import os
import hashlib
import shutil
import logging
import torch
from contextlib import asynccontextmanager
from typing import List
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, VideoMetadata
from app.schemas import VideoUploadResponse, QueryRequest, QueryResponse
from app.video_processor import process_video, query_frames, generate_tracked_video, generate_search_video
from app.middleware import (
    setup_middleware, setup_exception_handlers, add_health_endpoint,
    VideoAnalyticsException, VideoProcessingError, InvalidQueryError,
    VideoNotFoundError
)
from app.utils import (
    compute_file_hash, validate_video, get_video_properties, 
    validate_query, validate_threshold, format_size, save_json, load_json
)

# ====================== LOGGING ======================
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# ====================== LIFESPAN ======================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    logger.info("[START] Video Analytics API starting")
    logger.info(f"Device: {settings.device.upper()}")
    
    # GPU Info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        logger.info(f"CUDA Version: {torch.version.cuda}")
    else:
        logger.info("[WARN] Running on CPU (slower processing)")
    
    logger.info(f"Storage: {settings.video_storage_path}")
    
    # Startup
    init_db()
    logger.info("[OK] Database initialized")
    
    yield
    
    # Shutdown
    logger.info("[STOP] Shutting down")

# ====================== APP INITIALIZATION ======================

app = FastAPI(
    title="Advanced Video Analytics with Persistent Tracking",
    description="Semantic Search + Target Tracking API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Setup middleware and error handlers
setup_middleware(app, enable_cors=True, enable_logging=True, 
                enable_performance=True, enable_rate_limit=False)
setup_exception_handlers(app)
health_monitor = add_health_endpoint(app)

# ====================== UTILITIES ======================

def compute_file_hash_internal(file_path: str) -> str:
    """Compute SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def cleanup_video_files(video_id: str):
    """Background task to cleanup video files"""
    try:
        video_dir = settings.video_storage_path
        frames_dir = settings.frame_storage_path
        
        # Remove video
        for f in os.listdir(video_dir):
            if video_id in f:
                os.remove(os.path.join(video_dir, f))
        
        # Remove frames
        for f in os.listdir(frames_dir):
            if video_id in f:
                os.remove(os.path.join(frames_dir, f))
        
        logger.info(f"✅ Cleaned up files for {video_id}")
    except Exception as e:
        logger.error(f"Error cleaning up {video_id}: {e}")

# ====================== ENDPOINTS ======================

@app.post("/upload", response_model=VideoUploadResponse, tags=["Video"])
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload and process video
    
    - Extracts frames at target FPS
    - Generates CLIP embeddings
    - Stores in vector database
    - Detects duplicate videos by hash
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(status_code=400, detail="Only video files allowed")
    
    logger.info(f"📹 Receiving upload: {file.filename}")
    
    # Save temp file
    temp_file_path = os.path.join(settings.video_storage_path, file.filename)
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    # Validate video
    valid, message = validate_video(temp_file_path)
    if not valid:
        os.remove(temp_file_path)
        raise HTTPException(status_code=400, detail=message)
    
    # Compute hash
    try:
        file_hash = compute_file_hash_internal(temp_file_path)
    except Exception as e:
        os.remove(temp_file_path)
        logger.error(f"Hash computation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to compute file hash")
    
    # Check for duplicates
    existing = db.query(VideoMetadata).filter(VideoMetadata.file_hash == file_hash).first()
    if existing:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        
        logger.info(f"Duplicate detected: {file_hash[:8]}")
        return VideoUploadResponse(
            message="Duplicate video detected. Using existing record.",
            video_id=existing.filename,
            file_hash=file_hash,
            status="duplicate"
        )
    
    # Rename file
    video_id = f"vid_{file_hash[:8]}"
    final_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")
    
    if os.path.exists(final_path):
        os.remove(final_path)
    
    os.rename(temp_file_path, final_path)
    
    # Save to database
    new_video = VideoMetadata(filename=video_id, file_hash=file_hash)
    db.add(new_video)
    db.commit()
    
    # Process video
    try:
        frames_extracted = process_video(final_path, video_id)
        
        logger.info(f"✅ Processed {frames_extracted} frames for {video_id}")
        
        return VideoUploadResponse(
            message=f"Video processed successfully. Extracted {frames_extracted} frames.",
            video_id=video_id,
            file_hash=file_hash,
            status="processed"
        )
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        
        # Cleanup
        db.delete(new_video)
        db.commit()
        
        if os.path.exists(final_path):
            os.remove(final_path)
        
        raise VideoProcessingError(f"Failed to process video: {str(e)}")

@app.post("/query", response_model=QueryResponse, tags=["Search"])
async def query_video(request: QueryRequest):
    """
    Semantic search for frames matching query
    
    Returns matching frames with similarity scores and timestamps
    """
    # Validate inputs
    valid, msg = validate_query(request.query)
    if not valid:
        raise InvalidQueryError(msg)
    
    valid, msg = validate_threshold(request.threshold)
    if not valid:
        raise InvalidQueryError(msg)
    
    logger.info(f"🔍 Query: '{request.query}' | Video: {request.video_id} | Threshold: {request.threshold}")
    
    try:
        results = query_frames(request.query, request.threshold, request.video_id)
        
        return QueryResponse(
            query=request.query,
            results=results,
            message=f"Found {len(results)} matching frames."
        )
    
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise VideoProcessingError(f"Search failed: {str(e)}")

@app.post("/generate_tracked", tags=["Tracking"])
async def generate_tracked(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Generate full video with target person tracked
    
    Creates video with persistent green bounding boxes around target person
    """
    logger.info(f"🎬 Generating tracked video for {request.video_id}")
    
    try:
        output_path = generate_tracked_video(request.video_id, request.query)
        
        return {
            "status": "success",
            "message": "Tracked video generated successfully!",
            "video_path": output_path,
            "video_id": request.video_id,
            "query": request.query,
            "timestamp": datetime.now().isoformat()
        }
    
    except FileNotFoundError:
        raise VideoNotFoundError(request.video_id)
    except Exception as e:
        logger.error(f"Tracked video generation failed: {e}")
        raise VideoProcessingError(f"Failed to generate tracked video: {str(e)}")

@app.post("/generate_search_video", tags=["Tracking"])
async def generate_search_video_endpoint(video_id: str):
    """
    Generate video with YOLO bounding boxes on ALL detected persons
    
    Creates video with blue bounding boxes around all detected persons
    Includes timestamp and confidence scores on each box
    """
    logger.info(f"🎬 Generating search video with YOLO boxes for {video_id}")
    
    try:
        output_path = generate_search_video(video_id)
        
        return {
            "status": "success",
            "message": "Search video with YOLO boxes generated successfully!",
            "video_path": output_path,
            "video_id": video_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        logger.error(f"Invalid video: {e}")
        raise VideoNotFoundError(str(e))
    except Exception as e:
        logger.error(f"Search video generation failed: {e}")
        raise VideoProcessingError(f"Failed to generate search video: {str(e)}")

@app.get("/videos", tags=["Video"])
async def list_videos(db: Session = Depends(get_db)):
    """Get list of all processed videos"""
    videos = db.query(VideoMetadata).all()
    return {
        "count": len(videos),
        "videos": [
            {
                "id": v.filename,
                "hash": v.file_hash[:8],
                "uploaded": v.upload_time.isoformat()
            }
            for v in videos
        ]
    }

@app.get("/videos/{video_id}", tags=["Video"])
async def get_video_details(video_id: str, db: Session = Depends(get_db)):
    """Get details for specific video"""
    video = db.query(VideoMetadata).filter(VideoMetadata.filename == video_id).first()
    
    if not video:
        raise VideoNotFoundError(video_id)
    
    video_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")
    
    if os.path.exists(video_path):
        props = get_video_properties(video_path)
        file_size = os.path.getsize(video_path)
        
        return {
            "id": video.filename,
            "hash": video.file_hash,
            "uploaded": video.upload_time.isoformat(),
            "file_size": format_size(file_size),
            "properties": props
        }
    
    return {"id": video.filename, "hash": video.file_hash, "uploaded": video.upload_time.isoformat()}

@app.delete("/videos/{video_id}", tags=["Video"])
async def delete_video(video_id: str, db: Session = Depends(get_db), 
                      background_tasks: BackgroundTasks = BackgroundTasks()):
    """Delete video and associated data"""
    video = db.query(VideoMetadata).filter(VideoMetadata.filename == video_id).first()
    
    if not video:
        raise VideoNotFoundError(video_id)
    
    # Delete from database
    db.delete(video)
    db.commit()
    
    # Schedule cleanup
    background_tasks.add_task(cleanup_video_files, video_id)
    
    logger.info(f"🗑️ Scheduled deletion of {video_id}")
    
    return {"status": "success", "message": f"Video {video_id} marked for deletion"}
@app.get("/download/tracked/{video_id}", tags=["Download"])
async def download_tracked_video(video_id: str):
    """Download tracked person video"""
    video_path = os.path.join(settings.frame_storage_path, f"{video_id}_tracked.mp4")
    
    if not os.path.exists(video_path):
        raise VideoNotFoundError(f"Tracked video not found for {video_id}")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}_tracked.mp4"
    )

@app.get("/download/yolo/{video_id}", tags=["Download"])
async def download_yolo_video(video_id: str):
    """Download YOLO detection boxes video"""
    video_path = os.path.join(settings.frame_storage_path, f"{video_id}_yolo_boxes.mp4")
    
    if not os.path.exists(video_path):
        raise VideoNotFoundError(f"YOLO video not found for {video_id}")
    
    return FileResponse(
        video_path,
        media_type="video/mp4",
        filename=f"{video_id}_yolo_boxes.mp4"
    )
@app.get("/stats", tags=["Statistics"])
async def get_statistics(db: Session = Depends(get_db)):
    """Get system statistics"""
    videos = db.query(VideoMetadata).all()
    
    total_size = 0
    for video in videos:
        video_path = os.path.join(settings.video_storage_path, f"{video.filename}.mp4")
        if os.path.exists(video_path):
            total_size += os.path.getsize(video_path)
    
    return {
        "total_videos": len(videos),
        "total_storage_used": format_size(total_size),
        "device": settings.device.upper(),
        "health": health_monitor.get_status()
    }

@app.get("/config", tags=["System"])
async def get_config():
    """Get current configuration"""
    return {
        "device": settings.device,
        "clip_model": settings.clip_model_id,
        "yolo_model": settings.yolo_model,
        "target_fps": settings.target_fps,
        "reid_threshold": settings.reid_threshold,
        "storage_paths": {
            "videos": settings.video_storage_path,
            "frames": settings.frame_storage_path,
            "chroma": settings.chroma_db_path
        }
    }

@app.post("/batch_search", tags=["Batch"])
async def batch_search(queries: List[str], video_id: str, threshold: float = 0.2):
    """Search multiple queries in video"""
    results = {}
    
    for query in queries:
        valid, msg = validate_query(query)
        if valid:
            try:
                matches = query_frames(query, threshold, video_id)
                results[query] = {
                    "status": "success",
                    "matches": len(matches),
                    "results": matches
                }
            except Exception as e:
                results[query] = {
                    "status": "error",
                    "error": str(e)
                }
        else:
            results[query] = {
                "status": "error",
                "error": msg
            }
    
    return {
        "video_id": video_id,
        "threshold": threshold,
        "results": results
    }

# ====================== ROOT ======================

@app.get("/", tags=["Info"])
async def root():
    """API information"""
    return {
        "name": "Advanced Video Analytics",
        "version": "2.0.0",
        "description": "Semantic Search + Persistent Tracking",
        "docs": "/docs",
        "health": "/health",
        "metrics": "/metrics"
    }

# ====================== RUN ======================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
