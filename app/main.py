import os
import hashlib
import shutil
import logging
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session

from app.config import settings
from app.database import init_db, get_db, VideoMetadata
from app.schemas import VideoUploadResponse, QueryRequest, QueryResponse
from app.video_processor import process_video, query_frames, generate_tracked_video

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing database...")
    init_db()
    yield
    logger.info("Shutting down.")

app = FastAPI(
    title="Advanced Video Analytics with Persistent Tracking",
    description="Semantic Search + Target Tracking",
    version="2.0.0",
    lifespan=lifespan
)

def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

@app.post("/upload", response_model=VideoUploadResponse)
async def upload_video(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    logger.info(f"Receiving upload: {file.filename}")
    temp_file_path = os.path.join(settings.video_storage_path, file.filename)

    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    file_hash = compute_file_hash(temp_file_path)
    existing = db.query(VideoMetadata).filter(VideoMetadata.file_hash == file_hash).first()

    if existing:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return VideoUploadResponse(
            message="Duplicate video detected.", 
            video_id=existing.filename, 
            file_hash=file_hash, 
            status="duplicate"
        )

    video_id = f"vid_{file_hash[:8]}"
    final_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")

    if os.path.exists(final_path):
        os.remove(final_path)
    os.rename(temp_file_path, final_path)

    new_video = VideoMetadata(filename=video_id, file_hash=file_hash)
    db.add(new_video)
    db.commit()

    try:
        frames_extracted = process_video(final_path, video_id)
        logger.info(f"Processed {frames_extracted} frames for {video_id}")
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        db.delete(new_video)
        db.commit()
        if os.path.exists(final_path):
            os.remove(final_path)
        raise HTTPException(status_code=500, detail=str(e))

    return VideoUploadResponse(
        message=f"Video processed successfully. Extracted {frames_extracted} frames.",
        video_id=video_id,
        file_hash=file_hash,
        status="processed"
    )

@app.post("/query", response_model=QueryResponse)
async def query_video(request: QueryRequest):
    results = query_frames(request.query, request.threshold, request.video_id)
    return QueryResponse(
        query=request.query, 
        results=results, 
        message=f"Found {len(results)} matches."
    )

# ==================== NEW ENDPOINT ====================
@app.post("/generate_tracked")
async def generate_tracked(request: QueryRequest):
    """Generate full video with target person tracked (green box)"""
    try:
        output_path = generate_tracked_video(request.video_id, request.query)
        return {
            "status": "success",
            "message": "Tracked video generated successfully!",
            "video_path": output_path,
            "video_id": request.video_id
        }
    except Exception as e:
        logger.error(f"Tracked video generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# =====================================================

@app.get("/videos")
async def list_videos(db: Session = Depends(get_db)):
    videos = db.query(VideoMetadata).all()
    return [v.filename for v in videos]