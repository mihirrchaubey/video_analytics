import cv2
import torch
import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import chromadb
import warnings

# Suppress PyTorch security warning for older versions
os.environ['DISABLE_TRANSFORMERS_TORCH_LOAD_CHECK'] = '1'
warnings.filterwarnings('ignore')

from app.config import settings
from app.tracker import ObjectTracker

# ====================== GPU SETUP ======================
device = settings.device
print(f"[GPU] Using device: {device.upper()}")

# Load CLIP model and processor
processor = CLIPProcessor.from_pretrained(settings.clip_model_id)
model = CLIPModel.from_pretrained(settings.clip_model_id).to(device)

# Initialize Chroma vector database
chroma_client = chromadb.PersistentClient(path=settings.chroma_db_path)
collection = chroma_client.get_or_create_collection(
    name="video_frames", metadata={"hnsw:space": "cosine"}
)

# Initialize tracker with CLIP components
tracker = ObjectTracker(processor=processor, model=model, device=device)

# ====================== FEATURE EXTRACTION ======================

def get_image_features(pil_image):
    """Extract CLIP image embedding"""
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        return features / features.norm(p=2, dim=-1, keepdim=True)

def get_text_features(text):
    """Extract CLIP text embedding"""
    inputs = processor(text=text, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        return features / features.norm(p=2, dim=-1, keepdim=True)

# ====================== VIDEO PROCESSING ======================

def process_video(video_path: str, video_id: str) -> int:
    """
    Extract frames and embeddings from video with batch processing
    
    Args:
        video_path: Path to input video
        video_id: Unique video identifier
        
    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, int(round(fps / settings.target_fps)))

    print(f"[PROCESS] Processing video: {video_path}")
    print(f"   FPS: {fps}, Total frames: {total_frames}, Skip: {frame_skip}")
    print(f"   Batch size: 16 frames (faster GPU processing)")

    frame_count = extracted_count = 0
    embeddings_batch, metadatas_batch, ids_batch = [], [], []
    
    # Buffer for batch CLIP processing (16 frames at a time)
    frame_buffer = []
    frame_info_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at target FPS
        if frame_count % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            
            # Add to buffer
            frame_buffer.append(pil)
            frame_info_buffer.append({
                "frame_id": extracted_count,
                "timestamp": extracted_count / settings.target_fps
            })
            
            extracted_count += 1
            
            # Process batch of 16 frames together
            if len(frame_buffer) >= 16:
                _process_frame_batch(video_id, frame_buffer, frame_info_buffer, 
                                    embeddings_batch, metadatas_batch, ids_batch)
                frame_buffer, frame_info_buffer = [], []
                print(f"   Processed {extracted_count} frames...")

        frame_count += 1

    # Process remaining frames
    if frame_buffer:
        _process_frame_batch(video_id, frame_buffer, frame_info_buffer, 
                            embeddings_batch, metadatas_batch, ids_batch)

    # Add all embeddings to Chroma
    if embeddings_batch:
        print(f"[STORE] Storing {len(embeddings_batch)} embeddings in Chroma...")
        collection.add(embeddings=embeddings_batch, metadatas=metadatas_batch, ids=ids_batch)

    cap.release()
    print(f"[OK] Extraction complete: {extracted_count} frames")
    return extracted_count


def _process_frame_batch(video_id: str, frame_buffer, frame_info_buffer, 
                         embeddings_batch, metadatas_batch, ids_batch):
    """Process batch of frames together for faster GPU processing"""
    # Get CLIP embeddings for all frames at once
    inputs = processor(images=frame_buffer, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        features = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
        features = features / features.norm(p=2, dim=-1, keepdim=True)
    
    # Add to batch
    for i, (pil_image, frame_info) in enumerate(zip(frame_buffer, frame_info_buffer)):
        frame_path = os.path.join(settings.frame_storage_path, 
                                  f"{video_id}_frame_{frame_info['frame_id']:06d}.jpg")
        pil_image.save(frame_path)
        
        embeddings_batch.append(features[i].cpu().numpy().tolist())
        metadatas_batch.append({
            "frame_path": frame_path,
            "timestamp": f"{frame_info['timestamp']:.2f}",
            "video_id": video_id
        })
        ids_batch.append(f"{video_id}_{frame_info['frame_id']}")

# ====================== SEMANTIC SEARCH ======================

def query_frames(query_text: str, threshold: float, video_id: str, top_k: int = 10):
    """
    Search for frames matching query using semantic similarity
    
    Args:
        query_text: Natural language search query
        threshold: Similarity threshold (0-1)
        video_id: Video to search in
        top_k: Maximum results to return
        
    Returns:
        List of matching frames with metadata
    """
    try:
        print(f"[QUERY] Querying: '{query_text}' (threshold: {threshold})")
        
        # Get text embedding
        text_features = get_text_features([query_text])
        text_embedding = text_features.cpu().numpy().tolist()[0]
        
        # Query Chroma
        results = collection.query(
            query_embeddings=[text_embedding],
            n_results=top_k,
            where={"video_id": video_id},
            include=["metadatas", "distances"]
        )

        # Process results
        valid_matches = []
        for dist, meta in zip(results.get("distances", [[]])[0], results.get("metadatas", [[]])[0]):
            # Convert distance to similarity (cosine distance to similarity)
            similarity = 1.0 - (dist / 2.0)
            
            if similarity >= threshold:
                valid_matches.append({
                    "frame_path": meta["frame_path"],
                    "timestamp": meta["timestamp"],
                    "similarity_score": round(similarity, 4),
                    "has_target": False  # Can be updated if tracking info available
                })
        
        print(f"   Found {len(valid_matches)} matches")
        return valid_matches
        
    except Exception as e:
        print(f"[ERROR] Query error: {e}")
        return []

# ====================== TRACKED VIDEO GENERATION ======================

def generate_tracked_video(video_id: str, query_text: str):
    """
    Generate video with persistent person tracking
    
    Args:
        video_id: Video to process
        query_text: Query describing target person
        
    Returns:
        Path to output tracked video
    """
    # Find video file
    video_path = None
    for f in os.listdir(settings.video_storage_path):
        if video_id in f and f.endswith('.mp4'):
            video_path = os.path.join(settings.video_storage_path, f)
            break

    if not video_path:
        raise ValueError(f"Video not found for ID: {video_id}")

    print(f"[GENERATE] Generating tracked video for {video_id}...")
    
    # Reset tracker state
    tracker.reference_embedding = None
    tracker.target_id = None
    tracker.target_timestamps = []

    # Generate tracked output
    output_path = tracker.generate_tracked_video(video_path, video_id, query_text)
    return output_path

def generate_search_video(video_id: str):
    """
    Generate video with YOLO bounding boxes on ALL detected persons
    
    Args:
        video_id: Video to process
        
    Returns:
        Path to output video with YOLO boxes
    """
    # Find video file
    video_path = None
    for f in os.listdir(settings.video_storage_path):
        if video_id in f and f.endswith('.mp4'):
            video_path = os.path.join(settings.video_storage_path, f)
            break

    if not video_path:
        raise ValueError(f"Video not found for ID: {video_id}")

    print(f"[GENERATE] Generating search video with YOLO boxes for {video_id}...")
    
    # Generate video with all detections
    output_path = tracker.generate_all_detections_video(video_path, video_id)
    return output_path
