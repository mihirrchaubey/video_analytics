import os
import hashlib
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import torch
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

# ====================== FILE UTILITIES ======================

def compute_file_hash(file_path: str, algorithm: str = "sha256") -> str:
    """Compute hash of file for deduplication"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def get_file_size(file_path: str) -> int:
    """Get file size in bytes"""
    return os.path.getsize(file_path)

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes"""
    return get_file_size(file_path) / (1024 * 1024)

def safe_remove(file_path: str) -> bool:
    """Safely remove file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
    except Exception as e:
        logger.error(f"Error removing {file_path}: {e}")
    return False

def ensure_directory(dir_path: str) -> bool:
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(dir_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error creating directory {dir_path}: {e}")
        return False

# ====================== VIDEO UTILITIES ======================

def get_video_properties(video_path: str) -> Dict[str, Any]:
    """Extract video properties"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    props = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
        "duration_seconds": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
    }
    cap.release()
    return props

def validate_video(video_path: str) -> Tuple[bool, str]:
    """Validate video file"""
    if not os.path.exists(video_path):
        return False, "File not found"
    
    if get_file_size_mb(video_path) > 1000:  # 1GB max
        return False, "File too large (max 1GB)"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Cannot open video file"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    
    if frame_count < 1:
        return False, "Video has no frames"
    
    if fps < 1:
        return False, "Invalid FPS"
    
    return True, "Valid"

def extract_video_thumbnail(video_path: str, frame_idx: int = 0) -> Optional[np.ndarray]:
    """Extract single frame from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    return frame if ret else None

def calculate_video_stats(video_path: str) -> Dict[str, Any]:
    """Calculate video statistics"""
    props = get_video_properties(video_path)
    if not props:
        return {}
    
    return {
        "resolution": f"{props['width']}x{props['height']}",
        "aspect_ratio": f"{props['width']/props['height']:.2f}",
        "fps": props['fps'],
        "total_frames": props['frame_count'],
        "duration_seconds": props['duration_seconds'],
        "file_size_mb": get_file_size_mb(video_path),
        "bitrate_mbps": (get_file_size_mb(video_path) * 8) / props['duration_seconds'] if props['duration_seconds'] > 0 else 0
    }

# ====================== IMAGE UTILITIES ======================

def resize_image(image: np.ndarray, width: Optional[int] = None, height: Optional[int] = None) -> np.ndarray:
    """Resize image maintaining aspect ratio"""
    if width is None and height is None:
        return image
    
    h, w = image.shape[:2]
    
    if width is None:
        ratio = height / h
        width = int(w * ratio)
    elif height is None:
        ratio = width / w
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def save_image(image: np.ndarray, path: str, quality: int = 95) -> bool:
    """Save image with quality control"""
    try:
        _, ext = os.path.splitext(path)
        if ext.lower() == '.jpg':
            cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            cv2.imwrite(path, image)
        return True
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}")
        return False

def load_image(path: str) -> Optional[np.ndarray]:
    """Load image safely"""
    try:
        return cv2.imread(path)
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None

# ====================== TENSOR UTILITIES ======================

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor to numpy array"""
    return tensor.detach().cpu().numpy()

def numpy_to_tensor(array: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Convert numpy array to torch tensor"""
    return torch.from_numpy(array).to(device)

def normalize_embedding(embedding: torch.Tensor) -> torch.Tensor:
    """L2 normalize embedding"""
    return embedding / embedding.norm(p=2, dim=-1, keepdim=True)

def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> float:
    """Calculate cosine similarity between two embeddings"""
    return torch.cosine_similarity(emb1, emb2).item()

# ====================== CACHING ======================

@lru_cache(maxsize=128)
def cached_get_video_properties(video_path: str) -> Dict:
    """Cached version of get_video_properties"""
    return get_video_properties(video_path)

class EmbeddingCache:
    """Cache for video embeddings"""
    def __init__(self, cache_dir: str = "./cache/embeddings"):
        self.cache_dir = cache_dir
        ensure_directory(cache_dir)
    
    def get(self, video_id: str) -> Optional[Dict]:
        """Get cached embeddings"""
        cache_file = os.path.join(self.cache_dir, f"{video_id}.pkl")
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading cache {cache_file}: {e}")
        return None
    
    def set(self, video_id: str, data: Dict) -> bool:
        """Cache embeddings"""
        cache_file = os.path.join(self.cache_dir, f"{video_id}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving cache {cache_file}: {e}")
            return False
    
    def clear(self, video_id: Optional[str] = None) -> bool:
        """Clear cache"""
        try:
            if video_id:
                cache_file = os.path.join(self.cache_dir, f"{video_id}.pkl")
                return safe_remove(cache_file)
            else:
                import shutil
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                ensure_directory(self.cache_dir)
                return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

# ====================== BATCH PROCESSING ======================

def batch_iterator(items: List, batch_size: int):
    """Iterate through items in batches"""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

class BatchProcessor:
    """Process items in batches with threading"""
    def __init__(self, batch_size: int = 32, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
    
    def process(self, items: List, process_func, *args, **kwargs) -> List:
        """Process items in parallel batches"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for batch in batch_iterator(items, self.batch_size):
                future = executor.submit(process_func, batch, *args, **kwargs)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.extend(result if isinstance(result, list) else [result])
                except Exception as e:
                    logger.error(f"Batch processing error: {e}")
        
        return results

# ====================== DATA STRUCTURES ======================

class FrameMetadata:
    """Store frame metadata"""
    def __init__(self, frame_id: str, timestamp: float, frame_path: str, 
                 video_id: str, embedding: Optional[np.ndarray] = None):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.frame_path = frame_path
        self.video_id = video_id
        self.embedding = embedding
        self.created_at = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "frame_path": self.frame_path,
            "video_id": self.video_id,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(
            frame_id=data["frame_id"],
            timestamp=data["timestamp"],
            frame_path=data["frame_path"],
            video_id=data["video_id"]
        )

class DetectionResult:
    """Store detection results"""
    def __init__(self, class_id: int, class_name: str, confidence: float,
                 bbox: Tuple[int, int, int, int], track_id: Optional[int] = None):
        self.class_id = class_id
        self.class_name = class_name
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.track_id = track_id
    
    def to_dict(self) -> Dict:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": list(self.bbox),
            "track_id": self.track_id
        }

# ====================== VALIDATION ======================

def validate_query(query: str, min_length: int = 3, max_length: int = 500) -> Tuple[bool, str]:
    """Validate search query"""
    if not query:
        return False, "Query cannot be empty"
    
    if len(query) < min_length:
        return False, f"Query too short (min {min_length} chars)"
    
    if len(query) > max_length:
        return False, f"Query too long (max {max_length} chars)"
    
    return True, "Valid"

def validate_threshold(threshold: float) -> Tuple[bool, str]:
    """Validate similarity threshold"""
    if not isinstance(threshold, (int, float)):
        return False, "Threshold must be a number"
    
    if threshold < 0 or threshold > 1:
        return False, "Threshold must be between 0 and 1"
    
    return True, "Valid"

def validate_video_id(video_id: str) -> Tuple[bool, str]:
    """Validate video ID format"""
    if not video_id or not isinstance(video_id, str):
        return False, "Invalid video ID"
    
    if not video_id.startswith("vid_"):
        return False, "Video ID must start with 'vid_'"
    
    return True, "Valid"

# ====================== FORMATTING ======================

def format_duration(seconds: float) -> str:
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def format_size(bytes_size: float) -> str:
    """Format bytes to human readable"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"

def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

# ====================== JSON SERIALIZATION ======================

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def to_json(obj: Any, indent: int = 2) -> str:
    """Convert to JSON with numpy support"""
    return json.dumps(obj, cls=NumpyEncoder, indent=indent)

def save_json(obj: Any, path: str) -> bool:
    """Save object to JSON file"""
    try:
        with open(path, 'w') as f:
            json.dump(obj, f, cls=NumpyEncoder, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON {path}: {e}")
        return False

def load_json(path: str) -> Optional[Dict]:
    """Load JSON file"""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON {path}: {e}")
        return None
