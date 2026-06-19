import os
import torch
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Storage paths with defaults
    video_storage_path: str = "./storage/videos"
    frame_storage_path: str = "./storage/frames"
    chroma_db_path: str = "./storage/chroma_db"
    
    # Database URL with SQLite default
    database_url: str = "sqlite:///./storage/metadata.db"
    
    # CLIP model for semantic embeddings
    clip_model_id: str = "openai/clip-vit-base-patch32"
    
    # Video processing
    target_fps: int = 1
    yolo_model: str = "yolov8n.pt"
    
    # Re-ID threshold (0.55-0.75 recommended)
    reid_threshold: float = 0.70
    target_class: str = "person"
    
    # GPU Support
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False, 
        extra='ignore',
        # Allow .env to be optional - use defaults if not found
    )

# Load settings (uses .env if exists, otherwise defaults)
settings = Settings()

print(f"[GPU] Using device: {settings.device.upper()}")

# Create storage directories if they don't exist
for path in [settings.video_storage_path, settings.frame_storage_path, settings.chroma_db_path]:
    os.makedirs(path, exist_ok=True)
    print(f"[STORAGE] Created/verified directory: {path}")