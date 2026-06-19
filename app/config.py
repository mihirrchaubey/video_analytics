import os
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    video_storage_path: str
    frame_storage_path: str
    chroma_db_path: str
    database_url: str
    clip_model_id: str
    target_fps: int = 1
    reid_threshold: float = 0.55  # Aggressive: 0.55 to handle challenging occlusions and person switches
    target_class: str = "person"
    yolo_model: str = "yolov8n.pt"
    
    # GPU Support
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False, 
        extra='ignore'
    )

# Load settings
settings = Settings()

print(f"[GPU] Using device: {settings.device.upper()}")

# Create storage directories
for path in [settings.video_storage_path, settings.frame_storage_path, settings.chroma_db_path]:
    os.makedirs(path, exist_ok=True)