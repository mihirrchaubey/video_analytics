from pydantic import BaseModel, Field
from typing import List, Optional

class VideoUploadResponse(BaseModel):
    message: str
    video_id: str
    file_hash: str
    status: str

class FrameResult(BaseModel):
    frame_path: str
    timestamp: str
    similarity_score: float
    has_target: Optional[bool] = False

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    video_id: Optional[str] = None

class QueryResponse(BaseModel):
    query: str
    results: List[FrameResult]
    message: str