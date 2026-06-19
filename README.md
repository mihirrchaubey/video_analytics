# 🎥 Advanced Video Analytics with Persistent Tracking

A production-grade video analytics system combining **YOLOv8 person detection**, **CLIP-based person re-identification**, **ByteTrack multi-object tracking**, and **semantic search** for intelligent video analysis.

**Key Features:**
- 🎯 **Persistent Target Tracking** - Track specific individuals across frames using CLIP embeddings
- 🔍 **Semantic Video Search** - Query videos using natural language (e.g., "man in black suit")
- 🤖 **GPU-Accelerated** - YOLO detection + CLIP embeddings on CUDA for 10x+ speedup
- 📊 **Multi-Object Tracking** - ByteTrack maintains identity across occlusions
- 🎬 **Dual Video Output** - Generated tracked video + YOLO detection boxes video
- 💾 **Persistent Storage** - ChromaDB for frame embeddings + SQLite for metadata

---

## 🏗️ Architecture

```
┌─────────────────┐
│  Streamlit UI   │ (Port 8501)
│   - Upload      │
│   - Search      │
│   - Download    │
└────────┬────────┘
         │
┌─────────▼────────────────────────────┐
│  FastAPI Backend (Port 8001)         │
│  ├─ /upload - Process video          │
│  ├─ /query - Search frames           │
│  ├─ /generate_tracked - Re-ID video  │
│  ├─ /generate_search_video - YOLO    │
│  └─ /download/* - Stream results     │
└─────────┬────────────────────────────┘
          │
    ┌─────┴──────────────────┬──────────────────┐
    │                        │                  │
┌───▼───┐  ┌──────────┐  ┌──▼──┐  ┌──────────┐
│ YOLO  │  │   CLIP   │  │ ByteTrack  │ ChromaDB
│Detect │  │ Embedder │  │(Tracker)   │ Vector DB
└───────┘  └──────────┘  └──────┘   └──────────┘
  (GPU)      (GPU)        (CPU)        (Disk)
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- CUDA 12.1 (for GPU acceleration)
- FFmpeg (for video processing)
- 8GB+ VRAM recommended

### Installation

```bash
# Clone repository
git clone https://github.com/mihirrchaubey/video_analytics.git
cd video_analytics

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Setup configuration
copy .env.example .env  # Configure if needed
```

### Run the Application

**Start Backend (FastAPI)**
```bash
python -m uvicorn app.main_enhanced:app --reload --host 127.0.0.1 --port 8001
```
✅ Backend runs on `http://127.0.0.1:8001`

**Start Frontend (Streamlit)** - *In another terminal*
```bash
streamlit run app/ui.py --server.port 8501 --server.address 127.0.0.1
```
✅ UI opens at `http://127.0.0.1:8501`

---

## 💻 Usage

### 1. Upload & Process Video
- Select MP4/AVI/MOV/MKV file (max 200MB)
- Click **"⬆️ Upload & Process Video"**
- System extracts frames at 1 FPS and generates CLIP embeddings

### 2. Search for Objects
- Enter natural language query: *"man in black suit"*, *"person with glasses"*
- Adjust confidence slider (default 0.2)
- Click **"🔍 Search Matching Frames"**
- View matching frames in grid with similarity scores

### 3. Generate Tracked Video (Re-ID)
- Click **"Generate Tracked Video (Re-ID)"**
- System finds first match, creates CLIP baseline embedding
- Tracks person throughout video using:
  - ByteTrack for frame-to-frame continuity
  - CLIP embeddings for re-detection on occlusions
  - Adaptive checking (every 2 frames typically, every frame if low confidence)
- Output: Green boxes with "TARGET" labels on matched person

### 4. Generate YOLO Detection Video
- Click **"Generate YOLO Boxes Video"**
- Shows all detected persons with blue boxes
- Useful for validation and analysis

### 5. Download Results
- Click download button under generated videos
- Streams MP4 directly to your machine

---

## 🔧 Key Components

### **YOLOv8n Detection**
- Ultra-nano model (~3.2M params)
- ~20-30ms per frame on RTX 4060
- Class 0 = Person (only detection used)
- Moved to GPU at startup

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
results = model(frame)  # Returns bounding boxes
```

### **CLIP Re-Identification**
- `openai/clip-vit-base-patch32` for semantic embeddings
- Extracts 512-D embedding from person crops
- Cosine similarity matching against reference embedding
- Threshold: 0.70 (tuned for accuracy vs speed)

```python
from transformers import CLIPProcessor, CLIPModel
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

# Single embedding
embedding = model.get_image_features(processor(images=[pil_image], return_tensors="pt"))

# Batch processing (16 crops simultaneously)
embeddings = get_embeddings_batch(crops_list)  # ~10x faster
```

### **ByteTrack Multi-Object Tracking**
- Maintains unique track_id per detected person
- Handles occlusions via IoU matching
- Automatically recovers lost tracks when person reappears

```python
from supervision import ByteTrack
tracker = ByteTrack()
detections = tracker.update_with_detections(detections)  # Adds track_id
```

### **ChromaDB Vector Database**
- Stores frame embeddings with metadata
- Cosine similarity search for semantic matching
- Persistent storage at `chroma_db/`

```python
from chromadb.config import Settings
collection = client.get_or_create_collection(
    name="video_frames",
    metadata={"hnsw:space": "cosine"}
)
```

---

## 📊 Performance Metrics

| Component | Hardware | Speed | Notes |
|-----------|----------|-------|-------|
| YOLO Detection | RTX 4060 | 20-30ms/frame | ~33-50 FPS |
| CLIP Embedding (single) | RTX 4060 | 50-100ms | Used for re-ID |
| CLIP Batch (16 crops) | RTX 4060 | 60-80ms | ~200ms/frame with overhead |
| ByteTrack Update | CPU | <1ms | Frame-to-frame tracking |
| Full Pipeline | RTX 4060 | ~5-10 frames/sec | ~6400 frames ≈ 10-20 minutes |

**GPU Detection:**
```
CUDA Available: ✅
GPU: NVIDIA RTX 4060 (8.6 GB VRAM)
Compute Capability: 8.9
```

---

## 📁 File Structure

```
video_analytics/
├── app/
│   ├── main_enhanced.py      # FastAPI backend (6 endpoints)
│   ├── ui.py                 # Streamlit frontend
│   ├── tracker.py            # ObjectTracker class (YOLO+CLIP+ByteTrack)
│   ├── video_processor.py    # Video I/O & CLIP embeddings
│   ├── config.py             # Pydantic settings
│   ├── middleware.py         # Request logging
│   ├── database.py           # SQLite + ChromaDB setup
│   ├── schemas.py            # Request/response models
│   └── ...
├── requirements.txt          # Python dependencies
├── docker-compose.yml        # Docker orchestration
├── Dockerfile                # Backend container
├── setup.bat                 # Windows setup script
├── run.bat                   # Windows startup script
└── README.md                 # This file
```

---

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# Access services
# Frontend: http://localhost:8501
# Backend: http://localhost:8001
```

---

## 🔌 API Endpoints

### POST `/upload`
Upload and process video.
```bash
curl -F "file=@video.mp4" http://localhost:8001/upload
# Response: {"video_id": "vid_xxxx", "frames_stored": 6422}
```

### POST `/query`
Search for matching frames.
```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{"query_text": "man in black suit", "video_id": "vid_xxxx", "threshold": 0.2}'
# Response: [{"frame_path": "...", "timestamp": 209.0, "similarity": 0.639}, ...]
```

### POST `/generate_tracked`
Generate tracked video with person re-ID.
```bash
curl -X POST http://localhost:8001/generate_tracked \
  -H "Content-Type: application/json" \
  -d '{"video_id": "vid_xxxx", "query_text": "man in black suit"}'
```

### POST `/generate_search_video`
Generate YOLO detection boxes video.
```bash
curl -X POST http://localhost:8001/generate_search_video \
  -H "Content-Type: application/json" \
  -d '{"video_id": "vid_xxxx"}'
```

### GET `/download/tracked/{video_id}`
Stream tracked video.
```bash
curl http://localhost:8001/download/tracked/vid_xxxx -o output_tracked.mp4
```

### GET `/download/yolo/{video_id}`
Stream YOLO detection video.
```bash
curl http://localhost:8001/download/yolo/vid_xxxx -o output_yolo.mp4
```

---

## 🧪 Testing

Run the included test pipeline:
```bash
python app/test_pipeline.py
```

Or test with API directly:
```bash
python test_api.py
```

---

## 🔍 Algorithm Details

### Re-ID (Person Re-Identification) Flow

1. **First Frame Detection**
   - Run YOLO → Get bounding boxes
   - Extract crop of first detected person
   - Generate CLIP embedding → Store as `reference_embedding`
   - Initialize `target_id` from ByteTrack

2. **Frame-by-Frame Processing**
   - Run YOLO → Get detections for current frame
   - Update ByteTrack with detections
   - Check if `target_id` still exists in current detections
   - If found: Draw green box, continue tracking
   - If not found (occlusion/out-of-frame):
     - For each detected person crop:
       - Generate CLIP embedding
       - Calculate cosine similarity to `reference_embedding`
       - If similarity > threshold (0.70): Re-detect target, draw "[RE-DETECTED]"
   - Adaptive checking: Increase frequency if confidence drops

3. **Output**
   - Draws 4px green boxes on target person
   - Labels: "TARGET" or "TARGET [RE-DETECTED]"
   - Confidence score overlay
   - Resolution: 1x1 (original video resolution preserved)

### Semantic Search Flow

1. Convert query text → CLIP embedding via text encoder
2. Search ChromaDB with cosine similarity
3. Return top-K matching frames with scores
4. User selects frame for tracking

---

## 📝 Configuration

Edit `app/config.py` to adjust:

```python
# Video processing
target_fps = 1                          # Extract 1 frame per second
video_storage_path = "storage/videos"
frame_storage_path = "storage/frames"

# Re-ID tuning
reid_threshold = 0.70                   # Cosine similarity threshold
                                        # Higher = stricter matching
                                        # Lower = looser (more false positives)

# CLIP batch processing
clip_batch_size = 16                    # Process 16 crops simultaneously

# GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|----------|
| **CUDA not detected** | Install PyTorch 2.5.1+cu121 + CUDA 12.1 drivers |
| **Video generation slow** | Reduce video resolution or use `target_fps = 0.5` |
| **Re-ID losing target** | Lower `reid_threshold` from 0.70 to 0.55-0.60 |
| **Memory errors** | Reduce `clip_batch_size` from 16 to 8 or 4 |
| **Port already in use** | Change port in startup command: `--port 8002` |

---

## 📚 Dependencies

- **PyTorch 2.5.1+cu121** - GPU compute framework
- **Ultralytics 8.2+** - YOLOv8 detection
- **Transformers 4.30+** - CLIP model
- **Supervision 0.20+** - ByteTrack integration
- **OpenCV 4.8+** - Video processing
- **ChromaDB 0.4.24+** - Vector database
- **FastAPI 0.110+** - REST backend
- **Streamlit 1.32+** - Web frontend
- **Pydantic 2.0+** - Data validation

See `requirements.txt` for complete list.

---

## 🎯 Use Cases

1. **Interview Submission** - Demonstrates full ML pipeline with GPU optimization
2. **Security Surveillance** - Track specific individuals in video feeds
3. **Sports Analytics** - Track players throughout games
4. **Crowd Analysis** - Monitor person movement in crowded scenes
5. **Video Search** - Find scenes matching descriptions in archives

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👨‍💻 Author

Built for portfolio/interview demonstration of advanced computer vision techniques.

**Highlights:**
- ✅ GPU acceleration (10x+ speedup)
- ✅ Production-grade REST API
- ✅ Semantic search capability
- ✅ Multi-object tracking with occlusion handling
- ✅ Person re-identification with CLIP embeddings
- ✅ Docker deployment ready
- ✅ Streamlit interactive UI

---

## 📧 Support

For issues, questions, or improvements:
1. Check the **Troubleshooting** section
2. Review `app.log` for error details
3. Test individual components with `test_api.py`
4. Check CUDA availability with `nvidia-smi`

---

**Last Updated:** 2026-06-19  
**Repository:** https://github.com/mihirrchaubey/video_analytics
