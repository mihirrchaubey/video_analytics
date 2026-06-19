import os
import shutil
import hashlib
import cv2
import numpy as np

def run_tests():
    """Run complete test pipeline"""
    print("🧪 Starting test pipeline...\n")
    
    # Cleanup
    print("🧹 Cleaning up old test data...")
    for path in ["./storage/videos", "./storage/frames", "./storage/chroma"]:
        if os.path.exists(path):
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)

    if os.path.exists("./storage/metadata.db"):
        os.remove("./storage/metadata.db")

    # Import after cleanup
    from app.config import settings
    from app.database import init_db, VideoMetadata, SessionLocal
    from app.video_processor import process_video, query_frames

    init_db()
    print("✅ Database initialized\n")

    # Create dummy video
    test_video_path = os.path.join(settings.video_storage_path, "raw_test_video.mp4")
    create_dummy_video(test_video_path, duration_sec=8, fps=15)

    # Hash and rename video
    file_hash = compute_file_hash(test_video_path)
    video_id = f"vid_{file_hash[:8]}"
    final_video_path = os.path.join(settings.video_storage_path, f"{video_id}.mp4")
    os.rename(test_video_path, final_video_path)

    print(f"\n📊 Processing test video: {video_id}\n")

    # Process video
    db = SessionLocal()
    try:
        new_video = VideoMetadata(filename=video_id, file_hash=file_hash)
        db.add(new_video)
        db.commit()

        frames_extracted = process_video(final_video_path, video_id)
        print(f"\n✅ Processed {frames_extracted} frames\n")

        # Test semantic search queries
        test_queries = [
            ("red circle", 0.15),
            ("blue square", 0.15),
            ("colored shape", 0.10),
        ]
        
        print("🔍 Testing semantic queries:\n")
        for query_text, threshold in test_queries:
            results = query_frames(query_text, threshold=threshold, video_id=video_id)
            print(f"   Query: '{query_text}' | Found: {len(results)} frames")
            
            for idx, res in enumerate(results[:3]):
                print(f"      {idx+1}. Time: {res['timestamp']}s | Score: {res['similarity_score']:.3f}")

        # Verify results
        all_results = query_frames("colored shape", threshold=0.05, video_id=video_id)
        assert len(all_results) > 0, "No results found for any query!"
        
        print(f"\n🎉 All tests passed! ✅")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        db.close()


def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def create_dummy_video(filename: str, duration_sec: int = 8, fps: int = 15):
    """
    Create test video with colored geometric shapes
    
    Args:
        filename: Output video file path
        duration_sec: Video duration in seconds
        fps: Frames per second
    """
    print(f"🎨 Generating test video: {filename}")
    
    width, height = 640, 480
    total_frames = duration_sec * fps
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    if not out.isOpened():
        raise RuntimeError(f"Failed to create video: {filename}")
    
    # Create frames with animated shapes
    for frame_idx in range(total_frames):
        # Create blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (20, 20, 20)  # Dark background
        
        # Progress in animation (0 to 1)
        progress = frame_idx / total_frames
        
        # Red circle - moving left to right
        circle_x = int(150 + progress * 350)
        circle_y = 150
        cv2.circle(frame, (circle_x, circle_y), 50, (0, 0, 255), -1)
        
        # Blue square - moving top to bottom
        square_size = 60
        square_x = 100
        square_y = int(100 + progress * 280)
        cv2.rectangle(frame, 
                     (square_x, square_y),
                     (square_x + square_size, square_y + square_size),
                     (255, 0, 0), -1)
        
        # Green triangle - rotating
        angle = progress * 360
        center = (500, 300)
        radius = 70
        
        # Calculate triangle vertices
        angles = [0, 120, 240]
        vertices = []
        for a in angles:
            rad = np.radians(a + angle)
            x = int(center[0] + radius * np.cos(rad))
            y = int(center[1] + radius * np.sin(rad))
            vertices.append([x, y])
        
        vertices = np.array(vertices, dtype=np.int32)
        cv2.polylines(frame, [vertices], True, (0, 255, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_idx + 1}/{total_frames}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = frame_idx / fps
        cv2.putText(frame, f"Time: {timestamp:.2f}s", 
                   (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"✅ Test video created: {total_frames} frames @ {fps}fps")


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
