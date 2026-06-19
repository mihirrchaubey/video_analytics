import requests
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VideoAnalyticsClient:
    """Python client for Video Analytics API"""
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        """
        Initialize client
        
        Args:
            base_url: API endpoint URL
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Tuple[bool, Dict]:
        """Make HTTP request"""
        url = f"{self.base_url}{endpoint}"
        kwargs.setdefault('timeout', self.timeout)
        
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return True, response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return False, {"error": str(e)}
    
    def upload_video(self, video_path: str) -> Tuple[bool, Dict]:
        """
        Upload and process video
        
        Args:
            video_path: Path to video file
            
        Returns:
            (success, response) tuple
        """
        if not Path(video_path).exists():
            return False, {"error": "File not found"}
        
        try:
            with open(video_path, 'rb') as f:
                files = {'file': f}
                return self._request('POST', '/upload', files=files)
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False, {"error": str(e)}
    
    def search_frames(self, video_id: str, query: str, 
                     threshold: float = 0.2) -> Tuple[bool, Dict]:
        """
        Search for frames matching query
        
        Args:
            video_id: Video ID from upload
            query: Search query
            threshold: Similarity threshold (0-1)
            
        Returns:
            (success, response) tuple
        """
        payload = {
            "query": query,
            "threshold": threshold,
            "video_id": video_id
        }
        
        return self._request('POST', '/query', json=payload)
    
    def generate_tracked_video(self, video_id: str, query: str,
                              threshold: float = 0.2) -> Tuple[bool, Dict]:
        """
        Generate video with persistent tracking
        
        Args:
            video_id: Video ID from upload
            query: Query describing target person
            threshold: Similarity threshold (0-1)
            
        Returns:
            (success, response) tuple
        """
        payload = {
            "query": query,
            "video_id": video_id,
            "threshold": threshold
        }
        
        return self._request('POST', '/generate_tracked', json=payload)
    
    def list_videos(self) -> Tuple[bool, List[str]]:
        """
        Get list of processed videos
        
        Returns:
            (success, video_ids) tuple
        """
        success, response = self._request('GET', '/videos')
        if success:
            return True, response
        return False, []
    
    def get_video_info(self, video_id: str) -> Dict:
        """Get video metadata"""
        return {"video_id": video_id}
    
    def close(self):
        """Close session"""
        self.session.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class BatchProcessor:
    """Batch process multiple videos"""
    
    def __init__(self, client: VideoAnalyticsClient):
        self.client = client
    
    def upload_videos(self, video_dir: str) -> List[Dict]:
        """Upload all videos in directory"""
        results = []
        video_dir = Path(video_dir)
        
        for video_file in video_dir.glob("*.mp4"):
            logger.info(f"Uploading {video_file.name}...")
            success, response = self.client.upload_video(str(video_file))
            
            if success:
                results.append({
                    "file": video_file.name,
                    "status": "success",
                    "video_id": response.get("video_id"),
                    "message": response.get("message")
                })
            else:
                results.append({
                    "file": video_file.name,
                    "status": "error",
                    "error": response.get("error")
                })
        
        return results
    
    def search_all_videos(self, query: str, threshold: float = 0.2) -> Dict:
        """Search query in all videos"""
        results = {}
        
        success, videos = self.client.list_videos()
        if not success:
            return results
        
        for video_id in videos:
            logger.info(f"Searching {video_id}...")
            success, response = self.client.search_frames(video_id, query, threshold)
            
            if success:
                results[video_id] = {
                    "status": "success",
                    "matches": len(response.get("results", [])),
                    "results": response.get("results", [])
                }
            else:
                results[video_id] = {
                    "status": "error",
                    "error": response.get("error")
                }
        
        return results


# ====================== EXAMPLE USAGE ======================

if __name__ == "__main__":
    # Simple usage
    with VideoAnalyticsClient("http://localhost:8000") as client:
        # Upload video
        success, response = client.upload_video("test_video.mp4")
        if success:
            video_id = response["video_id"]
            print(f"Video uploaded: {video_id}")
            
            # Search
            success, results = client.search_frames(
                video_id=video_id,
                query="person in red shirt",
                threshold=0.2
            )
            
            if success:
                for match in results["results"]:
                    print(f"Match at {match['timestamp']}s - Score: {match['similarity_score']}")
            
            # Generate tracked video
            success, tracked = client.generate_tracked_video(
                video_id=video_id,
                query="person in red shirt"
            )
            
            if success:
                print(f"Tracked video: {tracked['video_path']}")
    
    # Batch processing
    with VideoAnalyticsClient("http://localhost:8000") as client:
        batch = BatchProcessor(client)
        
        # Upload multiple videos
        results = batch.upload_videos("./videos/")
        for result in results:
            print(result)
        
        # Search all videos
        search_results = batch.search_all_videos("person walking")
        for video_id, result in search_results.items():
            print(f"{video_id}: {result['matches']} matches")
