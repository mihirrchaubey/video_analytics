import sqlite3
import json
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# ====================== VIDEO ANALYTICS ======================

class VideoAnalytics:
    """Analyze video processing statistics"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_processing_stats(self) -> Dict:
        """Get video processing statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Total videos
            cursor.execute("SELECT COUNT(*) as count FROM videos")
            total = cursor.fetchone()['count']
            
            # Videos by day
            cursor.execute("""
                SELECT DATE(upload_time) as day, COUNT(*) as count
                FROM videos
                GROUP BY DATE(upload_time)
                ORDER BY day DESC
            """)
            by_day = {row['day']: row['count'] for row in cursor.fetchall()}
            
            # Latest uploads
            cursor.execute("""
                SELECT filename, upload_time 
                FROM videos 
                ORDER BY upload_time DESC 
                LIMIT 5
            """)
            latest = [{"id": row['filename'], "time": row['upload_time']} for row in cursor.fetchall()]
            
            conn.close()
            
            return {
                "total_videos": total,
                "by_day": by_day,
                "latest_uploads": latest
            }
        
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {}

class SearchAnalytics:
    """Analyze search queries and results"""
    
    def __init__(self):
        self.queries = []
        self.results_cache = {}
    
    def log_query(self, query: str, video_id: str, threshold: float, result_count: int, duration: float):
        """Log search query"""
        self.queries.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "video_id": video_id,
            "threshold": threshold,
            "result_count": result_count,
            "duration_ms": duration * 1000
        })
    
    def get_popular_queries(self, limit: int = 10) -> List[Dict]:
        """Get most popular queries"""
        from collections import Counter
        
        query_counts = Counter(q['query'] for q in self.queries)
        
        return [
            {"query": query, "count": count}
            for query, count in query_counts.most_common(limit)
        ]
    
    def get_performance_stats(self) -> Dict:
        """Get search performance statistics"""
        if not self.queries:
            return {}
        
        durations = [q['duration_ms'] for q in self.queries]
        
        return {
            "total_queries": len(self.queries),
            "avg_duration_ms": np.mean(durations),
            "min_duration_ms": np.min(durations),
            "max_duration_ms": np.max(durations),
            "median_duration_ms": np.median(durations),
            "total_results": sum(q['result_count'] for q in self.queries)
        }

class EmbeddingAnalytics:
    """Analyze embedding statistics"""
    
    @staticmethod
    def calculate_embedding_stats(embeddings: List[np.ndarray]) -> Dict:
        """Calculate statistics for embeddings"""
        if not embeddings:
            return {}
        
        embeddings = np.array(embeddings)
        
        return {
            "count": len(embeddings),
            "dimension": embeddings.shape[1] if len(embeddings.shape) > 1 else 0,
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "min_value": float(np.min(embeddings)),
            "max_value": float(np.max(embeddings)),
            "mean_value": float(np.mean(embeddings)),
            "sparsity": float(np.sum(embeddings == 0) / embeddings.size)
        }
    
    @staticmethod
    def calculate_similarity_distribution(embeddings: List[np.ndarray]) -> Dict:
        """Calculate similarity distribution"""
        if len(embeddings) < 2:
            return {}
        
        similarities = []
        embeddings = np.array(embeddings)
        
        # Sample pairs for efficiency
        sample_size = min(1000, len(embeddings) * (len(embeddings) - 1) // 2)
        
        for i in range(min(100, len(embeddings))):
            for j in range(i + 1, min(i + 100, len(embeddings))):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        similarities = np.array(similarities[:sample_size])
        
        return {
            "min": float(np.min(similarities)),
            "max": float(np.max(similarities)),
            "mean": float(np.mean(similarities)),
            "median": float(np.median(similarities)),
            "std": float(np.std(similarities)),
            "percentile_25": float(np.percentile(similarities, 25)),
            "percentile_75": float(np.percentile(similarities, 75))
        }

# ====================== PERFORMANCE TRACKING ======================

class PerformanceTracker:
    """Track performance metrics"""
    
    def __init__(self):
        self.metrics = {}
    
    def start_timer(self, name: str):
        """Start timing operation"""
        import time
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append({'start': time.time(), 'end': None})
    
    def end_timer(self, name: str):
        """End timing operation"""
        import time
        if name in self.metrics and self.metrics[name]:
            self.metrics[name][-1]['end'] = time.time()
    
    def get_duration(self, name: str) -> float:
        """Get last operation duration"""
        if name in self.metrics and self.metrics[name]:
            m = self.metrics[name][-1]
            if m['end'] is not None:
                return m['end'] - m['start']
        return 0
    
    def get_stats(self, name: str) -> Dict:
        """Get stats for operation"""
        if name not in self.metrics:
            return {}
        
        durations = [
            m['end'] - m['start'] 
            for m in self.metrics[name] 
            if m['end'] is not None
        ]
        
        if not durations:
            return {}
        
        return {
            "operation": name,
            "count": len(durations),
            "total_time": sum(durations),
            "avg_time": np.mean(durations),
            "min_time": np.min(durations),
            "max_time": np.max(durations),
            "median_time": np.median(durations)
        }
    
    def get_all_stats(self) -> Dict:
        """Get all operation stats"""
        return {name: self.get_stats(name) for name in self.metrics}

# ====================== REPORT GENERATION ======================

class ReportGenerator:
    """Generate analysis reports"""
    
    @staticmethod
    def generate_video_report(video_id: str, frames_count: int, 
                             storage_used: float, processing_time: float) -> str:
        """Generate video processing report"""
        report = f"""
╔════════════════════════════════════════╗
║   Video Processing Report              ║
╚════════════════════════════════════════╝

Video ID:         {video_id}
Timestamp:        {datetime.now().isoformat()}

Processing Metrics:
  - Frames Extracted:    {frames_count}
  - Processing Time:     {processing_time:.2f}s
  - Avg Time/Frame:      {processing_time/frames_count:.3f}s
  - Storage Used:        {storage_used:.2f} MB
  - Avg Size/Frame:      {storage_used*1024/frames_count:.1f} KB

Throughput:
  - Frames/Second:       {frames_count/processing_time:.1f}
  - MB/Second:           {storage_used/processing_time:.2f}
"""
        return report
    
    @staticmethod
    def generate_search_report(query: str, results_count: int, 
                              search_time: float, similarity_scores: List[float]) -> str:
        """Generate search report"""
        if similarity_scores:
            avg_score = np.mean(similarity_scores)
            max_score = np.max(similarity_scores)
            min_score = np.min(similarity_scores)
        else:
            avg_score = max_score = min_score = 0
        
        report = f"""
╔════════════════════════════════════════╗
║   Search Report                        ║
╚════════════════════════════════════════╝

Query:            "{query}"
Timestamp:        {datetime.now().isoformat()}

Results:
  - Total Matches:       {results_count}
  - Search Time:         {search_time:.3f}s
  - Time/Result:         {search_time/results_count:.4f}s (if results > 0)

Similarity Scores:
  - Min:                 {min_score:.4f}
  - Max:                 {max_score:.4f}
  - Average:             {avg_score:.4f}
"""
        return report
    
    @staticmethod
    def generate_system_report(total_videos: int, total_frames: int,
                              storage_used: float, device: str) -> str:
        """Generate system report"""
        report = f"""
╔════════════════════════════════════════╗
║   System Report                        ║
╚════════════════════════════════════════╝

Generated:        {datetime.now().isoformat()}
Device:           {device}

Videos:
  - Total:                {total_videos}
  - Total Frames:         {total_frames}
  - Avg Frames/Video:     {total_frames/total_videos if total_videos > 0 else 0:.0f}

Storage:
  - Total Used:           {storage_used:.2f} MB
  - Avg/Video:            {storage_used/total_videos if total_videos > 0 else 0:.2f} MB
"""
        return report

# ====================== LOGGING UTILITIES ======================

class QueryLogger:
    """Log all queries for analysis"""
    
    def __init__(self, log_file: str = "queries.jsonl"):
        self.log_file = log_file
    
    def log(self, query: str, video_id: str, threshold: float, 
           result_count: int, duration: float):
        """Log query execution"""
        record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "video_id": video_id,
            "threshold": threshold,
            "result_count": result_count,
            "duration_ms": duration * 1000
        }
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception as e:
            logger.error(f"Error logging query: {e}")
    
    def get_logs(self, limit: int = 100) -> List[Dict]:
        """Get recent logs"""
        try:
            logs = []
            with open(self.log_file, 'r') as f:
                for line in f:
                    try:
                        logs.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            
            return logs[-limit:]
        except FileNotFoundError:
            return []
        except Exception as e:
            logger.error(f"Error reading logs: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics from logs"""
        logs = self.get_logs(1000)
        
        if not logs:
            return {}
        
        durations = [log['duration_ms'] for log in logs]
        
        return {
            "total_queries": len(logs),
            "avg_duration_ms": np.mean(durations),
            "min_duration_ms": np.min(durations),
            "max_duration_ms": np.max(durations),
            "median_duration_ms": np.median(durations),
            "total_results": sum(log['result_count'] for log in logs)
        }

# ====================== USAGE ======================

if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker()
    
    # Time an operation
    tracker.start_timer("process_video")
    import time
    time.sleep(0.1)
    tracker.end_timer("process_video")
    
    # Get stats
    stats = tracker.get_stats("process_video")
    print(stats)
    
    # Generate report
    report = ReportGenerator.generate_video_report(
        video_id="vid_12345",
        frames_count=240,
        storage_used=50.0,
        processing_time=5.2
    )
    print(report)
