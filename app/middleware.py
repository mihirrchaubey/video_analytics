import time
import logging
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
import traceback

logger = logging.getLogger(__name__)

# ====================== CUSTOM EXCEPTIONS ======================

class VideoAnalyticsException(Exception):
    """Base exception for application"""
    def __init__(self, message: str, status_code: int = 500, error_code: str = "INTERNAL_ERROR"):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        super().__init__(self.message)

class VideoProcessingError(VideoAnalyticsException):
    """Video processing error"""
    def __init__(self, message: str):
        super().__init__(message, 400, "VIDEO_PROCESSING_ERROR")

class InvalidQueryError(VideoAnalyticsException):
    """Invalid search query"""
    def __init__(self, message: str):
        super().__init__(message, 400, "INVALID_QUERY")

class VideoNotFoundError(VideoAnalyticsException):
    """Video not found"""
    def __init__(self, video_id: str):
        super().__init__(f"Video {video_id} not found", 404, "VIDEO_NOT_FOUND")

class InsufficientMemoryError(VideoAnalyticsException):
    """Out of memory"""
    def __init__(self):
        super().__init__("Insufficient memory for operation", 503, "INSUFFICIENT_MEMORY")

class ModelLoadError(VideoAnalyticsException):
    """Failed to load model"""
    def __init__(self, model_name: str):
        super().__init__(f"Failed to load model: {model_name}", 500, "MODEL_LOAD_ERROR")

# ====================== LOGGING MIDDLEWARE ======================

class LoggingMiddleware(BaseHTTPMiddleware):
    """Log all requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Log request
        logger.info(f"{request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Log response
            process_time = time.time() - start_time
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.3f}s"
            )
            
            response.headers["X-Process-Time"] = str(process_time)
            return response
            
        except Exception as e:
            logger.error(f"Error processing {request.method} {request.url.path}: {e}")
            raise

# ====================== PERFORMANCE MONITORING ======================

class PerformanceMiddleware(BaseHTTPMiddleware):
    """Monitor and track performance metrics"""
    
    def __init__(self, app, slow_request_threshold: float = 5.0):
        super().__init__(app)
        self.slow_request_threshold = slow_request_threshold
        self.metrics = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Track metrics
        endpoint = f"{request.method} {request.url.path}"
        if endpoint not in self.metrics:
            self.metrics[endpoint] = {
                "count": 0,
                "total_time": 0,
                "max_time": 0,
                "min_time": float('inf')
            }
        
        self.metrics[endpoint]["count"] += 1
        self.metrics[endpoint]["total_time"] += process_time
        self.metrics[endpoint]["max_time"] = max(self.metrics[endpoint]["max_time"], process_time)
        self.metrics[endpoint]["min_time"] = min(self.metrics[endpoint]["min_time"], process_time)
        
        # Log slow requests
        if process_time > self.slow_request_threshold:
            logger.warning(
                f"Slow request detected: {endpoint} took {process_time:.3f}s "
                f"(threshold: {self.slow_request_threshold}s)"
            )
        
        return response
    
    def get_metrics(self) -> dict:
        """Get performance metrics"""
        summary = {}
        for endpoint, data in self.metrics.items():
            avg_time = data["total_time"] / data["count"] if data["count"] > 0 else 0
            summary[endpoint] = {
                "requests": data["count"],
                "avg_time_s": round(avg_time, 3),
                "max_time_s": round(data["max_time"], 3),
                "min_time_s": round(data["min_time"], 3) if data["min_time"] != float('inf') else 0
            }
        return summary

# ====================== RATE LIMITING ======================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple rate limiting"""
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.client_requests = {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_ip = request.client.host
        current_minute = int(time.time() // 60)
        
        key = f"{client_ip}:{current_minute}"
        
        if key not in self.client_requests:
            self.client_requests[key] = 0
        
        self.client_requests[key] += 1
        
        if self.client_requests[key] > self.requests_per_minute:
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                {"error": "Rate limit exceeded"},
                status_code=429
            )
        
        # Cleanup old entries
        current_key = f"{client_ip}:{int(time.time() // 60)}"
        keys_to_delete = [k for k in self.client_requests.keys() if k != current_key]
        for k in keys_to_delete:
            del self.client_requests[k]
        
        return await call_next(request)

# ====================== ERROR HANDLERS ======================

async def video_analytics_exception_handler(request: Request, exc: VideoAnalyticsException):
    """Handle custom exceptions"""
    logger.error(f"{exc.error_code}: {exc.message}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "message": exc.message,
            "path": str(request.url.path)
        }
    )

async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}\n{traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "path": str(request.url.path)
        }
    )

async def http_exception_handler(request: Request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP_ERROR",
            "message": str(exc.detail),
            "path": str(request.url.path)
        }
    )

# ====================== SETUP FUNCTIONS ======================

def setup_exception_handlers(app):
    """Register exception handlers"""
    app.add_exception_handler(VideoAnalyticsException, video_analytics_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

def setup_middleware(app, enable_cors: bool = True, 
                     enable_logging: bool = True,
                     enable_performance: bool = True,
                     enable_rate_limit: bool = False):
    """Setup all middleware"""
    
    # CORS
    if enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    # Performance monitoring
    if enable_performance:
        app.add_middleware(PerformanceMiddleware)
    
    # Rate limiting
    if enable_rate_limit:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # Logging
    if enable_logging:
        app.add_middleware(LoggingMiddleware)

# ====================== MONITORING ======================

class HealthMonitor:
    """Monitor system health"""
    
    def __init__(self):
        self.last_check = time.time()
        self.status = {
            "database": "unknown",
            "storage": "unknown",
            "models": "unknown",
            "memory": "unknown"
        }
    
    def check_database(self, db_session) -> bool:
        """Check database connectivity"""
        try:
            db_session.execute("SELECT 1")
            self.status["database"] = "healthy"
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            self.status["database"] = "unhealthy"
            return False
    
    def check_storage(self, storage_path: str) -> bool:
        """Check storage accessibility"""
        try:
            import os
            if os.access(storage_path, os.W_OK):
                self.status["storage"] = "healthy"
                return True
        except Exception as e:
            logger.error(f"Storage health check failed: {e}")
        
        self.status["storage"] = "unhealthy"
        return False
    
    def check_memory(self) -> bool:
        """Check available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            if memory.percent > 90:
                self.status["memory"] = "warning"
                logger.warning(f"High memory usage: {memory.percent}%")
            else:
                self.status["memory"] = "healthy"
            
            return True
        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            self.status["memory"] = "unknown"
            return False
    
    def get_status(self) -> dict:
        """Get overall system status"""
        return {
            "timestamp": time.time(),
            "components": self.status,
            "overall": "healthy" if all(v != "unhealthy" for v in self.status.values()) else "unhealthy"
        }

# ====================== ENDPOINT ======================

def add_health_endpoint(app):
    """Add health check endpoint"""
    monitor = HealthMonitor()
    
    @app.get("/health")
    async def health_check():
        """System health check endpoint"""
        return monitor.get_status()
    
    @app.get("/metrics")
    async def get_metrics():
        """Get performance metrics"""
        if hasattr(app, 'performance_middleware'):
            return app.performance_middleware.get_metrics()
        return {}
    
    return monitor
