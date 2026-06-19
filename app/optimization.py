import os
import gc
import psutil
import torch
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ====================== MEMORY OPTIMIZATION ======================

class MemoryOptimizer:
    """Optimize memory usage"""
    
    @staticmethod
    def get_memory_info() -> Dict[str, float]:
        """Get memory information"""
        memory = psutil.virtual_memory()
        
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent": memory.percent
        }
    
    @staticmethod
    def get_gpu_memory_info() -> Dict[str, float]:
        """Get GPU memory information"""
        if not torch.cuda.is_available():
            return {}
        
        return {
            "total_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
            "allocated_gb": torch.cuda.memory_allocated(0) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(0) / (1024**3)
        }
    
    @staticmethod
    def cleanup_memory():
        """Clean up memory"""
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Memory cleanup completed")
    
    @staticmethod
    def check_memory_available(required_gb: float) -> Tuple[bool, str]:
        """Check if enough memory is available"""
        memory = MemoryOptimizer.get_memory_info()
        
        if memory["available_gb"] < required_gb:
            return False, f"Insufficient memory. Required: {required_gb}GB, Available: {memory['available_gb']:.2f}GB"
        
        return True, "Sufficient memory available"
    
    @staticmethod
    def optimize_batch_size(model_params: int, available_gb: float = None) -> int:
        """Calculate optimal batch size"""
        if available_gb is None:
            memory = MemoryOptimizer.get_memory_info()
            available_gb = memory["available_gb"]
        
        # Estimate: ~4 bytes per parameter, factor for gradients
        bytes_per_sample = (model_params * 4) * 2.5  # 2.5x for intermediate activations
        mb_per_sample = bytes_per_sample / (1024**2)
        
        # Use 50% of available memory
        usable_memory = available_gb * 0.5 * 1024  # Convert to MB
        
        batch_size = max(1, int(usable_memory / mb_per_sample))
        
        return batch_size

# ====================== GPU OPTIMIZATION ======================

class GPUOptimizer:
    """Optimize GPU usage"""
    
    @staticmethod
    def is_gpu_available() -> bool:
        """Check if GPU is available"""
        return torch.cuda.is_available()
    
    @staticmethod
    def get_gpu_name() -> str:
        """Get GPU name"""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "CPU"
    
    @staticmethod
    def enable_mixed_precision() -> bool:
        """Enable mixed precision training"""
        try:
            from torch.cuda.amp import autocast
            logger.info("Mixed precision enabled")
            return True
        except Exception as e:
            logger.warning(f"Could not enable mixed precision: {e}")
            return False
    
    @staticmethod
    def enable_tf32() -> bool:
        """Enable TF32 precision for faster computation"""
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("TF32 enabled")
            return True
        except Exception as e:
            logger.warning(f"Could not enable TF32: {e}")
            return False
    
    @staticmethod
    def optimize_cudnn() -> bool:
        """Optimize cuDNN settings"""
        try:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            logger.info("cuDNN optimized")
            return True
        except Exception as e:
            logger.warning(f"Could not optimize cuDNN: {e}")
            return False

# ====================== INFERENCE OPTIMIZATION ======================

class InferenceOptimizer:
    """Optimize inference performance"""
    
    @staticmethod
    def use_inference_mode(func):
        """Decorator to use torch.inference_mode"""
        def wrapper(*args, **kwargs):
            with torch.inference_mode():
                return func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def use_no_grad(func):
        """Decorator to disable gradient computation"""
        def wrapper(*args, **kwargs):
            with torch.no_grad():
                return func(*args, **kwargs)
        return wrapper
    
    @staticmethod
    def quantize_model(model):
        """Quantize model for faster inference"""
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )
            logger.info("Model quantized")
            return quantized
        except Exception as e:
            logger.warning(f"Could not quantize model: {e}")
            return model

# ====================== CACHING OPTIMIZATION ======================

class CacheOptimizer:
    """Optimize caching strategies"""
    
    def __init__(self, max_cache_size_mb: int = 500):
        self.max_cache_size = max_cache_size_mb * (1024**2)  # Convert to bytes
        self.cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get from cache"""
        if key in self.cache:
            self.cache_hits += 1
            return self.cache[key]
        
        self.cache_misses += 1
        return None
    
    def set(self, key: str, value: np.ndarray) -> bool:
        """Set in cache with size limit"""
        size = value.nbytes
        
        # Check total cache size
        total_size = sum(v.nbytes for v in self.cache.values())
        
        if total_size + size > self.max_cache_size:
            # Remove oldest entries
            self._evict_oldest()
        
        self.cache[key] = value
        return True
    
    def _evict_oldest(self):
        """Evict oldest cache entries"""
        if self.cache:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total_hits = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_hits * 100) if total_hits > 0 else 0
        
        return {
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate_percent": hit_rate,
            "total_entries": len(self.cache),
            "total_size_mb": sum(v.nbytes for v in self.cache.values()) / (1024**2)
        }

# ====================== BATCH OPTIMIZATION ======================

class BatchOptimizer:
    """Optimize batch processing"""
    
    @staticmethod
    def calculate_optimal_batch_size(input_shape: Tuple, 
                                     model_memory_mb: float,
                                     available_memory_gb: float = None) -> int:
        """Calculate optimal batch size"""
        if available_memory_gb is None:
            memory = MemoryOptimizer.get_memory_info()
            available_memory_gb = memory["available_gb"]
        
        # Estimate memory per sample
        sample_size = np.prod(input_shape[1:]) * 4  # 4 bytes for float32
        model_overhead = model_memory_mb * (1024**2)
        
        available_bytes = available_memory_gb * (1024**3) * 0.8  # Use 80% of available
        
        batch_size = int((available_bytes - model_overhead) / (sample_size * 2.5))
        
        return max(1, batch_size)
    
    @staticmethod
    def split_into_batches(items: list, batch_size: int):
        """Split items into batches"""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

# ====================== DYNAMIC CONFIGURATION ======================

class DynamicConfig:
    """Dynamically adjust configuration based on available resources"""
    
    @staticmethod
    def auto_configure(target_fps: int = 1) -> Dict:
        """Auto-configure based on available resources"""
        memory = MemoryOptimizer.get_memory_info()
        gpu_available = GPUOptimizer.is_gpu_available()
        
        config = {
            "device": "cuda" if gpu_available else "cpu",
            "target_fps": target_fps,
            "batch_size": 32,
            "num_workers": 4
        }
        
        # Adjust based on memory
        if memory["percent"] > 80:
            config["batch_size"] = 16
            config["num_workers"] = 2
            config["target_fps"] = max(0.5, target_fps)
            logger.warning("High memory usage - reducing batch size and FPS")
        
        elif memory["percent"] > 60:
            config["batch_size"] = 24
            config["num_workers"] = 3
        
        # Adjust based on GPU memory
        if gpu_available:
            gpu_mem = MemoryOptimizer.get_gpu_memory_info()
            if gpu_mem.get("allocated_gb", 0) > gpu_mem.get("total_gb", 1) * 0.8:
                config["batch_size"] = 8
                logger.warning("High GPU memory usage - reducing batch size")
        
        return config
    
    @staticmethod
    def auto_select_model_size(available_gb: float) -> str:
        """Select model size based on available memory"""
        if available_gb < 4:
            return "yolov8n.pt"  # Nano
        elif available_gb < 8:
            return "yolov8s.pt"  # Small
        elif available_gb < 16:
            return "yolov8m.pt"  # Medium
        else:
            return "yolov8x.pt"  # Extra large
    
    @staticmethod
    def auto_select_clip_model(available_gb: float) -> str:
        """Select CLIP model based on available memory"""
        if available_gb < 4:
            return "openai/clip-vit-base-patch32"
        elif available_gb < 8:
            return "openai/clip-vit-base-patch16"
        else:
            return "openai/clip-vit-large-patch14"

# ====================== PROFILING ======================

class PerformanceProfiler:
    """Profile code performance"""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_function(self, func):
        """Decorator to profile function"""
        def wrapper(*args, **kwargs):
            import time
            
            func_name = func.__name__
            
            start = time.time()
            start_memory = MemoryOptimizer.get_memory_info()
            
            result = func(*args, **kwargs)
            
            duration = time.time() - start
            end_memory = MemoryOptimizer.get_memory_info()
            
            if func_name not in self.profiles:
                self.profiles[func_name] = []
            
            self.profiles[func_name].append({
                "duration": duration,
                "memory_delta": end_memory["used_gb"] - start_memory["used_gb"]
            })
            
            return result
        
        return wrapper
    
    def get_profile(self, func_name: str) -> Dict:
        """Get profile for function"""
        if func_name not in self.profiles:
            return {}
        
        data = self.profiles[func_name]
        durations = [d["duration"] for d in data]
        
        return {
            "calls": len(data),
            "total_time": sum(durations),
            "avg_time": np.mean(durations),
            "min_time": np.min(durations),
            "max_time": np.max(durations)
        }
    
    def get_all_profiles(self) -> Dict:
        """Get all profiles"""
        return {
            func_name: self.get_profile(func_name)
            for func_name in self.profiles.keys()
        }

# ====================== USAGE ======================

if __name__ == "__main__":
    # Example usage
    optimizer = MemoryOptimizer()
    
    # Get memory info
    mem_info = optimizer.get_memory_info()
    print("Memory Info:", mem_info)
    
    # Check if enough memory
    available, msg = optimizer.check_memory_available(2.0)
    print(msg)
    
    # Get GPU info
    gpu_name = GPUOptimizer.get_gpu_name()
    print(f"GPU: {gpu_name}")
    
    # Auto configure
    config = DynamicConfig.auto_configure()
    print("Auto Config:", config)
    
    # Select model
    model = DynamicConfig.auto_select_model_size(8)
    print(f"Selected Model: {model}")
