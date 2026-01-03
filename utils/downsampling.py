# weld_seg_project/utils/downsampling.py
# Farthest Point Sampling (FPS) with cache for efficient downsampling
import numpy as np
from typing import Tuple

# Global cache: key = (points_hash, k), value = indices
# Using hash of points array shape and first/last few points to identify identical inputs
_FPS_CACHE = {}
_CACHE_SIZE_LIMIT = 1000  # Limit cache size to avoid memory issues


def _hash_points(points: np.ndarray) -> str:
    """Create a hash key for points array based on shape and sample points."""
    # Use shape and a sample of points for hashing (not full array to save time)
    n = points.shape[0]
    sample_indices = [0, n // 4, n // 2, 3 * n // 4, n - 1] if n > 5 else list(range(n))
    sample_points = points[sample_indices].tobytes()
    return f"{points.shape}:{hash(sample_points)}"


def fps_with_cache(points: np.ndarray, k: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) with caching.
    
    Args:
        points: (N, 3) numpy array of 3D points
        k: number of points to sample (must be <= N)
    
    Returns:
        indices: (k,) numpy array of indices into points array
    """
    if k <= 0:
        raise ValueError(f"k must be > 0, got {k}")
    
    n = points.shape[0]
    if k >= n:
        return np.arange(n, dtype=np.int64)
    
    # Create cache key
    cache_key = (_hash_points(points), k)
    
    # Check cache
    if cache_key in _FPS_CACHE:
        return _FPS_CACHE[cache_key].copy()
    
    # Compute FPS
    indices = _fps_impl(points, k)
    
    # Cache result (with size limit)
    if len(_FPS_CACHE) >= _CACHE_SIZE_LIMIT:
        # Clear cache if too large (simple strategy: clear all)
        _FPS_CACHE.clear()
    _FPS_CACHE[cache_key] = indices
    
    return indices.copy()


def _fps_impl(points: np.ndarray, k: int) -> np.ndarray:
    """
    Core FPS implementation (no caching).
    
    Algorithm:
    1. Start with a random point (or first point)
    2. Iteratively select the point farthest from all selected points
    3. Use squared distances for efficiency
    """
    n = points.shape[0]
    indices = np.zeros(k, dtype=np.int64)
    distances = np.full(n, np.inf, dtype=np.float32)
    
    # Start with first point (or random for more diverse results)
    current = 0
    indices[0] = current
    
    # Iteratively select farthest points
    for i in range(1, k):
        # Compute distances from current point to all points
        point_current = points[current]
        dists_new = np.sum((points - point_current) ** 2, axis=1)
        
        # Update minimum distances (each point's distance to nearest selected point)
        distances = np.minimum(distances, dists_new)
        
        # Select farthest point (largest minimum distance)
        current = np.argmax(distances)
        indices[i] = current
    
    return indices

