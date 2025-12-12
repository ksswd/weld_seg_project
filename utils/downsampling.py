import numpy as np
import hashlib
_FPS_CACHE = {}
def _hash_points(points_np: np.ndarray):
    """给点云生成 hash，用于缓存 key"""
    return hashlib.sha256(points_np.tobytes()).hexdigest()

def fps_numpy(points: np.ndarray, k: int):
    """
    最远点采样（FPS）算法的 Numpy 实现
    """
    N = points.shape[0]
    if k >= N:
        return np.arange(N, dtype=np.int64)
    dists = np.ones(N) * 1e10
    idx = np.zeros(k, dtype=np.int64)
    current = 0
    for i in range(k):
        idx[i] = current
        dist = np.sum((points - points[current]) ** 2, axis=1)
        dists = np.minimum(dists, dist)
        current = np.argmax(dists)
    return idx


def fps_with_cache(points: np.ndarray, k: int):
    """
    带缓存的最远点采样（FPS）
    """
    assert points.ndim == 2 and points.shape[1] == 3
    key = (_hash_points(points), k)
    if key in _FPS_CACHE:
        idx_np = _FPS_CACHE[key]
    else:
        idx_np = fps_numpy(points, k)
        _FPS_CACHE[key] = idx_np
    return idx_np
