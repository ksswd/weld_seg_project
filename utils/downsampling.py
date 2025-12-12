import torch

@torch.no_grad()
def fps(xyz: torch.Tensor, M: int):
    """
    xyz: (N, 3) float tensor on CUDA
    M: number of samples

    return index: (M,) long tensor, selected indices
    """
    N = xyz.shape[0]
    if M >= N:
        return torch.arange(N, device=xyz.device)
    centroids = torch.zeros(M, dtype=torch.long, device=xyz.device)  
    distances = torch.full((N,), 1e10, device=xyz.device)  
    farthest = torch.randint(0, N, (1,), device=xyz.device)
    for i in range(M):
        centroids[i] = farthest
        centroid = xyz[farthest].view(1, 3)                  # (1,3)
        dist = torch.sum((xyz - centroid)**2, dim=1)         # (N,)
        distances = torch.minimum(distances, dist)
        farthest = torch.argmax(distances)
    return centroids

@torch.no_grad()
def sample_with_fps(item: dict, M: int):
    """
    item: 一个数据样本的 dict，包含 features, normals, curvature 等字段
    M: 采样后的点数

    return: 新的 item dict（所有字段都同步采样）
    """
    xyz = torch.from_numpy(item["features"][:, :3]).cuda()   # (N,3)
    # Get FPS indices
    idx = fps(xyz, M)    # (M,)
    out = {}
    for key, val in item.items():
        if isinstance(val, (list, tuple)):
            out[key] = val
            continue
        arr = torch.from_numpy(val) if not torch.is_tensor(val) else val
        arr = arr.cuda()
        if arr.ndim == 2:
            # (N, C)
            out[key] = arr[idx]
        else:
            out[key] = arr
    return out
