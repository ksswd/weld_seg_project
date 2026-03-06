# 分块掩码预训练方案 V2（最终版）

## 核心任务

**输入：xyz坐标**  
**输出：曲率（curvature）**

## 分块策略

### 方案：固定大小块 + 高曲率点计数

#### 1. 块划分方法
```python
# 方法A：网格划分（推荐）
# 将点云空间划分成固定大小的网格块
block_size = 0.01  # 每个块的边长（米）
grid_blocks = divide_into_grid(points, block_size)

# 方法B：固定点数块
# 将点云分成固定点数的块（使用FPS或随机）
points_per_block = 1000
fixed_size_blocks = divide_into_fixed_size(points, points_per_block)
```

#### 2. 块分类
```python
def classify_block(block_points, block_curvature, high_curv_threshold=0.01, min_high_curv_points=5):
    """
    根据块内高曲率点数量分类块
    
    Args:
        block_points: 块内的点坐标
        block_curvature: 块内点的曲率
        high_curv_threshold: 高曲率阈值
        min_high_curv_points: 高曲率点的最小数量阈值
    
    Returns:
        'weld' or 'background'
    """
    high_curv_mask = block_curvature > high_curv_threshold
    num_high_curv = high_curv_mask.sum()
    
    if num_high_curv >= min_high_curv_points:
        return 'weld'  # 高曲率块（焊缝块）
    else:
        return 'background'  # 背景块
```

#### 3. 块间部分处理
- **不考虑块间部分**：只对完整的块进行分类和mask
- 块边界外的点直接忽略

## Mask策略

### 策略3：交替mask（按epoch）

```python
def generate_block_mask(blocks, block_labels, epoch, mask_ratio=0.3):
    """
    交替mask策略
    
    Args:
        blocks: List[List[int]] - 每个块是点的索引列表
        block_labels: List[str] - 每个块的标签 ('weld' or 'background')
        epoch: 当前epoch
        mask_ratio: mask的块比例
    
    Returns:
        mask: (N,) bool - 点的mask（True表示被mask）
    """
    if epoch % 2 == 0:
        # 偶数epoch：mask焊缝块
        target_blocks = [i for i, label in enumerate(block_labels) if label == 'weld']
    else:
        # 奇数epoch：mask背景块
        target_blocks = [i for i, label in enumerate(block_labels) if label == 'background']
    
    # 随机选择要mask的块
    num_blocks_to_mask = max(1, int(len(target_blocks) * mask_ratio))
    masked_block_ids = random.sample(target_blocks, min(num_blocks_to_mask, len(target_blocks)))
    
    # 生成点的mask
    mask = torch.zeros(N, dtype=torch.bool)
    for block_id in masked_block_ids:
        mask[blocks[block_id]] = True
    
    return mask
```

## Loss计算

### 按块计算 + 点数加权平均

```python
def block_recon_criterion(recon, batch, mask, blocks, block_labels):
    """
    块级重建损失（预测曲率）
    
    Args:
        recon: (B, N, 1) - 预测的曲率
        batch: 包含curvature的batch
        curvature: (B, N, 1) - 真实曲率
        mask: (B, N) - 点的mask
        blocks: List[List[int]] - 每个样本的块列表
        block_labels: List[str] - 每个块的标签
    
    Returns:
        loss: scalar
    """
    curvature = batch['curvature']  # (B, N, 1)
    
    # 只计算被mask块的loss
    block_losses = []
    block_weights = []
    
    for b in range(B):  # 遍历batch
        sample_blocks = blocks[b]
        sample_labels = block_labels[b]
        sample_mask = mask[b]  # (N,)
        sample_recon = recon[b]  # (N, 1)
        sample_curv = curvature[b]  # (N, 1)
        
        for block_id, block_indices in enumerate(sample_blocks):
            # 检查这个块是否被mask
            block_mask = sample_mask[block_indices]
            if block_mask.sum() == 0:
                continue  # 这个块没有被mask，跳过
            
            # 计算这个块的loss
            block_recon = sample_recon[block_indices][block_mask]  # (M, 1)
            block_curv = sample_curv[block_indices][block_mask]  # (M, 1)
            
            # MSE loss for this block
            block_loss = F.mse_loss(block_recon, block_curv, reduction='mean')
            
            # 权重 = 块内被mask的点数
            block_weight = block_mask.sum().float()
            
            block_losses.append(block_loss)
            block_weights.append(block_weight)
    
    # 按点数加权平均
    if len(block_losses) == 0:
        return torch.tensor(0.0, device=recon.device)
    
    block_losses = torch.stack(block_losses)
    block_weights = torch.stack(block_weights)
    
    # 加权平均
    total_loss = (block_losses * block_weights).sum() / block_weights.sum()
    
    return total_loss
```

## 可视化方案

### 1. 块颜色标注

```python
def visualize_blocks(points, blocks, block_labels, save_path):
    """
    用不同颜色标注每个块，输出点云
    
    Args:
        points: (N, 3) - 点坐标
        blocks: List[List[int]] - 每个块是点的索引列表
        block_labels: List[str] - 每个块的标签
        save_path: 保存路径
    """
    import open3d as o3d
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 为每个块分配颜色
    colors = np.zeros((len(points), 3))
    
    # 定义颜色映射
    weld_color = [1.0, 0.0, 0.0]  # 红色 - 焊缝块
    background_color = [0.0, 0.0, 1.0]  # 蓝色 - 背景块
    
    for block_id, block_indices in enumerate(blocks):
        if block_labels[block_id] == 'weld':
            colors[block_indices] = weld_color
        else:
            colors[block_indices] = background_color
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved block visualization to {save_path}")
```

### 2. 预测结果可视化

```python
def visualize_predictions(points, blocks, block_labels, pred_curvature, gt_curvature, mask, save_path):
    """
    可视化预测结果
    
    Args:
        points: (N, 3) - 点坐标
        blocks: List[List[int]] - 每个块是点的索引列表
        block_labels: List[str] - 每个块的标签
        pred_curvature: (N, 1) - 预测的曲率
        gt_curvature: (N, 1) - 真实曲率
        mask: (N,) - 点的mask
        save_path: 保存路径
    """
    import open3d as o3d
    
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 计算误差
    error = np.abs(pred_curvature - gt_curvature)
    
    # 只对被mask的点着色（显示预测误差）
    colors = np.ones((len(points), 3)) * 0.5  # 灰色 - 未mask的点
    
    # 对被mask的点，用误差值着色
    masked_error = error[mask]
    if len(masked_error) > 0:
        # 归一化误差到[0, 1]
        error_min, error_max = masked_error.min(), masked_error.max()
        if error_max > error_min:
            normalized_error = (masked_error - error_min) / (error_max - error_min)
        else:
            normalized_error = np.zeros_like(masked_error)
        
        # 使用colormap（红色=高误差，绿色=低误差）
        colors[mask] = np.stack([
            normalized_error,  # R
            1 - normalized_error,  # G
            np.zeros_like(normalized_error)  # B
        ], axis=1)
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 保存
    o3d.io.write_point_cloud(save_path, pcd)
    print(f"Saved prediction visualization to {save_path}")
```

## 实现细节

### 1. BlockSegmenter类

```python
class BlockSegmenter:
    def __init__(self, 
                 block_size=0.01,  # 网格块大小（米）
                 method='grid',  # 'grid' or 'fixed_points'
                 high_curv_threshold=0.01,  # 高曲率阈值
                 min_high_curv_points=5,  # 高曲率点的最小数量
                 points_per_block=100):  # 固定点数块的点数
        self.block_size = block_size
        self.method = method
        self.high_curv_threshold = high_curv_threshold
        self.min_high_curv_points = min_high_curv_points
        self.points_per_block = points_per_block
    
    def segment(self, points, curvature):
        """
        将点云分割成块并分类
        
        Args:
            points: (N, 3) - 点坐标
            curvature: (N, 1) - 曲率
        
        Returns:
            blocks: List[List[int]] - 每个块是点的索引列表
            block_labels: List[str] - 每个块的标签
        """
        if self.method == 'grid':
            blocks = self._grid_segment(points)
        else:
            blocks = self._fixed_points_segment(points)
        
        # 分类每个块
        block_labels = []
        for block_indices in blocks:
            block_curv = curvature[block_indices]
            label = self._classify_block(block_curv)
            block_labels.append(label)
        
        return blocks, block_labels
    
    def _grid_segment(self, points):
        """网格划分"""
        # 计算网格范围
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        
        # 计算网格数量
        grid_size = (max_coords - min_coords) / self.block_size
        grid_size = np.ceil(grid_size).astype(int)
        
        # 将点分配到网格
        grid_indices = ((points - min_coords) / self.block_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)
        
        # 构建块
        blocks = {}
        for i, (x, y, z) in enumerate(grid_indices):
            key = (x, y, z)
            if key not in blocks:
                blocks[key] = []
            blocks[key].append(i)
        
        # 转换为列表
        blocks = list(blocks.values())
        
        return blocks
    
    def _fixed_points_segment(self, points):
        """固定点数划分"""
        N = len(points)
        blocks = []
        
        # 使用FPS或随机采样
        for i in range(0, N, self.points_per_block):
            end_idx = min(i + self.points_per_block, N)
            block_indices = list(range(i, end_idx))
            blocks.append(block_indices)
        
        return blocks
    
    def _classify_block(self, block_curvature):
        """分类块"""
        high_curv_mask = block_curvature > self.high_curv_threshold
        num_high_curv = high_curv_mask.sum()
        
        if num_high_curv >= self.min_high_curv_points:
            return 'weld'
        else:
            return 'background'
```

### 2. BlockMasker类

```python
class BlockMasker:
    def __init__(self, mask_ratio=0.3, strategy='alternate'):
        self.mask_ratio = mask_ratio
        self.strategy = strategy
    
    def generate_mask(self, blocks, block_labels, epoch=None):
        """
        生成块级mask
        
        Args:
            blocks: List[List[int]] - 每个块是点的索引列表
            block_labels: List[str] - 每个块的标签
            epoch: 当前epoch（用于交替策略）
        
        Returns:
            mask: (N,) bool - 点的mask
        """
        N = sum(len(block) for block in blocks)
        mask = torch.zeros(N, dtype=torch.bool)
        
        if self.strategy == 'alternate':
            if epoch is None:
                epoch = 0
            
            if epoch % 2 == 0:
                # 偶数epoch：mask焊缝块
                target_blocks = [i for i, label in enumerate(block_labels) if label == 'weld']
            else:
                # 奇数epoch：mask背景块
                target_blocks = [i for i, label in enumerate(block_labels) if label == 'background']
            
            # 随机选择要mask的块
            num_blocks_to_mask = max(1, int(len(target_blocks) * self.mask_ratio))
            masked_block_ids = random.sample(target_blocks, min(num_blocks_to_mask, len(target_blocks)))
            
            # 生成点的mask
            for block_id in masked_block_ids:
                for point_idx in blocks[block_id]:
                    mask[point_idx] = True
        
        return mask
```

### 3. 修改模型输出

```python
# 修改recon_head，只输出曲率（1维）
self.recon_head = nn.Sequential(
    nn.Linear(config.D_MODEL, config.D_MODEL * 2),
    nn.ReLU(),
    nn.Linear(config.D_MODEL * 2, config.D_MODEL),
    nn.ReLU(),
    nn.Linear(config.D_MODEL, 1)  # 只输出曲率
)
```

### 4. 修改训练流程

```python
# 在pretrain.py中
def run_pretrain(config):
    # ... 初始化 ...
    
    # 初始化块分割器
    segmenter = BlockSegmenter(
        block_size=config.BLOCK_SIZE,
        method=config.BLOCK_METHOD,
        high_curv_threshold=config.HIGH_CURV_THRESHOLD,
        min_high_curv_points=config.MIN_HIGH_CURV_POINTS
    )
    
    # 初始化块masker
    block_masker = BlockMasker(
        mask_ratio=config.BLOCK_MASK_RATIO,
        strategy='alternate'
    )
    
    for epoch in range(config.NUM_EPOCHS):
        for batch in train_loader:
            # 对每个样本进行块分割
            blocks_list = []
            block_labels_list = []
            
            for b in range(batch['features'].shape[0]):
                points = batch['coordinate'][b].cpu().numpy()
                curvature = batch['curvature'][b].cpu().numpy()
                valid_mask = batch['mask'][b].cpu().numpy()
                
                # 只对有效点进行分割
                valid_points = points[valid_mask]
                valid_curvature = curvature[valid_mask]
                
                # 块分割
                blocks, block_labels = segmenter.segment(valid_points, valid_curvature)
                
                # 转换回原始索引
                valid_indices = np.where(valid_mask)[0]
                blocks_original = [[valid_indices[idx] for idx in block] for block in blocks]
                
                blocks_list.append(blocks_original)
                block_labels_list.append(block_labels)
            
            # 生成mask
            masks = []
            for blocks, block_labels in zip(blocks_list, block_labels_list):
                mask = block_masker.generate_mask(blocks, block_labels, epoch)
                masks.append(mask)
            
            # 前向传播
            # ... model forward ...
            
            # 计算loss
            loss = block_recon_criterion(recon, batch, masks, blocks_list, block_labels_list)
            
            # ... backward and update ...
```

## 配置参数

```python
# 添加到config.py
# === 块级预训练参数 ===
BLOCK_SIZE = 0.01  # 网格块大小（米）
BLOCK_METHOD = "grid"  # "grid" or "fixed_points"
HIGH_CURV_THRESHOLD = 0.01  # 高曲率阈值
MIN_HIGH_CURV_POINTS = 5  # 高曲率点的最小数量
BLOCK_MASK_RATIO = 0.3  # mask的块比例
POINTS_PER_BLOCK = 100  # 固定点数块的点数（如果使用fixed_points方法）
```

## 可视化脚本

```python
# scripts/visualize_blocks.py
def visualize_blocks_from_csv(csv_path, config, save_dir):
    """从CSV文件加载数据，可视化块分割"""
    # 加载数据
    data = load_features_from_csv(csv_path)
    points = data['coordinate']
    curvature = data['curvature']
    
    # 块分割
    segmenter = BlockSegmenter(
        block_size=config.BLOCK_SIZE,
        method=config.BLOCK_METHOD,
        high_curv_threshold=config.HIGH_CURV_THRESHOLD,
        min_high_curv_points=config.MIN_HIGH_CURV_POINTS
    )
    blocks, block_labels = segmenter.segment(points, curvature)
    
    # 可视化
    visualize_blocks(points, blocks, block_labels, 
                     os.path.join(save_dir, 'blocks.ply'))
    
    print(f"Total blocks: {len(blocks)}")
    print(f"Weld blocks: {sum(1 for label in block_labels if label == 'weld')}")
    print(f"Background blocks: {sum(1 for label in block_labels if label == 'background')}")
```

## 总结

### 关键点
1. **任务**：输入xyz → 输出曲率
2. **分块**：固定大小块 + 高曲率点计数分类
3. **Mask**：交替mask（偶数epoch mask焊缝块，奇数epoch mask背景块）
4. **Loss**：按块计算，点数加权平均
5. **可视化**：不同颜色标注块，输出点云

### 实现步骤
1. 实现`BlockSegmenter`类
2. 实现`BlockMasker`类
3. 实现`block_recon_criterion`函数
4. 修改`recon_head`输出（1维）
5. 修改训练流程集成块分割
6. 实现可视化脚本
7. 测试和调参
