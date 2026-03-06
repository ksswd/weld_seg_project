# 分块掩码预训练方案

## 核心思路

将点云根据曲率分割成不同的块（焊缝区域 vs 非焊缝区域），然后对某些块进行mask，让模型预测被mask块的xyz坐标。

## 方案设计

### 1. 分块策略（Block Segmentation）

#### 方案A：基于曲率阈值的简单分割
```python
# 根据曲率阈值将点云分成两类
high_curvature_blocks = points[curvature > threshold]  # 焊缝区域
low_curvature_blocks = points[curvature <= threshold]   # 非焊缝区域
```

**优点**：
- 简单直接，计算快
- 符合焊缝检测的语义（高曲率=焊缝）

**缺点**：
- 可能产生不连续的块（同一区域可能被分成多个小块）
- 阈值需要手动调整

#### 方案B：曲率 + 空间聚类（推荐）
```python
# 1. 根据曲率阈值初步分类
high_curv_mask = curvature > threshold_high
low_curv_mask = curvature < threshold_low

# 2. 对高曲率点进行空间聚类（DBSCAN/K-means）
#    将高曲率点聚成连续的块（焊缝块）
high_curv_points = points[high_curv_mask]
weld_blocks = cluster_spatial(high_curv_points, method='dbscan')

# 3. 对低曲率点也进行聚类（背景块）
low_curv_points = points[low_curv_mask]
background_blocks = cluster_spatial(low_curv_points, method='dbscan')
```

**优点**：
- 产生连续的、有意义的块
- 更符合点云的空间结构
- 可以控制块的大小

**缺点**：
- 计算复杂度稍高
- 需要调整聚类参数

#### 方案C：纯空间聚类 + 曲率特征
```python
# 1. 先进行空间聚类（不考虑曲率）
blocks = cluster_spatial(points, method='dbscan')

# 2. 计算每个块的平均曲率
block_curvature = [mean(curvature[block]) for block in blocks]

# 3. 根据平均曲率分类
weld_blocks = [block for block, curv in zip(blocks, block_curvature) if curv > threshold]
background_blocks = [block for block, curv in zip(blocks, block_curvature) if curv <= threshold]
```

**优点**：
- 保证块的空间连续性
- 可以处理复杂的点云结构

**缺点**：
- 可能将焊缝和背景混在一起

### 2. Mask策略（Block-wise Masking）

#### 策略1：随机mask块
```python
# 随机选择一定比例的块进行mask
num_blocks_to_mask = int(len(blocks) * mask_ratio)
masked_block_ids = random.sample(range(len(blocks)), num_blocks_to_mask)
```

#### 策略2：优先mask高曲率块（焊缝块）
```python
# 优先mask焊缝块，让模型学习从背景预测焊缝
weld_block_ids = [i for i, block in enumerate(blocks) if is_weld_block(block)]
num_weld_to_mask = int(len(weld_block_ids) * weld_mask_ratio)
masked_weld_ids = random.sample(weld_block_ids, num_weld_to_mask)

# 也mask一些背景块
background_block_ids = [i for i, block in enumerate(blocks) if not is_weld_block(block)]
num_bg_to_mask = int(len(background_block_ids) * bg_mask_ratio)
masked_bg_ids = random.sample(background_block_ids, num_bg_to_mask)
```

#### 策略3：交替mask（推荐）
```python
# 交替mask焊缝块和背景块，让模型学习双向预测
if epoch % 2 == 0:
    # 偶数epoch：mask焊缝块，预测焊缝
    masked_block_ids = select_weld_blocks(blocks, mask_ratio)
else:
    # 奇数epoch：mask背景块，预测背景
    masked_block_ids = select_background_blocks(blocks, mask_ratio)
```

### 3. 预测目标

#### 选项A：只预测xyz坐标（简化版）
```python
# 只预测被mask块的xyz坐标
target = xyz[masked_blocks]
prediction = model(features, masked_features)
loss = MSE(prediction, target)
```

**优点**：
- 任务简单，容易训练
- 专注于几何重建

**缺点**：
- 可能信息不够丰富

#### 选项B：预测xyz + curvature（完整版）
```python
# 预测被mask块的xyz和curvature
target = concat([xyz[masked_blocks], curvature[masked_blocks]])
prediction = model(features, masked_features)
loss = weighted_MSE(prediction, target, weights=[1.0, 2.0])
```

**优点**：
- 更丰富的学习信号
- 同时学习几何和特征

**缺点**：
- 任务更复杂

### 4. 实现细节

#### 4.1 块分割实现
```python
class BlockSegmenter:
    def __init__(self, method='curvature_cluster', 
                 curvature_threshold_high=0.01,
                 curvature_threshold_low=0.005,
                 cluster_method='dbscan',
                 min_points_per_block=10):
        self.method = method
        self.curvature_threshold_high = curvature_threshold_high
        self.curvature_threshold_low = curvature_threshold_low
        self.cluster_method = cluster_method
        self.min_points_per_block = min_points_per_block
    
    def segment(self, points, curvature):
        """
        将点云分割成块
        
        Returns:
            blocks: List[List[int]] - 每个块是点的索引列表
            block_labels: List[str] - 每个块的标签 ('weld' or 'background')
        """
        if self.method == 'curvature_cluster':
            return self._curvature_cluster_segment(points, curvature)
        elif self.method == 'spatial_cluster':
            return self._spatial_cluster_segment(points, curvature)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _curvature_cluster_segment(self, points, curvature):
        # 方案B的实现
        # ...
        pass
```

#### 4.2 块级Mask实现
```python
class BlockMasker:
    def __init__(self, mask_ratio=0.3, 
                 strategy='random',  # 'random', 'weld_first', 'alternate'
                 weld_mask_ratio=0.5,
                 bg_mask_ratio=0.2):
        self.mask_ratio = mask_ratio
        self.strategy = strategy
        self.weld_mask_ratio = weld_mask_ratio
        self.bg_mask_ratio = bg_mask_ratio
    
    def generate_mask(self, blocks, block_labels, epoch=None):
        """
        生成块级mask
        
        Returns:
            mask: (N,) bool - 点的mask（True表示被mask）
        """
        if self.strategy == 'random':
            return self._random_mask(blocks)
        elif self.strategy == 'weld_first':
            return self._weld_first_mask(blocks, block_labels)
        elif self.strategy == 'alternate':
            return self._alternate_mask(blocks, block_labels, epoch)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
```

#### 4.3 Loss计算修改
```python
def block_recon_criterion(recon, batch, mask, blocks):
    """
    块级重建损失
    
    Args:
        recon: (B, N, 3) - 预测的xyz坐标
        batch: 包含xyz的batch
        mask: (B, N) - 点的mask
        blocks: List[List[int]] - 每个样本的块列表
    """
    xyz = batch['coordinate']  # (B, N, 3)
    
    # 只计算被mask块的loss
    masked_xyz = xyz[mask]  # (M, 3)
    masked_recon = recon[mask]  # (M, 3)
    
    # 可以按块计算loss（块内平均）
    block_losses = []
    for block in blocks:
        block_mask = mask[block]
        if block_mask.sum() > 0:  # 如果这个块被mask了
            block_xyz = xyz[block][block_mask]
            block_recon = recon[block][block_mask]
            block_loss = MSE(block_recon, block_xyz)
            block_losses.append(block_loss)
    
    # 平均所有被mask块的loss
    loss = mean(block_losses)
    return loss
```

## 方案对比

| 方案 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **方案A：曲率阈值** | 简单快速 | 块可能不连续 | ⭐⭐ |
| **方案B：曲率+聚类** | 块连续，语义清晰 | 需要调参 | ⭐⭐⭐⭐⭐ |
| **方案C：纯空间聚类** | 保证连续性 | 可能混入背景 | ⭐⭐⭐ |

## 推荐方案

### 分块策略：方案B（曲率+空间聚类）
- 使用DBSCAN对高曲率点进行聚类，形成焊缝块
- 使用DBSCAN对低曲率点进行聚类，形成背景块
- 控制最小块大小，过滤噪声块

### Mask策略：策略3（交替mask）
- 偶数epoch：mask焊缝块，让模型从背景预测焊缝
- 奇数epoch：mask背景块，让模型从焊缝预测背景
- 这样可以学习双向的几何关系

### 预测目标：选项A（只预测xyz）
- 先简化任务，只预测xyz坐标
- 如果效果好，再考虑加入curvature

## 实现步骤

1. **实现BlockSegmenter类**
   - 曲率阈值分割
   - DBSCAN空间聚类
   - 块标签生成

2. **实现BlockMasker类**
   - 块级mask生成
   - 多种mask策略

3. **修改pretrain.py**
   - 集成块分割和mask
   - 修改loss计算（只计算xyz）
   - 修改recon_head输出（只输出3维）

4. **测试和调参**
   - 测试块分割效果
   - 调整mask比例
   - 验证训练效果

## 潜在问题和解决方案

### 问题1：块大小不均匀
- **问题**：有些块很大，有些块很小
- **解决**：设置最小/最大块大小，过滤异常块

### 问题2：块数量过多/过少
- **问题**：DBSCAN参数不当导致块数量异常
- **解决**：动态调整DBSCAN参数，或使用K-means固定块数

### 问题3：训练不稳定
- **问题**：块级mask可能导致训练不稳定
- **解决**：使用渐进式mask（开始时mask比例小，逐渐增大）

### 问题4：计算开销
- **问题**：DBSCAN聚类计算开销大
- **解决**：对下采样后的点云进行聚类，或使用更快的聚类方法

## 预期效果

1. **更好的空间理解**：模型需要理解块之间的空间关系
2. **更强的泛化能力**：从背景预测焊缝，从焊缝预测背景
3. **更符合下游任务**：焊缝检测本质上就是识别高曲率区域

## 下一步

1. 确认方案是否符合需求
2. 实现BlockSegmenter和BlockMasker
3. 修改训练代码
4. 测试效果
