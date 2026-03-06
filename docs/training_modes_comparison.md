# 两种训练模式对比

## 当前实现的两种模式

### 相同点
- ✅ 都使用masked输入（部分特征被mask）
- ✅ 都预测被mask的区域（curvature + xyz）
- ✅ 都使用真实值作为标签（ground truth）
- ✅ 都计算MSE损失：`loss = MSE(预测值, 真实值)`
- ✅ 都只在masked区域计算loss

### 唯一区别：权重调整方式

## 1. "self_supervised" 模式（RMS动态平衡）

### Loss计算流程
```python
# 1. 计算每个通道的MSE
diff = (recon - gt) * mask
per_channel_mse = (diff ** 2).sum() / num_masked

# 2. 计算每个通道的RMS（误差尺度）
channel_rms = sqrt(mean(diff^2))  # 每个通道的RMS

# 3. 动态调整权重
rms_weights = 1.0 / channel_rms  # RMS越小，权重越大
effective_weights = base_weights * rms_weights

# 4. 加权求和
loss = sum(per_channel_mse * effective_weights)
```

### 特点
- **动态权重调整**：根据每个通道的误差尺度自动调整权重
- **自动平衡**：误差大的通道权重自动减小，误差小的通道权重自动增大
- **自适应**：训练过程中权重会动态变化

### 示例
假设：
- curvature通道：RMS = 0.1（误差小）
- x通道：RMS = 5.0（误差大）
- base_weights = [2.0, 2.0, 2.0, 2.0]

计算：
- curvature: rms_weight = 1/0.1 = 10.0 → effective_weight = 2.0 * 10.0 = 20.0
- x: rms_weight = 1/5.0 = 0.2 → effective_weight = 2.0 * 0.2 = 0.4

结果：curvature通道的权重是x通道的50倍，自动平衡了训练

## 2. "supervised" 模式（固定权重）

### Loss计算流程
```python
# 1. 计算每个通道的MSE
diff = (recon - gt) * mask
per_channel_mse = (diff ** 2).sum() / num_masked

# 2. 使用固定权重
effective_weights = base_weights  # 固定不变

# 3. 加权求和
loss = sum(per_channel_mse * effective_weights)
```

### 特点
- **固定权重**：所有通道使用相同的权重（或手动设置的权重）
- **简单直接**：不需要计算RMS，计算更快
- **需要手动调整**：如果某个通道误差大，需要手动调整权重

### 示例
假设：
- base_weights = [2.0, 2.0, 2.0, 2.0]
- curvature通道：MSE = 0.01
- x通道：MSE = 25.0

计算：
- curvature贡献：0.01 * 2.0 = 0.02
- x贡献：25.0 * 2.0 = 50.0

结果：x通道的贡献是curvature的2500倍，可能导致训练不平衡

## 对比总结

| 方面 | "self_supervised" | "supervised" |
|------|------------------|--------------|
| **权重调整** | 动态（RMS平衡） | 固定 |
| **计算复杂度** | 稍高（需要计算RMS） | 低 |
| **自适应能力** | 强（自动平衡） | 弱（需要手动调整） |
| **适用场景** | 不同通道误差尺度差异大 | 不同通道误差尺度相近 |
| **训练稳定性** | 更好（自动平衡） | 可能不平衡 |

## 实际效果

### RMS动态平衡的优势
1. **自动平衡**：不同通道的误差尺度差异大时，自动调整权重
2. **训练稳定**：避免某个通道主导训练
3. **无需调参**：不需要手动调整权重

### 固定权重的优势
1. **简单直接**：计算更快，逻辑更清晰
2. **可控性强**：可以精确控制每个通道的权重
3. **易于调试**：权重固定，更容易分析问题

## 建议

### 使用 "self_supervised" 模式（RMS平衡）如果：
- ✅ 不同通道的误差尺度差异大（如你的情况：curvature误差小，xyz误差大）
- ✅ 希望自动平衡训练
- ✅ 不想手动调整权重

### 使用 "supervised" 模式（固定权重）如果：
- ✅ 不同通道的误差尺度相近
- ✅ 需要精确控制每个通道的权重
- ✅ 希望训练过程更可控

## 你的情况

根据你的观察：
- curvature误差：~0.02（很小）
- xyz误差：~5.0（很大）

**建议使用 "self_supervised" 模式（RMS平衡）**，因为：
1. 不同通道误差尺度差异很大（250倍）
2. RMS平衡可以自动调整权重，让训练更平衡
3. 不需要手动调整权重
