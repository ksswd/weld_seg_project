# 焊缝点云分割项目总文档（持续更新）

> 文档目的：集中记录项目动机、代码实现细节、修改日志、全流程数学描述与当前进展，作为后续迭代与复现实验的统一入口。

---

## 第一部分：项目动机与介绍

### 1.1 研究背景
焊接是汽车制造、航空航天、船舶与装备制造中的关键工艺。面向智能焊接机器人的免示教作业场景，核心前提是**准确、稳定、实时**地完成焊缝区域感知与定位。与传统“示教-回放”路径不同，免示教焊接需要系统从传感数据中自动识别焊缝并支持后续路径/姿态调整。

从感知角度看，焊缝提取在点云中属于典型的**小目标分割 + 类别不平衡 + 几何细结构识别**问题：焊缝通常呈细长线状，区域占比较低，且边界定义存在不确定性。工业场景还存在干扰（噪声、反光、遮挡、采样不均），使得通用点云分割方法直接迁移时容易出现误检、漏检与泛化不稳。

### 1.2 项目动机与目标
结合开题终稿，本项目采用“**自监督预训练 + 小样本监督微调**”技术路线，目标是在有限标注条件下提升焊缝分割鲁棒性，并形成可复用工程流程：

1. 点云采集与ROI提取，构建焊缝点云数据集（有标签 + 无标签）；
2. 点云几何特征工程（8维输入特征 + 线性度/主方向辅助信息）；
3. 通过掩码重建进行预训练，提升几何表征能力；
4. 基于少量标注数据进行监督微调；
5. 输出逐点焊缝概率并完成推理与结果导出。

当前流程已完整跑通（预处理 + 预训练 + 微调 + 测试），但效果仍有提升空间。现阶段的主要动机是：

- 适配**线激光新数据域**（与旧深度相机数据分布差异显著）；
- 验证并量化各几何先验模块的真实收益；
- 在标注成本可控前提下提升F1与召回率，降低阈值敏感性与跨工件波动。

### 1.3 任务定义
给定点云点集
\[
\mathcal{P}=\{\mathbf{p}_i\in\mathbb{R}^3\}_{i=1}^{N}
\]
目标是学习点级二分类函数
\[
f_\theta: \mathbf{p}_i \mapsto y_i\in\{0,1\}
\]
其中 \(y_i=1\) 表示焊缝点，\(y_i=0\) 表示背景点。

---

## 第二部分：代码实现细节（基于当前主分支）

> 本节按“入口—数据—模型—训练—测试”展开，便于快速定位与二次开发。

### 2.1 项目入口与模式
- 入口：`main.py`
- 支持模式：`preprocess`、`pretrain`、`finetune`、`train`（预训练+微调）、`test`

典型调用：
- `python main.py --mode preprocess`
- `python main.py --mode pretrain`
- `python main.py --mode finetune`
- `python main.py --mode test`

### 2.2 配置管理
统一配置在 `utils/config.py`：

- 模型参数：`MODEL_TYPE`、`D_MODEL`、`N_HEADS`、`N_LAYERS`；
- 预训练参数：`USE_BLOCK_MASK`、`BLOCK_MASK_STRATEGY`、`PRETRAIN_MODE`；
- 微调参数：`USE_FOCAL_LOSS`、`NEG_TO_POS_RATIO`、`MODEL_SELECTION`；
- 推理参数：`PREDICTION_THRESHOLD`。

### 2.3 数据预处理模块
核心文件：
- `preprocess/geometric_feature.py`
- `preprocess/preprocess.py`

#### 2.3.1 预处理输出
每个点最终输出（归一化后）特征列：
- `x, y, z`
- `nx, ny, nz`
- `curvature`
- `density`

同时额外保存：
- `linearity`
- `principal_dir_x, principal_dir_y, principal_dir_z`

#### 2.3.2 几何特征计算
1) 法向：Open3D KNN估计。  
2) 曲率：邻域协方差矩阵特征值最小值占比。  
3) 局部密度：固定半径邻域点数并归一化。  
4) 线性度：特征值主次差异比。  
5) 主方向：局部主特征向量。

### 2.4 模型结构
核心文件：
- `model/model.py`
- `model/transformer_block.py`
- `model/attention/*`

当前支持两种主模型：
1. `transformer`（默认）  
2. `mlp`（对照实验）

Transformer主干层内交替混合三类注意力：
- 几何先验多头注意力（`MultiHeadGeometryAttention`）
- 局部注意力（`LocalAttention`）
- 全局注意力（`GlobalAttention`）

输出头：
- 分类头（点级二分类）
- 重建头（当前为曲率重建，1维）

### 2.5 预训练实现（train/pretrain.py）
当前主要采用**块级掩码重建**：

1. 先对有效点按网格分块（`BlockSegmenter`）；
2. 按高曲率点数将块标记为 `weld/background`；
3. 使用 `BlockMasker` 按策略（当前常用 `mixed`）mask部分块；
4. 将被mask区域输入特征与几何侧信息置零；
5. 模型执行 `task='recon'` 预测被mask区域曲率；
6. 用 `block_recon_criterion` 计算块级加权MSE。

### 2.6 微调实现（train/finetune.py）
关键设计：
- 仅使用有标签CSV（文件名含 `_label_`）；
- 按工件组（去除 `_augXX`）划分 train/val，避免增强泄露；
- 采样策略优先保留正类点，负类按比例采样；
- 支持 hard-negative 负样本优先采样（基于负类曲率分位）；
- 支持 Focal Loss 或带 `pos_weight` 的 BCE；
- 在验证集上搜索阈值（0~1）并以F1选最优模型。

hard-negative 采样说明（用于抑制搭接焊缝“边缘误检”）：
- 先从负类点中按曲率分位数划分 hard/easy 两池；
- hard 池定义：`curvature >= percentile(curvature_neg, HARD_NEG_CURV_PERCENTILE)`；
- 负类采样时按 `HARD_NEG_RATIO` 优先从 hard 池抽样，不足部分再由 easy/hard 回填；
- 仍受 `NEG_TO_POS_RATIO` 与 `MAX_NEG_PER_SAMPLE` 约束，保证训练稳定。

### 2.7 测试与推理（test/test.py）
流程：
1. 载入 `weights/best_finetune.pth`；
2. 对超大点云做FPS下采样；
3. 得到logits后叠加曲率引导项；
4. sigmoid + 阈值化输出二值预测；
5. 导出 `*_pred.csv`。

> 注意：当前 `test/test.py` 中模型调用参数顺序与训练阶段接口应再次核对，建议后续作为优先排查项之一。

---

## 第三部分：修改日志（从现在开始）

> 记录格式建议：日期 + 变更摘要 + 涉及文件 + 动机 + 结果。

### 2026-02-27
- 新建并完善 `docs/main.md` 项目总文档（本文件）。
- 本次整理内容：
  - 汇总项目动机与任务定义；
  - 梳理完整实现链路（预处理、预训练、微调、测试）；
  - 建立“流程数学公式 + 进展问题”的持续记录框架；
  - 新增参考文献占位，便于后续你给我论文后持续补充。

### 2026-03-01
- 配置瘦身与清理（第一轮）：`utils/config.py`
- 动机：减少历史参数冗余，提升配置可读性，避免调参误用。
- 处理：
  - 将当前主流程仍在使用的参数保留在主配置区；
  - 将静态扫描“未直接引用”的参数集中归档到“兼容保留”区，明确后续可删除候选。
- 结果：配置结构更清晰，不影响当前预处理/预训练/重建测试流程。

### 2026-03-01（重建验证）
- 使用 `test/recon_test.py` 对快速预训练权重进行首轮重建验证（样本：`downsampled_weld_1_aug_68`）。
- 输出文件：
  - `data/new/recon_test/output/downsampled_weld_1_aug_68_recon_vis.csv`
  - `data/new/recon_test/output/downsampled_weld_1_aug_68_recon_metrics.csv`
- 指标摘要（curvature，重点 masked）：
  - masked: MAE=9.6731, RMSE=9.9346, Corr=0.5234
  - unmasked: MAE=5.2076, RMSE=5.5715, Corr=0.6144
- 当前结论：模型已学习到一定相关性（Corr>0.5），但 masked 区域重建误差仍较大，重建质量暂不理想；需继续做多样本复测与目标/损失策略优化。

### 2026-03-07（5-fold group-aware 划分机制）
- 新增 `utils/fold_data_split.py`，实现固定 `fold_1~fold_5` 的 group 级划分与统一文件收集入口。
- 适配命名规则：
  - 无标注增强：`downsampled_weld_{i}_aug_{k}.csv` → `unlabeled_aug`
  - 有标注（当前主命名）：`downsampled_labeled_{i}_label_*.csv` → `labeled_original`
  - 兼容命名：`i-k` / `i-labeled` / `i-k-labeled`
- 新增能力：
  - `parse_group_id`：统一提取 group_id；
  - `detect_sample_type`：区分 `unlabeled_aug` / `labeled_original` / `labeled_aug`；
  - `collect_files_by_fold`：按阶段筛选文件（pretrain / finetune_train / finetune_val / test）；
  - 防泄漏检查：禁止 train 与 val/test group 交叉、禁止错误类型文件混入；
  - 统计日志打印：fold、group 列表、按类型与按group计数。
- 接入改造：
  - `main.py` 新增 `--fold_id` 参数（可覆盖配置）；
  - `train/pretrain.py`：预训练文件改为按 fold 的 train groups 仅收集 `unlabeled_aug`；
  - `train/finetune.py`：微调 train/val 文件改为 fold-aware 收集，val 仅 `labeled_original`；
  - `test/test.py`：测试文件改为 fold-aware 收集，test 仅 `labeled_original`。
- 配置新增（`utils/config.py`）：
  - `USE_FOLD_SPLIT`、`FOLD_ID`、`FOLD_SPLIT_STRICT`、`INCLUDE_LABELED_AUG_IN_FINETUNE`
  - `PRETRAIN_DATA_DIR`、`LABELED_DATA_DIR`
- 结果：训练与评估流程从“文件级划分”升级为“原始样本 group 级划分”，显著降低数据泄露风险并提高可复现性。

### 2026-03-08（前两项落地：评估双口径 + soft加权loss）
- 评估脚本升级：`test/per_case_metrics.py`
  - 新增 hard/soft 双口径同步统计；
  - hard：基于 `label`（0/1）计算；
  - soft-aware：基于 `label_soft > soft_thr` 计算（无 `label_soft` 时自动回退hard）；
  - 新增参数 `--soft_thr`（默认0.5）；
  - 全局汇总与逐样例输出均改为 hard/soft 双轨字段。
- 训练损失升级：`train/finetune.py` + `utils/config.py`
  - 新增 soft 标签加权loss：边界带降权（`1 - λ·4s(1-s)`） + 高soft区域召回增强（`1 + γ·s`）；
  - 目标：鼓励焊缝覆盖完整性，同时避免在边界不确定区过拟合硬判定；
  - 新增配置：
    - `USE_SOFT_WEIGHTED_LOSS`
    - `SOFT_BOUNDARY_DOWNWEIGHT_LAMBDA`
    - `SOFT_RECALL_BOOST_GAMMA`
- 运行脚本更新：`run.sh`
  - `per_case_metrics` 增加 `--soft_thr 0.5`，统一输出 hard/soft 两套评估结果。

### 2026-03-09（搭接焊缝误检修正：hard-negative 采样）
- 采样策略升级：`train/finetune.py`
  - 在微调 `collate_fn` 中新增 hard-negative 负样本优先采样；
  - 负类点按曲率分位数切分 hard/easy 两池，优先抽取 hard 池（高曲率非焊缝）；
  - 仍保持“正类尽量保留 + 负类受比例与上限约束”的整体框架。
- 配置新增：`utils/config.py`
  - `HARD_NEG_RATIO`：负类采样中 hard-negative 目标占比（默认 0.6）；
  - `HARD_NEG_CURV_PERCENTILE`：hard-negative 曲率分位阈值（默认 80）。
- 动机与预期：
  - 重点缓解搭接焊缝场景下“把工件边缘预测成焊缝”的系统性误检；
  - 通过提升困难负样本覆盖，提高模型对“高曲率但非焊缝”区域的判别能力。

---

## 第四部分：总流程（含数学公式、当前进展与问题）

### 4.1 总体流程图（文本）
结合开题终稿，焊缝提取主线可概括为：

数据采集与ROI提取 \(\rightarrow\) 数据预处理与几何特征工程 \(\rightarrow\) 自监督预训练 \(\rightarrow\) 小样本监督微调 \(\rightarrow\) 测试推理与结果输出

其中当前项目已实现“预处理 + 预训练 + 微调 + 测试”闭环，后续重点是面向线激光数据域做模型适配与增量优化。

### 4.2 步骤一：数据预处理

#### 4.2.1 邻域协方差与曲率
对点 \(\mathbf{p}_i\) 的邻域 \(\mathcal{N}(i)\) 计算协方差：
\[
\mathbf{C}_i = \frac{1}{|\mathcal{N}(i)|}\sum_{\mathbf{p}_j\in\mathcal{N}(i)}(\mathbf{p}_j-\bar{\mathbf{p}}_i)(\mathbf{p}_j-\bar{\mathbf{p}}_i)^T
\]
设特征值 \(\lambda_1\ge\lambda_2\ge\lambda_3\)，曲率定义为
\[
\kappa_i = \frac{\lambda_3}{\lambda_1+\lambda_2+\lambda_3+\epsilon}
\]

#### 4.2.2 线性度
\[
\text{Linearity}_i = \frac{\lambda_1-\lambda_2}{\lambda_1+\epsilon}
\]

#### 4.2.3 局部密度
在半径 \(r\) 邻域中统计点数：
\[
\rho_i = |\{\mathbf{p}_j\mid \|\mathbf{p}_j-\mathbf{p}_i\|_2<r\}|-1
\]
并做min-max归一化。

**当前进展**：已完成并稳定运行。  
**当前问题**：不同数据域的曲率分布差异较大，统一阈值可能不鲁棒。

### 4.3 步骤二：掩码预训练（块级）

#### 4.3.1 分块与掩码
将点云划分为块集合 \(\{\mathcal{B}_m\}\)，对部分块采样掩码集合 \(\mathcal{M}\)。被mask点输入置零，仅通过上下文恢复目标。

#### 4.3.2 重建目标与损失
当前实现以曲率重建为主：
\[
\mathcal{L}_{\text{pre}} = \frac{1}{\sum_m w_m}\sum_{m\in\mathcal{M}} w_m\cdot \frac{1}{|\mathcal{B}_m|}\sum_{i\in\mathcal{B}_m}(\hat{\kappa}_i-\kappa_i)^2
\]
其中 \(w_m\) 为块内被mask点数权重。

**当前进展**：块级mask、混合策略、块级loss均已接入训练。  
**当前问题**：预训练对下游分类增益不稳定，需进一步验证mask策略与目标设计是否最优。

### 4.4 步骤三：监督微调

二分类概率：
\[
\hat{p}_i = \sigma(z_i)
\]

BCE损失：
\[
\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_i \left[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]
\]

Focal形式（可选）：
\[
\mathcal{L}_{\text{Focal}} = -\frac{1}{N}\sum_i \alpha_i(1-p_{t,i})^{\gamma}\log(p_{t,i})
\]

**当前进展**：已实现分组划分、类别不平衡处理、阈值搜索。  
**当前问题**：
- 焊缝点占比低，仍存在召回不足；
- 最优阈值在不同批次数据波动较大；
- 对某些工件几何形态泛化仍偏弱。

### 4.5 步骤四：模型测试
推理阶段使用
\[
\tilde{z}_i = z_i + \alpha\cdot \tilde{\kappa}_i
\]
进行曲率引导，再经sigmoid与阈值 \(\tau\) 二值化：
\[
\hat{y}_i = \mathbb{I}(\sigma(\tilde{z}_i)>\tau)
\]

**当前进展**：可批量输出预测CSV。  
**当前问题**：推理增益依赖 \(\alpha,\tau\) 参数，需系统调参与交叉验证。

### 4.6 下一步建议（待执行）
1. 统一核查 `test/test.py` 的模型输入顺序与 `GeometryAwareTransformer.forward` 是否完全一致；
2. 进行“点级mask vs 块级mask”、“mixed vs alternate”消融；
3. 增加以工件为单位的分层验证统计（按类别/工件输出F1、Recall）；
4. 尝试联合重建目标（曲率 + 坐标）与损失重加权；
5. 将推理阈值从固定值改为按验证集自动选择并随权重版本存档。

---

## 第五部分：当前情况、问题梳理与任务列表（进行中）

### 5.1 当前数据说明
- 数据来源由深度相机迁移到线激光扫描，点云精度和几何结构可辨性显著提升；
- 新数据对象为铁板拼接焊接结构，覆盖直线/曲线V型、搭接、T型、拐角等焊缝类型；
- 尺寸量级（单位mm）：板件约 200 × 150，厚度约 3-9，焊缝宽度约 1-3；
- 经过增强、ROI提取与降采样后，当前可用数据约 1581 条；
- 已将新数据接入现有流程并开始跑通实验。
- 数据及存储路径
- 没有预处理的
  - augmentation ：无标注没有进行降采样的，含有边缘的点云，数据增强80
  - augmentation2 ：无标注有进行降采样的，含有边缘的点云，数据增强80
  - augmentation3 ：无标注有进行降采样的，含有边缘的点云，数据增强10
  - augmentation4 ：有标注有进行降采样的，无边缘的点云，数据增强10 （最新！！！）
  - augmentation——labeled : 有标注有进行降采样的，含有边缘的点云，数据增强10
  - noside ： 有标注无边缘点云，无增强
  - downsample : 无标注降采样有边缘点云，无增强
  - labeled ： 有标注点云，有边缘
  - original ： 最原始数据
- 经过预处理的
  - processed_csv : 有边缘无标注80增强  
  - ..3 :  有边缘无标注10增强 
  - ..4 : 无边缘软标注无增强 
  - ..5 : 预处理不进行线性化的4
  - ..labeled : 有边缘有标注无增强 
  - ..labeled_soft : +软标签 
  - ..labeled2 : 10增强

#### 样本与类型映射
1. V型直线焊缝（9mm深）
2. V型直线焊缝（6mm深）
3. V型直线焊缝（3mm深）
4. V型S线焊缝（3mm深）
5. V型S线焊缝（3mm深，加宽）
6. V型S线焊缝（3mm深，变窄）
7. V型弧线焊缝（3mm深）
8. V型弧线焊缝（3mm深，加宽）
9. 搭接焊缝（直线）
10. 搭接焊缝（3层直线）
11. 搭接焊缝（两条非平行直线）
12. 弧形搭接
13. 弧形搭接（2层）
14. S形搭接
15. S形搭接（2层）
16. S形搭接 + 直线搭接
17. 拐角焊缝 （弃用，效果差）
18. T型焊缝
19. T型焊缝

#### 当前观察结论（按类型）
- **1-5（V型直线/S线）**：识别相对较好，可基本还原焊缝形状。
- **6（变窄S线）**：表现较差，基本未检出焊缝。
- **7-8（弧线V型）**：存在局部误检，疑似将弧形工件上的铁锈区域识别为焊缝。
- **9-16（搭接系列）**：仅能识别大体轮廓，预测分布不均匀，连续性和完整性较差。
- **17（拐角）**：能检出焊缝线，但有不连续；并将工件边缘和面上激光纹理误检为焊缝。
- **18-19（T型）**：焊缝主线可被识别，但存在位置偏移；且工件边缘误检明显。

#### 当前阶段总体判断
- 模型已具备“焊缝大致定位”能力；
- 但对搭接、窄缝、边缘/纹理干扰场景鲁棒性不足；
- 下一步应优先进行：
  1. 搭接与窄缝样本定向补标（难例优先）；
  2. 类型分层评估与重采样训练；
  3. 边缘误检抑制（数据规则 + 训练策略联合改进）。

### 5.6 对比实验

#### A. 实验基本信息
- 模型权重路径：`weight/best_pretrain.pth` 有边缘预训练+无边缘微调
- 数据来源：`data/new/` 
- ours现状:
  - 存在大量误检,但是召回率较高
  - 边缘误检+焊缝周围误检 

- 评价指标
  - accurancy: 准确率 
  - recall: 召回率
  - F1: 
  - mIoU: 交并比

#### B. 对比方法
- 数学方法:两方法都效果不行
  - 曲率+法向量阈值
  - RANSAC+DBScan
- Pointnet2：无边缘较弱，稍强于未调参的 ours；有边缘很好。env：`pointcept`
- Point Transformer（Pointcept-copy）：只测了无边缘，效果很好。env：`pointcept`
- PointMLP：待统一测试脚本测试。env：`pointmlp`
- PointConv：env：`pointcept`
- DGCNN：env：`weld_seg`
- 本项目方法：env：`weld_seg`

#### C. 下一步可能提高性能的方法
- 整合我们的机制到PTv3重新做,但是现在初步试了一下效果是负的
- ptv3也拿无边缘的训练推理,看看效果 没提高多少
- ours调参? 目前只调了最外端的阈值
- ours继续加参数量,加层数,堆量,目前的参数量还是太少了 但是加了一些就OOM
- 预训练也用无边缘的训练 已完成,但没太多提升
- 要不要预训练?

- 指标定位iou
- 数学方法是否make sense
- 多个split一起跑,平均,然后再调参
- 几个baseline，fold1-3一起跑全量，再看效果
- ours上次跑的无线性化再看看效果和调bug

### 5.7 需要优先调整的超参表（预训练 + 微调 + 推理）

> 说明：下面这张表优先整理“**不大改逻辑、但最可能影响指标**”的参数。建议先按优先级从高到低逐批扫描，而不是一次性全量网格搜索。

| 优先级 | 参数名 | 当前值 | 建议搜索范围 | 作用阶段 | 说明 |
|---|---|---:|---|---|---|
| P0 | `LEARNING_RATE` | `5e-5` | `1e-5, 3e-5, 5e-5, 1e-4, 3e-4` | 预训练 / 微调 | 最优先，通常对收敛和最终指标最敏感 |
| P0 | `FINETUNE_LR` | `5e-5` | `1e-5, 3e-5, 5e-5, 1e-4` | 微调 | 预训练和微调的最优学习率通常不同 |
| P0 | `PREDICTION_THRESHOLD` | `0.80` | `0.3 ~ 0.9` | 推理 | 对 Precision / Recall / F1 影响很大，先做验证集搜索 |
| P0 | `NEG_TO_POS_RATIO` | `2` | `1, 2, 4, 8` | 微调 | 控制类别不平衡，直接影响召回与误检平衡 |
| P0 | `HARD_NEG_RATIO` | `0.6` | `0.3, 0.5, 0.6, 0.8` | 微调 | 用于抑制“边缘/纹理误检”为焊缝 |
| P0 | `FOCAL_GAMMA` | `2.0` | `1.0, 2.0, 3.0` | 微调 | focal loss 聚焦难样本强度 |
| P0 | `USE_FOCAL_LOSS` | `True` | `True / False` | 微调 | 与 BCE / pos_weight 组合做对比 |
| P0 | `USE_POS_WEIGHT` | `True` | `True / False` | 微调 | 正类稀缺时常有帮助，但需和 focal 联合看 |
| P0 | `POS_WEIGHT_MAX` | `50.0` | `10, 20, 50` | 微调 | 正类权重上限，避免过度偏置 |
| P0 | `USE_SOFT_WEIGHTED_LOSS` | `True` | `True / False` | 微调 | 边界软标签加权，适合焊缝边界模糊场景 |
| P0 | `SOFT_BOUNDARY_DOWNWEIGHT_LAMBDA` | `0.4` | `0.2, 0.4, 0.6` | 微调 | 边界带降权强度 |
| P0 | `SOFT_RECALL_BOOST_GAMMA` | `1.0` | `0.5, 1.0, 2.0` | 微调 | 高 soft 区域召回增强强度 |
| P1 | `MASK_RATIO` | `0.3` | `0.15, 0.3, 0.45, 0.6` | 预训练 | 影响预训练难度与表征质量 |
| P1 | `MASK_TYPE` | `random` | `random / curvature` | 预训练 | 随机 mask 和高曲率 mask 做消融 |
| P1 | `USE_BLOCK_MASK` | `True` | `True / False` | 预训练 | 块级 mask 与点级 mask 的效果对比 |
| P1 | `BLOCK_MASK_RATIO` | `0.3` | `0.15, 0.3, 0.45` | 预训练 | 块级预训练中最核心的 mask 强度之一 |
| P1 | `BLOCK_MASK_STRATEGY` | `mixed` | `mixed / alternate` | 预训练 | 适合做消融，观察 mask 组织方式差异 |
| P1 | `TARGET_POINTS_PER_BLOCK` | `1000` | `500, 1000, 1500, 2000` | 预训练 | 控制块大小，影响局部上下文与计算量 |
| P1 | `HIGH_CURV_THRESHOLD` | `0.01` | `0.005, 0.01, 0.02` | 预训练 | 高曲率块判定阈值 |
| P1 | `MIN_HIGH_CURV_POINTS` | `5` | `3, 5, 10` | 预训练 | 避免高曲率块过于稀疏 |
| P1 | `D_MODEL` | `32` | `32, 48, 64` | 预训练 / 微调 | 模型容量核心参数，可能存在欠拟合 |
| P1 | `N_HEADS` | `4` | `4, 8` | 预训练 / 微调 | 与 `D_MODEL` 联动，需保证可整除 |
| P1 | `N_LAYERS` | `9` | `6, 9, 12` | 预训练 / 微调 | 深度与容量控制 |
| P1 | `FFN_DIM` | `128` | `128, 256, 512` | 预训练 / 微调 | Transformer 前馈层宽度 |
| P2 | `K_NEIGHBORS` | `50` | `20, 30, 50, 80` | 预处理 | 邻域大小会影响法向/曲率/密度特征 |
| P2 | `RADIUS_RATIO` | `2.0` | `1.5, 2.0, 2.5, 3.0` | 预处理 | 局部半径相关，影响密度统计与结构感知 |
| P2 | `PREPROCESS_STANDARDIZE` | `False` | `False / True` | 预处理 | 是否对预处理特征做全局标准化 |
| P2 | `PRETRAIN_MAX_POINTS` | `10000` | `4096, 8192, 10000, 12000` | 预训练 | 控制单样本最大点数，影响上下文保留 |
| P2 | `SUBSAMPLE_METHOD` | `fps` | `fps / random` | 预训练 | 点云子采样策略，影响保真度 |
| P2 | `PRETRAIN_USE_NORM_LOSS` | `True` | `True / False` | 预训练 | 归一化损失是否启用 |
| P2 | `PRETRAIN_CURV_TARGET` | `log` | `log / linear`（按代码支持情况） | 预训练 | 曲率重建目标的尺度形式 |
| P2 | `PRETRAIN_CURV_EPS` | `1e-5` | `1e-6, 1e-5, 1e-4` | 预训练 | 曲率 log 变换稳定项 |
| P2 | `SOFT_LABEL_RADIUS` | `0.2` | `0.1, 0.2, 0.3` | 微调 | soft label 边界带宽度 |
| P2 | `CURV_AUX_WEIGHT` | `0.0` | `0.0, 0.1, 0.5, 1.0` | 微调 | 若启用曲率辅助头，可再做扫参 |
| P2 | `FREEZE_BACKBONE` | `False` | `True / False` | 微调 | 预训练后是否冻结骨干网络 |
| P2 | `USE_SOFT_LABELS` | `False` | `True / False` | 微调 | 是否使用软标签监督 |
| P2 | `INCLUDE_LABELED_AUG_IN_FINETUNE` | `True` | `True / False` | 训练划分 | 增强样本是否进入微调训练集 |
| P3 | `PRETRAIN_INCLUDE_LABELED` | `True` | `True / False` | 预训练 | 是否把同 fold 内有标注样本作为无监督特征学习数据 |
| P3 | `WELD_WIDTH_RANGE` | `[0.005, 0.02]` | `[0.003, 0.015] / [0.005, 0.02] / [0.01, 0.03]` | 预处理 / 先验 | 焊缝宽度先验范围，可能影响几何门控 |
| P3 | `ALPHA0` | `2.0` | `1.0, 2.0, 3.0` | 模型先验 | 几何先验强度相关 |
| P3 | `BETA0` | `0.5` | `0.25, 0.5, 1.0` | 模型先验 | 背景抑制/平衡项相关 |
| P3 | `GAMMA` | `1.0` | `0.5, 1.0, 2.0` | 模型先验 | 引导强度相关 |
| P3 | `SIGMA` | `0.1` | `0.05, 0.1, 0.2` | 模型先验 | 平滑/距离衰减相关 |
| P3 | `CURV_GAIN` | `4.0` | `1.0, 2.0, 4.0, 6.0` | 微调 / 推理 | 曲率引导强度，影响误检和召回 |
| P3 | `CURVATURE_THRESHOLD` | `0.0008` | `0.0005, 0.0008, 0.001, 0.002` | 推理 | 曲率阈值引导参数 |
| P3 | `MAX_NEG_PER_SAMPLE` | `2000` | `1000, 2000, 4000` | 微调 | 单样本负类上限，影响训练稳定性 |
| P3 | `EMPHASIS_FACTOR` | `3.0` | `1.5, 2.0, 3.0, 4.0` | 微调 | 采样/损失强调因子，需结合代码确认生效位置 |

#### 建议的扫参顺序
1. 先做 **P0**：学习率、采样不平衡、损失、阈值。
2. 再做 **P1**：预训练 mask 策略与模型容量。
3. 最后做 **P2/P3**：预处理、先验、推理细项。

#### 实验建议
- 每次只改一组参数，避免耦合过强导致结论不清晰。
- 先固定一个 fold 跑通，再对 `fold_1 ~ fold_5` 做平均。
- 先用验证集搜索 `PREDICTION_THRESHOLD`，再决定最终测试阈值。
- 对 `USE_FOCAL_LOSS`、`USE_POS_WEIGHT`、`USE_SOFT_WEIGHTED_LOSS` 这类强耦合项，优先做小型组合实验，而不是单独盲扫。


---

## 第六部分：参考文献（持续追加）

> 说明：本节先建立引用框架。后续你给我文献题目/链接/BibTeX后，我会直接补全并统一编号。

[1] 开题报告相关文档：《开题报告_终稿+修改说明.docx》（项目内部资料）。  
[2] He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.  
[3] Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.  
[4] Vaswani, A., et al. (2017). Attention Is All You Need. *NeurIPS*.

---

## 附：文档维护约定

1. 每次代码改动后，同步更新“第三部分 修改日志”；
2. 若流程或损失函数变化，同步更新“第四部分”对应公式与问题记录；
3. 每引入新论文，在“第六部分”追加标准引用；
4. 重要实验建议补充：日期、配置摘要、核心指标、结论四要素。
