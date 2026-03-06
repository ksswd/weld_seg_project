# weld_seg_project/utils/config.py 配置文件
from dataclasses import dataclass

@dataclass
class Config:
    # === 数据路径 ===
    RAW_DATA_DIR = "data/new/labeled"
    PROCESSED_DATA_DIR = "data/new/processed_csv_labeled"
    LABEL_DATA_DIR = "data/new/processed_csv_labeled_soft"
    TEST_DATA_DIR = "data/test"
    PREDICTED_DATA_DIR = "data/predictions"
    WEIGHTS_SAVE_DIR = "weights2"
    LOG_DIR = "logs"

    # === 数据预处理参数 ===
    K_NEIGHBORS = 20
    RADIUS_RATIO = 2.0

    # === 模型架构 ===
    MODEL_TYPE = "transformer"  # "transformer" 或 "mlp" - 模型类型选择
    SIDE_GATE_TYPE = "qwen"
    INPUT_DIM = 8
    D_MODEL = 12
    N_HEADS = 3
    N_LAYERS = 3
    FFN_DIM = 64

    # === 注意力机制参数 ===
    ALPHA0 = 2.0
    BETA0 = 0.5
    GAMMA = 1.0
    SIGMA = 0.1
    WELD_WIDTH_RANGE = [0.005, 0.02]

    # === 训练参数 ===
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    FINETUNE_LR = 5e-5
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 300
    NUM_WORKERS = 0

    # === 预训练参数 ===
    MASK_RATIO = 0.3
    MASK_TYPE = "random"  # "random" or "curvature" - mask策略类型，随机 or 高曲率
    PRETRAIN_MAX_POINTS = 15000
    SUBSAMPLE_METHOD = "fps"
    USE_AMP = True
    ACCUM_STEPS = 1
    DEBUG_SINGLE_BATCH = False

    # === 块级预训练参数 ===
    USE_BLOCK_MASK = True  # 是否使用块级mask（True）还是点级mask（False）
    TARGET_POINTS_PER_BLOCK = 1000  # 每个块的目标点数
    HIGH_CURV_THRESHOLD = 0.01  # 高曲率阈值
    MIN_HIGH_CURV_POINTS = 5  # 高曲率点的最小数量（用于分类块）
    BLOCK_MASK_RATIO = 0.3  # mask的块比例（当strategy='mixed'时，这是总比例）
    BLOCK_MASK_STRATEGY = "mixed"  # "mixed"（推荐）或 "alternate"
    # "mixed": 每个epoch同时mask两种块，让模型同时学习两种任务（推荐）
    # "alternate": 交替mask（偶数epoch mask焊缝块，奇数epoch mask背景块）
    WELD_MASK_RATIO = None  # 焊缝块的mask比例（None时使用BLOCK_MASK_RATIO * 0.5）
    BG_MASK_RATIO = None  # 背景块的mask比例（None时使用BLOCK_MASK_RATIO * 0.5）
    GRID_ALIGN_BASE = 0.001  # 网格对齐基数（米），使块边界明显

    # === 预训练重建损失参数 ===
    PRETRAIN_CURV_TARGET = "raw"
    PRETRAIN_CURV_EPS = 1e-6
    # [curvature, x, y, z] - 增加坐标权重以改善xyz重建效果
    PRETRAIN_RECON_WEIGHTS = [2.0, 2.0, 2.0, 2.0]  # [curvature, x, y, z]
    PRETRAIN_USE_NORM_LOSS = True  # 启用RMS归一化以平衡不同通道

    # === 训练模式 ===
    # 注意：两种模式本质上都是监督学习（都使用真实值作为标签）
    # 区别在于loss计算方式：
    # "self_supervised": RMS动态平衡权重 - 自动调整不同通道的权重（原始方法）
    # "supervised": 固定权重 - 使用固定的权重，需要手动调整
    PRETRAIN_MODE = "self_supervised"

    # === 微调参数 ===
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    USE_POS_WEIGHT = True
    POS_WEIGHT_MAX = 50.0
    FREEZE_BACKBONE = False
    LABEL_MASK_RATIO = 0.0
    CURV_AUX_WEIGHT = 0.0
    MODEL_SELECTION = "fps"
    USE_SOFT_LABELS = True  # 微调时是否优先使用label_soft作为loss监督（评估仍用硬标签）

    # === 微调平衡参数 ===
    NEG_TO_POS_RATIO = 2
    MAX_NEG_PER_SAMPLE = 2000

    # === 推理参数 ===
    PREDICTION_THRESHOLD = 0.50

    # === 点云处理 ===
    MAX_POINTS = 10000

    # === 权重路径 ===
    PRETRAINED_WEIGHTS = "weights2/best_pretrain1.pth"

    # === 兼容保留（当前代码未直接引用，后续确认后可删除） ===
    SPLITS_DIR = "data/splits/new"
    PRETRAIN_DENSITY_TARGET = "norm"
    PRETRAIN_LINEARITY_TARGET = "norm"
    RECON_WEIGHTS = [0.12, 0.12, 0.12, 0.06, 0.06, 0.06, 0.4, 0.06]
    CURV_L1_WEIGHT = 1.0
    EMPHASIS_FACTOR = 3.0
    CURV_GAIN = 4.0
    CURVATURE_THRESHOLD = 0.0008
    TEST_WEIGHTS = "weights2/best_finetune1.pth"
    SUPERVISED_TRAIN = True
    FINETUNE_CLASSIFIER = True
    ONLY_USE_LABELED = True
    CLASS_LOSS_WEIGHT = 1.0
    REINIT_CLASSIFIER_ON_FINETUNE = True
