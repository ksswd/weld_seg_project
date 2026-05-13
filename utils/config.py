# weld_seg_project/utils/config.py 配置文件
import os
from dataclasses import dataclass

@dataclass
class Config:
    # === 数据路径（统一由运行脚本/环境变量注入） ===
    RAW_DATA_DIR = os.environ.get("RAW_DATA_DIR", "data/new/augmentation_4")
    PROCESSED_DATA_DIR = os.environ.get("PROCESSED_DATA_DIR", "data/new/processed_csv_5")
    PRETRAIN_DATA_DIR = os.environ.get("PRETRAIN_DATA_DIR", "data/new/processed_csv_5")
    LABELED_DATA_DIR = os.environ.get("LABELED_DATA_DIR", "data/new/processed_csv_5")
    WEIGHTS_SAVE_DIR = os.environ.get("WEIGHTS_SAVE_DIR", "results/weights")
    LOG_DIR = os.environ.get("LOG_DIR", "logs") 
    PRETRAINED_WEIGHTS = os.environ.get("PRETRAINED_WEIGHTS", "weights/best_pretrain_800epoches.pth")
    TEST_WEIGHTS = os.environ.get("TEST_WEIGHTS", "results/weights/best_finetune.pth")

    # === Fold划分参数 ===
    USE_FOLD_SPLIT = True
    FOLD_ID = os.environ.get("FOLD_ID", "fold_1")
    FOLD_SPLIT_STRICT = True
    INCLUDE_LABELED_AUG_IN_FINETUNE = True

    # === 数据预处理参数 ===
    K_NEIGHBORS = 50
    RADIUS_RATIO = 2.0
    # 预处理是否做全局z-score标准化（你当前需求：关闭，直接保留原始数值）
    PREPROCESS_STANDARDIZE = True

    # === 模型架构 ===
    MODEL_TYPE = "transformer"  # "transformer" 或 "mlp"
    SIDE_GATE_TYPE = "qwen"
    INPUT_DIM = 8
    D_MODEL = 32
    N_HEADS = 4
    N_LAYERS = 9
    FFN_DIM = 128

    # === 注意力机制参数 ===
    ALPHA0 = 2.0
    BETA0 = 0.5
    GAMMA = 1.0
    SIGMA = 0.1
    WELD_WIDTH_RANGE = [0.005, 0.02]

    # === 训练参数 ===
    BATCH_SIZE = 1 ## batch怎么改
    LEARNING_RATE = 5e-5
    FINETUNE_LR = 1e-5
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 200
    NUM_WORKERS = 0
    EARLY_STOP_PATIENCE = 50
    EARLY_STOP_MIN_DELTA = 0.0
    EVAL_TEST_EVERY_EPOCH = True

    # === 预训练参数 ===
    MASK_RATIO = 0.3
    MASK_TYPE = "random"  # "random" or "curvature"
    PRETRAIN_MAX_POINTS = 10000
    SUBSAMPLE_METHOD = "fps"
    USE_AMP = False
    ACCUM_STEPS = 1
    DEBUG_SINGLE_BATCH = False

    # === 块级预训练参数 ===
    USE_BLOCK_MASK = True
    TARGET_POINTS_PER_BLOCK = 1000
    HIGH_CURV_THRESHOLD = 0.01
    MIN_HIGH_CURV_POINTS = 5
    BLOCK_MASK_RATIO = 0.3
    BLOCK_MASK_STRATEGY = "mixed"  # "mixed" 或 "alternate"
    WELD_MASK_RATIO = None
    BG_MASK_RATIO = None
    GRID_ALIGN_BASE = 0.001

    # === 预训练重建损失参数 ===
    PRETRAIN_CURV_TARGET = "log"
    PRETRAIN_CURV_EPS = 1e-5
    PRETRAIN_DENSITY_TARGET = "norm"
    PRETRAIN_LINEARITY_TARGET = "norm"
    PRETRAIN_RECON_WEIGHTS = [2.0]
    PRETRAIN_USE_NORM_LOSS = True

    # === 训练模式 ===
    PRETRAIN_MODE = "self_supervised"
    PRETRAIN_INCLUDE_LABELED = True  # 预训练是否纳入同fold训练组内的有标注样本（仅用特征，不用label监督）
    CURV_L1_WEIGHT = 1.0

    # === 微调参数 ===
    USE_FOCAL_LOSS = True
    FOCAL_GAMMA = 2.0
    USE_POS_WEIGHT = True
    POS_WEIGHT_MAX = 50.0
    FREEZE_BACKBONE = False
    LABEL_MASK_RATIO = 0.0
    CURV_AUX_WEIGHT = 0.0
    MODEL_SELECTION = "miou"
    USE_SOFT_LABELS = False  # 微调时是否优先使用label_soft作为loss监督（评估仍用硬标签）
    # 预处理阶段是否自动从label生成label_soft（仅当存在label列）
    GENERATE_SOFT_LABEL_IN_PREPROCESS = True
    SOFT_LABEL_RADIUS = 0.2
    # === soft标签加权loss（覆盖增强 + 边界降权）===
    USE_SOFT_WEIGHTED_LOSS = True
    SOFT_BOUNDARY_DOWNWEIGHT_LAMBDA = 0.4  # λ：边界带降权强度
    SOFT_RECALL_BOOST_GAMMA = 1.0          # γ：高soft区域召回增强强度

    # === 微调平衡参数 ===
    NEG_TO_POS_RATIO = 2
    MAX_NEG_PER_SAMPLE = 2000
    # 难负样本采样：优先从高曲率负类中采样，抑制“工件边缘=焊缝”误检
    HARD_NEG_RATIO = 0.6              # 负样本中 hard negative 占比
    HARD_NEG_CURV_PERCENTILE = 80.0   # 负类中曲率>=该分位阈值视为 hard negative
    EMPHASIS_FACTOR = 3.0
    CURV_GAIN = 4.0

    # === 推理参数 ===
    PREDICTION_THRESHOLD = 0.80
    CURVATURE_THRESHOLD = 0.0008

    # === 点云处理 ===
    MAX_POINTS = 8192
    # === 其他设置 ===
    SUPERVISED_TRAIN = True
    FINETUNE_CLASSIFIER = True
    ONLY_USE_LABELED = True
    CLASS_LOSS_WEIGHT = 1.0
    REINIT_CLASSIFIER_ON_FINETUNE = True
