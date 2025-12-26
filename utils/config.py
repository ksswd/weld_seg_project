# weld_seg_project/utils/config.py 配置文件，用于储存路径和训练参数等
from dataclasses import dataclass
@dataclass
class Config:
    # --- Data and Preprocessing ---
    LABEL_DATA_DIR = "data/processed/label"
    RAW_DATA_DIR = "data/raw/new/augmented_ply"
    PROCESSED_DATA_DIR = "data/processed/new"
    TEST_DATA_DIR = LABEL_DATA_DIR
    TEST_DATA_DIR = "data/test/new"  # Using correctly normalized training samples
    PREDICTED_DATA_DIR = "data/predictions/new"
    SPLITS_DIR = "data/splits/new"
    K_NEIGHBORS = 20
    RADIUS_RATIO = 2.0

    # --- Model Architecture ---
    # Type of side gate to use
    SIDE_GATE_TYPE = "qwen"
    # Input feature channels (from preprocessed features: x,y,z,nx,ny,nz,kappa,rho)
    INPUT_DIM = 8
    # Internal model feature dimension. Must be divisible by N_HEADS.
    D_MODEL = 12  # increased so it's divisible by 3 when using 3 heads
    N_HEADS = 3  # T-head, N-head, C-head
    N_LAYERS = 3 # Geometry -> Local -> Geometry -> Global
    FFN_DIM = 64 # Feed-forward network dimension in Transformer block

    # --- Attention Parameters ---
    ALPHA0 = 2.0 
    BETA0 = 0.5
    GAMMA = 1.0
    SIGMA = 0.1
    WELD_WIDTH_RANGE = [0.005, 0.02] # in meters

    # --- Training ---
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4  # for pretraining
    FINETUNE_LR = 1e-4    # learning rate for finetuning (increased from 5e-5 for faster convergence)
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 150  # increased for better convergence
    MASK_RATIO = 0.1 # fraction of points to mask per-sample
    CURVATURE_THRESHOLD = 0.0008 # for pseudo-labeling
    # Number of DataLoader worker processes. Use 0 on Windows for stability/debug.
    NUM_WORKERS = 0

    # Focal Loss parameters
    USE_FOCAL_LOSS = True      # Use Focal Loss instead of BCE (better for extreme imbalance)
    FOCAL_GAMMA = 2.0          # Focusing parameter (2.0 is standard)
    FREEZE_BACKBONE = False    # UNFREEZE backbone for this short finetune run

    # Maximum points to keep per sample (for downsampling to control attention memory).
    # NOTE: the training code must implement sampling from this value to take effect.
    MAX_POINTS = 4096

    # Use automatic mixed precision (AMP) to reduce memory consumption. Requires
    # PyTorch with AMP support. The training loop must check USE_AMP to enable it.
    USE_AMP = True

    # Gradient accumulation steps (keeps physical batch small while simulating larger batch)
    ACCUM_STEPS = 1

    # If True, training will run only a single batch then exit (useful for debugging)
    DEBUG_SINGLE_BATCH = False

    # --- Inference ---
    PREDICTION_THRESHOLD = 0.50 # threshold for binary classification from sigmoid output

    # --- Finetune balancing & emphasis (temporary experimental settings) ---
    NEG_TO_POS_RATIO = 2       # target negatives per positive during downsampling
    MAX_NEG_PER_SAMPLE = 2000  # cap negatives per sample
    EMPHASIS_FACTOR = 3.0      # increase loss weight for emphasized files (e.g., halfv/lap)
    EMPHASIS_KEYS = ['halfv', 'half_v', 'lap']
    CURV_GAIN = 4.0            # curvature channel gain applied during finetune
    FINETUNE_LR = 5e-5         # smaller lr when unfreezing backbone
    # NUM_EPOCHS = 2             # short run for experiment

    # --- Paths ---
    WEIGHTS_SAVE_DIR = "weights"
    # Logging
    LOG_DIR = "logs"

    # Reconstruction loss per-channel weights: length must equal INPUT_DIM, or None for equal weights
    # Example: give more weight to XYZ and curvature
    # Example default: emphasize XYZ and curvature (kappa at index 6)
    RECON_WEIGHTS = [0.12, 0.12, 0.12, 0.06, 0.06, 0.06, 0.4, 0.06]

    # Additional weight to add an explicit L1 loss on curvature channel to preserve peaks
    CURV_L1_WEIGHT = 1.0

    SUPERVISED_TRAIN = True

    FINETUNE_CLASSIFIER = True

    ONLY_USE_LABELED = True

    CLASS_LOSS_WEIGHT = 1.0

    REINIT_CLASSIFIER_ON_FINETUNE = True
    
    PRETRAINED_WEIGHTS = "weights/best_pretrain.pth"

    TEST_WEIGHTS = "weights/best_finetune.pth"