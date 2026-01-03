# weld_seg_project/utils/config.py 配置文件，用于储存路径和训练参数等
from dataclasses import dataclass
@dataclass
class Config:
    # --- Data and Preprocessing ---
    LABEL_DATA_DIR = "data/processed/label"
    RAW_DATA_DIR = "data/aug_ply"
    PROCESSED_DATA_DIR = "data/processed_csv"
    TEST_DATA_DIR = "data/test"  # Using correctly normalized training samples
    PREDICTED_DATA_DIR = "data/predictions"
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
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4  # for pretraining
    FINETUNE_LR = 1e-4    # learning rate for finetuning (increased from 5e-5 for faster convergence)
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS = 300  # increased for better convergence
    MASK_RATIO = 0.1 # fraction of points to mask per-sample
    CURVATURE_THRESHOLD = 0.0008 # for pseudo-labeling
    # Number of DataLoader worker processes. Use 0 on Windows for stability/debug.
    NUM_WORKERS = 0

    # Focal Loss parameters
    USE_FOCAL_LOSS = True      # Use Focal Loss instead of BCE (better for extreme imbalance)
    FOCAL_GAMMA = 2.0          # Focusing parameter (2.0 is standard)
    FREEZE_BACKBONE = False    # UNFREEZE backbone for this short finetune run

    # If not using focal loss, use BCEWithLogitsLoss with pos_weight (neg/pos) to fight imbalance.
    USE_POS_WEIGHT = True
    POS_WEIGHT_MAX = 50.0

    # Maximum points to keep per sample (for downsampling to control attention memory).
    # NOTE: the training code must implement sampling from this value to take effect.
    MAX_POINTS = 10000

    # --- Pretraining practicality ---
    # This repo includes quadratic-attention blocks (Geometry/Global attention).
    # Pretraining must cap N to keep memory/time reasonable.
    # Reduced from 8192 to 2048 for memory constraints (O(N^2) attention is expensive)
    PRETRAIN_MAX_POINTS = 2048

    # Subsampling method when N > *_MAX_POINTS. Options: "random" (fast, default), "first".
    SUBSAMPLE_METHOD = "random"

    # Use automatic mixed precision (AMP) to reduce memory consumption. Requires
    # PyTorch with AMP support. The training loop must check USE_AMP to enable it.
    USE_AMP = True

    # Gradient accumulation steps (keeps physical batch small while simulating larger batch)
    ACCUM_STEPS = 1

    # During finetune, optionally ignore a fraction of point labels (simulate "masked labels").
    # 0.0 = use all labels; 0.5 = only half points contribute to loss.
    LABEL_MASK_RATIO = 0.0

    # Optional auxiliary loss that biases high-curvature points towards positive predictions.
    # WARNING: can introduce many false positives; keep at 0 unless you know it's helping.
    CURV_AUX_WEIGHT = 0.0

    # Which validation protocol to use for selecting the best checkpoint:
    # - "fps": validation uses unbiased FPS sampling (recommended)
    # - "balanced": validation uses label-aware balanced sampling (optimistic; mainly for debugging)
    MODEL_SELECTION = "fps"

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

    # --- Pretrain recon target scaling ---
    # Curvature raw is very small (~1e-4..1e-2). Reconstructing in log-space is much more stable.
    # Options: "raw" or "log"
    PRETRAIN_CURV_TARGET = "log"
    PRETRAIN_CURV_EPS = 1e-6
    
    # Normalization for other channels (density and linearity)
    # Options: "raw", "norm" (min-max to [0,1]), "log" (for density, use log(1+x))
    PRETRAIN_DENSITY_TARGET = "norm"  # local_density is in [0,1] range, normalize for better training
    PRETRAIN_LINEARITY_TARGET = "norm"  # linearity is in [0,1] range, normalize for better training
    
    # Per-channel loss weights for reconstruction (curvature, density, linearity)
    # Higher weight on curvature since it has larger error
    PRETRAIN_RECON_WEIGHTS = [5.0, 1.0, 1.5]  # [curvature, density, linearity]
    
    # Use normalized MSE loss (divide by channel std for stable training)
    PRETRAIN_USE_NORM_LOSS = True

    SUPERVISED_TRAIN = True

    FINETUNE_CLASSIFIER = True

    ONLY_USE_LABELED = True

    CLASS_LOSS_WEIGHT = 1.0

    REINIT_CLASSIFIER_ON_FINETUNE = True
    
    PRETRAINED_WEIGHTS = "weights/best_pretrain.pth"

    TEST_WEIGHTS = "weights/best_finetune.pth"