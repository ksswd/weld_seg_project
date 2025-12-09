# weld_seg_project/train/train.py 训练主函数
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from model.model import GeometryAwareTransformer
from train.mask_strategy import HighCurvatureMasker
from utils.metric_utils import calculate_metrics
from torch.utils.tensorboard import SummaryWriter
from utils.config import Config as GlobalConfig

class WeldDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        # features expected layout: [x, y, z, nx, ny, nz, kappa, rho]
        features = data['features'].astype(np.float32)

        # Validate features shape
        if features.ndim == 1:
            # single point -> make (1, C)
            features = features[None, :]
        if features.ndim != 2:
            raise ValueError(f"Unexpected features ndim={features.ndim} in file {self.file_list[idx]}")
        if features.shape[1] < 3:
            raise ValueError(f"Expected at least 3 feature channels (XYZ), got {features.shape[1]} in {self.file_list[idx]}")

        # Some older .npz files may not include 'normals' or 'local_density' as separate arrays.
        # Provide fallbacks by slicing from the features array when missing.
        if 'normals' in data:
            normals = data['normals'].astype(np.float32)
        else:
            normals = features[:, 3:6]

        if 'local_density' in data:
            local_density = data['local_density'].astype(np.float32)
        else:
            # rho is at index 7 in features
            local_density = features[:, 7:8] if features.ndim == 2 else features[:, 7]

        # curvature and principal_dir/linearity should exist; if not, try to extract curvature (kappa) from features
        if 'curvature' in data:
            curvature = data['curvature'].astype(np.float32)
        else:
            curvature = features[:, 6:7] if features.ndim == 2 else features[:, 6]

        # optional per-point labels for supervised classification (0/1). If missing,
        # fill with -1 to indicate unlabeled points.
        if 'labels' in data:
            labels = data['labels'].astype(np.float32)
            # ensure shape (N,1)
            if labels.ndim == 1:
                labels = labels[:, None]
        else:
            labels = -1.0 * np.ones((features.shape[0], 1), dtype=np.float32)

        principal_dir = data['principal_dir'].astype(np.float32) if 'principal_dir' in data else np.zeros_like(normals)
        linearity = data['linearity'].astype(np.float32) if 'linearity' in data else np.zeros((features.shape[0], 1), dtype=np.float32)

        return {
            'features': features,
            'principal_dir': principal_dir,
            'curvature': curvature,
            'local_density': local_density,
            'normals': normals,
            'linearity': linearity,
            'labels': labels,
        }


def collate_fn(batch):
    """Pad variable-length point clouds in a batch to the same number of points.
    Returns a dict of tensors and a mask where 1 indicates valid points.
    """
    # determine max points in this batch
    # respect global MAX_POINTS from config if available
    global_max = getattr(GlobalConfig, 'MAX_POINTS', None)
    max_pts = max(item['features'].shape[0] for item in batch)
    max_pts = min(max_pts, global_max)

    def pad_batch(arr_list, out_shape, pad_value=0.0):
        # arr_list: list of numpy arrays, each with shape (N, C) or (N,) or (N,1)
        out = np.full(out_shape, pad_value, dtype=arr_list[0].dtype)
        expected_C = out_shape[-1]
        for i, arr in enumerate(arr_list):
            a = arr
            # normalize 1D arrays to (N,1)
            if a.ndim == 1:
                a = a[:, None]

            # If channel dim doesn't match expected, pad or truncate columns
            if a.ndim == 2 and a.shape[1] != expected_C:
                Cc = a.shape[1]
                if Cc < expected_C:
                    tmp = np.full((a.shape[0], expected_C), pad_value, dtype=a.dtype)
                    tmp[:, :Cc] = a
                    a = tmp
                else:
                    # truncate extra channels
                    a = a[:, :expected_C]

            out[i, :a.shape[0], ...] = a
        return out

    batch_size = len(batch)

    # Optionally downsample per-sample to global_max before padding to limit N
    sampled_features = []
    sampled_normals = []
    sampled_principal = []
    sampled_curvature = []
    sampled_density = []
    sampled_linearity = []
    sampled_labels = []

    for item in batch:
        n = item['features'].shape[0]
        if global_max is not None and n > global_max:
            inds = np.random.choice(n, global_max, replace=False)
            inds = np.sort(inds)
            sampled_features.append(item['features'][inds])
            sampled_normals.append(item['normals'][inds])
            sampled_principal.append(item['principal_dir'][inds])
            sampled_curvature.append(item['curvature'][inds])
            sampled_density.append(item['local_density'][inds])
            sampled_linearity.append(item['linearity'][inds])
            sampled_labels.append(item['labels'][inds])
        else:
            sampled_features.append(item['features'])
            sampled_normals.append(item['normals'])
            sampled_principal.append(item['principal_dir'])
            sampled_curvature.append(item['curvature'])
            sampled_density.append(item['local_density'])
            sampled_linearity.append(item['linearity'])
            sampled_labels.append(item['labels'])

    # features -> (B, max_pts, C)
    feat_dim = sampled_features[0].shape[1]
    features = torch.from_numpy(pad_batch(sampled_features, (batch_size, max_pts, feat_dim))).float()

    # normals (B, max_pts, 3)
    normals = torch.from_numpy(pad_batch(sampled_normals, (batch_size, max_pts, sampled_normals[0].shape[1]))).float()

    # principal_dir (B, max_pts, 3)
    principal_dir = torch.from_numpy(pad_batch(sampled_principal, (batch_size, max_pts, sampled_principal[0].shape[1]))).float()

    # curvature (B, max_pts, 1)
    curvature = torch.from_numpy(pad_batch(sampled_curvature, (batch_size, max_pts, sampled_curvature[0].shape[1]))).float()

    # local_density (B, max_pts, 1)
    local_density = torch.from_numpy(pad_batch(sampled_density, (batch_size, max_pts, sampled_density[0].shape[1]))).float()

    # linearity (B, max_pts, 1)
    linearity = torch.from_numpy(pad_batch(sampled_linearity, (batch_size, max_pts, sampled_linearity[0].shape[1]))).float()

    # labels (B, max_pts, 1) - may be -1 for unlabeled points
    labels = torch.from_numpy(pad_batch(sampled_labels, (batch_size, max_pts, sampled_labels[0].shape[1]), pad_value=-1.0)).float()

    # mask: 1 for valid points, 0 for padding
    mask = torch.zeros((batch_size, max_pts), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item['features'].shape[0]
        mask[i, :n] = 1

    return {
        'features': features,
        'principal_dir': principal_dir,
        'curvature': curvature,
        'local_density': local_density,
        'normals': normals,
        'linearity': linearity,
        'labels': labels,
        'mask': mask,
    }

def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Datasets and DataLoaders
    # Only include original preprocessed .npz files; exclude prediction outputs (e.g., *_pred.npz)
    # Only include original preprocessed .npz files; exclude prediction outputs (e.g., *_pred.npz)
    all_files = [os.path.join(config.PROCESSED_DATA_DIR, f)
                   for f in os.listdir(config.PROCESSED_DATA_DIR)
                   if f.endswith('.npz') and ('_pred' not in f)]
    
     # Model, Loss, Optimizer
    model = GeometryAwareTransformer(config).to(device)
    masker = HighCurvatureMasker(mask_ratio=config.MASK_RATIO)
    # Optionally load pretrained weights for backbone initialization.
    # Set `PRETRAINED_WEIGHTS` in your config to a path (absolute or relative) or
    # to a filename located in `WEIGHTS_SAVE_DIR`.
    pretrained_path = getattr(config, 'PRETRAINED_WEIGHTS', None)
    if pretrained_path:
        # if given a relative name, try weights dir as well
        candidate_paths = [pretrained_path]
        if not os.path.isabs(pretrained_path):
            candidate_paths.append(os.path.join(getattr(config, 'WEIGHTS_SAVE_DIR', ''), pretrained_path))
        loaded = False
        for cand in candidate_paths:
            if cand and os.path.exists(cand):
                try:
                    print(f"Loading pretrained weights from: {cand}")
                    state = torch.load(cand, map_location=device)
                    # load with strict=False so that missing or extra heads (classifier/recon)
                    # won't break loading; this allows swapping/reinitializing heads for finetune.
                    model.load_state_dict(state, strict=False)
                    loaded = True
                    break
                except Exception as e:
                    print(f"Warning: failed to load pretrained weights from {cand}: {e}")
        if not loaded:
            print(f"PRETRAINED_WEIGHTS specified but no valid file found among: {candidate_paths}")
        else:
            # Optionally reinitialize classifier if user prefers fresh classification head on finetune
            if getattr(config, 'REINIT_CLASSIFIER_ON_FINETUNE', False):
                try:
                    print('REINIT_CLASSIFIER_ON_FINETUNE=True: reinitializing classifier head')
                    for m in model.classifier.modules():
                        if isinstance(m, torch.nn.Linear):
                            torch.nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                torch.nn.init.zeros_(m.bias)
                except Exception:
                    pass
    # For reconstruction training use MSELoss per-feature; keep BCE available if needed
    recon_criterion = torch.nn.MSELoss(reduction='none')
    # class_criterion will be created later (after pos_weight computation) so it
    # can optionally include an automatically computed pos_weight tensor.
    class_criterion = None
    # supervised training flags
    supervised = getattr(config, 'SUPERVISED_TRAIN', False)
    finetune_classifier = getattr(config, 'FINETUNE_CLASSIFIER', False)
    class_loss_weight = float(getattr(config, 'CLASS_LOSS_WEIGHT', 1.0))
    # reconstruction per-channel weights (list or None) - default equal weights
    recon_weights = getattr(config, 'RECON_WEIGHTS', None)
    if recon_weights is not None:
        recon_weights = torch.tensor(recon_weights, device=device).float().view(1, 1, -1)
    else:
        recon_weights = None
    # Optionally freeze backbone and only train classifier (create optimizer accordingly)
    if finetune_classifier:
        for name, p in model.named_parameters():
            p.requires_grad = False
        for name, p in model.classifier.named_parameters():
            p.requires_grad = True

    # optimizer and scheduler will be created after dataset split so we can set param groups
    optimizer = None
    scheduler = None

    best_val_f1 = 0.0
    best_val_loss = float('inf')
    os.makedirs(config.WEIGHTS_SAVE_DIR, exist_ok=True)

    # If requested, restrict supervised training to only files that contain labels
    only_labeled = supervised and getattr(config, 'ONLY_USE_LABELED', False)
    if only_labeled:
        labeled_files = [os.path.join(config.LABEL_DATA_DIR, f)
                   for f in os.listdir(config.LABEL_DATA_DIR)
                   if f.endswith('.npz') and ('_pred' not in f)]      
        print('Found', len(labeled_files), 'labeled files for supervised training.') 
        if len(labeled_files) == 0:
            print('WARNING: ONLY_USE_LABELED=True but no labeled files found; falling back to all files')
            files_to_use = all_files
        else:
            files_to_use = labeled_files
    else:
        files_to_use = all_files

    # If supervised and AUTO_POS_WEIGHT enabled, compute positive/negative counts
    auto_pos_weight = supervised and getattr(config, 'AUTO_POS_WEIGHT', False)
    pos_weight_tensor = None
    if auto_pos_weight:
        total_pos = 0
        total_neg = 0
        for p in files_to_use:
            try:
                d = np.load(p)
                if 'labels' in d.files:
                    labs = d['labels'].reshape(-1)
                    valid = labs >= 0
                    total_pos += int(((labs==1) & valid).sum())
                    total_neg += int(((labs==0) & valid).sum())
            except Exception:
                continue
        if total_pos == 0:
            print('AUTO_POS_WEIGHT: no positive labels found; skipping pos_weight')
        else:
            pw = float(total_neg) / float(total_pos + 1e-8)
            import torch as _torch
            pos_weight_tensor = _torch.tensor(pw, device=device)
            print(f"AUTO_POS_WEIGHT computed: pos={total_pos}, neg={total_neg}, pos_weight={pw:.4f}")

    # For simplicity, we use a single dataset for train and val by splitting files_to_use
    train_dataset = WeldDataset(files_to_use[:int(0.8*len(files_to_use))])
    val_dataset = WeldDataset(files_to_use[int(0.8*len(files_to_use)):])
    # train_dataset = WeldDataset(train_files[:int(1*len(train_files))])
    # val_dataset = train_dataset
    
    # Use pinned memory to speed up host->device copies when using CUDA
    pin_memory = True if device.type == 'cuda' else False
    num_workers = getattr(config, 'NUM_WORKERS', 4) if pin_memory else 0
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, pin_memory=pin_memory, num_workers=num_workers)

    # If class_criterion deferred creation (to include pos_weight), create it now
    if class_criterion is None:
        if pos_weight_tensor is not None:
            class_criterion = torch.nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight_tensor)
            print('Using BCEWithLogitsLoss with pos_weight')
        else:
            class_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    # Build optimizer after loaders so we can create param groups for different LRs
    ft_class_lr = getattr(config, 'FT_CLASS_LR', None)
    ft_backbone_lr = getattr(config, 'FT_BACKBONE_LR', None)
    if ft_class_lr is not None or ft_backbone_lr is not None:
        backbone_params = [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad]
        classifier_params = [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad]
        param_groups = []
        if backbone_params:
            param_groups.append({'params': backbone_params, 'lr': ft_backbone_lr if ft_backbone_lr is not None else config.LEARNING_RATE})
        if classifier_params:
            param_groups.append({'params': classifier_params, 'lr': ft_class_lr if ft_class_lr is not None else config.LEARNING_RATE})
        optimizer = torch.optim.AdamW(param_groups, weight_decay=config.WEIGHT_DECAY)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    # scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    # AMP scaler if requested
    # Use torch.amp GradScaler with explicit device_type to avoid deprecation warnings
    scaler = torch.amp.GradScaler('cuda') if getattr(config, 'USE_AMP', False) and device.type == 'cuda' else None

    for epoch in range(config.NUM_EPOCHS):
        # create writer on first epoch
        if epoch == 0:
            log_dir = getattr(config, 'LOG_DIR', getattr(GlobalConfig, 'LOG_DIR', 'logs'))
            writer = SummaryWriter(log_dir=log_dir)
        model.train()
        # If requested, freeze backbone and train only classifier
        if finetune_classifier:
            for name, p in model.named_parameters():
                p.requires_grad = False
            for name, p in model.classifier.named_parameters():
                p.requires_grad = True
        train_loss = 0.0
        if getattr(config, 'DEBUG_SINGLE_BATCH', False):
            print('DEBUG_SINGLE_BATCH enabled: will run only first batch and then exit')
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        # training loop with optional AMP and gradient accumulation
        step_count = 0
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")):
            if getattr(config, 'DEBUG_SINGLE_BATCH', False) and batch_idx > 0:
                break

            # Move data to device (use non_blocking for pinned memory)
            batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # Generate mask and prepare masked features
            mask = masker.generate_mask(batch['curvature'])
            print('batch_idx', batch_idx, 'mask any?', mask.any().item(), 'mask.sum', mask.sum().item())
            if mask.dim() == 3 and mask.size(-1) == 1:
                mask = mask.squeeze(-1)
            mask = mask.to(torch.bool)

            # Mask only XYZ channels for masked points (self-supervised coordinate reconstruction)
            masked_features = batch['features'].clone()
            # masked_features[mask] selects all masked points flattened; zero only first 3 channels (x,y,z)
            if masked_features.dim() == 3:
                masked_features[mask][:, :3] = 0.0
            else:
                # fallback: treat as (B*N, C)
                masked_features[mask] = masked_features[mask]

            # Ground-truth coordinates for masked points (for reconstruction loss)
            gt_masked_coords = batch['features'][mask][:, :3]

            use_amp = getattr(config, 'USE_AMP', False) and device.type == 'cuda'

            # Forward: ask model to reconstruct features
            if use_amp:
                with torch.amp.autocast(device_type='cuda'):
                    recon = model(
                        masked_features,
                        batch['principal_dir'],
                        batch['curvature'],
                        batch['local_density'],
                        batch['normals'],
                        batch['linearity'],
                        task='recon'
                    )
                    # recon: (B, N, C)
                    # compute per-element MSE only on XYZ channels
                    per_elem = recon_criterion(recon[..., :3], batch['features'][..., :3])  # (B,N,3)
                    # apply per-channel weights if provided (slice to XYZ)
                    if recon_weights is not None:
                        per_elem = per_elem * recon_weights[..., :3]
                    # sum over xyz channels then mean over masked points
                    per_point = per_elem.sum(dim=-1)  # (B,N)
                    masked_per_point = per_point[mask]
                    if masked_per_point.numel() == 0:
                        # create a zero loss that has a grad_fn (so backward works)
                        loss = recon[..., :3].sum() * 0.0
                    else:
                        loss = masked_per_point.mean()

                    # supervised classification loss (optional)
                    # initialize class_loss as zero with grad_fn to keep graph intact
                    class_loss = recon[..., :3].sum() * 0.0
                    if supervised:
                        # compute logits using full (unmasked) features
                        class_logits = model(
                            batch['features'],
                            batch['principal_dir'],
                            batch['curvature'],
                            batch['local_density'],
                            batch['normals'],
                            batch['linearity'],
                            task='class'
                        )
                        # class_logits: (B,N,1), labels: (B,N,1) with -1 for unlabeled
                        labels = batch['labels']
                        labeled_mask = (labels >= 0.0)
                        if labeled_mask.any():
                            # BCEWithLogitsLoss reduction='none' -> (B,N)
                            cls_per_elem = class_criterion(class_logits.squeeze(-1), labels.squeeze(-1))
                            cls_vals = cls_per_elem[labeled_mask.squeeze(-1)]
                            if cls_vals.numel() > 0:
                                class_loss = cls_vals.mean()

                    total_loss = loss + class_loss_weight * class_loss

                # scale the loss, backward, and step according to ACCUM_STEPS
                if scaler is not None:
                    scaler.scale(total_loss / getattr(config, 'ACCUM_STEPS', 1)).backward()
                else:
                    (total_loss / getattr(config, 'ACCUM_STEPS', 1)).backward()
            else:
                recon = model(
                    masked_features,
                    batch['principal_dir'],
                    batch['curvature'],
                    batch['local_density'],
                    batch['normals'],
                    batch['linearity'],
                    task='recon'
                )
                per_elem = recon_criterion(recon[..., :3], batch['features'][..., :3])
                if recon_weights is not None:
                    per_elem = per_elem * recon_weights[..., :3]
                per_point = per_elem.sum(dim=-1)
                masked_per_point = per_point[mask]
                if masked_per_point.numel() == 0:
                    loss = recon[..., :3].sum() * 0.0
                else:
                    loss = masked_per_point.mean()
                # supervised classification loss (optional)
                class_loss = recon[..., :3].sum() * 0.0
                if supervised:
                    class_logits = model(
                        batch['features'],
                        batch['principal_dir'],
                        batch['curvature'],
                        batch['local_density'],
                        batch['normals'],
                        batch['linearity'],
                        task='class'
                    )
                    labels = batch['labels']
                    labeled_mask = (labels >= 0.0)
                    if labeled_mask.any():
                        cls_per_elem = class_criterion(class_logits.squeeze(-1), labels.squeeze(-1))
                        cls_vals = cls_per_elem[labeled_mask.squeeze(-1)]
                        if cls_vals.numel() > 0:
                            class_loss = cls_vals.mean()

                total_loss = loss + class_loss_weight * class_loss
                (total_loss / getattr(config, 'ACCUM_STEPS', 1)).backward()

            step_count += 1
            if step_count % getattr(config, 'ACCUM_STEPS', 1) == 0:
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            # compute per-channel masked MSE for logging (only XYZ channels)
            try:
                per_elem = recon_criterion(recon[..., :3], batch['features'][..., :3])  # (B,N,3)
                if recon_weights is not None:
                    per_elem = per_elem * recon_weights[..., :3]
                masked_sel = per_elem[mask]  # (num_masked, 3)
                if masked_sel.numel() > 0:
                    per_channel_mean = masked_sel.mean(dim=0).detach().cpu().numpy()
                else:
                    per_channel_mean = np.zeros((3,), dtype=np.float32)
            except Exception:
                per_channel_mean = np.zeros((3,), dtype=np.float32)

            # training classification metrics/logging
            train_class_loss_val = 0.0
            if supervised:
                try:
                    # class_logits may only be defined when supervised; compute if not present
                    if 'class_logits' not in locals():
                        class_logits = model(
                            batch['features'],
                            batch['principal_dir'],
                            batch['curvature'],
                            batch['local_density'],
                            batch['normals'],
                            batch['linearity'],
                            task='class'
                        )
                    labels = batch['labels']
                    labeled_mask = (labels >= 0.0)
                    if labeled_mask.any():
                        cls_preds = torch.sigmoid(class_logits.squeeze(-1))
                        cls_per_elem = class_criterion(class_logits.squeeze(-1), labels.squeeze(-1))
                        cls_vals = cls_per_elem[labeled_mask.squeeze(-1)]
                        if cls_vals.numel() > 0:
                            train_class_loss_val = float(cls_vals.mean().detach().cpu().numpy())
                except Exception:
                    train_class_loss_val = 0.0

            train_loss += loss.item() * batch['features'].size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        # supervised classification validation accumulators
        val_class_pred_list = []
        val_class_label_list = []
        # Validation: collect curvature samples for diagnostics (up to total_sample_limit)
        curvature_samples = []
        total_sample_limit = 5000
        with torch.no_grad():
            # collect reconstructed-curvature scores across validation for threshold tuning
            recon_scores_list = []
            true_scores_list = []
            for batch_idx, batch in enumerate(val_loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                # collect curvature samples (flattened) for stats (sample if too large)
                try:
                    cur_cpu = batch['curvature'].cpu().numpy().ravel()
                    if cur_cpu.size > 0 and len(curvature_samples) < total_sample_limit:
                        need = total_sample_limit - len(curvature_samples)
                        if cur_cpu.size <= need:
                            curvature_samples.append(cur_cpu)
                        else:
                            inds = np.random.choice(cur_cpu.size, need, replace=False)
                            curvature_samples.append(cur_cpu[inds])
                except Exception:
                    pass

                gt_labels = (batch['curvature'] > config.CURVATURE_THRESHOLD).float()

                # For validation, run the same masked reconstruction as training so metrics reflect
                # the model's ability to recover masked features.
                v_mask = masker.generate_mask(batch['curvature'])
                if v_mask.dim() == 3 and v_mask.size(-1) == 1:
                    v_mask = v_mask.squeeze(-1)
                v_mask = v_mask.to(torch.bool)

                v_masked_features = batch['features'].clone()
                # zero only xyz for validation masked points
                v_masked_features[v_mask][:, :3] = 0.0

                recon = model(
                    v_masked_features,
                    batch['principal_dir'],
                    batch['curvature'],
                    batch['local_density'],
                    batch['normals'],
                    batch['linearity'],
                    task='recon'
                )

                per_elem = recon_criterion(recon[..., :3], batch['features'][..., :3])
                if recon_weights is not None:
                    per_elem = per_elem * recon_weights[..., :3]
                per_point = per_elem.sum(dim=-1)  # (B,N)
                # compute mean over masked points to get val loss (mirrors training)
                masked_per_point = per_point[v_mask]
                if masked_per_point.numel() == 0:
                    vloss = torch.tensor(0.0, device=device)
                else:
                    vloss = masked_per_point.mean()
                # compute per-channel masked MSE for validation logging
                try:
                    per_elem = recon_criterion(recon[..., :3], batch['features'][..., :3])
                    if recon_weights is not None:
                        per_elem = per_elem * recon_weights[..., :3]
                    masked_sel = per_elem[v_mask]
                    if masked_sel.numel() > 0:
                        val_per_channel_mean = masked_sel.mean(dim=0).detach().cpu().numpy()
                    else:
                        val_per_channel_mean = np.zeros((3,), dtype=np.float32)
                except Exception:
                    val_per_channel_mean = np.zeros((3,), dtype=np.float32)

                val_loss += vloss.item() * batch['features'].size(0)

                # supervised classification validation (collect predictions and labels)
                if supervised:
                    try:
                        class_logits = model(
                            batch['features'],
                            batch['principal_dir'],
                            batch['curvature'],
                            batch['local_density'],
                            batch['normals'],
                            batch['linearity'],
                            task='class'
                        )
                        labels = batch['labels']
                        labeled_mask = (labels >= 0.0)
                        if labeled_mask.any():
                            print("Computing supervised classification metrics on validation batch...")
                            probs = torch.sigmoid(class_logits.squeeze(-1)).detach().cpu().numpy().reshape(-1)
                            labs = labels.cpu().numpy().reshape(-1)
                            mask_flat = labeled_mask.cpu().numpy().reshape(-1)
                            if mask_flat.sum() > 0:
                                val_class_pred_list.append(probs[mask_flat])
                                val_class_label_list.append(labs[mask_flat])
                    except Exception:
                        pass

                # collect reconstructed curvature scores and true labels for threshold tuning
                try:
                    recon_curv = recon[:, :, 6]
                    probs = recon_curv.detach().cpu().numpy().reshape(-1)
                    trues = batch['curvature'].cpu().numpy().reshape(-1)
                    recon_scores_list.append(probs)
                    true_scores_list.append((trues > config.CURVATURE_THRESHOLD).astype(np.float32))
                except Exception:
                    pass

                # print a small example from the first validation batch to help debugging
                if batch_idx == 0:
                    try:
                        example_curv = batch['curvature'][0, :min(10, batch['curvature'].shape[1]), 0].cpu().numpy()
                        # print(f"Example curvature (first sample, up to 10 values): {example_curv}")
                    except Exception:
                        pass

        # After validation, aggregate curvature samples and print statistics
        try:
            if len(curvature_samples) > 0:
                cur_all = np.concatenate(curvature_samples)
                # print("Curvature stats (sampled): min={:.6f}, 25%={:.6f}, median={:.6f}, 75%={:.6f}, 90%={:.6f}, 95%={:.6f}, max={:.6f}, mean={:.6f}, std={:.6f}".format(
                #     cur_all.min(), np.percentile(cur_all, 25), np.percentile(cur_all, 50), np.percentile(cur_all, 75), np.percentile(cur_all, 90), np.percentile(cur_all, 95), cur_all.max(), cur_all.mean(), cur_all.std()
                # ))
        except Exception:
            pass
                
        val_loss /= len(val_loader.dataset)
        # compute best threshold from recon scores (simple numpy search) if we have data
        metrics = None
        best_thresh = config.CURVATURE_THRESHOLD
        try:
            if len(recon_scores_list) > 0:
                recon_scores = np.concatenate(recon_scores_list)
                true_scores = np.concatenate(true_scores_list)
                # perform simple threshold sweep to maximize F1
                order = np.argsort(-recon_scores)
                sorted_scores = recon_scores[order]
                sorted_labels = true_scores[order]
                tp = 0
                fp = 0
                fn = int(sorted_labels.sum())
                tn = len(sorted_labels) - fn
                best_f1 = 0.0
                best_t = best_thresh
                # sweep thresholds at unique scores
                prev_score = None
                for s, lab in zip(sorted_scores, sorted_labels):
                    if lab == 1:
                        tp += 1
                        fn -= 1
                    else:
                        fp += 1
                        tn -= 1
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * prec * rec / (prec + rec + 1e-12) if (prec + rec) > 0 else 0.0
                    if f1 > best_f1:
                        best_f1 = f1
                        best_t = s
                best_thresh = float(best_t)
                # produce binary preds and labels for metrics
                all_preds = (recon_scores > best_thresh).astype(np.float32).reshape(-1, 1)
                all_labels = true_scores.reshape(-1, 1)
                metrics = calculate_metrics(all_preds, all_labels)
                # log best threshold and F1
                try:
                    writer.add_scalar('threshold/best_thresh', best_thresh, epoch+1)
                    writer.add_scalar('metrics/val_best_f1', best_f1, epoch+1)
                except Exception:
                    pass
        except Exception:
            pass
        if metrics is None:
            all_preds = np.vstack(all_preds)
            all_labels = np.vstack(all_labels)
            metrics = calculate_metrics(all_preds, all_labels)
        
        # If supervised labels were provided, compute classification metrics on val set
        class_metrics = None
        if supervised and len(val_class_pred_list) > 0:
            print("Computing supervised classification metrics on validation set...")
            try:
                val_preds = np.concatenate(val_class_pred_list)
                val_labels = np.concatenate(val_class_label_list)
                # binarize preds at 0.5 for metrics
                val_pred_bin = (val_preds > 0.5).astype(np.float32).reshape(-1, 1)
                val_label_bin = val_labels.reshape(-1, 1)
                class_metrics = calculate_metrics(val_pred_bin, val_label_bin)
                try:
                    writer.add_scalar('metrics/val_class_accuracy', class_metrics['accuracy'], epoch+1)
                    writer.add_scalar('metrics/val_class_f1', class_metrics['f1_score'], epoch+1)
                except Exception:
                    pass
            except Exception:
                class_metrics = None
        open(os.path.join(config.WEIGHTS_SAVE_DIR, "best_thresh.txt"), "w").write(str(best_thresh))
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {metrics['accuracy']:.4f}, Val F1: {metrics['f1_score']:.4f}")
        if class_metrics is not None:
            print(f"Epoch {epoch+1}, Val Class Acc: {class_metrics['accuracy']:.4f}, Val Class F1: {class_metrics['f1_score']:.4f}")

        # write to tensorboard
        try:
            writer.add_scalar('loss/train', train_loss, epoch+1)
            writer.add_scalar('loss/val', val_loss, epoch+1)
            if supervised:
                try:
                    writer.add_scalar('loss/train_class', float(train_class_loss_val), epoch+1)
                except Exception:
                    pass
            writer.add_scalar('metrics/val_accuracy', metrics['accuracy'], epoch+1)
            writer.add_scalar('metrics/val_f1', metrics['f1_score'], epoch+1)
            # per-channel (log up to INPUT_DIM channels)
            for ci in range(min(len(per_channel_mean), getattr(config, 'INPUT_DIM', per_channel_mean.shape[0]))):
                writer.add_scalar(f'per_channel/train_ch{ci}', float(per_channel_mean[ci]), epoch+1)
            for ci in range(min(len(val_per_channel_mean), getattr(config, 'INPUT_DIM', val_per_channel_mean.shape[0]))):
                writer.add_scalar(f'per_channel/val_ch{ci}', float(val_per_channel_mean[ci]), epoch+1)
        except Exception:
            pass

        # Diagnostics: show number of true positives and predicted positives in validation
        try:
            total_true = int(all_labels.sum())
            total_pred = int(all_preds.sum())
            print(f"Validation positives: true={total_true}, predicted={total_pred}")
            if total_true == 0:
                print("Warning: no positive samples in validation labels (check CURVATURE_THRESHOLD or label generation). F1 will be 0.")
        except Exception:
            pass

        # Save best model by validation loss (and by F1 if it improves)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, "best_model_by_loss.pth"))
            print(f"Saved best model by val loss: {best_val_loss:.4f}")

        if metrics['f1_score'] > best_val_f1:
            best_val_f1 = metrics['f1_score']
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, "best_model_by_f1.pth"))
            print(f"Saved best model with F1-score: {best_val_f1:.4f}")
            
        scheduler.step()