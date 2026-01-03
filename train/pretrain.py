# pretrain.py
import os, torch, numpy as np, torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model import GeometryAwareTransformer
from train.mask_strategy import HighCurvatureMasker
from utils.config import Config as GlobalConfig
from utils.io_utils import load_features_from_csv

class WeldDataset(Dataset):
    """从CSV文件加载预处理后的点云数据"""
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
       return load_features_from_csv(self.file_list[idx])

def collate_fn(batch):
    # Pretraining must cap N (quadratic attention blocks).
    global_max = getattr(GlobalConfig, 'PRETRAIN_MAX_POINTS', None) or getattr(GlobalConfig, 'MAX_POINTS', None)
    subsample_method = getattr(GlobalConfig, 'SUBSAMPLE_METHOD', 'random')
    max_pts = max(item['features'].shape[0] for item in batch)
    max_pts = min(max_pts, global_max) if global_max else max_pts
    # Subsample per-sample to max_pts to avoid bias from "first N points".
    def maybe_subsample(item, n_keep):
        n = item['features'].shape[0]
        if n <= n_keep:
            return item
        if subsample_method == 'first':
            idx = np.arange(n_keep)
        else:
            # random subsample (fast). Replace with FPS later if needed.
            idx = np.random.choice(n, size=n_keep, replace=False)
        out = {}
        for k, v in item.items():
            if v is None:
                out[k] = None
            else:
                out[k] = v[idx]
        return out
    batch = [maybe_subsample(it, max_pts) for it in batch]
    def pad(arr_list, shape):
        out = np.full(shape, 0.0, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            n = min(arr.shape[0], shape[1])
            out[i, :n] = arr[:n]
        return out
    b, c = len(batch), batch[0]['features'].shape[1]
    feats = pad([b['features'] for b in batch], (b, max_pts, c))
    coordinate = pad([b['coordinate'] for b in batch], (b, max_pts, 3))
    normals = pad([b['normals'] for b in batch], (b, max_pts, 3))
    principal = pad([b['principal_dir'] for b in batch], (b, max_pts, 3))
    curvature = pad([b['curvature'] for b in batch], (b, max_pts, 1))
    density = pad([b['local_density'] for b in batch], (b, max_pts, 1))
    linearity = pad([b['linearity'] for b in batch], (b, max_pts, 1))
    mask = torch.zeros(b, max_pts, dtype=torch.bool)
    for i, item in enumerate(batch):
        mask[i, :item['features'].shape[0]] = 1
    return {
        'features': torch.from_numpy(feats),
        'coordinate': torch.from_numpy(coordinate),
        'normals': torch.from_numpy(normals),
        'principal_dir': torch.from_numpy(principal),
        'curvature': torch.from_numpy(curvature),
        'local_density': torch.from_numpy(density),
        'linearity': torch.from_numpy(linearity),
        'mask': mask
    }

def run_pretrain(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_files = [os.path.join(config.PROCESSED_DATA_DIR, f)
                 for f in os.listdir(config.PROCESSED_DATA_DIR)
                 if f.endswith('.csv') and '_pred' not in f]
    train_files = all_files[:int(0.8 * len(all_files))]
    val_files = all_files[int(0.8 * len(all_files)):]
    train_loader = DataLoader(WeldDataset(train_files), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=getattr(config, 'NUM_WORKERS', 0), pin_memory=True)
    val_loader = DataLoader(WeldDataset(val_files), batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=getattr(config, 'NUM_WORKERS', 0), pin_memory=True)

    model = GeometryAwareTransformer(config).to(device)
    masker = HighCurvatureMasker(mask_ratio=config.MASK_RATIO)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    # Step LR per-iteration to avoid scheduler warnings when AMP skips a step,
    # and to behave consistently regardless of early breaks.
    total_steps = max(1, int(getattr(config, 'NUM_EPOCHS', 1)) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    # torch.cuda.amp.* is deprecated in recent PyTorch; use torch.amp.*
    use_amp = bool(getattr(config, 'USE_AMP', False)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    writer = SummaryWriter(os.path.join(getattr(config, 'LOG_DIR', 'logs'), 'pretrain'))
    best_loss = float('inf')
    os.makedirs(config.WEIGHTS_SAVE_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"PT Epoch {epoch+1}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # 1) 清零梯度（必须）
            optimizer.zero_grad(set_to_none=True)
            # 2) 生成 mask & masked input
            valid_mask = batch['mask'].bool()
            mask = masker.generate_mask(batch['curvature'], valid_mask=valid_mask).squeeze(-1).bool()
            mask = mask & valid_mask  # never mask padding

            # Mask ALL feature channels that can leak targets (including curvature_norm/density_norm).
            masked_feat = batch['features'].clone()
            masked_feat[mask] = 0.0

            # Also mask per-point geometric side inputs at masked locations.
            masked_curv = batch['curvature'].clone()
            masked_dens = batch['local_density'].clone()
            masked_lin = batch['linearity'].clone()
            masked_normals = batch['normals'].clone()
            masked_principal = batch['principal_dir'].clone()

            masked_curv[mask] = 0.0
            masked_dens[mask] = 0.0
            masked_lin[mask] = 0.0
            masked_normals[mask] = 0.0
            masked_principal[mask] = 0.0
            # 3) 前向 + 反向（考虑 AMP）
            if scaler:
                with torch.amp.autocast('cuda'):
                    recon = model(
                        masked_feat,
                        batch['coordinate'],     # keep coordinates visible; we do NOT regress xyz to avoid leakage
                        masked_principal,
                        masked_curv,
                        masked_dens,
                        masked_normals,
                        masked_lin,
                        task='recon'
                    )
                    if not torch.isfinite(recon).all():
                        raise RuntimeError("Non-finite values in model output 'recon'")
                    loss = recon_criterion(recon, batch, mask)
                scaler.scale(loss).backward()
                # 在 unscale 之后裁剪梯度
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # step optimizer & scaler
                scaler.step(optimizer)
                scaler.update()
                # PyTorch may not update optimizer._step_count when stepping through GradScaler,
                # which triggers a noisy lr_scheduler warning. Bump it manually for scheduler bookkeeping.
                if getattr(optimizer, "_step_count", 0) < 1:
                    optimizer._step_count = 1
                scheduler.step()
            else:
                recon = model(
                    masked_feat,
                    batch['coordinate'],
                    masked_principal,
                    masked_curv,
                    masked_dens,
                    masked_normals,
                    masked_lin,
                    task='recon'
                )
                if not torch.isfinite(recon).all():
                    raise RuntimeError("Non-finite values in model output 'recon' (no AMP)")
                loss = recon_criterion(recon, batch, mask)
                if not torch.isfinite(loss):
                    print("Warning: non-finite loss detected (no AMP). setting to zero.")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
            total_loss += float(loss.item()) * batch['features'].size(0)

            if getattr(config, 'DEBUG_SINGLE_BATCH', False):
                print("DEBUG_SINGLE_BATCH=True, stopping after one batch.")
                break

        # 验证、记录、保存与 lr scheduler（与你原逻辑一致）
        val_loss = validate(model, val_loader, device, masker)
        writer.add_scalar('loss/train', total_loss / len(train_loader.dataset), epoch+1)
        writer.add_scalar('loss/val', val_loss, epoch+1)
        print(f"Epoch {epoch+1} | Train {total_loss/len(train_loader.dataset):.6f} | Val {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, "best_pretrain.pth"))
            print("Saved best_pretrain.pth")
        # scheduler stepped per-iteration above

def recon_criterion(recon, batch, mask):
    """
    Self-supervised reconstruction loss with normalization and per-channel weights.
    Only compute loss on masked points.

    Target (rotation-invariant):
      - curvature_raw (log-normalized)
      - local_density_raw (normalized)
      - linearity (normalized)
    """
    # recon: (B, N, 3) -> [curvature_target, density_target, linearity_target]
    B, N, _ = recon.shape
    
    # Normalize curvature
    curv = batch['curvature']
    curv_mode = str(getattr(GlobalConfig, "PRETRAIN_CURV_TARGET", "log")).lower().strip()
    if curv_mode == "log":
        eps = float(getattr(GlobalConfig, "PRETRAIN_CURV_EPS", 1e-6))
        curv_t = torch.log(curv.clamp_min(0) + eps)
    else:
        curv_t = curv
    
    # Normalize density
    dens = batch['local_density']
    dens_mode = str(getattr(GlobalConfig, "PRETRAIN_DENSITY_TARGET", "norm")).lower().strip()
    if dens_mode == "norm":
        # Min-max normalize to [0, 1] for stable training
        dens_min = dens.min(dim=1, keepdim=True)[0].min(dim=0, keepdim=True)[0]  # (1, 1, 1)
        dens_max = dens.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]  # (1, 1, 1)
        dens_range = (dens_max - dens_min).clamp_min(1e-8)
        dens_t = (dens - dens_min) / dens_range
    elif dens_mode == "log":
        dens_t = torch.log(dens.clamp_min(1e-8) + 1.0)
    else:
        dens_t = dens
    
    # Normalize linearity
    lin = batch['linearity']
    lin_mode = str(getattr(GlobalConfig, "PRETRAIN_LINEARITY_TARGET", "norm")).lower().strip()
    if lin_mode == "norm":
        lin_min = lin.min(dim=1, keepdim=True)[0].min(dim=0, keepdim=True)[0]
        lin_max = lin.max(dim=1, keepdim=True)[0].max(dim=0, keepdim=True)[0]
        lin_range = (lin_max - lin_min).clamp_min(1e-8)
        lin_t = (lin - lin_min) / lin_range
    else:
        lin_t = lin
    
    # Concatenate targets
    gt = torch.cat([curv_t, dens_t, lin_t], dim=-1)  # (B, N, 3)
    
    # Mask: only compute loss on masked points
    m = mask.unsqueeze(-1).float()  # (B, N, 1)
    num_masked = mask.sum().clamp(min=1).float()
    
    # Per-channel weights
    weights = torch.tensor(
        getattr(GlobalConfig, "PRETRAIN_RECON_WEIGHTS", [1.0, 1.0, 1.0]),
        device=recon.device,
        dtype=recon.dtype
    ).reshape(1, 1, 3)  # (1, 1, 3)
    
    # Compute per-channel MSE
    diff = (recon - gt) * m  # (B, N, 3)
    use_norm_loss = getattr(GlobalConfig, "PRETRAIN_USE_NORM_LOSS", True)
    
    if use_norm_loss:
        # Normalize by per-channel std (computed only on masked points for stability)
        # This makes loss scale-invariant and helps balance different channels
        channel_std = (diff ** 2).sum(dim=(0, 1), keepdim=True).sqrt().clamp_min(1e-8)  # (1, 1, 3)
        norm_diff = diff / channel_std
        per_channel_mse = (norm_diff ** 2).sum(dim=(0, 1)) / num_masked  # (3,)
    else:
        per_channel_mse = (diff ** 2).sum(dim=(0, 1)) / num_masked  # (3,)
    
    # Weighted sum
    loss = (per_channel_mse * weights.squeeze()).sum()
    
    return loss


@torch.no_grad()
def validate(model, loader, device, masker):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid_mask = batch['mask'].bool()
        mask = masker.generate_mask(batch['curvature'], valid_mask=valid_mask).squeeze(-1).bool()
        mask = mask & valid_mask

        masked_feat = batch['features'].clone()
        masked_feat[mask] = 0.0

        masked_curv = batch['curvature'].clone()
        masked_dens = batch['local_density'].clone()
        masked_lin = batch['linearity'].clone()
        masked_normals = batch['normals'].clone()
        masked_principal = batch['principal_dir'].clone()
        masked_curv[mask] = 0.0
        masked_dens[mask] = 0.0
        masked_lin[mask] = 0.0
        masked_normals[mask] = 0.0
        masked_principal[mask] = 0.0

        recon = model(
            masked_feat,
            batch['coordinate'],
            masked_principal,
            masked_curv,
            masked_dens,
            masked_normals,
            masked_lin,
            task='recon'
        )
        vloss = recon_criterion(recon, batch, mask)
        total += vloss.item() * batch['features'].size(0)
    return total / len(loader.dataset)

if __name__ == "__main__":
    from utils.config import Config
    run_pretrain(Config)