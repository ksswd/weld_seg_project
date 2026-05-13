# pretrain.py
import os, torch, numpy as np, torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model import GeometryAwareTransformer
from train.mask_strategy import HighCurvatureMasker, RandomMasker
from train.block_segmenter import BlockSegmenter
from train.block_masker import BlockMasker
from train.block_loss import block_recon_criterion as block_loss_fn
from utils.config import Config as GlobalConfig
from utils.io_utils import load_features_from_csv
from utils.downsampling import fps_with_cache as fps
from utils.fold_data_split import collect_files_by_fold

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
    def subsample(item, n_keep):
        n = item['features'].shape[0]
        if n <= n_keep:
            return item
        if subsample_method == 'fps':
            idx = fps(item['coordinate'], n_keep)
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
    batch = [subsample(it, max_pts) for it in batch]
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
    use_fold_split = bool(getattr(config, 'USE_FOLD_SPLIT', False))
    if use_fold_split:
        fold_id = getattr(config, 'FOLD_ID', 'fold_1')
        strict = bool(getattr(config, 'FOLD_SPLIT_STRICT', True))
        pretrain_dir = getattr(config, 'PRETRAIN_DATA_DIR', config.PROCESSED_DATA_DIR)
        include_labeled_in_pretrain = bool(getattr(config, 'PRETRAIN_INCLUDE_LABELED', False))
        train_files = collect_files_by_fold(
            mode='pretrain',
            fold_id=fold_id,
            data_dir=pretrain_dir,
            include_labeled_in_pretrain=include_labeled_in_pretrain,
            strict=strict,
            print_stats=True,
        )
        # 预训练验证集仍保持来自训练组（无泄漏前提下切分），用于监控收敛
        split_idx = max(1, int(0.9 * len(train_files)))
        val_files = train_files[split_idx:]
        train_files = train_files[:split_idx]
        if len(val_files) == 0:
            val_files = train_files[-1:]
            train_files = train_files[:-1] if len(train_files) > 1 else train_files
    else:
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
    
    # 根据配置选择mask策略（块级或点级）
    use_block_mask = getattr(config, 'USE_BLOCK_MASK', False)
    
    if use_block_mask:
        # 块级mask策略
        segmenter = BlockSegmenter(
            target_points_per_block=getattr(config, 'TARGET_POINTS_PER_BLOCK', 1000),
            high_curv_threshold=getattr(config, 'HIGH_CURV_THRESHOLD', 0.01),
            min_high_curv_points=getattr(config, 'MIN_HIGH_CURV_POINTS', 5),
            align_grid=True,
            grid_align_base=getattr(config, 'GRID_ALIGN_BASE', 0.001)
        )
        block_masker = BlockMasker(
            mask_ratio=getattr(config, 'BLOCK_MASK_RATIO', 0.3),
            strategy=getattr(config, 'BLOCK_MASK_STRATEGY', 'mixed'),  # 'mixed' or 'alternate'
            weld_mask_ratio=getattr(config, 'WELD_MASK_RATIO', None),
            bg_mask_ratio=getattr(config, 'BG_MASK_RATIO', None)
        )
        masker = None  # 不使用点级masker
        print("Using block-wise masking strategy")
    else:
        # 点级mask策略（原有方式）
        mask_type = getattr(config, 'MASK_TYPE', 'random').lower()
        if mask_type == 'random':
            masker = RandomMasker(mask_ratio=config.MASK_RATIO)
        elif mask_type == 'curvature':
            masker = HighCurvatureMasker(mask_ratio=config.MASK_RATIO)
        else:
            raise ValueError(f"Unknown MASK_TYPE: {mask_type}. Use 'random' or 'curvature'")
        segmenter = None
        block_masker = None
        print("Using point-wise masking strategy")
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
            
            if use_block_mask:
                # 块级mask策略
                # 对每个样本进行块分割
                blocks_list = []
                block_labels_list = []
                masks_list = []
                
                for b in range(batch['features'].shape[0]):
                    # 获取有效点
                    valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1).cpu().numpy()
                    if len(valid_indices) == 0:
                        blocks_list.append([])
                        block_labels_list.append([])
                        masks_list.append(torch.zeros(batch['features'].shape[1], dtype=torch.bool, device=device))
                        continue
                    
                    # 提取有效点的数据
                    points = batch['coordinate'][b][valid_indices].cpu().numpy()
                    curvature = batch['curvature'][b][valid_indices].cpu().numpy()
                    
                    # 块分割
                    blocks, block_labels = segmenter.segment(points, curvature)
                    
                    # 转换回原始索引
                    blocks_original = [[int(valid_indices[idx]) for idx in block] for block in blocks]
                    
                    # 生成块级mask
                    mask = block_masker.generate_mask(blocks_original, block_labels, epoch=epoch)
                    mask = mask.to(device)
                    # 确保只mask有效点
                    mask = mask & valid_mask[b]
                    
                    blocks_list.append(blocks_original)
                    block_labels_list.append(block_labels)
                    masks_list.append(mask)
                
                # 合并mask
                mask = torch.stack(masks_list)  # (B, N)
            else:
                # 点级mask策略（原有方式）
                mask = masker.generate_mask(batch['curvature'], valid_mask=valid_mask).squeeze(-1).bool()
                mask = mask & valid_mask  # never mask padding
                blocks_list = None
                block_labels_list = None

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
                        batch['coordinate'],     # coordinates visible as input for spatial context
                        masked_principal,
                        masked_curv,
                        masked_dens,
                        masked_normals,
                        masked_lin,
                        task='recon'  # outputs: [curvature, x, y, z] for masked points
                    )
                    if not torch.isfinite(recon).all():
                        raise RuntimeError("Non-finite values in model output 'recon'")
                    if use_block_mask:
                        loss = block_loss_fn(recon, batch, masks_list, blocks_list, block_labels_list)
                    else:
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
                    batch['coordinate'],  # coordinates visible as input
                    masked_principal,
                    masked_curv,
                    masked_dens,
                    masked_normals,
                    masked_lin,
                    task='recon'  # outputs: [curvature, x, y, z]
                )
                if not torch.isfinite(recon).all():
                    raise RuntimeError("Non-finite values in model output 'recon' (no AMP)")
                if use_block_mask:
                    loss = block_loss_fn(recon, batch, masks_list, blocks_list, block_labels_list)
                else:
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
        if use_block_mask:
            val_loss = validate_block(model, val_loader, device, segmenter, block_masker, epoch)
        else:
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
    Reconstruction loss for pretraining (curvature-only).

    Current recon head outputs one channel (B, N, 1), so this criterion is aligned
    to curvature-only target to avoid target-head mismatch.
    """
    # recon: (B, N, 1) -> [curvature_target]
    if recon.shape[-1] != 1:
        raise ValueError(f"Expected recon last dim = 1 for curvature-only pretrain, got {recon.shape[-1]}")

    # Normalize curvature target
    curv = batch['curvature']
    curv_mode = str(getattr(GlobalConfig, "PRETRAIN_CURV_TARGET", "log")).lower().strip()
    if curv_mode == "log":
        eps = float(getattr(GlobalConfig, "PRETRAIN_CURV_EPS", 1e-6))
        gt = torch.log(curv.clamp_min(0) + eps)
    else:
        gt = curv

    # Masked regression on curvature only
    m = mask.unsqueeze(-1).float()  # (B, N, 1)
    num_masked = mask.sum().clamp(min=1).float()
    diff = (recon - gt) * m

    train_mode = str(getattr(GlobalConfig, "PRETRAIN_MODE", "self_supervised")).lower()

    # Use first recon weight for curvature; keep backward compatibility with existing config.
    recon_weights = getattr(GlobalConfig, "PRETRAIN_RECON_WEIGHTS", [2.0])
    base_w = float(recon_weights[0]) if len(recon_weights) > 0 else 1.0
    weight = torch.tensor(base_w, device=recon.device, dtype=recon.dtype)

    if train_mode == "supervised":
        per_ch_mse = (diff ** 2).sum() / num_masked
        loss = per_ch_mse * weight
    elif train_mode == "self_supervised":
        per_ch_mse = (diff ** 2).sum() / num_masked
        use_norm_loss = bool(getattr(GlobalConfig, "PRETRAIN_USE_NORM_LOSS", True))
        if use_norm_loss:
            rms = per_ch_mse.sqrt().clamp_min(1e-8)
            loss = per_ch_mse * (weight / rms)
        else:
            loss = per_ch_mse * weight
    else:
        raise ValueError(f"Unknown PRETRAIN_MODE: {train_mode}. Use 'self_supervised' or 'supervised'")

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

@torch.no_grad()
def validate_block(model, loader, device, segmenter, block_masker, epoch):
    """块级mask的验证函数"""
    model.eval()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid_mask = batch['mask'].bool()
        
        # 对每个样本进行块分割
        blocks_list = []
        block_labels_list = []
        masks_list = []
        
        for b in range(batch['features'].shape[0]):
            # 获取有效点
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1).cpu().numpy()
            if len(valid_indices) == 0:
                blocks_list.append([])
                block_labels_list.append([])
                masks_list.append(torch.zeros(batch['features'].shape[1], dtype=torch.bool, device=device))
                continue
            
            # 提取有效点的数据
            points = batch['coordinate'][b][valid_indices].cpu().numpy()
            curvature = batch['curvature'][b][valid_indices].cpu().numpy()
            
            # 块分割
            blocks, block_labels = segmenter.segment(points, curvature)
            
            # 转换回原始索引
            blocks_original = [[int(valid_indices[idx]) for idx in block] for block in blocks]
            
            # 生成块级mask
            mask = block_masker.generate_mask(blocks_original, block_labels, epoch=epoch)
            mask = mask.to(device)
            # 确保只mask有效点
            mask = mask & valid_mask[b]
            
            blocks_list.append(blocks_original)
            block_labels_list.append(block_labels)
            masks_list.append(mask)
        
        # 合并mask
        mask = torch.stack(masks_list)  # (B, N)
        
        # Mask输入
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
        vloss = block_loss_fn(recon, batch, masks_list, blocks_list, block_labels_list)
        total += vloss.item() * batch['features'].size(0)
    return total / len(loader.dataset)

if __name__ == "__main__":
    from utils.config import Config
    run_pretrain(Config)