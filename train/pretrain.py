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
    def __init__(self, file_list):
        self.file_list = file_list
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
       return load_features_from_csv(self.file_list[idx])

def collate_fn(batch):
    global_max = getattr(GlobalConfig, 'PRETRAIN_MAX_POINTS', None) or getattr(GlobalConfig, 'MAX_POINTS', None)
    subsample_method = getattr(GlobalConfig, 'SUBSAMPLE_METHOD', 'random')
    max_pts = max(item['features'].shape[0] for item in batch)
    max_pts = min(max_pts, global_max) if global_max else max_pts
    def subsample(item, n_keep):
        n = item['features'].shape[0]
        if n <= n_keep:
            return item
        if subsample_method == 'fps':
            idx = fps(item['coordinate'], n_keep)
        else:
            idx = np.random.choice(n, size=n_keep, replace=False)
        out = {}
        for k, v in item.items():
            out[k] = None if v is None else v[idx]
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
    pretrain_dir = getattr(config, 'PRETRAIN_DATA_DIR', config.PROCESSED_DATA_DIR)

    pretrain_train_dir = os.path.join(pretrain_dir, 'train')
    pretrain_val_dir = os.path.join(pretrain_dir, 'validation')
    if os.path.isdir(pretrain_train_dir):
        train_files = sorted([
            os.path.join(pretrain_train_dir, f)
            for f in os.listdir(pretrain_train_dir)
            if f.endswith('.csv') and '_pred' not in f
        ])
        if os.path.isdir(pretrain_val_dir):
            val_files = sorted([
                os.path.join(pretrain_val_dir, f)
                for f in os.listdir(pretrain_val_dir)
                if f.endswith('.csv') and '_pred' not in f
            ])
        else:
            split_idx = max(1, int(0.9 * len(train_files)))
            val_files = train_files[split_idx:]
            train_files = train_files[:split_idx]
        print(f"[Pretrain pre-split] train={len(train_files)} val={len(val_files)} dir={pretrain_dir}")
    elif use_fold_split:
        fold_id = getattr(config, 'FOLD_ID', 'fold_1')
        strict = bool(getattr(config, 'FOLD_SPLIT_STRICT', True))
        include_labeled_in_pretrain = bool(getattr(config, 'PRETRAIN_INCLUDE_LABELED', False))
        train_files = collect_files_by_fold(
            mode='pretrain', fold_id=fold_id, data_dir=pretrain_dir,
            include_labeled_in_pretrain=include_labeled_in_pretrain, strict=strict, print_stats=True,
        )
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
    use_block_mask = getattr(config, 'USE_BLOCK_MASK', False)
    if use_block_mask:
        segmenter = BlockSegmenter(
            target_points_per_block=getattr(config, 'TARGET_POINTS_PER_BLOCK', 1000),
            high_curv_threshold=getattr(config, 'HIGH_CURV_THRESHOLD', 0.01),
            min_high_curv_points=getattr(config, 'MIN_HIGH_CURV_POINTS', 5),
            align_grid=True,
            grid_align_base=getattr(config, 'GRID_ALIGN_BASE', 0.001)
        )
        block_masker = BlockMasker(
            mask_ratio=getattr(config, 'BLOCK_MASK_RATIO', 0.3),
            strategy=getattr(config, 'BLOCK_MASK_STRATEGY', 'mixed'),
            weld_mask_ratio=getattr(config, 'WELD_MASK_RATIO', None),
            bg_mask_ratio=getattr(config, 'BG_MASK_RATIO', None)
        )
        masker = None
    else:
        mask_type = getattr(config, 'MASK_TYPE', 'random').lower()
        masker = RandomMasker(mask_ratio=config.MASK_RATIO) if mask_type == 'random' else HighCurvatureMasker(mask_ratio=config.MASK_RATIO)
        segmenter = None
        block_masker = None

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = max(1, int(getattr(config, 'NUM_EPOCHS', 1)) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    use_amp = bool(getattr(config, 'USE_AMP', False)) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    writer = SummaryWriter(os.path.join(getattr(config, 'LOG_DIR', 'logs'), 'pretrain'))
    best_loss = float('inf')
    os.makedirs(config.WEIGHTS_SAVE_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        model.train(); total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"PT Epoch {epoch+1}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            valid_mask = batch['mask'].bool()
            if use_block_mask:
                masks_list = []
                blocks_list = []
                block_labels_list = []
                for b in range(batch['features'].shape[0]):
                    valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1).cpu().numpy()
                    if len(valid_indices) == 0:
                        blocks_list.append([]); block_labels_list.append([])
                        masks_list.append(torch.zeros(batch['features'].shape[1], dtype=torch.bool, device=device)); continue
                    points = batch['coordinate'][b][valid_indices].cpu().numpy()
                    curvature = batch['curvature'][b][valid_indices].cpu().numpy()
                    blocks, block_labels = segmenter.segment(points, curvature)
                    blocks_original = [[int(valid_indices[idx]) for idx in block] for block in blocks]
                    mask = block_masker.generate_mask(blocks_original, block_labels, epoch=epoch).to(device)
                    masks_list.append(mask & valid_mask[b]); blocks_list.append(blocks_original); block_labels_list.append(block_labels)
                mask = torch.stack(masks_list)
            else:
                mask = masker.generate_mask(batch['curvature'], valid_mask=valid_mask).squeeze(-1).bool() & valid_mask
                blocks_list = block_labels_list = None
            masked_feat = batch['features'].clone(); masked_feat[mask] = 0.0
            masked_curv = batch['curvature'].clone(); masked_curv[mask] = 0.0
            masked_dens = batch['local_density'].clone(); masked_dens[mask] = 0.0
            masked_lin = batch['linearity'].clone(); masked_lin[mask] = 0.0
            masked_normals = batch['normals'].clone(); masked_normals[mask] = 0.0
            masked_principal = batch['principal_dir'].clone(); masked_principal[mask] = 0.0
            if scaler:
                with torch.amp.autocast('cuda'):
                    recon = model(masked_feat, batch['coordinate'], masked_principal, masked_curv, masked_dens, masked_normals, masked_lin, task='recon')
                    loss = block_loss_fn(recon, batch, masks_list, blocks_list, block_labels_list) if use_block_mask else recon_criterion(recon, batch, mask)
                scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step()
            else:
                recon = model(masked_feat, batch['coordinate'], masked_principal, masked_curv, masked_dens, masked_normals, masked_lin, task='recon')
                loss = block_loss_fn(recon, batch, masks_list, blocks_list, block_labels_list) if use_block_mask else recon_criterion(recon, batch, mask)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step(); scheduler.step()
            total_loss += float(loss.item()) * batch['features'].size(0)

        val_loss = validate_block(model, val_loader, device, segmenter, block_masker, epoch) if use_block_mask else validate(model, val_loader, device, masker)
        writer.add_scalar('loss/train', total_loss / len(train_loader.dataset), epoch+1)
        writer.add_scalar('loss/val', val_loss, epoch+1)
        print(f"Epoch {epoch+1} | Train {total_loss/len(train_loader.dataset):.6f} | Val {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, "best_pretrain.pth"))


def recon_criterion(recon, batch, mask):
    curv = batch['curvature']
    curv_mode = str(getattr(GlobalConfig, "PRETRAIN_CURV_TARGET", "log")).lower().strip()
    if curv_mode == "log":
        eps = float(getattr(GlobalConfig, "PRETRAIN_CURV_EPS", 1e-6))
        gt = torch.log(curv.clamp_min(0) + eps)
    else:
        gt = curv
    m = mask.unsqueeze(-1).float()
    num_masked = mask.sum().clamp(min=1).float()
    diff = (recon - gt) * m
    recon_weights = getattr(GlobalConfig, "PRETRAIN_RECON_WEIGHTS", [2.0])
    base_w = float(recon_weights[0]) if len(recon_weights) > 0 else 1.0
    per_ch_mse = (diff ** 2).sum() / num_masked
    return per_ch_mse * torch.tensor(base_w, device=recon.device, dtype=recon.dtype)

@torch.no_grad()
def validate(model, loader, device, masker):
    model.eval(); total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid_mask = batch['mask'].bool()
        mask = masker.generate_mask(batch['curvature'], valid_mask=valid_mask).squeeze(-1).bool() & valid_mask
        masked_feat = batch['features'].clone(); masked_feat[mask] = 0.0
        masked_curv = batch['curvature'].clone(); masked_curv[mask] = 0.0
        masked_dens = batch['local_density'].clone(); masked_dens[mask] = 0.0
        masked_lin = batch['linearity'].clone(); masked_lin[mask] = 0.0
        masked_normals = batch['normals'].clone(); masked_normals[mask] = 0.0
        masked_principal = batch['principal_dir'].clone(); masked_principal[mask] = 0.0
        recon = model(masked_feat, batch['coordinate'], masked_principal, masked_curv, masked_dens, masked_normals, masked_lin, task='recon')
        total += recon_criterion(recon, batch, mask).item() * batch['features'].size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def validate_block(model, loader, device, segmenter, block_masker, epoch):
    model.eval(); total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid_mask = batch['mask'].bool()
        blocks_list = []; block_labels_list = []; masks_list = []
        for b in range(batch['features'].shape[0]):
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1).cpu().numpy()
            if len(valid_indices) == 0:
                blocks_list.append([]); block_labels_list.append([])
                masks_list.append(torch.zeros(batch['features'].shape[1], dtype=torch.bool, device=device)); continue
            points = batch['coordinate'][b][valid_indices].cpu().numpy()
            curvature = batch['curvature'][b][valid_indices].cpu().numpy()
            blocks, block_labels = segmenter.segment(points, curvature)
            blocks_original = [[int(valid_indices[idx]) for idx in block] for block in blocks]
            mask = block_masker.generate_mask(blocks_original, block_labels, epoch=epoch).to(device) & valid_mask[b]
            blocks_list.append(blocks_original); block_labels_list.append(block_labels); masks_list.append(mask)
        mask = torch.stack(masks_list)
        masked_feat = batch['features'].clone(); masked_feat[mask] = 0.0
        masked_curv = batch['curvature'].clone(); masked_curv[mask] = 0.0
        masked_dens = batch['local_density'].clone(); masked_dens[mask] = 0.0
        masked_lin = batch['linearity'].clone(); masked_lin[mask] = 0.0
        masked_normals = batch['normals'].clone(); masked_normals[mask] = 0.0
        masked_principal = batch['principal_dir'].clone(); masked_principal[mask] = 0.0
        recon = model(masked_feat, batch['coordinate'], masked_principal, masked_curv, masked_dens, masked_normals, masked_lin, task='recon')
        total += block_loss_fn(recon, batch, masks_list, blocks_list, block_labels_list).item() * batch['features'].size(0)
    return total / len(loader.dataset)

if __name__ == "__main__":
    from utils.config import Config
    run_pretrain(Config)
