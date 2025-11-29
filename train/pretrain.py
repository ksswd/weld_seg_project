# pretrain.py
import os, torch, numpy as np, torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model import GeometryAwareTransformer
from train.mask_strategy import HighCurvatureMasker
from utils.config import Config as GlobalConfig

class WeldDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        data = np.load(self.file_list[idx])
        feat = data['features'].astype(np.float32)
        if feat.ndim == 1:
            feat = feat[None, :]
        normals = data['normals'].astype(np.float32) if 'normals' in data else feat[:, 3:6]
        curvature = data['curvature'].astype(np.float32) if 'curvature' in data else feat[:, 6:7]
        local_density = data['local_density'].astype(np.float32) if 'local_density' in data else feat[:, 7:8]
        principal_dir = data['principal_dir'].astype(np.float32) if 'principal_dir' in data else np.zeros_like(normals)
        linearity = data['linearity'].astype(np.float32) if 'linearity' in data else np.zeros((feat.shape[0], 1), dtype=np.float32)
        return {
            'features': feat, 'normals': normals, 'curvature': curvature,
            'local_density': local_density, 'principal_dir': principal_dir, 'linearity': linearity
        }

def collate_fn(batch):
    global_max = getattr(GlobalConfig, 'MAX_POINTS', None)
    max_pts = max(item['features'].shape[0] for item in batch)
    max_pts = min(max_pts, global_max) if global_max else max_pts
    def pad(arr_list, shape):
        out = np.full(shape, 0.0, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            n = min(arr.shape[0], shape[1])
            out[i, :n] = arr[:n]
        return out
    b, c = len(batch), batch[0]['features'].shape[1]
    feats = pad([b['features'] for b in batch], (b, max_pts, c))
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
                 if f.endswith('.npz') and '_pred' not in f]
    train_files = all_files[:int(0.8 * len(all_files))]
    val_files = all_files[int(0.8 * len(all_files)):]
    train_loader = DataLoader(WeldDataset(train_files), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(WeldDataset(val_files), batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = GeometryAwareTransformer(config).to(device)
    masker = HighCurvatureMasker(mask_ratio=config.MASK_RATIO)
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler() if getattr(config, 'USE_AMP', False) and device.type == 'cuda' else None
    writer = SummaryWriter(os.path.join(getattr(config, 'LOG_DIR', 'logs'), 'pretrain'))
    best_loss = float('inf')
    os.makedirs(config.WEIGHTS_SAVE_DIR, exist_ok=True)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"PT Epoch {epoch+1}"):
            # 转到设备
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # 检查输入是否有 NaN/Inf（排查来源）
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and not torch.isfinite(v).all():
                    raise RuntimeError(f"Non-finite values found in batch tensor '{k}'")

            # 1) 清零梯度（必须）
            optimizer.zero_grad(set_to_none=True)

            # 2) 生成 mask & masked input
            mask = masker.generate_mask(batch['curvature']).squeeze(-1).bool()
            masked_feat = batch['features'].clone()
            mask_xyz = mask.unsqueeze(-1).expand_as(masked_feat[..., :3])
            masked_feat[..., :3][mask_xyz] = 0.0

            # 3) 前向 + 反向（考虑 AMP）
            if scaler:
                with torch.cuda.amp.autocast():
                    recon = model(masked_feat, batch['principal_dir'], batch['curvature'],
                                  batch['local_density'], batch['normals'], batch['linearity'], task='recon')
                    if not torch.isfinite(recon).all():
                        raise RuntimeError("Non-finite values in model output 'recon'")
                    loss = recon_criterion(criterion, recon, batch, mask)

                # 如果 loss 本身是 NaN/Inf，保护性处理
                if not torch.isfinite(loss):
                    print("Warning: non-finite loss detected. replacing with zero-loss to avoid crash.")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)

                scaler.scale(loss).backward()

                # 在 unscale 之后裁剪梯度
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # step optimizer & scaler
                scaler.step(optimizer)
                scaler.update()
            else:
                recon = model(masked_feat, batch['principal_dir'], batch['curvature'],
                              batch['local_density'], batch['normals'], batch['linearity'], task='recon')
                if not torch.isfinite(recon).all():
                    raise RuntimeError("Non-finite values in model output 'recon' (no AMP)")
                loss = recon_criterion(criterion, recon, batch, mask)
                if not torch.isfinite(loss):
                    print("Warning: non-finite loss detected (no AMP). setting to zero.")
                    loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += float(loss.item()) * batch['features'].size(0)

        # 验证、记录、保存与 lr scheduler（与你原逻辑一致）
        val_loss = validate(model, val_loader, device, masker, criterion)
        writer.add_scalar('loss/train', total_loss / len(train_loader.dataset), epoch+1)
        writer.add_scalar('loss/val', val_loss, epoch+1)
        print(f"Epoch {epoch+1} | Train {total_loss/len(train_loader.dataset):.6f} | Val {val_loss:.6f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, "best_pretrain.pth"))
            print("Saved best_pretrain.pth")
        scheduler.step()

def recon_criterion(criterion, recon, batch, mask, gt_coords_flat=None):
    """
    Weighted reconstruction loss.
    We reconstruct all 8 dimensions:
    [x,y,z, nx,ny,nz, curvature, density]
    """

    B, N, C = recon.shape
    # flatten
    recon_flat = recon.view(-1, C)
    feat_flat  = batch['features'].view(-1, C)
    mask_flat  = mask.view(-1)

    # only masked positions
    pred = recon_flat[mask_flat]   # [num_mask, C]
    gt   = feat_flat[mask_flat]    # [num_mask, C]

    if pred.numel() == 0:
        return recon.sum() * 0.0  # no mask

    # ====== 你的权重（device-aware） ======
    RECON_WEIGHTS = torch.tensor(
        [0.12,0.12,0.12,    # xyz
         0.06,0.06,0.06,    # normals
         0.40,             # curvature
         0.06],            # density
        device=pred.device
    ).view(1, -1)  # shape [1,8]

    # weighted mse: (pred - gt)^2 * weight
    loss = ((pred - gt) ** 2 * RECON_WEIGHTS).sum(-1).mean()

    # safety clamp
    if not torch.isfinite(loss):
        loss = recon.sum() * 0.0
    else:
        loss = torch.clamp(loss, max=1e4)

    return loss


@torch.no_grad()
def validate(model, loader, device, masker, criterion):
    model.eval()
    total = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        mask = masker.generate_mask(batch['curvature']).squeeze(-1).bool()
        masked_feat = batch['features'].clone()
        mask_xyz = mask.unsqueeze(-1).expand_as(masked_feat[..., :3])
        masked_feat[..., :3][mask_xyz] = 0.0
        recon = model(masked_feat, batch['principal_dir'], batch['curvature'],
                      batch['local_density'], batch['normals'], batch['linearity'], task='recon')
        vloss = recon_criterion(criterion, recon, batch, mask)
        total += vloss.item() * batch['features'].size(0)
    return total / len(loader.dataset)

if __name__ == "__main__":
    from utils.config import Config
    run_pretrain(Config)