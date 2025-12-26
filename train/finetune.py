# finetune.py
import os, torch, numpy as np, torch.nn as nn
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from model.model import GeometryAwareTransformer
from utils.metric_utils import calculate_metrics
from utils.config import Config as GlobalConfig

class WeldDataset(Dataset):
    """从CSV文件加载带标签的点云数据"""
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # 读取CSV文件
        df = pd.read_csv(self.file_list[idx])

        # 提取8维特征: [x, y, z, nx, ny, nz, curvature, density]
        feat = df[['x', 'y', 'z', 'nx', 'ny', 'nz', 'curvature', 'density']].values.astype(np.float32)

        # 提取各个几何特征
        normals = df[['nx', 'ny', 'nz']].values.astype(np.float32)
        curvature = df[['curvature']].values.astype(np.float32)
        local_density = df[['density']].values.astype(np.float32)

        # 主方向和线性度
        if 'principal_dir_x' in df.columns:
            principal_dir = df[['principal_dir_x', 'principal_dir_y', 'principal_dir_z']].values.astype(np.float32)
        else:
            principal_dir = np.zeros_like(normals)

        if 'linearity' in df.columns:
            linearity = df[['linearity']].values.astype(np.float32)
        else:
            linearity = np.zeros((feat.shape[0], 1), dtype=np.float32)

        # 标签 (finetune需要)
        if 'label' in df.columns:
            labels = df[['label']].values.astype(np.float32)
        else:
            labels = -1.0 * np.ones((feat.shape[0], 1), dtype=np.float32)

        return {
            'features': feat, 'normals': normals, 'curvature': curvature,
            'local_density': local_density, 'principal_dir': principal_dir,
            'linearity': linearity, 'labels': labels
        }

def collate_fn(batch):
    global_max = getattr(GlobalConfig, 'MAX_POINTS', None)
    max_pts = max(item['features'].shape[0] for item in batch)
    max_pts = min(max_pts, global_max) if global_max else max_pts
    def pad(arr_list, shape, pad_val=0.0):
        out = np.full(shape, pad_val, dtype=np.float32)
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
    labels = pad([b['labels'] for b in batch], (b, max_pts, 1), pad_val=-1.0)
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
        'labels': torch.from_numpy(labels),
        'mask': mask
    }

def run_finetune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_files = [os.path.join(config.LABEL_DATA_DIR, f)
                     for f in os.listdir(config.LABEL_DATA_DIR)
                     if f.endswith('.csv') and '_pred' not in f]
    train_files = labeled_files[:int(0.8 * len(labeled_files))]
    val_files = labeled_files[int(0.8 * len(labeled_files)):]
    train_loader = DataLoader(WeldDataset(train_files), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(WeldDataset(val_files), batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True)

    model = GeometryAwareTransformer(config).to(device)
    # ---------- 加载预训练权重 ----------
    pretrained = getattr(config, 'PRETRAINED_WEIGHTS', None)
    if pretrained and os.path.isfile(pretrained):
        model.load_state_dict(torch.load(pretrained, map_location=device), strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
    else:
        raise FileNotFoundError("PRETRAINED_WEIGHTS not found – required for finetune")
    # ---------- 可选重新初始化分类头 ----------
    if getattr(config, 'REINIT_CLASSIFIER_ON_FINETUNE', True):
        for m in model.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        print("Re-initialized classifier head")
    # ---------- 训练分类头和backbone ----------
    if getattr(config, 'FINETUNE_CLASSIFIER', True):
        for p in model.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True
    # ---------- 损失 & 优化器 ----------
    pos_weight = None
    if getattr(config, 'AUTO_POS_WEIGHT', True):
        pos, neg = 0, 0
        for p in labeled_files:
            df = pd.read_csv(p)
            if 'label' in df.columns:
                labs = df['label'].values.reshape(-1)
                valid = labs >= 0
                pos += int(((labs == 1) & valid).sum())
                neg += int(((labs == 0) & valid).sum())
        if pos > 0:
            # clamp to avoid extremely large weights
            raw = float(neg) / (float(pos) + 1e-8)
            raw = max(raw, 1.0)           # at least 1.0
            raw = min(raw, 10.0)          # clamp upper bound (tunable)
            pos_weight = torch.tensor([raw], device=device)
            print(f"AUTO_POS_WEIGHT (clamped) = {pos_weight.item():.4f}")
        else:
            print("No positive samples found in labeled data; skipping pos_weight")

    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=1e-5,
                                  weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    writer = SummaryWriter(os.path.join(getattr(config, 'LOG_DIR', 'logs'), 'finetune'))
    best_f1 = 0.0
    os.makedirs(config.WEIGHTS_SAVE_DIR, exist_ok=True)

    accum_steps = getattr(config, 'ACCUM_STEPS', 1)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss_sum = 0.0
        total_labeled_pts = 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"FT Epoch {epoch+1}")):

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            # ===== 曲率增强 =====
            CURV_GAIN = 4.0   # 可调 3~6
            features = batch["features"].clone()
            features[:, :, 6] = features[:, :, 6] * CURV_GAIN   # 第6维是曲率

            logits = model(
                features, batch['principal_dir'], batch['curvature'],
                batch['local_density'], batch['normals'], batch['linearity'],
                task='class'
            )  # (B,N,1)


            labels = batch['labels']          # (B,N,1)
            labeled_mask = (labels >= 0)

            if labeled_mask.any():

                loss_per_point = criterion(logits.squeeze(-1), labels.squeeze(-1))  # (B,N)
                mask2d = labeled_mask.squeeze(-1)

                loss_sum_batch = loss_per_point[mask2d].sum()
                n_labeled = int(mask2d.sum().item())

                loss_for_backward = loss_sum_batch / max(1, n_labeled)

                if labeled_mask.any():
                    loss_per_point = criterion(logits.squeeze(-1), labels.squeeze(-1))  # (B,N)
                    mask2d = labeled_mask.squeeze(-1)

                    loss_sum_batch = loss_per_point[mask2d].sum()
                    n_labeled = int(mask2d.sum().item())

                    # ===== 曲率引导辅助 loss =====
                    curv_norm = batch["curvature"].squeeze(-1) / (batch["curvature"].max() + 1e-6)
                    pred_sigmoid = torch.sigmoid(logits.squeeze(-1))
                    # 对高曲率点的错误预测增加 penalty
                    loss_aux = ((1 - pred_sigmoid)**2 * curv_norm).mean() * 0.3  # 0.2~0.4 可调
                    loss_for_backward = (loss_sum_batch / max(1, n_labeled)) + loss_aux

                else:
                    loss_for_backward = torch.tensor(0.0, device=device, requires_grad=True)
                    loss_sum_batch = torch.tensor(0.0)
                    n_labeled = 0

            else:
                loss_for_backward = torch.tensor(0.0, device=device, requires_grad=True)
                loss_sum_batch = torch.tensor(0.0)
                n_labeled = 0

            loss_for_backward = loss_for_backward / accum_steps
            loss_for_backward.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss_sum += float(loss_sum_batch.item())
            total_labeled_pts += n_labeled

        # --- Epoch training loss ---
        train_loss = (total_loss_sum / total_labeled_pts) if total_labeled_pts > 0 else 0.0

        # --- Validation ---
        val_loss, val_metrics = validate(model, val_loader, device, criterion)

        writer.add_scalar('loss/train', train_loss, epoch+1)
        writer.add_scalar('loss/val', val_loss, epoch+1)
        writer.add_scalar('metrics/val_f1', val_metrics["f1_score"], epoch+1)

        print(f"Epoch {epoch+1} | Train {train_loss:.6f} | Val {val_loss:.6f} | F1 {val_metrics['f1_score']:.4f}")

        if val_metrics["f1_score"] > best_f1:
            best_f1 = val_metrics["f1_score"]
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, "best_finetune.pth"))
            print("Saved best_finetune.pth")

        scheduler.step()

@torch.no_grad()
def validate(model, loader, device, criterion):
    model.eval()
    total_loss_sum = 0.0
    total_labeled_pts = 0
    all_preds, all_labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(batch['features'], batch['principal_dir'], batch['curvature'],
                       batch['local_density'], batch['normals'], batch['linearity'], task='class')
        labels = batch['labels']
        labeled_mask = (labels >= 0)
        if labeled_mask.any():
            loss_per_point = criterion(logits.squeeze(-1), labels.squeeze(-1))
            labeled_mask2d = labeled_mask.squeeze(-1)
            loss_sum_batch = loss_per_point[labeled_mask2d].sum().item()
            n_labeled = int(labeled_mask2d.sum().item())
        else:
            loss_sum_batch = 0.0
            n_labeled = 0

        total_loss_sum += loss_sum_batch
        total_labeled_pts += n_labeled

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labs = labels.cpu().numpy().reshape(-1)
        mask = labeled_mask.cpu().numpy().reshape(-1)
        if mask.any():
            all_preds.append(probs[mask])
            all_labels.append(labs[mask])

    val_loss = (total_loss_sum / total_labeled_pts) if total_labeled_pts > 0 else 0.0

    if len(all_preds):
        all_preds = np.concatenate(all_preds) > 0.5
        all_labels = np.concatenate(all_labels)
        metrics = calculate_metrics(all_preds.reshape(-1, 1), all_labels.reshape(-1, 1))
    else:
        metrics = {'f1_score': 0.0, 'accuracy': 0.0}
    return val_loss, metrics

if __name__ == "__main__":
    from utils.config import Config
    run_finetune(Config)