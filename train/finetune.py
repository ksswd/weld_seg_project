# finetune.py
import os, torch, numpy as np, torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model.model import GeometryAwareTransformer
from utils.metric_utils import calculate_metrics
from utils.config import Config as GlobalConfig
from utils.downsampling import fps_with_cache as fps

class WeldDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        filepath = self.file_list[idx]
        data = np.load(filepath)
        feat = data['features'].astype(np.float32)
        if feat.ndim == 1:
            feat = feat[None, :]
        normals = data['normals'].astype(np.float32) #if 'normals' in data else feat[:, 3:6]
        curvature = data['curvature'].astype(np.float32) #if 'curvature' in data else feat[:, 6:7]
        local_density = data['local_density'].astype(np.float32) #if 'local_density' in data else feat[:, 7:8]
        principal_dir = data['principal_dir'].astype(np.float32) #if 'principal_dir' in data else np.zeros_like(normals)
        linearity = data['linearity'].astype(np.float32) #if 'linearity' in data else np.zeros((feat.shape[0], 1), dtype=np.float32)
        labels = data['labels'].astype(np.float32) #if 'labels' in data else -1.0 * np.ones((feat.shape[0], 1), dtype=np.float32)
        if labels.ndim == 1:
            labels = labels[:, None]
        return {
            'features': feat, 'normals': normals, 'curvature': curvature,
            'local_density': local_density, 'principal_dir': principal_dir,
            'linearity': linearity, 'labels': labels
        }

def collate_fn(batch):
    M = GlobalConfig.MAX_POINTS
    # 准备列表
    feats_list = []
    normals_list = []
    principal_list = []
    curvature_list = []
    density_list = []
    linearity_list = []
    labels_list = []
    for item in batch:
        # ---- 1) FPS（只用 xyz = feats[:, :3]）----
        idx = fps(item['features'][:, :3], M)

        # ---- 2) 同步采样所有 field ----
        feats = item['features'][idx]
        normals = item['normals'][idx]
        principal = item['principal_dir'][idx]
        curvature = item['curvature'][idx]
        density = item['local_density'][idx]
        linearity = item['linearity'][idx]
        labels = item['labels'][idx]

        # ---- 3) 放入 batch 列表 ----
        feats_list.append(feats)
        normals_list.append(normals)
        principal_list.append(principal)
        curvature_list.append(curvature)
        density_list.append(density)
        linearity_list.append(linearity)
        labels_list.append(labels)

    return {
        "features": torch.from_numpy(np.stack(feats_list, axis=0)),
        "normals": torch.from_numpy(np.stack(normals_list, axis=0)),
        "principal_dir": torch.from_numpy(np.stack(principal_list, axis=0)),
        "curvature": torch.from_numpy(np.stack(curvature_list, axis=0)),
        "local_density": torch.from_numpy(np.stack(density_list, axis=0)),
        "linearity": torch.from_numpy(np.stack(linearity_list, axis=0)),
        "labels": torch.from_numpy(np.stack(labels_list, axis=0)),
    }

def run_finetune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_files = [os.path.join(config.LABEL_DATA_DIR, f)
                     for f in os.listdir(config.LABEL_DATA_DIR)
                     if f.endswith('.npz') and '_pred' not in f and not f.startswith('T1') and not f.startswith('T2') and not f.startswith('T3')]
    train_files = labeled_files[:int(0.8 * len(labeled_files))]
    val_files = labeled_files[int(0.8 * len(labeled_files)):]
    # num_workers 设置为默认值 0 以避免多进程下_FPS_CACHE不共享的问题
    train_loader = DataLoader(WeldDataset(train_files), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(WeldDataset(val_files), batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, pin_memory=True)

    model = GeometryAwareTransformer(config).to(device)
    # ---------- 加载预训练权重 ----------
    pretrained = getattr(config, 'PRETRAINED_WEIGHTS', None)
    if pretrained and os.path.isfile(pretrained):
        model.load_state_dict(torch.load(pretrained, map_location=device), strict=False)
        print(f"Loaded pretrained weights from {pretrained}")
    else:
        raise FileNotFoundError("PRETRAINED_WEIGHTS not found – required for finetune")

    # ---------- FREEZE BACKBONE (only train classifier head) ----------
    freeze_backbone = getattr(config, 'FREEZE_BACKBONE', True)
    if freeze_backbone:
        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad = False
        # Unfreeze only classifier head
        for p in model.classifier.parameters():
            p.requires_grad = True
        print("Froze backbone - only training classifier head")
    else:
        # Train all parameters
        for p in model.parameters():
            p.requires_grad = True
        print("Training full model (backbone + classifier)")

    # Use BCEWithLogitsLoss for binary classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_f1 = 0.0

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        train_step = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            logits = model(
                batch['features'], batch['principal_dir'], batch['curvature'],
                batch['local_density'], batch['normals'], batch['linearity'],
                task='class'
            )  # (B,N,1)
            labels = batch['labels']          # (B,N,1)
            loss_per_point = criterion(logits.squeeze(-1), labels.squeeze(-1))  # (B,N)
            loss_sum_batch = loss_per_point.sum()
            n_labeled = int(labels.numel())
            # ===== 曲率引导辅助 loss =====
            curv_norm = batch["curvature"].squeeze(-1) / (batch["curvature"].max() + 1e-6)
            pred_sigmoid = torch.sigmoid(logits.squeeze(-1))
            # 对高曲率点的错误预测增加 penalty
            loss_aux = ((1 - pred_sigmoid)**2 * curv_norm).mean() * 0.3  # 0.2~0.4 可调
            loss_for_backward = (loss_sum_batch / max(1, n_labeled)) + loss_aux

            loss_for_backward = loss_for_backward / accum_steps
            loss_for_backward.backward()

            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                train_step += 1

        avg_train_loss = total_loss / train_step if train_step > 0 else 0.0
        val_loss, metrics = validate(model, val_loader, device, criterion)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f}, Val Loss: {val_loss:.6f}, F1: {metrics['f1_score']:.4f}, Best Threshold: {metrics['best_threshold']:.4f}")
        if metrics['f1_score'] > best_f1:
            best_f1 = metrics['f1_score']
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, 'best_finetune.pth'))
            print(f"Save best model with F1: {best_f1:.4f}")

        scheduler.step()

    print("Finetuning complete.")

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
        loss_per_point = criterion(logits.squeeze(-1), labels.squeeze(-1))
        loss_sum_batch = loss_per_point.sum().item()
        n_labeled = int(labels.numel())

        total_loss_sum += loss_sum_batch
        total_labeled_pts += n_labeled

        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        labs = labels.cpu().numpy().reshape(-1)
        all_preds.append(probs)
        all_labels.append(labs)

    val_loss = (total_loss_sum / total_labeled_pts) if total_labeled_pts > 0 else 0.0
    if len(all_preds):
        all_probs = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        # print(all_labels)
        best_f1 = 0.0
        best_t = 0.5
        for t in np.linspace(0.01, 0.5, 50):
            preds_t = (all_probs > t).astype(np.int32)
            try:
                f1 = calculate_metrics(preds_t.reshape(-1, 1), all_labels.reshape(-1, 1))['f1_score']
            except Exception:
                f1 = 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        all_preds_bin = (all_probs > best_t)
        metrics = calculate_metrics(all_preds_bin.reshape(-1, 1), all_labels.reshape(-1, 1))
        metrics['best_threshold'] = float(best_t)
    else:
        metrics = {'f1_score': 0.0, 'accuracy': 0.0}

    return val_loss, metrics

if __name__ == "__main__":
    from utils.config import Config
    run_finetune(Config)