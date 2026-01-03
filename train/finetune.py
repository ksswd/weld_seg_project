# finetune.py
import os, torch, numpy as np, torch.nn as nn
from tqdm import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from model.model import GeometryAwareTransformer
from utils.metric_utils import calculate_metrics
from utils.config import Config as GlobalConfig
from utils.downsampling import fps_with_cache as fps
from utils.io_utils import load_features_from_csv


def collate_fn_fps(batch):
    """
    Plain FPS sampling to MAX_POINTS + padding + valid mask.
    Use this for *unbiased* validation metrics (closer to deployment distribution).
    """
    max_points = int(getattr(GlobalConfig, "MAX_POINTS", 1024))
    sampled = []
    for item in batch:
        if item.get("labels", None) is None:
            raise KeyError("This CSV has no 'label' column. Finetune expects *_label_*.csv files.")
        n = item["coordinate"].shape[0]
        k = min(max_points, n)
        idx = fps(item["coordinate"], k)
        sampled.append({
            "features": item["features"][idx],
            "coordinate": item["coordinate"][idx],
            "normals": item["normals"][idx],
            "principal_dir": item["principal_dir"][idx],
            "curvature": item["curvature"][idx],
            "local_density": item["local_density"][idx],
            "linearity": item["linearity"][idx],
            "labels": item["labels"][idx],
        })

    b = len(sampled)
    max_n = max(s["features"].shape[0] for s in sampled)
    max_n = min(max_n, max_points)

    def pad(arr_list, shape):
        out = np.full(shape, 0.0, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            n = min(arr.shape[0], shape[1])
            out[i, :n] = arr[:n]
        return out

    feats = pad([s["features"] for s in sampled], (b, max_n, sampled[0]["features"].shape[1]))
    coordinate = pad([s["coordinate"] for s in sampled], (b, max_n, 3))
    normals = pad([s["normals"] for s in sampled], (b, max_n, 3))
    principal = pad([s["principal_dir"] for s in sampled], (b, max_n, 3))
    curvature = pad([s["curvature"] for s in sampled], (b, max_n, 1))
    density = pad([s["local_density"] for s in sampled], (b, max_n, 1))
    linearity = pad([s["linearity"] for s in sampled], (b, max_n, 1))
    labels = pad([s["labels"] for s in sampled], (b, max_n, 1))

    mask = torch.zeros(b, max_n, dtype=torch.bool)
    for i, s in enumerate(sampled):
        mask[i, : min(s["features"].shape[0], max_n)] = True

    return {
        "features": torch.from_numpy(feats),
        "coordinate": torch.from_numpy(coordinate),
        "normals": torch.from_numpy(normals),
        "principal_dir": torch.from_numpy(principal),
        "curvature": torch.from_numpy(curvature),
        "local_density": torch.from_numpy(density),
        "linearity": torch.from_numpy(linearity),
        "labels": torch.from_numpy(labels),
        "mask": mask,
    }

class WeldDataset(Dataset):
    def __init__(self, file_list):
        self.file_list = file_list
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        return load_features_from_csv(self.file_list[idx])

def collate_fn(batch):
    """
    Positive-preserving sampler:
    - Keep (almost) all positive points (label==1)
    - Sample negatives with a cap and a target neg:pos ratio
    - Pad to batch max length and return a valid mask

    This avoids FPS wiping out rare positives and also reduces N (speeding up O(N^2) attention).
    """
    max_points = int(getattr(GlobalConfig, "MAX_POINTS", 1024))
    neg_to_pos = int(getattr(GlobalConfig, "NEG_TO_POS_RATIO", 2))
    max_neg = int(getattr(GlobalConfig, "MAX_NEG_PER_SAMPLE", 2000))

    def choose_indices(item):
        if item.get("labels", None) is None:
            raise KeyError("This CSV has no 'label' column. Finetune expects *_label_*.csv files.")
        y = item["labels"].reshape(-1)
        n = y.shape[0]
        pos_idx = np.where(y > 0.5)[0]
        neg_idx = np.where(y <= 0.5)[0]

        if pos_idx.size == 0:
            # fallback: no positives in this file; just keep up to max_points by FPS
            k = min(max_points, n)
            return fps(item["coordinate"], k)

        # keep all positives (or FPS within positives if too many)
        if pos_idx.size > max_points:
            pos_xyz = item["coordinate"][pos_idx]
            pos_keep = pos_idx[fps(pos_xyz, max_points)]
            return pos_keep

        # sample negatives with ratio/cap
        target_neg = min(neg_idx.size, max_neg, neg_to_pos * int(pos_idx.size))
        if target_neg > 0:
            neg_keep = np.random.choice(neg_idx, size=target_neg, replace=False)
            idx = np.concatenate([pos_idx, neg_keep], axis=0)
        else:
            idx = pos_idx

        # if still too many, downsample but preserve all positives
        if idx.size > max_points:
            remaining = max_points - pos_idx.size
            remaining = max(0, remaining)
            if remaining == 0:
                return pos_idx[:max_points]
            # pick remaining from negatives using FPS for coverage
            neg_pool = np.setdiff1d(idx, pos_idx, assume_unique=False)
            neg_xyz = item["coordinate"][neg_pool]
            neg_sel = neg_pool[fps(neg_xyz, min(remaining, neg_pool.size))]
            return np.concatenate([pos_idx, neg_sel], axis=0)

        return idx

    sampled = []
    for item in batch:
        idx = choose_indices(item)
        out = {
            "features": item["features"][idx],
            "coordinate": item["coordinate"][idx],
            "normals": item["normals"][idx],
            "principal_dir": item["principal_dir"][idx],
            "curvature": item["curvature"][idx],
            "local_density": item["local_density"][idx],
            "linearity": item["linearity"][idx],
            "labels": item["labels"][idx],
        }
        sampled.append(out)

    # pad to max length in this batch
    b = len(sampled)
    max_n = max(s["features"].shape[0] for s in sampled)
    max_n = min(max_n, max_points)

    def pad(arr_list, shape):
        out = np.full(shape, 0.0, dtype=np.float32)
        for i, arr in enumerate(arr_list):
            n = min(arr.shape[0], shape[1])
            out[i, :n] = arr[:n]
        return out

    feats = pad([s["features"] for s in sampled], (b, max_n, sampled[0]["features"].shape[1]))
    coordinate = pad([s["coordinate"] for s in sampled], (b, max_n, 3))
    normals = pad([s["normals"] for s in sampled], (b, max_n, 3))
    principal = pad([s["principal_dir"] for s in sampled], (b, max_n, 3))
    curvature = pad([s["curvature"] for s in sampled], (b, max_n, 1))
    density = pad([s["local_density"] for s in sampled], (b, max_n, 1))
    linearity = pad([s["linearity"] for s in sampled], (b, max_n, 1))
    labels = pad([s["labels"] for s in sampled], (b, max_n, 1))

    mask = torch.zeros(b, max_n, dtype=torch.bool)
    for i, s in enumerate(sampled):
        mask[i, : min(s["features"].shape[0], max_n)] = True

    return {
        "features": torch.from_numpy(feats),
        "coordinate": torch.from_numpy(coordinate),
        "normals": torch.from_numpy(normals),
        "principal_dir": torch.from_numpy(principal),
        "curvature": torch.from_numpy(curvature),
        "local_density": torch.from_numpy(density),
        "linearity": torch.from_numpy(linearity),
        "labels": torch.from_numpy(labels),
        "mask": mask,
    }

def run_finetune(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_files = sorted(
        [os.path.join(config.PROCESSED_DATA_DIR, f)
                     for f in os.listdir(config.PROCESSED_DATA_DIR)
         if f.endswith(".csv") and "_label_" in f]
    )

    # Hidden pitfall #1: leakage across augmentations.
    # Split by *group* (base name without _augXX) to prevent same part's augmentations in both train/val.
    import re
    groups = {}
    for p in labeled_files:
        base = os.path.basename(p)
        key = re.sub(r"_aug\\d+\\.csv$", ".csv", base)
        groups.setdefault(key, []).append(p)

    group_keys = sorted(groups.keys())
    n_val_groups = max(1, int(0.2 * len(group_keys)))
    val_group_keys = set(group_keys[-n_val_groups:])
    train_files = [p for k in group_keys if k not in val_group_keys for p in groups[k]]
    val_files = [p for k in group_keys if k in val_group_keys for p in groups[k]]
    print(f"Grouped split: {len(group_keys)} groups | train files {len(train_files)} | val files {len(val_files)}")

    # num_workers 设置为默认值 0 以避免多进程下_FPS_CACHE不共享的问题
    train_loader = DataLoader(WeldDataset(train_files), batch_size=config.BATCH_SIZE,
                              shuffle=True, collate_fn=collate_fn, pin_memory=True)
    # Validation on balanced sampling (optimistic) + FPS sampling (closer to real inference)
    val_loader_balanced = DataLoader(WeldDataset(val_files), batch_size=config.BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn, pin_memory=True)
    val_loader_fps = DataLoader(WeldDataset(val_files), batch_size=config.BATCH_SIZE,
                                shuffle=False, collate_fn=collate_fn_fps, pin_memory=True)

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

    def per_point_loss(logits: torch.Tensor, labels: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """
        logits: (B,N) float
        labels: (B,N) float in {0,1}
        keep:   (B,N) float mask in {0,1} selecting points that contribute to loss
        returns scalar loss
        """
        # Focal loss (recommended for heavy imbalance)
        if bool(getattr(config, "USE_FOCAL_LOSS", False)):
            gamma = float(getattr(config, "FOCAL_GAMMA", 2.0))
            # CE per point
            ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")  # (B,N)
            pt = torch.exp(-ce)  # pt = sigmoid(logit) if y=1 else 1-sigmoid(logit)
            loss = ((1.0 - pt) ** gamma) * ce
        else:
            # pos_weight BCE
            if bool(getattr(config, "USE_POS_WEIGHT", True)):
                # compute pos_weight on the kept subset
                pos = (labels * keep).sum()
                neg = keep.sum() - pos
                pw = (neg / (pos + 1e-6)).clamp(max=float(getattr(config, "POS_WEIGHT_MAX", 50.0)))
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pw, reduction="none")
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        denom = keep.sum().clamp(min=1.0)
        return (loss * keep).sum() / denom
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.FINETUNE_LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)

    best_f1 = 0.0
    model_selection = str(getattr(config, "MODEL_SELECTION", "fps")).lower().strip()

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        train_step = 0
        update_steps = 0
        accum_steps = int(getattr(config, "ACCUM_STEPS", 1))
        label_mask_ratio = float(getattr(config, "LABEL_MASK_RATIO", 0.0))  # 0.0 means use all points
        curv_aux_w = float(getattr(config, "CURV_AUX_WEIGHT", 0.0))
        optimizer.zero_grad(set_to_none=True)
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            valid_mask = batch.get("mask", None)
            if valid_mask is None:
                valid_mask = torch.ones(batch["labels"].shape[:2], device=device, dtype=torch.bool)
            valid_mask_f = valid_mask.float()

            logits = model(
                batch['features'], batch['coordinate'], batch['principal_dir'], batch['curvature'],
                batch['local_density'], batch['normals'], batch['linearity'],
                task='class'
            )  # (B,N,1)
            labels = batch['labels']          # (B,N,1)
            logits_ = logits.squeeze(-1)      # (B,N)
            labels_ = labels.squeeze(-1)      # (B,N)
            if label_mask_ratio > 0:
                keep = (torch.rand_like(labels_) > label_mask_ratio).float()
            else:
                keep = torch.ones_like(labels_)
            keep = keep * valid_mask_f
            loss_main = per_point_loss(logits_, labels_, keep)
            # ===== 曲率引导辅助 loss =====
            loss = loss_main
            if curv_aux_w > 0:
                curv_norm = batch["curvature"].squeeze(-1) / (batch["curvature"].max() + 1e-6)
                pred_sigmoid = torch.sigmoid(logits_)
                # Only encourage positives on high-curvature *positive* points to avoid flooding false positives
                loss_aux = ((1.0 - pred_sigmoid) ** 2 * curv_norm * labels_ * valid_mask_f).sum() / valid_mask_f.sum().clamp(min=1.0)
                loss = loss + curv_aux_w * loss_aux
            (loss / accum_steps).backward()

            train_step += 1
            if (train_step % accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += float(loss.item())
                update_steps += 1

        avg_train_loss = total_loss / update_steps if update_steps > 0 else 0.0
        val_loss, metrics = validate(model, val_loader_balanced, device, per_point_loss, config)
        val_loss_fps, metrics_fps = validate(model, val_loader_fps, device, per_point_loss, config)
        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | "
            f"Val(bal) Loss: {val_loss:.6f}, F1: {metrics['f1_score']:.4f}, Thr: {metrics['best_threshold']:.4f} | "
            f"Val(fps) Loss: {val_loss_fps:.6f}, F1: {metrics_fps['f1_score']:.4f}, Thr: {metrics_fps['best_threshold']:.4f}"
        )
        score = metrics_fps["f1_score"] if model_selection == "fps" else metrics["f1_score"]
        if score > best_f1:
            best_f1 = score
            torch.save(model.state_dict(), os.path.join(config.WEIGHTS_SAVE_DIR, 'best_finetune.pth'))
            print(f"Save best model with selection={model_selection} F1: {best_f1:.4f}")

        scheduler.step()

    print("Finetuning complete.")

@torch.no_grad()
def validate(model, loader, device, per_point_loss_fn, config):
    model.eval()
    total_loss_sum = 0.0
    total_labeled_pts = 0
    all_preds, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        valid_mask = batch.get("mask", None)
        if valid_mask is None:
            valid_mask = torch.ones(batch["labels"].shape[:2], device=device, dtype=torch.bool)
        valid_mask_f = valid_mask.float()
        logits = model(batch['features'], batch['coordinate'], batch['principal_dir'], batch['curvature'],
                       batch['local_density'], batch['normals'], batch['linearity'], task='class')
        labels = batch['labels']
        logits_ = logits.squeeze(-1)
        labels_ = labels.squeeze(-1)
        keep = torch.ones_like(labels_) * valid_mask_f
        loss_batch = per_point_loss_fn(logits_, labels_, keep)
        n_labeled = int(valid_mask.sum().item())

        total_loss_sum += float(loss_batch.item()) * n_labeled
        total_labeled_pts += n_labeled

        probs = torch.sigmoid(logits_).detach().cpu().numpy().reshape(-1)
        labs = labels_.detach().cpu().numpy().reshape(-1)
        vm = valid_mask.detach().cpu().numpy().reshape(-1).astype(bool)
        probs = probs[vm]
        labs = labs[vm]
        all_preds.append(probs)
        all_labels.append(labs)

    val_loss = (total_loss_sum / total_labeled_pts) if total_labeled_pts > 0 else 0.0
    if len(all_preds):
        all_probs = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        best_f1 = 0.0
        best_t = 0.5
        # Search threshold on [0,1] (the old 0.01-0.5 range can hide issues and bias results).
        for t in np.linspace(0.0, 1.0, 201):
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