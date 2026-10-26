#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import time
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from model import DGCNN_semseg_scannet
from util import IOStream, cal_loss
from weld_dataset import WeldCSVDataset, SingleWeldCSV, normalize_xyz, read_ply_xyz_label


def _init_(exp_name):
    os.makedirs('outputs', exist_ok=True)
    os.makedirs(f'outputs/{exp_name}', exist_ok=True)
    os.makedirs(f'outputs/{exp_name}/models', exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser('DGCNN Weld Semantic Segmentation')
    parser.add_argument('--exp_name', type=str, default='semseg_weld')
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--fold', type=str, default='fold_1', choices=['dataset', 'fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5', 'fold_6'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, default='cos', choices=['cos', 'step'])
    parser.add_argument('--step_size', type=int, default=20)
    parser.add_argument('--lr_decay', type=float, default=0.5)
    parser.add_argument('--num_points', type=int, default=8192)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--emb_dims', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--protocol_seed', type=int, default=20260823)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    parser.add_argument('--use_soft_label', action='store_true')
    parser.add_argument('--soft_threshold', type=float, default=0.5)

    parser.add_argument('--infer_csv', type=str, default='')
    parser.add_argument('--save_pred_csv', type=str, default='')
    parser.add_argument('--save_pred_xyz_label', type=str, default='')
    parser.add_argument('--save_pred_fullpred_csv', type=str, default='')
    parser.add_argument('--save_all_pred_xyz_label_dir', type=str, default='')
    parser.add_argument('--save_all_fullpred_dir', type=str, default='')
    parser.add_argument('--benchmark_infer_time', action='store_true')
    parser.add_argument('--timing_csv', type=str, default='')

    return parser.parse_args()


def set_seed(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(pred, target, num_classes=2, weld_class=1):
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)

    conf = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(target, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            conf[t, p] += 1

    total = conf.sum()
    oa = np.trace(conf) / float(total + 1e-8)

    class_iou = []
    class_acc = []
    for c in range(num_classes):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp
        iou = tp / float(tp + fp + fn + 1e-8)
        acc = tp / float(conf[c, :].sum() + 1e-8)
        class_iou.append(float(iou))
        class_acc.append(float(acc))

    miou = float(np.mean(class_iou))
    macc = float(np.mean(class_acc))

    tp = conf[weld_class, weld_class]
    fn = conf[weld_class, :].sum() - tp
    fp = conf[:, weld_class].sum() - tp
    weld_precision = tp / float(tp + fp + 1e-8)
    weld_recall = tp / float(tp + fn + 1e-8)
    weld_f1 = 2 * weld_precision * weld_recall / float(weld_precision + weld_recall + 1e-8)
    weld_iou = class_iou[weld_class]

    return {
        'oa': float(oa),
        'macc': float(macc),
        'miou': float(miou),
        'class_iou': class_iou,
        'class_acc': class_acc,
        'weld_precision': float(weld_precision),
        'weld_recall': float(weld_recall),
        'weld_f1': float(weld_f1),
        'weld_iou': float(weld_iou),
    }


def evaluate_model(model, loader, device, criterion, num_classes=2):
    model.eval()
    loss_sum = 0.0
    n_batch = 0
    all_pred = []
    all_gt = []

    with torch.no_grad():
        for points, target in loader:
            points = points.float().to(device)
            target = target.long().to(device)
            points = points.transpose(2, 1)

            logits, _ = model(points)  # (B, N, C)
            b, n, c = logits.shape
            logits_flat = logits.contiguous().view(-1, c)
            target_flat = target.view(-1)

            loss = criterion(logits_flat, target_flat, smoothing=False, ignore_index=255)
            loss_sum += loss.item()
            n_batch += 1

            pred_flat = torch.argmax(logits_flat, dim=1)
            all_pred.append(pred_flat.cpu().numpy())
            all_gt.append(target_flat.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_gt = np.concatenate(all_gt, axis=0)

    metrics = compute_metrics(all_pred, all_gt, num_classes=num_classes, weld_class=1)
    metrics['loss'] = float(loss_sum / max(n_batch, 1))
    return metrics


def print_metrics(io, metrics):
    io.cprint(f"oa: {metrics['oa']:.6f}")
    io.cprint(f"macc: {metrics['macc']:.6f}")
    io.cprint(f"miou: {metrics['miou']:.6f}")
    io.cprint(f"class_iou: [{metrics['class_iou'][0]:.6f}, {metrics['class_iou'][1]:.6f}]")
    io.cprint(f"class_acc: [{metrics['class_acc'][0]:.6f}, {metrics['class_acc'][1]:.6f}]")
    io.cprint(f"weld_precision: {metrics['weld_precision']:.6f}")
    io.cprint(f"weld_recall: {metrics['weld_recall']:.6f}")
    io.cprint(f"weld_f1: {metrics['weld_f1']:.6f}")
    io.cprint(f"weld_iou: {metrics['weld_iou']:.6f}")


def sync_if_cuda(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def build_model(args, device):
    model = DGCNN_semseg_scannet(num_classes=2, k=args.k, emb_dims=args.emb_dims, dropout=args.dropout).to(device)
    model = nn.DataParallel(model)
    return model


def train(args, io):
    train_set = WeldCSVDataset(
        root=args.data_root,
        split='train',
        fold=args.fold,
        num_point=args.num_points,
        normalize=True,
        augment=True,
        use_soft_label=args.use_soft_label,
        soft_threshold=args.soft_threshold,
    )
    val_set = WeldCSVDataset(
        root=args.data_root,
        split='val',
        fold=args.fold,
        num_point=args.num_points,
        normalize=True,
        augment=False,
        use_soft_label=args.use_soft_label,
        soft_threshold=args.soft_threshold,
    )
    test_set = WeldCSVDataset(
        root=args.data_root,
        split='test',
        fold=args.fold,
        num_point=args.num_points,
        normalize=True,
        augment=False,
        use_soft_label=args.use_soft_label,
        soft_threshold=args.soft_threshold,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)

    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    model = build_model(args, device)

    if args.optimizer == 'SGD':
        optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    else:
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.lr_decay)

    best_miou = -1.0
    best_path = f'outputs/{args.exp_name}/models/best_model.pth'

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n_batch = 0

        for points, target in train_loader:
            points = points.float().to(device)
            target = target.long().to(device)
            points = points.transpose(2, 1)

            optimizer.zero_grad()
            logits, _ = model(points)  # (B, N, C)
            b, n, c = logits.shape
            logits_flat = logits.contiguous().view(-1, c)
            target_flat = target.view(-1)

            loss = cal_loss(logits_flat, target_flat, smoothing=False, ignore_index=255)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            n_batch += 1

        scheduler.step()
        train_loss = loss_sum / max(n_batch, 1)

        val_metrics = evaluate_model(model, val_loader, device, cal_loss, num_classes=2)

        io.cprint(
            f"[Epoch {epoch + 1:03d}/{args.epochs:03d}] "
            f"train_loss={train_loss:.6f} val_loss={val_metrics['loss']:.6f} "
            f"val_oa={val_metrics['oa']:.6f} val_miou={val_metrics['miou']:.6f} "
            f"val_weld_f1={val_metrics['weld_f1']:.6f} time={time.time() - t0:.1f}s"
        )

        if val_metrics['miou'] > best_miou:
            best_miou = val_metrics['miou']
            torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_miou': best_miou}, best_path)
            io.cprint(f"==> Save best model to {best_path}")

    io.cprint('==== Final test with best model ====')
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    test_metrics = evaluate_model(model, test_loader, device, cal_loss, num_classes=2)
    print_metrics(io, test_metrics)


def evaluate_split(args, io):
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    model = build_model(args, device)

    if not args.model_path:
        raise ValueError('--model_path is required when --eval')

    ckpt = torch.load(args.model_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    ds = WeldCSVDataset(
        root=args.data_root,
        split=args.split,
        fold=args.fold,
        num_point=args.num_points,
        normalize=True,
        augment=False,
        use_soft_label=args.use_soft_label,
        soft_threshold=args.soft_threshold,
    )
    loader = DataLoader(ds, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
    metrics = evaluate_model(model, loader, device, cal_loss, num_classes=2)
    print_metrics(io, metrics)

    if args.save_all_pred_xyz_label_dir:
        os.makedirs(args.save_all_pred_xyz_label_dir, exist_ok=True)
    if args.save_all_fullpred_dir:
        os.makedirs(args.save_all_fullpred_dir, exist_ok=True)

    if args.save_all_pred_xyz_label_dir or args.save_all_fullpred_dir or args.benchmark_infer_time:
        model.eval()
        infer_times = []
        timing_rows = []
        total_points = 0
        for csv_path in ds.files:
            sync_if_cuda(device)
            t0 = time.perf_counter()
            xyz_full, label_full, pred_full, prob_weld = _predict_full_csv(
                model, device, csv_path, num_points=args.num_points,
                protocol_seed=args.protocol_seed)
            sync_if_cuda(device)
            infer_time = time.perf_counter() - t0
            infer_times.append(infer_time)
            total_points += xyz_full.shape[0]
            timing_rows.append({'sample': os.path.basename(csv_path),
                                'num_points': int(xyz_full.shape[0]),
                                'infer_time_sec': float(infer_time)})

            stem = os.path.splitext(os.path.basename(csv_path))[0]
            if args.save_all_pred_xyz_label_dir:
                out = np.concatenate([xyz_full, label_full.reshape(-1, 1), pred_full.reshape(-1, 1)], axis=1)
                out_path = os.path.join(args.save_all_pred_xyz_label_dir, f'{stem}_pred_xyz_label.txt')
                np.savetxt(out_path, out, fmt='%.6f %.6f %.6f %d %d')
            if args.save_all_fullpred_dir:
                out_path = os.path.join(args.save_all_fullpred_dir, f'{stem}_fullpred.csv')
                out_df = pd.DataFrame({
                    'x': xyz_full[:, 0],
                    'y': xyz_full[:, 1],
                    'z': xyz_full[:, 2],
                    'label': label_full.astype(np.int64),
                    'pred': pred_full.astype(np.int64),
                    'prob': prob_weld.astype(np.float32),
                })
                out_df.to_csv(out_path, index=False)

        if args.timing_csv:
            timing_dir = os.path.dirname(args.timing_csv)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            pd.DataFrame(timing_rows).to_csv(args.timing_csv, index=False)
            io.cprint(f'Saved inference timing to: {args.timing_csv}')

        if args.save_all_pred_xyz_label_dir:
            io.cprint(f'Saved all xyz+label predictions to: {args.save_all_pred_xyz_label_dir}')
        if args.save_all_fullpred_dir:
            io.cprint(f'Saved all fullpred csv to: {args.save_all_fullpred_dir}')
        if infer_times:
            infer_times_np = np.asarray(infer_times, dtype=np.float64)
            io.cprint(
                'Inference time :: '
                f'samples={len(infer_times)}, total_points={total_points}, '
                f'total={infer_times_np.sum():.6f}s, '
                f'avg_per_sample={infer_times_np.mean():.6f}s, '
                f'median={np.median(infer_times_np):.6f}s, '
                f'min={infer_times_np.min():.6f}s, '
                f'max={infer_times_np.max():.6f}s'
            )


def _predict_full_csv(model, device, csv_path, num_points=8192, protocol_seed=20260823):
    if csv_path.lower().endswith('.ply'):
        xyz, normal, label = read_ply_xyz_label(csv_path)
    else:
        df = pd.read_csv(csv_path)
        need_cols = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        missing = [c for c in need_cols if c not in df.columns]
        if missing:
            raise ValueError(f'Missing columns in {csv_path}: {missing}')
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        normal = df[['nx', 'ny', 'nz']].to_numpy(dtype=np.float32)
        label = df['label'].to_numpy(dtype=np.int64)

    xyz_norm = normalize_xyz(xyz)

    points = np.concatenate([xyz, normal, xyz_norm], axis=1).astype(np.float32)

    n = points.shape[0]
    probs = np.zeros((n, 2), dtype=np.float32)
    order = np.random.default_rng(protocol_seed).permutation(n)
    for start in range(0, n, num_points):
        end = min(start + num_points, n)
        valid_n = end - start
        valid_idx = order[start:end]
        pts = points[valid_idx]
        if valid_n < num_points:
            pad_idx = np.arange(num_points - valid_n) % max(valid_n, 1)
            pts = np.concatenate([pts, pts[pad_idx]], axis=0)
        inp = torch.from_numpy(pts).float().unsqueeze(0).to(device)
        inp = inp.transpose(2, 1)
        with torch.no_grad():
            logits, _ = model(inp)  # (1, N, 2)
            probs[valid_idx] = torch.softmax(logits.squeeze(0), dim=1).cpu().numpy()[:valid_n]
    pred = np.argmax(probs, axis=1).astype(np.int64)
    prob_weld = probs[:, 1].astype(np.float32)

    return xyz, label, pred, prob_weld


def infer_single(args, io):
    device = torch.device('cuda' if (torch.cuda.is_available() and not args.no_cuda) else 'cpu')
    model = build_model(args, device)

    if not args.model_path:
        raise ValueError('--model_path is required for --infer_csv')

    ckpt = torch.load(args.model_path, map_location=device)
    if 'state_dict' in ckpt:
        model.load_state_dict(ckpt['state_dict'])
    else:
        model.load_state_dict(ckpt)

    ds = SingleWeldCSV(
        csv_path=args.infer_csv,
        num_point=args.num_points,
        normalize=True,
        use_soft_label=args.use_soft_label,
        soft_threshold=args.soft_threshold,
    )

    model.eval()
    with torch.no_grad():
        pts, lbl, sample_idx, _ = ds[0]
        points = torch.from_numpy(pts).float().unsqueeze(0).to(device)  # (1, N, 9)
        points = points.transpose(2, 1)
        logits, _ = model(points)  # (1, N, C)
        pred = torch.argmax(logits.squeeze(0), dim=1).cpu().numpy().astype(np.int64)

    if lbl.min() >= 0:
        metrics = compute_metrics(pred, lbl, num_classes=2, weld_class=1)
        print_metrics(io, metrics)
    else:
        io.cprint('No ground-truth label in infer csv, skip metrics.')

    if args.save_pred_csv:
        df = pd.read_csv(args.infer_csv)
        pred_full = np.full((len(df),), -1, dtype=np.int64)
        pred_full[sample_idx] = pred
        df['pred'] = pred_full
        df.to_csv(args.save_pred_csv, index=False)
        io.cprint(f'Saved prediction csv to: {args.save_pred_csv}')

    if args.save_pred_xyz_label:
        xyz_full, _, pred_full, prob_weld = _predict_full_csv(
            model, device, args.infer_csv, num_points=args.num_points,
            protocol_seed=args.protocol_seed)
        out = np.concatenate([xyz_full, pred_full.reshape(-1, 1)], axis=1)
        out_dir = os.path.dirname(args.save_pred_xyz_label)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        np.savetxt(args.save_pred_xyz_label, out, fmt='%.6f %.6f %.6f %d')
        io.cprint(f'Saved xyz+label file to: {args.save_pred_xyz_label}')

    if args.save_pred_fullpred_csv:
        xyz_full, label_full, pred_full, prob_weld = _predict_full_csv(
            model, device, args.infer_csv, num_points=args.num_points,
            protocol_seed=args.protocol_seed)
        out_df = pd.DataFrame({
            'x': xyz_full[:, 0],
            'y': xyz_full[:, 1],
            'z': xyz_full[:, 2],
            'label': label_full,
            'pred': pred_full.astype(np.int64),
            'prob': prob_weld.astype(np.float32),
        })
        out_dir = os.path.dirname(args.save_pred_fullpred_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df.to_csv(args.save_pred_fullpred_csv, index=False)
        io.cprint(f'Saved fullpred csv to: {args.save_pred_fullpred_csv}')


def main():
    args = parse_args()
    _init_(args.exp_name)

    io = IOStream(f'outputs/{args.exp_name}/run.log')
    io.cprint(str(args))

    set_seed(args.seed)

    with open(f'outputs/{args.exp_name}/args.json', 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    if args.infer_csv:
        infer_single(args, io)
    elif args.eval:
        evaluate_split(args, io)
    else:
        train(args, io)


if __name__ == '__main__':
    main()
