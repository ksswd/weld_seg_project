from __future__ import print_function
import os
import argparse
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm

import model as models
from util.data_util import PartNormalDataset, WeldPartDataset
from util.util import to_categorical, compute_overall_iou, IOStream


def _init_(exp_name):
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(f'checkpoints/{exp_name}', exist_ok=True)


def build_dataset(args, split, return_meta=False):
    if args.dataset == 'weld':
        return WeldPartDataset(
            root=args.data_root,
            split=split,
            npoints=args.num_points,
            normalize=not args.disable_xyz_normalize,
            use_soft_label=args.use_soft_label,
            soft_threshold=args.soft_threshold,
            return_meta=return_meta,
        )
    mapped_split = 'trainval' if split == 'train' else split
    return PartNormalDataset(npoints=args.num_points, split=mapped_split, normalize=False)


def weight_init(m):
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Conv1d)):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def compute_binary_metrics(pred, target):
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)
    conf = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(target, pred):
        if 0 <= t < 2 and 0 <= p < 2:
            conf[t, p] += 1

    oa = np.trace(conf) / float(conf.sum() + 1e-8)
    class_iou, class_acc = [], []
    for c in range(2):
        tp = conf[c, c]
        fn = conf[c, :].sum() - tp
        fp = conf[:, c].sum() - tp
        class_iou.append(float(tp / float(tp + fp + fn + 1e-8)))
        class_acc.append(float(tp / float(conf[c, :].sum() + 1e-8)))

    miou = float(np.mean(class_iou))
    macc = float(np.mean(class_acc))

    tp = conf[1, 1]
    fn = conf[1, :].sum() - tp
    fp = conf[:, 1].sum() - tp
    weld_precision = tp / float(tp + fp + 1e-8)
    weld_recall = tp / float(tp + fn + 1e-8)
    weld_f1 = 2 * weld_precision * weld_recall / float(weld_precision + weld_recall + 1e-8)

    return {
        'oa': float(oa),
        'macc': float(macc),
        'miou': float(miou),
        'class_iou': class_iou,
        'class_acc': class_acc,
        'weld_precision': float(weld_precision),
        'weld_recall': float(weld_recall),
        'weld_f1': float(weld_f1),
        'weld_iou': float(class_iou[1]),
    }


def train_epoch(args, train_loader, model, opt, scheduler, epoch, io):
    num_part = args.num_part
    num_classes = args.num_classes
    train_loss, count, shape_ious = 0.0, 0.0, 0.0
    accuracy = []
    model.train()

    for points, label, target, norm_plt in tqdm(train_loader, total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points = Variable(points.float()).transpose(2, 1).cuda(non_blocking=True)
        label = Variable(label.long()).squeeze(1).cuda(non_blocking=True)
        target = Variable(target.long()).cuda(non_blocking=True)
        norm_plt = Variable(norm_plt.float()).transpose(2, 1).cuda(non_blocking=True)

        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)

        opt.zero_grad()
        loss.backward()
        opt.step()

        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target_flat = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target_flat.data).sum()

        shape_ious += batch_shapeious.item()
        count += batch_size
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))

    if args.scheduler == 'cos':
        scheduler.step()
    else:
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5

    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])
    io.cprint('Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch + 1, train_loss / count, np.mean(accuracy), shape_ious / count))


def test_epoch(args, test_loader, model, epoch, io):
    num_part = args.num_part
    num_classes = args.num_classes
    test_loss, count, shape_ious = 0.0, 0.0, 0.0
    accuracy = []
    model.eval()
    all_pred, all_gt = [], []

    for points, label, target, norm_plt in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points = Variable(points.float()).transpose(2, 1).cuda(non_blocking=True)
        label = Variable(label.long()).squeeze(1).cuda(non_blocking=True)
        target = Variable(target.long()).cuda(non_blocking=True)
        norm_plt = Variable(norm_plt.float()).transpose(2, 1).cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))

        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)

        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target_flat = target.view(-1, 1)[:, 0]
        loss = F.nll_loss(seg_pred.contiguous(), target_flat.contiguous())

        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target_flat.data).sum()

        shape_ious += batch_ious.item()
        count += batch_size
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))

        all_pred.append(pred_choice.detach().cpu().numpy())
        all_gt.append(target_flat.detach().cpu().numpy())

    metrics = {'accuracy': np.mean(accuracy), 'shape_avg_iou': shape_ious / count}
    io.cprint('Test %d, loss: %f, test acc: %f  test ins_iou: %f' % (epoch + 1, test_loss / count, metrics['accuracy'], metrics['shape_avg_iou']))

    if args.dataset == 'weld':
        pred = np.concatenate(all_pred, axis=0)
        gt = np.concatenate(all_gt, axis=0)
        metrics.update(compute_binary_metrics(pred, gt))

    return metrics


def train(args, io):
    device = torch.device('cuda' if args.cuda else 'cpu')
    model = models.__dict__[args.model](args.num_part, cls_in_channels=args.num_classes).to(device)
    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)
    io.cprint(f"Let's use {torch.cuda.device_count()} GPUs!")

    if args.resume:
        ckpt = torch.load(f"checkpoints/{args.exp_name}/best_insiou_model.pth", map_location=torch.device('cpu'))
        state_dict = ckpt['model']
        model.load_state_dict(state_dict)
        io.cprint('Resume training model...')
    else:
        io.cprint('Training from scratch...')

    train_data = build_dataset(args, 'train')
    val_data = build_dataset(args, 'val')
    io.cprint(f"The number of training data is:{len(train_data)}")
    io.cprint(f"The number of validation data is:{len(val_data)}")

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    if args.use_sgd:
        io.cprint('Use SGD')
        opt = optim.SGD(model.parameters(), lr=args.lr * 100, momentum=args.momentum, weight_decay=0)
    else:
        io.cprint('Use Adam')
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr / 100) if args.scheduler == 'cos' else StepLR(opt, step_size=args.step, gamma=0.5)

    best_acc, best_instance_iou = 0.0, 0.0
    for epoch in range(args.epochs):
        train_epoch(args, train_loader, model, opt, scheduler, epoch, io)
        val_metrics = test_epoch(args, val_loader, model, epoch, io)

        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            io.cprint('Max Acc:%.5f' % best_acc)
            torch.save({'model': model.module.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'val_acc': best_acc}, f'checkpoints/{args.exp_name}/best_acc_model.pth')

        if val_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = val_metrics['shape_avg_iou']
            io.cprint('Max instance iou:%.5f' % best_instance_iou)
            torch.save({'model': model.module.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'val_instance_iou': best_instance_iou}, f'checkpoints/{args.exp_name}/best_insiou_model.pth')

    io.cprint('Final Max Acc:%.5f' % best_acc)
    io.cprint('Final Max instance iou:%.5f' % best_instance_iou)


def test(args, io):
    need_export = args.eval_only or args.export_pred
    test_data = build_dataset(args, 'test', return_meta=need_export)
    io.cprint(f"The number of test data is:{len(test_data)}")
    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    device = torch.device('cuda' if args.cuda else 'cpu')
    model = models.__dict__[args.model](args.num_part, cls_in_channels=args.num_classes).to(device)
    io.cprint(str(model))

    ckpt = torch.load(f"checkpoints/{args.exp_name}/best_{args.model_type}_model.pth", map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    if need_export:
        os.makedirs(args.pred_xyz_label_dir, exist_ok=True)
        os.makedirs(args.fullpred_csv_dir, exist_ok=True)

    all_pred, all_gt = [], []
    total_inference_time = 0.0
    total_samples = 0
    timing_rows = []
    for batch in tqdm(test_loader, total=len(test_loader), smoothing=0.9):
        if need_export:
            points, label, target, norm_plt, file_paths, sampled_indices = batch
        else:
            points, label, target, norm_plt = batch

        points = Variable(points.float()).transpose(2, 1).cuda(non_blocking=True)
        # Keep the batch dimension when test_batch_size == 1.  An unrestricted
        # squeeze turns [1, 1] into a scalar, so one-hot encoding becomes [1]
        # and cls_map/Conv1d receives a 2-D tensor instead of [B, C, 1].
        label = Variable(label.long()).squeeze(1).cuda(non_blocking=True)
        target = Variable(target.long()).cuda(non_blocking=True)
        norm_plt = Variable(norm_plt.float()).transpose(2, 1).cuda(non_blocking=True)

        # Deterministic full-cloud inference in fixed-size chunks.  This makes
        # --num_points the actual number of points per model forward and covers
        # every original point exactly once.
        seg_pred = torch.empty(
            (points.size(0), points.size(2), args.num_part),
            dtype=points.dtype, device=points.device)
        batch_inference_time = 0.0
        total_n = points.size(2)
        order_np = np.random.default_rng(args.protocol_seed).permutation(total_n)
        order = torch.from_numpy(order_np).long().to(points.device)
        for start in range(0, total_n, args.num_points):
            end = min(start + args.num_points, total_n)
            valid_n = end - start
            valid_idx = order[start:end]
            point_chunk = points[:, :, valid_idx]
            normal_chunk = norm_plt[:, :, valid_idx]
            if valid_n < args.num_points:
                pad_idx = torch.arange(args.num_points - valid_n, device=points.device) % max(valid_n, 1)
                point_chunk = torch.cat([point_chunk, point_chunk[:, :, pad_idx]], dim=2)
                normal_chunk = torch.cat([normal_chunk, normal_chunk[:, :, pad_idx]], dim=2)
            if args.cuda:
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            with torch.no_grad():
                chunk_pred = model(point_chunk, normal_chunk, to_categorical(label, args.num_classes))
            if args.cuda:
                torch.cuda.synchronize()
            batch_inference_time += time.perf_counter() - inference_start
            seg_pred[:, valid_idx, :] = chunk_pred[:, :valid_n, :]
        total_inference_time += batch_inference_time
        total_samples += points.size(0)

        pred_point = seg_pred.data.max(2)[1].detach().cpu().numpy()
        gt_point = target.detach().cpu().numpy()

        seg_pred_flat = seg_pred.contiguous().view(-1, args.num_part)
        target_flat = target.view(-1, 1)[:, 0]
        all_pred.append(seg_pred_flat.data.max(1)[1].detach().cpu().numpy())
        all_gt.append(target_flat.data.cpu().numpy())

        if need_export:
            sampled_indices = sampled_indices.detach().cpu().numpy()
            for bi in range(pred_point.shape[0]):
                src_csv = file_paths[bi]
                stem = os.path.splitext(os.path.basename(src_csv))[0]
                df_src = pd.read_csv(src_csv)

                idx = sampled_indices[bi]
                xyz = df_src[['x', 'y', 'z']].to_numpy()[idx]
                gt = gt_point[bi]
                pred = pred_point[bi]

                txt_out = os.path.join(args.pred_xyz_label_dir, f"{stem}.txt")
                arr = np.concatenate([xyz, gt.reshape(-1, 1), pred.reshape(-1, 1)], axis=1)
                np.savetxt(txt_out, arr, fmt=['%.6f', '%.6f', '%.6f', '%d', '%d'])

                df_out = pd.DataFrame({'x': xyz[:, 0], 'y': xyz[:, 1], 'z': xyz[:, 2], 'gt': gt, 'pred': pred})
                csv_out = os.path.join(args.fullpred_csv_dir, f"{stem}.csv")
                df_out.to_csv(csv_out, index=False)
                timing_rows.append({'sample': stem, 'num_points': int(len(gt)),
                                    'infer_time_sec': float(batch_inference_time / pred_point.shape[0])})

    metrics = compute_binary_metrics(np.concatenate(all_pred, axis=0), np.concatenate(all_gt, axis=0))
    io.cprint(f"oa: {metrics['oa']:.6f}")
    io.cprint(f"macc: {metrics['macc']:.6f}")
    io.cprint(f"miou: {metrics['miou']:.6f}")
    io.cprint(f"class_iou: [{metrics['class_iou'][0]:.6f}, {metrics['class_iou'][1]:.6f}]")
    io.cprint(f"class_acc: [{metrics['class_acc'][0]:.6f}, {metrics['class_acc'][1]:.6f}]")
    io.cprint(f"weld_precision: {metrics['weld_precision']:.6f}")
    io.cprint(f"weld_recall: {metrics['weld_recall']:.6f}")
    io.cprint(f"weld_f1: {metrics['weld_f1']:.6f}")
    io.cprint(f"weld_iou: {metrics['weld_iou']:.6f}")
    average_inference_time = total_inference_time / max(total_samples, 1)
    io.cprint(f"Average inference time per sample: {average_inference_time:.6f} s")
    io.cprint(f"Total inference time: {total_inference_time:.6f} s")
    io.cprint(f"Total samples: {total_samples}")
    if need_export:
        io.cprint(f"Exported txt to: {args.pred_xyz_label_dir}")
        io.cprint(f"Exported csv to: {args.fullpred_csv_dir}")
    if args.timing_csv:
        timing_dir = os.path.dirname(args.timing_csv)
        if timing_dir:
            os.makedirs(timing_dir, exist_ok=True)
        pd.DataFrame(timing_rows).to_csv(args.timing_csv, index=False)
        io.cprint(f"Saved inference timing to: {args.timing_csv}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation')
    parser.add_argument('--model', type=str, default='pointMLP')
    parser.add_argument('--exp_name', type=str, default='weld_baseline', metavar='N')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--use_sgd', type=bool, default=False)
    parser.add_argument('--scheduler', type=str, default='step')
    parser.add_argument('--step', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--no_cuda', type=bool, default=False)
    parser.add_argument('--manual_seed', type=int, metavar='S')
    parser.add_argument('--protocol_seed', type=int, default=20260823)
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--eval_only', action='store_true', help='run evaluation only and export prediction files')
    parser.add_argument('--export_pred', action='store_true', help='export per-sample prediction files during eval')
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--model_type', type=str, default='insiou', choices=['insiou', 'acc'])

    parser.add_argument('--dataset', type=str, default='weld', choices=['weld', 'shapenet'])
    parser.add_argument('--data_root', type=str, default=os.environ.get('POINTMLP_DATA_ROOT', '../../data/dataset_processed'))
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--use_soft_label', action='store_true')
    parser.add_argument('--soft_threshold', type=float, default=0.5)
    parser.add_argument('--disable_xyz_normalize', action='store_true')
    parser.add_argument('--num_part', type=int, default=2)
    parser.add_argument('--num_classes', type=int, default=1)
    parser.add_argument('--pred_xyz_label_dir', type=str, default='pred_xyz_label_test')
    parser.add_argument('--fullpred_csv_dir', type=str, default='fullpred_test')
    parser.add_argument('--timing_csv', type=str, default='')

    args = parser.parse_args()
    args.exp_name = args.model + '_' + args.exp_name
    if args.eval_only:
        args.eval = True

    _init_(args.exp_name)

    io = IOStream(f"checkpoints/{args.exp_name}/{args.exp_name}_{'test' if args.eval else 'train'}.log")
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    io.cprint('Using GPU' if args.cuda else 'Using CPU')
    if args.cuda and args.manual_seed is not None:
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)

    if not args.eval:
        train(args, io)
    else:
        test(args, io)
