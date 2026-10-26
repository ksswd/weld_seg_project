import os
import time
import json
import argparse
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from data_utils.WeldDataset import WeldDataset
from model.pointconv_sem_seg import get_model, get_loss


def parse_args():
    parser = argparse.ArgumentParser("PointConv Weld Semantic Segmentation")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--gpu", type=str, default="0", help="gpu device")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--log_dir", type=str, default="log/weld_semseg_pointconv", help="log path")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--npoint", type=int, default=4096, help="point number")
    parser.add_argument("--step_size", type=int, default=20, help="lr decay step")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="lr decay rate")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--root", type=str, default="../Pointnet_Pointnet2_pytorch/data/weld", help="dataset root")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="", help="checkpoint path for eval_only")
    parser.add_argument("--save_pred", action="store_true", help="save per-sample prediction files on test split")
    parser.add_argument("--time_inference", action="store_true", help="measure model forward inference time on test split")
    parser.add_argument("--time_warmup", type=int, default=5, help="number of warmup batches excluded from timing")
    parser.add_argument("--timing_csv", type=str, default="")
    parser.add_argument("--protocol_seed", type=int, default=20260823)
    return parser.parse_args()


def compute_segmentation_metrics(pred, target, num_classes=2, weld_class=1):
    pred = pred.astype(np.int64)
    target = target.astype(np.int64)

    metrics = {}
    total_correct = np.sum(pred == target)
    total_seen = target.shape[0]
    oa = total_correct / float(total_seen + 1e-8)

    class_iou = []
    class_acc = []
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for t, p in zip(target, pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            confusion[t, p] += 1

    for cls in range(num_classes):
        tp = confusion[cls, cls]
        fn = confusion[cls, :].sum() - tp
        fp = confusion[:, cls].sum() - tp
        denom_iou = tp + fp + fn
        denom_acc = confusion[cls, :].sum()

        iou = tp / float(denom_iou + 1e-8)
        acc = tp / float(denom_acc + 1e-8)

        class_iou.append(iou)
        class_acc.append(acc)

    miou = float(np.mean(class_iou))
    macc = float(np.mean(class_acc))

    tp = confusion[weld_class, weld_class]
    fn = confusion[weld_class, :].sum() - tp
    fp = confusion[:, weld_class].sum() - tp

    precision = tp / float(tp + fp + 1e-8)
    recall = tp / float(tp + fn + 1e-8)
    f1 = 2 * precision * recall / float(precision + recall + 1e-8)
    weld_iou = class_iou[weld_class]

    metrics["oa"] = float(oa)
    metrics["macc"] = float(macc)
    metrics["miou"] = float(miou)
    metrics["class_iou"] = [float(x) for x in class_iou]
    metrics["class_acc"] = [float(x) for x in class_acc]
    metrics["weld_precision"] = float(precision)
    metrics["weld_recall"] = float(recall)
    metrics["weld_f1"] = float(f1)
    metrics["weld_iou"] = float(weld_iou)

    return metrics


def print_metrics_like_job(metrics):
    print(f"oa: {metrics['oa']:.6f}")
    print(f"macc: {metrics['macc']:.6f}")
    print(f"miou: {metrics['miou']:.6f}")
    print("class_iou: [" + ", ".join(f"{x:.6f}" for x in metrics["class_iou"]) + "]")
    print("class_acc: [" + ", ".join(f"{x:.6f}" for x in metrics["class_acc"]) + "]")
    print(f"weld_precision: {metrics['weld_precision']:.6f}")
    print(f"weld_recall: {metrics['weld_recall']:.6f}")
    print(f"weld_f1: {metrics['weld_f1']:.6f}")
    print(f"weld_iou: {metrics['weld_iou']:.6f}")


def evaluate(model, loader, device, num_classes=2, save_pred_dir=None, time_inference=False,
             time_warmup=5, protocol_seed=20260823):
    model.eval()
    loss_sum = 0.0
    count = 0
    all_pred = []
    all_target = []
    timed_batches = 0
    timed_samples = 0
    total_infer_time = 0.0
    timing_rows = []
    chunk_points = int(getattr(loader.dataset, "num_point", 4096))

    if save_pred_dir is not None:
        os.makedirs(save_pred_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if len(batch) == 3:
                points, target, sample_names = batch
            else:
                points, target = batch
                sample_names = [f"sample_{batch_idx}_{i}" for i in range(points.shape[0])]

            # Full-cloud deterministic chunk inference. PointConv constructs an
            # N x N density matrix, so passing a ~260k-point cloud would require
            # hundreds of GiB. Each original point is evaluated exactly once.
            points_np_batch = points.float().numpy()
            target_np_batch = target.long().numpy()
            for sample_idx in range(points_np_batch.shape[0]):
                sample_points = points_np_batch[sample_idx]
                sample_target = target_np_batch[sample_idx]
                n_points = int(sample_points.shape[0])
                pred_choice = np.zeros(n_points, dtype=np.int64)
                order = np.random.default_rng(protocol_seed).permutation(n_points)
                sample_elapsed = 0.0
                sample_loss_sum = 0.0
                sample_loss_points = 0

                for start in range(0, n_points, chunk_points):
                    end = min(start + chunk_points, n_points)
                    valid_n = end - start
                    valid_idx = order[start:end]
                    chunk = sample_points[valid_idx]
                    if valid_n < chunk_points:
                        pad_idx = np.arange(chunk_points - valid_n) % max(valid_n, 1)
                        chunk = np.concatenate([chunk, chunk[pad_idx]], axis=0)
                    chunk_tensor = torch.from_numpy(chunk[None]).float().to(device).transpose(2, 1)

                    if time_inference and device.type == "cuda":
                        torch.cuda.synchronize()
                    start_time = time.perf_counter()
                    pred_chunk = model(chunk_tensor)
                    if time_inference and device.type == "cuda":
                        torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start_time
                    sample_elapsed += elapsed

                    pred_valid = pred_chunk[:, :valid_n, :]
                    target_valid = torch.from_numpy(sample_target[valid_idx]).long().to(device)
                    sample_loss_sum += get_loss(pred_valid.reshape(-1, num_classes), target_valid).item() * valid_n
                    sample_loss_points += valid_n
                    pred_choice[valid_idx] = pred_valid.argmax(dim=2).squeeze(0).cpu().numpy()

                all_pred.append(pred_choice)
                all_target.append(sample_target)
                loss_sum += sample_loss_sum / max(sample_loss_points, 1)
                count += 1

                if time_inference and batch_idx >= time_warmup:
                    timed_batches += int(np.ceil(n_points / chunk_points))
                    timed_samples += 1
                    total_infer_time += sample_elapsed
                    timing_rows.append({"sample": str(sample_names[sample_idx]),
                                        "num_points": n_points,
                                        "infer_time_sec": float(sample_elapsed)})

                if save_pred_dir is not None:
                    sample_name = str(sample_names[sample_idx])
                    xyz = sample_points[:, :3]
                    out = np.concatenate([xyz, sample_target[:, None], pred_choice[:, None]], axis=1)
                    out_path = os.path.join(save_pred_dir, f"{sample_name}.txt")
                    np.savetxt(out_path, out, fmt="%.6f %.6f %.6f %d %d")

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    metrics = compute_segmentation_metrics(all_pred, all_target, num_classes=num_classes, weld_class=1)
    metrics["loss"] = float(loss_sum / max(count, 1))
    if time_inference:
        metrics["timed_batches"] = int(timed_batches)
        metrics["timed_samples"] = int(timed_samples)
        metrics["total_infer_time_sec"] = float(total_infer_time)
        metrics["avg_infer_time_per_batch_ms"] = float(total_infer_time / max(timed_batches, 1) * 1000.0)
        metrics["avg_infer_time_per_sample_ms"] = float(total_infer_time / max(timed_samples, 1) * 1000.0)
        metrics["timing_rows"] = timing_rows
    return metrics


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = args.log_dir
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    print("==== Load dataset ====")
    train_dataset = WeldDataset(root=args.root, split="train", num_point=args.npoint, use_normal=True, normalize_xyz=True, augment=True)
    val_dataset = WeldDataset(root=args.root, split="validation", num_point=args.npoint, use_normal=True, normalize_xyz=True, augment=False)
    test_dataset = WeldDataset(root=args.root, split="test", num_point=args.npoint, use_normal=True, normalize_xyz=True, augment=False, return_name=args.save_pred)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    num_classes = 2
    print("==== Build model ====")
    classifier = get_model(num_classes=num_classes, input_dim=9)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate)
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)

    lr_clip = 1e-5
    best_val_miou = -1.0
    best_epoch = -1
    best_model_path = os.path.join(checkpoints_dir, "best_model.pth")

    if args.eval_only:
        ckpt_path = args.checkpoint if args.checkpoint else best_model_path
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        classifier.load_state_dict(ckpt["model_state_dict"])
        pred_dir = os.path.join(experiment_dir, "pred_xyz_label_test") if args.save_pred else None
        test_metrics = evaluate(
            classifier,
            test_loader,
            device,
            num_classes=num_classes,
            save_pred_dir=pred_dir,
            time_inference=args.time_inference,
            time_warmup=args.time_warmup,
            protocol_seed=args.protocol_seed,
        )
        print("==== Test Metrics ====")
        print_metrics_like_job(test_metrics)
        if args.time_inference:
            print(f"timed_batches: {test_metrics['timed_batches']}")
            print(f"timed_samples: {test_metrics['timed_samples']}")
            print(f"total_infer_time_sec: {test_metrics['total_infer_time_sec']:.6f}")
            print(f"avg_infer_time_per_batch_ms: {test_metrics['avg_infer_time_per_batch_ms']:.6f}")
            print(f"avg_infer_time_per_sample_ms: {test_metrics['avg_infer_time_per_sample_ms']:.6f}")
        if pred_dir is not None:
            print(f"Saved test predictions to: {pred_dir}")
        if args.time_inference and args.timing_csv:
            timing_dir = os.path.dirname(args.timing_csv)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            pd.DataFrame(test_metrics["timing_rows"]).to_csv(args.timing_csv, index=False)
            print(f"Saved inference timing to: {args.timing_csv}")
        return

    print("==== Start training ====")
    for epoch in range(args.epoch):
        classifier.train()
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), lr_clip)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        loss_sum = 0.0
        total_correct = 0
        total_seen = 0
        start_time = time.time()

        for points, target in train_loader:
            optimizer.zero_grad()
            points = points.float().to(device)
            target = target.long().to(device)
            points = points.transpose(2, 1)

            pred = classifier(points)
            pred_reshape = pred.contiguous().view(-1, num_classes)
            target_reshape = target.view(-1)

            loss = get_loss(pred_reshape, target_reshape)
            loss.backward()
            optimizer.step()

            pred_choice = pred_reshape.max(1)[1]
            correct = pred_choice.eq(target_reshape).sum().item()
            total_correct += correct
            total_seen += target_reshape.numel()
            loss_sum += loss.item()

        train_loss = loss_sum / max(len(train_loader), 1)
        train_acc = total_correct / float(total_seen + 1e-8)
        val_metrics = evaluate(classifier, val_loader, device, num_classes=num_classes)
        epoch_time = time.time() - start_time

        print(f"[Epoch {epoch + 1:03d}/{args.epoch:03d}] lr={lr:.6f} train_loss={train_loss:.6f} train_acc={train_acc:.6f} val_loss={val_metrics['loss']:.6f} val_oa={val_metrics['oa']:.6f} val_miou={val_metrics['miou']:.6f} val_weld_iou={val_metrics['weld_iou']:.6f} val_weld_f1={val_metrics['weld_f1']:.6f} time={epoch_time:.1f}s")

        if val_metrics["miou"] > best_val_miou:
            best_val_miou = val_metrics["miou"]
            best_epoch = epoch + 1
            torch.save({"epoch": best_epoch, "class_iou": val_metrics["class_iou"], "best_val_miou": best_val_miou, "model_state_dict": classifier.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, best_model_path)
            print(f"==> Save best model to: {best_model_path}")

        if ((epoch + 1) % args.save_interval) == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"epoch_{epoch + 1:03d}.pth")
            torch.save({"epoch": epoch + 1, "model_state_dict": classifier.state_dict(), "optimizer_state_dict": optimizer.state_dict()}, ckpt_path)

    print("==== Training finished ====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val mIoU: {best_val_miou:.6f}")

    print("==== Final test with best model ====")
    ckpt = torch.load(best_model_path, map_location=device)
    classifier.load_state_dict(ckpt["model_state_dict"])
    pred_dir = os.path.join(experiment_dir, "pred_xyz_label_test") if args.save_pred else None
    test_metrics = evaluate(
        classifier,
        test_loader,
        device,
        num_classes=num_classes,
        save_pred_dir=pred_dir,
        time_inference=args.time_inference,
        time_warmup=args.time_warmup,
        protocol_seed=args.protocol_seed,
    )
    print_metrics_like_job(test_metrics)
    if args.time_inference:
        print(f"timed_batches: {test_metrics['timed_batches']}")
        print(f"timed_samples: {test_metrics['timed_samples']}")
        print(f"total_infer_time_sec: {test_metrics['total_infer_time_sec']:.6f}")
        print(f"avg_infer_time_per_batch_ms: {test_metrics['avg_infer_time_per_batch_ms']:.6f}")
        print(f"avg_infer_time_per_sample_ms: {test_metrics['avg_infer_time_per_sample_ms']:.6f}")
        if args.timing_csv:
            timing_dir = os.path.dirname(args.timing_csv)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            pd.DataFrame(test_metrics['timing_rows']).to_csv(args.timing_csv, index=False)
            print(f"Saved inference timing to: {args.timing_csv}")
    if pred_dir is not None:
        print(f"Saved test predictions to: {pred_dir}")


if __name__ == "__main__":
    main()
