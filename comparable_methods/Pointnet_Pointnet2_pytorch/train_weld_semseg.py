import os
import sys
import time
import copy
import math
import json
import argparse
import importlib
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, "models"))
sys.path.append(os.path.join(ROOT_DIR, "data_utils"))

from data_utils.WeldDataset import WeldDataset


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find("ReLU") != -1:
        m.inplace = True


def parse_args():
    parser = argparse.ArgumentParser("PointNet++ Weld Semantic Segmentation")
    parser.add_argument("--model", type=str, default="pointnet2_sem_seg", help="model name")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--epoch", default=100, type=int, help="number of epochs")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--gpu", type=str, default="0", help="gpu device")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--log_dir", type=str, default="log/weld_semseg", help="log path")
    parser.add_argument("--decay_rate", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--npoint", type=int, default=4096, help="point number")
    parser.add_argument("--step_size", type=int, default=20, help="lr decay step")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="lr decay rate")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--root", type=str, default="data/weld", help="dataset root")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--seed", type=int, default=444)
    parser.add_argument("--disable_class_weights", action="store_true")
    return parser.parse_args()


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_segmentation_metrics(pred, target, num_classes=2, weld_class=1):
    """
    pred: (N,)
    target: (N,)
    """
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

    # weld class metrics
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


def evaluate(model, criterion, loader, device, num_classes=2, class_weights=None):
    model.eval()

    loss_sum = 0.0
    count = 0

    all_pred = []
    all_target = []

    with torch.no_grad():
        for points, target in loader:
            points = points.float().to(device)       # (B, N, 9)
            target = target.long().to(device)        # (B, N)

            points = points.transpose(2, 1)          # (B, 9, N)

            pred, trans_feat = model(points)         # pred: (B, N, C)
            pred_reshape = pred.contiguous().view(-1, num_classes)
            target_reshape = target.view(-1)

            loss = criterion(pred_reshape, target_reshape, trans_feat, class_weights)
            loss_sum += loss.item()
            count += 1

            pred_choice = pred_reshape.max(1)[1].detach().cpu().numpy()
            target_np = target_reshape.detach().cpu().numpy()

            all_pred.append(pred_choice)
            all_target.append(target_np)

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    metrics = compute_segmentation_metrics(all_pred, all_target, num_classes=num_classes, weld_class=1)
    metrics["loss"] = float(loss_sum / max(count, 1))
    return metrics


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    experiment_dir = args.log_dir
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    writer = SummaryWriter(log_dir=os.path.join(experiment_dir, "tensorboard"))

    print("==== Load dataset ====")
    train_dataset = WeldDataset(
        root=args.root,
        split="train",
        num_point=args.npoint,
        use_normal=True,
        normalize_xyz=True,
        augment=True,
    )
    val_dataset = WeldDataset(
        root=args.root,
        split="val",
        num_point=args.npoint,
        use_normal=True,
        normalize_xyz=True,
        augment=False,
    )
    test_dataset = WeldDataset(
        root=args.root,
        split="test",
        num_point=args.npoint,
        use_normal=True,
        normalize_xyz=True,
        augment=False,
    )
    class_weights = None if args.disable_class_weights else torch.tensor(
        train_dataset.labelweights, dtype=torch.float32)
    print(f"class_weights: {None if class_weights is None else class_weights.tolist()}")
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    num_classes = 2

    print("==== Build model ====")
    model_module = importlib.import_module(args.model)
    classifier = model_module.get_model(num_classes)
    criterion = model_module.get_loss()
    classifier.apply(inplace_relu)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = classifier.to(device)
    criterion = criterion.to(device)
    if class_weights is not None:
        class_weights = class_weights.to(device)

    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate,
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=0.9,
            weight_decay=args.decay_rate,
        )

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    best_val_miou = -1.0
    best_val_weld_f1 = -1.0
    best_epoch = -1

    best_model_path = os.path.join(checkpoints_dir, "best_model.pth")

    if args.eval_only:
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"best model not found: {best_model_path}")
        ckpt = torch.load(best_model_path, map_location=device)
        classifier.load_state_dict(ckpt["model_state_dict"])
        test_metrics = evaluate(classifier, criterion, test_loader, device,
                                num_classes=num_classes, class_weights=class_weights)
        print("==== Test Metrics ====")
        for k, v in test_metrics.items():
            print(f"{k}: {v}")
        return

    print("==== Start training ====")
    global_step = 0

    for epoch in range(args.epoch):
        classifier.train()

        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        loss_sum = 0.0
        total_correct = 0
        total_seen = 0

        start_time = time.time()

        for i, (points, target) in enumerate(train_loader):
            optimizer.zero_grad()

            points = points.float().to(device)   # (B, N, 9)
            target = target.long().to(device)    # (B, N)

            points = points.transpose(2, 1)      # (B, 9, N)

            pred, trans_feat = classifier(points)          # (B, N, C)
            pred_reshape = pred.contiguous().view(-1, num_classes)
            target_reshape = target.view(-1)

            loss = criterion(pred_reshape, target_reshape, trans_feat, class_weights)
            loss.backward()
            optimizer.step()

            pred_choice = pred_reshape.max(1)[1]
            correct = pred_choice.eq(target_reshape).sum().item()

            total_correct += correct
            total_seen += target_reshape.numel()
            loss_sum += loss.item()

            writer.add_scalar("train/batch_loss", loss.item(), global_step)
            global_step += 1

        train_loss = loss_sum / max(len(train_loader), 1)
        train_acc = total_correct / float(total_seen + 1e-8)

        val_metrics = evaluate(classifier, criterion, val_loader, device,
                               num_classes=num_classes, class_weights=class_weights)

        epoch_time = time.time() - start_time

        print(
            f"[Epoch {epoch + 1:03d}/{args.epoch:03d}] "
            f"lr={lr:.6f} "
            f"train_loss={train_loss:.6f} "
            f"train_acc={train_acc:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_oa={val_metrics['oa']:.6f} "
            f"val_miou={val_metrics['miou']:.6f} "
            f"val_weld_iou={val_metrics['weld_iou']:.6f} "
            f"val_weld_f1={val_metrics['weld_f1']:.6f} "
            f"time={epoch_time:.1f}s"
        )

        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/epoch_acc", train_acc, epoch)
        writer.add_scalar("train/lr", lr, epoch)

        writer.add_scalar("val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("val/oa", val_metrics["oa"], epoch)
        writer.add_scalar("val/macc", val_metrics["macc"], epoch)
        writer.add_scalar("val/miou", val_metrics["miou"], epoch)
        writer.add_scalar("val/weld_iou", val_metrics["weld_iou"], epoch)
        writer.add_scalar("val/weld_precision", val_metrics["weld_precision"], epoch)
        writer.add_scalar("val/weld_recall", val_metrics["weld_recall"], epoch)
        writer.add_scalar("val/weld_f1", val_metrics["weld_f1"], epoch)

        is_best = val_metrics["miou"] > best_val_miou
        if is_best:
            best_val_miou = val_metrics["miou"]
            best_val_weld_f1 = val_metrics["weld_f1"]
            best_epoch = epoch + 1

            state = {
                "epoch": best_epoch,
                "seed": args.seed,
                "input_layout": "xyz_norm_normal_xyz_norm",
                "class_weights": None if class_weights is None else class_weights.detach().cpu().tolist(),
                "class_iou": val_metrics["class_iou"],
                "best_val_miou": best_val_miou,
                "best_val_weld_f1": best_val_weld_f1,
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, best_model_path)
            print(f"==> Save best model to: {best_model_path}")

        if ((epoch + 1) % args.save_interval) == 0:
            ckpt_path = os.path.join(checkpoints_dir, f"epoch_{epoch + 1:03d}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": classifier.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                ckpt_path,
            )

    print("==== Training finished ====")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val mIoU: {best_val_miou:.6f}")
    print(f"Best val weld F1: {best_val_weld_f1:.6f}")

    print("==== Final test with best model ====")
    ckpt = torch.load(best_model_path, map_location=device)
    classifier.load_state_dict(ckpt["model_state_dict"])

    test_metrics = evaluate(classifier, criterion, test_loader, device,
                            num_classes=num_classes, class_weights=class_weights)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    writer.close()


if __name__ == "__main__":
    main()
