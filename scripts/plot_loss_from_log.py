#!/usr/bin/env python3
"""从一个或多个训练日志中提取 epoch-loss 并画到同一张图。

支持的常见日志格式：
1) [Epoch 001/200] ... train_loss=0.123 ... val_loss=0.456 ...
2) [Epoch 001/200] train_loss=0.123 val_loss=0.456 ...
3) Epoch 1 | Train 0.123 | Val 0.456
4) Train 193, loss: 0.030039
   Test 193, loss: 0.205728

用法示例：
    python scripts/plot_loss_from_log.py \
        comparable_methods/dgcnn.pytorch/job.70225.out \
        comparable_methods/Pointcept-copy/job.70523.out \
        comparable_methods/pointconv_pytorch/job.70497.out \
        comparable_methods/pointMLP-pytorch/job.70490.out \
        comparable_methods/Pointnet_Pointnet2_pytorch/job.70524.out \
        --out all_loss.png --csv all_loss.csv
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


# 旧脚本格式：Epoch 1 | Train 0.123 | Val 0.456
RE_PIPE = re.compile(
    r"Epoch\s+(\d+)\s*\|\s*Train\s*([+-]?\d*\.?\d+)\s*\|\s*Val\s*([+-]?\d*\.?\d+)"
)

# 常见格式：[Epoch 001/200] ... train_loss=0.123 ... val_loss=0.456
RE_TRAIN_VAL_IN_BRACKET_EPOCH = re.compile(
    r"\[Epoch\s*(\d+)/(\d+)\].*?train_loss=([+-]?\d*\.?\d+).*?val_loss=([+-]?\d*\.?\d+)"
)

# Pointcept 风格：Train 193, loss: 0.030039
RE_TRAIN_LINE = re.compile(r"Train\s+(\d+),\s*loss:\s*([+-]?\d*\.?\d+)")

# Pointcept / pointMLP 风格：Test 193, loss: 0.205728  test ins_iou: 0.773270
RE_TEST_LINE = re.compile(r"Test\s+(\d+),\s*loss:\s*([+-]?\d*\.?\d+)")

# pointMLP 风格：Train 193, loss: ..., train ins_iou: 0.888671
RE_TRAIN_INS_IOU = re.compile(r"Train\s+(\d+),.*?train\s+ins_iou:\s*([+-]?\d*\.?\d+)", re.IGNORECASE)

# pointMLP 风格：Test 193, loss: ..., test ins_iou: 0.773270
RE_TEST_INS_IOU = re.compile(r"Test\s+(\d+),.*?test\s+ins_iou:\s*([+-]?\d*\.?\d+)", re.IGNORECASE)

# [Epoch ...] ... val_miou=... val_weld_iou=...
RE_VAL_MIOU = re.compile(r"\[Epoch\s*(\d+)/(\d+)\].*?val_miou=([+-]?\d*\.?\d+)")
RE_VAL_WELD_IOU = re.compile(r"\[Epoch\s*(\d+)/(\d+)\].*?val_weld_iou=([+-]?\d*\.?\d+)")


def parse_log(path: Path):
    """返回字典，包含 loss 与 iou 相关序列（按 epoch 对齐，缺失为 None）。"""
    train_map = {}
    val_map = {}

    train_iou_map = {}
    val_iou_map = {}
    val_miou_map = {}

    # 用于 Pointcept 这种按 iteration 打印 loss 的日志：
    # Train: [epoch/total][iter/iter_total] ... loss: x
    # 我们会按 epoch 聚合成平均 train loss。
    epoch_train_losses = defaultdict(list)
    epoch_val_losses = defaultdict(list)

    re_pointcept_train_iter = re.compile(
        r"Train:\s*\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?loss:\s*([+-]?\d*\.?\d+)"
    )
    re_pointcept_val_iter = re.compile(
        r"Val:\s*\[(\d+)/(\d+)\]\[(\d+)/(\d+)\].*?loss:\s*([+-]?\d*\.?\d+)"
    )

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # 注意：同一行可能同时包含 loss 和 iou（如 [Epoch ...] 行），
            # 所以这里不要用互斥的 continue，而是逐项尝试匹配。

            # 1) [Epoch ...] train_loss=... val_loss=...
            m = RE_TRAIN_VAL_IN_BRACKET_EPOCH.search(line)
            if m:
                ep = int(m.group(1))
                tr = float(m.group(3))
                va = float(m.group(4))
                train_map[ep] = tr
                val_map[ep] = va

            # 2) Epoch ... | Train ... | Val ...
            m = RE_PIPE.search(line)
            if m:
                ep = int(m.group(1))
                tr = float(m.group(2))
                va = float(m.group(3))
                train_map[ep] = tr
                val_map[ep] = va

            # 3) Train <epoch>, loss: ...
            m = RE_TRAIN_LINE.search(line)
            if m:
                ep = int(m.group(1))
                tr = float(m.group(2))
                train_map[ep] = tr

            # 4) Test <epoch>, loss: ...
            m = RE_TEST_LINE.search(line)
            if m:
                ep = int(m.group(1))
                va = float(m.group(2))
                val_map[ep] = va

            # 4.1) pointMLP: Train ... train ins_iou: ...
            m = RE_TRAIN_INS_IOU.search(line)
            if m:
                ep = int(m.group(1))
                train_iou_map[ep] = float(m.group(2))

            # 4.2) pointMLP: Test ... test ins_iou: ...
            m = RE_TEST_INS_IOU.search(line)
            if m:
                ep = int(m.group(1))
                val_iou_map[ep] = float(m.group(2))

            # 4.3) [Epoch ...] ... val_miou=...
            m = RE_VAL_MIOU.search(line)
            if m:
                ep = int(m.group(1))
                val_miou_map[ep] = float(m.group(3))

            # 4.4) [Epoch ...] ... val_weld_iou=...
            m = RE_VAL_WELD_IOU.search(line)
            if m:
                ep = int(m.group(1))
                val_iou_map[ep] = float(m.group(3))

            # 5) Pointcept 迭代级 train 日志：按 epoch 聚合均值
            m = re_pointcept_train_iter.search(line)
            if m:
                ep = int(m.group(1))
                loss = float(m.group(5))
                epoch_train_losses[ep].append(loss)

            # 6) Pointcept 迭代级 val 日志（如果有）：按 epoch 聚合均值
            m = re_pointcept_val_iter.search(line)
            if m:
                ep = int(m.group(1))
                loss = float(m.group(5))
                epoch_val_losses[ep].append(loss)

    # 仅当没有直接解析到 train_map 时，才使用 iteration 聚合值，避免覆盖更可靠的 epoch 汇总值
    if not train_map and epoch_train_losses:
        for ep, losses in epoch_train_losses.items():
            train_map[ep] = sum(losses) / len(losses)

    if not val_map and epoch_val_losses:
        for ep, losses in epoch_val_losses.items():
            val_map[ep] = sum(losses) / len(losses)

    all_epochs = sorted(
        set(train_map.keys())
        | set(val_map.keys())
        | set(train_iou_map.keys())
        | set(val_iou_map.keys())
        | set(val_miou_map.keys())
    )

    train = [train_map.get(ep) for ep in all_epochs]
    val = [val_map.get(ep) for ep in all_epochs]
    train_iou = [train_iou_map.get(ep) for ep in all_epochs]
    val_iou = [val_iou_map.get(ep) for ep in all_epochs]
    val_miou = [val_miou_map.get(ep) for ep in all_epochs]

    return {
        "epochs": all_epochs,
        "train_loss": train,
        "val_loss": val,
        "train_iou": train_iou,
        "val_iou": val_iou,
        "val_miou": val_miou,
    }


def to_xy(epochs, values, min_epoch=1):
    """把可能包含 None 的序列转换为可绘图点，并按最小 epoch 过滤。"""
    x, y = [], []
    for e, v in zip(epochs, values):
        if v is not None and e >= min_epoch:
            x.append(e)
            y.append(v)
    return x, y


def default_label(path: Path) -> str:
    """默认曲线名：目录名/文件名（不含后缀）。"""
    parent = path.parent.name
    stem = path.stem
    return f"{parent}/{stem}" if parent else stem


def save_csv(out_csv: Path, rows):
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "log",
                "epoch",
                "train_loss",
                "val_loss",
                "train_iou",
                "val_iou",
                "val_miou",
            ]
        )
        writer.writerows(rows)


def plot_logs(log_files, out_png: Path, show=False, only="both", title="Loss Curves", min_epoch=1):
    fig, ax = plt.subplots(figsize=(11, 7))
    csv_rows = []

    plotted_any = False

    for p in log_files:
        parsed = parse_log(p)
        epochs = parsed["epochs"]
        if not epochs:
            print(f"[WARN] 未在日志中识别到可用指标: {p}")
            continue

        label_base = default_label(p)

        train = parsed["train_loss"]
        val = parsed["val_loss"]
        train_iou = parsed["train_iou"]
        val_iou = parsed["val_iou"]
        val_miou = parsed["val_miou"]

        tx, ty = to_xy(epochs, train, min_epoch=min_epoch)
        vx, vy = to_xy(epochs, val, min_epoch=min_epoch)
        tix, tiy = to_xy(epochs, train_iou, min_epoch=min_epoch)
        vix, viy = to_xy(epochs, val_iou, min_epoch=min_epoch)
        vmx, vmy = to_xy(epochs, val_miou, min_epoch=min_epoch)

        # 记录 CSV
        train_map = {e: v for e, v in zip(tx, ty)}
        val_map = {e: v for e, v in zip(vx, vy)}
        train_iou_map = {e: v for e, v in zip(tix, tiy)}
        val_iou_map = {e: v for e, v in zip(vix, viy)}
        val_miou_map = {e: v for e, v in zip(vmx, vmy)}
        for ep in epochs:
            csv_rows.append(
                [
                    label_base,
                    ep,
                    train_map.get(ep),
                    val_map.get(ep),
                    train_iou_map.get(ep),
                    val_iou_map.get(ep),
                    val_miou_map.get(ep),
                ]
            )

        if only in ("both", "train") and tx:
            ax.plot(tx, ty, linewidth=1.8, label=f"{label_base} | train_loss")
            plotted_any = True

        if only in ("both", "val") and vx:
            ax.plot(vx, vy, linewidth=1.8, linestyle="--", label=f"{label_base} | val_loss")
            plotted_any = True

        print(
            f"[OK] {p}: epochs={len(epochs)}, train_loss_pts={len(tx)}, val_loss_pts={len(vx)}, "
            f"train_iou_pts={len(tix)}, val_iou_pts={len(vix)}, val_miou_pts={len(vmx)}"
        )

    if not plotted_any:
        raise SystemExit("没有可绘制的loss曲线，请检查日志格式。")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"Saved loss plot to {out_png}")

    if show:
        plt.show()

    return csv_rows


def plot_iou_logs(log_files, out_png: Path, show=False, title="IoU Curves", min_epoch=1):
    fig, ax = plt.subplots(figsize=(11, 7))
    plotted_any = False

    for p in log_files:
        parsed = parse_log(p)
        epochs = parsed["epochs"]
        if not epochs:
            continue

        label_base = default_label(p)

        tix, tiy = to_xy(epochs, parsed["train_iou"], min_epoch=min_epoch)
        vix, viy = to_xy(epochs, parsed["val_iou"], min_epoch=min_epoch)
        vmx, vmy = to_xy(epochs, parsed["val_miou"], min_epoch=min_epoch)

        if tix:
            ax.plot(tix, tiy, linewidth=1.8, label=f"{label_base} | train_iou")
            plotted_any = True
        if vix:
            ax.plot(vix, viy, linewidth=1.8, linestyle="--", label=f"{label_base} | val_iou")
            plotted_any = True
        if vmx:
            ax.plot(vmx, vmy, linewidth=1.8, linestyle=":", label=f"{label_base} | val_miou")
            plotted_any = True

    if not plotted_any:
        print("[WARN] 没有识别到可绘制的IoU指标，跳过IoU图。")
        return

    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    print(f"Saved IoU plot to {out_png}")

    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description="从多个日志提取 epoch-loss/IoU 并合并绘图")
    p.add_argument(
        "logfiles",
        type=Path,
        nargs="+",
        help="一个或多个日志文件路径",
    )
    p.add_argument("--out", type=Path, default=Path("loss_plot_all.png"), help="Loss 图 PNG 路径")
    p.add_argument("--out-iou", type=Path, default=Path("iou_plot_all.png"), help="IoU 图 PNG 路径")
    p.add_argument("--skip-iou", action="store_true", help="跳过 IoU 曲线图")
    p.add_argument("--csv", type=Path, default=None, help="可选：导出合并后的 CSV")
    p.add_argument("--show", action="store_true", help="交互式显示图像")
    p.add_argument(
        "--only",
        choices=["both", "train", "val"],
        default="both",
        help="仅画 train、仅画 val 或两者都画（仅作用于 loss 图）",
    )
    p.add_argument("--title", type=str, default="Train/Val Loss per Epoch", help="Loss 图标题")
    p.add_argument("--title-iou", type=str, default="IoU per Epoch", help="IoU 图标题")
    p.add_argument("--min-epoch", type=int, default=1, help="从第几个 epoch 开始画（默认 1）")
    args = p.parse_args()

    missing = [x for x in args.logfiles if not x.exists()]
    if missing:
        miss = "\n".join(str(x) for x in missing)
        raise SystemExit(f"以下日志不存在:\n{miss}")

    rows = plot_logs(
        args.logfiles,
        args.out,
        show=args.show,
        only=args.only,
        title=args.title,
        min_epoch=args.min_epoch,
    )

    if not args.skip_iou:
        plot_iou_logs(
            args.logfiles,
            args.out_iou,
            show=args.show,
            title=args.title_iou,
            min_epoch=args.min_epoch,
        )

    if args.csv:
        save_csv(args.csv, rows)
        print(f"Saved CSV to {args.csv}")


if __name__ == "__main__":
    main()
