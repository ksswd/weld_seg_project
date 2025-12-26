#!/usr/bin/env python3
"""Parse a training log and plot Train Loss and Val Loss per epoch.

Usage:
    python scripts/plot_loss_from_log.py /path/to/job.41087.out --out loss_plot.png

This script looks for lines like:
    Epoch 1, Train Loss: 0.5472, Val Loss: 0.5409, ...
and extracts epoch number, train loss and val loss.
"""
import re
import argparse
from pathlib import Path
import csv
import matplotlib.pyplot as plt


LOSS_LINE_RE = re.compile(
    r"Epoch\s+(\d+)\s+\|\s+Train Loss:\s+([\d.]+)\s*,\s*Val Loss:\s+([\d.]+)\s*,\s*F1:\s+([\d.]+)\s*,\s*Best Threshold:\s+([\d.]+)"
)


def parse_log(path):
    epochs = []
    train = []
    val = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            m = LOSS_LINE_RE.search(line)
            if m:
                ep = int(m.group(1))
                tr = float(m.group(2))
                va = float(m.group(3))
                epochs.append(ep)
                train.append(tr)
                val.append(va)
    return epochs, train, val


def save_csv(out_csv: Path, epochs, train, val):
    with open(out_csv, 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['epoch', 'train_loss', 'val_loss'])
        for e, t, v in zip(epochs, train, val):
            writer.writerow([e, t, v])


def plot_and_save(epochs, train, val, out_png: Path, show=False):
    # plt.style.use('seaborn-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train, marker='o', label='Train Loss')
    ax.plot(epochs, val, marker='s', label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train and Validation Loss per Epoch')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"Saved plot to {out_png}")
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser(description='Plot Train/Val loss from a training log file')
    p.add_argument('logfile', type=Path, default='train_log.txt', help='Path to the training log file' )
    p.add_argument('--out', type=Path, default=Path('loss_plot_ft.png'), help='Output PNG path')
    p.add_argument('--csv', type=Path, default=None, help='Optional CSV output path')
    p.add_argument('--show', action='store_true', help='Show the plot interactively')
    args = p.parse_args()

    if not args.logfile.exists():
        raise SystemExit(f"Log file not found: {args.logfile}")

    epochs, train, val = parse_log(args.logfile)
    if len(epochs) == 0:
        raise SystemExit("No loss lines found in the log. Please check the logfile format.")

    # sort by epoch in case the log is not strictly ordered
    paired = sorted(zip(epochs, train, val), key=lambda x: x[0])
    epochs, train, val = zip(*paired)

    if args.csv:
        save_csv(args.csv, epochs, train, val)
        print(f"Saved CSV to {args.csv}")

    plot_and_save(epochs, train, val, args.out, show=args.show)


if __name__ == '__main__':
    main()
