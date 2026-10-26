import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List

import pandas as pd

_THIS_DIR = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from postprocess.postprocess_eval import (
    calc_baseline_metrics_from_conf,
    calc_confusion,
    postprocess_mask,
)


@dataclass
class EvalSample:
    file: str
    probs: object
    labels: object
    coords: object


def _parse_bool_list(s: str) -> List[bool]:
    vals = []
    for x in s.split(","):
        t = x.strip().lower()
        if t in {"1", "true", "t", "yes", "y"}:
            vals.append(True)
        elif t in {"0", "false", "f", "no", "n"}:
            vals.append(False)
        else:
            raise ValueError(f"无法解析布尔值: {x}")
    return vals


def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _match_pattern(filename: str, pattern: str) -> bool:
    if pattern == "*_fullpred.csv":
        return filename.endswith("_fullpred.csv")
    if pattern == "*.csv":
        return filename.endswith(".csv")
    if pattern.startswith("*") and pattern.endswith(".csv"):
        return filename.endswith(pattern[1:])
    return filename == pattern


def load_samples(input_dir: str, pattern: str) -> List[EvalSample]:
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv") and _match_pattern(f, pattern)])
    if not files:
        raise SystemExit(f"未找到文件: {input_dir}/{pattern}")

    samples: List[EvalSample] = []
    for fn in files:
        path = os.path.join(input_dir, fn)
        df = pd.read_csv(path)
        required = {"x", "y", "z", "label", "prob"}
        if not required.issubset(df.columns):
            raise ValueError(f"{path} 缺少必须列: {required}")

        labels = (df["label"].to_numpy(dtype="float32") > 0.5).astype("int32")
        probs = df["prob"].to_numpy(dtype="float32")
        coords = df[["x", "y", "z"]].to_numpy(dtype="float32")
        samples.append(EvalSample(file=fn, probs=probs, labels=labels, coords=coords))
    return samples


def evaluate_one_config(samples: Iterable[EvalSample], args_ns: argparse.Namespace) -> Dict[str, float]:
    g_before_tp = g_before_tn = g_before_fp = g_before_fn = 0
    g_after_tp = g_after_tn = g_after_fp = g_after_fn = 0

    for s in samples:
        pred_before = (s.probs > args_ns.fixed_thr).astype("int32")
        pred_after = postprocess_mask(s.coords, s.probs, args_ns).astype("int32")

        btp, btn, bfp, bfn = calc_confusion(s.labels, pred_before)
        atp, atn, afp, afn = calc_confusion(s.labels, pred_after)

        g_before_tp += btp
        g_before_tn += btn
        g_before_fp += bfp
        g_before_fn += bfn

        g_after_tp += atp
        g_after_tn += atn
        g_after_fp += afp
        g_after_fn += afn

    before = calc_baseline_metrics_from_conf(g_before_tp, g_before_tn, g_before_fp, g_before_fn)
    after = calc_baseline_metrics_from_conf(g_after_tp, g_after_tn, g_after_fp, g_after_fn)

    return {
        "hard_weld_iou_before": before["hard_weld_iou"],
        "hard_weld_iou_after": after["hard_weld_iou"],
        "hard_weld_iou_gain": after["hard_weld_iou"] - before["hard_weld_iou"],
        "hard_miou_before": before["hard_miou"],
        "hard_miou_after": after["hard_miou"],
        "hard_miou_gain": after["hard_miou"] - before["hard_miou"],
        "hard_oa_before": before["hard_oa"],
        "hard_oa_after": after["hard_oa"],
        "hard_weld_f1_before": before["hard_weld_f1"],
        "hard_weld_f1_after": after["hard_weld_f1"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="后处理参数扫参并输出最优参数")
    ap.add_argument("--input_dir", required=True, help="包含 *_fullpred.csv 的目录")
    ap.add_argument("--pattern", default="*_fullpred.csv")
    ap.add_argument("--output_dir", default="", help="默认 input_dir/postprocess_grid")

    ap.add_argument("--fixed_thr", type=float, default=0.53)

    ap.add_argument("--grid_enable_remove_small_components", default="true")
    ap.add_argument("--grid_comp_radius", default="0.08,0.10,0.12")
    ap.add_argument("--grid_min_comp_size", default="8,12,16")

    ap.add_argument("--grid_enable_bridge", default="false,true")
    ap.add_argument("--grid_min_comp_size_for_bridge", default="30,40")
    ap.add_argument("--grid_bridge_max_gap", default="0.08,0.10")
    ap.add_argument("--grid_bridge_corridor_radius", default="0.015,0.02")
    ap.add_argument("--grid_bridge_prob_min", default="0.15,0.2")
    ap.add_argument("--grid_bridge_dir_cos_min", default="0.75,0.85")

    ap.add_argument("--grid_enable_fill_holes", default="false,true")
    ap.add_argument("--grid_fill_k", default="12,16")
    ap.add_argument("--grid_fill_pos_ratio", default="0.55,0.65")
    ap.add_argument("--grid_fill_prob_min", default="0.1,0.15")

    ap.add_argument("--grid_enable_remove_sparse", default="false")
    ap.add_argument("--grid_sparse_radius", default="0.02,0.025")
    ap.add_argument("--grid_sparse_min_neighbors", default="1,2")

    ap.add_argument("--rank_metric", default="hard_miou_gain", choices=["hard_miou_gain", "hard_weld_iou_gain", "hard_weld_f1_after", "hard_miou_after"])
    ap.add_argument("--topk", type=int, default=20)
    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    out_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, "postprocess_grid")
    os.makedirs(out_dir, exist_ok=True)

    samples = load_samples(args.input_dir, args.pattern)

    grid = {
        "enable_remove_small_components": _parse_bool_list(args.grid_enable_remove_small_components),
        "comp_radius": _parse_float_list(args.grid_comp_radius),
        "min_comp_size": _parse_int_list(args.grid_min_comp_size),
        "enable_bridge": _parse_bool_list(args.grid_enable_bridge),
        "min_comp_size_for_bridge": _parse_int_list(args.grid_min_comp_size_for_bridge),
        "bridge_max_gap": _parse_float_list(args.grid_bridge_max_gap),
        "bridge_corridor_radius": _parse_float_list(args.grid_bridge_corridor_radius),
        "bridge_prob_min": _parse_float_list(args.grid_bridge_prob_min),
        "bridge_dir_cos_min": _parse_float_list(args.grid_bridge_dir_cos_min),
        "enable_fill_holes": _parse_bool_list(args.grid_enable_fill_holes),
        "fill_k": _parse_int_list(args.grid_fill_k),
        "fill_pos_ratio": _parse_float_list(args.grid_fill_pos_ratio),
        "fill_prob_min": _parse_float_list(args.grid_fill_prob_min),
        "enable_remove_sparse": _parse_bool_list(args.grid_enable_remove_sparse),
        "sparse_radius": _parse_float_list(args.grid_sparse_radius),
        "sparse_min_neighbors": _parse_int_list(args.grid_sparse_min_neighbors),
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    print(f"总组合数: {len(combos)}")

    rows = []
    best_row = None
    for i, combo in enumerate(combos, start=1):
        cfg = dict(zip(keys, combo))

        ns = argparse.Namespace(**cfg)
        ns.fixed_thr = args.fixed_thr

        metrics = evaluate_one_config(samples, ns)
        row = {**cfg, **metrics, "config_id": i}
        rows.append(row)

        if best_row is None or row[args.rank_metric] > best_row[args.rank_metric]:
            best_row = row

        if i % 20 == 0 or i == len(combos):
            print(f"进度: {i}/{len(combos)} 当前最优 {args.rank_metric}={best_row[args.rank_metric]:.6f} (config_id={best_row['config_id']})")

    df = pd.DataFrame(rows).sort_values(args.rank_metric, ascending=False)
    rank_path = os.path.join(out_dir, "grid_ranking.csv")
    df.to_csv(rank_path, index=False)

    top_df = df.head(args.topk)
    top_path = os.path.join(out_dir, f"grid_top{args.topk}.csv")
    top_df.to_csv(top_path, index=False)

    best_path = os.path.join(out_dir, "best_config.csv")
    pd.DataFrame([best_row]).to_csv(best_path, index=False)

    print("\n=== 最优参数 ===")
    for k in keys:
        print(f"{k}: {best_row[k]}")
    print(f"{args.rank_metric}: {best_row[args.rank_metric]:.6f}")
    print(f"hard_miou_before/after: {best_row['hard_miou_before']:.6f} -> {best_row['hard_miou_after']:.6f}")
    print(f"hard_weld_iou_before/after: {best_row['hard_weld_iou_before']:.6f} -> {best_row['hard_weld_iou_after']:.6f}")
    print(f"输出: {rank_path}")
    print(f"输出: {top_path}")
    print(f"输出: {best_path}")


if __name__ == "__main__":
    main()
