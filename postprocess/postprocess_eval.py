import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


@dataclass
class CaseMetrics:
    file: str
    n_points: int
    pos_ratio: float
    pred_ratio_before: float
    pred_ratio_after: float
    hard_weld_iou_before: float
    hard_weld_iou_after: float
    hard_miou_before: float
    hard_miou_after: float
    hard_oa_before: float
    hard_oa_after: float
    hard_weld_f1_before: float
    hard_weld_f1_after: float
    hard_weld_precision_before: float
    hard_weld_precision_after: float
    hard_weld_recall_before: float
    hard_weld_recall_after: float


def calc_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_pred = y_pred.astype(np.int32).reshape(-1)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, tn, fp, fn


def calc_baseline_metrics_from_conf(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    oa = (tp + tn) / max(tp + tn + fp + fn, 1)
    class_iou_0 = tn / max(tn + fp + fn, 1)
    class_iou_1 = tp / max(tp + fp + fn, 1)
    miou = (class_iou_0 + class_iou_1) / 2.0

    weld_precision = tp / max(tp + fp, 1)
    weld_recall = tp / max(tp + fn, 1)
    weld_f1 = 2 * weld_precision * weld_recall / max(weld_precision + weld_recall, 1e-12)

    return {
        "hard_oa": float(oa),
        "hard_miou": float(miou),
        "hard_weld_iou": float(class_iou_1),
        "hard_weld_precision": float(weld_precision),
        "hard_weld_recall": float(weld_recall),
        "hard_weld_f1": float(weld_f1),
    }


def build_neighbor_indices(points: np.ndarray, radius: float) -> List[np.ndarray]:
    if points.shape[0] == 0:
        return []
    nn = NearestNeighbors(radius=radius, algorithm="auto")
    nn.fit(points)
    inds = nn.radius_neighbors(points, return_distance=False)
    return inds


def connected_components_from_mask(coords: np.ndarray, mask: np.ndarray, radius: float) -> List[np.ndarray]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []

    sub_pts = coords[idx]
    neigh = build_neighbor_indices(sub_pts, radius=radius)
    visited = np.zeros(idx.size, dtype=bool)
    components_sub = []

    for i in range(idx.size):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in neigh[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(int(v))
        components_sub.append(np.array(comp, dtype=np.int32))

    return [idx[c] for c in components_sub]


def remove_small_components(coords: np.ndarray, pred_mask: np.ndarray, comp_radius: float, min_comp_size: int) -> np.ndarray:
    comps = connected_components_from_mask(coords, pred_mask, radius=comp_radius)
    out = pred_mask.copy()
    for c in comps:
        if c.size < min_comp_size:
            out[c] = False
    return out


def get_component_direction(pts: np.ndarray) -> np.ndarray:
    if pts.shape[0] < 3:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(pts.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    d = eigvecs[:, np.argmax(eigvals)]
    n = np.linalg.norm(d)
    return d / n if n > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)


def component_endpoints(pts: np.ndarray, direction: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    proj = pts @ direction
    return pts[np.argmin(proj)], pts[np.argmax(proj)]


def bridge_components(
    coords: np.ndarray,
    probs: np.ndarray,
    pred_mask: np.ndarray,
    comp_radius: float,
    min_comp_size_for_bridge: int,
    bridge_max_gap: float,
    bridge_corridor_radius: float,
    bridge_prob_min: float,
    bridge_dir_cos_min: float,
) -> np.ndarray:
    out = pred_mask.copy()
    comps = connected_components_from_mask(coords, out, radius=comp_radius)
    comps = [c for c in comps if c.size >= min_comp_size_for_bridge]
    if len(comps) < 2:
        return out

    comp_info = []
    for c in comps:
        pts = coords[c]
        d = get_component_direction(pts)
        p0, p1 = component_endpoints(pts, d)
        comp_info.append({"idx": c, "dir": d, "endpoints": (p0, p1)})

    background_idx = np.where(~out)[0]
    if background_idx.size == 0:
        return out
    background_pts = coords[background_idx]

    for i in range(len(comp_info)):
        for j in range(i + 1, len(comp_info)):
            a = comp_info[i]
            b = comp_info[j]

            cosv = float(abs(np.dot(a["dir"], b["dir"])))
            if cosv < bridge_dir_cos_min:
                continue

            ends_a = a["endpoints"]
            ends_b = b["endpoints"]
            best = None
            best_dist = 1e18
            for ea in ends_a:
                for eb in ends_b:
                    d = float(np.linalg.norm(ea - eb))
                    if d < best_dist:
                        best_dist = d
                        best = (ea, eb)

            if best is None or best_dist > bridge_max_gap:
                continue

            p0, p1 = best
            seg = p1 - p0
            seg_len2 = float(np.dot(seg, seg))
            if seg_len2 < 1e-12:
                continue

            vec = background_pts - p0
            t = np.clip((vec @ seg) / seg_len2, 0.0, 1.0)
            proj = p0[None, :] + t[:, None] * seg[None, :]
            dist = np.linalg.norm(background_pts - proj, axis=1)

            candidates = (dist <= bridge_corridor_radius) & (probs[background_idx] >= bridge_prob_min)
            if np.any(candidates):
                out[background_idx[candidates]] = True

    return out


def fill_inner_holes_knn(
    coords: np.ndarray,
    probs: np.ndarray,
    pred_mask: np.ndarray,
    k: int,
    fill_pos_ratio: float,
    fill_prob_min: float,
) -> np.ndarray:
    out = pred_mask.copy()
    unknown_idx = np.where(~out)[0]
    if unknown_idx.size == 0:
        return out

    nn = NearestNeighbors(n_neighbors=min(k + 1, coords.shape[0]), algorithm="auto")
    nn.fit(coords)
    neigh = nn.kneighbors(coords[unknown_idx], return_distance=False)
    neigh = neigh[:, 1:] if neigh.shape[1] > 1 else neigh
    neigh_pred_ratio = pred_mask[neigh].mean(axis=1) if neigh.size > 0 else np.zeros(unknown_idx.size)

    add = (probs[unknown_idx] >= fill_prob_min) & (neigh_pred_ratio >= fill_pos_ratio)
    out[unknown_idx[add]] = True
    return out


def remove_sparse_points(pred_mask: np.ndarray, coords: np.ndarray, sparse_radius: float, sparse_min_neighbors: int) -> np.ndarray:
    out = pred_mask.copy()
    idx = np.where(out)[0]
    if idx.size == 0:
        return out

    pts = coords[idx]
    neigh = build_neighbor_indices(pts, radius=sparse_radius)
    keep_local = np.array([len(n) - 1 >= sparse_min_neighbors for n in neigh], dtype=bool)
    out[idx[~keep_local]] = False
    return out


def postprocess_mask(coords: np.ndarray, probs: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    pred = probs > args.fixed_thr

    if args.enable_remove_small_components:
        pred = remove_small_components(coords, pred, args.comp_radius, args.min_comp_size)

    if args.enable_bridge:
        pred = bridge_components(
            coords=coords,
            probs=probs,
            pred_mask=pred,
            comp_radius=args.comp_radius,
            min_comp_size_for_bridge=args.min_comp_size_for_bridge,
            bridge_max_gap=args.bridge_max_gap,
            bridge_corridor_radius=args.bridge_corridor_radius,
            bridge_prob_min=args.bridge_prob_min,
            bridge_dir_cos_min=args.bridge_dir_cos_min,
        )

    if args.enable_fill_holes:
        pred = fill_inner_holes_knn(coords, probs, pred, args.fill_k, args.fill_pos_ratio, args.fill_prob_min)

    if args.enable_remove_sparse:
        pred = remove_sparse_points(pred, coords, args.sparse_radius, args.sparse_min_neighbors)

    if args.enable_remove_small_components:
        pred = remove_small_components(coords, pred, args.comp_radius, args.min_comp_size)

    return pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对 fullpred 结果进行后处理并按统一口径评估")
    parser.add_argument("--input_dir", required=True, help="包含 *_fullpred.csv 的目录")
    parser.add_argument("--pattern", default="*_fullpred.csv", help="输入文件匹配模式")
    parser.add_argument("--output_dir", default="", help="输出目录，默认 input_dir/postprocessed")

    parser.add_argument("--fixed_thr", type=float, default=0.53, help="与 test/per_case_metrics.py 对齐的固定阈值")

    parser.add_argument("--enable_remove_small_components", action="store_true", help="启用小连通域剔除")
    parser.add_argument("--comp_radius", type=float, default=0.06, help="连通域邻接半径")
    parser.add_argument("--min_comp_size", type=int, default=25, help="最小连通域点数")

    parser.add_argument("--enable_bridge", action="store_true", help="启用组件桥接")
    parser.add_argument("--min_comp_size_for_bridge", type=int, default=40, help="参与桥接的最小组件点数")
    parser.add_argument("--bridge_max_gap", type=float, default=0.12, help="桥接端点最大距离")
    parser.add_argument("--bridge_corridor_radius", type=float, default=0.02, help="桥接走廊半径")
    parser.add_argument("--bridge_prob_min", type=float, default=0.25, help="桥接候选点最小概率")
    parser.add_argument("--bridge_dir_cos_min", type=float, default=0.75, help="两组件方向余弦绝对值最小值")

    parser.add_argument("--enable_fill_holes", action="store_true", help="启用 kNN 补洞")
    parser.add_argument("--fill_k", type=int, default=16, help="补洞时的 k 近邻")
    parser.add_argument("--fill_pos_ratio", type=float, default=0.7, help="近邻中正类比例阈值")
    parser.add_argument("--fill_prob_min", type=float, default=0.2, help="补洞点最小概率")

    parser.add_argument("--enable_remove_sparse", action="store_true", help="启用稀疏离群点剔除")
    parser.add_argument("--sparse_radius", type=float, default=0.035, help="离群判定半径")
    parser.add_argument("--sparse_min_neighbors", type=int, default=3, help="半径内最小邻居数(不含自身)")

    parser.add_argument("--save_per_file_csv", action="store_true", help="保存每个样本的后处理csv")
    parser.add_argument("--summary_name", default="summary_postprocess_per_case.csv", help="逐样本汇总")
    parser.add_argument("--global_name", default="summary_postprocess_global.csv", help="全局汇总")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not files:
        raise SystemExit(f"未找到文件: {args.input_dir}/{args.pattern}")

    output_dir = args.output_dir if args.output_dir else os.path.join(args.input_dir, "postprocessed")
    os.makedirs(output_dir, exist_ok=True)

    rows: List[CaseMetrics] = []

    g_before_tp = g_before_tn = g_before_fp = g_before_fn = 0
    g_after_tp = g_after_tn = g_after_fp = g_after_fn = 0

    for f in files:
        df = pd.read_csv(f)
        required = {"x", "y", "z", "label", "prob"}
        if not required.issubset(df.columns):
            raise ValueError(f"{f} 缺少必须列: {required}")

        probs = df["prob"].to_numpy(dtype=np.float32)
        labels = (df["label"].to_numpy(dtype=np.float32) > 0.5).astype(np.int32)
        coords = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

        pred_before = (probs > args.fixed_thr).astype(np.int32)
        pred_after = postprocess_mask(coords, probs, args).astype(np.int32)

        btp, btn, bfp, bfn = calc_confusion(labels, pred_before)
        atp, atn, afp, afn = calc_confusion(labels, pred_after)

        m_before = calc_baseline_metrics_from_conf(btp, btn, bfp, bfn)
        m_after = calc_baseline_metrics_from_conf(atp, atn, afp, afn)

        g_before_tp += btp
        g_before_tn += btn
        g_before_fp += bfp
        g_before_fn += bfn
        g_after_tp += atp
        g_after_tn += atn
        g_after_fp += afp
        g_after_fn += afn

        row = CaseMetrics(
            file=os.path.basename(f),
            n_points=int(labels.size),
            pos_ratio=float(labels.mean()),
            pred_ratio_before=float(pred_before.mean()),
            pred_ratio_after=float(pred_after.mean()),
            hard_weld_iou_before=float(m_before["hard_weld_iou"]),
            hard_weld_iou_after=float(m_after["hard_weld_iou"]),
            hard_miou_before=float(m_before["hard_miou"]),
            hard_miou_after=float(m_after["hard_miou"]),
            hard_oa_before=float(m_before["hard_oa"]),
            hard_oa_after=float(m_after["hard_oa"]),
            hard_weld_f1_before=float(m_before["hard_weld_f1"]),
            hard_weld_f1_after=float(m_after["hard_weld_f1"]),
            hard_weld_precision_before=float(m_before["hard_weld_precision"]),
            hard_weld_precision_after=float(m_after["hard_weld_precision"]),
            hard_weld_recall_before=float(m_before["hard_weld_recall"]),
            hard_weld_recall_after=float(m_after["hard_weld_recall"]),
        )
        rows.append(row)

        if args.save_per_file_csv:
            out_df = df.copy()
            out_df["pred_before"] = pred_before
            out_df["pred_after"] = pred_after
            out_path = os.path.join(output_dir, os.path.basename(f).replace(".csv", "_post.csv"))
            out_df.to_csv(out_path, index=False)

        print(
            f"[{os.path.basename(f)}] "
            f"hard_weld_iou {m_before['hard_weld_iou']:.4f} -> {m_after['hard_weld_iou']:.4f}, "
            f"hard_miou {m_before['hard_miou']:.4f} -> {m_after['hard_miou']:.4f}, "
            f"hard_weld_f1 {m_before['hard_weld_f1']:.4f} -> {m_after['hard_weld_f1']:.4f}, "
            f"P {m_before['hard_weld_precision']:.4f} -> {m_after['hard_weld_precision']:.4f}, "
            f"R {m_before['hard_weld_recall']:.4f} -> {m_after['hard_weld_recall']:.4f}"
        )

    df_case = pd.DataFrame([r.__dict__ for r in rows]).sort_values("hard_weld_iou_after", ascending=False)
    df_case.to_csv(os.path.join(output_dir, args.summary_name), index=False)

    g_before = calc_baseline_metrics_from_conf(g_before_tp, g_before_tn, g_before_fp, g_before_fn)
    g_after = calc_baseline_metrics_from_conf(g_after_tp, g_after_tn, g_after_fp, g_after_fn)

    df_global = pd.DataFrame([
        {
            "n_files": len(rows),
            "n_points": int(df_case["n_points"].sum()),
            "fixed_thr": float(args.fixed_thr),
            "hard_weld_iou_before": float(g_before["hard_weld_iou"]),
            "hard_weld_iou_after": float(g_after["hard_weld_iou"]),
            "hard_weld_iou_gain": float(g_after["hard_weld_iou"] - g_before["hard_weld_iou"]),
            "hard_miou_before": float(g_before["hard_miou"]),
            "hard_miou_after": float(g_after["hard_miou"]),
            "hard_miou_gain": float(g_after["hard_miou"] - g_before["hard_miou"]),
            "hard_oa_before": float(g_before["hard_oa"]),
            "hard_oa_after": float(g_after["hard_oa"]),
            "hard_weld_f1_before": float(g_before["hard_weld_f1"]),
            "hard_weld_f1_after": float(g_after["hard_weld_f1"]),
            "hard_weld_precision_before": float(g_before["hard_weld_precision"]),
            "hard_weld_precision_after": float(g_after["hard_weld_precision"]),
            "hard_weld_recall_before": float(g_before["hard_weld_recall"]),
            "hard_weld_recall_after": float(g_after["hard_weld_recall"]),
            "tp_before": int(g_before_tp),
            "tn_before": int(g_before_tn),
            "fp_before": int(g_before_fp),
            "fn_before": int(g_before_fn),
            "tp_after": int(g_after_tp),
            "tn_after": int(g_after_tn),
            "fp_after": int(g_after_fp),
            "fn_after": int(g_after_fn),
        }
    ])
    df_global.to_csv(os.path.join(output_dir, args.global_name), index=False)

    print("\n=== GLOBAL SUMMARY (ALIGNED) ===")
    print(f"hard_weld_iou: {g_before['hard_weld_iou']:.6f} -> {g_after['hard_weld_iou']:.6f}")
    print(f"hard_miou    : {g_before['hard_miou']:.6f} -> {g_after['hard_miou']:.6f}")
    print(f"hard_oa      : {g_before['hard_oa']:.6f} -> {g_after['hard_oa']:.6f}")
    print(f"hard_weld_f1 : {g_before['hard_weld_f1']:.6f} -> {g_after['hard_weld_f1']:.6f}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
