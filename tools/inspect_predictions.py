import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train.finetune import WeldDataset, collate_fn
from model.model import GeometryAwareTransformer
from utils.config import Config


def inspect_one(sample_idx=0, topk=500, threshold=None):
    config = Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = [os.path.join(config.LABEL_DATA_DIR, f)
             for f in os.listdir(config.LABEL_DATA_DIR) if f.endswith('.npz') and '_pred' not in f]
    if len(files) == 0:
        print('No labeled files')
        return
    if sample_idx >= len(files):
        print('sample_idx out of range', sample_idx, 'len', len(files))
        return
    path = files[sample_idx]
    data = np.load(path)
    feats = data['features'].astype(np.float32)
    labels = data['labels'].reshape(-1)
    print('File:', os.path.basename(path))
    print('N points:', feats.shape[0])
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    unknown = np.where(labels < 0)[0]
    print('pos/neg/unknown counts:', len(pos), len(neg), len(unknown))

    model = GeometryAwareTransformer(config).to(device)
    model_path = getattr(config, 'WEIGHTS_TO_TEST', None) or 'weights/best_finetune.pth'
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print('Loaded model:', model_path)
    else:
        print('Model not found:', model_path)
        return
    model.eval()

    # prepare batch (no padding needed)
    f = torch.from_numpy(feats[None]).to(device)
    pd = torch.from_numpy(data['principal_dir'].astype(np.float32)[None]).to(device)
    curv = torch.from_numpy(data['curvature'].astype(np.float32)[None]).to(device)
    den = torch.from_numpy(data['local_density'].astype(np.float32)[None]).to(device)
    nor = torch.from_numpy(data['normals'].astype(np.float32)[None]).to(device)
    lin = torch.from_numpy(data['linearity'].astype(np.float32)[None]).to(device)

    with torch.no_grad():
        # If a curvature gain is set in Config, apply it to the feature vector to match training
        curv_gain = getattr(Config, 'CURV_GAIN', None)
        f_in = f.clone()
        if curv_gain is not None:
            try:
                f_in[:, :, 6] = f_in[:, :, 6] * float(curv_gain)
            except Exception:
                pass

        logits = model(f_in, pd, curv, den, nor, lin, task='class')
        probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)

    # pick threshold if not provided: use median of positive probs or low quantile
    if threshold is None:
        # heuristic: if many small probs, choose 0.1 or choose median pos prob
        if len(pos) > 0:
            median_pos = np.median(probs[pos])
            threshold = max(0.01, min(0.5, median_pos))
        else:
            threshold = 0.5
    print('Using threshold:', threshold)

    preds_bin = (probs > threshold).astype(np.int32)
    tp = np.logical_and(preds_bin == 1, labels == 1).sum()
    fp = np.logical_and(preds_bin == 1, labels == 0).sum()
    fn = np.logical_and(preds_bin == 0, labels == 1).sum()
    tn = np.logical_and(preds_bin == 0, labels == 0).sum()
    print('TP/FP/FN/TN:', tp, fp, fn, tn)
    if len(pos) > 0:
        recall = tp / (tp + fn + 1e-12)
        prec = tp / (tp + fp + 1e-12)
        f1 = 2 * prec * recall / (prec + recall + 1e-12)
        print(f'precision={prec:.4f}, recall={recall:.4f}, f1={f1:.4f}')

    # show topk predicted points and whether they are true positive
    idx_sorted = np.argsort(-probs)
    topk_idx = idx_sorted[:topk]
    tp_in_topk = np.intersect1d(topk_idx, pos)
    print(f'top{topk} contains {len(tp_in_topk)} true positives out of {len(pos)} positives')

    # compare feature stats between predicted and true points
    def stats(idx):
        s = feats[idx]
        return dict(mean=s.mean(axis=0), std=s.std(axis=0))
    print('Feature mean (all):', feats.mean(axis=0))
    if len(pos) > 0:
        print('Feature mean (true pos):', stats(pos)['mean'])
    print('Feature mean (pred pos):', stats(np.where(preds_bin==1)[0])['mean'] if preds_bin.sum()>0 else 'none')

    # list some indices of mismatch (FPs and FNs)
    fp_idx = np.where(np.logical_and(preds_bin==1, labels==0))[0]
    fn_idx = np.where(np.logical_and(preds_bin==0, labels==1))[0]
    print('Some FP indices:', fp_idx[:10])
    print('Some FN indices:', fn_idx[:10])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--sample', type=int, default=0)
    p.add_argument('--topk', type=int, default=500)
    p.add_argument('--threshold', type=float, default=None)
    args = p.parse_args()
    inspect_one(sample_idx=args.sample, topk=args.topk, threshold=args.threshold)
