import os
import sys
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train.finetune import WeldDataset, collate_fn
from model.model import GeometryAwareTransformer
from utils.config import Config


def per_file_eval(threshold=None):
    config = Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    files = [os.path.join(config.LABEL_DATA_DIR, f) for f in os.listdir(config.LABEL_DATA_DIR) if f.endswith('.npz') and '_pred' not in f]
    model = GeometryAwareTransformer(config).to(device)
    model_path = getattr(config, 'WEIGHTS_TO_TEST', None) or 'weights/best_finetune.pth'
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print('Loaded model:', model_path)
    else:
        print('Model not found:', model_path)
        return
    model.eval()
    rows = []
    # if threshold not provided, compute per-file optimal threshold by searching 0.01..0.5
    for path in files:
        data = np.load(path)
        feats = data['features'].astype(np.float32)
        labels = data['labels'].reshape(-1)
        if (labels >= 0).sum() == 0:
            continue
        # chunk inference to avoid OOM for large point clouds
        max_pts = getattr(Config, 'MAX_POINTS', 4096)
        N = feats.shape[0]
        probs = np.zeros(N, dtype=np.float32)
        with torch.no_grad():
            if N <= max_pts:
                f = torch.from_numpy(feats[None]).to(device)
                pd = torch.from_numpy(data['principal_dir'].astype(np.float32)[None]).to(device)
                curv = torch.from_numpy(data['curvature'].astype(np.float32)[None]).to(device)
                den = torch.from_numpy(data['local_density'].astype(np.float32)[None]).to(device)
                nor = torch.from_numpy(data['normals'].astype(np.float32)[None]).to(device)
                lin = torch.from_numpy(data['linearity'].astype(np.float32)[None]).to(device)
                logits = model(f, pd, curv, den, nor, lin, task='class')
                probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            else:
                num_chunks = (N + max_pts - 1) // max_pts
                for c in range(num_chunks):
                    s = c * max_pts
                    e = min(N, (c + 1) * max_pts)
                    f = torch.from_numpy(feats[s:e][None]).to(device)
                    pd = torch.from_numpy(data['principal_dir'].astype(np.float32)[s:e][None]).to(device)
                    curv = torch.from_numpy(data['curvature'].astype(np.float32)[s:e][None]).to(device)
                    den = torch.from_numpy(data['local_density'].astype(np.float32)[s:e][None]).to(device)
                    nor = torch.from_numpy(data['normals'].astype(np.float32)[s:e][None]).to(device)
                    lin = torch.from_numpy(data['linearity'].astype(np.float32)[s:e][None]).to(device)
                    logits = model(f, pd, curv, den, nor, lin, task='class')
                    probs[s:e] = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        mask = (labels >= 0)
        probs_m = probs[mask]
        labs_m = labels[mask]
        if threshold is None:
            best_f1 = -1
            best_t = 0.5
            for t in np.linspace(0.01,0.5,50):
                preds = (probs_m > t).astype(int)
                try:
                    f1 = f1_score(labs_m, preds, pos_label=1)
                except Exception:
                    f1 = 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
        else:
            best_t = threshold
            preds = (probs_m > best_t).astype(int)
            try:
                best_f1 = f1_score(labs_m, preds, pos_label=1)
            except Exception:
                best_f1 = 0.0
        # compute precision/recall at best_t
        preds = (probs_m > best_t).astype(int)
        prec = precision_score(labs_m, preds, zero_division=0)
        rec = recall_score(labs_m, preds, zero_division=0)
        rows.append((os.path.basename(path), int((labs_m==1).sum()), int((labs_m==0).sum()), float(best_t), float(best_f1), float(prec), float(rec)))
    # sort by f1 ascending
    rows = sorted(rows, key=lambda x: x[4])
    out = os.path.join('logs', 'per_file_eval.csv')
    os.makedirs('logs', exist_ok=True)
    with open(out, 'w') as fh:
        fh.write('file,pos,neg,best_t,f1,prec,rec\n')
        for r in rows:
            fh.write(','.join(map(str,r)) + '\n')
    print('Wrote per-file eval to', out)

if __name__ == '__main__':
    per_file_eval()
