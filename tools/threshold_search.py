import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train.finetune import WeldDataset, collate_fn
from model.model import GeometryAwareTransformer
from utils.config import Config


def gather_probs():
    config = Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_files = [os.path.join(config.LABEL_DATA_DIR, f)
                     for f in os.listdir(config.LABEL_DATA_DIR)
                     if f.endswith('.npz') and '_pred' not in f]
    loader = torch.utils.data.DataLoader(WeldDataset(labeled_files), batch_size=getattr(config,'BATCH_SIZE',8),
                                         shuffle=False, collate_fn=collate_fn, num_workers=2)
    model = GeometryAwareTransformer(config).to(device)
    model_path = getattr(config, 'WEIGHTS_TO_TEST', None) or 'weights/best_finetune.pth'
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print('Loaded model from', model_path)
    else:
        print('Model path not found:', model_path)
    model.eval()
    probs_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch['features'], batch['principal_dir'], batch['curvature'],
                           batch['local_density'], batch['normals'], batch['linearity'], task='class')
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            labs = batch['labels'].cpu().numpy().reshape(-1)
            mask = (labs >= 0)
            if mask.any():
                probs_all.append(probs[mask])
                labels_all.append(labs[mask])
    if len(probs_all)==0:
        print('No labeled points found')
        return None, None
    probs_all = np.concatenate(probs_all)
    labels_all = np.concatenate(labels_all)
    return probs_all, labels_all


def search():
    probs, labels = gather_probs()
    if probs is None:
        return
    best_f1 = -1
    best_t = None
    for t in np.linspace(0.1, 0.8, 71):
        preds = (probs > t).astype(np.int32)
        f1 = f1_score(labels, preds, pos_label=1)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    print('best threshold:', best_t, 'best f1:', best_f1)
    # print baseline at 0.5
    base_preds = (probs > 0.5).astype(int)
    print('f1 at 0.5:', f1_score(labels, base_preds, pos_label=1), 'accuracy:', accuracy_score(labels, base_preds))

if __name__ == '__main__':
    search()
