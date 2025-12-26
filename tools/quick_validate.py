import os
import sys
import torch
import numpy as np
# ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train import finetune
from train.finetune import WeldDataset, collate_fn
from model.model import GeometryAwareTransformer
from utils.config import Config


def quick_validate():
    config = Config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labeled_files = [os.path.join(config.LABEL_DATA_DIR, f)
                     for f in os.listdir(config.LABEL_DATA_DIR)
                     if f.endswith('.npz') and '_pred' not in f]
    if len(labeled_files) == 0:
        print('No labeled files found in', config.LABEL_DATA_DIR)
        return
    loader = torch.utils.data.DataLoader(WeldDataset(labeled_files), batch_size=getattr(config,'BATCH_SIZE',8),
                                         shuffle=False, collate_fn=collate_fn, num_workers=2)
    model = GeometryAwareTransformer(config).to(device)
    # load pretrained or finetuned if available
    model_path = getattr(config, 'WEIGHTS_TO_TEST', None) or 'weights/best_finetune.pth'
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print('Loaded model from', model_path)
    else:
        print('Model path not found:', model_path, '\nUsing randomly initialized classifier (may be poor).')

    model.eval()
    # build a criterion similar to training (use FocalLoss if available in config)
    use_focal = getattr(Config, 'USE_FOCAL_LOSS', True)
    if use_focal:
        # compute alpha as in finetune
        pos, neg = 0, 0
        for p in labeled_files:
            labs = np.load(p)['labels'].reshape(-1)
            valid = labs >= 0
            pos += int(((labs == 1) & valid).sum())
            neg += int(((labs == 0) & valid).sum())
        alpha = pos / (pos + neg + 1e-8)
        criterion = finetune.FocalLoss(alpha=alpha, gamma=getattr(Config,'FOCAL_GAMMA',2.0), reduction='none')
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    print('Running validate() from finetune with constructed criterion...')
    val_loss, metrics = finetune.validate(model, loader, device, criterion)
    print('validate() returned -> val_loss:', val_loss, 'metrics:', metrics)

if __name__ == '__main__':
    quick_validate()
