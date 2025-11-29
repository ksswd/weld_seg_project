# debug_mask_scan.py
import numpy as np, torch, os, glob
from utils.config import Config
from train.mask_strategy import HighCurvatureMasker

files = sorted([os.path.join(Config.PROCESSED_DATA_DIR, f) for f in os.listdir(Config.PROCESSED_DATA_DIR) if f.endswith('.npz')])
masker = HighCurvatureMasker(mask_ratio=getattr(Config, 'MASK_RATIO', 0.1))
bad = []
for f in files:
    d = np.load(f)
    if 'curvature' not in d.files:
        bad.append((f, 'no curvature'))
        continue
    cur = d['curvature'].reshape(1, -1, 1).astype(np.float32)
    cur_t = torch.from_numpy(cur)
    m = masker.generate_mask(cur_t)
    if m.dim()==3 and m.size(-1)==1:
        m = m.squeeze(-1)
    if not m.any():
        bad.append((f, 'mask empty'))
print('checked', len(files), 'files, bad count', len(bad))
for p, reason in bad[:20]:
    print(p, reason)