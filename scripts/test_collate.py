import sys, os
proj = r"d:\work\研一\welding\welding seam extraction\weld_seg_project"
if proj not in sys.path:
    sys.path.insert(0, proj)

from train.train import WeldDataset, collate_fn
from torch.utils.data import DataLoader
import glob

files = glob.glob(os.path.join(proj, 'data', 'processed', '*.npz'))
if len(files) < 2:
    print('Need at least 2 .npz files to test, found', len(files))
    raise SystemExit(1)

ds = WeldDataset(files[:2])
dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

batch = next(iter(dl))
for k,v in batch.items():
    try:
        print(k, type(v), getattr(v, 'shape', None))
    except Exception as e:
        print('Error printing', k, e)

print('OK')
