import os, sys
proj = r"d:\work\研一\welding\welding seam extraction\weld_seg_project"
if proj not in sys.path:
    sys.path.insert(0, proj)

from utils.config import Config
from train.train import WeldDataset, collate_fn
from torch.utils.data import DataLoader
import glob
import torch
from model.model import GeometryAwareTransformer

print('Starting smoke test')

files = glob.glob(os.path.join(proj, 'data', 'processed', '*.npz'))
print('Found npz files:', len(files))
if len(files) == 0:
    raise SystemExit('No processed files found')

cfg = Config()

# Use small batch
ds = WeldDataset(files[:4])
dl = DataLoader(ds, batch_size=2, collate_fn=collate_fn)

print('Created DataLoader; attempting to get one batch...')
for i, batch in enumerate(dl):
    print('Got batch', i)
    for k,v in batch.items():
        print(k, type(v), getattr(v, 'shape', None))
    # instantiate model
    print('Instantiating model...')
    model = GeometryAwareTransformer(cfg)
    print('Model instantiated; running forward on CPU...')
    with torch.no_grad():
        out = model(batch['features'], batch['principal_dir'], batch['curvature'], batch['local_density'], batch['normals'], batch['linearity'])
    print('Forward done. output shape:', out.shape)
    break

print('Smoke test finished')
