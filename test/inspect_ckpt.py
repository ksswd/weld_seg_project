import os
import torch
import numpy as np
from model.model import GeometryAwareTransformer
from utils.config import Config


def summarize_state_dict(sd, prefix=''):
    stats = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            arr = v.detach().cpu().numpy()
            stats[prefix + k] = {
                'shape': arr.shape,
                'mean': float(arr.mean()),
                'std': float(arr.std()),
                'l2': float((arr**2).sum())
            }
    return stats


def compare_ckpts(path_a, path_b):
    a = torch.load(path_a, map_location='cpu')
    b = torch.load(path_b, map_location='cpu')

    keys = set(a.keys()) & set(b.keys())
    print(f"Common keys: {len(keys)}")

    diffs = {}
    for k in sorted(keys):
        va = a[k].detach().cpu().numpy()
        vb = b[k].detach().cpu().numpy()
        diff = vb - va
        diffs[k] = {
            'shape': va.shape,
            'a_mean': float(va.mean()),
            'b_mean': float(vb.mean()),
            'mean_diff': float(diff.mean()),
            'l2_diff': float((diff**2).sum())
        }

    # Print summary for classifier params first
    classifier_keys = [k for k in diffs.keys() if k.startswith('classifier')]
    print('\nClassifier parameter differences:')
    for k in classifier_keys:
        d = diffs[k]
        print(f"{k}: shape={d['shape']} a_mean={d['a_mean']:.6e} b_mean={d['b_mean']:.6e} mean_diff={d['mean_diff']:.6e} l2_diff={d['l2_diff']:.6e}")

    # Print small summary for a few backbone keys
    print('\nSome backbone parameter diffs (first 10):')
    backbone_keys = [k for k in diffs.keys() if not k.startswith('classifier')]
    for k in backbone_keys[:10]:
        d = diffs[k]
        print(f"{k}: shape={d['shape']} mean_diff={d['mean_diff']:.6e} l2_diff={d['l2_diff']:.6e}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_a', help='Path to first checkpoint')
    parser.add_argument('ckpt_b', help='Path to second checkpoint')
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt_a) or not os.path.isfile(args.ckpt_b):
        raise SystemExit('Please provide two valid checkpoint paths')

    compare_ckpts(args.ckpt_a, args.ckpt_b)
