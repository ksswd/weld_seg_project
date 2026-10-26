# Weld segmentation source snapshot

This repository contains the current geometry-aware line-laser weld segmentation source and local baseline adaptations. This is a **source-only backup**, not a new experimental release or evidence of measured performance.

## Source layout

- main.py: command-line entry point.
- model/, train/, preprocess/, utils/: network, training and data-processing code.
- test/, scripts/, postprocess/: evaluation, visualization and post-processing.
- comparable_methods/: ordinary source snapshots of baseline implementations, including local weld adaptations. Nested .git directories are not included.
- data/: where present in Git, only selected processing source files are included; actual datasets are excluded.

Dependencies are recorded in environment.yml and requirements.txt. Existing configuration and run scripts contain server-specific paths; review utils/config.py and shell scripts before running on a different machine.

## Baseline provenance and licenses

The local baseline copies include modifications and additional files beyond their upstream commits. Both Pointcept copies are retained because their local adaptations differ. Original LICENSE/NOTICE files are preserved; the licenses listed here apply to the corresponding upstream sources, not a blanket relicensing of this project.

| Source snapshot | Upstream | Upstream commit | License |
|---|---|---|---|
| Pointnet_Pointnet2_pytorch | https://github.com/yanx27/Pointnet_Pointnet2_pytorch | eb64fe0b4c24055559cea26299cb485dcb43d8dd | MIT |
| Pointcept | https://github.com/Pointcept/Pointcept | 04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2 | MIT |
| Pointcept-copy | https://github.com/Pointcept/Pointcept | 04a0232b70f5c7091ffdc6bfe7a476e3eb7daff2 | MIT |
| pointMLP-pytorch | https://github.com/ma-xu/pointMLP-pytorch | 1ab7f04e88d2aeb8fd16921a2b2fc4fba950b6df | Apache-2.0 |
| dgcnn.pytorch | https://github.com/antao97/dgcnn.pytorch | 07d534c2702905010ec9991619f552d8cacae45b | MIT |
| pointconv_pytorch | https://github.com/DylanWusee/pointconv_pytorch | 4a01ee9e1742d4b837ab0ef44896be4e31c63378 | MIT |

## Excluded artifacts

No datasets, checkpoints, weights, prediction point clouds, generated metrics, plots, cache files or nested Git metadata are added in this sync. Local manuscript files and hypothetical/simulated result tables are not included. Existing repository history is not rewritten.

## Known pre-existing syntax issues

Static checking of the source snapshot on 2026-10-26 found these existing Python syntax errors:

- comparable_methods/Pointcept/pointcept/models/default.py:90: empty else block.
- scripts/csv_fix.py:117: empty for block.
- test/recon_test.py:191: empty if block.

These files are preserved as-is for backup. This sync does not repair model logic, run training, or establish that all baseline methods execute successfully. Other selected Python files passed parsing and selected shell scripts passed bash -n; dependency installation, CUDA compilation and runtime correctness are not verified by these checks.

## Snapshot policy

The snapshot follows the existing backup/full-workspace-20260306 branch and preserves the current source state, including previously deleted tracked scripts. Publication-grade experiment claims should be based on their own actual logs and predictions, not this source backup.

