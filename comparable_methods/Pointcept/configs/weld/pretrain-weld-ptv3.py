_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 2
num_worker = 4
mix_prob = 0.0
empty_cache = False
enable_amp = True
evaluate = False

# model settings
model = pretrain_model = dict(
    type="DefaultReconstructorV2",
    recon_channels=6,
    backbone_out_channels=64,
    recon_keys=("curvature", "density", "linearity", "principal_dir"),
    recon_use_mask=True,
    loss_weight=1.0,
    backbone=dict(
        type="PT-v3m1",
        in_channels=12,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        enable_flash=False,
        upcast_attention=False,
        upcast_softmax=False,
        geom_bias_cfg=dict(
            enable=True,
            alpha=1.0,
            gamma1=1.0,
            gamma2=1.0,
            gamma3=1.0,
            gamma4=1.0,
            sigma_kappa=1.0,
            density_scale=1.0,
            enable_gate=True,
            gate_a1=2.0,
            gate_a2=0.5,
            gate_a3=1.0,
            gate_alpha=1.0,
            gate_beta=1.0,
            w_min=0.005,
            w_max=0.02,
            gate_scale=1.0,
        ),
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("Weld"),
    ),
)

# scheduler settings
epoch = 600

# hooks (disable seg evaluator for recon pretrain)
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="CheckpointSaver", save_freq=None),
]
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# dataset settings
dataset_type = "DefaultDataset"
data_root = "data/weld"

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["background", "weld"],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="BlockReconMask",
                target_points_per_block=1000,
                high_curv_threshold=0.01,
                min_high_curv_points=5,
                align_grid=True,
                grid_align_base=0.001,
                mask_ratio=0.3,
                strategy="mixed",
                weld_mask_ratio=None,
                bg_mask_ratio=None,
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=204800, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "normal",
                    "curvature",
                    "density",
                    "linearity",
                    "principal_dir",
                    "recon_mask",
                ),
                feat_keys=(
                    "color",
                    "normal",
                    "curvature",
                    "density",
                    "linearity",
                    "principal_dir",
                ),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="BlockReconMask", mask_ratio=0.0),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "normal",
                    "curvature",
                    "density",
                    "linearity",
                    "principal_dir",
                    "recon_mask",
                ),
                feat_keys=(
                    "color",
                    "normal",
                    "curvature",
                    "density",
                    "linearity",
                    "principal_dir",
                ),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="test",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="BlockReconMask", mask_ratio=0.0),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=(
                    "coord",
                    "grid_coord",
                    "curvature",
                    "density",
                    "linearity",
                    "principal_dir",
                    "recon_mask",
                ),
                feat_keys=(
                    "color",
                    "normal",
                    "curvature",
                    "density",
                    "linearity",
                    "principal_dir",
                ),
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.02,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
                return_inverse=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="NormalizeColor"),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "normal", "recon_mask"),
                    feat_keys=(
                        "color",
                        "normal",
                        "curvature",
                        "density",
                        "linearity",
                        "principal_dir",
                    ),
                ),
            ],
            aug_transform=[[dict(type="RandomScale", scale=[0.95, 1.05])]],
        ),
    ),
)
