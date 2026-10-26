import re
import csv
from pathlib import Path

# ====================== 你只需要改这里 ======================
# 格式：
# "方法名": {
#     "log_path": "日志文件路径",
#     "pattern": 正则表达式（稍后你填）
# }

METHODS = {
    "Ours": {
        "log_path": "job.70989.out",
        "pattern": r"Val\(fps\).*?mIoU: (\d+\.\d+)", # 稍后你填正则]
        "folds": 2
    },
    "PointNet++": {
        "log_path": "comparable_methods/Pointnet_Pointnet2_pytorch/job.70679.out",
        "pattern": r"val_miou=(\d+\.\d+)",
        "folds": 3
    },
    "DGCNN": {
        "log_path": "comparable_methods/dgcnn.pytorch/dgcnn.out",
        "pattern": r"val_miou=(\d+\.\d+)",
        "folds": 3
    },
    # "Point Transformer v3":: {
    #     "log_path": "logs/dgcnn.log",
    #     "pattern": r""
    # },
    "PointMLP": {
        "log_path": "comparable_methods/pointMLP-pytorch/job.70680.out",
        "pattern": r"train ins_iou: (\d+\.\d+)",
        "folds": 3
    },
    "Pointconv": {
        "log_path": "comparable_methods/pointconv_pytorch/job.70674.out",
        "pattern": r"val_miou=(\d+\.\d+)",
        "folds": 3
    },
}

OUTPUT_CSV = "miou_epochs_results.csv"
MAX_EPOCH = 200
# ==========================================================

def extract_miou_from_log(log_path, pattern):
    """从日志提取所有miou（按出现顺序：fold1全部epoch → fold2 → fold3）"""
    if not Path(log_path).exists():
        print(f"⚠️ 警告：文件不存在 {log_path}")
        return []
    
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    
    results = re.findall(pattern, text)
    miou_list = []
    for val in results:
        try:
            miou_list.append(float(val))
        except:
            pass
    return miou_list

def split_folds(miou_list, num_folds, max_epoch):
    """把一整串miou切成 fold1 / fold2 / fold3，每段200epoch"""
    folds = []
    for i in range(num_folds):
        start = i * max_epoch
        end = start + max_epoch
        fold_data = miou_list[start:end]

        # 不足200补空
        while len(fold_data) < max_epoch:
            fold_data.append("")
        folds.append(fold_data)
    return folds

def write_csv():
    # 表头
    header = ["Method"] + [f"Epoch_{i}" for i in range(1, MAX_EPOCH+1)]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # 遍历每个方法
        for method, cfg in METHODS.items():
            print(f"📦 处理：{method}")
            miou_list = extract_miou_from_log(cfg["log_path"], cfg["pattern"])
            folds = split_folds(miou_list, cfg["folds"], MAX_EPOCH)

            # 每个fold写一行：方法名_fold1 / 方法名_fold2...
            for idx, fold_data in enumerate(folds):
                fold_name = f"{method}_fold{idx+1}"
                row = [fold_name] + fold_data
                writer.writerow(row)

    print(f"\n✅ 全部完成！CSV 已保存到：{OUTPUT_CSV}")

if __name__ == "__main__":
    write_csv()