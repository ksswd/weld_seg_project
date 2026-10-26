import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===================== 配置 =====================
CSV_PATH = "miou_epochs_results.csv"
MAX_EPOCH = 200
LINE_WIDTH = 2
FONT_SIZE = 12
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
# ==================================================

# 读取CSV
df = pd.read_csv(CSV_PATH)

# 按 fold 分组画图
for fold in [1, 2]:
    # 筛选当前 fold 的所有方法
    fold_rows = df[df["Method"].str.endswith(f"_fold{fold}")]
    if fold_rows.empty:
        print(f"ℹ️ Fold{fold} 无数据，跳过")
        continue

    plt.rcParams.update({
        "font.size": FONT_SIZE,
        "figure.figsize": (10, 6),
        "axes.spines.top": False,
        "axes.spines.right": False
    })
    plt.figure()

    epochs = np.arange(1, MAX_EPOCH + 1)

    # 绘制每个方法
    for i, (_, row) in enumerate(fold_rows.iterrows()):
        method_name = row["Method"].replace(f"_fold{fold}", "")
        miou = row[1:MAX_EPOCH+1].values.astype(float)

        plt.plot(
            epochs, miou,
            label=method_name,
            linewidth=LINE_WIDTH,
            color=colors[i % len(colors)]
        )

    plt.xlabel("Epoch")
    plt.ylabel("mIoU / ins.IoU")
    plt.title(f"mIoU Training Curve - Fold {fold}")
    plt.xlim(1, MAX_EPOCH)
    plt.grid(alpha=0.3, linestyle="--")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()

    # 保存图片
    plt.savefig(f"miou_curve_fold{fold}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ Fold {fold} 图已保存：miou_curve_fold{fold}.png")