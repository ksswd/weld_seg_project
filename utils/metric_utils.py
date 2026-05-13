# weld_seg_project/utils/metric_utils.py 评估指标
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def calculate_metrics(predictions, labels):
    """
    计算分类任务的基本指标。
    :param predictions: 模型预测的标签 (N,)
    :param labels: 真实标签 (N,)
    :return: 一个包含指标的字典
    """
    # 确保输入是一维数组
    predictions = predictions.flatten().astype(np.int32)
    labels = labels.flatten().astype(np.int32)

    # 计算准确率
    acc = accuracy_score(labels, predictions)

    # 计算F1（关注焊缝正类=1）
    f1 = f1_score(labels, predictions, average="binary", pos_label=1)

    # 计算二类 IoU 与 mIoU
    tp = int(((labels == 1) & (predictions == 1)).sum())
    tn = int(((labels == 0) & (predictions == 0)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    iou_bg = tn / max(tn + fp + fn, 1)
    iou_weld = tp / max(tp + fp + fn, 1)
    miou = 0.5 * (iou_bg + iou_weld)

    return {
        "accuracy": float(acc),
        "f1_score": float(f1),
        "miou": float(miou),
        "iou_bg": float(iou_bg),
        "iou_weld": float(iou_weld),
    }