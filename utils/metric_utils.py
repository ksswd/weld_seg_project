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
    predictions = predictions.flatten()
    labels = labels.flatten()
    # print(labels)
    # unique, counts = np.unique(labels, return_counts=True)
    # label_counts = dict(zip(unique, counts))
    # print(label_counts)

    # 计算准确率 (Accuracy)
    acc = accuracy_score(labels, predictions)

    # 计算F1分数 (F1-Score)，适用于不平衡数据集
    # 'binary' 表示二分类，'pos_label'=1表示我们关注的是焊缝(1)的F1分数
    f1 = f1_score(labels, predictions, average='binary', pos_label=1)

    return {
        'accuracy': acc,
        'f1_score': f1
    }