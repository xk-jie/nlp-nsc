import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    roc_curve,
    auc,
    matthews_corrcoef
)

import torch

from configs.config import Config
from utils.helper import id_to_label

config = Config.yaml_config()

def search_f1_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), f'Error with label length {len(y_true)} vs {len(y_pred)}'

    # 根据 classification report 获取 B-ASP 的 f1_score 值
    score = search_cls_rep(y_true, y_pred)

    return score

def search_acc_score(y_true, y_pred):
    assert len(y_true) == len(y_pred), f'Error with label length {len(y_true)} vs {len(y_pred)}'

    score = accuracy_score(y_true, y_pred)
    print('acc', score)
    return score

def search_threshold(y_true, y_pred):
    assert len(y_true) == len(y_pred), f'Error with label length {len(y_true)} vs {len(y_pred)}'

    scores = []
    thresholds = [i / 100 for i in range(100)]
    for threshold in thresholds:
        y_pred = (y_pred > threshold).astype(int)
        score = f1_score(y_true, y_pred, average='macro')
        scores.append(score)

    threshold = thresholds[np.argmax(scores)]

    return threshold

def search_cls_rep(y_true, y_pred):
    # 标签进行转换， 同时剔除 [CLS] 的相应值
    y_true, y_pred = id_to_label(y_true, y_pred)
    cls_rep = classification_report(y_true, y_pred)

    # 取第二行的 f1_score, 第一行为 head
    score = round(float(cls_rep.split()[7]), 2)

    return score

def search_f1_acc(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, labels=[0, 1, 2], average='macro')

    return score

def search_f1_auc(y_true, y_pred):
    """
    as the metrics when save model
    """
    P = y_true.sum()
    R = y_pred.sum()
    TP = ((y_true + y_pred) > 1).sum()

    pre = TP / P
    rec = TP / R

    return 2 * (pre * rec) / (pre + rec)

def search_auc(y_true, y_pred):
    """
    fp: rarray, shape = [>2]
        Increasing false positive rates such that element i is the false positive rate
        of predictions with score >= thresholds[i].
    tpr: array, shape = [>2]
        Increasing true positive rates such that element i is the true positive rate
        of predictions with score >= thresholds[i].
    thresholds: array, shape = [n_thresholds]
        Decreasing thresholds on the decision function used to compute fpr and tpr.
        thresholds[0] represents no instances being predicted and is arbitrarily set to max(y_pred) + 1.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return auc(fpr, tpr)

def map(y_true, y_pred):
    """
    map: mean of ap
    """
    def ap(y_true, y_pred):
        """
        ap: average precision
        指的是在各个召回率上的正确率的平均值
        """
        y_true = torch.tensor(y_true, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        _, idx = torch.sort(y_pred, descending=True)
        y_true = y_true[idx].round()

        total = 0.
        for i in range(len(y_pred)):
            j = i + 1
            if y_true[i]:
                total += y_true[:j].sum().item() / j

        true_sum = y_true.sum()
        if true_sum != 0.:
            return total / true_sum.item()

        return 0.

    assert len(y_true) == len(y_pred), f'Error with label length {len(y_true)} vs {len(y_pred)}'

    res = []
    res.append(ap(y_true, y_pred))

    return np.mean(res)

def mrr(y_true, y_pred):
    """
    mrr: mean of rr
    根据rr, 再对所有的问题取平均
    """
    def rr(y_true, y_pred):
        """
        rr: reciprocal rank
        把标准答案在被评价系统给出结果的排序取倒数作为准确度
        """
        y_true = torch.tensor(y_true, dtype=torch.float32)
        y_pred = torch.tensor(y_pred, dtype=torch.float32)

        _, idx = torch.sort(y_pred, descending=True)
        best = y_true[idx].nonzero().squeeze().min().item()

        return 1.0 / (best + 1)

    assert len(y_true) == len(y_pred), f'Error with label length {len(y_true)} vs {len(y_pred)}'

    res = []
    res.append(rr(y_true, y_pred))

    return np.mean(res)

def search_mcc(y_true, y_pred):
    """
    mcc: Matthews correlation coefficient
    The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications
    Binary and multiclass labels are supported. Only in the binary case does this relate to information about true and false positives and negatives
    指标：它的取值范围为[-1,1]，取值为1时表示对受试对象的完美预测，取值为0时表示预测的结果还不如随机预测的结果，-1是指预测分类和实际分类完全不一致
    """
    mcc = matthews_corrcoef(y_true, y_pred)

    return mcc
