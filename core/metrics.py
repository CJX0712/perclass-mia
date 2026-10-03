"""纯 numpy 指标计算（作者：晨星）。

不依赖 sklearn 即可复现 ROC-AUC / TPR@FPR，保证离线兜底下指标口径与
可选 sklearn 后端完全一致（单测交叉验证）。
"""
from __future__ import annotations

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC，分数越大越判为阳性（member）。

    采用排序 + 梯形法，对并列分数取平均位置（与 sklearn 一致）。
    当某类为空（全部 member 或全部 non-member）时退化为 0.5（无区分信息）。
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if scores.shape != labels.shape or scores.size == 0:
        raise ValueError("scores 与 labels 形状必须一致且非空")
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    l_sorted = labels[order]

    # 累计：沿升序扫描，统计每个阈值下的 TPR/FPR
    # 用 rank 法（Mann-Whitney U）更稳：AUC = (sum_rank_pos - n_pos*(n_pos+1)/2)/ (n_pos*n_neg)
    # 对并列分数做平均秩。
    # 重新实现：按分数降序赋平均秩。
    # 简单稳健实现：基于 ranks
    ranks = _average_ranks(s_sorted)
    # MWU: AUC = (R_pos - n_pos*(n_pos+1)/2) / (n_pos*n_neg)，R_pos = sum of ranks of positives
    r_pos = float(ranks[l_sorted == 1].sum())
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _average_ranks(sorted_vals: np.ndarray) -> np.ndarray:
    """对升序数组返回平均秩（1-based）。"""
    n = sorted_vals.size
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based 平均秩
        ranks[i:j + 1] = avg
        i = j + 1
    return ranks


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr_target: float = 0.1) -> float:
    """在指定 FPR 下的最大 TPR（越大越好）。"""
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.int64).ravel()
    n_neg = max(int((1 - labels).sum()), 1)
    order = np.argsort(-scores, kind="mergesort")  # 分数降序
    tp = 0
    fp = 0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        if fp / n_neg >= fpr_target:
            return tp / max(int(labels.sum()), 1)
    return tp / max(int(labels.sum()), 1)


def paired_one_sided_t(diffs: np.ndarray) -> tuple[float, float]:
    """配对单侧 t 检验（H1: 均值差 > 0）。

    返回 (mean_diff, t_stat)。用于 DoD 显著性：同 seed 下旗舰 AUC − 基线 AUC 的配对差，
    检验是否显著 > 0。t ≥ 临界值（df=n-1, α=0.05 单侧）即显著。
    """
    diffs = np.asarray(diffs, dtype=np.float64).ravel()
    n = diffs.size
    if n < 2:
        return 0.0, 0.0
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1))
    if sd < 1e-12:
        t = float("inf") if mean > 0 else 0.0
    else:
        t = mean / (sd / np.sqrt(n))
    return mean, t
