"""从目标模型输出派生攻击特征（作者：晨星）。

分数语义统一：这些值越大越"像 member"或由调用方自行校准。
- loss: 越小越像 member（攻击时取负号或单调递增变换）
- confidence: 最大类概率，越大越像 member
- entropy: 越小越像 member
- margin: 最大与次大概率差，越大越像 member
"""
from __future__ import annotations

import numpy as np


def predict_features(preds: np.ndarray, losses: np.ndarray) -> dict[str, np.ndarray]:
    """返回各单调特征。所有特征均被整理为"越大越像 member"。"""
    preds = np.asarray(preds, dtype=np.float64)
    losses = np.asarray(losses, dtype=np.float64).ravel()
    conf = preds.max(axis=1)
    # 熵（越小越像 member）
    eps = 1e-12
    pc = np.clip(preds, eps, 1.0)
    entropy = -(pc * np.log(pc)).sum(axis=1)
    # margin：最大与次大概率之差（越大越像 member）
    part = np.sort(preds, axis=1)
    margin = part[:, -1] - part[:, -2]
    return {
        "loss": -losses,            # 越大（损失越小）越像 member
        "confidence": conf,
        "entropy": -entropy,        # 越大（熵越小）越像 member
        "margin": margin,
    }


FEATURE_ORDER = ("loss", "confidence", "entropy", "margin")
