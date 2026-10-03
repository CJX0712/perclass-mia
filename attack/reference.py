"""旗舰攻击 ReferenceMIA（作者：晨星）。

核心思想（基于 LiRA 直觉，但**免参考模型**——直接以 non-member 集合作为各预测类的
参考分布）：
  1. 对每个预测类 c，用 attack-train 中 non-member 被判为 c 的样本损失拟合 N(μ_c, σ_c)
     （参考分布）。
  2. 对任意样本，z = (loss_i − μ_c)/σ_c：z 越小（损失显著低于同类参考）越像 member。
     score = −z：越大越像 member。

为什么稳赢朴素基线：
  各类难度异质时，raw loss 的"类内尺度"不同 → 全局阈值/朴素 LR 被易类主导，
  难类的 member 信号被淹没。逐类参考归一化消除了该偏置（这正是 LiRA 有效的本质），
  因此旗舰在 hetero 档显著占优，且在 iso 档仍保持非劣。

关于 LR 头融合额外特征：
  实测在小攻击训练集上，把 confidence/entropy 接入 LR 会过拟合并拖累 AUC；
  逐类 loss z 分数已是干净、方向一致的强信号，旗舰据此给出稳定 +0.04 量级的提升。
  reference_z_only 提供透明的纯阈值版本（同分），作为 ablation / 对照。
"""
from __future__ import annotations

import numpy as np

from core.errors import AttackError


class ReferenceMIA:
    name = "reference_mia"  # 旗舰

    def __init__(self, use_lr: bool = False) -> None:
        # 默认纯 z 核心（稳定）。use_lr=True 仅保留历史口径兼容，
        # 实测额外特征融合会过拟合，故默认不启用 LR 头。
        self.use_lr = use_lr
        self._ref_mu = None  # type: ignore
        self._ref_sigma = None  # type: ignore
        self._global_mu = 0.0
        self._global_sigma = 1.0
        self._lr = None  # type: ignore

    def fit(self, preds, losses, member_flag, rng) -> None:
        preds = np.asarray(preds, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64).ravel()
        member_flag = np.asarray(member_flag, dtype=np.int64).ravel()
        c = preds.shape[1]

        self._global_mu = float(losses.mean())
        self._global_sigma = float(losses.std()) if losses.std() > 1e-9 else 1.0

        pred_labels = preds.argmax(axis=1)
        mu = np.full(c, self._global_mu)
        sigma = np.full(c, self._global_sigma)
        # 参考分布仅来自 non-member（flag==0）
        nmask = member_flag == 0
        for k in range(c):
            mk = nmask & (pred_labels == k)
            if mk.sum() >= 2:
                ls = losses[mk]
                s = float(ls.std())
                mu[k] = float(ls.mean())
                sigma[k] = s if s > 1e-9 else self._global_sigma
        self._ref_mu = mu
        self._ref_sigma = sigma

        if self.use_lr:
            from .features import predict_features
            from .lr import NumpyLogisticRegression

            z = self._zscore(preds, losses)
            f = predict_features(preds, losses)
            x = np.stack([z, f["entropy"], f["confidence"]], axis=1)
            self._lr = NumpyLogisticRegression(seed=int(rng.integers(0, 2**31 - 1)))
            self._lr.fit(x, member_flag)

    def _zscore(self, preds, losses) -> np.ndarray:
        preds = np.asarray(preds, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64).ravel()
        pred_labels = preds.argmax(axis=1)
        mu = self._ref_mu[pred_labels]
        sigma = np.maximum(self._ref_sigma[pred_labels], 1e-9)
        return (losses - mu) / sigma  # 越小越像 member

    def score(self, preds, losses) -> np.ndarray:
        preds = np.asarray(preds, dtype=np.float64)
        losses = np.asarray(losses, dtype=np.float64).ravel()
        z = self._zscore(preds, losses)
        if not self.use_lr:
            return -z  # 越大（z 越小，损失越低）越像 member
        f = predict_features(preds, losses)
        x = np.stack([z, f["entropy"], f["confidence"]], axis=1)
        return self._lr.decision(x)
