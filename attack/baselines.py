"""基线攻击（作者：晨星）。

按"强度递增 + 口径透明"组织，最强基线 = NaiveStackedLR（多特征拼接 LR），
用于 DoD 性能门槛对照。所有基线零依赖、可离线。
"""
from __future__ import annotations

import numpy as np

from core.errors import AttackError
from .features import FEATURE_ORDER, predict_features
from .lr import NumpyLogisticRegression


class RandomAttack:
    """空基线：恒等分数 → AUC 恒为 0.5（无区分信息）。"""

    name = "random"

    def fit(self, preds, losses, member_flag, rng) -> None:  # noqa: D401
        pass

    def score(self, preds, losses) -> np.ndarray:
        preds = np.asarray(preds)
        return np.zeros(preds.shape[0], dtype=np.float64)


class SingleFeatureAttack:
    """单特征攻击：直接以某一单调特征为分数（参数-free 阈值攻击的泛化）。"""

    def __init__(self, feature: str) -> None:
        if feature not in FEATURE_ORDER:
            raise AttackError(f"未知特征: {feature}")
        self.feature = feature
        self.name = f"single_{feature}"

    def fit(self, preds, losses, member_flag, rng) -> None:
        pass

    def score(self, preds, losses) -> np.ndarray:
        feats = predict_features(preds, losses)
        return np.asarray(feats[self.feature], dtype=np.float64).ravel()


class GlobalLossLR:
    """全局损失逻辑回归：仅用 raw loss 一个特征（经典 threshold 攻击的 LR 版）。"""

    name = "global_loss_lr"

    def __init__(self) -> None:
        self._lr = None  # type: ignore

    def fit(self, preds, losses, member_flag, rng) -> None:
        x = predict_features(preds, losses)["loss"].reshape(-1, 1)
        self._lr = NumpyLogisticRegression(seed=int(rng.integers(0, 2**31 - 1)))
        self._lr.fit(x, member_flag)

    def score(self, preds, losses) -> np.ndarray:
        x = predict_features(preds, losses)["loss"].reshape(-1, 1)
        return self._lr.decision(x)


class NaiveStackedLR:
    """朴素多特征 LR：将 [loss,confidence,entropy,margin] 直接拼接（不做类内归一化）。

    这是**最强基线**：多特征未经逐类参考校准，仍受"类难度异质"偏置。
    """

    name = "naive_stacked_lr"

    def __init__(self) -> None:
        self._lr = None  # type: ignore

    @staticmethod
    def _stack(preds, losses) -> np.ndarray:
        f = predict_features(preds, losses)
        cols = [f[k] for k in FEATURE_ORDER]
        return np.stack(cols, axis=1)

    def fit(self, preds, losses, member_flag, rng) -> None:
        x = self._stack(preds, losses)
        self._lr = NumpyLogisticRegression(seed=int(rng.integers(0, 2**31 - 1)))
        self._lr.fit(x, member_flag)

    def score(self, preds, losses) -> np.ndarray:
        x = self._stack(preds, losses)
        return self._lr.decision(x)
