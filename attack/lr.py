"""纯 numpy 逻辑回归（作者：晨星）。

用作攻击模型的分类头，零 sklearn 依赖即可离线复现；同时提供 sklearn 后端可选
路径（available_sklearn）用于单测口径交叉验证。L2 正则 + 全批量梯度下降，
确定性由 rng 初始化保证。
"""
from __future__ import annotations

import numpy as np


class NumpyLogisticRegression:
    def __init__(self, lr: float = 0.1, n_iter: int = 400, l2: float = 1e-3,
                 seed: int = 0) -> None:
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.seed = int(seed)
        self._w = None  # type: ignore
        self._b = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumpyLogisticRegression":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self._w = rng.normal(0.0, 0.01, size=d)
        self._b = 0.0
        ybin = (y == 1).astype(np.float64)
        for _ in range(self.n_iter):
            scores = X @ self._w + self._b
            p = 1.0 / (1.0 + np.exp(-scores))
            err = p - ybin
            grad_w = (X.T @ err) / n + self.l2 * self._w
            grad_b = err.mean()
            self._w -= self.lr * grad_w
            self._b -= self.lr * grad_b
        return self

    def decision(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        return X @ self._w + self._b  # 越大越判为阳性（member）


def available_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def sklearn_lr(X: np.ndarray, y: np.ndarray) -> "SklearnLRShim":
    """可选 sklearn 后端（仅用于口径交叉验证；不参与主交付路径）。"""
    from sklearn.linear_model import LogisticRegression

    from core.errors import AttackError

    try:
        m = LogisticRegression(max_iter=1000)
        m.fit(X, y)
        return SklearnLRShim(m)
    except Exception as exc:  # noqa: BLE001
        raise AttackError(f"sklearn LR 失败: {exc}") from exc


class SklearnLRShim:
    def __init__(self, model) -> None:
        self.model = model

    def decision(self, X: np.ndarray) -> np.ndarray:
        return self.model.decision_function(np.asarray(X, dtype=np.float64))
