"""被攻击的目标模型（作者：晨星）。

纯 numpy 单隐层 MLP（tanh 隐藏 + softmax 输出 + 交叉熵 + 小批量 SGD）。
零 torch 依赖，可在受限 Windows + 无 GPU 环境确定性复现。
目标模型是"黑盒"：攻击者只能调用 predict_proba / loss，看不到权重。
"""
from __future__ import annotations

import numpy as np

from core.errors import TargetError


class NumpyMLPTarget:
    def __init__(self, n_classes: int, hidden: int = 32, epochs: int = 320,
                 lr: float = 0.05, batch: int = 64) -> None:
        self.n_classes = int(n_classes)
        self.hidden = int(hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch = int(batch)
        self._W1 = self._b1 = self._W2 = self._b2 = None  # type: ignore

    def _init_params(self, d: int, rng: np.random.Generator) -> None:
        # 缩放初始化（Glorot-ish），固定 rng 保证确定性
        lim1 = np.sqrt(6.0 / (d + self.hidden))
        lim2 = np.sqrt(6.0 / (self.hidden + self.n_classes))
        self._W1 = rng.uniform(-lim1, lim1, size=(d, self.hidden))
        self._b1 = np.zeros(self.hidden)
        self._W2 = rng.uniform(-lim2, lim2, size=(self.hidden, self.n_classes))
        self._b2 = np.zeros(self.n_classes)

    @staticmethod
    def _forward(X: np.ndarray, W1, b1, W2, b2):
        h = np.tanh(X @ W1 + b1)
        logits = h @ W2 + b2
        # 数值稳定 softmax
        z = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        return h, logits, p

    def fit(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> "NumpyMLPTarget":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).ravel()
        n, d = X.shape
        if X.shape[0] != y.shape[0]:
            raise TargetError("X/y 样本数不一致")
        self._init_params(d, rng)
        Y = np.zeros((n, self.n_classes), dtype=np.float64)
        Y[np.arange(n), y] = 1.0

        idx = np.arange(n)
        for _ in range(self.epochs):
            rng.shuffle(idx)
            for s in range(0, n, max(self.batch, 1)):
                bix = idx[s:s + self.batch]
                if bix.size == 0:
                    continue
                Xb, Yb = X[bix], Y[bix]
                h, logits, p = self._forward(Xb, self._W1, self._b1, self._W2, self._b2)
                # 梯度
                dp = p - Yb                      # (B, C)
                dW2 = h.T @ dp
                db2 = dp.sum(axis=0)
                dh = (dp @ self._W2.T) * (1.0 - h * h)  # tanh'
                dW1 = Xb.T @ dh
                db1 = dh.sum(axis=0)
                self._W2 -= self.lr * dW2 / bix.size
                self._b2 -= self.lr * db2 / bix.size
                self._W1 -= self.lr * dW1 / bix.size
                self._b1 -= self.lr * db1 / bix.size
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._W1 is None:
            raise TargetError("目标模型尚未 fit")
        X = np.asarray(X, dtype=np.float64)
        _, _, p = self._forward(X, self._W1, self._b1, self._W2, self._b2)
        return p

    def loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        y = np.asarray(y, dtype=np.int64).ravel()
        # 逐样本交叉熵（clip 防 log(0)）
        pc = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
        return -np.log(pc)
