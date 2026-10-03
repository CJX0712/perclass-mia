"""抽象契约（作者：晨星）。

调用链单向无环：cli -> pipeline -> {data, attack(target)} -> core。
所有可替换组件通过 Protocol 解耦，默认实现零依赖，生产实现一键切换。
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from .types import AttackResult, Dataset


@runtime_checkable
class TargetModel(Protocol):
    """被攻击的目标模型。攻击者只能看到 predict_proba / 逐样本 loss，看不到权重。"""

    n_classes: int

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """返回 (n, C) 概率矩阵。"""
        ...

    def loss(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """返回 (n,) 逐样本交叉熵损失。"""
        ...


@runtime_checkable
class DataGenerator(Protocol):
    def make(self, cfg, rng: np.random.Generator) -> Dataset:
        ...


@runtime_checkable
class Attack(Protocol):
    name: str

    def fit(self, preds: np.ndarray, losses: np.ndarray, member_flag: np.ndarray,
            rng: np.random.Generator) -> None:
        """用 attack-train 子集拟合攻击模型（无泄漏：仅用该子集）。

        preds: (n, C) 目标模型概率；losses: (n,) 逐样本损失；
        member_flag: (n,) 0/1（1=member）。各攻击内部自行派生所需特征。
        """
        ...

    def score(self, preds: np.ndarray, losses: np.ndarray) -> np.ndarray:
        """返回 (n,) 攻击分数，分数越大越判为 member。"""
        ...
