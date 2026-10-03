"""核心数据类型（作者：晨星）。

所有跨模块传递的载体都定义为 dataclass，字段全部是 plain numpy / 基本类型，
保证可序列化（benchmark.json）且接口语义清晰。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Dataset:
    """合成基准数据集（已切分为成员/非成员两半）。

    X_member / y_member : 目标模型训练时见过的样本（members）。
    X_nonmember / y_nonmember : 同源分布、但目标模型未见过（non-members）。
    """

    name: str
    X_member: np.ndarray
    y_member: np.ndarray
    X_nonmember: np.ndarray
    y_nonmember: np.ndarray
    n_classes: int

    def probe(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """拼接 probe 集：返回 (X, y_true_label, member_flag)。

        member_flag: 1=member, 0=non-member。攻击者的监督标签。
        """
        X = np.vstack([self.X_member, self.X_nonmember]).astype(np.float64)
        y_lab = np.concatenate([self.y_member, self.y_nonmember]).astype(np.int64)
        flag = np.concatenate(
            [np.ones(len(self.X_member)), np.zeros(len(self.X_nonmember))]
        ).astype(np.int64)
        return X, y_lab, flag


@dataclass
class AttackResult:
    """单个攻击在某 probe 集上的评测结果。分数语义：AUC 越大越准（越大越好）。"""

    name: str
    auc: float
    tpr_at_fpr01: float
    n_samples: int
    scores: np.ndarray = field(default_factory=lambda: np.empty(0))
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkCell:
    """单个 (dataset, attack, seed) 单元的结果。"""

    dataset: str
    attack: str
    seed: int
    auc: float
    tpr_at_fpr01: float


@dataclass
class BenchmarkResult:
    """多数据集 × 多攻击 × 多 seed 的聚合结果。核心指标 = AUC（越大越好）。"""

    cells: list[BenchmarkCell] = field(default_factory=list)
    elapsed_sec: float = 0.0

    def aggregate(self) -> dict[str, dict[str, float]]:
        """按 attack 聚合 mean±std 的 AUC。"""
        from collections import defaultdict

        buckets: dict[str, list[float]] = defaultdict(list)
        for c in self.cells:
            buckets[c.attack].append(c.auc)
        out: dict[str, dict[str, float]] = {}
        for a, vals in buckets.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[a] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
                "n": len(arr),
            }
        return out
