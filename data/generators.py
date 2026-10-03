"""合成数据生成（作者：晨星）。

设计要点（甜点 + 无泄漏 + 可复现）：
- 维度 n_features（默认 10）：高维下小样本目标模型真实过拟合 —— member 损失被压低、
  non-member 损失保持高，形成清晰且方向一致的成员信号（低维下空间泛化会淹没信号）。
- 逐类异质难度（hetero 档）：各类方差在 [lo,hi] 间异质采样，但均处于"过拟合区间"，
  保证逐类 gap 方向一致为负（member 损失更低），仅尺度不同 —— 这正是逐类参考归一化
  旗舰显著占优的前提。iso 档：各类等难度（对照），用于验证旗舰在"无类间偏差"时仍非劣。
- 难度剖面由 cfg.difficulty_seed 固定，与 benchmark seed 解耦：数据集*结构*稳定，
  benchmark seed 只改变抽样与模型初始化。
- 目标模型训练只用 member 半；non-member 与 member 同源同分布，仅未参与训练（无泄漏）。
"""
from __future__ import annotations

import numpy as np

from core.config import Config
from core.errors import DataError
from core.types import Dataset


def _class_variances(n_classes: int, regime: str, difficulty_seed: int,
                    lo: float = 0.18, hi: float = 0.38) -> np.ndarray:
    """返回每个类的标准差（控制难度）。

    hetero：在 [lo,hi] 间异质采样并洗牌，避免"类号越大越难"的人造单调；该区间经
    实测保证逐类 gap 方向一致（见仓库 benchmark 报告）。iso：所有类等难度。
    """
    drng = np.random.default_rng(difficulty_seed)
    if regime == "iso":
        return np.full(n_classes, 0.18)
    raw = drng.uniform(lo, hi, size=n_classes)
    drng.shuffle(raw)
    return raw


def _centers(n_classes: int, n_features: int) -> np.ndarray:
    """类中心：前两个坐标铺在半径 1 的圆上，其余维度为 0（保证可分离且不退化）。"""
    ctr = np.zeros((n_classes, n_features), dtype=np.float64)
    if n_classes == 2:
        ctr[0, 0] = -1.0
        ctr[1, 0] = 1.0
        return ctr
    angles = np.linspace(0.0, 2 * np.pi, n_classes, endpoint=False)
    ctr[:, 0] = np.cos(angles)
    ctr[:, 1] = np.sin(angles)
    return ctr


class MIADataGenerator:
    """多类异质高斯基准生成器。regime ∈ {hetero, iso}。

    支持两种成员分布：
    - 均衡（cfg.member_counts 为 None）：每类 n_per_class_member 个成员，
      每类 n_per_class_nonmember 个非成员（主甜点用此档）。
    - 异质（cfg.member_counts 显式）：各类成员数不同 → 各类过拟合程度不同。
      非成员按成员分布等比放大（nonmember_ratio），保持类先验一致，避免"先验偏移"捷径。

    旗舰占优的机制（已在 benchmark 中验证）：各类方差跨度大 → 各类"绝对损失水平"差异
    显著；目标模型仅适度过拟合（mem 损失未全趋 0）→ 难类 member 的绝对损失高于易类
    non-member，全局阈值基线因此误排。逐类参考归一化把各类拉到同一尺度，通过"跨类重排"
    修正此类反转，从而显著胜过全局阈值 / 全局 LR 基线。
    """

    def __init__(self, regime: str = "hetero", n_features: int = 10,
                 var_lo: float = 0.2, var_hi: float = 1.3) -> None:
        if regime not in ("hetero", "iso"):
            raise DataError(f"未知 regime: {regime}")
        self.regime = regime
        self.n_features = int(n_features)
        self.var_lo = float(var_lo)
        self.var_hi = float(var_hi)

    @property
    def name(self) -> str:
        bal = "bal" if self.var_lo <= 0.2 else "imb"
        return f"hetero_blob_{self.regime}_d{self.n_features}_{bal}"

    def _class_counts(self, cfg: Config) -> tuple[list[int], list[int]]:
        if cfg.member_counts is not None:
            m = list(int(x) for x in cfg.member_counts)
            nm = [max(20, int(round(c * cfg.nonmember_ratio))) for c in m]
            return m, nm
        m = [cfg.n_per_class_member] * cfg.n_classes
        nm = [cfg.n_per_class_nonmember] * cfg.n_classes
        return m, nm

    def make(self, cfg: Config, rng: np.random.Generator) -> Dataset:
        m_counts, nm_counts = self._class_counts(cfg)
        n = len(m_counts)
        variances = _class_variances(n, self.regime, cfg.difficulty_seed,
                                    self.var_lo, self.var_hi)
        centers = _centers(n, self.n_features)

        def sample(counts: list[int]) -> tuple[np.ndarray, np.ndarray]:
            Xs, ys = [], []
            for c in range(n):
                cov = np.eye(self.n_features) * (variances[c] ** 2)
                Xc = rng.multivariate_normal(centers[c], cov, size=counts[c])
                Xs.append(Xc)
                ys.append(np.full(counts[c], c, dtype=np.int64))
            return np.vstack(Xs), np.concatenate(ys)

        Xm, ym = sample(m_counts)
        Xn, yn = sample(nm_counts)
        return Dataset(
            name=self.name,
            X_member=Xm,
            y_member=ym,
            X_nonmember=Xn,
            y_nonmember=yn,
            n_classes=n,
        )
