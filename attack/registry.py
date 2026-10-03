"""攻击注册表与可用后端探测（作者：晨星）。

available_* 用于离线降级：本系统核心攻击零依赖，sklearn 仅作口径交叉验证后端，
缺省不影响交付。
"""
from __future__ import annotations

from .baselines import (
    GlobalLossLR,
    NaiveStackedLR,
    RandomAttack,
    SingleFeatureAttack,
)
from .lr import available_sklearn
from .reference import ReferenceMIA


def build_attack(name: str):  # noqa: ANN201
    """按名构造攻击实例（每次返回全新实例，避免跨 cell 状态污染）。

    返回的实例 .name 恒等于请求的注册名 —— 同一实现类可注册多个别名（如
    reference_z_only 是 ReferenceMIA 的对照档），别名即其在基准表中的身份。
    """
    if name == "random":
        atk = RandomAttack()
    elif name.startswith("single_"):
        atk = SingleFeatureAttack(name[len("single_"):])
    elif name == "global_loss_lr":
        atk = GlobalLossLR()
    elif name == "naive_stacked_lr":
        atk = NaiveStackedLR()
    elif name in ("reference_mia", "reference_z_only"):
        # 旗舰：逐类参考归一化 z 分数；z_only 为同一实现的透明对照档
        atk = ReferenceMIA(use_lr=False)
    else:
        raise KeyError(f"未知攻击: {name}")
    atk.name = name
    return atk


# 主基准包含：空基线、单特征、最强朴素基线、旗舰、z-only 对照
DEFAULT_ATTACKS = (
    "random",
    "single_loss",
    "naive_stacked_lr",
    "reference_z_only",
    "reference_mia",
)

# 强基线口径（DoD 性能门槛对照对象）
STRONG_BASELINE = "naive_stacked_lr"
FLAGSHIP = "reference_mia"


def available_sklearn_backend() -> bool:
    return available_sklearn()
