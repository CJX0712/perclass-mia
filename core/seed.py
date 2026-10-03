"""全局确定性种子入口（作者：晨星）。

所有 rng 必须经由本模块创建，禁止在模块顶层直接调用 np.random.* 或 random.*。
set_all(seed) 一次性钉死：numpy 全局 RNG、标准库 random、以及返回一个可传递的
np.random.Generator。目标模型 / 数据生成 / 攻击模型的随机性全部从同一个 root 派生，
保证同 seed 两次运行核心指标逐位一致（elapsed_sec 除外）。
"""
from __future__ import annotations

import random as _random
import numpy as np


def set_all(seed: int) -> np.random.Generator:
    """钉死全局随机源，并返回本次会话的主 Generator。

    返回的主 Generator 应当作为唯一的可变随机源在各模块间传递；模块内部
    若需独立子流，使用 rng.spawn / default_rng(rng.integers(...)) 派生，避免
    直接触碰 np.random 全局状态，从而保证跨 numpy 版本 / BLAS 的稳定复现。
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed 必须为非负整数，收到: {seed!r}")
    _random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    return np.random.default_rng(seed)
