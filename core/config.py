"""运行配置与 ENV_XXX_* 覆盖（作者：晨星）。

所有可被运行期覆盖的超参集中在此；schema 校验保证非法值早失败而非静默错算。
环境变量前缀 ENV_MIA_ 允许在 CI / 部署时覆盖（例如 ENV_MIA_SEEDS=0,1,2）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .errors import ConfigError


@dataclass
class Config:
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    n_per_class_member: int = 60
    n_per_class_nonmember: int = 150
    # 类训练样本数（成员）分布：None=均衡；否则逐类显式指定（额外的难度异质维度；
    # 非成员按同分布等比放大，避免"先验偏移"捷径）。主甜点使用均衡分布即可。
    member_counts: "tuple[int, ...] | None" = None
    # 非成员按成员分布等比放大（保持类先验一致），最小 20。
    nonmember_ratio: float = 2.5
    attack_train_frac: float = 0.5  # probe 内用于拟合攻击模型的比例（其余评测，降方差）
    n_classes: int = 4
    # 适度过拟合（非完全记忆）：mem 损失不全趋 0，类间绝对损失水平差异显著 →
    # 逐类参考归一化（旗舰）才能通过"跨类重排"显著胜过全局阈值基线。
    target_hidden: int = 16
    target_epochs: int = 800
    target_lr: float = 0.05
    target_batch: int = 32
    difficulty_seed: int = 7  # 数据集甜点固定的异质难度种子（与 bench seed 解耦）
    determinism_guard: bool = True

    @staticmethod
    def from_env() -> "Config":
        cfg = Config()
        mapping = {
            "ENV_MIA_SEEDS": ("seeds", _parse_int_tuple),
            "ENV_MIA_N_PER_CLASS_MEMBER": ("n_per_class_member", int),
            "ENV_MIA_N_PER_CLASS_NONMEMBER": ("n_per_class_nonmember", int),
            "ENV_MIA_ATTACK_TRAIN_FRAC": ("attack_train_frac", float),
            "ENV_MIA_N_CLASSES": ("n_classes", int),
            "ENV_MIA_TARGET_EPOCHS": ("target_epochs", int),
            "ENV_MIA_TARGET_LR": ("target_lr", float),
        }
        for env_key, (attr, caster) in mapping.items():
            raw = os.environ.get(env_key)
            if raw is None:
                continue
            try:
                setattr(cfg, attr, caster(raw))
            except (ValueError, TypeError) as exc:
                raise ConfigError(f"{env_key}={raw!r} 解析失败: {exc}") from exc
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.attack_train_frac <= 0.0 or self.attack_train_frac >= 1.0:
            raise ConfigError("attack_train_frac 必须 ∈ (0,1)")
        if self.n_classes < 2:
            raise ConfigError("n_classes 至少 2")
        if self.member_counts is not None:
            if len(self.member_counts) != self.n_classes:
                raise ConfigError("member_counts 长度必须与 n_classes 一致")
            if any(c < 15 for c in self.member_counts):
                raise ConfigError("member_counts 每类至少 15（保证参考分布可估）")
        else:
            if self.n_per_class_member < 20 or self.n_per_class_nonmember < 20:
                raise ConfigError("每类样本数过少（<20），统计无意义")
        if self.nonmember_ratio <= 0:
            raise ConfigError("nonmember_ratio 必须 > 0")
        if self.target_epochs < 1:
            raise ConfigError("target_epochs 必须 ≥ 1")


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip() != "")
