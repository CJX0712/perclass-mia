"""端到端攻击评测流水线（作者：晨星）。

调用链单向无环：pipeline -> {data, attack(target)} -> core。
无泄漏保证：目标模型仅在 member 半训练；攻击模型仅在 attack-train 子集上拟合，
在 attack-test 子集上评分（按 member_flag 分层抽样）。
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from core.config import Config
from core.errors import PipelineError
from core.metrics import roc_auc, tpr_at_fpr
from core.types import AttackResult, BenchmarkCell, BenchmarkResult, Dataset
from attack.registry import DEFAULT_ATTACKS, build_attack
from attack.targets import NumpyMLPTarget


def _stratified_split(flag: np.ndarray, frac: float, rng: np.random.Generator):
    """按 member_flag 分层抽样。返回 (train_idx, test_idx)。"""
    flag = np.asarray(flag)
    tr, te = [], []
    for v in (0, 1):
        ix = np.where(flag == v)[0]
        rng.shuffle(ix)
        k = int(round(len(ix) * frac))
        tr.append(ix[:k])
        te.append(ix[k:])
    return np.concatenate(tr), np.concatenate(te)


def _stable_seed(name: str, seed: int) -> int:
    """确定性 32-bit 种子（不使用进程盐化的内置 hash，保证跨进程复现）。"""
    h = hashlib.sha256(f"{name}:{seed}".encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big")


def run_one(dataset: Dataset, cfg: Config, attack_names,
            rng_data: np.random.Generator, rng_target: np.random.Generator,
            rng_attack: np.random.Generator) -> list[AttackResult]:
    # 1) 训练目标模型（仅 member 半）
    target = NumpyMLPTarget(
        n_classes=dataset.n_classes, hidden=cfg.target_hidden,
        epochs=cfg.target_epochs, lr=cfg.target_lr, batch=cfg.target_batch,
    ).fit(dataset.X_member, dataset.y_member, rng_target)

    # 2) 在完整 probe 上取目标输出（成员/非成员均可见）
    X, y_lab, flag = dataset.probe()
    preds = target.predict_proba(X)
    losses = target.loss(X, y_lab)

    # 3) 无泄漏划分
    tr_idx, te_idx = _stratified_split(flag, cfg.attack_train_frac, rng_attack)

    results: list[AttackResult] = []
    for name in attack_names:
        atk = build_attack(name)
        atk.fit(preds[tr_idx], losses[tr_idx], flag[tr_idx], rng_attack)
        sc = atk.score(preds[te_idx], losses[te_idx])
        sc = np.asarray(sc, dtype=np.float64).ravel()
        auc = roc_auc(sc, flag[te_idx])
        tpr = tpr_at_fpr(sc, flag[te_idx], 0.1)
        results.append(AttackResult(name=name, auc=auc, tpr_at_fpr01=tpr,
                                    n_samples=len(te_idx), scores=sc))
    return results


def benchmark(cfg: Config, generators, attack_names=DEFAULT_ATTACKS) -> BenchmarkResult:
    """多 (数据集×seed) 基准。返回聚合前的逐 cell 结果。"""
    t0 = time.perf_counter()
    cells: list[BenchmarkCell] = []
    for gen in generators:
        for seed in cfg.seeds:
            base = _stable_seed(gen.name, seed)
            rng_data = np.random.default_rng(base * 3 + 1)
            rng_target = np.random.default_rng(base * 3 + 2)
            rng_attack = np.random.default_rng(base * 3 + 3)
            ds = gen.make(cfg, rng_data)
            res = run_one(ds, cfg, attack_names, rng_data, rng_target, rng_attack)
            for r in res:
                cells.append(BenchmarkCell(
                    dataset=gen.name, attack=r.name, seed=seed,
                    auc=r.auc, tpr_at_fpr01=r.tpr_at_fpr01))
    return BenchmarkResult(cells=cells, elapsed_sec=time.perf_counter() - t0)


def run(cfg: Config, generators, attack_names=DEFAULT_ATTACKS) -> list[AttackResult]:
    """单 seed 单数据集的端到端演示（默认取首个生成器与首个 seed）。"""
    gen = generators[0]
    seed = cfg.seeds[0]
    base = _stable_seed(gen.name, seed)
    rng_data = np.random.default_rng(base * 3 + 1)
    rng_target = np.random.default_rng(base * 3 + 2)
    rng_attack = np.random.default_rng(base * 3 + 3)
    ds = gen.make(cfg, rng_data)
    return run_one(ds, cfg, attack_names, rng_data, rng_target, rng_attack)
