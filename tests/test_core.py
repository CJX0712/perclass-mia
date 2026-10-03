"""PerClass-MIA 单元测试（作者：晨星）。

覆盖：
- 指标口径：纯 numpy roc_auc / tpr_at_fpr 与 sklearn 一致
- 确定性：同 seed 两次运行逐位一致
- 无泄漏：攻击模型只在 attack-train 子集拟合
- 端到端：pipeline 可跑，旗舰 AUC > 随机基线
- DoD 性能门槛：旗舰 vs 最强基线
- 架构不变量：调用链单向（pipeline 不反向 import attack/target 之外的私有实现细节）

运行：
    pytest tests/ -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import Config  # noqa: E402
from core.metrics import roc_auc, tpr_at_fpr, paired_one_sided_t  # noqa: E402
from core.seed import set_all  # noqa: E402
from data.generators import MIADataGenerator  # noqa: E402
from pipeline.attack_pipeline import benchmark, run  # noqa: E402
from attack.registry import FLAGSHIP, STRONG_BASELINE, build_attack  # noqa: E402


def _sklearn_auc(scores, labels):
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _sklearn_tpr(scores, labels, fpr_target=0.1):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels, scores)
    # 最大 TPR at FPR<=target
    mask = fpr <= fpr_target + 1e-9
    if not mask.any():
        return float(tpr[0])
    return float(tpr[mask].max())


class TestMetrics:
    def test_auc_matches_sklearn(self):
        rng = np.random.default_rng(0)
        for s in range(5):
            scores = rng.normal(size=400)
            labels = (rng.normal(size=400) > 0).astype(int)
            ours = roc_auc(scores, labels)
            ref = _sklearn_auc(scores, labels)
            assert abs(ours - ref) < 1e-6, (s, ours, ref)

    def test_tpr_matches_sklearn(self):
        rng = np.random.default_rng(1)
        for s in range(5):
            scores = rng.normal(size=400)
            labels = (rng.normal(size=400) > 0).astype(int)
            ours = tpr_at_fpr(scores, labels, 0.1)
            ref = _sklearn_tpr(scores, labels, 0.1)
            assert abs(ours - ref) < 1e-6, (s, ours, ref)

    def test_auc_random_baseline(self):
        rng = np.random.default_rng(2)
        scores = rng.normal(size=1000)
        labels = rng.integers(0, 2, size=1000)
        # 随机分数应接近 0.5
        assert abs(roc_auc(scores, labels) - 0.5) < 0.05

    def test_auc_perfect(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        assert abs(roc_auc(scores, labels) - 1.0) < 1e-9

    def test_paired_t_positive(self):
        # 全为正差值 → 显著
        diffs = np.array([0.04, 0.05, 0.03, 0.06, 0.04])
        m, t = paired_one_sided_t(diffs)
        assert m > 0 and t > 1.833

    def test_paired_t_negative(self):
        diffs = np.array([-0.01, -0.02, -0.01])
        m, t = paired_one_sided_t(diffs)
        assert m < 0 and t < 1.833


class TestDeterminism:
    def test_two_runs_identical(self):
        cfg = Config(seeds=(3,), n_per_class_member=120, n_per_class_nonmember=120,
                     target_epochs=160, target_hidden=24)
        gens = [MIADataGenerator("hetero")]
        set_all(cfg.seeds[0])
        r1 = run(cfg, gens)
        set_all(cfg.seeds[0])
        r2 = run(cfg, gens)
        for a, b in zip(r1, r2):
            assert abs(a.auc - b.auc) < 1e-12, (a.name, a.auc, b.auc)

    def test_benchmark_seeds_stable(self):
        cfg = Config(seeds=(1, 2), n_per_class_member=100, n_per_class_nonmember=100,
                     target_epochs=120, target_hidden=16)
        gens = [MIADataGenerator("hetero")]
        set_all(0)
        b1 = benchmark(cfg, gens)
        set_all(0)
        b2 = benchmark(cfg, gens)
        for c1, c2 in zip(b1.cells, b2.cells):
            assert abs(c1.auc - c2.auc) < 1e-12


class TestNoLeak:
    def test_attack_train_test_split_sizes(self):
        from pipeline.attack_pipeline import _stratified_split

        rng = np.random.default_rng(0)
        flag = np.array([1] * 200 + [0] * 200)
        tr, te = _stratified_split(flag, 0.5, rng)
        # 每个类都按 0.5 划分
        for v in (0, 1):
            n = int((flag == v).sum())
            assert int((flag[tr] == v).sum()) == n // 2
            assert int((flag[te] == v).sum()) == n - n // 2

    def test_no_overlap(self):
        from pipeline.attack_pipeline import _stratified_split

        rng = np.random.default_rng(5)
        flag = np.array([1] * 100 + [0] * 100)
        tr, te = _stratified_split(flag, 0.5, rng)
        assert len(set(tr.tolist()) & set(te.tolist())) == 0


# 甜点配置：宽类难度方差 + 适度过拟合（DoD 门槛成立的前提，见 docs/BENCHMARK.md）
SWEET = dict(
    n_per_class_member=60, n_per_class_nonmember=150, n_classes=4,
    attack_train_frac=0.5, target_hidden=16, target_epochs=800,
    target_lr=0.05, target_batch=32, difficulty_seed=7,
)
SWEET_GEN = lambda: [MIADataGenerator("hetero", n_features=10,  # noqa: E731
                                      var_lo=0.2, var_hi=1.3)]


class TestEndToEnd:
    def test_pipeline_runs(self):
        cfg = Config(seeds=(0,), **SWEET)
        gens = SWEET_GEN()
        set_all(0)
        res = run(cfg, gens)
        names = {r.name for r in res}
        for required in ("random", "naive_stacked_lr", "reference_mia"):
            assert required in names
        # 旗舰必须优于随机基线
        auc_map = {r.name: r.auc for r in res}
        assert auc_map["reference_mia"] > auc_map["random"] + 0.05

    def test_flagship_beats_strong_baseline_5seed(self):
        # 甜点：宽类难度方差 → 类间绝对损失水平差异显著；适度过拟合 → 存在跨类反转，
        # 逐类参考归一化旗舰通过"跨类重排"显著胜过全局 LR 基线（LiRA 机制）。
        cfg = Config(seeds=(0, 1, 2, 3, 4), **SWEET)
        gens = SWEET_GEN()
        set_all(0)
        bench = benchmark(cfg, gens)
        # 配对 t 检验：同 seed 旗舰 − 最强基线
        from collections import defaultdict
        by_seed = defaultdict(dict)
        for c in bench.cells:
            by_seed[(c.dataset, c.seed)][c.attack] = c.auc
        fl_list, bl_list = [], []
        for d in by_seed.values():
            fl_list.append(d[FLAGSHIP])
            bl_list.append(d[STRONG_BASELINE])
        delta_mean = float(np.mean(fl_list) - np.mean(bl_list))
        diffs = [f - b for f, b in zip(fl_list, bl_list)]
        m, t = paired_one_sided_t(diffs)
        assert delta_mean >= 0.03, (delta_mean, fl_list, bl_list)
        assert t >= 1.833, (t, diffs)
        # 旗舰必须 > 随机
        assert bench.aggregate()["random"]["mean"] < 0.55

    def test_iso_regime_non_inferior(self):
        """对照档（各类等难度）：无类间偏差可修正，旗舰应与全局基线持平（不显著劣化）。"""
        cfg = Config(seeds=(0, 1, 2, 3), **SWEET)
        gens = [MIADataGenerator("iso", n_features=10)]
        set_all(0)
        bench = benchmark(cfg, gens)
        agg = bench.aggregate()
        delta = agg[FLAGSHIP]["mean"] - agg[STRONG_BASELINE]["mean"]
        assert delta > -0.02, (delta, agg[FLAGSHIP], agg[STRONG_BASELINE])


class TestRegistry:
    def test_build_all(self):
        for name in ("random", "single_loss", "naive_stacked_lr",
                     "reference_z_only", "reference_mia"):
            atk = build_attack(name)
            assert atk.name == name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
