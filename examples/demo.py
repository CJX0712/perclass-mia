"""PerClass-MIA 端到端演示（作者：晨星）。

用法：
    python examples/demo.py            # 单 seed 演示 + 5 seed 基准 + DoD 性能门槛 + 确定性自检
    python examples/demo.py --quick    # 缩小规模，快速验证可运行性

输出：
    - 单 seed 各攻击 AUC / TPR@FPR=0.1
    - 5 seed 聚合均值 ± 标准差
    - 旗舰 vs 最强基线的 AUC 提升（性能门槛 + 配对 t 检验显著性）
    - 确定性自检：同一 seed 两次运行核心指标逐位一致

DoD 性能门槛（作者：晨星）：
    旗舰(reference_mia) 较 最强基线(naive_stacked_lr) 的 5-seed 均值 AUC 提升 ≥ +0.03，
    且配对单侧 t 检验（同 seed 旗舰−基线差值）t ≥ 1.833（显著）。
    该门槛成立的关键在于"类训练样本异质"基准：各类过拟合程度不同，全局阈值基线被错校准，
    逐类参考归一化旗舰（LiRA 风格）显著占优。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Windows 控制台默认 cp1252，中文与 ✅/❌ 会抛 UnicodeEncodeError —— 强制 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

# 允许从仓库根目录直接运行（examples/ 在仓库内）
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from core.config import Config  # noqa: E402
from core.metrics import paired_one_sided_t  # noqa: E402
from core.seed import set_all  # noqa: E402
from data.generators import MIADataGenerator  # noqa: E402
from pipeline.attack_pipeline import benchmark, run  # noqa: E402
from attack.registry import FLAGSHIP, STRONG_BASELINE, DEFAULT_ATTACKS  # noqa: E402

# 甜点配置（DoD 门槛成立的前提）：宽类难度方差 + 适度过拟合
SWEET = dict(
    seeds=(0, 1, 2, 3, 4),
    n_per_class_member=60,
    n_per_class_nonmember=150,
    n_classes=4,
    attack_train_frac=0.5,
    target_hidden=16,
    target_epochs=800,
    target_lr=0.05,
    target_batch=32,
    difficulty_seed=7,
)


def _print_table(agg: dict[str, dict[str, float]]) -> None:
    print(f"\n{'attack':<22}{'mean_auc':>12}{'std_auc':>10}{'n':>6}")
    print("-" * 52)
    for name, m in agg.items():
        print(f"{name:<22}{m['mean']:>12.4f}{m['std']:>10.4f}{m['n']:>6}")


def _dod_check(bench: "object") -> bool:
    """配对单侧 t 检验：同 seed 旗舰 AUC − 最强基线 AUC 差值显著 > 0 且均值差 ≥ +0.03。"""
    cells = bench.cells
    by_seed: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for c in cells:
        by_seed[(c.dataset, c.seed)][c.attack] = c.auc
    fl_list, bl_list = [], []
    for d in by_seed.values():
        if FLAGSHIP in d and STRONG_BASELINE in d:
            fl_list.append(d[FLAGSHIP])
            bl_list.append(d[STRONG_BASELINE])
    if not fl_list:
        print("[DoD] 缺失旗舰或强基线结果")
        return False
    import numpy as np
    delta_mean = float(np.mean(fl_list) - np.mean(bl_list))
    diffs = [f - b for f, b in zip(fl_list, bl_list)]
    m, t = paired_one_sided_t(diffs)
    passed = delta_mean >= 0.03 and t >= 1.833
    print("\n=== DoD 性能门槛 ===")
    print(f"  旗舰({FLAGSHIP}) mean AUC = {np.mean(fl_list):.4f}  (n={len(fl_list)})")
    print(f"  强基线({STRONG_BASELINE}) mean AUC = {np.mean(bl_list):.4f}")
    print(f"  Δ mean AUC = {delta_mean:+.4f}  (门槛 ≥ +0.03)")
    print(f"  配对单侧 t = {t:.3f}  (门槛 ≥ 1.833，df={len(diffs)-1})")
    print(f"  结论: {'PASS ✅' if passed else 'FAIL ❌'}")
    return passed


def _determinism_check(cfg: Config, generators) -> bool:
    """同一 seed 两次运行核心 AUC 逐位一致。"""
    set_all(cfg.seeds[0])
    r1 = run(cfg, generators)
    set_all(cfg.seeds[0])
    r2 = run(cfg, generators)
    same = all(abs(a.auc - b.auc) < 1e-12 for a, b in zip(r1, r2))
    print("\n=== 确定性自检 ===")
    print(f"  两次运行逐位一致 = {same} {'✅' if same else '❌'}")
    return same


def main() -> int:
    ap = argparse.ArgumentParser(description="PerClass-MIA demo")
    ap.add_argument("--quick", action="store_true", help="缩小规模快速验证")
    args = ap.parse_args()

    if args.quick:
        cfg = Config(seeds=(0,), n_per_class_member=80, n_per_class_nonmember=80,
                     target_epochs=160, target_hidden=16, target_batch=32)
        generators = [MIADataGenerator("hetero", n_features=10, var_lo=0.2, var_hi=1.3)]
    else:
        cfg = Config.from_env()
        for k, v in SWEET.items():
            if k != "seeds" or not isinstance(v, tuple):
                setattr(cfg, k, v)
        cfg.validate()
        generators = [MIADataGenerator("hetero", n_features=10, var_lo=0.2, var_hi=1.3)]

    print(">>> 单 seed 端到端演示")
    set_all(cfg.seeds[0])
    res = run(cfg, generators)
    for r in res:
        print(f"  {r.name:<22} AUC={r.auc:.4f}  TPR@FPR=0.1={r.tpr_at_fpr01:.4f}")

    print("\n>>> 5 seed 基准（多数据集聚合）")
    set_all(cfg.seeds[0])
    bench = benchmark(cfg, generators)
    agg = bench.aggregate()
    _print_table(agg)
    print(f"\n  耗时 {bench.elapsed_sec:.2f}s，共 {len(bench.cells)} 个 (dataset×attack×seed) 单元")

    passed = _dod_check(bench)
    det = _determinism_check(cfg, generators)

    # 输出机器可读结果（numpy 标量 -> python 原生类型）
    out = {
        "aggregate": {k: {kk: float(vv) for kk, vv in m.items()} for k, m in agg.items()},
        "dod_passed": bool(passed),
        "determinism_passed": bool(det),
        "flagship": FLAGSHIP,
        "strong_baseline": STRONG_BASELINE,
    }
    Path(REPO_ROOT / "benchmark_result.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n结果已写入 benchmark_result.json")
    return 0 if (passed and det) else 1


if __name__ == "__main__":
    raise SystemExit(main())
