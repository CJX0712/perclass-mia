# MIAForge 使用手册（作者：晨星）

## 安装

```bash
python -m venv .venv
# Windows
.venv/Scripts/pip install -r requirements.lock.txt
# Linux / macOS
source .venv/bin/activate && pip install -r requirements.lock.txt
```

核心攻击链路只依赖 numpy；sklearn 仅用于指标口径交叉验证，缺失不影响功能。

## 三件事

```bash
python examples/demo.py          # 演示 + 5 seed 基准 + DoD 门槛 + 确定性自检
python examples/demo.py --quick  # 冒烟
python -m pytest tests/ -q       # 14 项测试
```

`demo.py` 退出码：`0` = DoD 与确定性自检全通过；`1` = 有门槛未过。可直接当 CI 门禁。

## Python API

### 跑一次基准

```python
from core.config import Config
from core.seed import set_all
from data.generators import MIADataGenerator
from pipeline.attack_pipeline import benchmark

cfg = Config(seeds=(0, 1, 2, 3, 4),
             n_per_class_member=60, n_per_class_nonmember=150,
             attack_train_frac=0.5, n_classes=4,
             target_hidden=16, target_epochs=800, target_lr=0.05)
gen = MIADataGenerator("hetero", n_features=10, var_lo=0.2, var_hi=1.3)
set_all(0)
agg = benchmark(cfg, [gen]).aggregate()
print(agg["reference_mia"]["mean"])   # 0.7271
```

### 单 seed 端到端

```python
from pipeline.attack_pipeline import run
for r in run(cfg, [gen]):
    print(f"{r.name:<20} AUC={r.auc:.4f}  TPR@FPR=0.1={r.tpr_at_fpr01:.4f}")
```

### 配对显著性检验

```python
from core.metrics import paired_one_sided_t
mean_diff, t = paired_one_sided_t([f - b for f, b in zip(flagship_aucs, baseline_aucs)])
```

## 配置覆盖

| 环境变量 | 对应字段 |
|---|---|
| `ENV_MIA_SEEDS` | `seeds`（逗号分隔，如 `0,1,2`） |
| `ENV_MIA_N_PER_CLASS_MEMBER` | `n_per_class_member` |
| `ENV_MIA_N_PER_CLASS_NONMEMBER` | `n_per_class_nonmember` |
| `ENV_MIA_ATTACK_TRAIN_FRAC` | `attack_train_frac` |
| `ENV_MIA_N_CLASSES` | `n_classes` |
| `ENV_MIA_TARGET_EPOCHS` | `target_epochs` |
| `ENV_MIA_TARGET_LR` | `target_lr` |

`Config.from_env()` 会自动 `validate()`，非法值抛 `ConfigError`。

## 可用攻击

| 注册名 | 说明 |
|---|---|
| `random` | 空基线（校准点，AUC 应恒为 0.5） |
| `single_loss` | 全局 loss 阈值（Yeom-2018 风格） |
| `global_loss_lr` | 全局 loss 上拟合 LR |
| `naive_stacked_lr` | **最强基线**：loss/confidence/entropy/margin 全局 LR |
| `reference_z_only` | 旗舰的透明纯阈值版（对照档，同分） |
| `reference_mia` | **旗舰**：逐类参考归一化 z 分数 |

`single_<feature>` 支持任意单特征，如 `single_confidence`。

## 新增攻击

```python
class MyAttack:
    name = "my_attack"
    def fit(self, preds, losses, member_flag, rng) -> None:
        ...   # 只能用 member_flag 做监督
    def score(self, preds, losses) -> np.ndarray:
        ...   # 越大越像 member
```

在 `attack/registry.py` 的 `build_attack` 加分支并加入 `DEFAULT_ATTACKS` 即可，
pipeline 无需改动。

## 双档数据集

- `MIADataGenerator("hetero", var_lo=0.2, var_hi=1.3)` —— 类难度异质（**旗舰占优档**）
- `MIADataGenerator("iso")` —— 各类等难度（对照档，旗舰应持平而非劣化）

想验证机制，改这两个旋钮即可（详见 `docs/BENCHMARK.md`）：

- `target_epochs=3000, target_hidden=256` → 完全记忆 → 旗舰优势**消失**
- `MIADataGenerator("iso")` → 无类间偏差 → 旗舰退化为持平

## 常见问题

**Q：为什么我的改动让 AUC 只有 0.5？**
大概率是破坏了过拟合条件：数据量变大、维度变低、epoch 变少、容量变大都会让
member/non-member loss 分布重合。先用 `docs/BENCHMARK.md` 的甜点配置跑通再改。

**Q：为什么换台机器结果对不上？**
检查有没有引入 Python 内置 `hash()` 做种子派生（它有进程盐）。本项目统一用
`sha256` 派生。

**Q：sklearn 装不上怎么办？**
只影响 `tests/` 里的口径交叉验证两项；核心功能与 demo 不受影响。
