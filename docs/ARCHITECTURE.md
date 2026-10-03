# MIAForge 架构说明（作者：晨星）

## 1. 分层与依赖方向

```
        pipeline/
            │
     ┌──────┴──────┐
     ▼             ▼
   data/        attack/
     │             │
     └──────┬──────┘
            ▼
          core/
```

**单向无环**：`pipeline → {data, attack} → core`。`core` 不 import 任何上层模块，
因此指标、配置、类型可以脱离攻击逻辑独立测试。

| 层 | 职责 | 禁止 |
|---|---|---|
| `core/` | 配置、类型、指标、播种、异常、接口协议 | import `data/` `attack/` `pipeline/` |
| `data/` | 合成基准生成 | import `attack/` `pipeline/` |
| `attack/` | 目标模型 + 攻击实现 + 注册表 | import `pipeline/` |
| `pipeline/` | 编排：训练目标 → 取预测 → 划分 → 拟合攻击 → 评分 | 被上层反向 import |

## 2. 关键模块

### 2.1 `core/config.py`
所有超参集中一处。`Config.from_env()` 支持 `ENV_MIA_*` 覆盖（CI / 部署用）。
`validate()` 做 schema 校验——**早失败而非静默错算**（例如 `attack_train_frac=1.0`
会让 test 子集为空、AUC 静默变成随机数，必须在入口挡掉）。

### 2.2 `core/metrics.py`
- `roc_auc` / `tpr_at_fpr`：纯 NumPy 实现，与 sklearn 逐位对齐（有测试断言，容差 1e-6）。
  离线实现是刻意的：核心攻击链路不应因 sklearn 缺失而不可用。
- `paired_one_sided_t(diffs)`：配对单侧 t 检验。**这是本项目的统计正确口径**。
  逐 seed 独立估计方差很大（数据分布本身在变），必须用同 seed 的差值做配对检验。

### 2.3 `core/seed.py`
全局播种。种子派生用 `hashlib.sha256`，**禁用 Python 内置 `hash()`**——后者对 str
加进程随机盐，跨进程不可复现，是这类项目最常见的"确定性假象"来源。

### 2.4 `data/generators.py`
`MIADataGenerator(regime, n_features, var_lo, var_hi)`：

- `hetero`：各类标准差在 `[var_lo, var_hi]` 间采样并**洗牌**（避免"类号越大越难"的
  人造单调结构）。难度剖面由 `cfg.difficulty_seed` 固定，与 benchmark seed 解耦。
- `iso`：各类等难度，作为**对照档**——验证旗舰在无类间偏差时不劣化。
- 支持 `cfg.member_counts` 显式指定类样本分布（额外的难度异质维度）；主甜点用均衡档。

### 2.5 `attack/targets.py`
纯 NumPy MLP（`NumpyMLPTarget`），零 torch 依赖。刻意用小容量 + 适中 epoch 制造
**适度过拟合**（见 `docs/BENCHMARK.md`——完全记忆反而摧毁旗舰优势）。

### 2.6 `attack/reference.py`（旗舰）
```
fit:  对每预测类 c，用 attack-train 中 flag==0 且被判为 c 的样本 loss → (μ_c, σ_c)
score: z = (loss − μ_c)/σ_c ；return −z   （越大越像 member）
```
参考分布严格只取 non-member，**免训练任何影子模型**（对比 LiRA 的 N× 成本）。
`use_lr=True` 保留历史口径（融合 confidence/entropy），实测过拟合，默认关闭。

### 2.7 `attack/registry.py`
`build_attack(name)` 每次返回**全新实例**（避免跨 cell 状态污染），并把 `name`
写回实例——同一实现类可注册多个别名（`reference_z_only` 是旗舰的透明对照档），
别名即其在基准表中的身份。

## 3. 不变量清单

系统正确性由以下可独立断言的性质保证（全部有对应测试）：

| 不变量 | 断言位置 |
|---|---|
| `roc_auc` / `tpr_at_fpr` 与 sklearn 一致（1e-6） | `TestMetrics` |
| 随机分数 AUC ≈ 0.5；完美分离 AUC = 1.0 | `TestMetrics` |
| 同 seed 两次运行 AUC 逐位一致（1e-12） | `TestDeterminism` |
| attack-train / test 按 member_flag 分层且无交集 | `TestNoLeak` |
| 旗舰 AUC > 随机 + 0.05 | `TestEndToEnd` |
| 旗舰 vs 强基线：Δ ≥ +0.03 且配对 t ≥ 1.833 | `TestEndToEnd` |
| iso 档：旗舰不显著劣化（Δ > −0.02） | `TestEndToEnd` |
| 所有注册名可构造且 `instance.name == 注册名` | `TestRegistry` |

## 4. 扩展点

新增攻击只需实现两个方法并在 `registry.py` 注册，pipeline 无需改动：

```python
class MyAttack:
    name = "my_attack"
    def fit(self, preds, losses, member_flag, rng) -> None: ...
    def score(self, preds, losses) -> np.ndarray: ...   # 越大越像 member
```

新增数据集：实现 `make(cfg, rng) -> Dataset` 即可接入 `benchmark`。
