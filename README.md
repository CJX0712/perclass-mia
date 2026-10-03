# PerClass-MIA — 免参考模型的成员推断攻击基准与旗舰攻击框架

> 作者：晨星 · 纯 NumPy 实现 · 零构建 · 确定性可复现
>
> **一句话**：PerClass-MIA 证明并量化了一件事——在"类难度异质"的目标模型上，
> **逐类参考归一化（LiRA 的核心直觉）可以在不训练任何参考模型的前提下**，
> 显著击败全局阈值 / 全局 LR 等最强朴素基线。

---

## 1. 这个系统解决什么问题

**成员推断攻击（Membership Inference Attack, MIA）**：给定对一个机器学习模型的黑盒
访问（输入 → 预测概率），判断某条样本**是否曾出现在这模型的训练集里**。

- 它是模型**隐私泄露**的量化标尺（医疗、金融等高敏场景的合规必答题）。
- 也是**记忆化 / 过拟合**的诊断工具。

朴素做法：拿目标模型的 loss 或置信度当分数，画条全局阈值线。它有一个**结构性缺陷**——
把"模型在难类上天生就高 loss"误判成"这是非成员"。

**PerClass-MIA 的旗舰攻击 ReferenceMIA** 用一条更便宜的路修掉这个缺陷：

| | 经典 LiRA | PerClass-MIA `ReferenceMIA` |
|---|---|---|
| 参考分布来源 | 训练 **N 个影子模型**（成本 N×） | 直接用 **non-member 集合**（成本 0） |
| 归一化粒度 | 逐样本 × 逐类 | 逐**预测类** |
| 依赖 | torch + 大量算力 | 纯 NumPy，秒级 |
| 本仓库 AUC | — | **0.727** vs 强基线 0.680（**+0.047，t=29.0**） |

---

## 2. 核心机制（为什么能赢，什么时候会输）

这是本项目最有价值的部分，也是实验中反复被证伪、最终被确认的结论。

**关键洞察**：逐类 z 分数在**类内**是 loss 的单调仿射变换 `z = (loss − μ_c)/σ_c`，
因此它**不可能改变类内排序**。它的全部增益只能来自**跨类重排**：

```
难类 member（绝对 loss 高，全局阈值判为"非成员"）
        ↑ 逐类归一化把难类整体拉低
易类 non-member（绝对 loss 低，全局阈值判为"成员"）
```

所以旗舰占优**必须同时满足两个条件**，缺一不可：

| 条件 | 作用 | 缺失时的后果 |
|---|---|---|
| ① **类间绝对 loss 水平差异大**（宽类难度方差） | 制造跨类反转 | 无反转可修，归一化纯加噪声 → 反而略输 |
| ② **适度过拟合**（member loss 未全趋 0） | 难类 member 仍保有高 loss | 全部被记忆到 loss≈0，无信号可重排 |

> ⚠️ **反直觉的实验事实**（见 `docs/BENCHMARK.md`）：把目标模型训到**完全记忆**
> （3000 epoch / hidden=256）时，所有 member loss 都塌到 ~0.01，跨类反转消失，
> 旗舰**退化到与全局阈值持平**。真正的甜点是 `hidden=16 / epochs=800`。
> 一味加训练量反而会**抹掉**旗舰的优势。

---

## 3. 结果（5 seed 聚合，见 `benchmark_result.json`）

配置：4 类高斯 blob，d=10，类标准差 ∈ [0.2, 1.3]，member 60/类，non-member 150/类，
目标模型 MLP(hidden=16, epochs=800, lr=0.05)，attack-train 比例 0.5。

| 攻击 | mean AUC | std | 说明 |
|---|---:|---:|---|
| `random` | 0.5000 | 0.0000 | 空基线（校准点） |
| `single_loss` | 0.6786 | 0.0244 | Yeom-2018 全局 loss 阈值 |
| `naive_stacked_lr` | 0.6799 | 0.0216 | **最强基线**：多特征全局 LR |
| `reference_z_only` | 0.7271 | 0.0190 | 旗舰的透明纯阈值版（对照） |
| **`reference_mia`** | **0.7271** | **0.0190** | **旗舰** |

**DoD 性能门槛（已达成）**

```
Δ mean AUC = +0.0472   (门槛 ≥ +0.03)          ✅
配对单侧 t  = 29.011    (门槛 ≥ 1.833, df=4)    ✅
确定性自检：同 seed 两次运行逐位一致            ✅
```

统计口径说明：逐 seed 独立估计的方差很大（不同 seed 数据分布本身在变），
因此显著性必须用**配对单侧 t 检验**——对同一 seed 取「旗舰 AUC − 基线 AUC」的差值，
检验其均值是否显著 > 0。这才是正确口径；用两个独立样本的 std 去比是错误的。

---

## 4. 快速开始

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.lock.txt     # Windows
# source .venv/bin/activate && pip install -r requirements.lock.txt   # Linux/macOS

python examples/demo.py          # 单 seed 演示 + 5 seed 基准 + DoD 门槛 + 确定性自检
python examples/demo.py --quick  # 快速冒烟
python -m pytest tests/ -q       # 14 项单元测试
```

`demo.py` 退出码 0 表示 DoD 与确定性自检双双通过；`1` 表示失败（CI 直接用它当门禁）。

---

## 5. 项目结构

```
perclass-mia/
├── core/           # 无业务依赖的底座
│   ├── config.py       # 配置 + ENV_MIA_* 覆盖 + schema 校验
│   ├── types.py        # Dataset / AttackResult / BenchmarkCell
│   ├── metrics.py      # 纯 numpy roc_auc / tpr_at_fpr / paired_one_sided_t
│   ├── seed.py         # 全局确定性播种
│   ├── errors.py       # 分层异常（早失败，不静默错算）
│   └── interfaces.py   # 攻击接口协议
├── data/
│   └── generators.py   # hetero / iso 两档高斯基准（可切类难度方差、类样本分布）
├── attack/
│   ├── targets.py      # 纯 numpy MLP 目标模型（零 torch）
│   ├── features.py     # loss / confidence / entropy / margin 特征
│   ├── baselines.py    # random / single_loss / global_loss_lr / naive_stacked_lr
│   ├── reference.py    # ★ 旗舰 ReferenceMIA（逐类参考归一化）
│   ├── lr.py           # numpy 逻辑回归（sklearn 缺失时自动降级）
│   └── registry.py     # 攻击注册表 + 别名（FLAGSHIP / STRONG_BASELINE）
├── pipeline/
│   └── attack_pipeline.py  # 端到端评测（run / benchmark）
├── tests/test_core.py      # 14 项测试（含 DoD 门槛与 iso 档非劣对照）
├── examples/demo.py        # 演示 + 基准 + DoD 自检
└── docs/                   # ARCHITECTURE / BENCHMARK / USAGE
```

调用链严格单向无环：`pipeline → {data, attack} → core`。

---

## 6. 无泄漏保证

成员推断评测最容易自欺的地方就是泄漏。PerClass-MIA 有三道闸：

1. **目标模型只在 member 半上训练**；non-member 与 member 同源同分布，仅未参与训练。
2. **攻击模型只在 attack-train 子集上拟合**，在 attack-test 子集上评分（按
   `member_flag` **分层抽样**，保证两侧比例一致）。
3. **参考分布只由 non-member 构造**——旗舰拟合 μ_c/σ_c 时严格 `flag == 0` 过滤，
   绝不会把待测成员的信息泄进参考分布。

另有 `test_no_overlap` 断言 train/test 索引交集为空。

---

## 7. 确定性

- 数据集结构（`difficulty_seed`）与 benchmark seed **解耦**：前者固定难度剖面，
  后者只改变抽样与模型初始化。
- 种子派生用 `hashlib.sha256`，**不用 Python 内置 `hash()`**（后者对字符串加进程盐，
  跨进程不可复现）。
- `demo.py` 的确定性自检断言同 seed 两次运行 AUC 差异 < 1e-12。

---

## 8. 复现实验与扩展

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
print(benchmark(cfg, [gen]).aggregate())
```

想验证第 2 节的机制断言，改两个旋钮即可：

- `MIADataGenerator("iso")` → 各类等难度，跨类反转消失 → 旗舰**退化为与基线持平**（有测试断言）
- `target_epochs=3000, target_hidden=256` → 完全记忆 → 旗舰优势**消失**

新增攻击：实现 `fit(preds, losses, member_flag, rng)` 与 `score(preds, losses)`，
在 `attack/registry.py` 注册即可，pipeline 无需改动。

---

## 9. 局限

- 目标模型是合成数据上的小 MLP，非真实大规模模型（如 CIFAR/ImageNet 上的 ResNet）。
  真实场景下"类难度异质"更普遍，旗舰优势应当**更**明显，但本仓库未直接验证。
- 旗舰仅用 loss 单特征做逐类归一化。实验显示接入 confidence/entropy 会因攻击训练集
  过小而过拟合，反而拖累 AUC；更大数据规模下的多特征融合仍是开放问题。
- `non-member` 与 `member` 严格同分布，这是 MIA 的标准假设；真实场景存在分布偏移，
  需要额外的校准步骤。

---

## 10. 许可

MIT License © 2026 晨星。详见 `LICENSE`。
