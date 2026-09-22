# Phase 1 Data Model: 参数重分配与分组查询注意力

**Feature**: `001-param-realloc-gqa` | **Date**: 2026-09-19
**Input**: [spec.md](./spec.md) 的 Key Entities 一节

本文件把 spec 中的概念实体落为**可核对的数据结构**。不含代码，只定义字段、约束与关系。

---

## E1. 模型配置（ModelConfig）

区分基线与各候选版本的唯一标识，随 checkpoint 一同记录。

| 字段 | 类型 | 约束 | 说明 |
|---|---|---|---|
| `vocab` | int | **必须为 2048** | 契约固定，任何改动即违规 |
| `context` | int | **必须为 256** | 契约固定，任何改动即违规 |
| `width` | int | ≥ 1，且能被 `heads` 整除 | 模型宽度 `d` |
| `heads` | int | ≥ 1 | 注意力头数 `h` |
| `depth` | int | ≥ 1 | Transformer 层数 `L` |
| `kv_heads` | int | 1 ≤ `kv_heads` ≤ `heads`；**缺省值为 `heads`** | KV 头数 `k`。`k = heads` 即标准多头注意力（MHA）；`k = 1` 即多查询注意力（**MQA**，是分组查询注意力 GQA 在 `k=1` 时的特例）；`1 < k < heads` 为一般 GQA |

**派生量（不入库，由公式计算并写入运行记录）**

- `parameters`：`2306d + L·(attn + mlp + 4d)`
- `macs_per_token`：`L·(attn_macs + 8d²)`

**校验规则**

- `context == 256 and vocab == 2048`，否则 `common.make_model` 会抛错（[common.py:57](../../code/common.py)）
- `kv_heads` 字段缺失时 MUST 按 `heads` 处理，以保证既有的 `configs/baseline.json`（无该字段）
  在 `--implementation student` 下与基线行为一致
- 参数量与基线差异超过 ±3% 时，该配置**不得**用于 SC-002 的公平对比，只能作为披露差异的消融臂

---

## E2. Checkpoint

评估方重建预测器的唯一入口。字段由 [train.py:77-79](../../code/train.py) 写入。

| 字段 | 类型 | 约束 | 用途 |
|---|---|---|---|
| `protocol` | str | **必须等于** `7506-mp1-wt2-v2` | 评估器校验；不匹配直接拒绝 |
| `implementation` | str | 可导入的模块名 | 决定 `importlib` 加载哪个模块的 `build_model` |
| `config` | ModelConfig | 满足 E1 约束 | 重建模型结构 |
| `model` | state_dict | 键与结构一致 | 权重 |
| `seed` | int | — | 可复现性 |
| `train_tokens` | int | — | 已处理目标数，用于成本披露 |

**校验规则**

- 提交包 MUST 包含 `implementation` 指向的模块及其全部依赖，否则评估方无法重建
- 不需要优化器状态
- `implementation` 指向的模块必须在**无本项目其它私有文件**的情况下可导入

---

## E3. 运行记录（RunRecord）

一次训练的完整度量，是成本披露与数字核对的数据源。由 `metrics.json` 承载。

| 字段 | 说明 |
|---|---|
| `implementation` / `config` / `seed` | 运行标识 |
| `parameters` | 参数量（核对 E1 的派生量） |
| `train_tokens` | 已处理训练目标数 |
| `precision` | 训练精度（bf16 / fp32） |
| `preparation_seconds` | 数据加载耗时（约 20 s 固定开销） |
| `train_seconds` | 纯训练耗时 |
| `validation` | 验证集结果对象（含 `bpb`、`targets`、`utf8_bytes`、`seconds`） |
| `history` | 每 100 步的 loss 与累计耗时 |
| `checkpoint_sha256` / `implementation_sha256` | 产物哈希，用于追溯 |
| `torch_version` / `threads` / `device_*` | 环境记录 |

**校验规则**

- `checkpoint_sha256` MUST 与运行目录中 checkpoint 的实际哈希一致
- `implementation_sha256` MUST 与提交的模型模块一致；不一致表示代码在产出后被改动
- 训练集选择 MUST NOT 出现测试集指标

---

## E4. 评估产物（EvalResult）

由 [evaluate.py:57-66](../../code/evaluate.py) 写出，是"分数"的载体。

| 字段 | 说明 |
|---|---|
| `bpb` | **排名指标**，越低越好 |
| `token_ppl` | 仅供内部比较，**不得**用于排名 |
| `nll_nats` / `targets` / `utf8_bytes` | `bpb = nll_nats / ln2 / utf8_bytes`，可反算核对 |
| `split` / `precision` | 必须为 `test` / `fp32` 才算正式测试分 |
| `evaluator_sha256` / `tokenizer_sha256` | 证明评估口径未被改动 |
| `seconds` | 用于预算核对 |

**配套文件**: `<split>_<device>_<precision>.window-nll.npy`，逐窗口 NLL，用于误差分析。

**校验规则**

- 正式提交分数 MUST 来自 `--split test --precision fp32` 的产物
- 验证集产物只用于开发选择，MUST NOT 出现在报告中作为最终分数

---

## E5. 实验单元（ExperimentUnit）

一次受控比较的定义。是消融与对照的记账单位。

| 字段 | 说明 |
|---|---|
| `arm_id` | 如 `BASE` / `C3` / `C4` |
| `config` | 指向 ModelConfig |
| `seed` | 单一随机源 |
| `processed_targets` | **固定为 9,830,400** 才可与基线对比 |
| `precision` | 训练精度 |
| `parent_checkpoints` | 沿革关系（如 EMA 或续训的父 checkpoint） |

**校验规则**

- 用于 SC-002 的对比中，`processed_targets` MUST 完全相等
- 每个实验单元 MUST 有独立 `run-dir`（[train.py:30-31](../../code/train.py) 会拒绝非空目录）
- `parent_checkpoints` 非空时，祖先的训练成本 MUST 计入总成本（宪法第 VI 条）

---

## E6. 基线锚点（BaselineAnchor）

所有对比的参照，MUST 在本机重测而非引用他人数值。

| 字段 | 本机实测值 |
|---|---:|
| `test_bpb` | 2.1013 |
| `validation_bpb` | 2.0711 |
| `train_seconds_cpu4` | 667.6 |
| `eval_seconds_cpu_fp32` | 11.86 |
| `peak_eval_ram_gib` | 1.56 |
| `asset_mib` | 4.17 |

**派生上限**: 打分 ≤ 60 s、内存 ≤ 4 GiB、资产 ≤ 64 MiB。

**校验规则**

- 参照环境（torch 版本、线程数、设备）变更时 MUST 重新建立锚点
- 锚点 MUST 与 `runs/baseline/` 中的产物哈希绑定

---

## 实体关系

```text
BaselineAnchor ──derives──> 预算上限
      ▲
      │ compare
ModelConfig ──build──> 模型 ──train──> Checkpoint ──evaluate──> EvalResult
      │                                    ▲                        │
      └──used by──> ExperimentUnit ────────┘                        │
                          │                                        │
                          └── recorded in ──> RunRecord <─── verify ┘
```

- 一个 `ExperimentUnit` 产生一个 `RunRecord` 与一个 `Checkpoint`
- 一个 `Checkpoint` 可被多次评估（冻结后允许重复评估用于计时/复现）
- `EvalResult` 的 `checkpoint_sha256` 反向锚定到具体 `Checkpoint`，保证分数可追溯