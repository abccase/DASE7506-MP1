# Model Interface Contract

**Feature**: `001-param-realloc-gqa` | **Date**: 2026-09-19
**Enforced by**: [code/tests/test_contract.py](../../code/tests/test_contract.py)（不可修改）

本契约是模型侧**不可协商**的接口约定。`common.py` 与 `evaluate.py` 依赖它，
违反即导致打分器抛错或分数无效。

---

## C1. 工厂函数 `build_model(config)`

```text
输入:  config — 满足 data-model.md E1 的字典
输出:  PyTorch nn.Module
前置:  config['context'] == 256 且 config['vocab'] == 2048
后置:  返回对象必须暴露属性 .context == 256
```

**违规后果**: `common.make_model` 抛出 `ValueError`（[common.py:57-58](../../code/common.py)）。

---

## C2. 训练接口 `forward(ids)`

```text
输入:  ids  — LongTensor [batch, time]，取值范围 [0, 2048)
输出:  logits — FloatTensor [batch, time, 2048]，未归一化
要求:  对位置 t，输出只能依赖 ids[:, :t+1]
用途:  由 train.py 计算交叉熵损失；需支持反向传播
```

**契约测试**: `test_shifted_loss_produces_gradients` —— 损失有限、梯度存在且有限、梯度非全零。

---

## C3. 评估接口 `predict_log_probs(ids)`

```text
输入:  ids  — LongTensor [batch, time]
输出:  logp — FloatTensor [batch, time, 2048]，归一化自然对数概率
硬性要求:
  (a) 形状必须严格等于 [batch, time, 2048]
  (b) 全部元素有限（torch.isfinite().all()）
  (c) 归一化：torch.logsumexp(logp, dim=-1).abs().max() <= 1e-3
  (d) 因果性：位置 t 只能使用 ids[:, :t+1]
  (e) 样本独立：batch 内样本互不影响
  (f) 状态重置：每次调用后不保留任何临时状态
```

**用途**: 由 evaluate.py 的唯一打分入口调用（[evaluate.py:21](../../code/evaluate.py)）。
**违规后果**: (a)(b)(c) 违反 → `ValueError`；(d)(e)(f) 违反 → 分数无效，且构成对评测协议的破坏。

---

## C4. 契约测试映射

| 测试 | 验证条目 | 失败含义 |
|---|---|---|
| `test_future_inputs_cannot_change_earlier_predictions` | C3(d) | 存在未来 token 泄漏 |
| `test_probabilities_are_normalized_and_examples_independent` | C3(a)(c)(e) | 分布未归一化或样本间串扰 |
| `test_state_resets_between_windows` | C3(f) | 跨窗口状态残留 |
| `test_shifted_loss_produces_gradients` | C2 | 训练路径断裂 |
| `test_every_target_is_counted_once_including_last_short_window` | 窗口切分 | 目标重复计数或遗漏（针对 `common.windows`） |

**强制流程**: 任何模型结构改动之后 MUST 重跑 `python -m unittest discover -s tests -v`，
5 项全部通过方可继续（宪法第 III 条）。

---

## C5. 打分协议契约（不可修改）

```text
协议:      7506-mp1-wt2-v2
窗口:      独立 256-token 因果窗口，含最后一个短窗口
目标:      除每个 split 的第一个 token 外，每个目标恰好计一次
BPB:       所有目标负 log2 概率之和 / 该 split 的原始 UTF-8 字节数
精度:      正式排名使用 FP32
分母:      验证集 1,148,007 bytes；测试集 1,292,013 bytes
```

**共享边界 token**: 相邻窗口共享一个边界 token，但**不携带状态**。这是协议设计，不是可用状态通道。

---

## C6. 提交包契约

```text
必需内容:
  1. implementation 字段指向的模块（含全部依赖模块）
  2. 匹配的 checkpoint
  3. 精确的安装 / 训练 / 评估命令
禁止:
  - 依赖运行时网络访问
  - 依赖训练步骤即可完成的评估流程
  - 超过 64 MiB 未压缩的推理资产
```

**验证方式**: 在干净环境中按 quickstart.md 执行，不进行任何训练，
复现测试 BPB 与报告值差异 < 1e-3。