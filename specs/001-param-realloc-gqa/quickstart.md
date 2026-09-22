# Quickstart: 参数重分配与分组查询注意力

**Feature**: `001-param-realloc-gqa` | **Date**: 2026-09-19

所有命令在 `code/` 目录下执行。本机 PowerShell **禁止执行脚本**（ExecutionPolicy），
因此不使用 `Activate.ps1`，改为直接调用解释器的绝对路径。

---

## 环境

| 用途 | 解释器路径 | 版本 |
|---|---|---|
| **评估**（官方口径，锁定） | `C:\Users\Administrator\.conda\envs\gpt\python.exe` | Python 3.12.14 + torch 2.7.1+cpu |
| **训练**（GPU 加速） | `D:\anaconda\envs\pytorch\python.exe` | Python 3.10 + torch 2.8.0+cu128 |

两者均已安装 `tokenizers==0.21.4`（版本必须一致，否则分词结果可能不同）
与 `numpy`。评估环境额外锁定 `numpy==2.5.3`。

---

## 1. 契约测试（每次结构改动后必跑）

```powershell
C:\Users\Administrator\.conda\envs\gpt\python.exe -m unittest discover -s tests -v
```

期望：`Ran 5 tests ... OK`。任何一项失败即不得继续。

---

## 2. 复现基线锚点

```powershell
# 训练（CPU，与官方参考口径一致；约 668 秒）
C:\Users\Administrator\.conda\envs\gpt\python.exe train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline

# 打分（FP32，CPU；约 12 秒）
C:\Users\Administrator\.conda\envs\gpt\python.exe evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

期望：测试 BPB ≈ **2.1013**，验证 BPB ≈ 2.0711。

---

## 3. 快速训练（GPU，用于筛选）

```powershell
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c3.json --device cuda --precision bf16 --seed 17 --steps 1200 --run-dir runs/c3-s17
```

- 全协议 1200 步 = 9,830,400 targets，GPU 上约 23 秒（加约 20 秒数据加载）
- 每次都**必须使用新的 `--run-dir`**，否则 [train.py:30-31](../../code/train.py) 会拒绝执行
- `--precision bf16` 与 `fp32` 的选择见 research.md R6

### 3b. 最终提交配置（12,000 步）

`--steps` 会把 cosine 调度整体拉长（"同调度形状、更多目标数"）。已处理目标数
= `steps × batch × 256`，逐档对应 research.md R12 的实验 1。

```powershell
# 最终配置 L2 = (width=168, depth=2, MHA)，12,000 步 = 98,304,000 targets
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/w2.json --device cuda --precision bf16 --seed 17 --steps 12000 --eval-every 1200 --run-dir runs/w2-s17-r12000

# 等参数深度阶梯（12,000 步，含新建的 MHA 配置）
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/mhal5.json --device cuda --precision bf16 --seed 17 --steps 12000 --run-dir runs/mhal5-s17-r12000

# 三 seed 复现（最终选型判据）
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/w2.json --device cuda --precision bf16 --seed 42 --steps 12000 --run-dir runs/w2-s42-r12000
```

训练时长在评估预算之外不受限制（GUIDE 第 3 节），但**必须披露**：见 `code/RUN_LOG.csv`。
注意笔记本 GPU 在持续负载下会降频（实测 1,250 → 6,170 ns/token），故 wall clock 只作上限记录。

---

## 4. 评估候选（CPU FP32，官方口径）

```powershell
# 开发期选择用验证集
C:\Users\Administrator\.conda\envs\gpt\python.exe evaluate.py --checkpoint runs/c3-s17/checkpoint.pt --device cpu --precision fp32 --split validation

# 方法冻结后才允许跑测试集
C:\Users\Administrator\.conda\envs\gpt\python.exe evaluate.py --checkpoint runs/c3-s17/checkpoint.pt --device cpu --precision fp32 --split test

# 最终提交配置（已冻结）：L2 = (168, 2, MHA)，12,000 步
C:\Users\Administrator\.conda\envs\gpt\python.exe evaluate.py --checkpoint runs/w2-s17-r12000/checkpoint.pt --device cpu --precision fp32 --split test --threads 4
```

**纪律**: 冻结前只能看 `validation`；`test` 只在方法冻结后运行，且每个冻结模型只跑一次
（复跑仅用于 SC-008 可复现性校验）。提交的分数取 `test_cpu_fp32.json` 中的 `bpb`，不用 `token_ppl`。

---

## 5. 预算实测（每次结构改动后）

```powershell
# 打分时间与内存：见 spec.md SC-004 / SC-005，上限 60 秒 / 4 GiB
# 推理资产体积：
Get-Item runs/c3-s17/checkpoint.pt | Select-Object Name,@{N='MiB';E={[math]::Round($_.Length/1MB,2)}}
# 上限 64 MiB
```

打分时间直接读评估产物中的 `seconds` 字段。

---

## 6. 多 seed 确认性复跑

```powershell
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c3.json --device cuda --seed 42 --run-dir runs/c3-s42
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c3.json --device cuda --seed 123 --run-dir runs/c3-s123
```

---

## 7. 待产出（本计划尚未创建）

- `configs/c3.json` 等候选配置（在 `/sp.tasks` 与 `/sp.implement` 阶段创建）
- `student.py` 中的 MQA 实现（训练时保持 5 项契约测试通过）
- 提交用 git 仓库（当前项目根尚**不是** git 仓库，见 plan.md 风险 #8）

---

## 常见错误

| 现象 | 原因 |
|---|---|
| `Run directory already contains results` | 复用旧 `--run-dir`；换一个新目录 |
| `The main board fixes context=256 and vocab=2048` | 配置中的 `context` 或 `vocab` 被改动 |
| `The output is not a normalized probability distribution` | `predict_log_probs` 未归一化 |
| `Return finite log probabilities of shape [...]` | 输出形状错误或含 NaN/Inf |
| `Checkpoint belongs to a different course protocol` | checkpoint 的 `protocol` 字段被改动 |
| `Changed benchmark file: ...` | `data/` 下文件被修改，哈希校验失败 |