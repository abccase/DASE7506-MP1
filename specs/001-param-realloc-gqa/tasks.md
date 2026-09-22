# Tasks: 参数重分配与分组查询注意力改进

**Input**: Design documents from `/specs/001-param-realloc-gqa/`
**Prerequisites**: [plan.md](./plan.md)（必需）、[spec.md](./spec.md)（用户故事）、[research.md](./research.md)、[data-model.md](./data-model.md)、[contracts/model-interface.md](./contracts/model-interface.md)
**Constitution**: `.specify/memory/constitution.md` **v1.1.0**

**Tests**: 本作业的测试由课程提供且**不可修改**（[code/tests/test_contract.py](../../code/tests/test_contract.py)）。
因此不存在"编写测试任务"，而是把**运行既有契约测试**作为强制验证门。
唯一自行新增的测试是 `tests/test_equivalence.py`（数值等价性防回归），见 T006。

**术语约定**: 本项目实际使用 `kv_heads = 1`，即**多查询注意力（MQA）**，
是分组查询注意力（GQA）在 `kv_heads = 1` 时的特例。文中统一以 MQA 指代该机制。

**plan 阶段映射**: plan 的阶段字母 A–G 与本文 Phase 的对应见 [plan.md](./plan.md) 的"阶段与任务编号映射"。
**阶段 F（EMA / 延长训练 / 1D 免 weight decay）已显式延后，不纳入本任务集。**

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）
- **[Story]**: 所属用户故事（US1–US4）
- 所有路径以 `code/` 或 `specs/` 为基准

---

## Phase 1: Setup

**Purpose**: 建立候选配置，使各配置可被 `train.py --config` 直接消费

- [X] T001 [P] 创建主候选配置 `code/configs/c3.json`（`width=112, depth=6, heads=4, kv_heads=1, vocab=2048, context=256`）
- [X] T002 [P] 创建消融臂配置 `code/configs/c4.json`（`d=112, L=6, kv_heads=4`，即 MHA）与 `code/configs/c1.json`（`d=128, L=4, kv_heads=1`）
- [X] T003 [P] 创建变体配置 `code/configs/c6.json`（`d=120, L=5, kv_heads=1`）与 `code/configs/c5.json`（`d=96, L=8, kv_heads=1`）
- [X] T004 [P] 创建 GPU 对照基线配置 `code/configs/base.json`（内容等价于 `configs/baseline.json`，显式写出 `kv_heads=4`，用于与候选同精度对照）

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: 统一注意力实现、防回归验证与协议完整性校验。**此阶段完成前不得开始任何用户故事。**

**⚠️ CRITICAL**: 未通过 T007 的 5 项契约测试，后续所有训练结果均无效。

- [X] T005 [P] 在 `code/student.py` 实现统一注意力模型：支持 `kv_heads` 参数，`kv_heads == heads` 时退化为标准多头注意力；**`kv_heads` 字段缺失时按 `heads` 处理**（保证 `configs/baseline.json` 兼容，见 data-model.md E1 校验规则）
- [X] T006 [P] 在 `code/tests/test_equivalence.py` 添加数值等价性检查：将基线融合的 `qkv` 权重拆分为 q/k/v 后载入学生模型，比较 `forward` 与 `predict_log_probs` 输出。**容差为 atol=1e-5 / rtol=1e-4**（非原定的 1e-6）：基线用单个 fused `[3d, d]` 矩阵，学生模型用三个 `[d, d]` 矩阵，数学等价但在 fp32 下非逐位相等
- [X] T007 运行契约测试并确认 5 项全通过：`python -m unittest discover -s tests -v`
- [X] T008 核对实现的实际参数量与 `research.md` R2 表格一致（BASE/C1/C2/C3/C4/C5/C6 逐一比对）
- [X] T009 **协议不可变性校验**（宪法第 II 条）：比对 `code/PACKAGE_MANIFEST.json` 中的 sha256，确认 `code/common.py`、`code/evaluate.py`、`code/model.py`、`code/tests/test_contract.py`、`code/configs/baseline.json`、`code/data/*` 全部未被修改。**此项在每次提交前必须重跑**

**Checkpoint**: 实现可信、参照基准可信，开始实验。

---

## Phase 3: User Story 1 - 同 token 预算下的公平改进证据 (Priority: P1) 🎯 MVP

**Goal**: 在与基线完全相同的已处理训练目标数（9,830,400）与近似相同的参数量下，证明改进配置的验证集 BPB 严格更低。

**Independent Test**: 比较 `runs/base-s17/` 与 `runs/c3-s17/` 的 `metrics.json` 中 `validation.bpb`，
以 T011 得到的门槛判断差异是否超过 2σ。

### Implementation for User Story 1

- [X] T010 [US1] 阶段 A：用 `code/configs/base.json` 在 GPU 上以 seed 17/42/123 各跑 1200 步（9,830,400 targets），产出 `code/runs/base-s17|s42|s123/`（实测 BPB：2.072792 / 2.075514 / 2.077814；单次约 35 s）
- [X] T011 [US1] 由 T010 三个验证 BPB 计算均值与标准差 σ；门槛取 `max(σ_3seed, 基线已知波动)`。**结果：均值 2.075373，σ=0.002514，门槛 = 0.0050 BPB（相对 0.24%）**；详见 research.md R8
- [X] T012 [US1] 阶段 B：用 `code/configs/c3.json` 跑 seed 17 完整协议，产出 `code/runs/c3-s17/`。**验证 BPB 2.111076**（基线 seed17 2.072792，**+0.038284**）
- [X] T013 [P] [US1] 阶段 B：用 `code/configs/c4.json` 跑 seed 17 完整协议，产出 `code/runs/c4-s17/`。**验证 BPB 2.112802**（**+0.040010**）
- [X] T014 [P] [US1] 阶段 B：用 `code/configs/c1.json` 跑 seed 17 完整协议，产出 `code/runs/c1-s17/`。**验证 BPB 2.079097**（**+0.006305**，略高于门槛）
- [X] T015 [P] [US1] 阶段 B：用 `code/configs/c6.json` 跑 seed 17 完整协议，产出 `code/runs/c6-s17/`。**验证 BPB 2.105199**（**+0.032407**）
- [X] T016 [P] [US1] 阶段 B：用 `code/configs/c5.json` 跑 seed 17 完整协议，产出 `code/runs/c5-s17/`。**验证 BPB 2.166496**（**+0.093704**）

> **⚠️ 阶段 B 结果：全部 5 个候选均劣于基线**，且沿等参数前沿呈单调趋势
> （宽度越窄、层数越多 → 越差）。主机制假设未获支持。详见 research.md R9。
> 这触发了 T017 的决策分叉，须先完成机制陈述改写与审批，再继续 T018 之后的任务。
- [X] T017 [US1] 汇总 T012–T016 的 `validation.bpb` 并选出排名前 2 名。**决策分叉已触发并处理**：最佳候选改进为**负值**（远低于 3%），据此按 spec 核对清单第 3 条回写机制陈述并重过审批门；**SC-001 门槛按决议保持原值不动**（不事后收紧、不放宽）。详见 PHR-010/PHR-011
- [X] T018 [US1] 对进入最终对比的前 2 名候选实测三项预算（打分时间 / 峰值内存 / 资产体积）（宪法 v1.1.0 第 V 条）。**实测（验证集，CPU FP32）**：W7 = 8.58 s / 1.801 GiB / 4.29 MiB；W5 = 8.87 s / 1.800 GiB / 4.17 MiB。基线同口径为 9.30 s / 1.56 GiB / 4.17 MiB → **三项均达标，且打分更快**
- [X] T019 [US1] 阶段 D：为胜出配置 **W7** 补跑 seed 42/123。**三 seed 结果：1.943094 / 1.948854 / 1.954273，均值 1.948740，σ = 0.005590**。与 BASE 三 seed 均值 2.075373 的差为 **0.126633**（相对 **−6.10%**）；差的标准误 0.003539，t ≈ 35.8 → 极显著。且 W7 最差 seed（1.954273）仍优于 BASE 最好 seed（2.072792）达 0.1185，两者完全分离
- [X] T020 [US1] 判定 SC-002：以 **W5 (d220 L1, 参数量 +0.25%, 验证 BPB 1.959555)** 作为满足 ±3% 参数带的对照点，相对 BASE 均值改进 0.115818，远超门槛 → **SC-002 通过**。**W7 为 +3.06%，微超自设带**，作为"带外探针"单独披露，未用于 SC-002 判定

> **⚠️ 机制第二次改写（2026-09-19）**：阶段 B 否定了"MQA 换深度"，阶段 B′ 反向搜索确认
> **宽度支配深度**，最优 `W7 = (224, 1, MHA)` 验证 BPB **1.9431（−6.26%）**。
> `spec.md` / `plan.md` 已回写，`research.md` R9/R10 记录了两次否定与完整前沿表。
> 下方 US2 阶段已按新机制重新设计。

---

## Phase 4: User Story 2 - 关键机制的消融证据 (Priority: P1)

**Goal**: 通过**在等参数前沿上移动容量分配**（宽度 vs 深度），测量该分配决策单独的贡献，证明改进的因果来源。

**Independent Test**: 比较等参数前沿上各配置（`(128,4)`、`(144,3)`、`(168,2)`、`(220,1)`、`(224,1)`）的
验证 BPB，检验序列是否严格单调且每一档差异均超过门槛（0.0050），并显式列出各点参数量与计算量。

### Implementation for User Story 2

- [X] T021 [US2] 组装等参数前沿消融表。**已完成**：完整表见 `research.md` R10——列出每个点的 (宽度, 层数)、参数量及相对基线偏差、每 token MACs 及偏差、验证 BPB 与相对改进；并显式标注 W6 (+6.03%) 与 W7 (+3.06%) 超出 ±3% 带。表头前提：所有权重与统计量仅派生自训练文本
- [X] T022 [US2] 判定主机制陈述：**已执行**——C4 (2.112802) 未击败 C3 (2.111076) 但差异 −0.0017 远小于门槛，故"KV 共享有益"**不成立**；且沿等参数前沿加深单调变差，初版机制被否定。机制已改写并回写 `spec.md`/`plan.md`
- [X] T023 [US2] 校验 SC-003：最优 (224,1) 三 seed 均值 1.948740 对基线 (128,4) 三 seed 均值 2.075373，差 **0.126633**，为 0.01 门槛的 12.7 倍 → **通过**。**最弱一档**校验：(144,3) 的 −0.031203 仍为噪声门槛 0.0050 的 6.2 倍 → 趋势可靠，非仅端点差异
- [X] T024 [US2] 记录"何种结果会否定新机制"的对应关系，作为报告的批判性分析素材；并**必须**同时保留两版被否定机制的否定证据（FR 与 US2 验收场景 4 要求）。**已完成**：可证伪清单以表格形式落于报告 §6.6（4 条预测 / 各自的反证观测 / 若观测到即否定该机制）；两版被否机制的否定证据分别落于 §6.1——MQA 中性（C3 vs C4 差 −0.001726，远小于门槛）与沿前沿加深单调变差（C1/C6/C3/C5 依次劣化）；并说明改写后的方向源自"利用失败的梯度"反向搜索

**Checkpoint**: US1 与 US2 均独立可交付——"是否更好"与"为什么更好"两个问题各有答案。

---

## Phase 5: User Story 3 - 预算合规且可复现的交付物 (Priority: P2)

**Goal**: 产出冻结的预测器与匹配的 checkpoint bundle，使第三方无需训练即可在预算内复现分数。

**Independent Test**: 在一台干净机器上按 `quickstart.md` 执行，不进行任何训练，
复现测试 BPB 与报告值差异 < 1e-3，且三项预算均达标。

### Implementation for User Story 3

- [X] T025 [US3] 对最终模型重跑契约测试并重跑协议不可变性校验。**结果：14/14 测试通过（含 5 项官方契约测试）；13/13 协议文件哈希匹配**
- [X] T026 [US3] 实测完整测试集 FP32 CPU 打分时间。**结果：9.79 s ≤ 60 s**（基线同口径 11.86 s，即 **0.83×，比基线更快**）
- [X] T027 [P] [US3] 实测评估流程峰值内存。**结果：1.801 GiB ≤ 4 GiB**
- [X] T028 [P] [US3] 实测推理资产体积。**结果：4.29 MiB ≤ 64 MiB**
- [X] T029 [US3] **冻结方法**并对测试集运行一次 FP32 CPU 打分。**已完成**：冻结配置 `W7 = (d224 L1 MHA)`，seed 17，checkpoint `runs/w7-s17/checkpoint.pt`；产出 `runs/w7-s17/test_cpu_fp32.json`，已评分 428,405 个目标（完整测试集）
- [X] T030 [US3] **判定 SC-001**：**测试 BPB = 1.974276 ≤ 2.038 → 通过**（余量 0.064）。相对基线 2.101260 改进 **−6.04%**。门槛按决议未做任何修改
- [X] T031 [US3] 重跑同一评估命令一次校验可复现性（SC-008）。**结果：两次测试 BPB 差异 = 0.000E+000（逐位完全相同）**，远优于 1e-3 要求
- [X] T032 [US3] 干净环境重建验证（SC-009）。**已核验**：`student.py` 自包含（仅依赖 torch，不导入 `model.py`）；checkpoint（4.29 MiB）已写入 `config` 与 `implementation` 字段；`evaluate.py` 仅凭 checkpoint + `student.py` 即可重建并复现分数，全程无训练步骤、无网络访问

**Checkpoint**: 分数可复现且预算合规——成绩可被承认。

---

## Phase 6: User Story 4 - 完整的方法与成本披露 (Priority: P3)

**Goal**: 产出 ≤10 页英文报告与完整运行记录，使第三方能重建方法描述、核对所有数字、追溯每个 checkpoint。

**Independent Test**: 依据报告与运行记录，第三方能核对任一数字到对应 `runs/` 产物，并追踪 checkpoint 来源。

### Implementation for User Story 4

- [X] T033 [US4] 汇总全部运行与搜索成本到**新建的** `code/RUN_LOG.csv`（以 `code/RUN_LOG_TEMPLATE.csv` 为表头参考，**不修改模板文件**），含所有 seed、processed targets、ancestry、筛选开销。**已完成**：24 列 × 24 行（表头 + 23 条记录），字段含 `seed`、`processed_targets_including_ancestry`、`parent_checkpoints`、`checkpoint_sha256`、`selected_on`、`notes`；覆盖 19 次训练（18 GPU + 1 CPU 基线）与 4 次评估。模板文件未修改
- [X] T034 [US4] 撰写英文报告（≤10 页）：方法、同 token 对照、消融、批判性分析（含预测质量与计算成本的权衡）。**已完成**：`code/REPORT.md`，3,916 词 / 6 表格 / 3 代码块（约 8 页，≤10 页限制内）；含 Abstract、协议与约束、基线诊断、方法、结果（同 token 对照 + 等参数前沿消融）、成本披露、批判性分析（含三解释未区分、n=3 局限、可证伪清单）、复现步骤、AI 辅助披露
- [X] T035 [US4] 交叉核对报告中每一个数字与 `runs/` 下的产物 JSON，确保一致；并确认报告页数 ≤10 且三项证据齐备（SC-010）。**已完成**：以独立脚本重算全部 MACs / 百分比 / 统计量 / 成本 / 页数并逐项比对 `runs/*/metrics.json` 与评估 JSON，修正 5 处四舍五入偏差（C1 −9.37→−9.38、C6 +1.40→+1.39、W2 −1.75→−1.76 与 −19.00→−18.99、W8 −5.26→−5.27）与 1 处条目错位（W7 每 token MACs 误填 W5 的 693,440/−33.9%，更正为 716,800/−31.6%，并同步回改 research.md、plan.md、PHR-011）；页数 ≤10 与三项证据（本机重测基线 / 同 token 对照 / 等参数前沿消融）均确认齐备

**Checkpoint**: 全部四项交付物齐备。

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: 提交所需的仓库形态与流程动作

- [ ] T036 初始化 git 仓库并完成首次提交（当前项目根**尚不是** git 仓库，见 plan.md 风险 #8）
- [ ] T037 [P] 编写 `code/README` 中的复现说明：精确安装、训练、评估命令与环境版本
- [ ] T038 [P] 在报告中声明 AI 辅助范围（依 GUIDE 第 3 节要求）
- [ ] T039 于 9 月 29 日前在课程网站提交学号与全测试 BPB，并完成生成的 GitHub issue；9 月 30 日前完成最终提交（不可变代码 + checkpoint bundle）

---

## Phase 8: 机制修正与重新冻结 (2026-09-21/22)

**Goal**: 执行 R10 中标注"未执行"的区分实验（给深模型更多步数），据实测结果修正机制表述、
在饱和步数下重新选型并重新冻结。

**Independent Test**: 同配置只改 `--steps` 是否使 W7 对基线的优势单调衰减并反转；在 12,000 步
重跑等参数阶梯后，最优点是否离开浅端。

### Implementation

- [X] T040 [US2] 训练目标数扫描（Regime 2）：`224/L1` 与 `128/L4` 成对跑到 1,200/2,400/3,600/4,800/6,000/12,000 步。**结果**：优势由 −0.129698 单调衰减至 +0.030654，线性插值过零点 **5,844 步（47,872,055 targets，13.2 epochs）**
- [X] T041 [US2] 在饱和步数（12,000）重跑等参数深度阶梯。**新建 4 个 MHA 配置**（`mhal5..8`，因原 `c6/c3/c5` 为 MQA，会混入第二个变量），8 点全部落在基线 ±3% 内。**结果**：最优由 L1(d224) 移到 **L3(d144)=1.676193**，L1 反而最差（+0.039020）
- [X] T042 [US2] 长训练机制下的三 seed 复现与选型。**结果**：单 seed 排名未复现（名义最优由 L3 移到 L2）；均值差 0.007212、差值标准误 0.005327、**t=1.35 不显著** → 按成本选型（L2 少 15,024 参数、少 118,272 MACs）。冻结取**预登记 seed 17**（非三 seed 中验证最优的 seed 123），避免 seed 挑选进入记录分
- [X] T043 [P] [US2] 三项中性杠杆验证：注意力头数（h=2/7/14 → 1.948981/1.954492/1.965600）、EMA（0.995，相对本 run 原始权重 +0.002284）、1D 参数免 weight decay（+0.003351）。**三者均亚门槛**
- [X] T044 [US3] **重新冻结**为 `L2 = (168, L2, MHA)`，seed 17，12,000 步，并重跑测试集与三项预算、契约测试、协议完整性校验。**结果**：测试 BPB **1.700525**（−19.07%）、10.833 s（基线 0.91×）、1.8004 GiB、4.09 MiB、复跑差 **0.000E+000**、14/14 测试、13/13 协议文件
- [X] T045 [US4] 回写 `research.md` R12（含机制修正、解释判定、选型、噪声底机制相关性、结论变更）、`spec.md` 第三次机制修订说明、`plan.md` 摘要、`quickstart.md`（12,000 步配方）、`REPORT.md`（全文改写）与 `RUN_LOG.csv`（新增 29 行）
- [X] T046 [P] `train.py` 新增 `--weight-decay-1d` 与 `--ema-decay` 两个开关。**默认值经验证为数值无操作**（对照运行 1.943105 vs 1.943087 vs 1.943094，差 1.8e-05）

**Checkpoint**: 机制表述与实测一致；最终配置、分数、预算与文档全部对齐。

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 (Setup)**: 无依赖，可立即开始
- **Phase 2 (Foundational)**: 依赖 Phase 1；**阻塞全部用户故事**（T007 未通过则一切训练结果无效）
- **Phase 3 (US1)**: 依赖 Phase 2
- **Phase 4 (US2)**: 依赖 Phase 3 的 T012–T016（消融臂数据即筛选臂数据，**不额外消耗训练成本**）
- **Phase 5 (US3)**: 依赖 Phase 3 的 T017–T019（胜出候选）
- **Phase 6 (US4)**: 依赖 Phase 3–5 全部结果
- **Phase 7 (Polish)**: 依赖 Phase 5、6
- **Phase 8 (机制修正)**: 依赖 Phase 5、6；其 T044 重新冻结**取代** T029 的冻结，T045 的
  文档回写**取代** R11 的结论表述；T029–T031 作为历史记录保留，不删除

### Critical Path

```
T001–T004 → T005 → T006 → T007 → T008 → T009
  → T010/T011 (噪声底，必须先于任何对比结论)
  → T012–T016 (筛选)
  → T017 → T018 → T019 → T020 (US1 完成)
  → T021 → T022 → T023 → T024 (US2 完成)
  → T025 → T026–T028 → T029 (冻结) → T030 → T031 → T032 (US3 完成)
  → T033 → T034 → T035 (US4 完成)
  → T040 → T041 → T042 → T043 → T044 (重新冻结) → T045 (文档回写) → T046
  → T036 → T037 → T038 → T039 (提交)
```

### 关键依赖说明

- **T011 必须先于 T017**：没有噪声底就无法判断候选差异是否真实（宪法第 VII 条精神）
- **T018 必须在 T017 之后、T019 之前**：先排除超预算结构，再投入确认性复跑
- **T029 冻结门是唯一不可逆闸门**：之后禁止任何基于测试集的选择；若需改动，必须解除冻结
  并重跑 T029。T044 正是按此执行：先解除冻结、重选配置、再重新冻结并重跑测试集
- **T042 必须先于 T044**：选型判据（三 seed 均值与差值标准误）必须在冻结之前确定；
  0.0050 门槛是为**单 seed 差值**标定的，不得直接套用于**均值差**
- **T022 是决策分叉点**：其结论可能回写 `spec.md`/`plan.md`，属框架允许的"实现回到计划"，
  但必须重过审批门。T045 适用同一规则
- **T009 在每次提交前重跑**：它是"参照基准未被污染"的唯一机械保证

---

## Parallel Opportunities

同一阶段内标记 `[P]` 的任务可并行执行。

```powershell
# Phase 1：4 个配置可同时创建（不同文件）
# T001 configs/c3.json / T002 c4.json+c1.json / T003 c6.json+c5.json / T004 base.json

# Phase 3 筛选：5 个候选互不依赖，可并行（GPU 显存充裕，单次峰值约 0.42 GB）
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c3.json --device cuda --seed 17 --run-dir runs/c3-s17
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c4.json --device cuda --seed 17 --run-dir runs/c4-s17
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c1.json --device cuda --seed 17 --run-dir runs/c1-s17
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c6.json --device cuda --seed 17 --run-dir runs/c6-s17
D:\anaconda\envs\pytorch\python.exe train.py --implementation student --config configs/c5.json --device cuda --seed 17 --run-dir runs/c5-s17

# Phase 5：三项预算测量互不依赖（T026 时间 / T027 内存 / T028 体积）
```

**注意**: `train.py` 会拒绝非空的 `--run-dir`，所以"并行"指不同 `run-dir` 同时执行，
而非同一目录重复运行。所有训练臂**必须使用 `--implementation student`**（含 BASE），
以保证基线与候选走同一代码路径——其等价性由 T006 保证。

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1 + Phase 2（配置、实现、校验，约 30 分钟）
2. Phase 3（噪声底 + 筛选 + 确认，约 7 分钟 GPU 时间）
3. **STOP and VALIDATE**: 确认 SC-002 → 此时已有一份完整可报告的结论

### Incremental Delivery

1. US1（P1）→ 独立验证"是否更好" → 可报告的最小成果
2. US2（P1）→ 独立验证"为什么更好" → 论文的因果论证闭环
3. US3（P2）→ 独立验证"分数可复现" → 成绩可被承认
4. US4（P3）→ 报告与披露 → 交付完成

### 成本纪律

- 筛选臂复用为消融臂（T012–T016 的数据同时服务 US1 与 US2），**不重复训练**
- BASE 的 3 个 seed 只在 T010 产出一次，T019 不得重跑
- 全协议筛选（1200 步）而非缩短训练，因 GPU 下单次仅约 43 秒
- 训练运行合计 **10 次**（阶段 A 3 + 阶段 B 5 + 阶段 D 2），约 7 分钟 GPU 时间
- 所有运行必须记入 `runs/*/metrics.json`，最终汇总到 T033

---

## Notes

- `[P]` 表示不同文件、无未完成依赖
- 契约测试由课程提供且不可修改；唯一新增测试为 `tests/test_equivalence.py`（防回归）
- 每次模型结构改动后必须重跑契约测试；每次提交前必须重跑 T009 哈希校验
- 冻结前只看 validation；test 仅在冻结后运行，且**每个冻结模型只跑一次**（复跑仅用于 SC-008）
- 若 T022 判定主机制不成立，按框架允许返回计划阶段并重过审批门，不得静默降低要求
- **阶段 F 已于 Phase 8 执行完毕**：延长训练（→12,000 步，有效）、EMA（亚门槛）、
  1D 免 weight decay（亚门槛），另加注意力头数扫描（亚门槛）。结论与否定证据一并保留
- **SC-003 的适用范围已明确限定**：仅在 9.83M-targets 机制下成立；在最终提交的 98.3M-targets
  机制下不成立。门槛**未被**事后放宽或重定义
- 测试集在本项目历史中被评分两次（W7 与 L2，各在冻结后一次，均逐位复现）；
  **没有任何测试值参与过任何选择**，此事实必须在报告中显式披露