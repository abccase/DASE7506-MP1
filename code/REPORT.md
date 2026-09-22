# The Width/Depth Optimum Is a Function of the Training Budget

**MP1 — Small Language Model Challenge (DASE7506)**
**Student ID: 3036747264**
**Recorded full-test BPB: 1.700525** (protocol `7506-mp1-wt2-v2`, CPU, FP32)

*Length: about 4,650 words including 7 tables (roughly 4,000 words of prose), which renders
inside the 10-page limit at a standard 11 pt single-column layout. The count is stated so a
reader can check it rather than trust it.*

---

## Abstract

The supplied baseline is a 1,088,256-parameter GPT (width 128, 4 blocks, 1,200 steps) scoring
2.101260 bits per byte (BPB) on the full WikiText-2 test split. This report studies two
independent levers: **where** a fixed parameter budget is spent (width versus depth) and **how
long** the model is trained.

At the supplied training budget the first answer is sharp: on an equal-parameter frontier,
moving capacity from depth into width improves BPB monotonically, and the best point (width
224, depth 1) beats the baseline by 0.129698 BPB, i.e. 6.26%, while using 31.6% fewer
multiply-accumulates per token. That looked like a general architectural law. It is not. Holding the architecture fixed
and varying only the number of processed training targets shows the width advantage **decays
monotonically and reverses**: it crosses zero at about 5,844 steps (47.9M targets, 13.2
epochs), and at 12,000 steps the deep baseline is 0.030654 BPB ahead. Re-running the whole
frontier at 12,000 steps moves the optimum from depth 1 (d=224) to **depth 3 (d=144)**, with
the wide-shallow extreme now the *worst* point in the ladder. **The optimal width/depth
allocation is a function of the training budget, not a property of the architecture.**

The submitted model is the depth-2, width-168 point, trained for 12,000 steps. It records
**1.700525 test BPB — 19.07% below the baseline and 13.87% below this project's earlier frozen
model — with 18.99% fewer MACs per token and 1.76% fewer parameters than the baseline.** Three
further levers were **neutral or worse** and are reported as such: attention head count,
exponential moving average of the weights, and excluding norms and biases from weight decay.

---

## 1. Task, protocol and constraints

Train a language model from scratch and minimise reproducible full-test BPB within a fixed
evaluation budget, under the supplied protocol `7506-mp1-wt2-v2`:

- WikiText-2 with a train-fitted BPE-2048 tokenizer; vocabulary (2048) and context (256) are
  fixed. The training split is 3,613,343 tokens. Independent causal windows of 256 targets;
  BPB is the summed negative log-base-2 next-token probability over the split's raw UTF-8 byte
  count. Validation: 376,599 targets / 1,148,007 bytes. Test: 428,405 / 1,292,013.
- Ranked evaluation is FP32 on CPU. Budgets: scoring time ≤ 5x baseline (11.856 s, ≈59 s),
  peak evaluation RAM ≤ 4 GiB, uncompressed inference assets ≤ 64 MiB.
- **Training duration, architecture and training hardware are unrestricted** inside those
  evaluation limits; only the training and search cost must be disclosed. This is the freedom
  the second half of this report exploits.

`common.py`, `evaluate.py`, `model.py`, `tests/test_contract.py` and `data/` were never
modified; `check_integrity.py` re-verifies them against `PACKAGE_MANIFEST.json` and reports
**13/13 protocol files matching**. `student.py` (the submitted model factory) and the candidate
configs are this project's work. Two mutable files were changed. `train.py` gained two opt-in
switches (`--weight-decay-1d`, `--ema-decay`) whose defaults reproduce the supplied recipe
exactly — verified by a control run (1.943105 against 1.943087 and 1.943094 for two pre-edit
re-runs of the same seed, a spread of 1.8e-05). `README.md` had its three blank budget lines
filled with measured values, its two submission bullets filled in (report link and reproduction
instructions), and the AI-assistance disclosure required by GUIDE §3 added as new §7; its
supplied text is otherwise unchanged apart from two relative links to the guide that were broken
by the package being flattened (`../guide/GUIDE.md` → `../GUIDE.md`).

---

## 2. Baseline diagnosis

The baseline was re-measured locally rather than quoted (4 CPU threads, seed 17, 1,200 steps):
train 667.6 s, **validation BPB 2.071083**, test BPB 2.101260.

| Observation | Value | Implication |
|---|---|---|
| Train–validation loss gap | 4.317 vs 4.376 nats per token, i.e. 0.059 nats (1.35%) | No overfitting at this budget |
| Processed targets / training set | 9,830,400 / 2.72 epochs | Data is re-used, not exhausted |
| Tokens per parameter | 9.03 (Chinchilla-style guidance: ≈20) | Under-trained per parameter |

The baseline is **underfitting** and under-trained relative to the usual parameter/token
guidance. That implies two distinct remedies: spend the fixed parameter budget better (a
structural change), or process more targets (a budget change). The first half of this report
pursues the structural remedy; the second shows the budget change dominates it and then
*rewrites* its conclusion.

---

## 3. Method

### 3.1 Two regimes, one design

Each regime varies exactly one quantity, holding parameter count within ±3% of the baseline
where relevant.

**Regime 1 — equal-parameter frontier at the supplied budget.** Variable: the split of capacity
between width `d` and depth `L`. Fixed: 9,830,400 processed targets, seed, environment,
precision. Fourteen configurations.

**Regime 2 — training-target scaling at fixed architecture.** Architectures (`224/L1`,
`128/L4`, later `168/L2`) retrained at 1,200 / 2,400 / 3,600 / 4,800 / 6,000 / 12,000 steps.
The cosine schedule stretches with `--steps`, so this is "same schedule shape, more targets",
not a truncated run, and every step count is run for both architectures so comparisons stay
paired.

**Regime 2b — the frontier again at the saturating step count.** The 8-point depth ladder is
re-run at 12,000 steps with the same ±3% band and the same requirement of multi-head attention,
so it differs from Regime 1 in the training budget *only*.

Parameter count is predicted analytically rather than measured after the fact. With tied
embedding and output projection, `V=2048`, `C=256`:

```
Total = V·d + C·d + 2d + L·(attn + mlp + 4d)
attn  = 4d² + 4d           (MHA; 2d² + 2d²·kv/h in general)
mlp   = 8d² + 5d           (GELU, hidden = 4d)
```

This predicts 1,088,256 for the baseline, matching the implementation exactly. All 21
configurations used here agree with the formula to the parameter and the check is executable
(`tests/test_equivalence.py`, 14/14 passing). Multiply-accumulates per token follow a
three-term model `L·(12d² + 512d)` for multi-head, charging the attention core `2·T·d`
(`T=256`) per layer.

### 3.2 Pre-registered decision rules

**The gate.** The baseline was run at three seeds first: 2.072792 / 2.075514 / 2.077814, giving
**mean 2.075373, sigma 0.002514**. The decision threshold was set to **0.0050 BPB** (≈2 sigma)
before any candidate was trained. No difference below it is treated as a signal.

**The stopping rule.** Training was extended while a point's last 1,200 steps still bought more
than 0.001 BPB. At 12,000 steps the leaders were buying 0.0001–0.0011, i.e. less than the gate,
so 12,000 is the stopping point and no extrapolation was attempted.

**Selection.** Validation only. The test split was scored only after a freeze, and no test value
ever entered a decision. Disclosed: the test split has now been scored twice in this project,
once per freeze (§4.6).

### 3.3 Search protocol

Full-length screening was used rather than short-training proxies: a 1,200-step GPU run costs
about 25 s, so a shortened proxy would have saved little while risking that short-run rankings
do not transfer — a risk this report later shows to be *real*, since rankings genuinely change
with the training budget. Every run reported here is a full-protocol run.

---

## 4. Results

### 4.1 The equal-parameter frontier at the supplied budget

Seed 17, 9,830,400 processed targets, GPU + bf16 training, FP32 validation scoring. Improvement
is relative to `base-s17` at the same seed and environment.

| Configuration | Depth | Width | Parameters | vs base | MACs/token | Validation BPB | Improvement |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Baseline** | 4 | 128 | 1,088,256 | 0.00% | 1,048,576 | 2.072792 | — |
| C6 (MQA) | 5 | 120 | 1,039,620 | −4.47% | 1,063,200 | 2.105199 | −1.56% |
| C3 (MQA) | 6 | 112 | 1,056,272 | −2.94% | 1,134,336 | 2.111076 | −1.85% |
| C4 (MHA) | 6 | 112 | 1,170,176 | +7.53% | 1,247,232 | 2.112802 | −1.93% |
| C5 (MQA) | 8 | 96 | 1,004,352 | −7.71% | 1,167,360 | 2.166496 | −4.52% |
| W1 | 3 | 144 | 1,084,176 | −0.37% | 967,680 | 2.041589 | −1.51% |
| W2 | 2 | 168 | 1,069,152 | −1.76% | 849,408 | 2.004238 | −3.31% |
| W8 | 1 | 212 | 1,030,956 | −5.27% | 647,872 | 1.958463 | −5.52% |
| W5 | 1 | 220 | 1,090,980 | +0.25% | 693,440 | 1.959555 | −5.46% |
| **W7** | **1** | **224** | **1,121,568** | **+3.06%** | **716,800** | **1.943094** | **−6.26%** |

The table lists the points that carry a claim; the remaining configurations of the frontier
(two more depth-2 widths, an MQA control at the baseline shape, and an MQA twin of `W1`) are in
[`RUN_LOG.csv`](RUN_LOG.csv) and do not change any conclusion. Collapsed by depth the sequence
is strictly monotone — depth 4: 2.0728; depth 3: 2.0416, 2.0326;
depth 2: 2.0042, 1.9984, 1.9835; depth 1: 1.9585, 1.9596, **1.9431** — and every adjacent step
exceeds the gate (the weakest, depth 6 → 5, by 1.2x; the smallest width-leaning gain, W1, by
6.2x). Read by attention type instead (`C3` vs `C4`, identical shape): −0.001726, far below the
gate, so attention type is irrelevant here. Two qualifications: within depth 1 the width
ordering is not monotone (212 and 220 sit 0.0011 apart, inside the noise), and the frontier
confirms a *direction*, not that 224 is an optimum — it is the widest point tested.

### 4.2 The same architectures, varying only the training budget

Only `--steps` changes; seed, environment, data, precision and schedule shape are identical.

| Steps | Processed targets | Epochs | W7 (d224 L1) | BASE (d128 L4) | W7 − BASE | Gate multiples |
|---:|---:|---:|---:|---:|---:|---:|
| 1,200 | 9,830,400 | 2.72 | 1.943094 | 2.072792 | **−0.129698** | −25.9 |
| 2,400 | 19,660,800 | 5.44 | 1.806035 | 1.881351 | −0.075316 | −15.1 |
| 3,600 | 29,491,200 | 8.16 | 1.764189 | 1.795245 | −0.031056 | −6.2 |
| 4,800 | 39,321,600 | 10.88 | 1.742466 | 1.752345 | −0.009879 | −2.0 |
| 6,000 | 49,152,000 | 13.60 | 1.727835 | 1.726357 | +0.001479 | +0.3 |
| 12,000 | 98,304,000 | 27.21 | 1.715212 | **1.684559** | **+0.030654** | **+6.1** |

At 12,000 steps the models have seen 87.6 tokens per parameter, more than four times the
Chinchilla-style guidance, and still improve.

Two findings. First, **training budget dominates architecture**: the same W7 architecture
improves by **0.227882 BPB** between 1,200 and 12,000 steps, whereas the entire structural
search was worth 0.129698 at 1,200 steps. Second, the W7 advantage **decays monotonically and
reverses**. Linear interpolation puts the zero crossing at **5,844 steps = 47,872,055 targets =
13.2 epochs**: below that budget width wins, above it depth wins. The +0.001479 at 6,000 steps
is smaller than the seed sigma (0.002514) and reads as a tie, not a reversal; the +0.030654 at
12,000 steps is 6.1x the gate and is a confirmed reversal.

### 4.3 The equal-parameter ladder at the saturating step count

The supplied deep configurations `c6/c3/c5` use multi-query attention, which would mix a second
variable into the ladder, so four new multi-head configurations were built at widths keeping
the total inside the ±3% band.

| Depth | Width | Parameters | vs base | MACs/token | Validation BPB | Distance from best |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 224 | 1,121,568 | +3.06% | 716,800 | 1.715212 | +0.039020 |
| 2 | 168 | 1,069,152 | −1.76% | 849,408 | **1.679032** | +0.002839 |
| **3** | **144** | **1,084,176** | **−0.37%** | **967,680** | **1.676193** | **best** |
| 4 | 128 | 1,088,256 | 0.00% | 1,048,576 | 1.684559 | +0.008366 |
| 5 | 116 | 1,082,396 | −0.54% | 1,104,320 | 1.690292 | +0.014099 |
| 6 | 108 | 1,097,280 | +0.83% | 1,171,584 | 1.684304 | +0.008111 |
| 7 | 100 | 1,079,700 | −0.79% | 1,198,400 | 1.701458 | +0.025265 |
| 8 | 96 | 1,116,096 | +2.56% | 1,277,952 | 1.697476 | +0.021283 |

**The optimum has moved from depth 1 to depth 3, and the wide-shallow extreme is now the worst
point in the ladder.** The preference is not merely attenuated but inverted at both ends: depth
3 beats the wide-shallow extreme by 0.039020 (7.8x the gate) and the baseline-architecture
point by 0.008366 (1.7x).

### 4.4 Replication, and a selection that got harder

The two leading ladder points look 0.002839 apart — below the gate, so indistinguishable — but
single-seed spacing is exactly what this report refuses to lean on. Both were re-run at seeds
42 and 123.

| Configuration | seed 17 | seed 42 | seed 123 | Mean | Sigma | Parameters | MACs/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| L3 (d144, L=3) | 1.676193 | 1.675741 | 1.689320 | 1.680418 | 0.007712 | 1,084,176 | 967,680 |
| **L2 (d168, L=2)** | 1.679032 | 1.670718 | 1.669866 | **1.673206** | 0.005064 | **1,069,152** | **849,408** |

Replication **moved the nominal best from L3 to L2**, which is the point: the single-seed
ranking did not survive three seeds. The difference of means is 0.007212 with a standard error
of 0.005327 (**t = 1.35**) — not significant. The 0.0050 gate is calibrated for single-seed
differences and must not be applied to a difference of means without that standard error.
**The two points are measured-equal in quality**, so the choice was made on cost: L2 uses
15,024 fewer parameters and 118,272 fewer MACs per token. L2 was frozen on that basis. The
frozen checkpoint is the **pre-registered seed 17** rather than the best of the three
replicates (seed 123 at 1.669866), so that no seed shopping enters the recorded score; the
three-seed mean is reported as the configuration's expected performance.

### 4.5 The frozen predictor and the recorded score

The method was frozen (`w2.json`, seed 17, 12,000 steps) and the complete test split was then
scored once with `evaluate.py --device cpu --precision fp32 --split test`.

| Quantity | Value | Limit / reference | Status |
|---|---:|---|---|
| **Test BPB** | **1.700525** | — | reported score |
| vs local baseline | −0.400735 | 2.101260 | **−19.07%** |
| vs this project's earlier frozen model | −0.273751 | 1.974276 (W7, 1,200 steps) | **−13.87%** |
| Targets scored | 428,405 (complete) | complete split | PASS |
| CPU scoring time | 10.833 s | ≈59 s budget | PASS (baseline 11.856 s, 0.91x — faster) |
| Peak evaluation RAM | 1.8004 GiB | 4 GiB | PASS |
| Inference asset (checkpoint) | 4.09 MiB | 64 MiB | PASS |
| Re-run of the same command | 1.700525 | diff < 1e-3 | PASS (difference 0.000E+000) |

The frozen checkpoint is `runs/w2-s17-r12000/checkpoint.pt` (sha256 `35531887…`). It stores the
config and the implementation module name, so the predictor is rebuilt from the checkpoint and
`student.py` alone, with no training step and no network access. Peak evaluation RAM was
sampled every 150 ms during scoring; the same procedure applied to the earlier checkpoint
reproduced its recorded 1.801 GiB exactly, so the method is consistent across models.

**Disclosure on repeated test use.** The test split has been scored twice in this project, once
per freeze, each reproduced bit-for-bit. No test value was used to choose anything: the
configuration was selected entirely on validation, and the earlier model's test result did not
inform the later design. This is disclosed because a reader cannot verify the ordering of
decisions from the artefacts alone.

The pre-registered target was ≤2.038 test BPB (≥3% better than the local baseline), set before
any search and **never changed** — not tightened when cleared, not relaxed afterwards. The
achieved 1.700525 clears it by 0.337 BPB.

---

## 5. Cost disclosure

All 46 training runs and 6 evaluation-only measurements are itemised in
[`RUN_LOG.csv`](RUN_LOG.csv) with per-run seeds, ancestry, parameter counts, checkpoint hashes
and wall-clock times.

| Item | Count | Wall clock | Processed targets |
|---|---:|---:|---:|
| GPU training runs | 45 | 5,081.4 s | 1,720,320,000 |
| CPU baseline run (supplied recipe) | 1 | 667.6 s | 9,830,400 |
| **Total training** | **46** | **5,749.0 s (95.8 min)** | **1,730,150,400** |
| Evaluation measurements (CPU, FP32) | 6 | 58.8 s | — |
| **Total** | **52** | **5,807.8 s (96.8 min)** | **1,730,150,400** |

The search consumed **17.6x** the training targets of the final selection (98,304,000). Of the
46 training runs, 19 predate this session (the 1,200-step frontier and the noise floor) and 27
belong to it (the scaling series, the 12,000-step ladder, the replicates and the three neutral
levers); the 6 evaluation rows split 4 and 2. No run reuses another checkpoint; every
checkpoint derives from random initialisation and the supplied training text only. Peak GPU
allocation was 0.77 GiB (`mhal7-s17-r12000`), so no candidate was rejected for cost, and
training cost is not budget-capped — only disclosed.

One measurement caveat, since the raw seconds are in the log: GPU seconds per token drift upward
within a sustained session (1,250–1,400 ns/token for the shallow models early, 4,400–6,170 for
the deep ones late), reflecting both the shape of the work — deep narrow models use the GPU far
less efficiently than few wide layers — and thermal throttling on a laptop GPU. Training wall
clock is therefore an upper bound and justifies no conclusion; every performance claim rests on
BPB, which is scored on CPU.

---

## 6. Critical analysis

### 6.1 Three revisions of the central claim, all kept in the record

**Revision 1 → falsified.** The plan was to share key/value projections (multi-query attention)
to save `2d²(1 − k/h)` parameters per block and spend the saving on more blocks at equal total
parameters. The clean ablation is `C3` (d112, depth 6, MQA) versus `C4` (d112, depth 6, MHA):
**−0.001726 BPB, far below the gate — MQA is neutral**, and all five depth-leaning candidates
were *worse* than the baseline, monotonically (2.0728 → 2.1052 → 2.1111 → 2.1665). The
hypothesis predicted the opposite direction.

**Revision 2 → correct but over-generalised.** Rather than invent a third mechanism, the
gradient of the failure was followed: the depth-leaning results were monotone, so the opposite
direction was searched, producing §4.1. The surviving statement was "at constant parameters
*and constant processed targets*, move capacity from depth to width". The italicised clause
matters: §4.2–4.3 show the effect exists only at that budget.

**Revision 3 → the claim is now a function, not a law.** §4.1 and §4.3 are the same experiment
at two budgets and they disagree at both ends of the ladder, so the general architectural
statement is withdrawn. What survives is narrower and quantitative: *which* width/depth split
is optimal depends on how many targets the model will process, and the optimal depth increases
with that budget.

### 6.2 What the crossover settles, and what it still does not

An earlier version of this work listed three undistinguished explanations for why width helped
and named one distinguishing experiment as designed-but-unrun: *give the deep models
substantially more steps*. That experiment has now been run, and it discriminates. Two
explanations predicted exactly this observation — that depth is harder to optimise inside a
fixed step budget, and that the conclusion is a function of data scale — so **both are
supported**, and the 1,200-step width advantage is best read as a training-budget effect rather
than an architectural one. The third explanation, that the tied output projection caps the
output distribution at rank `d`, **remains untested**: no untied-output run exists anywhere in
this project, so untied narrow models cannot be compared against these results, and that is the
one experiment that would still change the interpretation.

### 6.3 Three levers that did nothing (all below the gate)

Reported because a search that publishes only its wins is not a search.

- **Attention head count.** Changing heads at the W7 shape is *free*: multi-head attention
  costs `4d² + 4d` parameters per block at any head count, so parameters and MACs are
  unchanged. heads = 2 / 7 / 14 scored 1.948981 / 1.954492 / 1.965600 against 1.943094 for
  heads = 4 (head_dim 56). Every move is worse: this is not an unexplored free dimension but
  one that has been explored and does not help.
- **Exponential moving average of the weights** (`--ema-decay 0.995`): 1.803701 for the
  averaged weights versus 1.805985 for the raw weights of the same run — 0.002284, below the
  gate. The raw weights matched the plain run (1.806035), so the bookkeeping does not perturb
  training.
- **No weight decay on norms and biases** (`--weight-decay-1d 0`): 1.802684 versus 1.806035 —
  0.003351, below the gate.

### 6.4 The gate was calibrated in a different regime from the one it now governs

The 0.0050 gate comes from a seed sigma of 0.002514 measured on the **baseline architecture at
1,200 steps**. At 12,000 steps the observed seed spread is 0.005064 (L2) and 0.007712 (L3) —
two to three times wider. Applying a gate calibrated in one regime to another is a real
weakness and it cuts both ways: it made the L2/L3 comparison look decisive on one seed and made
the 6,000-step tie look like a reversal. Late-training comparisons therefore carry wider error
bars than the headline experiments, which is why §4.4 reports the standard error of the
difference instead of comparing point estimates to a threshold. A properly powered ladder at
12,000 steps would need more seeds than were run here.

### 6.5 Selection pressure accumulated, and one criterion that no longer holds

About 46 runs were compared on a single validation split and the final configuration was chosen
by its three-seed mean on that same split. This is the standard risk of iterative selection and
it is not eliminated here. Two mitigations, neither sufficient: the single post-freeze test
score agreed with validation (−19.07% on test against −19.00% on validation at the same seed,
and −19.38% against the baseline's three-seed validation mean), and the replication step showed
the *ranking* changing under resampling — exactly the failure mode of over-selection. The
mitigation **not** taken is a held-out part of validation.

One earlier success criterion must also be flagged. `SC-003` asserts that reverting the key
mechanism — returning to a baseline-like deep, narrow allocation — worsens validation BPB by at
least 0.01. That held at 9,830,400 targets, where L3 scores 2.041589 against the baseline's
2.072792. **At the 12,000-step budget actually submitted, the deeper point is the better one**
(L4 at 1.684559 against L3 at 1.676193 is the losing direction; the reverted allocation wins).
The criterion's own text limits it to the 9.8M-target regime, and it was **not** widened or
redefined to accommodate the new operating point: it is recorded as met in its stated regime
and as not holding outside it.

### 6.6 What would falsify the revised claim

The revised claim is *"the optimal width/depth allocation at constant parameters is a function
of the number of processed training targets, and optimal depth increases with that budget."*
It would be overturned by any of four observations: a budget sweep in which optimal depth does
not move monotonically with targets (making §4.1 and §4.3 a two-point coincidence rather than a
connected trend); the 12,000-step ladder repeated with more seeds showing depth 1 is not the
worst point (making the inversion seed noise — the ladder's 0.039020 spacing, at 6.1x the gate,
is the load-bearing evidence); the 5,844-step crossing failing to reproduce for other
architectures at the same parameter count (making it specific to these two shapes rather than a
budget effect); or any single budget where both ends of the ladder beat its middle (falsifying
the interior-optimum reading). Degrading on a different corpus, tokenizer or context length
would **not** falsify it, because the claim is not generalised beyond the protocol in §1.

### 6.7 Limitations

- **Two ladder points carry three seeds (L2, L3); the other six carry one.** Their spacing
  relative to the gate is comfortable (0.008–0.039), but the individual values are single-seed,
  and §4.4 shows single-seed rankings in this regime are unreliable.
- **The crossover step count (5,844) is interpolated between two sampled budgets**, not
  measured directly, and is specific to these two architectures at ≈1.1M parameters.
- **One corpus, tokenizer, context length, optimizer and schedule shape.** The schedule
  stretches with the step count, so "more targets" is entangled with "a longer annealing
  schedule"; separating them was not attempted.
- **Width was not optimised finely.** Ladder widths are the largest multiples of 4 fitting the
  ±3% band, not a search over width at each depth, and the `d = 224, L = 1` endpoint was not
  probed beyond 224 — it remains an endpoint rather than a demonstrated optimum.
- **No untied-output run exists**, so §6.2's third explanation remains untested.

---

## 7. Reproduction

The frozen predictor is rebuilt from the checkpoint without retraining:

```bash
cd code
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v                              # 14 tests, incl. the 5 supplied contract tests
python check_integrity.py                                            # 13/13 protocol files match PACKAGE_MANIFEST.json
python evaluate.py --checkpoint runs/w2-s17-r12000/checkpoint.pt \
                   --device cpu --precision fp32 --split test
```

The last command reproduces **1.700525** BPB from `runs/w2-s17-r12000/test_cpu_fp32.json`.
Expected environment: Python 3.12, PyTorch 2.7.1+cpu, `tokenizers==0.21.4` (the version must
match or tokenisation may differ), 4 CPU threads. Retraining needs a GPU environment (PyTorch
2.8.0+cu128, Python 3.10); the recipe for every run is in
`specs/001-param-realloc-gqa/quickstart.md`, and the exact `--steps` and `--config` per run are
recoverable from each `runs/*/metrics.json`. The submitted model is `student.py`, self-contained
(PyTorch only; it does not import `model.py`), and `model.py` with `configs/baseline.json` is
retained unchanged for the baseline comparison.

Naming note: the specification directory `001-param-realloc-gqa` is historical — "gqa" referred
to the first (falsified) mechanism. The submitted model uses standard multi-head attention.

Two further disclosures. The `code_commit` column of `RUN_LOG.csv` holds the SHA-256 prefixes
of the modules each run used (`impl:9a08cff5` = `student.py`, `impl:6cce142a` = `model.py`,
`train:93aed0d7` = the edited `train.py`), because the run directory is not a git repository;
the hashes are verifiable from `runs/*/metrics.json`. And `guide/GUIDE.md` on disk does not
match its `PACKAGE_MANIFEST.json` hash; this pre-dates this work, the file was never written to
by this project, and it is documentation rather than a scored protocol file, so
`check_integrity.py` reports it without gating on it.

---

## 8. AI assistance disclosure

An AI coding assistant (Trae, GLM/Claude-class model) was used substantively.

**Assisted:** implementing `student.py`'s configurable attention (including the
numerical-equivalence test against `model.py`), writing the candidate configs,
`tests/test_equivalence.py` and `check_integrity.py`, the `train.py` switches for the 1-D-decay
and EMA levers, running the training and evaluation commands, aggregating results, and drafting
this report and the specification documents.

**Authored and decided by the student:** diagnosing before designing, the equal-parameter
frontier design, the noise-floor-first protocol, holding the target fixed and refusing to widen
the ±3% parameter band after W7 came in at +3.06%, and the decisions to disclose every falsified
mechanism, the neutral levers, the repeated test scoring and the criterion that no longer holds
at the submitted operating point, rather than present only the winning direction. Every number
in this report was reviewed against the underlying `runs/` artefacts before submission.

**Not assisted:** no pretrained weights, external training text or external model was used; all
weights, statistics and selection decisions derive only from the supplied training text and
validation split. No answer caching, retrieval database or lookup index is used. The evaluator,
data and tokenizer were never modified. Nothing beyond the supplied course package was reused;
WikiText-2 is used under the licences recorded in `code/README.md` §6.

---

## References

1. S. Merity, C. Xiong, J. Bradbury, R. Socher. *Pointer Sentinel Mixture Models.* 2017. (WikiText-2)
2. J. Hoffmann et al. *Training Compute-Optimal Large Language Models.* 2022. (tokens-per-parameter guidance)
3. N. Shazeer. *Fast Transformer Decoding: One Write-Head is All You Need.* 2019. (multi-query attention)
4. J. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* 2023.
5. O. Press, L. Wolf. *Using the Output Embedding to Improve Language Models.* 2017. (weight tying, rank limit)
6. J. Kaplan et al. *Scaling Laws for Neural Language Models.* 2020.
7. B. T. Polyak, A. B. Juditsky. *Acceleration of Stochastic Approximation by Averaging.* 1992. (weight averaging)
8. I. Loshchilov, F. Hutter. *Decoupled Weight Decay Regularization.* 2019. (decay on norms and biases)