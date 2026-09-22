# MP1 code — installation and usage

Read [the project guide](../GUIDE.md) for the assignment, assessment, deadlines and peer review. This README contains the running instructions and technical rules. The package has only these two documents.

All commands below run from **code/**. Data and the tokenizer are included. No API key, pretrained weights or additional dataset download is needed; after installing dependencies, training and evaluation work offline.

## 1. Install

Use **Python 3.12**. From the extracted package directory:

```bash
cd code
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.

Install PyTorch for **one** device:

```bash
# Linux/Windows CPU: recommended; no GPU needed
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
```

For an NVIDIA GPU with a compatible driver, use this command **instead**:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
```

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. After installing PyTorch, install the remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C, measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**Your model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the corresponding device/split name) and per-window losses. Submit the **bpb** value from the complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add `--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserve for comparisons. |
| `student.py`, `train.py` | Your model factory and training recipe; add supporting code as needed. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; keep unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; keep unchanged. |
| `tests/test_contract.py` | Checks your model's causality, normalization, independence and gradients. |
| `RUN_LOG_TEMPLATE.csv` | Optional experiment-log template. |
| `PACKAGE_MANIFEST.json` | Release hashes; paths are relative to the package root containing code/ and guide/. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state between independent windows, examples and scoring passes. Compact training-derived assets may be reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and ensembles may change within the guide's constraints. Log all seeds, processed training targets, checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048, independent windows of 256 targets, including the final short window. Every target except the first token of each split is scored once. Input windows share a boundary token but carry no state. BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8 byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint/mixture selection. Weights, statistics and retrieval entries must derive only from training text. The public test text enables reproduction; it must not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for timing or reproduction. Token perplexity is not directly comparable with published word-level perplexity.

Measure all three limits for the same frozen predictor:

- **CPU time ≤5× baseline:** 10.83 s for the frozen predictor, against a locally re-measured
  baseline of 11.86 s (limit ≈ 59 s). PASS, at 0.91× the baseline.
- **Peak RAM ≤4 GiB:** 1.80 GiB, the process peak working set sampled every 150 ms while
  scoring the complete test split. PASS.
- **Inference assets ≤64 MiB uncompressed:** 4.09 MiB checkpoint. PASS. 

## 5. Prepare your submission and reproduce a peer

The [guide](../GUIDE.md) specifies the deadline and website workflow. Include the following in your immutable code repository:

- **Report, at most 10 pages including figures, tables and references** — [`REPORT.md`](REPORT.md)
- **Reproduction instructions** — §7 below and [`specs/001-param-realloc-gqa/quickstart.md`](../specs/001-param-realloc-gqa/quickstart.md)

Your final website submission must link to this code and the matching complete checkpoint bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare reproduced BPB with the reported score. Submit **Peer Review Report** with the reproduced score; optionally include the command, environment, difference and evidence/log link.  The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn bonus credit under the announced marking policy.

## 6. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in [Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext) identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the [GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); retain these notices when redistributing the data.

The supplied `wikitext-2-raw-v1` splits preserve revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8; the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These dataset notices do not assign a new license to the surrounding classroom code.

## 7. Submission record — student 3036747264

**Recorded score: 1.700525 test BPB** (protocol `7506-mp1-wt2-v2`, CPU, FP32), 19.07% below the
locally re-measured baseline (2.101260) and 13.87% below this project's earlier frozen model
(1.974276).

**Frozen predictor.** `configs/w2.json` — width 168, depth 2, standard multi-head attention,
1,069,152 parameters, 849,408 multiply-accumulates per token (18.99% fewer than the baseline,
at 1.76% fewer parameters). Seed 17, 12,000 steps = 98,304,000 processed training targets.
Checkpoint: `runs/w2-s17-r12000/checkpoint.pt`.

### Reproduce the recorded score

No training and no network access are required; only the checkpoint and `student.py` are needed
to rebuild the predictor.

```bash
cd code
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # 14 tests: the 5 supplied contract tests plus 9 project tests
python check_integrity.py                    # expect: 13 protocol files checked, 0 problems
python evaluate.py --checkpoint runs/w2-s17-r12000/checkpoint.pt --device cpu --precision fp32 --split test
```

The last command writes `runs/w2-s17-r12000/test_cpu_fp32.json`; its `bpb` field is **1.700525**,
reproduced bit-for-bit on re-run (difference `0.000E+000`). Reference environment: Python 3.12,
PyTorch 2.7.1+cpu, `tokenizers==0.21.4` (the version must match or tokenisation may differ),
4 CPU threads. Scoring the complete test split takes about 10.8 s and peaks at 1.80 GiB of
working set.

### Reproduce the training (optional)

Needs a GPU environment: PyTorch 2.8.0+cu128, Python 3.10.

```bash
python train.py --implementation student --config configs/w2.json --device cuda --precision bf16 \
                --seed 17 --steps 12000 --eval-every 1200 --run-dir runs/w2-s17-r12000
```

`--steps` stretches the cosine schedule rather than truncating a run, so every step count in
`RUN_LOG.csv` is a full-protocol run at its own budget. Retraining is not required to reproduce
the recorded score and is not expected to land on the same weights.

### Cost

46 training runs and 6 evaluation measurements are itemised in [`RUN_LOG.csv`](RUN_LOG.csv) with
per-run seeds, checkpoint ancestry, parameter counts, checkpoint hashes and wall-clock times:
**5,749 s of training over 1,730,150,400 processed targets — 17.6× the final run — plus 59 s of
evaluation.** Training duration is not budget-capped by the assignment, only disclosed.

### What this repository contains, and what it deliberately omits

Included: the code required to reproduce the recorded score (evaluator, model, configs, supplied
data and tokenizer, integrity manifest, frozen checkpoint); the evidence for every number in the
report (`runs/*/metrics.json`, the evaluation JSONs, `RUN_LOG.csv`); the report; and the
specification directory the report links to.

Omitted as search byproducts, all regenerable with the commands above: **45 of the 46 training
checkpoints (199 MB)** — only the frozen checkpoint is kept — and the per-window loss arrays
(`*.window-nll.npy`). Also omitted: agent and OS byproducts (`__pycache__/`, `*.pyc`, `.DS_Store`),
and the `history/` prompt logs and `.specify/` scaffolding, which document how the work was
arranged rather than how to reproduce it.

### AI assistance disclosure

GUIDE §3 requires substantive AI help to be acknowledged here. An AI coding assistant (Trae,
GLM/Claude-class model) was used substantively for: implementing `student.py`'s configurable
attention and its numerical-equivalence test against `model.py`; writing the candidate configs;
writing `tests/test_equivalence.py` and `check_integrity.py`; adding the two opt-in `train.py`
switches for the 1-D-decay and EMA experiments; running the training and evaluation commands and
aggregating results; and drafting `REPORT.md` with the specification documents.

Authored and decided by the student: the diagnosis-first approach, the equal-parameter frontier
design, the noise-floor-first protocol, holding the pre-registered target fixed and refusing to
widen the ±3% parameter band after the selected point came in at +3.06%, and the decisions to
disclose every falsified mechanism, the neutral levers, the repeated test scoring and the
success criterion that no longer holds at the submitted operating point — rather than present
only the winning direction. Every number in `REPORT.md` was reviewed against the underlying
`runs/` artefacts.

No pretrained weights, external training text or external model was used. All weights,
statistics and selection decisions derive only from the supplied training text and validation
split. `common.py`, `evaluate.py`, `model.py`, `tests/test_contract.py` and `data/` were never
modified, and `check_integrity.py` re-verifies their hashes. The three files this project did
change are `student.py` (the submitted model factory), `train.py` (the two opt-in switches, whose
defaults reproduce the supplied recipe exactly) and this README.
