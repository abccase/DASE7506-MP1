# DASE7506 MP1

Student ID **3036747264**. A language model trained from scratch under protocol
`7506-mp1-wt2-v2`, and a controlled study of *where* a fixed parameter budget is spent and *how
long* the model is trained.

**Recorded score: 1.700525 test BPB** — 19.07% below the supplied baseline (2.101260) and 13.87%
below this project's earlier frozen model (1.974276), while using **18.99% fewer
multiply-accumulates per token** and **1.76% fewer parameters** than the baseline.

| Budget | Measured | Limit | |
|---|---:|---:|---|
| CPU scoring time, full test split | 10.83 s | ≈59 s (5× baseline) | PASS |
| Peak evaluation RAM | 1.80 GiB | 4 GiB | PASS |
| Inference assets, uncompressed | 4.09 MiB | 64 MiB | PASS |
| Re-run of the identical command | 1.700525 | diff < 1e-3 | PASS (diff `0.000E+000`) |

This is the repository's single entry document: it doubles as the landing page and as the
protocol, installation and reproduction manual. Read [GUIDE.md](GUIDE.md) for the assignment,
assessment, deadlines and peer review.

## What the study found

Two levers were tested one at a time, each holding everything else fixed.

**At the supplied training budget, width beats depth.** On an equal-parameter frontier, moving
capacity from depth into width improves validation BPB monotonically; the best point (width 224,
depth 1) beats the baseline by 0.129698 BPB — 6.26% — at 31.6% fewer multiply-accumulates per
token. That looked like a general architectural law.

**A longer training budget reverses it.** Holding the architecture fixed and varying only the
number of processed targets shows that advantage decaying monotonically to zero at about **5,844
steps (47.9M targets, 13.2 epochs)**, and at 12,000 steps the deeper baseline is 0.030654 BPB
ahead. Re-running the whole equal-parameter frontier at 12,000 steps moves the optimum from depth
1 (d=224) to **depth 3 (d=144)**, with the wide-shallow extreme now the *worst* point in the
ladder.

**The optimal width/depth allocation is therefore a function of the training budget, not a
property of the architecture** — and training budget is the larger lever by far: the same
configuration improves by 0.227881 BPB from 1,200 to 12,000 steps, against 0.129698 for the
entire structural search.

The submitted model is the depth-2, width-168 point of the second frontier. Three further levers
were tested and did nothing measurable (attention head count, weight averaging, excluding norms
and biases from weight decay); they are reported with the negative results, the two falsified
mechanisms and the limitations in [`code/REPORT.md`](code/REPORT.md).

## Repository map

| Path | What it is |
|---|---|
| [`code/REPORT.md`](code/REPORT.md) | The report (about 7 pages). Method, results, critical analysis. |
| [`code/student.py`](code/student.py) | The submitted model factory. Self-contained; PyTorch only, does not import `model.py`. |
| [`code/RUN_LOG.csv`](code/RUN_LOG.csv) | All 46 training runs and 6 evaluation measurements: seeds, ancestry, parameter counts, processed targets, checkpoint hashes, wall-clock times. |
| [`code/runs/`](code/runs) | Per-run `metrics.json` and evaluation JSON — the evidence for every number in the report — plus the one frozen checkpoint. |
| [`code/`](code) | The supplied harness: `train.py`, `evaluate.py`, `common.py`, `model.py`, `configs/`, `data/`, `tests/`, `PACKAGE_MANIFEST.json`. |
| [`specs/001-param-realloc-gqa/`](specs/001-param-realloc-gqa) | Specification, plan, research log (R1–R12) and the per-run training recipes. |
| `.gitattributes` | Pins `* -text` so a clone is byte-identical to what was scored; see below. |

## 1. Install

Use **Python 3.12**. All commands in this document are run from the repository root and start by
entering `code/`. Data and the tokenizer are included; no API key, pretrained weights or extra
dataset download is needed, and after installing dependencies training and evaluation work
offline.

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

For macOS, install `torch==2.7.1` from the default PyPI index and run on CPU. Then install the
remaining dependencies and check the model:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

Linux CPU commands were verified with Python 3.12 and PyTorch 2.7.1+cpu. Windows/macOS timings
have not been measured.

## 2. Train and evaluate

**Quick installation check** — 10 training steps, then full-test evaluation:

```bash
python train.py --implementation model --steps 10 --run-dir runs/smoke
python evaluate.py --checkpoint runs/smoke/checkpoint.pt --split test
```

This checks that the pipeline works; its score is **not** the full baseline. Each training run
needs a new output directory.

**Full baseline** — 1,200 training updates, then evaluation:

```bash
python train.py --implementation model --device cpu --threads 4 --seed 17 --run-dir runs/baseline
python evaluate.py --checkpoint runs/baseline/checkpoint.pt --device cpu --precision fp32 --split test
```

The baseline has four GPT blocks, width 128, four attention heads and **1,088,256 parameters**, and
achieves approximately **2.10 test BPB**. On the reference four-thread Xeon Platinum 8457C,
measured training took about **311 seconds** and scoring **5.92 seconds**, excluding installation
and loading. These are reference measurements, not laptop guarantees or a fixed time allowance.

**A submission model** — edit `student.py` and supporting files, then:

```bash
python train.py --implementation student --seed 17 --eval-every 300 --run-dir runs/my-model
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split validation
# Freeze the final method before testing:
python evaluate.py --checkpoint runs/my-model/checkpoint.pt --split test
```

Training writes `checkpoint.pt` and `metrics.json`. Evaluation writes `test_cpu_fp32.json` (or the
corresponding device/split name) and per-window losses. Submit the **bpb** value from the
complete-test JSON, not token perplexity or validation BPB. Default evaluation is FP32. Add
`--device cuda` for GPU runs; training can use BF16, but ranked evaluation must use FP32 and
remain reproducible on CPU. The supplied CUDA runner caps PyTorch allocation at 20 GB; driver
overhead is additional.

## 3. Files and model interface

| Files | Use |
|---|---|
| `model.py`, `configs/baseline.json` | Runnable baseline; preserved here for comparisons. |
| `student.py`, `train.py` | The submitted model factory and training recipe. |
| `common.py`, `evaluate.py` | Fixed data checks, windows and scorer; unchanged. |
| `data/` | Supplied splits, tokenizer and dataset hashes; unchanged. |
| `tests/test_contract.py` | Checks the model's causality, normalization, independence and gradients. |
| `tests/test_equivalence.py` | Numeric equivalence with `model.py` plus per-config parameter counts (project file). |
| `check_integrity.py` | Re-verifies every released file against `PACKAGE_MANIFEST.json`. |
| `RUN_LOG_TEMPLATE.csv` | The supplied experiment-log template, left unmodified. |

- `build_model(config)` returns a PyTorch model with `context=256`.
- The supplied trainer calls `forward(ids)` for unnormalized logits; the scorer calls
  `predict_log_probs(ids)` for finite, normalized natural-log probabilities. Both outputs have
  shape `[batch, time, 2048]`.
- A prediction at position t may use only the observed prefix through t. Reset temporary state
  between independent windows, examples and scoring passes. Compact training-derived assets may be
  reused across windows; evaluation-prefix state may not.
- Checkpoints record the implementation module and configuration. Include that module and every
  required asset so the evaluator can reconstruct the submitted predictor. No optimizer state is
  required for direct evaluation.
- Training length, architecture, optimizer, regularization, self-trained weight averaging and
  ensembles may change within the guide's constraints. Log all seeds, processed training targets,
  checkpoint ancestry and search costs; reusing a checkpoint does not erase its training cost. No
  particular seed or score improvement is mandated.

## 4. Benchmark and resource measurements

**Fixed score.** Protocol `7506-mp1-wt2-v2`: WikiText-2 raw text, train-fitted BPE-2048,
independent windows of 256 targets, including the final short window. Every target except the
first token of each split is scored once. Input windows share a boundary token but carry no state.
BPB is summed negative log-base-2 next-token probability divided by the split's entire raw UTF-8
byte length, including the first token's bytes.

| Split | Scored targets | UTF-8 bytes |
|---|---:|---:|
| Validation | 376,599 | 1,148,007 |
| Test | 428,405 | 1,292,013 |

Use validation for all development and checkpoint selection. Weights, statistics and retrieval
entries must derive only from training text. The public test text enables reproduction; it must
not be used to tune the method. Once frozen, the same predictor may be evaluated repeatedly for
timing or reproduction. Token perplexity is not directly comparable with published word-level
perplexity.

All three limits were measured on the same frozen predictor:

- **CPU time ≤5× baseline:** 10.833 s for the frozen predictor, against a locally re-measured
  baseline of 11.856 s (limit ≈ 59 s). PASS, at 0.91× the baseline.
- **Peak RAM ≤4 GiB:** 1.8004 GiB, the process peak working set sampled every 150 ms while scoring
  the complete test split. PASS. The same procedure applied to the earlier frozen model reproduced
  its recorded 1.801 GiB, so the measurement method is consistent across models.
- **Inference assets ≤64 MiB uncompressed:** 4.09 MiB checkpoint. PASS.

## 5. Reproduce the recorded score

No training and no network access are required; only the checkpoint and `student.py` are needed to
rebuild the predictor.

```bash
cd code
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v      # 14 tests: the 5 supplied contract tests plus 9 project tests
python check_integrity.py                    # expect: 13 protocol files checked, 0 problems
python evaluate.py --checkpoint runs/w2-s17-r12000/checkpoint.pt \
                   --device cpu --precision fp32 --split test
```

The last command writes `runs/w2-s17-r12000/test_cpu_fp32.json`; its `bpb` field is **1.700525**,
reproduced bit-for-bit on re-run (difference `0.000E+000`). Reference environment: Python 3.12,
PyTorch 2.7.1+cpu, `tokenizers==0.21.4` (the version must match or tokenisation may differ), 4 CPU
threads.

Verified the way a reviewer would: a fresh `git clone` containing only the tracked files passes
14/14 tests and `check_integrity.py` at 13/13, and reproduces 1.700525 bit-for-bit.

**Why `.gitattributes` matters.** The integrity check hashes bytes and BPB divides by each split's
exact UTF-8 byte count. With `core.autocrlf=true` — the default on many Windows installs — a clone
would rewrite the text files to CRLF, inflating `wikitext_test.txt` from 1,292,013 to 1,299,261
bytes and breaking both the hash check and the score. Pinning `* -text` makes the rule travel with
the repository, so it holds regardless of the cloning machine's local git settings.

**Reproducing the training (optional).** Needs a GPU environment: PyTorch 2.8.0+cu128,
Python 3.10.

```bash
python train.py --implementation student --config configs/w2.json --device cuda --precision bf16 \
                --seed 17 --steps 12000 --eval-every 1200 --run-dir runs/w2-s17-r12000
```

`--steps` stretches the cosine schedule rather than truncating a run, so every step count in
`RUN_LOG.csv` is a full-protocol run at its own budget. Retraining is not required to reproduce the
recorded score and is not expected to land on the same weights. The recipe for every run is in
[`specs/001-param-realloc-gqa/quickstart.md`](specs/001-param-realloc-gqa/quickstart.md), and the
exact `--steps` and `--config` per run are recoverable from each `runs/*/metrics.json`.

## 6. Checkpoint bundle

The matching bundle for the website submission is built from this repository and contains only the
inference assets:

| Bundle file | Copied from |
|---|---|
| `checkpoint.pt` | `code/runs/w2-s17-r12000/checkpoint.pt` |
| `student.py` | `code/student.py` |
| `config.json` | `code/configs/w2.json` |
| `requirements.txt` | `code/requirements.txt` |
| `MANIFEST.sha256`, `README.md` | Written at build time |

**Download:** [Releases — v1.0](https://github.com/abccase/DASE7506-MP1/releases/tag/v1.0), asset
`DASE7506-MP1-checkpoint-bundle.zip`.

| Hash | Value |
|---|---|
| `DASE7506-MP1-checkpoint-bundle.zip` | `642b361a7bda2cba1fe88e84b9210f207ed1a83bf7e6d71e060cc890acdd59a8` |
| `checkpoint.pt` (inside the bundle and in `runs/w2-s17-r12000/`) | `355318877465ecc4e74bc39bb197baea64fa85044751ceb25eb22e90c855fc55` |

The checkpoint hash is the value recorded in `runs/w2-s17-r12000/metrics.json`. Every payload file
is byte-identical to its repository counterpart, the bundle verifies against its own
`MANIFEST.sha256`, and the archive is byte-reproducible: rebuilding it from the same inputs yields
the same sha256.

Verified end to end against the published copies: cloning the repository from GitHub yields 13/13
byte-identical protocol files, 14/14 tests and `check_integrity.py` at 13/13, and reproduces
1.700525; downloading the release asset back from GitHub gives the zip hash above, a manifest that
verifies 4/4, and a checkpoint that scores 1.700525 exactly.

## 7. Cost

46 training runs and 6 evaluation measurements are itemised in
[`code/RUN_LOG.csv`](code/RUN_LOG.csv) with per-run seeds, checkpoint ancestry, parameter counts,
checkpoint hashes and wall-clock times: **5,749 s of training over 1,730,150,400 processed
targets — 17.6× the final run — plus 59 s of evaluation.** Training duration is not budget-capped
by the assignment, only disclosed.

## 8. Submission and peer review

Include the following in the immutable code repository:

- **Report, at most 10 pages including figures, tables and references** — [`code/REPORT.md`](code/REPORT.md)
- **Reproduction instructions** — §5 above and [`specs/001-param-realloc-gqa/quickstart.md`](specs/001-param-realloc-gqa/quickstart.md)

The final website submission must link to this code and to the matching complete checkpoint
bundle. The website generates the Issue JSON automatically. Keep all inference assets downloadable
for verification.

To check a peer, obtain their exact code version and checkpoint, follow their installation
instructions, and run their frozen model with the supplied evaluator:

```bash
python evaluate.py --checkpoint /path/to/peer-checkpoint.pt --device cpu --precision fp32 --split test --output peer-test.json
```

Compare the reproduced BPB with the reported score. Submit a **Peer Review Report** with the
reproduced score; optionally include the command, environment, difference and evidence/log link.
The instructor adjudicates discrepancies. Confirmed discrepancies during the seven-day review earn
bonus credit under the announced marking policy.

## 9. What is deliberately not here

Included: the code required to reproduce the recorded score (evaluator, model, configs, supplied
data and tokenizer, integrity manifest, frozen checkpoint); the evidence for every number in the
report (`runs/*/metrics.json`, the evaluation JSONs, `RUN_LOG.csv`); the report; and the
specification directory the report links to.

Omitted as search byproducts, all regenerable with the commands above: **45 of the 46 training
checkpoints (199 MB)** — only the frozen checkpoint is kept — and the per-window loss arrays
(`*.window-nll.npy`). Also omitted: agent and OS byproducts (`__pycache__/`, `*.pyc`, `.DS_Store`,
`.trae/`), and the `history/` prompt logs and `.specify/` scaffolding, which document how the work
was arranged rather than how to reproduce it.

## 10. AI assistance disclosure

GUIDE §3 requires substantive AI help to be acknowledged here. An AI coding assistant (Trae,
GLM/Claude-class model) was used substantively for: implementing `student.py`'s configurable
attention and its numerical-equivalence test against `model.py`; writing the candidate configs;
writing `tests/test_equivalence.py` and `check_integrity.py`; adding the two opt-in `train.py`
switches for the 1-D-decay and EMA experiments; running the training and evaluation commands and
aggregating results; and drafting the report with the specification documents.

Authored and decided by the student: the diagnosis-first approach, the equal-parameter frontier
design, the noise-floor-first protocol, holding the pre-registered target fixed and refusing to
widen the ±3% parameter band after the selected point came in at +3.06%, and the decisions to
disclose every falsified mechanism, the neutral levers, the repeated test scoring, the success
criterion that no longer holds at the submitted operating point, and the limitations — rather than
present only the winning direction. Every number in the report was reviewed against the underlying
`runs/` artefacts.

No pretrained weights, external training text or external model was used. All weights, statistics
and selection decisions derive only from the supplied training text and validation split. No answer
caching, retrieval database or lookup index is used; the model is a plain causal transformer.
`common.py`, `evaluate.py`, `model.py`, `tests/test_contract.py` and `data/` were never modified,
and `check_integrity.py` re-verifies their hashes.

The three files this project did change are `student.py` (the submitted model factory), `train.py`
(the two opt-in switches, whose defaults reproduce the supplied recipe exactly) and the repository
documentation. The supplied `code/README.md` was superseded by this document and has been deleted,
so that the repository has a single documentation entry point; its content lives in §1–§4, §8, §10
and §11 below. [GUIDE.md](GUIDE.md) names that path, but its own link to it was already stale,
because the guide sits at the repository root after the package was flattened and `../code/`
therefore resolves outside the repository. GUIDE.md is supplied documentation and was left
unmodified.

## 11. Data attribution

WikiText-2 was introduced by Stephen Merity, Caiming Xiong, James Bradbury and Richard Socher in
[Pointer Sentinel Mixture Models](https://arxiv.org/abs/1609.07843). The text is by Wikipedia
contributors. The [upstream dataset](https://huggingface.co/datasets/Salesforce/wikitext)
identifies [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) and the
[GNU Free Documentation License](https://www.gnu.org/licenses/fdl-1.3.html); **retain these
notices when redistributing the data**.

The supplied `wikitext-2-raw-v1` splits preserve revision
`b08601e04326c79dfdd32d625aee71d232d685c3`. Rows are joined with newlines and encoded as UTF-8;
the tokenizer is fitted only to training text. Dataset hashes are in `data/manifest.json`. These
dataset notices do not assign a new license to the surrounding classroom code.