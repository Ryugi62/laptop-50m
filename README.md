# Laptop-50M

A 49.3M-parameter language model trained **from scratch on one Apple M3 iMac** (16 GB unified memory, MPS backend) for
GIBC V2 Track 01 (TECH: ≤ 50M parameters). The model trained in the background while the machine was in daily use,
and all compute ran locally at zero cost. Everything in this README can be reproduced with the scripts in this repository.

For scale, we also ran EleutherAI's Pythia-70M through the same harness on the same Mac
([comparison](#same-harness-same-mac-pythia-70m-as-a-reference-point)). It used about 1,450× our training compute. The two models
are level on HellaSwag (the gap is inside one standard error), Laptop-50M is 1.9 points higher on ARC-Easy (about two standard
errors, one run each), and Pythia-70M is ahead on PIQA, WinoGrande and WikiText. Our model is undertrained (about 6 tokens per
parameter), and that, not the architecture, limits the benchmark scores.

| | |
|---|---|
| Total trainable parameters | **49,295,872**, including the token embedding and the output head (tied) — `python count_params.py` |
| Training data | 294.9M tokens of FineWeb-Edu (`sample/10BT`, shard 000). No pretrained weights, no fine-tuning, no distillation |
| Hardware | iMac (Mac15,5), Apple M3 (8-core CPU, 10-core GPU), 16 GB, macOS 26.5.1, PyTorch 2.8.0 MPS, bf16 autocast |
| Training time | **39.6 h wall-clock** (process time incl. pausing for the user), **≈ 20.5 h at full speed** (294.9M tok / 3,995 tok/s median) |
| Compute | 6·N·D = 6 × 49.3M × 294.9M ≈ **8.7 × 10¹⁶ FLOPs** (non-embedding N: 7.2 × 10¹⁶) |
| Peak memory | ≈ 4.7 GB (batch 4 × 512, bf16) |
| Inference | 197 MB of fp32 weights; ≈ 31–38 new tokens/s on the M3 CPU without a KV cache (`generate_cli`, below) |

## Results (0-shot, lm-evaluation-harness 0.4.13, full evaluation sets)

| Task | Metric | Laptop-50M | ± stderr | Random chance |
|---|---|---|---|---|
| HellaSwag (val, 10,042) | acc / acc_norm | 26.51 / **27.30** | 0.44 / 0.44 | 25.0 |
| ARC-Easy (test, 2,376) | acc / acc_norm | **39.48** / 36.03 | 1.00 / 0.99 | 25.0 |
| PIQA (val, 1,838) | acc / acc_norm | **56.75** / 55.60 | 1.16 / 1.16 | 50.0 |
| WinoGrande (val, 1,267) | acc | **50.67** | 1.41 | 50.0 |
| WikiText-103 test (lm-eval, 62 docs) | word perplexity | **105.03** | — | — |
| | byte perplexity / bits per byte | 2.388 / 1.256 | — | — |
| WikiText-103 test, token-level (316,293 tokens) | perplexity per token (window 512, stride 256) | **46.05** | — | — |
| WikiText-103 validation, token-level (276,394 tokens) | perplexity per token (window 512, stride 256) | 48.76 | — | — |

Raw output: [`results/eval_l50m-v1.json`](results/eval_l50m-v1.json) (evaluation wall-clock: 535 s on the same M3 GPU).

## Same harness, same Mac: Pythia-70M as a reference point

To give the numbers above a scale, we ran EleutherAI **Pythia-70M** (final checkpoint `step143000`) through the same
lm-evaluation-harness 0.4.13 tasks, 0-shot, on the same iMac (CPU, context capped at 512 like ours):
[`baselines/run_pythia70m.sh`](baselines/run_pythia70m.sh) → [`results/baselines/pythia-70m.json`](results/baselines/pythia-70m.json).
Pythia-70M is only a measuring stick here. It is not used anywhere in our training, data or tokenizer.

| | **Laptop-50M** | Pythia-70M (reference) |
|---|---|---|
| Parameters (incl. embeddings) | **49,295,872** (tied head) | 70,426,624 (untied head, 50k vocabulary) |
| Training tokens | **294.9M** (FineWeb-Edu) | 299.9B (the Pile: 143,000 steps × 2,097,152 tokens) |
| Training compute, 6·N·D | **8.7 × 10¹⁶ FLOPs** (one M3 iMac) | 1.27 × 10²⁰ FLOPs (≈ 1,450× more) |
| HellaSwag acc_norm | **27.30** ± 0.44 | 27.38 ± 0.45 |
| ARC-Easy acc | **39.48** ± 1.00 | 37.54 ± 0.99 |
| PIQA acc | 56.75 ± 1.16 | **59.85** ± 1.14 |
| WinoGrande acc | 50.67 ± 1.41 | **52.88** ± 1.40 |
| WikiText-103 word perplexity / bits per byte | 105.03 / 1.256 | **68.89 / 1.142** |

What this shows and what it does not: with about 1/1,450 of the compute, the small-vocabulary, deeper design reaches the
same HellaSwag and a higher ARC-Easy score. Pythia is ahead on PIQA and WinoGrande, and clearly ahead on WikiText, whose
articles come from Wikipedia, which is part of the Pile; our training data is FineWeb-Edu only. This is one run of each model.

Notes on the metrics:
- **WikiText-103 perplexity is out-of-domain.** The WikiText dataset was not used for training (FineWeb-Edu is a web crawl and was not deduplicated against WikiText). The lm-eval `wikitext103` task in
  [`eval_tasks/`](eval_tasks/wikitext103) is lm-eval's built-in `wikitext` task (same detokenizer and metrics) pointed at
  the `wikitext-103-raw-v1` config. WikiText-2 and WikiText-103 share the same test articles.
- **Word and byte perplexity do not depend on the tokenizer**, so they are the numbers to compare across models.
  Per-token perplexity depends on the 16k BPE vocabulary. The value logged during training (57.44) was measured on 25
  random 512-token windows with no left context. The table uses a sliding window, so every token has up to 512 tokens of context.

![validation loss](results/loss_curve.svg)

Validation loss (FineWeb-Edu held-out: last 4 row groups, 4.57M tokens) was measured 36 times. It went up only twice, by at most 0.003 (at steps 5,500 and 7,000): 5.566 (step 250) → 3.725 (4,250) → 3.459 (8,500) → **3.410 (9,000)**.
The drop over the last 20% comes from the linear LR decay (WSD schedule).

## Design: spend the 50M budget on depth, not on the vocabulary

The rule counts the embedding and output head. At d=512, a GPT-2 vocabulary (50,257) with an untied head would use
51.5M parameters on its own, which is over the whole budget. Laptop-50M uses two choices:

1. **Tied input/output embedding.** The head reuses the embedding matrix, so it is counted once.
2. **16,384-token byte-level BPE** trained on the same data. Embedding = 8.4M (17% of the budget), and the other
   40.9M (83%) goes to 12 transformer blocks.

| Component | Parameters |
|---|---|
| Token embedding = output head (tied), 16,384 × 512 | 8,388,608 |
| 12 × block (attention 4d² + SwiGLU 3·d·1536 + 2 RMSNorm) | 40,906,752 |
| Final RMSNorm | 512 |
| **Total** | **49,295,872** |

Architecture: decoder-only, pre-norm RMSNorm, RoPE (θ = 10,000), SwiGLU (hidden 1,536), 8 heads, no biases,
context 512. [`configs/l50m-v1.json`](configs/l50m-v1.json) holds the exact config. `count_params.py` instantiates the PyTorch module and counts
`requires_grad` parameters (tied weights once). A unit test checks that the count matches the closed-form formula.

```
$ python count_params.py --config configs/l50m-v1.json
token embedding (tied with output head): 8,388,608
transformer blocks + final norm: 40,907,264
TOTAL trainable parameters: 49,295,872 <= 50,000,000: True
```

**Counting the released weights file directly.** In a tied model, the embedding and the output head are one tensor
stored under two keys. A naive `sum(v.numel() for v in state_dict.values())` counts it twice and prints 57,684,480, which
looks like it breaks the cap. `count_params.py --ckpt` counts each stored tensor once and checks it against the formula:

```
$ python count_params.py --ckpt laptop50m-step9000.pt      # the file from the v1.0-step9000 release
checkpoint: laptop50m-step9000.pt (87 state_dict keys)
tensors stored under two keys (tied, counted once): [('tok_emb.weight', 'lm_head.weight')]
naive sum over keys (counts the tied matrix twice): 57,684,480
unique parameters in the file == config formula (49,295,872): True
TOTAL trainable parameters: 49,295,872 <= 50,000,000: True
```

### Ablation: vocabulary vs depth when the cap counts the embedding (small proxy, CPU)

The 16k vocabulary is the main design choice, so we tested it at a smaller scale where four runs fit on a CPU.
Every run gets the same 8M-parameter cap (embedding counted, head tied), width 192, the same 20.05 MB of FineWeb-Edu
text (one pass, so every run sees the same bytes) and a BPE tokenizer trained on that text. The vocabulary changes, and the
depth fills what the embedding leaves. Runs are compared on bits per byte of 205 held-out documents (1.0 MB, the last row group
of shard 000), which does not depend on the tokenizer.
[`laptop50m/infrastructure/ablate_vocab.py`](laptop50m/infrastructure/ablate_vocab.py) → [`results/ablation_vocab.json`](results/ablation_vocab.json)

| Vocabulary | Layers | Parameters | Embedding share | Training tokens | Held-out bits per byte |
|---|---|---|---|---|---|
| 4,096 | 16 | 7,870,656 | 10% | 5.86M | 1.762 |
| 8,192 | 14 | 7,771,584 | 20% | 5.16M | 1.750 |
| **16,384** | 10 | 7,573,440 | 42% | 4.67M | **1.734** |
| 32,768 | 3 | 7,619,904 | 83% | 4.33M | 1.746 |

In this proxy, 16k was the best of the four. Both extremes lost: a very small vocabulary gives more layers but weaker tokens
(3.4 bytes per token vs 4.3), and a 32k vocabulary leaves only 3 layers. At full scale (d = 512) the same 16k vocabulary
takes only 17% of the budget. Limits: one seed per point, a tiny model trained on 20 MB, and gaps of 0.012 to 0.028 bits
per byte, so we read this as support for the choice, not as proof of an optimum.

## Training setup

| | |
|---|---|
| Tokens | 9,000 steps × 32,768 tokens (batch 4 × 512 × grad-accum 16) = 294,912,000 |
| Optimizer | AdamW β = (0.9, 0.95), weight decay 0.1 (matrices only), grad clip 1.0 |
| LR schedule | WSD: 300 warmup → 1e-3 constant → linear decay to 1e-4 over the last 20% |
| Precision | bf16 autocast on MPS (fp32 master weights) |
| Checkpointing | every 15 min + on SIGTERM; resumable with the optimizer state (one resume after a reboot at step 8,961) |
| Data | FineWeb-Edu `sample/10BT/000_00000.parquet` (ODC-By 1.0): the first 400 row groups (443.9M tokens) for training, the last 4 for validation; tokenizer trained on the first 60 row groups |

**Laptop-friendly training.** The training loop ran on a machine that was in use during the day:
- Heavy stages start only after the user has been idle for ≥ 120 s (`infrastructure/idle_gate.py`, reads `HIDIdleTime`).
- While the user is active, a duty-cycle pacer sleeps 3× the step time after each step (≈ 25% GPU duty) and
  releases the MPS cache (`infrastructure/pacer.py`). When the user is idle, training runs at full speed (≈ 4,000 tok/s).
- Everything runs under `nice 20`. This is why the wall-clock (39.6 h) is about twice the full-speed compute time (≈ 20.5 h).
- An earlier try with fp32 and batch 8 × 1024 used 14.3 GB and swapped (101–211 tok/s), so it was rejected in favour of bf16 at 4 × 512 (≈ 3,000–4,000 tok/s, 4.7 GB).

## Quick check with the released weights (no training, CPU, about 1 minute)

```sh
gh release download v1.0-step9000 -R Ryugi62/laptop-50m     # laptop50m-step9000.pt + tokenizer.json
.venv/bin/python -m laptop50m.infrastructure.eval_cli --ckpt laptop50m-step9000.pt \
    --tokenizer tokenizer.json --tasks arc_easy,piqa --device cpu --out results_check
```

On 2026-09-28 this printed ARC-Easy acc 0.39478 and PIQA acc 0.56746 in 70 s on the M3 CPU, the same values as the
table above (measured on the MPS GPU from the training checkpoint).

## Demo: generate text on a CPU

```sh
.venv/bin/python -m laptop50m.infrastructure.generate_cli --ckpt laptop50m-step9000.pt --tokenizer tokenizer.json \
    --prompt "The water cycle is" --new-tokens 40
```

Output on the M3 CPU (temperature 0.7, top-k 40, seed 0):

```
The water cycle is a major factor in influencing the water cycle. The main role of water is the formation of water and the
production of water. Water is a source of energy and energy. The water cycle is the
[40 new tokens in 1.05 s on CPU (4 threads) = 38.1 tok/s, no KV cache]
[peak process memory 631 MB, weights fp32 197 MB]
```

The text stays on topic and is grammatical, but it repeats itself and says little. That is what 295M training tokens buy at
this size, and we show it as it is.

## Reproduce

```sh
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
./download_data.sh                      # FineWeb-Edu shard 000 (2.15 GB) + WikiText-103 val/test
.venv/bin/python -m pytest -q           # 30 tests (23 when the demo video was recorded)
./run_pipeline.sh                       # tokenizer + tokenization + training -> runs/l50m-v1/ckpt.pt
.venv/bin/python count_params.py --config configs/l50m-v1.json
.venv/bin/python -m laptop50m.infrastructure.eval_cli --ckpt runs/l50m-v1/ckpt.pt --out results
.venv/bin/python -m laptop50m.infrastructure.plot_curve runs/l50m-v1/train_log.jsonl results/loss_curve.svg
baselines/run_pythia70m.sh              # reference point (needs: pip install "transformers>=4.44,<4.57" accelerate)
.venv/bin/python -m laptop50m.infrastructure.ablate_vocab --vocabs 4096,8192,16384,32768   # ablation (CPU, ≈ 1 h)
```

`eval_cli` runs lm-evaluation-harness through a custom `TemplateLM` adapter
([`laptop50m/adapters/lm_eval_adapter.py`](laptop50m/adapters/lm_eval_adapter.py)) with `num_fewshot=0` on
`hellaswag`, `arc_easy`, `piqa`, `winogrande` and `wikitext103`, then adds token-level sliding-window perplexity.
Log-likelihoods come from one causal forward pass per batch, and continuations are scored with right padding. Tests check
that the scores match a manual computation and do not change with batch size (`tests/test_eval.py`).
The trained checkpoint (`runs/l50m-v1/ckpt.pt`, 592 MB with optimizer state) is not tracked by git.

## Repository layout (clean architecture)

```
laptop50m/domain/          config + parameter budget, LR schedule (pure Python, no torch)
laptop50m/application/     training loop, sliding-window perplexity (ports injected)
laptop50m/adapters/        PyTorch model, memmap token shards, lm-eval adapter
laptop50m/infrastructure/  CLIs: prepare_data, train_cli, eval_cli, plot_curve, generate_cli, ablate_vocab, idle gate, pacer
eval_tasks/                lm-eval task config for WikiText-103
baselines/                 Pythia-70M reference run (evaluation only)
tests/                     acceptance tests AC-1 … AC-18 (see SPEC.md)
```

## AI tool use (disclosure)

AI coding assistants are allowed in this hackathon, and we disclose our use in full:
- **Claude Code (Anthropic Claude)** wrote this project as an AI agent working under the entrant's direction.
  It wrote the spec (`SPEC.md`), the tests, all source code, the data pipeline, and the training and evaluation runs, and it drafted this README.
  The entrant set the goal and the constraints (one laptop, zero cost, ≤ 50M parameters) and owns the submission.
- **No AI model is part of the trained model.** No pretrained weights, no fine-tuning, no distillation, and no
  AI-generated training data. The weights were initialised randomly and trained only on FineWeb-Edu text.
  The tokenizer was trained from scratch on the same data.
- All numbers in this README are copied from script output (`results/eval_l50m-v1.json`, `runs/l50m-v1/train_log.jsonl`,
  `count_params.py`, `results/baselines/pythia-70m.json`, `results/ablation_vocab.json`, `generate_cli`).
- Pythia-70M appears only as an evaluation reference, run by `baselines/run_pythia70m.sh`. It is not part of training.

## Research prototype

Laptop-50M is a research prototype. It is not a medical device, not a diagnostic tool, and not financial advice.
Its outputs can be wrong and must not be used for any decision.

## Licenses

Code: MIT. Data: FineWeb-Edu (ODC-By 1.0), WikiText-103 (CC BY-SA 3.0, evaluation only).
Evaluation: EleutherAI lm-evaluation-harness (MIT). Reference model: EleutherAI Pythia-70M (Apache-2.0), evaluation only.

## Weights
Final checkpoint (step 9000, fp32 state_dict, SHA-256 `930c08d5…70ea`) and tokenizer: https://github.com/Ryugi62/laptop-50m/releases/tag/v1.0-step9000
