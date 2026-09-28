# SPEC — Laptop-50M (GIBC V2 Track 01 TECH)

## §0 Purpose
Train a <=50,000,000-parameter decoder-only language model **from scratch** on one Apple M3 laptop (16 GB, MPS),
evaluate it with lm-evaluation-harness (HellaSwag, ARC-Easy, PIQA, WinoGrande) and WikiText-103 perplexity,
and document hardware, wall-clock time and compute (GIBC V2 rules, Track 01).

## §1 Success conditions (numbers)
- S1 parameter count printed by `count_params.py` <= 50,000,000 (includes token embedding and output head).
- S2 training wall-clock <= 14 h on the M3 (`nice 20`, background), checkpoint every <= 20 min, resumable.
- S3 validation loss strictly decreasing between the first and last logged eval (convergence gate, kill 2026-09-27 24:00 KST).
- S4 all tests green (`.venv/bin/python -m pytest -q`).
- S5 heavy stages start only when the user is idle (HIDIdleTime >= 120 s); never kill other processes.

## §2 Non-goals
- No pretrained weights, no fine-tuning, no distillation (rule "trained from scratch").
- No cloud GPU, no paid compute.
- No hosted inference API as the model.

## §3 Ubiquitous language
| term | meaning | code |
|---|---|---|
| ModelConfig | architecture hyper-parameters | `laptop50m.domain.config.ModelConfig` |
| param budget | 50,000,000 total trainable parameters | `PARAM_BUDGET` |
| analytic param count | closed-form count from ModelConfig | `analytic_param_count` |
| LR schedule | warmup -> constant -> linear decay (WSD) | `laptop50m.domain.schedule.lr_at` |
| token shard | uint16 memmap of token ids | `laptop50m.adapters.data.TokenShard` |
| idle gate | wait until HIDIdleTime >= threshold | `laptop50m.infrastructure.idle_gate` |

## §4 Model
Decoder-only transformer: RMSNorm (pre-norm), RoPE, SwiGLU MLP, no biases, **tied input/output embedding**,
BPE vocab 16,384 (byte-level). Budget goes to depth instead of a large vocabulary.

## §5 Given / When / Then
- AC-1 Given the default ModelConfig, When `analytic_param_count` runs, Then it is <= PARAM_BUDGET.
- AC-2 Given a ModelConfig whose count exceeds the budget, When `check_budget` runs, Then it raises ValueError.
- AC-3 Given the torch model built from a ModelConfig, When counting `requires_grad` parameters, Then it equals `analytic_param_count` exactly.
- AC-4 Given the torch model, When comparing embedding and head weights, Then they are the same tensor (tied).
- AC-5 Given tokens (B, T), When forward runs with targets, Then logits are (B, T, V) and loss is a finite scalar.
- AC-6 Given warmup W, total S, decay fraction d, When `lr_at` is called, Then lr rises linearly to max by W, stays constant, and decays linearly to min_lr at S.
- AC-7 Given a uint16 token file, When `TokenShard.batch` samples, Then x and y are (B, T) int64 and y == x shifted by one.
- AC-8 Given ioreg output text, When `parse_hid_idle_seconds` parses it, Then it returns seconds as int.
- AC-9 Given a checkpoint saved at step k, When training resumes, Then it continues from step k with the same optimizer state.
- AC-10 Given a run config JSON, When `python count_params.py --config <json>` runs, Then it prints the total trainable parameter count of the instantiated torch model (tied weights counted once) and whether it is <= 50,000,000; exit code 1 if over.
- AC-11 Given a (context, continuation) token pair, When `Laptop50MLM._loglikelihood_tokens` scores it, Then the log-likelihood equals the sum of log-softmax of the continuation tokens from one causal forward pass, and `is_greedy` is True iff every continuation token is the argmax; an empty context is conditioned on `<|endoftext|>`.
- AC-12 Given several requests, When scored with batch size 1 and batch size 3, Then results agree within 1e-4 (right padding is invisible to scored positions).
- AC-13 Given a context longer than `max_length`, When scored, Then only the last `max_length + 1` tokens of context+continuation are used (left truncation), without error.
- AC-14 Given a text, When `loglikelihood_rolling` runs, Then every token is scored exactly once in disjoint windows of `max_length` (lm-eval `get_rolling_token_windows`, context_len=1, EOT prefix).
- AC-15 Given a token stream, When `sliding_window_nll(window, stride)` runs, Then every token after the first is scored exactly once, the result equals one full forward pass when the stream fits one window, and batching windows does not change the result.

## §5b Evaluation (Track 01 rules)
- `python -m laptop50m.infrastructure.eval_cli` = lm-evaluation-harness 0-shot `hellaswag`, `arc_easy`, `piqa`, `winogrande` (acc, acc_norm, full sets)
  + `wikitext103` (local task = lm-eval `wikitext` on the `wikitext-103-raw-v1` document-level test split: word/byte perplexity, bits/byte)
  + token-level sliding-window perplexity (window 512, stride 256) on the WikiText-103 test/validation token streams. Output `results/eval_l50m-v1.json`.

## §6 Layers
`domain/` (pure python: config, schedule) <- `application/` (training loop over ports) <- `adapters/` (torch model, memmap data) <- `infrastructure/` (CLI, tokenizer training, idle gate). No reverse imports.

## §7 Data
FineWeb-Edu `sample/10BT` shard 000 (ODC-By 1.0), WikiText-103 validation/test (CC BY-SA 3.0) for held-out perplexity only.

## §8 v2 — more tokens on free-tier GPU (added 2026-09-28)
Purpose: raise axes 1-2 (Perplexity & Accuracy, Reasoning) by training the same <=50M architecture on
10x more tokens with a more compute-efficient optimizer, on free cloud GPUs (Colab / Kaggle T4, no spend).
Success (numbers): HellaSwag acc_norm >= 29, ARC-Easy acc >= 45, PIQA acc >= 60 (0-shot, full sets), WikiText-103
word ppl < 80, params <= 50,000,000, cloud spend $0. Non-goals: pretrained weights, distillation, synthetic LLM data.

Ubiquitous language (v2):
| term | meaning | code |
|---|---|---|
| Muon | momentum + Newton-Schulz orthogonalised update for 2-D hidden matrices | `laptop50m.adapters.muon.Muon` |
| QK-norm | RMS-normalise per-head queries and keys before RoPE (no parameters) | `ModelConfig.qk_norm` |
| multi-shard | several uint16 token files sampled in proportion to length | `laptop50m.adapters.data.MultiShard` |
| anneal mix | during LR decay, a fraction of rows come from the high-quality (int_score >= 4) shard | `TrainConfig.anneal_bins/anneal_frac` |
| decontamination | drop training docs sharing a word 13-gram with WikiText-103 val/test | `laptop50m.domain.decontam` |

- AC-16 Given `qk_norm=True`, When the model is built, Then the trainable parameter count equals `analytic_param_count` (QK-norm adds none) and forward is finite.
- AC-17 Given a 2-D matrix G, When `newton_schulz(G)` runs 5 steps, Then >= 90% of the singular values of the result lie in [0.5, 1.5] and none exceeds 1.5 (approximately orthogonal; near-zero singular values of a random square matrix stay small after 5 steps), for both tall and wide G, and a (3, d, d) stack is orthogonalised per slice.
- AC-18 Given a toy regression, When Muon + AdamW (hybrid) train for 50 steps, Then the loss decreases by >= 50%; Muon only receives 2-D block matrices; embedding and norms go to AdamW.
- AC-19 Given two token files of lengths 1:3, When `MultiShard.batch` samples many windows, Then about 25% come from the first (±5%), and x/y are shifted by one.
- AC-20 Given a word-13-gram index of reference texts, When `is_contaminated(doc)` runs, Then a doc containing a 13-word span of the reference (case/punctuation-insensitive) is flagged and an unrelated doc is not.
- AC-21 Given `anneal_frac=f` and a run in its decay phase, When batches are drawn, Then rows come from the anneal shard with fraction f; before the decay phase, 0.
- AC-22 Given `optimizer="muon"` and fp16 with a GradScaler on CPU-emulated settings, When training resumes from a checkpoint, Then both optimizers' states are restored and training continues to max_steps (extends AC-9).
