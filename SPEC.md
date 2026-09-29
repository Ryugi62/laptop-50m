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
- AC-23 Given a saved weights file (a bare state_dict or a training checkpoint with `model`), When `count_checkpoint_params(path)` runs, Then it counts each stored tensor once even when two keys share storage (tied embedding/head), the result equals `analytic_param_count` of the stored config, and the naive per-key sum (which counts the tied matrix twice) is reported separately.
- AC-24 Given a model and a prompt, When `sample(model, ids, new_tokens, temperature, top_k, seed)` runs, Then the same seed gives the same tokens, the output keeps the prompt and adds exactly `new_tokens` ids inside the vocabulary, and temperature 0 equals greedy argmax (the context is cropped to the model's window).
- AC-25 Given a parameter budget that counts the (tied) embedding, a width and a vocabulary size, When `layers_for_budget` runs, Then it returns the deepest config that fits (one more layer would not), larger vocabularies get fewer layers, a vocabulary that leaves no room for one block is rejected, and `bits_per_byte` converts a text's total NLL in nats to bits per UTF-8 byte (the tokenizer-independent metric the vocabulary ablation is compared on).

## §5b Evaluation (Track 01 rules)
- `python -m laptop50m.infrastructure.eval_cli` = lm-evaluation-harness 0-shot `hellaswag`, `arc_easy`, `piqa`, `winogrande` (acc, acc_norm, full sets)
  + `wikitext103` (local task = lm-eval `wikitext` on the `wikitext-103-raw-v1` document-level test split: word/byte perplexity, bits/byte)
  + token-level sliding-window perplexity (window 512, stride 256) on the WikiText-103 test/validation token streams. Output `results/eval_l50m-v1.json`.

## §6 Layers
`domain/` (pure python: config, schedule) <- `application/` (training loop over ports) <- `adapters/` (torch model, memmap data) <- `infrastructure/` (CLI, tokenizer training, idle gate). No reverse imports.

## §7 Data
FineWeb-Edu `sample/10BT` shard 000 (ODC-By 1.0), WikiText-103 validation/test (CC BY-SA 3.0) for held-out perplexity only.
