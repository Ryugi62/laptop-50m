#!/bin/bash
# Reference point, not part of our model: EleutherAI Pythia-70M (final checkpoint, 300B Pile tokens) through the same
# lm-evaluation-harness 0.4.13 tasks, 0-shot, on the same Mac (CPU), context capped at 512 like Laptop-50M.
# Needs: pip install "transformers>=4.44,<4.57" accelerate
set -e
cd "$(dirname "$0")/.."
OMP_NUM_THREADS=${OMP_NUM_THREADS:-4} TOKENIZERS_PARALLELISM=false .venv/bin/lm_eval --model hf \
  --model_args pretrained=EleutherAI/pythia-70m,revision=step143000,dtype=float32,max_length=512 \
  --tasks hellaswag,arc_easy,piqa,winogrande,wikitext103 --include_path eval_tasks \
  --num_fewshot 0 --device cpu --batch_size 16 --output_path results/baselines/pythia-70m
