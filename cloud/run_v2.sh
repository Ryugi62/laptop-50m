#!/bin/bash
# Laptop-50M v2 on a free cloud GPU (Colab T4 / Kaggle T4x2). Idempotent: re-run after a disconnect to resume.
#   OUT=<persistent dir> TOKENS=1.5e9 bash cloud/run_v2.sh
set -u
cd "$(dirname "$0")/.."
OUT=${OUT:-/content/drive/MyDrive/laptop50m-v2/run}
DATA=${DATA:-/content/data}
TOKENS=${TOKENS:-1.5e9}
NPROC=${NPROC:-$(python -c "import torch;print(max(1,torch.cuda.device_count()))")}
REL=https://github.com/Ryugi62/laptop-50m/releases/download/data-v2
mkdir -p "$DATA/tok2" "$DATA/tok" "$OUT"
python -c "import tokenizers, pyarrow" 2>/dev/null || pip -q install tokenizers pyarrow
for f in train_000.bin train_001.bin hq_002.bin hq_003.bin manifest.json; do
  [ -s "$DATA/tok2/$f" ] || curl -sSfL --retry 5 -o "$DATA/tok2/$f" "$REL/$f" || { echo "download failed $f"; exit 2; }
done
for f in fineweb_val.bin wikitext103_validation.bin wikitext103_test.bin tokenizer.json; do
  [ -s "$DATA/tok/$f" ] || curl -sSfL --retry 5 -o "$DATA/tok/$f" "$REL/$f" || { echo "download failed $f"; exit 2; }
done
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$OUT/stdout.log"
ARGS="--data $DATA/tok2 --val-dir $DATA/tok --out $OUT --tokens $TOKENS ${EXTRA:-}"
if [ "$NPROC" -gt 1 ]; then RUN="torchrun --standalone --nproc_per_node $NPROC -m laptop50m.infrastructure.train_cloud"; else RUN="python -m laptop50m.infrastructure.train_cloud"; fi
# try torch.compile first; if it dies, resume from the checkpoint without compile
$RUN $ARGS --compile 2>&1 | tee -a "$OUT/stdout.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "[run_v2] compile run failed -> eager" | tee -a "$OUT/stdout.log"
  $RUN $ARGS 2>&1 | tee -a "$OUT/stdout.log"
fi
