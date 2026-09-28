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
for f in fineweb_val.bin wikitext103_validation.bin wikitext103_test.bin tokenizer.json; do
  [ -s "$DATA/tok/$f" ] || curl -sSfL --retry 5 -o "$DATA/tok/$f" "$REL/$f" || { echo "download failed $f"; exit 2; }
done
# token shards: released pre-tokenized copy if present, otherwise tokenize here from the Hugging Face parquet
HF=https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT
WT=https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1
SPECS=""
for i in ${MAIN_IDS:-000 001}; do SPECS="$SPECS train_$i"; done
for i in ${HQ_IDS:-002 003}; do SPECS="$SPECS hq_$i"; done
for spec in $SPECS; do
  f=$spec.bin; id=${spec#*_}; kind=${spec%_*}
  [ -s "$DATA/tok2/$f" ] && continue
  curl -sSfL --retry 3 -o "$DATA/tok2/$f" "$REL/$f" && continue
  rm -f "$DATA/tok2/$f"; mkdir -p "$DATA/raw"
  [ -s "$DATA/raw/wikitext103_test.parquet" ] || { curl -sSfL -o "$DATA/raw/wikitext103_validation.parquet" "$WT/validation-00000-of-00001.parquet"; curl -sSfL -o "$DATA/raw/wikitext103_test.parquet" "$WT/test-00000-of-00001.parquet"; }
  [ -s "$DATA/raw/fineweb_edu_$id.parquet" ] || curl -sSfL --retry 5 -o "$DATA/raw/fineweb_edu_$id.parquet" "$HF/${id}_00000.parquet"
  if [ "$kind" = train ]; then M=$id; H=""; else M=""; H=$id; fi
  python -m laptop50m.infrastructure.prepare_v2 --raw "$DATA/raw" --out "$DATA/tok2" --tokenizer "$DATA/tok/tokenizer.json" --main "$M" --hq "$H" --workers "$(nproc)" | tee -a "$OUT/stdout.log"
  rm -f "$DATA/raw/fineweb_edu_$id.parquet"
done
[ -n "${PREP_ONLY:-}" ] && { echo "[run_v2] data ready"; ls -la "$DATA/tok2"; exit 0; }
nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a "$OUT/stdout.log"
ARGS="--data $DATA/tok2 --val-dir $DATA/tok --out $OUT --tokens $TOKENS ${EXTRA:-}"
if [ "$NPROC" -gt 1 ]; then RUN="torchrun --standalone --nproc_per_node $NPROC -m laptop50m.infrastructure.train_cloud"; else RUN="python -m laptop50m.infrastructure.train_cloud"; fi
# try torch.compile first; if it dies, resume from the checkpoint without compile
$RUN $ARGS --compile 2>&1 | tee -a "$OUT/stdout.log"
if [ "${PIPESTATUS[0]}" -ne 0 ]; then
  echo "[run_v2] compile run failed -> eager" | tee -a "$OUT/stdout.log"
  $RUN $ARGS 2>&1 | tee -a "$OUT/stdout.log"
fi

# evaluation on the same GPU once training reached its token target (re-running this script after the run = eval only)
if python - "$OUT/ckpt.pt" <<'PY'
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
tc = ck["train_config"]; sys.exit(0 if ck["step"] >= tc["max_steps"] else 1)
PY
then
  [ -s "$OUT/results/eval_l50m-v2.json" ] && { echo "[run_v2] eval exists"; exit 0; }
  python -c "import lm_eval" 2>/dev/null || pip -q install lm-eval==0.4.13
  python -m laptop50m.infrastructure.eval_cli --ckpt "$OUT/ckpt.pt" --tokenizer "$DATA/tok/tokenizer.json" \
    --tok-dir "$DATA/tok" --out "$OUT/results" --device cuda --seq-len 1024 --batch-size 64 --name l50m-v2 2>&1 | tee -a "$OUT/stdout.log"
fi
