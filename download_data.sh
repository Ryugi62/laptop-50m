#!/bin/sh
# Raw data used for Laptop-50M (sizes verified against the Hugging Face Hub on 2026-09-28).
#   FineWeb-Edu sample/10BT shard 000 (ODC-By 1.0)           2,152,819,114 bytes
#   WikiText-103 raw validation / test (CC BY-SA 3.0)          657,209 / 732,610 bytes
set -e
cd "$(dirname "$0")"
mkdir -p _data/raw
HF=https://huggingface.co/datasets
curl -L -C - -o _data/raw/fineweb_edu_000.parquet "$HF/HuggingFaceFW/fineweb-edu/resolve/main/sample/10BT/000_00000.parquet"
curl -L -o _data/raw/wikitext103_validation.parquet "$HF/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet"
curl -L -o _data/raw/wikitext103_test.parquet "$HF/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/test-00000-of-00001.parquet"
