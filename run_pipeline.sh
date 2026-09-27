#!/bin/sh
# Laptop-50M pipeline: prepare data -> train. Heavy stages wait for idle (>=120 s) and run at nice 20.
cd "$(dirname "$0")" || exit 1
PY=.venv/bin/python
nice -n 20 $PY -m laptop50m.infrastructure.prepare_data --raw _data/raw --out _data/tok || exit 2
exec nice -n 20 $PY -m laptop50m.infrastructure.train_cli --tok-dir _data/tok --out runs/l50m-v1 --max-steps 9000 --max-hours 72 --max-lr 1e-3
