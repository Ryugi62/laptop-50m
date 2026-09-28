"""Optional: the same v2 run on Modal (H100). Needs a Modal account; set a workspace budget before use.

    modal run cloud/modal_v2.py::prep      # CPU-only tokenization into the volume (~20 min)
    modal run --detach cloud/modal_v2.py::train --tokens 12e9
"""
import subprocess

import modal

app = modal.App("laptop50m-v2")
vol = modal.Volume.from_name("laptop50m-v2", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "curl")
         .pip_install("torch==2.8.0", "numpy", "tokenizers==0.23.2", "pyarrow")
         .run_commands("git clone -b v2-cloud https://github.com/Ryugi62/laptop-50m /repo"))
IDS = "MAIN_IDS='000 001 002 003 004 005 006 007 008 009 010 011' HQ_IDS='012 013'"


def _sh(cmd: str):
    subprocess.run(f"cd /repo && git pull -q && {IDS} OUT=/vol/run DATA=/vol/data {cmd}", shell=True, check=True)
    vol.commit()


@app.function(image=image, cpu=32, memory=65536, timeout=4 * 3600, volumes={"/vol": vol})
def prep():
    _sh("PREP_ONLY=1 bash cloud/run_v2.sh")


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=24 * 3600, volumes={"/vol": vol})
def train(tokens: float = 12e9):
    _sh(f"TOKENS={tokens} EXTRA='--global-batch 524288 --micro 64 --anneal-frac 0.1 --ckpt-every 1200' bash cloud/run_v2.sh")
