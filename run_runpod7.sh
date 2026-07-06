#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 7 (105분) — 시급 갭 9·10·12위
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode fs       # 갭14.94 [9위]
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode culture  # 갭14.82 [10위]
python RQ3_debug.py --model "$MODEL" --language Arabic   --mode culture  # 갭14.31 [12위]
