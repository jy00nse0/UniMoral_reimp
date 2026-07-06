#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 9 (105분) — RQ1 fs 시급분 + RQ3 English fs
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode fs       # 갭14.58 [11위]
python RQ1_debug.py --model "$MODEL" --language Arabic   --mode fs       # 갭14.08 [13위]
python RQ3_debug.py --model "$MODEL" --language English  --mode fs
