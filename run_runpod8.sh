#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 8 (120분) — RQ2 moral/culture 시급분
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode culture  # 갭22.87 [2위]
python RQ2_debug.py --model "$MODEL" --language Russian  --mode moral    # 갭17.22 [6위]
python RQ2_debug.py --model "$MODEL" --language English  --mode moral    # 갭13.35 [14위]
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode moral    # 갭25.89 [15위]
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode moral
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode moral
