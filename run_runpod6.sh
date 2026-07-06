#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 6 (105분) — 시급 갭 1·5·7위
python RQ3_debug.py --model "$MODEL" --language Russian  --mode fs       # 갭30.52 [1위]
python RQ3_debug.py --model "$MODEL" --language Russian  --mode culture  # 갭17.61 [5위]
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode culture  # 갭16.54 [7위]
