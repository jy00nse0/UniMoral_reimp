#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 14 (105분) — RQ3 나머지 + RQ1 Russian fs
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode moral
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode desc
python RQ1_debug.py --model "$MODEL" --language Russian  --mode fs
