#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 11 (105분) — RQ3 나머지
python RQ3_debug.py --model "$MODEL" --language Arabic   --mode fs
python RQ3_debug.py --model "$MODEL" --language Russian  --mode moral
python RQ3_debug.py --model "$MODEL" --language Russian  --mode desc
