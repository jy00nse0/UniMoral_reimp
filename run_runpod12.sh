#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 12 (105분) — RQ3 나머지
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode moral
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode desc
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode moral
