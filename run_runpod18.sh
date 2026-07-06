#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 18 (140분) — RQ1 나머지 + RQ2 Chinese 잔여
python RQ1_debug.py --model "$MODEL" --language Russian  --mode desc
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode culture
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode moral
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode desc
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode moral
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode culture
