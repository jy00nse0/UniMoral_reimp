#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 10 (105분) — RQ3 나머지
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode culture
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode fs
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode fs
