#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 15 (105분) — RQ1
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode fs
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode culture
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode moral
