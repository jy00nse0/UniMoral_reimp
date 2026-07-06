#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 17 (105분) — RQ1
python RQ1_debug.py --model "$MODEL" --language Arabic   --mode desc
python RQ1_debug.py --model "$MODEL" --language Russian  --mode culture
python RQ1_debug.py --model "$MODEL" --language Russian  --mode moral
