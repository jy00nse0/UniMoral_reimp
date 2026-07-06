#!/bin/bash
set -e
export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# Runpod 19 (160분) — RQ2 desc/culture 잔여분
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode desc
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode desc
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode desc
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode culture
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode desc
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode culture
python RQ2_debug.py --model "$MODEL" --language Russian  --mode desc
python RQ2_debug.py --model "$MODEL" --language Russian  --mode culture
