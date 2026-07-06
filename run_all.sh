#!/bin/bash
set -e

export PYTHONUNBUFFERED=1
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"

# ── RQ1 비영어권 전 모드 ───────────────────────────────
python RQ1_debug.py --model "$MODEL" --language Chinese  --mode desc
python RQ1_debug.py --model "$MODEL" --language Chinese  --mode moral
python RQ1_debug.py --model "$MODEL" --language Chinese  --mode culture
python RQ1_debug.py --model "$MODEL" --language Chinese  --mode fs
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode desc
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode moral
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode culture
python RQ1_debug.py --model "$MODEL" --language Spanish  --mode fs
python RQ1_debug.py --model "$MODEL" --language Arabic   --mode desc
python RQ1_debug.py --model "$MODEL" --language Arabic   --mode moral
python RQ1_debug.py --model "$MODEL" --language Arabic   --mode culture
python RQ1_debug.py --model "$MODEL" --language Arabic   --mode fs
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode desc
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode moral
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode culture
python RQ1_debug.py --model "$MODEL" --language Hindi    --mode fs
python RQ1_debug.py --model "$MODEL" --language Russian  --mode desc
python RQ1_debug.py --model "$MODEL" --language Russian  --mode moral
python RQ1_debug.py --model "$MODEL" --language Russian  --mode culture
python RQ1_debug.py --model "$MODEL" --language Russian  --mode fs

# ── RQ2 English fs + 비영어권 전 모드 ─────────────────
python RQ2_debug.py --model "$MODEL" --language English  --mode fs
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode desc
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode moral
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode culture
python RQ2_debug.py --model "$MODEL" --language Chinese  --mode fs
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode desc
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode moral
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode culture
python RQ2_debug.py --model "$MODEL" --language Spanish  --mode fs
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode desc
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode moral
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode culture
python RQ2_debug.py --model "$MODEL" --language Arabic   --mode fs
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode desc
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode moral
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode culture
python RQ2_debug.py --model "$MODEL" --language Hindi    --mode fs
python RQ2_debug.py --model "$MODEL" --language Russian  --mode desc
python RQ2_debug.py --model "$MODEL" --language Russian  --mode moral
python RQ2_debug.py --model "$MODEL" --language Russian  --mode culture
python RQ2_debug.py --model "$MODEL" --language Russian  --mode fs

# ── RQ3 English fs + 비영어권 전 모드 ─────────────────
python RQ3_debug.py --model "$MODEL" --language English  --mode fs
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode desc
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode moral
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode culture
python RQ3_debug.py --model "$MODEL" --language Chinese  --mode fs
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode desc
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode moral
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode culture
python RQ3_debug.py --model "$MODEL" --language Spanish  --mode fs
python RQ3_debug.py --model "$MODEL" --language Arabic   --mode desc
python RQ3_debug.py --model "$MODEL" --language Arabic   --mode moral
python RQ3_debug.py --model "$MODEL" --language Arabic   --mode culture
python RQ3_debug.py --model "$MODEL" --language Arabic   --mode fs
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode desc
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode moral
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode culture
python RQ3_debug.py --model "$MODEL" --language Hindi    --mode fs
python RQ3_debug.py --model "$MODEL" --language Russian  --mode desc
python RQ3_debug.py --model "$MODEL" --language Russian  --mode moral
python RQ3_debug.py --model "$MODEL" --language Russian  --mode culture
python RQ3_debug.py --model "$MODEL" --language Russian  --mode fs

# ── RQ4 비영어권 ───────────────────────────────────────
python RQ4_debug.py --model "$MODEL" --language Chinese
python RQ4_debug.py --model "$MODEL" --language Spanish
python RQ4_debug.py --model "$MODEL" --language Arabic
python RQ4_debug.py --model "$MODEL" --language Hindi
python RQ4_debug.py --model "$MODEL" --language Russian
