#!/bin/bash
set -e

echo "=== 패키지 설치 ==="
pip install \
    transformers==4.43.3 \
    torch==2.4.0 \
    pandas \
    scikit-learn \
    datasets \
    huggingface_hub \
    tqdm \
    numpy \
    bert-score \
    nltk \
    sacrebleu \
    accelerate

echo "=== NLTK 데이터 다운로드 ==="
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('punkt')"

echo "=== 설치 확인 ==="
python3 -c "
import transformers, torch, sklearn, datasets
print('transformers:', transformers.__version__)
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU count:', torch.cuda.device_count())
"

echo "=== 완료 ==="
