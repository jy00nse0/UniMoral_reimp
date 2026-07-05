import os
os.environ['HF_HOME'] = '<Path for HF cache>'

import transformers
from packaging import version

# ----------------------------------------------------------------------------
# Llama 3.1 rope_scaling 호환성 패치 (RQ1/RQ2/RQ3와 동일)
# ----------------------------------------------------------------------------
if version.parse(transformers.__version__) < version.parse("4.43"):
    from transformers.models.llama import configuration_llama as _llama_cfg
    from transformers.models.llama import modeling_llama as _llama_model

    def _patched_rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict):
            return
        if "rope_type" not in self.rope_scaling and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

    _llama_cfg.LlamaConfig._rope_scaling_validation = _patched_rope_scaling_validation

    _orig_init_rope = _llama_model.LlamaRotaryEmbedding._init_rope if hasattr(
        _llama_model, "LlamaRotaryEmbedding"
    ) else None

    if _orig_init_rope is not None:
        def _patched_init_rope(self):
            rope_type = getattr(self.config, "rope_scaling", {}).get("rope_type", None) \
                if getattr(self.config, "rope_scaling", None) else None
            if rope_type == "llama3":
                self.rope_init_fn = _llama_model._compute_dynamic_ntk_parameters \
                    if hasattr(_llama_model, "_compute_dynamic_ntk_parameters") else None
                if self.rope_init_fn is not None:
                    inv_freq, self.attention_scaling = self.rope_init_fn(
                        self.config, device=self.inv_freq.device if hasattr(self, "inv_freq") else None
                    )
                    self.register_buffer("inv_freq", inv_freq, persistent=False)
                    return
            _orig_init_rope(self)

        _llama_model.LlamaRotaryEmbedding._init_rope = _patched_init_rope

    print(f"[INFO] transformers {transformers.__version__} < 4.43 — Llama 3.1 rope_scaling 패치 적용")
else:
    print(f"[INFO] transformers {transformers.__version__} >= 4.43 — 패치 불필요")

from transformers import pipeline
from huggingface_hub import login
import argparse
import pandas as pd
import ast
from tqdm import tqdm
import json
import random
import numpy as np
import torch
import nltk
import math
import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from bert_score import score

# [수정] nltk 리소스(wordnet, omw-1.4) 다운로드 누락 — single_meteor_score가
# wordnet을 필요로 하므로 최초 실행 시 LookupError가 발생하지 않도록 미리 다운로드.
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('punkt', quiet=True)

access_token ='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'
login(token=access_token)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

act2idx = {'a': 'a', 'b': 'b', '1': 'a', '2': 'b', 1: 'a', 2: 'b'}
bs_lang_dict = {'English': 'en', 'Spanish': 'es', 'Arabic': 'ar', 'Russian': 'ru', 'Chinese': 'zh', 'Hindi': 'hi'}


def evaluate_metrics(predictions, ground_truths, bs_lang):
    """
    [수정] BERTScore를 샘플-참조 쌍마다 개별 호출하던 이중 루프를 제거하고,
    전체 (prediction, reference) 쌍을 한 번에 배치로 score()에 넘기도록 변경.
    기존 방식은 매 호출마다 모델을 다시 태우는 구조라 RQ1/RQ2 대비 런타임이
    크게 늘어나고, 긴 데이터셋에서는 세션 중단 위험이 컸음.
    """
    smoothing_function = SmoothingFunction().method1

    bleu_scores = []
    meteor_scores = []

    # BERTScore 배치 계산을 위한 평탄화 + 인덱스 매핑
    flat_preds = []
    flat_refs = []
    sample_idx_map = []

    for sample_idx, (pred, refs) in enumerate(zip(predictions, ground_truths)):
        # BLEU
        sample_bleu = []
        pred_tokens = pred.split()
        for ref in refs:
            ref_tokens = ref.split()
            score_value = sentence_bleu(
                [ref_tokens],
                pred_tokens,
                smoothing_function=smoothing_function
            )
            sample_bleu.append(score_value)
        max_bleu = max(sample_bleu) if sample_bleu else 0.0
        bleu_scores.append(max_bleu)

        # METEOR
        sample_meteor = []
        for ref in refs:
            score_value = single_meteor_score(ref.split(), pred.split())
            sample_meteor.append(score_value)
        max_meteor = max(sample_meteor) if sample_meteor else 0.0
        meteor_scores.append(max_meteor)

        # BERTScore용 평탄화
        for ref in refs:
            flat_preds.append(pred)
            flat_refs.append(ref)
            sample_idx_map.append(sample_idx)

    # [수정] BERTScore를 전체 배치로 한 번만 계산
    bert_f1_per_sample = [[] for _ in predictions]
    if flat_preds:
        P, R, F1 = score(flat_preds, flat_refs, lang=bs_lang, verbose=False)
        F1 = F1.tolist()
        for sample_idx, f1_val in zip(sample_idx_map, F1):
            bert_f1_per_sample[sample_idx].append(f1_val)

    bert_f1_scores = [max(scores) if scores else 0.0 for scores in bert_f1_per_sample]

    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0.0
    avg_meteor = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
    avg_bert = sum(bert_f1_scores) / len(bert_f1_scores) if bert_f1_scores else 0.0

    return {
        'bleu': bleu_scores,
        'avg_bleu': avg_bleu,
        'meteor': meteor_scores,
        'avg_meteor': avg_meteor,
        'bert_score': bert_f1_scores,
        'avg_bert_score': avg_bert
    }


def dict_to_list_of_vectors(data):
    data = ast.literal_eval(data)
    return list(data.values())

def parse_scenario(scenario_str):
    """CSV에 ['...'] 형태로 저장된 시나리오를 순수 문자열로 변환.
    Chinese의 <unk> 토큰도 동시에 제거."""
    try:
        parsed = ast.literal_eval(scenario_str)
        if isinstance(parsed, list) and len(parsed) > 0:
            scenario_str = parsed[0]
    except:
        pass
    scenario_str = scenario_str.replace('<unk>', '')
    return scenario_str


def read_data_RQ4(data_file_long):
    print("Reading data...")
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id', 'Annotator_id', 'Scenario', 'Possible_actions', 'Selected_action',
                 'Consequence', 'Moral_values', 'Cultural_values', 'Annotator_self_description']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

    scenario_consequences = {}

    unique_scenario_id = data['Scenario_id'].unique()
    for s_id in unique_scenario_id:
        s_data = data[data['Scenario_id'] == s_id]
        action_consequence = {}
        for acts, sel_act, con in zip(s_data['Possible_actions'], s_data['Selected_action'], s_data['Consequence']):
            possible_actions = ast.literal_eval(acts)
            if type(possible_actions[0]) != str:
                sel_act = act2idx[sel_act] + ": " + possible_actions[0][sel_act - 1]
            else:
                sel_act = act2idx[sel_act] + ": " + possible_actions[sel_act - 1]

            try:
                if math.isnan(float(con)):
                    continue
            except (ValueError, TypeError):
                pass

            con = con.replace('[', '')
            con = con.replace(']', '')
            con = " ".join(con.lower().strip().split())

            if sel_act not in action_consequence.keys():
                action_consequence[sel_act] = []
            action_consequence[sel_act].append(con)
        scenario_consequences[s_id] = action_consequence

    main_data = []
    for s_id, act_con in scenario_consequences.items():
        scenario = parse_scenario(data[data['Scenario_id'] == s_id]['Scenario'].values[0])
        for sel_act, cons in act_con.items():
            this_inst = {
                'scenario_id': s_id,
                'scenario': scenario,
                'selected_action': sel_act,
                'gt': cons
            }
            main_data.append(this_inst)

    return main_data


def parse_consequence(generated_text, is_instruct):
    """
    [수정 추가] 파싱 로직을 함수로 분리하고, 고정 키워드 'consequence of the action is'가
    없을 경우를 대비해 다중 후보 키워드와 정규식 fallback을 추가.
    기존 코드는 split 실패 시 원문 전체를 그대로 사용해 BLEU/METEOR/BERTScore가
    비정상적으로 낮아지는 문제가 있었음.
    """
    text = generated_text.lower().strip()

    candidate_markers = [
        "consequence of the action is",
        "the consequence is",
        "the likely consequence is",
        "consequence:",
    ]

    for marker in candidate_markers:
        if marker in text:
            return text.split(marker, 1)[1].strip()

    # 정규식 fallback: "consequence ... is" 형태 탐색
    match = re.search(r"consequence[^.]*?\bis\b[:\s]*(.*)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # 끝까지 못 찾으면 원문 그대로 반환 (단, 호출부에서 경고 로그를 남김)
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='Model name for pipeline')
    parser.add_argument('--language', default='English', type=str,
                         help='Language out of ["English", "Chinese", "Russian", "Arabic", "Spanish", and "Hindi"]')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size used for generation')
    parser.add_argument('--checkpoint_every', default=10, type=int,
                         help='[수정 추가] 몇 배치마다 중간 결과를 저장할지. max_new_tokens=2000으로 '
                              '소요 시간이 길어 런타임 중단 위험이 높으므로 기본값을 RQ3보다 짧게 설정.')
    parser.add_argument('--debug_only', action='store_true', help='프롬프트 샘플 출력 후 즉시 종료 (모델 로딩 없음)')
    args = parser.parse_args()

    RQ = 4
    language = args.language
    bs_lang = bs_lang_dict[language]
    model_name = args.model
    batch_size = args.batch_size
    checkpoint_every = args.checkpoint_every
    debug_only = args.debug_only

    # [수정] Final_results 디렉터리 미생성 문제 해결
    os.makedirs("Final_results", exist_ok=True)

    with open("PROMPTS4.txt", "r") as f:
        PROMPTS = f.read()
    PROMPTS = ast.literal_eval(PROMPTS)

    if not debug_only:
        pipe = pipeline("text-generation", model=model_name, device_map="auto", truncation=True)

    is_instruct = "instruct" in model_name.lower()

    data_file_rq4 = f"Final_data/{language}_long.csv"
    data = read_data_RQ4(data_file_rq4)

    formatted_prompts = []
    ground_truth = []

    for i, inst in tqdm(enumerate(data)):
        ground_truth.append(inst['gt'])

        prompt = PROMPTS[f"prompt_{language.lower()[:3]}_rq{RQ}"]

        prompt = prompt.replace("[SCENARIO]", inst['scenario'])
        prompt = prompt.replace("[SELECTED_ACTION]", inst['selected_action'])

        if is_instruct:
            formatted_prompt = f"""
                ### Instruction:
                {prompt}

                ### Response:
            """
        else:
            formatted_prompt = [
                {"role": "user", "content": prompt},
            ]

        if i < 5:
            print(f"\n{'='*60}")
            print(f"[DEBUG] Sample {i+1} / 5")
            print(f"{'='*60}")
            print(formatted_prompt)
            print(f"{'='*60}\n")

        formatted_prompts.append(formatted_prompt)

    if debug_only:
        print(f"[DEBUG] --debug_only 모드: 프롬프트 확인 완료. 종료합니다.")
        import sys; sys.exit(0)

    print("\nGenerating responses...")
    generated_outputs = []

    # [수정 추가] 체크포인트 저장 — max_new_tokens=2000으로 시간이 오래 걸리는 RQ4 특성상 필수.
    # 부분 생성 결과를 저장해 두면 런타임 중단 시 이어서 진행할 수 있는 토대가 된다.
    checkpoint_path = f"Final_results/_checkpoint_{language}_RQ{RQ}_{model_name.split('/')[-1]}.json"

    for i, batch_start in enumerate(tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating batches")):
        batch = formatted_prompts[batch_start:batch_start + batch_size]
        outputs = pipe(
            batch,
            max_new_tokens=2000
        )
        generated_outputs.extend(outputs)

        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"num_done": len(generated_outputs)}, f)

    predictions = []
    n_parse_failed = 0
    for output in generated_outputs:
        if is_instruct:
            generated_text = output[0]['generated_text']
            generated_text = generated_text.split("### Response:")[1].lower().strip()
        else:
            generated_text = output[0]['generated_text'][1]['content']

        parsed = parse_consequence(generated_text, is_instruct)
        if parsed == generated_text.lower().strip():
            # 어떤 마커로도 못 잘랐다는 뜻 — 원문이 그대로 들어간 경우
            n_parse_failed += 1

        predictions.append(parsed)

    print(f"Parsing fallback count (원문 그대로 사용된 샘플 수): {n_parse_failed} / {len(predictions)}")
    print(ground_truth, predictions)

    results = evaluate_metrics(predictions, ground_truth, bs_lang)
    print("Average BLEU Score:", results['avg_bleu'])
    print("Average METEOR Score:", results['avg_meteor'])
    print("Average BERTScore (F1):", results['avg_bert_score'])

    output_results = {
        'predictions': predictions,
        'ground truth': ground_truth,
        'n_parse_failed': n_parse_failed,
        'Per sample BLEU': results['bleu'],
        'Average BLEU': results['avg_bleu'],
        'Per sample METEOR': results['meteor'],
        'Average METEOR': results['avg_meteor'],
        'Per sample BERTScore': results['bert_score'],
        'Average BERTScore': results['avg_bert_score'],
    }

    with open(f"Final_results/{language}_RQ{RQ}_long_{model_name.split('/')[-1]}.json", "w") as f:
        json.dump(output_results, f, ensure_ascii=False)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
