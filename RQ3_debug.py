import os
os.environ['HF_HOME'] = '<Path for HF cache>'

import transformers
from packaging import version

# ----------------------------------------------------------------------------
# Llama 3.1 rope_scaling 호환성 패치
# transformers < 4.43 환경에서 rope_scaling 딕셔너리에 'rope_type', 'low_freq_factor'
# 등 Llama 3.1 전용 키가 있으면 ValueError가 발생하므로, 구버전일 때만 패치를 적용한다.
# transformers >= 4.43 이면 패치하지 않고 그대로 진행한다.
# ----------------------------------------------------------------------------
if version.parse(transformers.__version__) < version.parse("4.43"):
    from transformers.models.llama import configuration_llama as _llama_cfg
    from transformers.models.llama import modeling_llama as _llama_model

    def _patched_rope_scaling_validation(self):
        if self.rope_scaling is None:
            return
        if not isinstance(self.rope_scaling, dict):
            return
        # rope_type 키가 없으면 type 키를 rope_type으로 정규화
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
                # llama3 전용 동적 NTK 스케일링으로 대체
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
from collections import defaultdict  # [수정] 누락되어 있던 import 추가 (calculate_metrics에서 사용)

from pathlib import Path
_token_path = Path("access_token.txt")
with _token_path.open("r", encoding="utf-8") as _f:
    access_token = _f.readline().strip()
if not access_token:
    raise ValueError(f"Empty token in {_token_path.resolve()}")
login(token=access_token)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)

act2idx = {'a': 'a', 'b': 'b', '1': 'a', '2': 'b', 1: 'a', 2: 'b'}
idx2label = ['Emotions', 'Moral', 'Culture', 'Responsibilities', 'Relationships', 'Legality', 'Politeness', 'Sacred values']
label2idx = {lab: i for i, lab in enumerate(idx2label)}
idx2moral = ['Care', 'Equality', 'Proportionality', 'Loyalty', 'Authority', 'Purity']
idx2culture = ['Power Distance', 'Individualism', 'Motivation', 'Uncertainty Avoidance', 'Long Term Orientation', 'Indulgence']


def calculate_metrics(true_values, pred_values):
    """
    true_values: 샘플별 정답 라벨(복수 가능) 리스트의 리스트
    pred_values: 샘플별 예측 라벨(단일) 리스트

    반환: precision(dict), recall(dict), f1_score(dict), weighted_f1_score(float)
    [수정] 호출부에서 4개 반환값을 (accuracy, precision, recall, f1)으로 잘못 받던 문제를
           (precision, recall, f1, weighted_f1)로 명확히 정정. accuracy는 별도로 계산.
    """
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    total_true = defaultdict(int)

    for ground_truth, prediction in zip(true_values, pred_values):
        found = False
        for gt in ground_truth:
            total_true[gt] += 1
            if prediction == gt:
                tp[gt] += 1
                found = True
        if not found:
            fp[prediction] += 1
            for gt in ground_truth:
                fn[gt] += 1

    precision = {}
    recall = {}
    f1_score = {}

    all_classes = set(i for sublist in true_values for i in sublist)
    all_classes.update(pred_values)

    for cls in all_classes:
        precision[cls] = tp[cls] / (tp[cls] + fp[cls]) if (tp[cls] + fp[cls]) > 0 else 0
        recall[cls] = tp[cls] / (tp[cls] + fn[cls]) if (tp[cls] + fn[cls]) > 0 else 0
        if precision[cls] + recall[cls] > 0:
            f1_score[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
        else:
            f1_score[cls] = 0

    total_instances = sum(total_true.values())
    weighted_f1_score = sum((f1_score[cls] * total_true[cls] for cls in all_classes)) / total_instances if total_instances > 0 else 0.0

    return precision, recall, f1_score, weighted_f1_score


def calculate_accuracy(true_values, pred_values):
    """
    [수정 추가] RQ3는 멀티라벨(복수 정답 가능) 구조이므로,
    예측이 정답 집합 중 하나에 포함되면 맞은 것으로 간주하는 accuracy를 별도로 계산.
    """
    correct = sum(1 for gt, pred in zip(true_values, pred_values) if pred in gt)
    return correct / len(true_values) if len(true_values) > 0 else 0.0


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


def read_data_RQ3(data_file_long):
    print("Reading data...")
    data = pd.read_csv(data_file_long)
    data = data[['Scenario_id', 'Annotator_id', 'Scenario', 'Possible_actions', 'Selected_action',
                 'Contributing_factors', 'Moral_values', 'Cultural_values', 'Annotator_self_description']]
    print("Data read -- ", len(data), "\n", data.columns, "\n", data.head())

    main_data = []
    for i in tqdm(range(len(data))):
        possible_actions = ast.literal_eval(data['Possible_actions'][i])
        sel_idx = data['Selected_action'][i]

        # [점검] act2idx 인덱싱: Selected_action이 정수(1/2)든 문자('a'/'b')든
        # 항상 -1을 해서 0-indexed로 변환되도록 통일.
        # possible_actions가 중첩 리스트([[...],[...]]) 형태인 경우와
        # 단순 문자열 리스트([str, str]) 형태인 경우를 모두 처리.
        if type(possible_actions[0]) != str:
            sel_act = act2idx[sel_idx] + ": " + possible_actions[0][sel_idx - 1]
        else:
            sel_act = act2idx[sel_idx] + ": " + possible_actions[sel_idx - 1]

        moral_vector = dict_to_list_of_vectors(data['Moral_values'][i])
        culture_vector = dict_to_list_of_vectors(data['Cultural_values'][i])

        moral_idx = sorted(range(len(moral_vector)), key=lambda i: moral_vector[i], reverse=True)
        culture_idx = sorted(range(len(culture_vector)), key=lambda i: culture_vector[i], reverse=True)

        contributing_factor = data['Contributing_factors'][i]
        contributing_factor_list = [int(x) for x in ast.literal_eval(contributing_factor)]
        contributing_factor_list = np.array(contributing_factor_list)
        contributing_factor_list = np.where(contributing_factor_list == contributing_factor_list.max())[0]
        contributing_factor_list = [idx2label[x] for x in contributing_factor_list]

        this_inst = {
            'scenario_id': data['Scenario_id'][i],
            'annotator_id': data['Annotator_id'][i],
            'scenario': parse_scenario(data['Scenario'][i]),
            'actions': data['Possible_actions'][i],
            'action_selected': sel_act,
            'gt': contributing_factor_list,
            'desc': data['Annotator_self_description'][i],
            'moral_vector': moral_vector,
            'culture_vector': culture_vector,
            'moral_idx': [idx2moral[x] for x in moral_idx],
            'culture_idx': [idx2culture[x] for x in culture_idx]
        }
        main_data.append(this_inst)

    for inst in main_data:
        annotator = inst['annotator_id']
        annotator_data = data[data['Annotator_id'] == annotator]
        annotator_data = annotator_data[annotator_data['Scenario_id'] != inst['scenario_id']]
        try:
            annotator_data = annotator_data.sample(n=1, replace=True, random_state=42)
        except Exception:
            annotator_data = data[data['Annotator_id'] == annotator]
            annotator_data = annotator_data.sample(n=1, replace=True, random_state=42)
        annotator_data = annotator_data.reset_index(drop=True)

        fs_possible_actions = ast.literal_eval(annotator_data['Possible_actions'][0])
        fs_sel_idx = annotator_data['Selected_action'][0]

        if type(fs_possible_actions[0]) != str:
            sel_act = act2idx[fs_sel_idx] + ": " + fs_possible_actions[0][fs_sel_idx - 1]
        else:
            sel_act = act2idx[fs_sel_idx] + ": " + fs_possible_actions[fs_sel_idx - 1]

        contributing_factor = annotator_data['Contributing_factors'][0]
        contributing_factor_list = [int(x) for x in ast.literal_eval(contributing_factor)]
        contributing_factor_list = np.array(contributing_factor_list)
        contributing_factor_list = np.where(contributing_factor_list == contributing_factor_list.max())[0]
        contributing_factor_list = ", ".join([idx2label[x] for x in contributing_factor_list])

        inst['fs_data'] = {
            'fs_scenario': parse_scenario(annotator_data['Scenario'][0]),
            'fs_possible_actions': annotator_data['Possible_actions'][0],
            'fs_selected_action': sel_act,
            'fs_contributing_factor': contributing_factor_list
        }

    return main_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Meta-Llama-3-8B', type=str, help='Model name for pipeline')
    parser.add_argument('--language', default='English', type=str,
                         help='Language out of ["English", "Chinese", "Russian", "Arabic", "Spanish", and "Hindi"]')
    parser.add_argument('--mode', default='desc', type=str,
                         help='Mode out of ["desc", "moral", "culture", "desc_moral", "desc_culture", "moral_culture", "desc_moral_culture", "fs"]')
    parser.add_argument('--batch_size', default=4, type=int, help='Batch size used for generation')
    parser.add_argument('--checkpoint_every', default=20, type=int,
                         help='[수정 추가] 몇 배치마다 중간 결과를 저장할지. 런타임 중단 대비.')
    parser.add_argument('--debug_only', action='store_true', help='프롬프트 샘플 출력 후 즉시 종료 (모델 로딩 없음)')
    args = parser.parse_args()

    RQ = 3
    mode = args.mode
    language = args.language
    model_name = args.model
    batch_size = args.batch_size
    checkpoint_every = args.checkpoint_every
    debug_only = args.debug_only

    # [수정] Final_results 디렉터리 미생성 문제 — 실행 시작 시점에 미리 생성
    os.makedirs("Final_results", exist_ok=True)

    with open("PROMPTS3.txt", "r") as f:
        PROMPTS = f.read()
    PROMPTS = ast.literal_eval(PROMPTS)

    if not debug_only:
        pipe = pipeline("text-generation", model=model_name, device_map="auto", truncation=True, trust_remote_code=True)

    # [수정] instruct 모델 여부에 따라 max_new_tokens, dtype 등 일관성 유지
    is_instruct = "instruct" in model_name.lower()

    data_file_rq3 = f"Final_data/{language}_long.csv"
    data = read_data_RQ3(data_file_rq3)

    formatted_prompts = []
    ground_truth = []

    for i, inst in tqdm(enumerate(data)):
        ground_truth.append(inst['gt'])

        possible_actions = ast.literal_eval(inst['actions'])
        if type(possible_actions[0]) != str:
            possible_actions = "(a) " + possible_actions[0][0] + "; (b) " + possible_actions[1][0]
        else:
            possible_actions = "(a) " + possible_actions[0] + "; (b) " + possible_actions[1]

        prompt = PROMPTS[f"prompt_{mode}_{language.lower()[:3]}_rq{RQ}"]

        prompt = prompt.replace("[SCENARIO]", inst['scenario'])
        prompt = prompt.replace("[ACTIONS]", possible_actions)

        # [수정] action_selected는 항상 문자열("a: ...")이므로 fs 분기 없이 동일하게 처리.
        # 기존 코드는 mode != 'fs'일 때 [0]으로 인덱싱해 첫 글자만 사용하는 버그가 있었음(RQ2와 동일 패턴).
        prompt = prompt.replace("[SELECTED_ACTION]", inst['action_selected'])

        prompt = prompt.replace("[DESC]", inst['desc'])
        prompt = prompt.replace("[MORAL]", " ".join(map(str, inst['moral_vector'])))
        prompt = prompt.replace("[CULTURE]", " ".join(map(str, inst['culture_vector'])))

        prompt = prompt.replace("[MORAL_VALUE_1]", inst['moral_idx'][0])
        prompt = prompt.replace("[MORAL_VALUE_2]", inst['moral_idx'][1])
        prompt = prompt.replace("[MORAL_VALUE_3]", inst['moral_idx'][2])
        prompt = prompt.replace("[MORAL_VALUE_4]", inst['moral_idx'][3])
        prompt = prompt.replace("[MORAL_VALUE_5]", inst['moral_idx'][4])
        prompt = prompt.replace("[MORAL_VALUE_6]", inst['moral_idx'][5])

        prompt = prompt.replace("[CULTURAL_VALUE_1]", inst['culture_idx'][0])
        prompt = prompt.replace("[CULTURAL_VALUE_2]", inst['culture_idx'][1])
        prompt = prompt.replace("[CULTURAL_VALUE_3]", inst['culture_idx'][2])
        prompt = prompt.replace("[CULTURAL_VALUE_4]", inst['culture_idx'][3])
        prompt = prompt.replace("[CULTURAL_VALUE_5]", inst['culture_idx'][4])
        prompt = prompt.replace("[CULTURAL_VALUE_6]", inst['culture_idx'][5])

        prompt = prompt.replace("[FS_SCENARIO]", inst['fs_data']['fs_scenario'])
        prompt = prompt.replace("[FS_ACTION]", inst['fs_data']['fs_selected_action'])
        prompt = prompt.replace("[FS_CONTRIBUTING_FACTOR]", inst['fs_data']['fs_contributing_factor'])

        # [수정] RQ1/RQ2/RQ4와 동일하게, instruct 여부에 따라 프롬프트 포맷 분기.
        # 기존 RQ3는 무조건 ### Instruction 템플릿만 사용해 base 모델과의 일관성이 깨져 있었음.
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

    # [수정 추가] 체크포인트 저장 로직 — 런타임 중단 시 처음부터 재실행하지 않도록
    checkpoint_path = f"Final_results/_checkpoint_{language}_RQ{RQ}_{mode}_{model_name.split('/')[-1]}.json"

    for i, batch_start in enumerate(tqdm(range(0, len(formatted_prompts), batch_size), desc="Generating batches")):
        batch = formatted_prompts[batch_start:batch_start + batch_size]
        outputs = pipe(batch, max_new_tokens=2000)
        generated_outputs.extend(outputs)

        if checkpoint_every > 0 and (i + 1) % checkpoint_every == 0:
            with open(checkpoint_path, "w") as f:
                json.dump({"num_done": len(generated_outputs)}, f)

    unsure_preds = 0
    predictions = []
    for output in generated_outputs:
        if is_instruct:
            generated_text = output[0]['generated_text']
            generated_text = generated_text.split("### Response:")[1].strip()
        else:
            generated_text = output[0]['generated_text'][1]['content']

        try:
            generated_text = generated_text.split("Selected action is")[1].strip()
        except Exception:
            print("generated_text: ", generated_text)

        response = generated_text.lower().strip()
        if 'emotion' in response:
            predictions.append(label2idx['Emotions'])
        elif 'moral' in response:
            predictions.append(label2idx['Moral'])
        elif 'culture' in response:
            predictions.append(label2idx['Culture'])
        elif 'responsibilities' in response:
            predictions.append(label2idx['Responsibilities'])
        elif 'relationship' in response:
            predictions.append(label2idx['Relationships'])
        elif 'legal' in response:
            predictions.append(label2idx['Legality'])
        elif 'polite' in response:
            predictions.append(label2idx['Politeness'])
        elif 'sacred' in response:
            predictions.append(label2idx['Sacred values'])
        else:
            unsure_preds += 1
            # [수정] fallback 클래스를 'Emotions'로 고정하지 않고 None(-1)으로 표시해
            # 결과 집계 시 unsure 비율을 명확히 추적할 수 있도록 함.
            predictions.append(-1)

    ground_truth_idx = [[label2idx[x] for x in y] for y in ground_truth]

    print(f"Unsure predictions: {unsure_preds} / {len(predictions)}")
    print(ground_truth_idx, predictions)

    precision, recall, f1_score, weighted_f1 = calculate_metrics(ground_truth_idx, predictions)
    accuracy = calculate_accuracy(ground_truth_idx, predictions)

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score (per class): {f1_score}")
    print(f"Weighted F1: {weighted_f1}")

    results = {
        'predictions': predictions,
        'ground truth': ground_truth_idx,
        'unsure_preds': unsure_preds,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1 score': f1_score,
        'weighted f1': weighted_f1
    }

    with open(f"Final_results/{language}_RQ{RQ}_{mode}_{model_name.split('/')[-1]}.json", "w") as f:
        json.dump(results, f)

    # 완료 후 체크포인트 파일 정리
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
