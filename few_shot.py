import os
import json
import random
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

QWEN_MODEL_PATH = "your_model_path_here"

TRAIN_DATA_FILE = "dataset_elevator_train.json"
TEST_DATA_FILE = "dataset_elevator_test.json"
RESULT_FILE = "inference_result_few_shot.xlsx"

N_SHOTS = 1

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_random_shots(train_data, n=1):
    shots = random.sample(train_data, n)
    shot_text = ""

    for i, shot in enumerate(shots):
        shot_text += f"### 示例 {i + 1}:\n"
        shot_text += f"用户问题: {shot['instruction']}\n"
        shot_text += f"专家回答:\n{shot['output']}\n\n"

    return shot_text

def run_icl_inference(train_data, test_data):
    print(f"🚀 正在加载模型: {QWEN_MODEL_PATH} ...")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=4096,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        temperature=1.8,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    results = []
    prompts = []

    print(f"🔍 正在构建 {N_SHOTS}-Shot Prompts 并进行推理...")

    fixed_shots_text = get_random_shots(train_data, N_SHOTS)

    print(f"--- Few-shot Context Preview ---\n{fixed_shots_text[:200]}...\n--------------------------------")

    for item in test_data:
        target_instruction = item['instruction']

        user_content = (
            f"请参考以下电梯维修专家的思维链分析范例，回答最后的用户问题。\n\n"
            f"{fixed_shots_text}"
            f"### 待处理任务:\n"
            f"用户问题: {target_instruction}\n"
            f"专家回答:"
        )

        messages = [
            {"role": "system", "content": "你是一名电梯维修领域的资深专家。请严格模仿示例的格式和逻辑进行回答。"},
            {"role": "user", "content": user_content}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        results.append({
            "instruction": test_data[i]['instruction'],
            "ground_truth": test_data[i]['output'],
            "generated_text": generated_text,
            "prompt_used": prompts[i]
        })

    return results

if __name__ == "__main__":
    if not os.path.exists(TRAIN_DATA_FILE):
        print(f"❌ 训练集不存在 ({TRAIN_DATA_FILE})，无法抽取示例。请先运行之前的生成脚本。")
        exit()

    train_data = load_data(TRAIN_DATA_FILE)
    test_data = load_data(TEST_DATA_FILE)

    icl_results = run_icl_inference(train_data, test_data)

    df = pd.DataFrame(icl_results)
    df.to_excel(RESULT_FILE, index=False)

    print(f"\n✅ Few-shot ({N_SHOTS}-shot) 测试完成！结果已保存至 {RESULT_FILE}")
    print("📋 下一步：运行 evaluation_report_cot.py 进行打分。")