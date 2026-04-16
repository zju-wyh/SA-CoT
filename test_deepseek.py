import os
import json
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

DEEPSEEK_API_KEY = "your_deepseek_api_key_here"
BASE_URL = "your_base_url_here"

MODEL_NAME = "deepseek-r1"

TEST_DATA_FILE = "dataset_elevator_test.json"
RESULT_FILE = "inference_result_deepseek_r1.xlsx"

MAX_CONCURRENCY = 10

def load_test_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📉 已加载测试集: {len(data)} 条数据")
    return data

async def run_inference(data_list):
    client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=BASE_URL)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    results = []

    async def process_entry(entry):
        async with sem:
            instruction = entry.get('instruction', '')
            ground_truth = entry.get('output', '')

            messages = [
                {"role": "system", "content": "你是一名电梯维修专家。请根据用户描述进行故障分析，并给出详细的排查步骤。"},
                {"role": "user", "content": instruction}
            ]

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.6
                )

                output_content = response.choices[0].message.content

                return {
                    "instruction": instruction,
                    "ground_truth": ground_truth,
                    "generated_text": output_content
                }
            except Exception as e:
                print(f"⚠️ 请求失败: {e}")
                return {
                    "instruction": instruction,
                    "ground_truth": ground_truth,
                    "generated_text": "Error"
                }

    tasks = [process_entry(item) for item in data_list]

    results = await tqdm_asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    test_data = load_test_data(TEST_DATA_FILE)

    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print(f"🚀 开始使用 {MODEL_NAME} 进行推理...")
    inference_results = asyncio.run(run_inference(test_data))

    df = pd.DataFrame(inference_results)
    df.to_excel(RESULT_FILE, index=False)

    print(f"\n✅ 推理完成！结果已保存至 {RESULT_FILE}")
    print("接下来可以使用之前的 eval 脚本对这个 Excel 进行 BLEU/ROUGE/Judge 评测。")