import os
import json
import asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from huggingface_hub import HfFolder

HF_TOKEN = "your_hf_token_here"
HfFolder.save_token(HF_TOKEN)

BASE_URL = "https://router.huggingface.co/v1"
MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"
TEST_DATA_FILE = "dataset_elevator_test.json"
RESULT_FILE = "inference_result_llama3.xlsx"
MAX_CONCURRENCY = 1

def load_test_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ 文件不存在: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📉 已加载测试集: {len(data)} 条数据")
    return data

async def run_inference(data_list):
    client = AsyncOpenAI(
        base_url=BASE_URL,
        api_key=HF_TOKEN
    )

    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def process_entry(entry):
        async with sem:
            instruction = entry.get('instruction', '')
            ground_truth = entry.get('output', '')

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in elevator maintenance. Please analyze the fault based on the user's description and provide detailed troubleshooting steps. \nIMPORTANT: Please answer strictly in Chinese (中文)."
                },
                {
                    "role": "user",
                    "content": instruction
                }
            ]

            try:
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.6,
                    timeout=60.0
                )

                output_content = response.choices[0].message.content

                return {
                    "instruction": instruction,
                    "ground_truth": ground_truth,
                    "generated_text": output_content
                }
            except Exception as e:
                err_msg = str(e)
                if "loading" in err_msg.lower():
                    print(f"⏳ 模型正在加载中，跳过本条...")
                else:
                    print(f"⚠️ 请求失败: {err_msg}")

                return {
                    "instruction": instruction,
                    "ground_truth": ground_truth,
                    "generated_text": "Error"
                }

    tasks = [process_entry(item) for item in data_list]

    print(f"🚀 开始使用 {MODEL_NAME} 进行推理 (并发数: {MAX_CONCURRENCY})...")
    print("💡 提示：如果是免费 Token，速度可能会受限，请耐心等待。")

    results = await tqdm_asyncio.gather(*tasks)
    return results

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    if "hf_" not in HF_TOKEN:
        print("❌ 错误: 请先设置正确的 Hugging Face Access Token！")
        exit()

    test_data = load_test_data(TEST_DATA_FILE)

    inference_results = asyncio.run(run_inference(test_data))

    df = pd.DataFrame(inference_results)
    df.to_excel(RESULT_FILE, index=False)

    print(f"\n✅ 推理完成！结果已保存至 {RESULT_FILE}")
    print("📋 下一步：运行 evaluation_report_cot.py 进行打分。")