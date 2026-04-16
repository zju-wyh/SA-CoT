import os
import json
import re
import asyncio
import random
from copy import deepcopy
from typing import List
from openai import AsyncOpenAI

from new_openai_api import NewOpenAIChat

SYSTEM_PROMPT_COT = """
你是一个电梯维修领域的资深专家数据构建员。你的任务是读取提供的【语料片段】，生成用于训练大模型的 "思维链 (Chain-of-Thought)" 问答对。

【重要原则】：
1. 你的回答必须严格基于【语料片段】的内容，不要编造。
2. 每一个问答对必须包含详细的推理过程。
3. 必须在回答中指明该知识适用的场景或来源（根据提供的来源文档名判断）。

请生成 3-5 个不同维度（故障诊断、原理分析、操作步骤）的问答对。
必须以标准的 JSON 格式输出，结构如下：
{
  "qa_pairs": [
    {
      "instruction": "用户的问题（例如：迅达300P电梯急停继电器释放是什么原因？）",
      "output": "思维链：\\n1. 来源确认：该信息源自《[Filename]》...\\n2. 现象分析：...\\n3. 原理推导：...\\n\\n结论：..."
    }
  ]
}
确保 output 字段包含详细的推理步骤，而不仅仅是答案。如果语料片段中信息不足以生成问题，请返回空列表。
"""

AUGMENT_STYLES = {
    "novice": """
你是一个数据增强专家。请将用户输入的【标准工业问题】改写为【不懂技术的物业/小白描述】。
要求：
1. 去掉具体的继电器编号（如 KJT），改用“那个吸合的东西”或“安全开关”。
2. 描述要模糊，侧重于表面现象（灯灭了、不动了、有响声）。
3. 语气可以稍微困惑。
4. 仅输出改写后的问题文本，不要其他废话。
""",

    "expert": """
你是一个数据增强专家。请将用户输入的【标准工业问题】改写为【资深维修工的现场简报】。
要求：
1. 大量使用专业术语缩写（如 KJT, 110V, 门锁回路）。
2. 语言极度简练，类似电报风格。
3. 去掉“请问”、“是什么原因”等客套话，直接描述状态。
4. 仅输出改写后的问题文本，不要其他废话。
""",

    "noise": """
你是一个数据增强专家。请将用户输入的【标准工业问题】改写为【匆忙输入的手机短信】。
要求：
1. 模拟打字错误（如把“继电器”打成“继电七”）。
2. 句子不完整，省略标点。
3. 模拟语音转文字的口语风格。
4. 仅输出改写后的问题文本，不要其他废话。
"""
}

def parse_mixed_corpus(file_path):
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    pattern = r'={20,}\s+File:\s*(.*?)\s+={20,}'
    parts = re.split(pattern, raw_text, flags=re.DOTALL)

    documents = []
    for i in range(1, len(parts), 2):
        filename = parts[i].strip()
        content = parts[i + 1].strip()
        if content:
            documents.append({"source": filename, "content": content})

    print(f"📚 解析完成：共识别出 {len(documents)} 个独立来源的文档。")
    return documents

def split_text_into_chunks(text, chunk_size=1500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def process_corpus_and_generate(file_path):
    documents = parse_mixed_corpus(file_path)
    if not documents:
        print("⚠️ 未找到有效文档，请检查输入文件格式是否包含分隔符。")
        return []

    messages_list = []
    total_chunks = 0

    for doc in documents:
        chunks = split_text_into_chunks(doc['content'])
        total_chunks += len(chunks)

        for i, chunk in enumerate(chunks):
            user_msg = (
                f"【来源文档】：{doc['source']}\n"
                f"【文档片段 ({i + 1}/{len(chunks)})】：\n{chunk}\n\n"
                f"请基于上述特定文档内容生成 CoT 问答数据。"
            )
            messages_list.append([
                {"role": "system", "content": SYSTEM_PROMPT_COT},
                {"role": "user", "content": user_msg}
            ])

    print(f"📄 准备处理 {len(documents)} 个文档，共 {total_chunks} 个文本片段...")

    model = NewOpenAIChat(model_name='gpt-4o', max_tokens=4096, temperature=0.7)

    print("🚀 [Step 1] 开始请求 GPT-4o 生成基础 CoT 数据...")
    responses = model.batch_run(messages_list)

    all_data = []
    success_count = 0

    for resp in responses:
        if not resp: continue
        try:
            clean_resp = resp.replace("```json", "").replace("```", "").strip()
            data_json = json.loads(clean_resp)

            if "qa_pairs" in data_json:
                pairs = data_json["qa_pairs"]
                if pairs:
                    all_data.extend(pairs)
                    success_count += 1
        except Exception:
            continue

    print(f"✅ 基础数据生成完成！成功解析 {success_count}/{len(responses)} 个片段。")
    print(f"📊 获得种子数据: {len(all_data)} 条。")
    return all_data

def augment_dataset_with_styles(seed_data):
    if not seed_data:
        return []

    print(f"\n🌱 [Step 2] 正在基于 {len(seed_data)} 条种子数据构建表达增强请求...")

    model = NewOpenAIChat(model_name='gpt-4o', max_tokens=2048, temperature=0.8)

    messages_list = []
    tasks_metadata = []

    for entry in seed_data:
        original_instruction = entry.get('instruction', '')
        if not original_instruction: continue

        for style_name, system_prompt in AUGMENT_STYLES.items():
            messages_list.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"标准问题：{original_instruction}"}
            ])
            tasks_metadata.append({
                "original_entry": entry,
                "style": style_name
            })

    print(f"🚀 开始并发请求 GPT-4o 进行增强，共 {len(messages_list)} 个任务...")

    responses = model.batch_run(messages_list)

    augmented_entries = []
    for i, resp in enumerate(responses):
        if resp:
            new_instruction = resp.strip().strip('"').strip('“').strip('”')
            original_entry = tasks_metadata[i]['original_entry']
            new_entry = deepcopy(original_entry)
            new_entry['instruction'] = new_instruction
            augmented_entries.append(new_entry)

    print(f"✅ 增强完成！新增 {len(augmented_entries)} 条风格化数据。")

    final_dataset = seed_data + augmented_entries
    return final_dataset

def save_and_split_data(data, train_file, test_file, split_ratio=0.1):
    if not data:
        print("❌ 数据为空，无法保存。")
        return

    random.shuffle(data)

    split_index = int(len(data) * split_ratio)
    if split_index == 0 and len(data) > 0:
        split_index = 1

    test_data = data[:split_index]
    train_data = data[split_index:]

    with open(train_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"\n💾 数据保存完毕：")
    print(f"   - 训练集 ({train_file}): {len(train_data)} 条")
    print(f"   - 测试集 ({test_file}):  {len(test_data)} 条")
    print(f"   ⚠️ 请务必人工打开 {test_file} 进行核对，将其作为你的 Gold Standard 评测集！")

if __name__ == "__main__":
    INPUT_FILE = "elevator_raw_text.txt"
    TRAIN_FILE = "dataset_elevator_train.json"
    TEST_FILE = "dataset_elevator_test.json"

    seed_dataset = process_corpus_and_generate(INPUT_FILE)

    if seed_dataset:
        full_dataset = augment_dataset_with_styles(seed_dataset)
        print(f"\n📈 总数据量概览: 种子 {len(seed_dataset)} -> 最终 {len(full_dataset)}")
        save_and_split_data(full_dataset, TRAIN_FILE, TEST_FILE)
    else:
        print("❌ 未生成种子数据，终止流程。")