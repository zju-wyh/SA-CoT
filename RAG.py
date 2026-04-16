import os
import json
import pandas as pd
import torch
from typing import List
from tqdm import tqdm
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

QWEN_MODEL_PATH = "your_model_path_here"
EMBEDDING_MODEL_PATH = "./MiniLM"

TEST_DATA_FILE = "dataset_elevator_test.json"
PDF_CORPUS_FILE = "elevator_raw_text.txt"
RESULT_FILE = "inference_result_standard_rag.xlsx"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

def build_vector_store(corpus_path):
    print("🏗️ 正在构建 RAG 向量知识库...")

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"❌ 语料文件不存在: {corpus_path}")

    with open(corpus_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    texts = text_splitter.split_text(raw_text)

    docs = [Document(page_content=t, metadata={"source": "elevator_docs"}) for t in texts]
    print(f"📄 语料已切分为 {len(docs)} 个片段。")

    print(f"📥 加载 Embedding 模型: {EMBEDDING_MODEL_PATH} ...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    vector_store = FAISS.from_documents(docs, embeddings)
    print("✅ 向量库构建完成！")
    return vector_store

def load_test_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"📉 已加载测试集: {len(data)} 条数据")
    return data

def run_rag_inference(vector_store, test_data):
    print(f"🚀 正在加载生成模型: {QWEN_MODEL_PATH} ...")

    tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    llm = LLM(
        model=QWEN_MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=4096,
        dtype="bfloat16"
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id]
    )

    results = []
    prompts = []

    print("🔍 开始检索与生成...")

    retrieved_contexts = []
    for item in tqdm(test_data, desc="Retrieving"):
        query = item['instruction']

        docs = vector_store.similarity_search(query, k=TOP_K)
        context_str = "\n\n".join([f"片段 {i + 1}: {d.page_content}" for i, d in enumerate(docs)])
        retrieved_contexts.append(context_str)

        rag_prompt_content = f"""你是一个电梯维修专家。请基于以下【参考资料】回答用户的问题。如果参考资料中没有相关信息，请利用你的专业知识回答，但优先依据参考资料。

### 参考资料：
{context_str}

### 用户问题：
{query}

### 回答："""

        messages = [
            {"role": "system", "content": "你是一个基于知识库的智能助手。"},
            {"role": "user", "content": rag_prompt_content}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(text)

    print("⚡ 开始 vLLM 批量生成...")
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text.strip()
        results.append({
            "instruction": test_data[i]['instruction'],
            "ground_truth": test_data[i]['output'],
            "retrieved_context": retrieved_contexts[i],
            "generated_text": generated_text
        })

    return results

if __name__ == "__main__":
    vector_store = build_vector_store(PDF_CORPUS_FILE)

    test_data = load_test_data(TEST_DATA_FILE)

    rag_results = run_rag_inference(vector_store, test_data)

    df = pd.DataFrame(rag_results)
    df.to_excel(RESULT_FILE, index=False)

    print(f"\n✅ Standard RAG 测试完成！结果已保存至 {RESULT_FILE}")
    print("💡 提示：Excel 中包含了 'retrieved_context' 列，你可以检查检索是否准确。")
    print("📋 下一步：运行 evaluation_report_cot.py 进行打分。")