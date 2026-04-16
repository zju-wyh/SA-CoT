import os
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as calculate_bert_score
from transformers import AutoTokenizer
from openai import OpenAI

try:
    from vllm import LLM, SamplingParams
except ImportError:
    exit()

TEST_DATA_FILE = "dataset_elevator_test.json"
QWEN_MODEL_PATH = "your_model_path_here"
RESULT_FILE = "evaluation_report_cot.xlsx"
OPENAI_API_KEY = "your_api_key_here"
OPENAI_BASE_URL = "your_base_url_here"

def load_test_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    return df

def calculate_text_metrics(references, candidates):
    bleu_scores = []
    rouge_l_scores = []

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    smooth = SmoothingFunction().method1

    for ref, cand in zip(references, candidates):
        ref_tokens = list(ref)
        cand_tokens = list(cand)
        if len(cand_tokens) == 0: cand_tokens = [""]

        b_score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)
        bleu_scores.append(b_score)

        ref_spaced = " ".join(ref)
        cand_spaced = " ".join(cand)
        r_score = scorer.score(ref_spaced, cand_spaced)['rougeL'].fmeasure
        rouge_l_scores.append(r_score)

    return np.mean(bleu_scores), np.mean(rouge_l_scores)

def calculate_semantic_metrics(references, candidates):
    LOCAL_MODEL_PATH = "./bert_base_chinese"

    if not os.path.exists(LOCAL_MODEL_PATH):
        return 0.0

    try:
        P, R, F1 = calculate_bert_score(
            candidates,
            references,
            model_type=LOCAL_MODEL_PATH,
            num_layers=9,
            verbose=True,
            device="cuda:0"
        )
        return F1.mean().item()
    except Exception:
        return 0.0

def run_llm_judge(df_results):
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    judge_prompt_template = """
    你是一名**企业内部技术文档审核员**。你的任务是鉴别模型生成的回答是否真正基于**内部私有知识库**，还是在利用互联网通用知识进行“猜测”。

    【用户问题】：{instruction}
    【标准答案 (私有真实数据)】：{reference}
    【待测模型回答】：{candidate}

    请基于以下**内部合规性标准**进行打分 (1-5分)。

    ### 评分核心逻辑：
    *企业维护现场需要的是基于私有手册的精准指令。引用外部出版物名称（如“xxx第3版”）或罗列教科书式的“可能性列表”，均视为**未掌握私有数据**的表现，必须严厉扣分。*

    ### 评分维度：

    **1. 来源真实性鉴别 (Source Authenticity) - 权重 40% [关键项]**
    - **[符合内部规范 - 满分]**：引用了**具体的内部文件名**，通常带有文件后缀（如 **.docx**, .pdf）或具有非正式的内部命名特征（如“故障集锦”、“维修实用手册”）。
        - *判定标准：只要文件名看起来像内部流转文档，即使与标准答案不完全一致，视为正确引用。*
    - **[外部通用特征 - 零分]**：引用了**看起来像公开出版物**的名称（如“xx手册(**第3版**)”、“xx标准指南”、“xx原理大全”）。
        - *判定标准：这是典型的模型幻觉（Hallucination），表明模型没有检索到私有文件，而是编造了书名。本项直接 0 分。*
    - **[缺失]**：未引用来源，本项 0 分。

    **2. 诊断导向性 (Diagnosis Actionability) - 权重 40%**
    - **[专家级 - 高分]**：**敢于下判断**。直接指出某一个具体的故障元件（如“张紧轮偏心螺栓”），并给出单一的确切建议。
        - *判定标准：现场维修需要确定的指令。只要给出的元件属于该故障的合理工业原因（专业度高），视为合格。*
    - **[教科书式 - 低分]**：**“大撒网”式回答**。使用了“可能原因有：1... 2... 3...”这种列表，罗列了所有可能的机械和电气原因。
        - *判定标准：这种回答虽然全面但缺乏针对性，属于通用模型的典型特征。最高不超过 3 分。*

    **3. 格式纯净度 (Format Compliance) - 权重 20%**
    - **[高分]**：直接输出标准的“思维链”内容，无多余的废话。
    - **[扣分]**：包含了类似 `<|im_end|>`、`User:`、`Expert Answer:` 等对话残留标记，或者在回答前加了太多“好的，根据分析...”等客套铺垫。

    ### 综合打分执行表：
    - **5分 (完全合规)**：引用了 .docx/.pdf 格式的内部文件名，给出了**单一、具体**的硬件故障点，无废话。
    - **4分 (基本合规)**：引用格式正确，故障分析合理但略显简单。
    - **3分 (存在偏差)**：引用了“xxx手册(第x版)”这种疑似伪造的名称，或者罗列了3点以上的通用原因。**视为未掌握私有数据**。
    - **1-2分 (严重违规)**：完全错误的建议，或格式严重崩坏。

    请输出 JSON 格式：{{"score": 5, "reason": "模型引用了符合内部规范的文件名（.docx），并精准指出了'偏心螺栓'这一具体故障点，避免了泛泛而谈的罗列，符合内部专家诊断标准。"}}
    """

    scores = []
    reasons = []

    for index, row in tqdm(df_results.iterrows(), total=len(df_results)):
        prompt = judge_prompt_template.format(
            instruction=row['instruction'],
            reference=row['ground_truth'],
            candidate=row['generated_text']
        )

        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            res_json = json.loads(response.choices[0].message.content)

            score = res_json.get('score', res_json.get('logic', 0))
            reason = res_json.get('reason', '')

            scores.append(score)
            reasons.append(reason)
        except Exception:
            scores.append(0)
            reasons.append("Error")

    return scores, reasons

if __name__ == "__main__":
    df_res = pd.read_excel("inference_result_qwen.xlsx")

    refs = df_res['ground_truth'].tolist()
    cands = df_res['generated_text'].tolist()

    bleu, rouge_l = calculate_text_metrics(refs, cands)
    bert_f1 = calculate_semantic_metrics(refs, cands)

    scores, reasons = run_llm_judge(df_res)
    df_res['judge_score'] = scores
    df_res['judge_reason'] = reasons

    print("\n" + "=" * 40)
    print("📊 Evaluation Report (CoT Generation)")
    print("=" * 40)
    print(f"✅ BLEU-4:      {bleu:.4f}")
    print(f"✅ ROUGE-L:     {rouge_l:.4f}")
    print(f"✅ BERTScore:   {bert_f1:.4f}")
    print(f"✅ Judge Score: {np.mean(scores):.2f} / 5.0")
    print("=" * 40)

    df_res.to_excel(RESULT_FILE, index=False)