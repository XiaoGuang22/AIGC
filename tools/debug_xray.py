import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 初始化设置 (本地绝对路径)
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "weights/blip-large" # 确保这里是你的本地路径

print(f"Loading BLIP Model from LOCAL: {MODEL_ID} on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)
model.eval()

# ==========================================
# 2. 核心探测：自由采样 + 纯手工计算概率 (绝不报错)
# ==========================================
def inspect_image_captions_robust(image_path, num_return=10):
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  [Warning] Failed to open image: {image_path} - {e}")
        return []

    inputs = processor(raw_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # 自由发散模式
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,                  # 允许自由发挥
            top_p=0.9,                       # 核采样
            num_return_sequences=num_return, # 10 条
            output_scores=True,              # 必须开启，保留底层 logits
            return_dict_in_generate=True
        )

    generated_ids = outputs.sequences
    scores_tuple = outputs.scores # 这是一个元组，包含了每一步生成的 logits

    results = []

    # 遍历这 10 句话
    for seq_idx in range(num_return):
        seq_tokens = generated_ids[seq_idx][1:] # 跳过第一个 [BOS] 符

        token_probs = []
        words = []

        # 遍历这句话里的每一个词
        for step, token_id in enumerate(seq_tokens):
            # 遇到结束符或填充符，当前句子解析结束
            if token_id == processor.tokenizer.pad_token_id or token_id == processor.tokenizer.eos_token_id:
                break

            # 安全检查：防止句子提早结束导致越界
            if step >= len(scores_tuple):
                break

            # 【核心手工计算】提取当前步、当前句子的 logits
            step_logits = scores_tuple[step][seq_idx]

            # 手动执行 Softmax，转化为真实的 0~1 概率
            step_probs = F.softmax(step_logits, dim=-1)
            token_prob = step_probs[token_id].item()

            word = processor.tokenizer.decode([token_id]).strip()
            # 过滤掉空的 token（比如标点符号前后的空格残留）
            if word:
                words.append(word)
                token_probs.append(round(token_prob, 4))

        # 完整句子
        full_text = processor.tokenizer.decode(seq_tokens, skip_special_tokens=True).strip()

        # 计算统计量
        avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0.0
        min_prob = min(token_probs) if token_probs else 0.0

        # 拼接成直观格式：dog(0.99) | running(0.45)
        word_prob_pairs = [f"{w}({p:.2f})" for w, p in zip(words, token_probs)]

        results.append({
            "Full_Text": full_text,
            "Min_Prob": round(min_prob, 4),
            "Avg_Prob": round(avg_prob, 4),
            "Word_Prob_Details": " | ".join(word_prob_pairs)
        })

    return results

# ==========================================
# 3. 遍历测试集并导出
# ==========================================
def run_observation():
    json_dir = "../data/test_benchmarks_blip"

    if not os.path.exists(json_dir):
        print(f"Error: Directory {json_dir} not found!")
        return

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    all_records = []
    
    print(f"Found {len(json_files)} benchmark JSON files. Starting ROBUST extraction...")

    for json_file in json_files:
        json_path = os.path.join(json_dir, json_file)

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        generator_name = data[0].get("generator", json_file.replace("test_", "").replace("_blip.json", ""))
        print(f"\n>>> Processing Generator: {generator_name}")

        real_images = [item for item in data if item["label"] == 0][:10]
        fake_images = [item for item in data if item["label"] == 1][:10]

        tasks = [("real", item) for item in real_images] + [("fake", item) for item in fake_images]

        for label_name, item in tqdm(tasks, desc=f"Inferring {generator_name}"):
            img_path = item["image_path"]

            captions_data = inspect_image_captions_robust(img_path, num_return=10)

            for idx, cap in enumerate(captions_data):
                all_records.append({
                    "Generator": generator_name,
                    "Label": label_name,
                    "Image_Path": img_path,
                    "Caption_ID": idx + 1,
                    "Full_Text": cap["Full_Text"],
                    "Min_Prob": cap["Min_Prob"],
                    "Avg_Prob": cap["Avg_Prob"],
                    "Word_Prob_Details": cap["Word_Prob_Details"]
                })

    df = pd.DataFrame(all_records)
    save_path = "../blip_final_observation.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 终极跑通版提取完成！结果已保存至 {save_path}。准备迎接震撼的数据吧！")

if __name__ == "__main__":
    run_observation()