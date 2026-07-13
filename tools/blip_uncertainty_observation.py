import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. 初始化设置
# ==========================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "weights/blip-large"

print(f"Loading BLIP Model: {MODEL_ID} on {DEVICE}...")
processor = BlipProcessor.from_pretrained(MODEL_ID)
model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(DEVICE)

if not hasattr(model.config, "vocab_size"):
    model.config.vocab_size = model.config.text_config.vocab_size

model.eval()


# ==========================================
# 2. 核心探测函数：零干预 Beam Search + 真实概率
# ==========================================
def inspect_image_captions_pure(image_path, num_return=10):
    try:
        raw_image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"  [Warning] Failed to open image: {image_path} - {e}")
        return []

    inputs = processor(raw_image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        # 【核心修改】去除所有 do_sample 和 top_p，完全依靠纯数学的 Beam Search
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            num_beams=num_return,  # 开启 Beam Search，束宽设为 10
            num_return_sequences=num_return,  # 严格返回排名前 10 的句子
            output_scores=True,
            return_dict_in_generate=True
        )

    # 提取生成的 tokens
    generated_ids = outputs.sequences

    # 【核心修改】计算 Beam Search 下每个 Token 绝对真实的生成概率
    # transition_scores 返回的是对数概率 (log probabilities)
    transition_scores = model.compute_transition_scores(
        generated_ids, outputs.scores, normalize_logits=True
    )

    results = []

    for seq_idx in range(num_return):
        seq_tokens = generated_ids[seq_idx]
        seq_scores = transition_scores[seq_idx]

        token_probs = []
        words = []

        # zip 组合 token 和 对应的 score，注意跳过开头的 BOS token
        for token_id, score in zip(seq_tokens[1:], seq_scores):
            if token_id == processor.tokenizer.pad_token_id or token_id == processor.tokenizer.eos_token_id:
                break

            # 将对数概率转换为 0-1 的常规概率
            prob = torch.exp(score).item()

            word = processor.tokenizer.decode([token_id])
            words.append(word.strip())
            token_probs.append(round(prob, 4))

        # 绝对完整地解码整句话，无截断
        full_text = processor.tokenizer.decode(seq_tokens, skip_special_tokens=True)

        # 计算统计值
        avg_prob = sum(token_probs) / len(token_probs) if token_probs else 0
        min_prob = min(token_probs) if token_probs else 0

        # 将词和概率组合
        word_prob_pairs = [f"{w}({p:.2f})" for w, p in zip(words, token_probs)]

        results.append({
            "Full_Text": full_text.strip(),
            "Min_Prob": round(min_prob, 4),
            "Avg_Prob": round(avg_prob, 4),
            "Word_Prob_Details": " | ".join(word_prob_pairs)
        })

    return results


# ==========================================
# 3. 遍历测试集并导出
# ==========================================
def run_observation():
    json_dir = "../data/test_benchmarks_blip"  # 替换为你的目录

    if not os.path.exists(json_dir):
        print(f"Error: Directory {json_dir} not found!")
        return

    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    all_records = []

    print(f"Found {len(json_files)} benchmark files. Starting PURE extraction...")

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

            captions_data = inspect_image_captions_pure(img_path, num_return=10)

            for idx, cap in enumerate(captions_data):
                all_records.append({
                    "Generator": generator_name,
                    "Label": label_name,
                    "Image_Path": img_path,
                    "Rank": idx + 1,  # 这里改为了 Rank，因为 Beam Search 是按分数从高到低排的
                    "Full_Text": cap["Full_Text"],
                    "Min_Prob": cap["Min_Prob"],
                    "Avg_Prob": cap["Avg_Prob"],
                    "Word_Prob_Details": cap["Word_Prob_Details"]
                })

    df = pd.DataFrame(all_records)
    save_path = "blip_pure_beam_observation.csv"
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ 零干预纯粹版提取完成！结果已保存至 {save_path}。")


if __name__ == "__main__":
    run_observation()