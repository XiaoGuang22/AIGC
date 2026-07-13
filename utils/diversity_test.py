import torch
import json
import os
import random
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================= 配置 =================
INPUT_JSON = "/home/liangpeng/LYK/AIGCdetection/data/dataset_sdv4_blip.json"
IMAGE_ROOT = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 采样参数 (激发模型的多样性)
NUM_SAMPLES = 10  # 每张图生成几句话
TEMPERATURE = 1.0  # 温度：越高越狂野，1.0 是标准随机
TOP_P = 0.9  # 核采样
TOP_K = 50


def calculate_diversity(captions):
    """计算唯一句子的比例"""
    unique_caps = set(captions)
    return len(unique_caps) / len(captions)


def generate_multiple(model, processor, image, device):
    inputs = processor(images=image, return_tensors="pt").to(device)

    # 使用 sample 模式生成多条
    outputs = model.generate(
        **inputs,
        do_sample=True,  # 开启采样
        num_return_sequences=NUM_SAMPLES,  # 一次返回10条
        top_k=TOP_K,
        top_p=TOP_P,
        temperature=TEMPERATURE,
        max_length=30,
        return_dict_in_generate=True,
        output_scores=True
    )

    captions = []
    confidences = []

    # 解码文本
    decoded = processor.batch_decode(outputs.sequences, skip_special_tokens=True)

    # 计算每句话的置信度 (Sequence Score)
    # sequences_scores 是对数概率，转为概率需要 exp
    # 注意: generate 只有在 beam search 下才直接返回 sequences_scores
    # 在 sample 模式下我们需要自己估算，或者简单点，直接看文本的多样性
    # 这里我们主要关注文本内容

    for text in decoded:
        captions.append(text)

    return captions


def print_results(title, samples, model, processor):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

    for item in samples:
        img_path = os.path.join(IMAGE_ROOT, item['image_path']) if 'image_path' in item else item['path']
        print(f"\n�� Image: {os.path.basename(img_path)}")

        try:
            image = Image.open(img_path).convert('RGB')
            captions = generate_multiple(model, processor, image, DEVICE)

            # 计算多样性
            diversity_score = calculate_diversity(captions)

            # 打印生成的文本
            print(f"   [Diversity Score]: {diversity_score:.2f} (1.0表示全都不同, 0.1表示全都一样)")
            print(f"   ------------------------------------------------")
            for i, cap in enumerate(captions):
                # 简单查重：如果这句话之前出现过，标记一下
                is_duplicate = captions.count(cap) > 1
                marker = "��" if is_duplicate else "  "
                print(f"   {i + 1}. {marker} {cap}")

        except Exception as e:
            print(f"Error: {e}")


def main():
    print(">>> Loading Model for Diversity Test...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # 随机采样
    real_imgs = [x for x in data if x.get('label') == 0]
    fake_imgs = [x for x in data if x.get('label') == 1]

    sample_real = random.sample(real_imgs, min(5, len(real_imgs)))
    sample_fake = random.sample(fake_imgs, min(5, len(fake_imgs)))

    print_results("�� REAL IMAGES (预期: 描述更多样/更自然)", sample_real, model, processor)
    print_results("�� FAKE IMAGES (预期: 描述更死板 或 出现幻觉)", sample_fake, model, processor)


if __name__ == "__main__":
    main()