import json
import random
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import BlipProcessor, BlipForConditionalGeneration
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


# --- 1. 处理图像并提取最高置信度 ---
def process_and_evaluate_images(json_path, num_samples_per_class=100):
    print(f"Loading image data from: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"Total images found: {len(data)}")

    # 分离真假图像数据
    real_images_meta = [item for item in data if item['label'] == 0]
    fake_images_meta = [item for item in data if item['label'] == 1]

    # 随机抽样
    print(f"Sampling {num_samples_per_class} images from each class randomly...")
    sampled_real = random.sample(real_images_meta, min(len(real_images_meta), num_samples_per_class))
    sampled_fake = random.sample(fake_images_meta, min(len(fake_images_meta), num_samples_per_class))

    # 初始化 BLIP 模型
    model_id = "Salesforce/blip-image-captioning-base"
    print(f"Loading online model: {model_id}...")
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Model loaded and moved to {device}.")

    max_confidences_real = []
    max_confidences_fake = []

    def evaluate_class(samples, is_real):
        print(f"Processing {'real' if is_real else 'fake'} images...")
        for item in tqdm(samples):
            image_path = item['image_path']
            try:
                raw_image = Image.open(image_path).convert('RGB')
                inputs = processor(images=raw_image, return_tensors="pt").to(device)

                # 生成文本并输出每一步的原始得分 (logits)
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_length=50
                    )

                # 手动计算概率，完美避开 compute_transition_scores 的 Bug
                step_max_probs = []
                for step_logits in outputs.scores:
                    # step_logits shape: (batch_size, vocab_size)
                    # 将 logits 转换为概率分布 (Softmax)
                    step_probs = F.softmax(step_logits, dim=-1)
                    # 取出当前生成步中概率最大的那个词的置信度
                    max_prob = torch.max(step_probs[0]).item()
                    step_max_probs.append(max_prob)

                # 提取整句话中最高的 Token 置信度
                highest_token_prob = max(step_max_probs) if step_max_probs else 0.0

                if is_real:
                    max_confidences_real.append(highest_token_prob)
                else:
                    max_confidences_fake.append(highest_token_prob)

            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}. Skipping.")
            except Exception as e:
                print(f"Warning: Error processing {image_path}: {e}. Skipping.")

    evaluate_class(sampled_real, is_real=True)
    evaluate_class(sampled_fake, is_real=False)

    return max_confidences_real, max_confidences_fake


# --- 2. 绘制符合学术规范的分组柱状图 ---
def plot_confidence_distribution(max_conf_real, max_conf_fake, threshold=0.5):
    print("Preparing statistical plot...")

    # 设置区间 0.0-0.1, 0.1-0.2 ... 0.9-1.0
    bins = np.linspace(0, 1.0, 11)
    bin_labels = [f"{bins[i]:.1f}-{bins[i + 1]:.1f}" for i in range(len(bins) - 1)]

    # 统计每个区间内的数据量
    digitized_real = np.digitize(max_conf_real, bins, right=False)
    digitized_fake = np.digitize(max_conf_fake, bins, right=False)

    count_real = [(digitized_real == i).sum() for i in range(1, len(bins))]
    count_fake = [(digitized_fake == i).sum() for i in range(1, len(bins))]

    x = np.arange(len(bin_labels))
    width = 0.35

    # 设置学术字体和样式
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 1.2

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # 绘制分组柱状图
    rects1 = ax.bar(x - width / 2, count_real, width, label='Real Images',
                    color='#003366', alpha=0.85, edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width / 2, count_fake, width, label='AI Generated (Fake)',
                    color='#990000', alpha=0.85, edgecolor='black', linewidth=1)

    # 绘制阈值虚线 (0.5 位于索引 4.5 处)
    ax.axvline(x=4.5, color='#d62728', linestyle='--', linewidth=2.5, label=f'Truncation Threshold ({threshold})')

    # 【新增核心修改】：强制纵坐标范围为 0-100
    ax.set_ylim(0, 80)

    # 添加美化标签
    ax.set_xlabel('Maximum Token Confidence Interval', fontsize=14, fontweight='bold')
    ax.set_ylabel(f'Number of Images (Total N={len(max_conf_real)} per class)', fontsize=14, fontweight='bold')
    ax.set_title('Distribution of Maximum Token Confidence for Real and Fake Images', fontsize=16, fontweight='bold',
                 pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=12)
    ax.tick_params(axis='y', labelsize=12, direction='in')

    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

    # 使用 tight_layout 自动计算边缘，防止标签溢出被裁切
    plt.tight_layout()

    # 加入 bbox_inches='tight' 强制包裹画布所有内容
    plt.savefig('MJ.png', bbox_inches='tight')
    print("Plot saved as 'real_data_confidence_histogram.png'. Showing plot...")
    plt.show()


# --- 主程序 ---
if __name__ == "__main__":
    # JSON 路径
    your_json_file_path = "/home/liangpeng/LYK/AIGCdetection/data/test_benchmarks_blip/test_Midjourney_blip.json"

    # 1. 抽取图片并计算真实置信度
    real_max_probs, fake_max_probs = process_and_evaluate_images(your_json_file_path, num_samples_per_class=100)

    print(f"Processed valid real images: {len(real_max_probs)}")
    print(f"Processed valid fake images: {len(fake_max_probs)}")

    # 2. 画图
    plot_confidence_distribution(real_max_probs, fake_max_probs, threshold=0.5)