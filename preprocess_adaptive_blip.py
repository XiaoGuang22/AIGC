import torch
import json
import os
import random
import sys
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration

# ================= ⚙️ 运行模式 =================
# "test" = 5真5假测试
# "run"  = 全量运行
RUN_MODE = "run"

# ================= 配置 =================
INPUT_JSON = "data/dataset_sdv4_blip.json"
OUTPUT_JSON = "data/dataset_sdv4_detailed_final.json"  # 改个名，区分一下
IMAGE_ROOT = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#  强力描述参数
MIN_LENGTH = 12  # 强迫多说点
NUM_BEAMS = 3  # 稍微慢点，但质量高，能生成更连贯的长句
REPETITION_PENALTY = 1.2  # 防止车轱辘话


def generate_detailed_caption(model, processor, image, device):
    #  Prompt Engineering: 引导模型关注细节
    text_prompt = "a detailed photo of"

    inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)

    # 使用 Beam Search 生成高质量长文本
    outputs = model.generate(
        **inputs,
        min_length=MIN_LENGTH,
        max_length=40,  # 允许生成更长的句子
        num_beams=NUM_BEAMS,  # 集束搜索，寻找最优解
        repetition_penalty=REPETITION_PENALTY,  # 惩罚重复
        early_stopping=True
    )

    caption = processor.decode(outputs[0], skip_special_tokens=True)
    return caption


# ================= 测试模式 =================
def run_test_mode(model, processor, data):
    print("\n" + "=" * 60)
    print(" [TEST MODE] 捧杀策略测试：强制生成细节")
    print("=" * 60)

    real_imgs = [x for x in data if x.get('label') == 0]
    fake_imgs = [x for x in data if x.get('label') == 1]

    sample_real = random.sample(real_imgs, min(5, len(real_imgs)))
    sample_fake = random.sample(fake_imgs, min(5, len(fake_imgs)))

    print("\n 真实图像 (Real) - 预期：描述准确，与图像纹理一致")
    print("-" * 50)
    for item in sample_real:
        img_path = os.path.join(IMAGE_ROOT, item['image_path']) if 'image_path' in item else item['path']
        try:
            image = Image.open(img_path).convert('RGB')
            caption = generate_detailed_caption(model, processor, image, DEVICE)
            print(f"Path: .../{os.path.basename(img_path)}")
            print(f"Text: \033[92m{caption}\033[0m")
        except Exception as e:
            print(e)

    print("\n 虚假图像 (Fake) - 预期：描述虽然详细，但包含幻觉 (Conflict Source)")
    print("-" * 50)
    for item in sample_fake:
        img_path = os.path.join(IMAGE_ROOT, item['image_path']) if 'image_path' in item else item['path']
        try:
            image = Image.open(img_path).convert('RGB')
            caption = generate_detailed_caption(model, processor, image, DEVICE)
            print(f"Path: .../{os.path.basename(img_path)}")
            print(f"Text: \033[91m{caption}\033[0m")
        except Exception as e:
            print(e)


# ================= 全量模式 =================
def run_full_generation(model, processor, data):
    print(f"\n [FULL RUN] 正在为 {len(data)} 张图片生成详尽描述...")
    new_data = []

    for item in tqdm(data):
        img_path = os.path.join(IMAGE_ROOT, item['image_path']) if 'image_path' in item else item['path']
        try:
            image = Image.open(img_path).convert('RGB')
            caption = generate_detailed_caption(model, processor, image, DEVICE)
            item['caption'] = caption
            new_data.append(item)
        except:
            continue

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(new_data, f, indent=4)
    print("✅ 完成！")


# ================= 主程序 =================
if __name__ == "__main__":
    print(f">>> Mode: {RUN_MODE}")
    print(">>> Loading BLIP...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    with open(INPUT_JSON, 'r') as f:
        data_source = json.load(f)

    if RUN_MODE == "test":
        run_test_mode(model, processor, data_source)
    else:
        run_full_generation(model, processor, data_source)