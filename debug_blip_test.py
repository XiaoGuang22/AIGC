import torch
import json
import os
import random
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch.nn.functional as F

# ================= 配置区 =================
# 指向你的原始 JSON 和图片根目录
INPUT_JSON = "data/dataset_sdv4_blip.json"
IMAGE_ROOT = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 策略参数 (与主脚本保持一致)
CONFIDENCE_THRESHOLD = 0.70
PATIENCE = 2
EXPAND_THRESHOLD = 0.6
EXPAND_PATIENCE = 1
HANGING_WORDS = {"a", "an", "the", "in", "on", "at", "with", "by", "of", "and", "is", "are", "to", "for", "which",
                 "that"}


# ================= 核心算法 (复用 v5_fix 版本) =================

def clean_trailing_words(text):
    words = text.split()
    while words and words[-1].lower() in HANGING_WORDS:
        words.pop()
    return " ".join(words)


def has_repetition(text):
    words = text.split()
    if len(words) == 0: return False
    # 检查相邻重复
    for i in range(len(words) - 1):
        if words[i] == words[i + 1]:
            return True
    # 检查整体重复率
    unique_words = set(words)
    if len(words) > 4 and len(unique_words) < len(words) * 0.6:
        return True
    return False


def generate_step(model, processor, image, text_prompt, device, threshold, patience, max_new_tokens=20):
    if text_prompt:
        inputs = processor(images=image, text=text_prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(images=image, return_tensors="pt").to(device)

    image_embeds = model.vision_model(inputs.pixel_values)[0]
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)
    input_ids = inputs.input_ids.to(device)

    generated_ids = []
    low_conf_streak = 0
    last_token_id = -1

    model.eval()
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model.text_decoder(
                input_ids=input_ids,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
            )
            next_token_logits = outputs.logits[:, -1, :]

            # 重复惩罚
            if len(generated_ids) > 0:
                for past_id in set(generated_ids):
                    next_token_logits[0, past_id] /= 1.2

            probs = F.softmax(next_token_logits, dim=-1)
            max_prob, next_token = torch.max(probs, dim=-1)
            token_id = next_token.item()

            # 强制阻断
            if token_id == last_token_id: break
            if token_id in [100, 101, 102, 103]: break

            # 自适应截断
            if max_prob.item() < threshold:
                low_conf_streak += 1
            else:
                low_conf_streak = 0

            if low_conf_streak >= patience: break
            if token_id == model.text_decoder.config.eos_token_id: break

            generated_ids.append(token_id)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            last_token_id = token_id

    return processor.decode(generated_ids, skip_special_tokens=True)


def adaptive_generate_pipeline(model, processor, image, device):
    # Phase 1: Anchor
    base_prompt = "a photo of"
    base_suffix = generate_step(model, processor, image, base_prompt, device,
                                threshold=CONFIDENCE_THRESHOLD, patience=PATIENCE)
    base_caption = clean_trailing_words(f"{base_prompt} {base_suffix}")

    if has_repetition(base_caption): return "a photo"
    if len(base_caption.split()) < 4: return base_caption

    # Phase 2: Expand
    expand_prompt = base_caption + " which is"
    detail_suffix = generate_step(model, processor, image, expand_prompt, device,
                                  threshold=EXPAND_THRESHOLD, patience=EXPAND_PATIENCE, max_new_tokens=15)

    if has_repetition(detail_suffix): return base_caption

    full_caption = clean_trailing_words(f"{base_caption} which is {detail_suffix}")

    if len(detail_suffix.strip()) < 2: return base_caption

    return full_caption


# ================= �� 快速测试控制器 =================

def run_debug_test():
    print(">>> 正在加载模型 (只需加载一次)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)

    print(f">>> 读取数据集索引: {INPUT_JSON}")
    if not os.path.exists(INPUT_JSON):
        print("❌ 找不到 JSON 文件")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # 分离真假图
    real_imgs = [x for x in data if x['label'] == 0]
    fake_imgs = [x for x in data if x['label'] == 1]

    # 随机采样 (各取 5 张)
    sample_real = random.sample(real_imgs, min(5, len(real_imgs)))
    sample_fake = random.sample(fake_imgs, min(5, len(fake_imgs)))

    print("\n" + "=" * 60)
    print("�� 真实图像测试 (Real Images)")
    print("=" * 60)
    for item in sample_real:
        img_path = os.path.join(IMAGE_ROOT, item['image_path']) if 'image_path' in item else item['path']
        try:
            image = Image.open(img_path).convert('RGB')
            caption = adaptive_generate_pipeline(model, processor, image, DEVICE)
            print(f"Path: .../{os.path.basename(img_path)}")
            print(f"�� Result: \033[92m{caption}\033[0m")  # 绿色字体
            print("-" * 30)
        except Exception as e:
            print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("�� 虚假图像测试 (Fake/AI Images)")
    print("=" * 60)
    for item in sample_fake:
        img_path = os.path.join(IMAGE_ROOT, item['image_path']) if 'image_path' in item else item['path']
        try:
            image = Image.open(img_path).convert('RGB')
            caption = adaptive_generate_pipeline(model, processor, image, DEVICE)
            print(f"Path: .../{os.path.basename(img_path)}")
            print(f"�� Result: \033[91m{caption}\033[0m")  # 红色字体
            print("-" * 30)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_debug_test()