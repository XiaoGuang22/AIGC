import sys

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image
import os
import json
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def is_valid_image(img_path):
    """
    深度检测图片是否损坏。包含文件头检测、数据流截断检测、以及极小尺寸检测。
    """
    try:
        # 第一层防御：校验文件头是否完整
        with Image.open(img_path) as img:
            img.verify()

        with Image.open(img_path) as img:
            img.load()
            # 第三层防御：拦截尺寸过小，会导致卷积下采样时张量变为 0 的图片
            if img.size[0] < 16 or img.size[1] < 16:
                return False
        return True
    except Exception:
        # 只要有一丁点异常，直接判定为损坏
        return False
# ==========================================
# 1. 模型加载 (A100 专属 bfloat16 配置)
# ==========================================
model_id = "/home/liangpeng/LYK/AIGCdetection/weights/llava-1.5-7b-hf"
print("Loading LLaVA-1.5-7B model...")
processor = AutoProcessor.from_pretrained(model_id, use_fast=False)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
).to("cuda")

# 提取纯粹客观语义的 Prompt
prompt_text = "USER: <image>\nDescribe the main subjects in this image objectively. Focus on their physical properties, such as materials, surface textures, and the overall lighting of the scene. Do not judge the image quality or origin. Keep it concise. ASSISTANT:"


base_test_dir = "/home/liangpeng/LYK/dataset/AIGCDetect/AIGCDetect_testset/test"
sdv4_base_dir = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"


out_dir = "data/test_benchmarks_LLa3"
os.makedirs(out_dir, exist_ok=True)

# 定义所有需要执行的生成任务
tasks = [
    # --- 1. 训练集任务 (Train) ---
    {
        "output_json": f"{out_dir}/dataset_sdv4_LLa.json",
        "generator": "sdv4",
        "split": "train",
        "paths": {
            0: f"{sdv4_base_dir}/train/nature",  # Real
            1: f"{sdv4_base_dir}/train/ai"  # Fake
        }
    },
    # --- 2. 各个测试集任务 (Val) ---
    {
        "output_json": f"{out_dir}/test_stable_diffusion_v_1_4_LLa.json",
        "generator": "sdv4",
        "split": "val",
        "paths": {
            0: f"{sdv4_base_dir}/val/nature",
            1: f"{sdv4_base_dir}/val/ai"
        }
    },
    {
        "output_json": f"{out_dir}/test_stable_diffusion_v_1_5_LLa.json",
        "generator": "stable_diffusion_v1_5",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/stable_diffusion_v_1_5/0_real",  # 请核对你的实际文件夹名是否为大写
            1: f"{base_test_dir}/stable_diffusion_v_1_5/1_fake"
        }
    },
    {
        "output_json": f"{out_dir}/test_Midjourney_LLa.json",
        "generator": "Midjourney",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/Midjourney/0_real",
            1: f"{base_test_dir}/Midjourney/1_fake"
        }
    },
    {
        "output_json": f"{out_dir}/test_ADM_LLa.json",
        "generator": "ADM",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/ADM/0_real",
            1: f"{base_test_dir}/ADM/1_fake"
        }
    },
    {
        "output_json": f"{out_dir}/test_Glide_LLa.json",
        "generator": "Glide",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/Glide/0_real",
            1: f"{base_test_dir}/Glide/1_fake"
        }
    },
    {
        "output_json": f"{out_dir}/test_wukong_LLa.json",
        "generator": "wukong",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/wukong/0_real",
            1: f"{base_test_dir}/wukong/1_fake"
        }
    },
    {
        "output_json": f"{out_dir}/test_VQDM_LLa.json",
        "generator": "VQDM",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/VQDM/0_real",
            1: f"{base_test_dir}/VQDM/1_fake"
        }
    },
    {
        "output_json": f"{out_dir}/test_biggan_LLa.json",
        "generator": "biggan",
        "split": "val",
        "paths": {
            0: f"{base_test_dir}/biggan/0_real",
            1: f"{base_test_dir}/biggan/1_fake"
        }
    }
]

DEFAULT_BATCH_SIZE = 16

processor.tokenizer.padding_side = "left"
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

blacklist_file = "bad_images_blacklist.txt"
blacklist = set()
if os.path.exists(blacklist_file):
    with open(blacklist_file, "r", encoding="utf-8") as f:
        blacklist = set(line.strip() for line in f)

suspect_file = "current_batch.txt"
crash_suspects = set()
if os.path.exists(suspect_file):
    with open(suspect_file, "r", encoding="utf-8") as f:
        crash_suspects = set(line.strip() for line in f)
    if crash_suspects:
        print(f" {len(crash_suspects)}损坏")

for task in tasks:
    output_file = task["output_json"]
    generator_name = task["generator"]
    split_name = task["split"]

    task_results = []
    processed_images = set()

    # 断点续传
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                task_results = json.load(f)
            for item in task_results:
                processed_images.add(item["image_path"])
        except Exception:
            pass

    for label, path in task["paths"].items():
        if not os.path.exists(path):
            continue

        all_images = sorted(
            [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpeg', '.jpg', '.bmp', '.webp'))])

        # 排除掉已成功的和已拉黑的
        pending_images = [
            img for img in all_images
            if os.path.join(path, img) not in processed_images
               and os.path.join(path, img) not in blacklist
        ]

        if not pending_images:
            continue

        label_str = "Real(0)" if label == 0 else "Fake(1)"
        print(f"\n�� Processing {generator_name} {label_str}: {len(pending_images)} images left...")

        # 使用 while 循环以便动态调整 batch_size
        pbar = tqdm(total=len(pending_images), desc=f"{generator_name} {label_str}")
        i = 0

        while i < len(pending_images):
            current_img_path = os.path.join(path, pending_images[i])

            # 【核心神技】：如果当前图片是上次崩溃的嫌疑犯，强制降级为单步排雷！
            if current_img_path in crash_suspects:
                current_batch_size = 1
            else:
                current_batch_size = DEFAULT_BATCH_SIZE

            batch_img_names = pending_images[i: i + current_batch_size]
            batch_paths = [os.path.join(path, name) for name in batch_img_names]

            # 进 GPU 前，立下“生死状”
            with open(suspect_file, "w", encoding="utf-8") as f:
                for p in batch_paths:
                    f.write(p + "\n")

            batch_raw_images = []
            valid_batch_paths = []

            # 读取图片（如果有基础的损坏直接拉黑跳过）
            for p in batch_paths:
                try:
                    batch_raw_images.append(Image.open(p).convert("RGB"))
                    valid_batch_paths.append(p)
                except Exception as e:
                    with open(blacklist_file, "a", encoding="utf-8") as bf:
                        bf.write(p + "\n")
                    blacklist.add(p)

            if not batch_raw_images:
                i += current_batch_size
                pbar.update(current_batch_size)
                continue

            batch_prompts = [prompt_text] * len(batch_raw_images)

            try:
                # GPU 前向推理
                inputs = processor(batch_prompts, batch_raw_images, return_tensors='pt', padding=True).to("cuda",
                                                                                                          torch.bfloat16)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
                captions = processor.batch_decode(outputs, skip_special_tokens=True)

                for img_path, caption in zip(valid_batch_paths, captions):
                    clean_caption = caption.split("ASSISTANT:")[-1].strip()
                    task_results.append({
                        "image_path": img_path,
                        "label": label,
                        "split": split_name,
                        "generator": generator_name,
                        "caption": clean_caption
                    })

                # 平安无事，继续前进
                i += current_batch_size
                pbar.update(current_batch_size)

                # 实时保存
                if (i // DEFAULT_BATCH_SIZE) % 10 == 0:
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(task_results, f, indent=4)

            except Exception as e:
                # �� CUDA 崩溃了！
                # 如果此时 batch_size 是 1，说明我们精准抓住了唯一的凶手！
                if current_batch_size == 1 and len(valid_batch_paths) == 1:
                    killer = valid_batch_paths[0]
                    print(f"\n☠️ 抓到真凶！导致死机的毒药图片是: {killer}")
                    with open(blacklist_file, "a", encoding="utf-8") as bf:
                        bf.write(killer + "\n")

                # 保存目前的好数据，然后自尽，呼叫 Bash 重启
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(task_results, f, indent=4)
                sys.exit(1)

        pbar.close()

    # 整个文件跑完，正常保存
    if task_results:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(task_results, f, indent=4)

# 如果全流程完美结束，清理掉空的生死状
if os.path.exists(suspect_file):
    os.remove(suspect_file)

print("\n�� 全部数据集完美生成完毕！守护进程可以休息了！")