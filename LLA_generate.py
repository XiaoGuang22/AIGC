import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, LlavaForConditionalGeneration

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True

ImageFile.LOAD_TRUNCATED_IMAGES = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

base_test_dir = "/home/liangpeng/LYK/dataset/AIGCDetect/AIGCDetect_testset/test"
sdv4_base_dir = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"
model_id = "/home/liangpeng/LYK/AIGCdetection/weights/llava-1.5-7b-hf"

out_dir = "data/test_benchmarks_LLa"
os.makedirs(out_dir, exist_ok=True)

BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LLAVA_PROMPT = "USER: <image>\nDescribe the main subjects in this image objectively. Focus on their physical properties, such as materials, surface textures, and the overall lighting of the scene. Do not judge the image quality or origin. Keep it concise. ASSISTANT:"

tasks = [
    # {
    #     "output_json": f"{out_dir}/dataset_sdv4_LLa.json",
    #     "generator": "sdv4",
    #     "split": "train",
    #     "paths": {
    #         0: f"{sdv4_base_dir}/train/nature",
    #         1: f"{sdv4_base_dir}/train/ai"
    #     }
    # },
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
            0: f"{base_test_dir}/stable_diffusion_v_1_5/0_real",
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

# 数据集损坏
class CaptionDataset(Dataset):
    def __init__(self, data_list: List[Dict[str, Any]]):
        self.data_list = data_list
        self.dummy_image = Image.new('RGB', (336, 336))

    def __len__(self) -> int:
        return len(self.data_list)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int, bool]:
        item = self.data_list[idx]
        img_path = item['image_path']

        try:
            image = Image.open(img_path).convert('RGB')
            w, h = image.size
            if w < 16 or h < 16 or (w / h > 10) or (h / w > 10):
                raise ValueError(f"畸形图像尺寸 {w}x{h}")
            return image, idx, True

        except Exception:
            return self.dummy_image.copy(), idx, False


def build_raw_data_from_task(task: Dict) -> List[Dict]:
    """根据单个 task 的字典配置，扫描出所有底层图像路径"""
    dataset_data = []
    split_name = task["split"]
    generator_name = task["generator"]

    for label, folder_path in task["paths"].items():
        if not os.path.exists(folder_path):
            logger.warning(f"路径不存在，已跳过: {folder_path}")
            continue

        extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.webp', '*.JPG', '*.PNG', '*.JPEG']
        images = []
        for ext in extensions:
            images.extend(list(Path(folder_path).rglob(ext)))

        for img_obj in images:
            dataset_data.append({
                "image_path": str(img_obj.resolve()).replace('\\', '/'),
                "label": label,
                "split": split_name,
                "generator": generator_name,
                "caption": ""
            })
    return dataset_data


def process_single_task(model, processor, task: Dict):
    """处理单个任务字典"""
    output_json_path = task["output_json"]
    generator_name = task["generator"]
    split_name = task["split"]

    logger.info(f"开始处理任务: {generator_name} ({split_name})")
    raw_data = build_raw_data_from_task(task)

    if not raw_data:
        logger.warning(f"未能在 {generator_name} 中找到图像，跳过该任务。")
        return

    logger.info(f"      共扫描到 {len(raw_data)} 张图像。")

    processed_data = []
    start_index = 0

    # 断点续传
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding="utf-8") as f:
                processed_data = json.load(f)
            if len(processed_data) == len(raw_data):
                logger.info(f"   该任务已全部完成，跳过。")
                return
            start_index = len(processed_data)
            logger.info(f"   发现历史进度，从 {start_index}/{len(raw_data)} 继续...")
        except Exception:
            logger.warning("   历史文件损坏，将重新生成。")
            processed_data = []

    remaining_data = raw_data[start_index:]
    if not remaining_data:
        return

    dataset = CaptionDataset(remaining_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=lambda x: x)

    pbar = tqdm(dataloader, desc=f"   生成 {generator_name}", unit="batch")

    for batch_idx, batch in enumerate(pbar, start=1):
        images = [item[0] for item in batch]
        indices = [item[1] for item in batch]
        is_valid_flags = [item[2] for item in batch]

        prompts = [LLAVA_PROMPT] * len(images)

        # GPU 推理
        inputs = processor(prompts, images, return_tensors="pt").to(DEVICE, torch.bfloat16)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)

        captions = processor.batch_decode(outputs, skip_special_tokens=True)

        # 回填数据
        for i, cap in enumerate(captions):
            item = remaining_data[indices[i]]
            if not is_valid_flags[i]:
                item['caption'] = "ERROR_SKIPPED"
            else:
                item['caption'] = cap.split("ASSISTANT:")[-1].strip()
            processed_data.append(item)

        # 高频保存
        if batch_idx % 10 == 0:
            with open(output_json_path, 'w', encoding="utf-8") as f:
                json.dump(processed_data, f, indent=4)

        torch.cuda.empty_cache()

    # 最终保存
    with open(output_json_path, 'w', encoding="utf-8") as f:
        json.dump(processed_data, f, indent=4)

    logger.info(f"   任务保存成功: {output_json_path}")
    print("-" * 60)


def main():
    logger.info(f"正在 {DEVICE} 上初始化 LLaVA 模型...")
    processor = AutoProcessor.from_pretrained(model_id, use_fast=False)

    # processor.tokenizer.padding_side = "left"
    # if processor.tokenizer.pad_token is None:
    #     processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        #attn_implementation="flash_attention_2",
    ).to(DEVICE)
    model.eval()

    for task in tasks:
        process_single_task(model, processor, task)

    logger.info("�� 所有任务执行完毕，JSON 结构与截图完全一致！")


if __name__ == "__main__":
    main()