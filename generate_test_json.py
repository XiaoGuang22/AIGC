import json
import torch
import os
import glob
from pathlib import Path
from PIL import Image, ImageFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置区域 =================
# 测试集根目录
TEST_ROOT = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4"

OUTPUT_DIR = "data/dataset_sdv4.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 显卡配置
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

LABEL_MAP = {
    "0_real": 0,
    "1_fake": 1
}


# ==============================================

class CaptionDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']
        try:
            image = Image.open(img_path).convert('RGB')
            return image, idx  # 返回 idx 以便回填数据
        except Exception as e:
            # print(f"Error reading {img_path}: {e}")
            # 返回一张黑图占位，后面会过滤掉
            return Image.new('RGB', (224, 224)), idx


def get_image_list_from_folder(gen_path, gen_name):
    """
    扫描文件夹，构建待处理的图片列表
    支持：root/0_real 和 root/type/0_real 两种结构
    """
    dataset_data = []

    # 使用 os.walk 递归查找所有子文件夹
    for root, dirs, files in os.walk(gen_path):
        for dir_name in dirs:
            # 只要文件夹名字是 0_real 或 1_fake (忽略大小写)
            lower_name = dir_name.lower()
            if lower_name in LABEL_MAP:
                label = LABEL_MAP[lower_name]
                folder_path = Path(os.path.join(root, dir_name))

                # 递归查找该文件夹下的所有图片
                images = []
                extensions = ['*.jpg', '*.png', '*.jpeg', '*.bmp', '*.JPG', '*.PNG', '*.JPEG']
                for ext in extensions:
                    images.extend(list(folder_path.rglob(ext)))

                for img_obj in images:
                    img_path = str(img_obj).replace('\\', '/')
                    dataset_data.append({
                        "image_path": img_path,
                        "label": label,
                        "split": "val",  # 统一标记为 test
                        "generator": gen_name,
                        "caption": ""  # 占位，等待 BLIP 填充
                    })
    return dataset_data


def process_single_generator(model, processor, gen_name, raw_data):
    """
    处理单个生成器的数据：生成 Caption 并保存
    """
    output_json_path = os.path.join(OUTPUT_DIR, f"test_{gen_name}_blip.json")

    # --- 断点续传逻辑 ---
    start_index = 0
    processed_data = []

    if os.path.exists(output_json_path):
        print(f"Found existing file: {output_json_path}")
        try:
            with open(output_json_path, 'r') as f:
                processed_data = json.load(f)
            # 如果已处理的数量等于总数，直接跳过
            if len(processed_data) == len(raw_data):
                print(f"   ✅ Already completed. Skipping.")
                return

            # 否则从断点继续
            start_index = len(processed_data)
            print(f"   ⏩ Resuming from index {start_index}/{len(raw_data)}")
        except:
            print("   ⚠️ File corrupted, restarting.")
            processed_data = []

    # 准备剩余数据
    remaining_data = raw_data[start_index:]
    if not remaining_data:
        return

    # 构建 DataLoader
    dataset = CaptionDataset(remaining_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=lambda x: x)

    # 进度条
    pbar = tqdm(dataloader, desc=f"   Processing {gen_name}", unit="batch")

    for batch in pbar:
        images = [item[0] for item in batch]
        indices = [item[1] for item in batch]

        # 1. 预处理
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)

        # 2. 生成
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)

        # 3. 解码
        captions = processor.batch_decode(out, skip_special_tokens=True)

        # 4. 回填
        for i, cap in enumerate(captions):
            relative_idx = indices[i]
            item = remaining_data[relative_idx]
            item['caption'] = cap
            processed_data.append(item)

        # 每 5 个 Batch 保存一次，保证安全
        if len(processed_data) % (BATCH_SIZE * 5) == 0:
            with open(output_json_path, 'w') as f:
                json.dump(processed_data, f, indent=4)

    # 最后保存完整文件
    with open(output_json_path, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print(f"   �� Saved to {output_json_path}")


def main_pipeline():
    # 1. 加载模型 (只加载一次，节省时间)
    print(f"�� Loading BLIP model on {DEVICE}...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    model.eval()

    # 2. 扫描 TEST_ROOT 下的所有生成器文件夹
    if not os.path.exists(TEST_ROOT):
        print(f"❌ Error: Path not found {TEST_ROOT}")
        return

    sub_folders = sorted([d for d in os.listdir(TEST_ROOT) if os.path.isdir(os.path.join(TEST_ROOT, d))])
    print(f"�� Found {len(sub_folders)} generators: {sub_folders}\n")

    # 3. 循环处理每个生成器
    for gen_name in sub_folders:
        gen_path = os.path.join(TEST_ROOT, gen_name)
        print(f"[1/2] Scanning images for: {gen_name} ...")

        # 扫描并构建初始列表
        raw_data = get_image_list_from_folder(gen_path, gen_name)

        if len(raw_data) == 0:
            print(f"  No images found in {gen_name} (checked 0_real/1_fake). Skipping.")
            continue

        print(f"   Found {len(raw_data)} images.")

        print(f"▶️ [2/2] Generating Captions for: {gen_name} ...")
        # 执行 BLIP 生成
        process_single_generator(model, processor, gen_name, raw_data)
        print("-" * 50)

    print("\n�� All benchmarks processed!")


if __name__ == "__main__":
    main_pipeline()