import json
import torch
import os
from PIL import Image, ImageFile
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ================= 配置区域 =================
# 输入文件 (你之前的 JSON)
INPUT_JSON = "data/dataset_sdv4.json"
# 输出文件 (生成带有 BLIP 描述的新 JSON)
OUTPUT_JSON = "data/dataset_sdv4_blip.json"

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
            print(f"Error reading {img_path}: {e}")
            # 返回一张黑图占位，后面会过滤掉
            return Image.new('RGB', (224, 224)), idx


def generate_captions():
    print(f"�� Loading BLIP model on {DEVICE}...")
    # 加载模型
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(DEVICE)
    model.eval()

    # 读取原始数据
    print(f"�� Reading input json: {INPUT_JSON}")
    with open(INPUT_JSON, 'r') as f:
        raw_data = json.load(f)

    # 检查是否断点续传
    start_index = 0
    processed_data = []
    if os.path.exists(OUTPUT_JSON):
        print(f"�� Found existing output file. Checking for resume...")
        try:
            with open(OUTPUT_JSON, 'r') as f:
                processed_data = json.load(f)
            start_index = len(processed_data)
            print(f"⏩ Resuming from index {start_index}")
        except:
            print("⚠️ Output file corrupted, starting from scratch.")

    # 准备剩余数据
    remaining_data = raw_data[start_index:]
    if not remaining_data:
        print("✅ All data already processed!")
        return

    dataset = CaptionDataset(remaining_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, collate_fn=lambda x: x)

    print(f"�� Start captioning {len(remaining_data)} images...")

    # 批量推理
    for batch in tqdm(dataloader):
        # 解包 batch (list of tuples)
        images = [item[0] for item in batch]
        indices = [item[1] for item in batch]  # 这里的 index 是相对于 remaining_data 的

        # 预处理
        inputs = processor(images=images, return_tensors="pt").to(DEVICE)

        # 生成
        with torch.no_grad():
            # max_new_tokens=30 足够描述清楚内容了，太长浪费时间
            out = model.generate(**inputs, max_new_tokens=30)

        # 解码
        captions = processor.batch_decode(out, skip_special_tokens=True)

        # 填回数据
        for i, cap in enumerate(captions):
            relative_idx = indices[i]
            item = remaining_data[relative_idx]

            # �� [核心] 把生成的真实描述填进去
            item['caption'] = cap
            processed_data.append(item)

        # 每 10 个 batch 保存一次，防止崩溃白跑
        if len(processed_data) % (BATCH_SIZE * 10) == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(processed_data, f, indent=4)

    # 最后保存
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(processed_data, f, indent=4)

    print("=" * 40)
    print(f"✅ Done! Captions saved to {OUTPUT_JSON}")
    print(f"Sample: {processed_data[-1]['caption']}")


if __name__ == "__main__":
    generate_captions()