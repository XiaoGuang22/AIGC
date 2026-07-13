import json
import torch
import os
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

# ================= 配置区域 =================
INPUT_JSON = "data/dataset_sdv4_blip.json"
OUTPUT_JSON = "data/dataset_sdv4_inblip.json"

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 可选: "Salesforce/instructblip-vicuna-7b" 或 "Salesforce/instructblip-flan-t5-xl"
MODEL_PATH = "Salesforce/instructblip-flan-t5-xl"

# InstructBLIP 专用 Prompt
# 引导它输出客观物理属性
#PROMPT = "Briefly describe the main subjects in this image in detail. Focus on their physical properties, materials, surface textures, and lighting."
PROMPT = "Provide a comprehensive and natural description of this image. Sequentially detail the main subjects, their physical textures, the materials present, and the overall lighting. Ensure the flow is smooth and avoid repeating any information,limit your response to a maximum of 5 sentences."

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
            return image, idx
        except Exception as e:
            print(f"Error reading {img_path}: {e}")
            return Image.new('RGB', (224, 224)), idx


def generate_captions():
    print(f"�� Loading InstructBLIP model on {DEVICE}...")

    # 使用标准加载方式，无需 trust_remote_code，极度稳定
    processor = InstructBlipProcessor.from_pretrained(MODEL_PATH, use_fast = False)
    model = InstructBlipForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print(f"�� Reading input json: {INPUT_JSON}")
    with open(INPUT_JSON, 'r') as f:
        raw_data = json.load(f)

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

    remaining_data = raw_data[start_index:]
    if not remaining_data:
        print("✅ All data already processed!")
        return

    dataset = CaptionDataset(remaining_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=lambda x: x)

    print(f"�� Start captioning {len(remaining_data)} images...")

    for batch in tqdm(dataloader):
        images = [item[0] for item in batch]
        indices = [item[1] for item in batch]

        # 构造 Batch Prompt
        prompts = [PROMPT] * len(images)

        # 预处理：InstructBLIP 的接口非常干净，直接传入 images 和 text
        inputs = processor(
            images=images,
            text=prompts,
            return_tensors="pt"
        ).to(DEVICE)

        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=False,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3
            )

        # 解码
        captions = processor.batch_decode(outputs, skip_special_tokens=True)

        # 填回数据
        for i, caption in enumerate(captions):
            relative_idx = indices[i]
            item = remaining_data[relative_idx]
            # InstructBLIP 直接输出答案，不需要做复杂的 split 截断
            item['caption'] = caption.strip()
            processed_data.append(item)

        # 自动保存
        if len(processed_data) % (BATCH_SIZE * 10) == 0:
            with open(OUTPUT_JSON, 'w') as f:
                json.dump(processed_data, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_JSON, 'w') as f:
        json.dump(processed_data, f, indent=4, ensure_ascii=False)

    print("=" * 40)
    print(f"✅ Done! Captions saved to {OUTPUT_JSON}")
    print(f"Sample: {processed_data[-1]['caption']}")


if __name__ == "__main__":
    generate_captions()