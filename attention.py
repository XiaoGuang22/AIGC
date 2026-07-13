import os
import json
import torch
import torch.nn as nn
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import textwrap
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from networks.model_v5 import CrossModalDetector
from utils.dataset import GenImageDataset


# 如果你本地有 utils.cnn_aug，请取消注释下面这行。如果没有，我用了一个简单的替代品防止报错
# from utils.cnn_aug import CNNAugmentations
class DummyCNNAug:
    def __init__(self, **kwargs): pass

    def __call__(self, img): return img


CNNAugmentations = DummyCNNAug  # 临时占位符，防止你本地没有这个文件报错

# ==========================================
# 3. 注意力图生成与保存函数
# ==========================================
def save_aigc_attention_map(model, dataset, idx, target_word=None, save_dir="./"):
    model.eval()

    image_tensor, input_ids, attention_mask, label_tensor = dataset[idx]

    item = dataset.data[idx]
    img_path = item['image_path']
    raw_caption = item['caption']
    generator = item.get('generator', 'Unknown')
    label_val = item['label']
    label_str = "Real (0)" if label_val == 0 else "Fake/AIGC (1)"

    try:
        original_img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"❌ 无法读取图像: {img_path}")
        return

    original_img = original_img.resize((224, 224), Image.Resampling.BICUBIC)
    original_img_np = np.array(original_img)

    device = next(model.parameters()).device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _, _, attn_weights = model(image_tensor, input_ids, attention_mask)

    attn_weights = attn_weights.squeeze(0).cpu().numpy()
    tokens = dataset.tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().numpy())

    token_indices = []
    if target_word:
        for i, token in enumerate(tokens):
            clean_token = token.replace('Ġ', '').replace('##', '').replace('</w>', '')
            if target_word.lower() in clean_token.lower() and attention_mask.squeeze(0)[i] == 1:
                token_indices.append(i)
        if not token_indices:
            print(f"⚠️ 索引 {idx}: 在文本中未找到词汇 '{target_word}'，将保存对所有词汇的全局注意力。")
            target_word = None  # 回退到全局

    if not target_word:
        valid_lens = attention_mask.squeeze(0).sum().item()
        token_indices = list(range(1, int(valid_lens) - 1))

    patch_attn = attn_weights[1:, token_indices].mean(axis=1)
    grid_size = int(np.sqrt(patch_attn.shape[0]))
    attention_map = patch_attn.reshape(grid_size, grid_size)

    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
    attention_map = np.uint8(255 * attention_map)
    attention_map_resized = cv2.resize(attention_map, (224, 224), interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(attention_map_resized, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img_np, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(15, 6))
    wrapped_caption = "\n".join(textwrap.wrap(raw_caption, width=80))
    plt.suptitle(f"Source: {generator} | GT: {label_str}\nCaption: {wrapped_caption[:160]}...",
                 fontsize=12, y=0.98, fontweight='bold')

    plt.subplot(1, 3, 1)
    plt.imshow(original_img_np)
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(attention_map_resized, cmap='jet')
    focus_title = f"Focus Word: '{target_word}'" if target_word else "Focus: Global Average"
    plt.title(f"Attention Map\n({focus_title})")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay Result")
    plt.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.85])

    os.makedirs(save_dir, exist_ok=True)
    word_suffix = target_word if target_word else "global"
    filename = f"attn_idx{idx}_{generator}_label{label_val}_{word_suffix}.png"
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 成功保存: {save_path}")


# ==========================================
# 4. 主程序运行逻辑
# ==========================================
if __name__ == '__main__':
    # ---------------- ⚠️ 请修改以下路径 ⚠️ ----------------
    JSON_FILE_PATH = "data/test_benchmarks_inblip2/test_stable_diffusion_v_1_4_inblip.json"  # 你的 JSON 数据集路径
    MODEL_WEIGHTS = "checkpoints/test38-3/best_generalization_model.pth"  # 你训练好的模型权重 (.pth 文件)
    SAVE_DIRECTORY = "./"  # 图片保存路径 (默认当前根目录)
    # -----------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据集
    val_dataset = GenImageDataset(
        json_file=JSON_FILE_PATH,
        tokenizer_path="weights/bert-base-uncased",  # 本地路径或 HuggingFace ID
        tokenizer_type='bert',
        split='val',  # 请确保 JSON 文件中有 split="val" 的数据
        max_len=150
    )

    # 2. 初始化模型
    model = CrossModalDetector(
        vit_path="weights/clip-vit-large-patch14",
        text_model_path="weights/bert-base-uncased",
        num_classes=2,
        freeze_backbone=True,
        text_encoder_type='bert'
    ).to(device)

    # 3. 加载训练好的权重 (极度重要，否则是随机初始化)
    if os.path.exists(MODEL_WEIGHTS):
        print(f"Loading weights from {MODEL_WEIGHTS}...")
        state_dict = torch.load(MODEL_WEIGHTS, map_location=device)
        # 如果你训练时用了 DataParallel，可能需要处理 'module.' 前缀
        if 'module.' in list(state_dict.keys())[0]:
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print("✅ 模型权重加载成功！")
    else:
        print(f"⚠️ 找不到权重文件 {MODEL_WEIGHTS}，将使用未经训练的模型直接出图。")

    # 4. 提取注意力图并保存到本地
    print("\n�� 开始生成注意力图...")

    # 示例 A: 生成索引为 0 的图像，关注 "glass" 词汇
    save_aigc_attention_map(model, val_dataset, idx=0, target_word="glass", save_dir=SAVE_DIRECTORY)

    # 示例 B: 生成索引为 1 的图像，关注 "poodle" 词汇
    save_aigc_attention_map(model, val_dataset, idx=1, target_word="poodle", save_dir=SAVE_DIRECTORY)

    # 示例 C: 批量生成前 5 张图的全局注意力 (不指定单词)
    for i in range(min(5, len(val_dataset))):
        save_aigc_attention_map(model, val_dataset, idx=i, target_word=None, save_dir=SAVE_DIRECTORY)

    print("\n�� 全部任务完成！请去对应的目录查看生成的图片。")