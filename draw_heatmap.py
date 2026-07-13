import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import random
import torch.nn.functional as F

# ==========================================
# 1. 导入你自己的模型和数据集
# ==========================================
from utils.dataset import GenImageDataset
from networks.model_v5 import CrossModalDetector


# ==========================================
# 2. 核心绘图逻辑：纯净极简版热力图渲染
# ==========================================
def draw_overlay_heatmap(image_path, V_patch, A_patch, save_path="conflict_heatmap.png", vmax=0.3):
    """
    绘制并保存特征冲突热力图 (完全保留原始特征刻度，无多余归一化)
    """
    # 1. 准备底图 (保持与模型输入一致的 CenterCrop)
    img = Image.open(image_path).convert('RGB')
    transform_crop = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)
    ])
    img_cropped = np.array(transform_crop(img))  # shape: (224, 224, 3)

    # 2. 计算纯原始的冲突距离 (Cosine Distance)
    # 不做任何额外缩放，保留原始物理意义，范围严格在 [0, 2]
    V_norm = F.normalize(V_patch, p=2, dim=-1)
    A_norm = F.normalize(A_patch, p=2, dim=-1)
    conflict_score = 1.0 - torch.sum(V_norm * A_norm, dim=-1).cpu().numpy()  # [256]

    # 3. 空间重构与上采样 (保留 Float 原始精度)
    heatmap_16x16 = conflict_score.reshape(16, 16)
    heatmap_224x224 = cv2.resize(heatmap_16x16, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 4. 绘图 (交给 Matplotlib 的 vmin/vmax 机制处理绝对色彩映射)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 第一张：原图
    axes[0].imshow(img_cropped)
    axes[0].set_title("Original Image", fontsize=14)
    axes[0].axis('off')

    # 第二张：纯净的热力分布 (用 vmax 锁定绝对红色的阈值)
    im = axes[1].imshow(heatmap_224x224, cmap='jet', vmin=0.0, vmax=vmax)
    axes[1].set_title(f"Raw Conflict (Threshold={vmax})", fontsize=14)
    axes[1].axis('off')
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # 第三张：优雅的叠加效果 (Matplotlib 直接透明覆盖，无需 OpenCV 矩阵混合)
    axes[2].imshow(img_cropped)
    axes[2].imshow(heatmap_224x224, cmap='jet', vmin=0.0, vmax=vmax, alpha=0.55)
    axes[2].set_title("Overlay Result", fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 热力图已保存至: {save_path}")


# ==========================================
# 3. 主运行脚本 (抽取图片 -> 提取特征 -> 画热力图)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 配置路径 =====
    JSON_FILE = "data/test_benchmarks_inblip2/test_stable_diffusion_v_1_4_inblip.json"
    VIT_PATH = "weights/clip-vit-large-patch14"
    BERT_PATH = "weights/bert-base-uncased"
    MODEL_WEIGHTS = "checkpoints/test38-3/best_generalization_model.pth"
    # ====================

    # 1. 初始化数据集
    dataset = GenImageDataset(
        json_file=JSON_FILE,
        tokenizer_path=BERT_PATH,
        tokenizer_type='bert',
        split='val'
    )

    # 2. 初始化模型
    model = CrossModalDetector(
        vit_path=VIT_PATH,
        text_model_path=BERT_PATH,
        text_encoder_type='bert'
    ).to(device)

    if MODEL_WEIGHTS and os.path.exists(MODEL_WEIGHTS):
        model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
        print(f"Loaded trained weights from {MODEL_WEIGHTS}")

    model.eval()

    # ==========================================================
    # 【Hook机制】：使用 PyTorch Forward Hook 截取中间空间特征
    # ==========================================================
    spatial_features = {}

    def hook_visual_proj(module, input, output):
        spatial_features['V_patch'] = output[:, 1:, :]

    def hook_cross_attn(module, input, output):
        attn_output = output[0]
        spatial_features['A_patch'] = attn_output[:, 1:, :]

    model.visual_proj.register_forward_hook(hook_visual_proj)
    model.cross_attention.register_forward_hook(hook_cross_attn)

    # ==========================================================
    # 3. 随机抽取真实图像 (Label 0) 和 虚假图像 (Label 1)
    # ==========================================================
    real_indices = [i for i in range(len(dataset)) if dataset.data[i]['label'] == 0]
    fake_indices = [i for i in range(len(dataset)) if dataset.data[i]['label'] == 1]

    # 随机抽取
    real_idx = random.choice(real_indices) if real_indices else -1
    fake_idx = random.choice(fake_indices) if fake_indices else -1

    print(f"\nRandomly selected Real Image Index: {real_idx}")
    print(f"Randomly selected Fake Image Index: {fake_idx}")

    # ==========================================================
    # 4. 执行推理并生成热力图
    # ==========================================================
    for idx, label_name in zip([real_idx, fake_idx], ["Real", "Fake"]):
        if idx == -1:
            print(f"Could not find a {label_name} image in the dataset.")
            continue

        print(f"\nProcessing {label_name} Image...")
        image_tensor, input_ids, attention_mask, label = dataset[idx]
        img_path = dataset.data[idx]['image_path']

        image_tensor = image_tensor.unsqueeze(0).to(device)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

        # 前向传播
        with torch.no_grad():
            _ = model(image_tensor, input_ids, attention_mask)

        V_patch = spatial_features['V_patch'][0]  # [256, 768]
        A_patch = spatial_features['A_patch'][0]  # [256, 768]

        # 绘制并保存热力图 (传入 vmax 控制红色的敏感度)
        save_name = f"heatmap_overlay_{label_name}.png"
        draw_overlay_heatmap(img_path, V_patch, A_patch, save_path=save_name, vmax=0.3)

    print("\n�� All visualizations completed successfully!")