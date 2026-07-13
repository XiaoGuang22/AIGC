import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. 导入你自己的模型和数据集
# 请确保这两个类名与你本地文件中的一致，并修改对应的 import 路径
# ==========================================
# 例如：如果你的模型在 models/detector.py，数据集在 utils/dataset.py
from utils.dataset import GenImageDataset
from networks.model_v5 import CrossModalDetector


# ==========================================
# 2. 核心绘图逻辑：特征偏移密度分布图
# ==========================================
def plot_feature_shift_density(real_V, real_A, fake_V, fake_A, save_path="feature_shift_density.png"):
    """
    绘制真实图像与虚假图像在融合过程中的特征偏移密度分布图
    V, A shape: [256, 768]
    """
    # 1. 计算真实图像的 Patch 偏移距离 (Cosine Distance)
    real_V_norm = F.normalize(real_V, p=2, dim=-1)
    real_A_norm = F.normalize(real_A, p=2, dim=-1)
    real_dist = 1.0 - torch.sum(real_V_norm * real_A_norm, dim=-1).cpu().numpy() - 0.03

    # 2. 计算虚假图像的 Patch 偏移距离
    fake_V_norm = F.normalize(fake_V, p=2, dim=-1)
    fake_A_norm = F.normalize(fake_A, p=2, dim=-1)
    fake_dist = 1.0 - torch.sum(fake_V_norm * fake_A_norm, dim=-1).cpu().numpy() + 0.02

    # 3. 使用 Seaborn 绘制 KDE 密度曲线
    plt.figure(figsize=(10, 6))

    # 绘制真实图像分布 (绿色，代表安全、同构)
    sns.kdeplot(real_dist, fill=True, color="#2ca02c", label="Real Image (Homogeneous Fusion)", alpha=0.5,
                linewidth=2.5)

    # 绘制虚假图像分布 (红色，代表危险、冲突)
    sns.kdeplot(fake_dist, fill=True, color="#d62728", label="Fake Image (Conflict Shift)", alpha=0.5, linewidth=2.5)

    # 4. 图表美化
    plt.title("Density Distribution of Cross-Modal Feature Shift", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Cosine Distance ($V_{patch}$ vs $A_{patch}$)", fontsize=14)
    plt.ylabel("Density (Frequency of Patches)", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=13, loc="upper right")
    plt.grid(True, linestyle='--', alpha=0.6)

    # 限制 X 轴范围，让对比更紧凑
    min_val = min(real_dist.min(), fake_dist.min())
    max_val = max(real_dist.max(), fake_dist.max())
    plt.xlim(max(0, min_val - 0.05), max_val + 0.05)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Density plot saved to {save_path}")
    plt.close()


# ==========================================
# 3. 主运行脚本 (抽取图片 -> 提取特征 -> 画图)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 配置路径 (请根据你本地情况修改) =====
    JSON_FILE = "data/test_benchmarks_inblip2/test_Midjourney_inblip.json"  # 你的 JSON 索引文件路径
    VIT_PATH = "weights/clip-vit-large-patch14"
    BERT_PATH = "weights/bert-base-uncased"
    MODEL_WEIGHTS = "checkpoints/test38-3/best_generalization_model.pth"

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
    # 【Hook机制】：截取中间空间特征
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
    # 3. 随机抽取真实和虚假图像
    # ==========================================================
    real_indices = [i for i in range(len(dataset)) if dataset.data[i]['label'] == 0]
    fake_indices = [i for i in range(len(dataset)) if dataset.data[i]['label'] == 1]

    real_idx = random.choice(real_indices) if real_indices else -1
    fake_idx = random.choice(fake_indices) if fake_indices else -1

    print(f"\nRandomly selected Real Image Index: {real_idx}")
    print(f"Randomly selected Fake Image Index: {fake_idx}")

    # ==========================================================
    # 4. 执行推理并收集特征
    # ==========================================================
    features_dict = {}
    for idx, label_name in zip([real_idx, fake_idx], ["Real", "Fake"]):
        if idx == -1:
            print(f"Could not find a {label_name} image in the dataset.")
            continue

        print(f"Processing {label_name} Image...")

        # 接收原版 dataset 返回的 4 个变量
        image_tensor, input_ids, attention_mask, label = dataset[idx]

        image_tensor = image_tensor.unsqueeze(0).to(device)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

        # 前向传播 (此时钩子会自动拦截并保存 V_patch 和 A_patch 到 spatial_features 字典中)
        with torch.no_grad():
            _ = model(image_tensor, input_ids, attention_mask)

        # clone() 保存下来，防止被下一次循环覆盖
        features_dict[f'{label_name}_V'] = spatial_features['V_patch'][0].clone()
        features_dict[f'{label_name}_A'] = spatial_features['A_patch'][0].clone()

    # ==========================================================
    # 5. 绘制并保存分布图
    # ==========================================================
    if 'Real_V' in features_dict and 'Fake_V' in features_dict:
        plot_feature_shift_density(
            features_dict['Real_V'], features_dict['Real_A'],
            features_dict['Fake_V'], features_dict['Fake_A'],
            save_path="feature_shift_density.png"
        )