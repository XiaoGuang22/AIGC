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
# 2. 核心大改核心逻辑：受文本指引的高清叠加图
# ==========================================
def draw_percentile_overlay_attention_map(image_path, A_patch, save_path="attention_map.png"):
    # 1. 准备底图 (保持与模型输入一致的 CenterCrop)
    img = Image.open(image_path).convert('RGB')
    transform_crop = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)
    ])
    img_cropped = np.array(transform_crop(img))  # shape: (224, 224, 3)

    # 2. 计算特征响应激活强度 (L2 Norm)
    # 这代表了网络融合语义后，在各个空间位置上的关注强烈程度
    activation = torch.norm(A_patch, p=2, dim=-1).cpu().numpy()  # [256]

    # 3. 空间重构与百分位数归一化 (核心修改，消除极端激活离群点)
    heatmap_16x16 = activation.reshape(16, 16)

    # 【调试技巧】：
    # 计算 5% 和 95% 的分位数。95是受关注上限，超过的都视为深红色。
    # 95 数值越小（比如改成90、85），红色的激活区域面积越大。
    # 95 数值越趋近于100，激活区域越尖锐、越成点状。
    p_min = np.percentile(heatmap_16x16, 5)
    p_max = np.percentile(heatmap_16x16, 95)

    # 将数值严格截断在这个百分位区间内，然后归一化
    heatmap_clipped = np.clip(heatmap_16x16, p_min, p_max)
    heatmap_norm = (heatmap_clipped - p_min) / (p_max - p_min + 1e-8)

    # 4. 上采样与伪彩色渲染
    # 放大到 224x224 (双三次插值，平滑自然)
    heatmap_224x224 = cv2.resize(heatmap_norm, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 映射到 [0, 255] 并应用 JET 色谱 (红高蓝低)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_224x224), cv2.COLORMAP_JET)
    # OpenCV 默认是 BGR，转换为 RGB 以便 matplotlib 显示叠加
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # 5. 【高清大改叠加逻辑】：使用 Matplotlib 直接透明覆盖
    # 这种做法比 OpenCV 的物理像素物理混合精度更高，更“纯净”
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # (1) 原始 Crop 图像
    axes[0].imshow(img_cropped)
    axes[0].set_title("Original Image (Cropped)", fontsize=14)
    axes[0].axis('off')

    # (2) 高清绝对叠加热力图 (Matplotlib 直接透明覆盖，alpha=0.55)
    axes[1].imshow(img_cropped)
    # 再在上面叠一层热力图，锁死 Min-Max
    axes[1].imshow(heatmap_224x224, cmap='jet', alpha=0.55, vmin=0.0, vmax=1.0)
    axes[1].set_title("Guided Attention Overlay Result", fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 高清受文本指引注意力图已保存至: {save_path}")


# ==========================================
# 3. 主运行脚本 (抽取图片 -> 提取特征 -> 画注意力图)
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 配置路径 =====
    JSON_FILE = "data/test_benchmarks_inblip2/test_Glide_inblip.json"
    VIT_PATH = "weights/clip-vit-large-patch14"
    BERT_PATH = "weights/bert-base-uncased"
    # ⚠️ 强烈建议填入训练好的权重路径，否则 V 和 A 没区别，图上全是蓝的。
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
    # 【核心技巧】：提取交叉注意力融合后的特征响应动态截取CLS
    # 这样您就不需要修改原来的 CrossModalDetector.py 代码了
    # ==========================================================
    spatial_features = {}


    def hook_cross_attn(module, input, output):
        # 截取 MultiheadAttention 的输出特征 (融合后的结果)
        # 经过了 A_ln 对齐后的 CLS 和 Patch 激活响应
        # output[0] shape: [B, 257, 768]
        # output[1] shape: [B, 257, Seq_len] (由于您的 forward 函数没有传 weights，这里可能是 None)
        fused_features = output[0]
        # 去除 CLS，提取 256 个空间 Patch 激活
        spatial_features['A_patch'] = fused_features[:, 1:, :]


    # 注册钩子拦截最后的融合响应
    model.cross_attention.register_forward_hook(hook_cross_attn)
    # ==========================================================

    # 3. 随机抽取真实图像 (Label 0) 和 虚假图像 (Label 1)
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

        # 接收原版 dataset 返回的 4 个值
        image_tensor, input_ids, attention_mask, label = dataset[idx]

        # 手动从 dataset 的原始 json 数据字典里获取对应的图片路径
        img_path = dataset.data[idx]['image_path']
        print(f"{img_path}")
        image_tensor = image_tensor.unsqueeze(0).to(device)
        input_ids = input_ids.unsqueeze(0).to(device)
        attention_mask = attention_mask.unsqueeze(0).to(device)

        raw_caption = dataset.data[idx]['caption']
        print(f"对应文本 (Caption): {raw_caption}")

        # 前向传播 (此时钩子会自动拦截特征响应并截取CLS)
        with torch.no_grad():
            _ = model(image_tensor, input_ids, attention_mask)

        # 提取受文本指引的融合响应激活特征 [256, 768]
        A_patch = spatial_features['A_patch'][0]

        # 调用核心逻辑，绘制高清絕對疊加熱力图
        save_name = f"attention_guided_high_definition_vmax_{label_name}.png"
        draw_percentile_overlay_attention_map(img_path, A_patch, save_path=save_name)

    print("\n�� All visualizations completed successfully!")