import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

# 请替换为你的实际导入路径
from utils.dataset import GenImageDataset
from networks.model_v4_UFD import CrossModalDetector


# ==========================================
# 模块 1: 特征提取底层函数
# ==========================================
def extract_features(model, dataloader, target_label, color_id, desc, max_samples, device='cuda'):
    """按类别标签提取特征，凑够 max_samples 立即停止"""
    model.eval()
    features_list = []
    colors_list = []
    collected_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images, input_ids, attention_mask, labels = batch

            # 筛选符合目标标签的样本
            target_indices = (labels == target_label).nonzero(as_tuple=True)[0]
            if len(target_indices) == 0:
                continue

            # 核心优化：计算还差多少个凑够
            need = max_samples - collected_count
            if need <= 0:
                break

            if len(target_indices) > need:
                target_indices = target_indices[:need]

            # 仅提取截断后的目标样本
            images = images[target_indices].to(device)
            input_ids = input_ids[target_indices].to(device)
            attention_mask = attention_mask[target_indices].to(device)

            # 模型前向传播提取特征
            features, _ = model(images, input_ids, attention_mask, return_features=True)

            features_list.append(features.cpu().numpy())
            colors_list.extend([color_id] * features.size(0))

            collected_count += len(target_indices)

            # 提取完更新一下进度条信息，并检查是否可以下班
            if collected_count >= max_samples:
                print(f"[*] {desc} 已凑齐 {max_samples} 个样本！")
                break

    if len(features_list) == 0:
        return np.array([]), np.array([])
    return np.concatenate(features_list, axis=0), np.array(colors_list)


# ==========================================
# 模块 2: 模型与数据加载器构建
# ==========================================
def build_model(vit_path, bert_path, weight_path, device):
    """初始化模型并加载权重"""
    print(f"[*] 初始化模型并加载权重: {weight_path}")
    model = CrossModalDetector(vit_path=vit_path, bert_path=bert_path, text_encoder_type='clip')
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


def build_dataloader(json_path, tokenizer_path, batch_size):
    """构建标准的数据加载器"""
    print(f"[*] 加载数据集: {json_path}")
    dataset = GenImageDataset(json_path, tokenizer_path=tokenizer_path, tokenizer_type='clip', split='val')
    # 必须为 True 保证均匀采样
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


# ==========================================
# 模块 3: 高阶业务流 - 聚合多域特征
# ==========================================
def collect_all_features(model, loader_source, loader_target, max_per_class, device):
    """核心业务逻辑：真图全合并，假图分域提取"""

    half_max = max_per_class // 2

    # 1. 提取所有真实图像 (Label=0 -> Color=0 绿色)
    feat_r1, col_r1 = extract_features(model, loader_source, target_label=0, color_id=0, desc="提取真图 (源域)",
                                       max_samples=half_max, device=device)
    feat_r2, col_r2 = extract_features(model, loader_target, target_label=0, color_id=0, desc="提取真图 (未见域)",
                                       max_samples=max_per_class - half_max, device=device)

    # 过滤空数组并拼接
    r_feats = [f for f in [feat_r1, feat_r2] if f.size > 0]
    r_cols = [c for c in [col_r1, col_r2] if c.size > 0]
    feat_real = np.concatenate(r_feats, axis=0) if r_feats else np.array([])
    color_real = np.concatenate(r_cols, axis=0) if r_cols else np.array([])
    print(f" -> ✅ 共收集到 {len(feat_real)} 个真图特征\n")

    # 2. 提取源域假图 (Label=1 -> Color=1 蓝色)
    feat_sd_fake, color_sd_fake = extract_features(model, loader_source, target_label=1, color_id=1,
                                                   desc="提取假图 (源域)", max_samples=max_per_class, device=device)
    print(f" -> ✅ 共收集到 {len(feat_sd_fake)} 个源域假图特征\n")

    # 3. 提取未见域假图 (Label=1 -> Color=2 橙色)
    feat_mj_fake, color_mj_fake = extract_features(model, loader_target, target_label=1, color_id=2,
                                                   desc="提取假图 (未见域)", max_samples=max_per_class, device=device)
    print(f" -> ✅ 共收集到 {len(feat_mj_fake)} 个未见域假图特征\n")

    # 合并返回
    all_features = np.concatenate([feat_real, feat_sd_fake, feat_mj_fake], axis=0)
    all_colors = np.concatenate([color_real, color_sd_fake, color_mj_fake], axis=0)
    return all_features, all_colors


# ==========================================
# 模块 4: 降维与采样
# ==========================================
def perform_tsne(features, colors):
    """直接进行 t-SNE 降维"""
    print(f"\n[*] 开始对 {len(features)} 个特征点进行 t-SNE 降维 (可能需要几分钟)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca')
    reduced_features = tsne.fit_transform(features)
    return reduced_features, colors


# ==========================================
# 模块 5: 绘图渲染
# ==========================================
def plot_tsne(reduced_features, colors, legend_labels, output_path):
    """渲染高精度学术图表"""
    print(f"[*] 正在绘制图表并保存至 {output_path}...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors_hex = ['#2ca02c', '#1f77b4', '#ff7f0e']  # 绿(Real), 蓝(Source), 橙(Target)
    markers = ['o', 's', '^']

    for i in range(3):
        idx = (colors == i)
        if np.any(idx):
            ax.scatter(reduced_features[idx, 0], reduced_features[idx, 1],
                       c=colors_hex[i], label=legend_labels[i],
                       marker=markers[i], s=35, alpha=0.7, edgecolors='w', linewidth=0.5)

    ax.set_title('t-SNE Visualization of Cross-domain Joint Features', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.9, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ 图表已成功生成！")


# ==========================================
# �� 主程序与配置区 (以后你只需要改这里！)
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- 核心参数配置 ----------------
    VIT_PATH = "weights/clip-vit-large-patch14"
    BERT_PATH = "weights/bert-base-uncased"
    WEIGHT_PATH = "checkpoints/test28-2/best_generalization_model.pth"

    # �� 只要改这里，就能测试不同域的泛化效果！
    JSON_SOURCE = "data/test_benchmarks_blip/test_stable_diffusion_v_1_4_blip.json"
    JSON_TARGET = "data/test_benchmarks_blip/test_Glide_blip.json"  # <--- 改这里测试其他生成器

    # �� 对应图例名称
    NAME_SOURCE = "Source Domain (SD v1.4)"
    NAME_TARGET = "Unseen Domain (Glide)"  # <--- 图例名称跟着改

    OUTPUT_FILE = "fig/tsne_SD_vs_Glide.png"  # <--- 建议输出文件名也跟着改

    BATCH_SIZE = 64
    MAX_PER_CLASS = 1000  # 每个类别提取数量
    # ----------------------------------------------

    # 1. 准备模型与数据
    model = build_model(VIT_PATH, BERT_PATH, WEIGHT_PATH, device)
    loader_source = build_dataloader(JSON_SOURCE, VIT_PATH, BATCH_SIZE)
    loader_target = build_dataloader(JSON_TARGET, VIT_PATH, BATCH_SIZE)

    # 2. 提取特征
    features, colors = collect_all_features(model, loader_source, loader_target, MAX_PER_CLASS, device)

    # 3. 降维
    reduced_features, reduced_colors = perform_tsne(features, colors)

    # 4. 画图
    legend_labels = ['Real Nature Images', NAME_SOURCE, NAME_TARGET]
    plot_tsne(reduced_features, reduced_colors, legend_labels, OUTPUT_FILE)


if __name__ == '__main__':
    main()