import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# 请根据你的实际导入路径修改
from utils.dataset import GenImageDataset
from networks.model_v5_final import CrossModalDetector


# ==========================================
# 模块 1: 距离提取底层函数 (加入安全拦截机制)
# ==========================================
def extract_distances(model, dataloader, target_label, desc, max_samples=1000, device='cuda'):
    """
    按类别标签提取 V_ln 和 A_ln 之间的余弦距离。
    Cosine Distance = 1 - Cosine Similarity
    """
    model.eval()
    distances_list = []
    collected_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images, input_ids, attention_mask, labels = batch

            # 筛选符合目标标签的样本 (0=Real, 1=Fake)
            target_indices = (labels == target_label).nonzero(as_tuple=True)[0]
            if len(target_indices) == 0:
                continue

            need = max_samples - collected_count
            if need <= 0:
                break

            if len(target_indices) > need:
                target_indices = target_indices[:need]

            # 送入设备
            images = images[target_indices].to(device)
            input_ids = input_ids[target_indices].to(device)
            attention_mask = attention_mask[target_indices].to(device)

            # max_id = input_ids.max().item()
            # if max_id >= 30522:
            #     print(f"\n[FATAL ERROR] 拦截到越界词元！当前最大 ID 为 {max_id}")
            #     print(f"BERT 词表最大容量为 30521。请彻底检查 Dataset 中是否仍在错误使用 CLIP Tokenizer！")
            #     exit(1)

            # 前向传播，拿到模型返回的 V_ln 和 A_ln
            _, V_ln, A_ln = model(images, input_ids, attention_mask)

            # 计算余弦相似度与距离
            cosine_sim = F.cosine_similarity(V_ln, A_ln, dim=-1)
            cosine_dist = 1.0 - cosine_sim

            # diff_vector = torch.abs(V_ln - A_ln)  # [B, 768]
            #
            # K = 64
            # topk_diff, _ = torch.topk(diff_vector, k=K, dim=-1)  # [B, 64]
            #
            # conflict_energy = torch.mean(topk_diff, dim=-1)  # [B]

            distances_list.append(cosine_dist.cpu().numpy())
            collected_count += len(target_indices)

            if collected_count >= max_samples:
                print(f"[*] {desc} 已凑齐 {max_samples} 个样本距离！")
                break

    if len(distances_list) == 0:
        return np.array([])
    return np.concatenate(distances_list, axis=0)


# ==========================================
# 模块 2: 模型与数据加载器构建
# ==========================================
def build_model(vit_path, text_path, weight_path, device):
    print(f"[*] 初始化模型并加载权重: {weight_path}")
    model = CrossModalDetector(
        vit_path=vit_path,
        text_model_path=text_path,
        text_encoder_type='bert',
        freeze_backbone=True
    )
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model


def build_dataloader(json_path, tokenizer_path, batch_size):
    """构建标准的数据加载器 (必须 shuffle=True 保证随机抽样)"""
    print(f"[*] 加载数据集: {json_path}")
    # 强制设置 tokenizer_type 为 bert
    dataset = GenImageDataset(json_path, tokenizer_path=tokenizer_path, tokenizer_type='bert', split='val')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


# ==========================================
# 模块 3: 高清学术级绘图渲染 (KDE)
# ==========================================
def plot_distance_kde(dist_real, dist_source_fake, dist_unseen_fake, output_path):
    """使用 Seaborn 绘制高精度核密度估计分布图"""
    print(f"\n[*] 正在绘制 KDE 特征偏移分布图...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 配色方案：绿(Real), 蓝(Source Fake), 橙(Unseen Fake)
    colors = ['#2ca02c', '#1f77b4', '#ff7f0e']

    sns.kdeplot(dist_real, fill=True, color=colors[0], alpha=0.4, linewidth=2.5, label='Real Nature Images', ax=ax)
    sns.kdeplot(dist_source_fake, fill=True, color=colors[1], alpha=0.4, linewidth=2.5, label='Source Fake (SD v1.4)',
                ax=ax)
    sns.kdeplot(dist_unseen_fake, fill=True, color=colors[2], alpha=0.4, linewidth=2.5,
                label='Unseen Fake (Midjourney)', ax=ax)

    # 图表修饰
    ax.set_title('Distribution of Cross-Modal Feature Shift Distance', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(r'Cosine Distance ($Dis = 1 - Cos(V_{ln}, A_{ln})$)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)

    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='gray')
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ 图表已成功保存至: {output_path}")


# ==========================================
# 主程序配置区
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- 核心参数配置 ----------------
    VIT_PATH = "weights/clip-vit-large-patch14"
    TEXT_PATH = "weights/bert-base-uncased"

    WEIGHT_PATH = "checkpoints/test38-3/best_generalization_model.pth"

    JSON_SOURCE = "data/test_benchmarks_inblip2/test_stable_diffusion_v_1_4_inblip.json"
    JSON_TARGET = "data/test_benchmarks_inblip2/test_Midjourney_inblip.json"

    OUTPUT_FILE = "feature_shift_distance_KDE_MJ.png"

    BATCH_SIZE = 64
    MAX_PER_CLASS = 1000  # 每类抽取 1000 个样本计算距离
    # ----------------------------------------------

    # 1. 准备模型与数据
    model = build_model(VIT_PATH, TEXT_PATH, WEIGHT_PATH, device)

    # �� 关键修复：这里的数据集 tokenizer 路径必须传入 TEXT_PATH ！！！
    loader_source = build_dataloader(JSON_SOURCE, TEXT_PATH, BATCH_SIZE)
    loader_target = build_dataloader(JSON_TARGET, TEXT_PATH, BATCH_SIZE)

    # 2. 提取特征偏移距离
    print("\n开始进行前向传播并计算特征偏移距离...")

    half_max = MAX_PER_CLASS // 2
    dist_r1 = extract_distances(model, loader_source, target_label=0, desc="提取真图 (源域)", max_samples=half_max,
                                device=device)
    dist_r2 = extract_distances(model, loader_target, target_label=0, desc="提取真图 (未见域)", max_samples=half_max,
                                device=device)

    # 防止因为样本不足导致 concatenate 报错
    if len(dist_r1) > 0 and len(dist_r2) > 0:
        dist_real = np.concatenate([dist_r1, dist_r2])
    elif len(dist_r1) > 0:
        dist_real = dist_r1
    else:
        dist_real = dist_r2

    dist_source_fake = extract_distances(model, loader_source, target_label=1, desc="提取源域假图 (SDv1.4)",
                                         max_samples=MAX_PER_CLASS, device=device)
    dist_unseen_fake = extract_distances(model, loader_target, target_label=1, desc="提取未见域假图 (Midjourney)",
                                         max_samples=MAX_PER_CLASS, device=device)

    # 3. 绘制 KDE 分布图
    plot_distance_kde(dist_real, dist_source_fake, dist_unseen_fake, OUTPUT_FILE)


if __name__ == '__main__':
    main()