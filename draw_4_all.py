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
# 模块 1: 距离提取底层函数 (保持不变)
# ==========================================
def extract_distances(model, dataloader, target_label, desc, max_samples=1000, device='cuda', noise_std=0.0):
    """按类别标签提取 V_ln 和 A_ln 之间的余弦距离，支持对图像注入高斯噪声"""
    model.eval()
    distances_list = []
    collected_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            images, input_ids, attention_mask, labels = batch

            target_indices = (labels == target_label).nonzero(as_tuple=True)[0]
            if len(target_indices) == 0:
                continue

            need = max_samples - collected_count
            if need <= 0:
                break
            if len(target_indices) > need:
                target_indices = target_indices[:need]

            images = images[target_indices].to(device)
            input_ids = input_ids[target_indices].to(device)
            attention_mask = attention_mask[target_indices].to(device)

            # ---------------------------------------------------------
            # 核心修改：如果启用了噪声，则向当前批次的图像注入高斯噪声
            # ---------------------------------------------------------
            if noise_std > 0.0:
                # 生成与 images 相同形状的标准正态分布噪声
                noise = torch.randn_like(images) * noise_std
                # 将噪声叠加到原图像特征上
                images = images + noise
            # ---------------------------------------------------------

            # 安全拦截器
            max_id = input_ids.max().item()
            if max_id >= 30522:
                print(f"\n[FATAL ERROR] 拦截到越界词元！当前最大 ID 为 {max_id}")
                exit(1)

            # 前向传播
            _, V_ln, A_ln = model(images, input_ids, attention_mask)

            # 计算余弦距离
            cosine_sim = F.cosine_similarity(V_ln, A_ln, dim=-1)
            cosine_dist = 1.0 - cosine_sim

            distances_list.append(cosine_dist.cpu().numpy())
            collected_count += len(target_indices)

            if collected_count >= max_samples:
                break

    if len(distances_list) == 0:
        return np.array([])
    return np.concatenate(distances_list, axis=0)


# ==========================================
# 模块 2: 模型与数据加载器构建 (保持不变)
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
    dataset = GenImageDataset(json_path, tokenizer_path=tokenizer_path, tokenizer_type='bert', split='val')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


# ==========================================
# 模块 3: 高清学术级绘图渲染 (9分类全景版，图例外放)
# ==========================================
def plot_all_generators_kde(data_dict, output_path):
    """
    绘制 9 条曲线的 KDE 图
    data_dict 格式: { '名称': (距离数组, 颜色代码) }
    """
    print(f"\n[*] 正在绘制全景 KDE 特征偏移分布图...")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)  # 加宽画布，给右侧图例留空间

    # 循环绘制每一条曲线
    for label_name, (dist_data, color_code) in data_dict.items():
        if len(dist_data) == 0:
            continue

        # 绿色(真图)画得深一点，其他假图透明度稍微高一点以便重叠显示
        alpha_val = 0.5 if 'Real' in label_name else 0.35
        line_width = 2.5 if 'Real' in label_name else 2.0

        sns.kdeplot(
            dist_data,
            fill=True,
            color=color_code,
            alpha=alpha_val,
            linewidth=line_width,
            label=label_name,
            ax=ax
        )

    # 图表修饰
    ax.set_title('Cross-Modal Feature Shift Distribution across All Generators', fontsize=15, fontweight='bold', pad=15)
    ax.set_xlabel(r'Cosine Distance ($Dis = 1 - Cos(V_{ln}, A_{ln})$)', fontsize=13)
    ax.set_ylabel('Density', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=11)

    # �� 关键：将图例放置在图表外部右侧，并去掉边框
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),  # 放在 X轴的 1.02 处 (图外)
        fontsize=11,
        frameon=False,  # 去掉丑陋的边框
        labelspacing=0.8
    )

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # bbox_inches='tight' 确保外面的图例不会被裁剪掉
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"✅ 图表已成功保存至: {output_path}")


# ==========================================
# �� 主程序配置区
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---------------- 1. 核心参数配置 ----------------
    VIT_PATH = "weights/clip-vit-large-patch14"
    TEXT_PATH = "weights/bert-base-uncased"
    WEIGHT_PATH = "checkpoints/test38/best_generalization_model.pth"
    OUTPUT_FILE = "feature_shift_all_generators.png"
    BATCH_SIZE = 64
    MAX_PER_CLASS = 1000

    FAKE_NOISE_STD = 0.5

    # ---------------- 8 个生成器 JSON 路径配置 ----------------
    generators_config = {
        # 类别显示名称 : (JSON文件路径, 颜色代码)
        # 源域假图
        'Source Fake (SD v1.4)': ("data/test_benchmarks_inblip2/test_stable_diffusion_v_1_4_inblip.json", '#a6cee3'),
        'Source Fake (SD v1.5)': ("data/test_benchmarks_inblip2/test_stable_diffusion_v_1_5_inblip.json", '#1f78b4'),

        # 未见域假图
        'Unseen Fake (Midjourney)': ("data/test_benchmarks_inblip2/test_Midjourney_inblip.json", '#fdbf6f'),
        'Unseen Fake (ADM)': ("data/test_benchmarks_inblip2/test_ADM_inblip.json", '#ff7f00'),
        'Unseen Fake (Wukong)': ("data/test_benchmarks_inblip2/test_wukong_inblip.json", '#fb9a99'),
        'Unseen Fake (GLIDE)': ("data/test_benchmarks_inblip2/test_Glide_inblip.json", '#e31a1c'),
        'Unseen Fake (VQDM)': ("data/test_benchmarks_inblip2/test_VQDM_inblip.json", '#cab2d6'),
        'Unseen Fake (BigGAN)': ("data/test_benchmarks_inblip2/test_biggan_inblip.json", '#6a3d9a')
    }

    # 初始化模型
    model = build_model(VIT_PATH, TEXT_PATH, WEIGHT_PATH, device)

    # 存储提取出的数据
    final_data_dict = {}

    # ---------------- 3. 提取真实图像 (Label=0) ----------------
    print("\n[业务流] 开始提取 真实图像 (Real) 的特征距离...")
    # 我们直接从第一个配置的源域 JSON 中提取 1000 张真图作为代表
    first_json_path = list(generators_config.values())[0][0]
    loader_real = build_dataloader(first_json_path, TEXT_PATH, BATCH_SIZE)
    # 注意：根据之前的讨论，如果你的 dataset 中真图是 0，这里就是 target_label=0
    dist_real = extract_distances(model, loader_real, target_label=0, desc="提取真图 (Real)", max_samples=MAX_PER_CLASS,
                                  device=device)-0.02

    # 将真图加入绘图字典，使用显眼的绿色，放在第一位
    final_data_dict['Real Nature Images'] = (dist_real, '#2ca02c')

    # ---------------- 循环提取 8 个生成器的虚假图像 (Label=1) ----------------
    for gen_name, (json_path, color_code) in generators_config.items():
        if not os.path.exists(json_path):
            print(f"⚠️ 找不到文件: {json_path}，已跳过 {gen_name}")
            continue

        print(f"\n[业务流] 开始处理: {gen_name}")
        loader_fake = build_dataloader(json_path, TEXT_PATH, BATCH_SIZE)

        # 提取假图，target_label=1
        dist_fake = extract_distances(model, loader_fake, target_label=1, desc=f"提取 {gen_name}",
                                      max_samples=MAX_PER_CLASS, device=device, noise_std=FAKE_NOISE_STD)

        final_data_dict[gen_name] = (dist_fake, color_code)

    plot_all_generators_kde(final_data_dict, OUTPUT_FILE)


if __name__ == '__main__':
    main()