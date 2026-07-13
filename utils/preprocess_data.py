import re
import os
import json
import glob
import random

# ================= 自动路径配置 =================
# 1. 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 获取项目根目录 (往上一级)
project_root = os.path.dirname(current_dir)

# 3. 拼接数据集路径 (你的服务器绝对路径)
DATA_ROOT = "/home/liangpeng/LYK/dataset/GenImage/stable_diffusion_v_1_4/"

# 4. 输出文件放在项目根目录下的 data 文件夹
OUTPUT_JSON = os.path.join(project_root, "data/dataset_sdv4_blip.json")  # 建议改个名区分模型

print(f"Dataset Root: {DATA_ROOT}")
print(f"Output JSON:  {OUTPUT_JSON}")

# 固定随机种子，确保实验可复现！
random.seed(42)
# ===============================================

# 定义类别映射
CLASS_MAP = {
    "nature": 0,  # 真图
    "ai": 1  # 假图
}

# 文本描述模板
TRAIN_TEMPLATES = {
    "nature": [
        "A real photograph capturing natural scenes.",
        "A high-quality real-world image with consistent textures.",
        "A natural image without synthetic artifacts."
    ],
    "ai": [
        "a synthetic image containing artificial artifacts and digital noise.",
        "an AI-generated image with unnatural structural patterns.",
        "a fake image showing visual anomalies and synthesis glitches.",
        "an artificial image created by deep generative models."
    ]
}

VAL_CAPTION = "a photo don't know nature or ai"

def generate_dataset_json(root_dir, output_file):
    dataset = []

    if not os.path.exists(root_dir):
        print(f"❌ 错误: 找不到数据集路径: {root_dir}")
        return

    # 获取所有生成器子文件夹 (例如: imagenet_ai_0424_sdv5)
    sub_datasets = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if not sub_datasets:
        print(f"警告: 在 {root_dir} 下没找到任何子文件夹！请检查路径结构。")
        return

    print(f"发现 {len(sub_datasets)} 个子数据集: {sub_datasets}")

    # 详细统计
    stats = {"train": 0, "val": 0, "total": 0}

    # 遍历每个生成器文件夹
    for generator_name in sub_datasets:
        generator_path = os.path.join(root_dir, generator_name)

        # 简化生成器名字
        simple_gen_name = re.sub(r'(imagenet_|ai_|\d+_)', '', generator_name)

        # 遍历 train 和 val
        for split in ['train', 'val']:
            split_path = os.path.join(generator_path, split)
            if not os.path.exists(split_path): continue

            # 遍历 ai 和 nature
            for class_name, label in CLASS_MAP.items():
                class_path = os.path.join(split_path, class_name)
                if not os.path.exists(class_path): continue

                # 获取所有图片 (覆盖常见后缀)
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPEG', '*.PNG', '*.JPG']
                images = []
                for ext in image_extensions:
                    images.extend(glob.glob(os.path.join(class_path, ext)))

                for img_path in images:
                    # 统一转为 Linux 风格的正斜杠路径
                    img_path = img_path.replace('\\', '/')

                    # 生成描述
                    if split == 'train':
                        # 训练集：从模板中随机抽取，告诉模型这图是真是假，引导它关注特征
                        caption = random.choice(TRAIN_TEMPLATES[class_name])
                    else:
                        # 验证集：使用统一的中性文本，模拟真实场景，测试模型是否真的学会了看图
                        caption = VAL_CAPTION

                    dataset.append({
                        "image_path": img_path,
                        "label": label,
                        "split": split,
                        "generator": simple_gen_name,
                        "caption": caption
                    })

                    # 更新统计
                    stats[split] += 1
                    stats["total"] += 1

        print(f"  - 已处理子集: {generator_name}")

    # 自动创建输出目录，防止报错
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建目录: {output_dir}")

    # 保存 JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=4)

    print("=" * 40)
    print(f"成功生成索引文件: {output_file}")
    print(f"数据统计:")
    print(f"   - Train set: {stats['train']} 张")
    print(f"   - Val set  : {stats['val']} 张")
    print(f"   - Total    : {stats['total']} 张")
    print("=" * 40)


if __name__ == "__main__":
    generate_dataset_json(DATA_ROOT, OUTPUT_JSON)