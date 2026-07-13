import torch
import os
import shutil
import cv2  # 用于图片拼接和写字，如果没有请 pip install opencv-python
import numpy as np
from PIL import Image
from torchvision import transforms

# 引入你的数据集类
from utils.dataset import GenImageDataset

# ================= �� 配置区域 =================
# 请确保这些路径和 train.py 里的一致
JSON_PATH = "../data/dataset_sdv5_blip.json"
BERT_PATH = "/weights/bert-base-uncased"
OUTPUT_DIR = "debug_compare_vis"

# 归一化参数 (用于反归一化)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


# =============================================

def tensor_to_numpy(tensor):
    """把 Tensor 反归一化并转为 OpenCV 格式 (H, W, C)"""
    # 1. 反归一化: img = img * std + mean
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    img = tensor * std + mean

    # 2. 截断到 [0, 1]
    img = torch.clamp(img, 0, 1)

    # 3. 转 numpy: (C, H, W) -> (H, W, C)
    img_np = img.numpy().transpose(1, 2, 0)

    # 4. 转为 0-255 整数
    img_np = (img_np * 255).astype(np.uint8)

    # 5. RGB 转 BGR (OpenCV 默认是用 BGR)
    img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_cv2


def get_raw_image_cv2(path, size=(224, 224)):
    """读取原图并 Resize 到同样大小，不做增强"""
    try:
        img = Image.open(path).convert('RGB')
        img = img.resize(size)  # 保持和模型输入一样大
        img_np = np.array(img)
        img_cv2 = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        return img_cv2
    except Exception as e:
        print(f"Error loading raw image: {e}")
        return np.zeros((224, 224, 3), dtype=np.uint8)


def main():
    # 1. 准备目录
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(">>> 1. 初始化数据集 (split='train')...")
    # 这里 split='train' 就会触发 dataset.py 里的增强逻辑
    dataset = GenImageDataset(
        json_file=JSON_PATH,
        tokenizer_path=BERT_PATH,
        split='train'
    )

    print(f">>> 数据集加载完成，准备抽取 20 张进行对比...")

    # 2. 循环抽取样本
    # 直接遍历 dataset，这样我们可以同时拿到 Aug图 和 原始路径
    count = 0
    # 为了随机一点，我们可以跳着抽，或者就抽前20个
    indices = list(range(0, min(len(dataset), 20)))

    for idx in indices:
        # A. 获取增强后的 Tensor (这是喂给 GPU 的)
        # dataset[idx] 返回 (image, input_ids, attention_mask, label)
        aug_tensor, _, _, label = dataset[idx]

        # B. 获取原始图片路径 (这是 dataset 内部存的)
        # 注意：这里假设你的 dataset.data 是一个 list of dict
        raw_data_item = dataset.data[idx]
        raw_img_path = raw_data_item['image_path']

        # C. 处理图片
        img_aug = tensor_to_numpy(aug_tensor)  # 增强后的图
        img_raw = get_raw_image_cv2(raw_img_path)  # 原始图

        # D. 拼接 (水平拼接)
        # 中间加一条黑线分隔
        separator = np.zeros((224, 10, 3), dtype=np.uint8)
        combined = np.hstack([img_raw, separator, img_aug])

        # E. 加文字说明 (左下角)
        cv2.putText(combined, "Original", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(combined, "Augmented (Train)", (244, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # F. 保存
        filename = f"{OUTPUT_DIR}/compare_{idx}_label_{label}.jpg"
        cv2.imwrite(filename, combined)
        print(f"   已生成: {filename}")

    print(f"\n✅ 对比图已生成！请打开文件夹 [{OUTPUT_DIR}] 查看。")
    print("===========================================")
    print("左边是原图，右边是模型看到的图。")
    print("如果右边看起来【明显模糊】或【有马赛克】，说明增强生效了。")
    print("如果左右两边看起来【一模一样】，说明增强挂了！")
    print("===========================================")


if __name__ == "__main__":
    main()