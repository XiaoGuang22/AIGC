import json
import os
from PIL import Image
from tqdm import tqdm


def clean_dataset(input_json, output_json):
    """
    input_json: 原始包含坏图路径的 JSON 文件
    output_json: 清洗后生成的干净 JSON 文件
    """
    # 1. 加载原始数据
    if not os.path.exists(input_json):
        print(f"错误：找不到输入文件 {input_json}")
        return

    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"原始数据总量: {len(data)}")
    print("开始检查图片完整性")

    clean_data = []
    broken_count = 0

    # 2. 遍历检查
    for item in tqdm(data, desc="清洗进度"):
        img_path = item.get('image_path', '')

        # 检查逻辑 A：物理路径是否存在
        if not os.path.exists(img_path):
            broken_count += 1
            continue

        # 检查逻辑 B：图片内容是否损坏（尝试打开并验证）
        try:
            with Image.open(img_path) as img:
                # verify() 能检查文件是否截断或损坏，而不必完全解码，速度较快
                img.verify()

                # 如果走到这一步没报错，说明图是好的
            clean_data.append(item)
        except Exception:
            # 捕获所有 PIL 抛出的读取错误
            broken_count += 1
            continue

    # 3. 保存结果
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(clean_data, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"✅ 清洗完成！")
    print(f"剔除坏图数量: {broken_count}")
    print(f"剩余有效数据: {len(clean_data)}")
    print(f"新索引已保存至: {output_json}")
    print("-" * 30)


if __name__ == "__main__":
    # --- 你只需要修改这里 ---
    INPUT_PATH = "data/test_benchmarks_inblip2/test_Glide_inblip.json"  # 你的原始 JSON
    OUTPUT_PATH = "data/test_benchmarks_inblip2_clean/test_Glide_inblip_cleaned.json"  # 清洗后的 JSON
    # -----------------------

    clean_dataset(INPUT_PATH, OUTPUT_PATH)