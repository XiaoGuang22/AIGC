import json
import os
import glob

# ================= 配置区域 =================
# 你可以指定单个文件，或者使用包含通配符的路径批量处理
# 例如: "data/dataset_sdv4_InstructBLIP.json" 或 "data/*_InstructBLIP.json"
TARGET_JSON_PATTERN = "data/test_benchmarks_inblip2/*_inblip.json"
# "data/test_benchmarks_inblip/test_ADM_inblip.json"
# 是否覆盖原文件？建议先设为 False，测试没问题后再覆盖
OVERWRITE = True


# ============================================

def clean_truncated_captions(json_path):
    print(f"�� 正在检查文件: {json_path}")

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cleaned_count = 0
    for item in data:
        caption = item.get('caption', '').strip()

        if not caption:
            continue

        # 检查是否以规范的句子结束符结尾
        if not caption.endswith(('.', '!', '?', ',')):
            # 找到文本中最后一个句号、叹号或问号的索引
            last_punct = max(caption.rfind('.'), caption.rfind('!'), caption.rfind('?'), caption.rfind(','))

            if last_punct != -1:
                # 核心操作：切片保留到最后一个有效标点，抛弃后面的半截残句
                cleaned_caption = caption[:last_punct + 1].strip()
                item['caption'] = cleaned_caption
                cleaned_count += 1
            else:
                # 极端情况：整段话连一个句号都没有，说明模型生成了彻底的残句
                # 强行给它补上一个句号，使其符合语法基础闭环
                item['caption'] = caption + "."
                cleaned_count += 1

    # 决定输出路径
    output_path = json_path if OVERWRITE else json_path.replace(".json", "_cleaned.json")

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ 处理完成！共修复了 {cleaned_count} 条残缺句子。保存至: {output_path}")
    print("-" * 50)


if __name__ == "__main__":
    # 使用 glob 批量匹配文件
    json_files = glob.glob(TARGET_JSON_PATTERN)

    if not json_files:
        print("❌ 未找到匹配的 JSON 文件，请检查 TARGET_JSON_PATTERN 路径！")
    else:
        for jf in json_files:
            clean_truncated_captions(jf)