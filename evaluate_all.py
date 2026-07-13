import os
import glob
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from networks.model_v5 import CrossModalDetector
from networks.model_vision_only import VisionOnlyDetector
from utils.dataset import GenImageDataset

# ================= �� 配置 =================
CONFIG = {
    "BATCH_SIZE": 256,
    "NUM_WORKERS": 8,
    "MODEL_PATH": "checkpoints/test39/best_generalization_model.pth",
    "TOKENIZER_TYPE": "bert",
    "BENCHMARK_DIR": "data/test_benchmarks_inblip2",
    "TEXT_MODEL_PATH": "/home/liangpeng/LYK/AIGCdetection/weights/bert-base-uncased",
    "VIT_PATH": "/home/liangpeng/LYK/AIGCdetection/weights/clip-vit-large-patch14",

    "SAVE_DIR": "./results/test_results"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_one_dataset(model, json_path, tokenizer_path, tokenizer_type):
    # 这里 split='val'，对应 JSON 文件里的 split 标签
    dataset = GenImageDataset(
        json_file=json_path,
        tokenizer_path=tokenizer_path,
        split='val',
        tokenizer_type=tokenizer_type
    )

    if len(dataset) == 0:
        print(f"   ⚠️ 数据集为空: {os.path.basename(json_path)}")
        return 0.0, 0.0, 0.0

    loader = DataLoader(dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False, num_workers=CONFIG["NUM_WORKERS"])

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, ids, mask, labels in tqdm(loader, desc=f"Testing {os.path.basename(json_path)}", leave=False):
            imgs, ids, mask = imgs.to(DEVICE), ids.to(DEVICE), mask.to(DEVICE)

            outputs = model(imgs, ids, mask)
            #outputs, _, _ = model(imgs, ids, mask)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    try:
        acc = accuracy_score(all_labels, all_preds)
        ap = average_precision_score(all_labels, all_probs)
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        # print(f"   ⚠️ 计算指标出错: {e}")
        acc, ap, auc = 0.0, 0.0, 0.0

    return acc, ap, auc


def main():
    tokenizer_type = CONFIG["TOKENIZER_TYPE"]

    if tokenizer_type == "clip":
        current_tokenizer_path = CONFIG["VIT_PATH"]
        print(f"�� Configured Tokenizer: CLIP (from {current_tokenizer_path})")
    else:
        current_tokenizer_path = CONFIG["TEXT_MODEL_PATH"]
        print(f"�� Configured Tokenizer: {tokenizer_type.upper()} (from {current_tokenizer_path})")

    print(f"�� Loading Model: {CONFIG['MODEL_PATH']}")
    model = CrossModalDetector(
        text_model_path=CONFIG["TEXT_MODEL_PATH"], vit_path=CONFIG["VIT_PATH"], num_classes=2, freeze_backbone=True
        , text_encoder_type=tokenizer_type).to(DEVICE)

    # model = VisionOnlyDetector(
    #     vit_path=CONFIG["VIT_PATH"], num_classes=2, freeze_backbone=True
    # ).to(DEVICE)

    if not os.path.exists(CONFIG['MODEL_PATH']):
        print(f"❌ 找不到模型文件: {CONFIG['MODEL_PATH']}")
        return

    model.load_state_dict(torch.load(CONFIG['MODEL_PATH'], map_location=DEVICE))
    model.eval()

    json_files = sorted(glob.glob(os.path.join(CONFIG["BENCHMARK_DIR"], "*.json")))
    if not json_files:
        print("❌ 没找到测试文件，请先运行 generate_test_json.py")
        return

    results = []
    print(f"准备测试 {len(json_files)} 个数据集...\n")

    for json_file in json_files:
        name = os.path.basename(json_file).replace("test_", "").replace(".json", "")
        acc, ap, auc = evaluate_one_dataset(
            model,
            json_file,
            tokenizer_path=current_tokenizer_path,
            tokenizer_type=tokenizer_type
        )

        results.append({
            "Generator": name,
            "Acc (%)": round(acc * 100, 2),
            "AP (%)": round(ap * 100, 2),
            "AUC (%)": round(auc * 100, 2)
        })
        print(f"  ✅ {name:<20} | Acc: {acc * 100:.2f}% | AP: {ap * 100:.2f}%")

    # 保存结果
    df = pd.DataFrame(results)

    # 计算平均值行
    if len(df) > 0:
        avg_row = pd.DataFrame([{
            "Generator": "AVERAGE",
            "Acc (%)": df["Acc (%)"].mean(),
            "AP (%)": df["AP (%)"].mean(),
            "AUC (%)": df["AUC (%)"].mean()
        }])
        df = pd.concat([df, avg_row], ignore_index=True)

    csv_path = "final_benchmark_results.csv"
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 50)
    print(f"�� 测试完成！结果已保存至: {csv_path}")
    print("=" * 50)
    try:
        print(df.to_markdown(index=False))
    except ImportError:
        print(df)


if __name__ == "__main__":
    main()