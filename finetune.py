import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

# 复用你现有的模块
from networks.model import CrossModalDetector
from utils.dataset import GenImageDataset

# ================= 超参数配置 (Phase 1.5: Annealing) =================
CONFIG = {
    "BATCH_SIZE": 256,

    # ⬇️ [关键修正] 学习率降低 10 倍，进行精细微调
    "LR": 0.001,

    "NUM_EPOCHS": 15,  # 不需要跑太久，15 轮足够收敛
    "NUM_WORKERS": 8,
    "WEIGHT_DECAY": 0.05,
    "TRAINABLE_LAYERS": 0,  # 继续保持全冻结

    # ⬇️ [必须修改] 这里填你刚才跑出来的最佳模型路径 (Epoch 8 或 21)
    "PRETRAINED_PATH": "checkpoints/100/best_generalization_model.pth",

    "SAVE_DIR": "checkpoints/finetune_annealing",  # 保存到新目录
    "LOG_DIR": "results/logs_finetune",

    # 数据集路径保持不变
    "JSON_PATH": "data/dataset_sdv5_blip.json",
    "VAL_UNSEEN_JSON": "data/test_benchmarks_blip/test_Midjourney_blip.json",
    "BERT_PATH": "/home/liangpeng/LYK/AIGCdetection/weights/bert-base-uncased",
    "VIT_PATH": "/home/liangpeng/LYK/AIGCdetection/weights/clip-vit-base-patch16",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 日志工具 =================
def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f"finetune_log_{timestamp}.txt")

    def log_print(message):
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    return log_print


# ================= 验证函数 =================
def validate(model, loader, criterion, device, desc="Validation"):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc=desc, leave=False)
    with torch.no_grad():
        for images, input_ids, attention_mask, labels in loop:
            images, input_ids, attention_mask, labels = \
                images.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return val_loss / len(loader), 100 * correct / total


# ================= 微调主程序 =================
def main():
    log_print = get_logger(CONFIG["LOG_DIR"])
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

    log_print(f">>> Start Fine-tuning (Annealing Phase)")
    log_print(f"Target LR: {CONFIG['LR']} | Epochs: {CONFIG['NUM_EPOCHS']}")
    log_print(f"Loading Weights from: {CONFIG['PRETRAINED_PATH']}")

    # 1. 数据准备
    train_dataset = GenImageDataset(CONFIG["JSON_PATH"], CONFIG["BERT_PATH"], split='train')
    val_seen_dataset = GenImageDataset(CONFIG["JSON_PATH"], CONFIG["BERT_PATH"], split='val')
    val_unseen_dataset = GenImageDataset(CONFIG["VAL_UNSEEN_JSON"], CONFIG["BERT_PATH"], split='val')

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_seen_loader = DataLoader(val_seen_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                                 num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                                   num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    # 2. 模型初始化
    model = CrossModalDetector(
        bert_path=CONFIG["BERT_PATH"],
        vit_path=CONFIG["VIT_PATH"],
        num_classes=2,
        freeze_backbone=True
    ).to(DEVICE)

    # 3. 加载预训练权重 (核心步骤)
    if os.path.exists(CONFIG["PRETRAINED_PATH"]):
        checkpoint = torch.load(CONFIG["PRETRAINED_PATH"], map_location=DEVICE)
        # strict=False 防止有些无关层(如text_model unused weights)报错，通常没问题
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        log_print(f"✅ Weights Loaded. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        raise FileNotFoundError(f"❌ Checkpoint not found: {CONFIG['PRETRAINED_PATH']}")

    # 4. 优化器设置 (独立设置，不使用 build_optimizer 以确保完全控制)
    #    只优化 requires_grad=True 的参数 (Classifier + CrossAttn)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=CONFIG["LR"],
                            weight_decay=CONFIG["WEIGHT_DECAY"])

    # 5. 调度器设置 (Cosine Annealing With Warmup)
    #    微调时，稍微给一点 Warmup (1 epoch) 防止甚至 0.001 也会震荡，然后 Cosine 下降
    num_training_steps = len(train_loader) * CONFIG["NUM_EPOCHS"]
    num_warmup_steps = len(train_loader) * 1  # 1 Epoch Warmup

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)

    # 6. 训练循环
    best_unseen_acc = 0.0
    # 先测一次当前模型的性能作为基准
    log_print(">>> Benchmarking loaded model...")
    _, start_acc = validate(model, val_unseen_loader, criterion, DEVICE, desc="Baseline Test")
    log_print(f"Starting Baseline MJ Acc: {start_acc:.2f}%")
    best_unseen_acc = start_acc

    for epoch in range(CONFIG["NUM_EPOCHS"]):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Fine-tune [{epoch + 1}/{CONFIG['NUM_EPOCHS']}]", leave=True)

        for images, input_ids, attention_mask, labels in loop:
            images, input_ids, attention_mask, labels = \
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

            with autocast():
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # ✅ 保持 Gradient Clipping (安全第一)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # Validation
        loss_seen, acc_seen = validate(model, val_seen_loader, criterion, DEVICE, desc="Val Seen")
        loss_unseen, acc_unseen = validate(model, val_unseen_loader, criterion, DEVICE, desc="Val MJ")

        log_print(f"Epoch [{epoch + 1}] LR: {optimizer.param_groups[0]['lr']:.6f} | "
                  f"Loss: {train_loss / len(train_loader):.4f} | "
                  f"Seen Acc: {acc_seen:.2f}% | Gen Acc (MJ): {acc_unseen:.2f}%")

        # Save Best
        if acc_unseen > best_unseen_acc:
            best_unseen_acc = acc_unseen
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], "best_finetuned.pth"))
            log_print(f"  >>> �� New Best MJ Acc: {best_unseen_acc:.2f}%")

        # Save Last
        torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], "last_finetuned.pth"))

    log_print(f"Fine-tuning Finished. Best MJ Acc: {best_unseen_acc:.2f}%")


if __name__ == "__main__":
    main()