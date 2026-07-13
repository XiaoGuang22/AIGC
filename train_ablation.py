import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from networks.model import CrossModalDetector
from utils.dataset import GenImageDataset
from utils.optimizer_utils import build_optimizer
from transformers import get_cosine_schedule_with_warmup
from networks.model_vision_only import VisionOnlyDetector

# ================= 超参数配置 =================
CONFIG = {
    "BATCH_SIZE": 256,
    "LR": 0.01,
    "NUM_EPOCHS": 100,
    "NUM_WORKERS": 8,
    "WEIGHT_DECAY": 0.05,

    "SAVE_FREQ": 5,
    "SAVE_DIR": "checkpoints/100",
    "LOG_DIR": "results/logs",
    "JSON_PATH": "data/dataset_sdv5_blip.json",
    "VAL_UNSEEN_JSON": "data/test_benchmarks_blip/test_Midjourney_blip.json",
    "BERT_PATH": "/home/liangpeng/LYK/AIGCdetection/weights/bert-base-uncased",
    "VIT_PATH": "/home/liangpeng/LYK/AIGCdetection/weights/clip-vit-base-patch16",

    # 策略核心：全冻结
    "TRAINABLE_LAYERS": 0,

    # �� [修改 5] 注释掉预训练权重
    # 因为我们换了 Multi-Level 模型架构，维度变了，必须从头训练。
    # "PRETRAINED_PATH": "save_checkpoints/test4_clip-vit-best_generalization_model.pth"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 日志辅助工具 =================
def get_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join(log_dir, f"train_log_{timestamp}.txt")

    def log_print(message):
        print(message)
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    return log_print


# ================= 通用验证函数 =================
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

    avg_loss = val_loss / len(loader)
    acc = 100 * correct / total
    return avg_loss, acc


# ================= 训练主程序 =================
def train():
    log_print = get_logger(CONFIG["LOG_DIR"])
    os.makedirs(CONFIG["SAVE_DIR"], exist_ok=True)

    log_print("=" * 40)
    log_print(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Device    : {DEVICE}")
    log_print("-" * 40)
    log_print("Hyperparameters Configuration (UFD Strategy):")
    for k, v in CONFIG.items():
        log_print(f"  {k:<15} : {v}")
    log_print("=" * 40 + "\n")

    # 2. 加载数据
    log_print(">>> Initializing Datasets...")
    # A. 训练集 - 保持 Blur/JPEG 增强 (UFD 灵魂)
    train_dataset = GenImageDataset(
        json_file=CONFIG["JSON_PATH"],
        tokenizer_path=CONFIG["BERT_PATH"],
        split='train'
    )
    # B. 验证集
    val_seen_dataset = GenImageDataset(
        json_file=CONFIG["JSON_PATH"],
        tokenizer_path=CONFIG["BERT_PATH"],
        split='val'
    )
    # C. 泛化集
    if os.path.exists(CONFIG["VAL_UNSEEN_JSON"]):
        val_unseen_dataset = GenImageDataset(
            json_file=CONFIG["VAL_UNSEEN_JSON"],
            tokenizer_path=CONFIG["BERT_PATH"],
            split='val'
        )
        has_unseen = True
    else:
        has_unseen = False

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                              num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    val_seen_loader = DataLoader(val_seen_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                                 num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
    if has_unseen:
        val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                                       num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

    # 3. 初始化模型
    log_print(">>> Initializing VISION-ONLY Model (Ablation Study)...")
    model = VisionOnlyDetector(
        vit_path=CONFIG["VIT_PATH"],
        num_classes=2,
        freeze_backbone=True
    ).to(DEVICE)

    # �� [修改 7] 删除/注释掉原来的 load_state_dict 代码
    # 因为我们采用了新的模型架构（多层特征融合），输入维度变了，必须从头初始化分类头。
    print(">>> Training from SCRATCH (New Classification Head)")

    # 4. 定义优化器 & 调度器
    # 加入 Label Smoothing，防止模型在 SDv5 上过度自信
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 使用 build_optimizer 自动读取 CONFIG 里的 LR=1e-3 和 TRAINABLE_LAYERS=0
    optimizer, scheduler, scaler = build_optimizer(model, CONFIG, len(train_loader))

    # 5. 训练循环
    best_seen_acc = 0.0
    best_unseen_acc = 0.0
    start_time = time.time()

    log_print(f"\n>>> Start Training for {CONFIG['NUM_EPOCHS']} Epochs...")

    for epoch in range(CONFIG["NUM_EPOCHS"]):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{CONFIG['NUM_EPOCHS']}] Train", leave=True)

        for images, input_ids, attention_mask, labels in loop:
            images, input_ids, attention_mask, labels = \
                images.to(DEVICE), input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)

            with autocast():
                outputs = model(images, input_ids, attention_mask)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # 1. 先将梯度从缩放状态还原 (Unscale)
            scaler.unscale_(optimizer)

            # 2. 裁剪梯度 (防止梯度爆炸导致 NaN)
            # max_norm 建议设为 1.0 或 5.0，配合 0.01 的高 LR，设为 1.0 更安全
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            # �� [修改 9] 确保 scheduler 在这里更新 (Batch-wise)
            scheduler.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        # �� [修改 10] 确保 Epoch 循环外没有 scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # === 验证 ===
        loss_seen, acc_seen = validate(model, val_seen_loader, criterion, DEVICE, desc="Val Seen")

        if has_unseen:
            loss_unseen, acc_unseen = validate(model, val_unseen_loader, criterion, DEVICE, desc="Val Unseen")
            unseen_msg = f" | Gen Acc(MJ): {acc_unseen:.2f}%"
        else:
            acc_unseen = 0.0
            unseen_msg = ""

        log_msg = (f"Epoch [{epoch + 1}/{CONFIG['NUM_EPOCHS']}] "
                   f"LR: {current_lr:.6f} | "
                   f"Train Loss: {avg_train_loss:.4f} | "
                   f"Seen Acc: {acc_seen:.2f}%"
                   f"{unseen_msg}")
        log_print(log_msg)

        # === 保存策略 ===
        if acc_seen > best_seen_acc:
            best_seen_acc = acc_seen
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], "best_seen_model.pth"))

        if has_unseen and acc_unseen > best_unseen_acc:
            best_unseen_acc = acc_unseen
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], "best_generalization_model.pth"))
            log_print(f"  >>> �� New Best Generalization Model! (MJ Acc: {best_unseen_acc:.2f}%)")

        if (epoch + 1) % CONFIG["SAVE_FREQ"] == 0:
            torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], f"checkpoint_epoch_{epoch + 1}.pth"))

        torch.save(model.state_dict(), os.path.join(CONFIG["SAVE_DIR"], "last_model.pth"))

    total_time = time.time() - start_time
    log_print(f"\nTraining Finished in {total_time / 3600:.2f} hours.")
    log_print(f"Best Seen Acc (SDv5) : {best_seen_acc:.2f}%")
    if has_unseen:
        log_print(f"Best Gen Acc (MJ)    : {best_unseen_acc:.2f}%")
    log_print(f"Log saved to {CONFIG['LOG_DIR']}")


if __name__ == "__main__":
    train()