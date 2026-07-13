#!/usr/bin/env python3
"""
SpecXNet + 师兄数据（JSON 格式）的数据集适配与训练入口。
"""
import os, sys, json, argparse, time, csv
from copy import deepcopy
from pathlib import Path
from collections import Counter

# 加入 SpecXNet 代码路径
SPECXNET_PATH = "/home/liangpeng/LXG/SpecXNet/SpecXNet-repo"
sys.path.insert(0, SPECXNET_PATH)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ========== SpecXNet 模型（从修好的 model_zoo 导入）==========
sys.path.insert(0, SPECXNET_PATH)
# 先 import spectral 确保模块注册，再 import specXnet
import model_zoo  # __init__.py 已被我改为 from .specXnet import * + from .spectral import *
ffc_xception = model_zoo.ffc_xception

# ========== 归一化常数 ==========
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

INPUT_SIZE = 299  # SpecXNet (XceptionNet) 原生输入


class GenImageDataset(Dataset):
    """适配师兄 JSON 格式的纯视觉数据集（仅 image_path + label）"""

    def __init__(self, json_file, split='train', transform=None):
        with open(json_file, 'r') as f:
            all_data = json.load(f)
        self.data = [item for item in all_data if item['split'] == split]
        if len(self.data) == 0:
            raise ValueError(f"没有找到 split='{split}' 的样本")
        self.transform = transform or self._default_transform(split == 'train')
        print(f"[{split.upper()}] 加载 {len(self.data)} 个样本")

    @staticmethod
    def _default_transform(is_train=True):
        if is_train:
            return transforms.Compose([
                transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.7, 1.0)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomGrayscale(p=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(INPUT_SIZE + 34),
                transforms.CenterCrop(INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = Image.open(item['image_path']).convert('RGB')
        img = self.transform(img)
        label = item['label']
        return img, label


def build_test_loaders(benchmark_dir, batch_size=256, num_workers=8):
    """载入师兄的全部 8 个跨域测试集"""
    test_jsons = [
        ("SD v1.4", "test_stable_diffusion_v_1_4"),
        ("SD v1.5", "test_stable_diffusion_v_1_5"),
        ("Midjourney", "test_Midjourney"),
        ("GLIDE", "test_Glide"),
        ("Wukong", "test_wukong"),
        ("ADM", "test_ADM"),
        ("VQDM", "test_VQDM"),
        ("BigGAN", "test_biggan"),
    ]
    loaders = {}
    transform = GenImageDataset._default_transform(is_train=False)
    for name, jname in test_jsons:
        path = os.path.join(benchmark_dir, f"{jname}.json")
        if not os.path.exists(path):
            print(f"  ⚠️ 跳过 {name}：{path} 不存在")
            continue
        ds = GenImageDataset(path, split='val', transform=transform)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        loaders[name] = dl
    return loaders


def evaluate(model, test_loaders, device):
    """跨域评测，返回 {name: {acc, ap}}"""
    from sklearn.metrics import accuracy_score, average_precision_score
    model.eval()
    results = {}
    with torch.no_grad():
        for name, loader in test_loaders.items():
            all_labels, all_probs = [], []
            for imgs, labels in loader:
                imgs = imgs.to(device, non_blocking=True)
                outputs = model(imgs)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
            acc = accuracy_score(all_labels, (np.array(all_probs) > 0.5).astype(int))
            ap = average_precision_score(all_labels, all_probs)
            results[name] = {"acc": round(acc * 100, 2), "ap": round(ap * 100, 2)}
            print(f"    {name:15s}  Acc={acc*100:.2f}%  AP={ap*100:.2f}%")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", default="specxnet_baseline")
    parser.add_argument("--json-path", default="/home/liangpeng/LYK/AIGCdetection/data/dataset_sdv4_inblip.json")
    parser.add_argument("--benchmark-dir", default="/home/liangpeng/LYK/AIGCdetection/data/test_benchmarks")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--lr-steps", nargs="+", type=int, default=[30, 60, 80])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--lfu", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import numpy as np
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # ========== 日志目录 ==========
    run_dir = os.path.join("runs", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)

    # 保存配置
    cfg = vars(args)
    cfg["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # ========== 数据 ==========
    train_ds = GenImageDataset(args.json_path, split='train')
    val_ds   = GenImageDataset(args.json_path, split='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loaders = build_test_loaders(args.benchmark_dir)

    # ========== 模型 ==========
    model = ffc_xception(num_classes=2, ratio=args.ratio, lfu=args.lfu, use_se=False)
    model = model.to(device)

    # 为 A6000 开启自动混合精度
    scaler = torch.cuda.amp.GradScaler()

    # ========== 优化器 & 调度器 ==========
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    # ========== 训练循环 ==========
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as csf:
        cw = csv.writer(csf)
        cw.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

        best_val_acc = 0.0
        for epoch in range(args.epochs):
            model.train()
            total_loss, correct, total = 0, 0, 0

            for i, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_loss = total_loss / total
            train_acc = correct / total * 100

            # 验证
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total * 100

            scheduler.step()
            lr = optimizer.param_groups[0]['lr']

            cw.writerow([epoch, round(train_loss, 4), round(train_acc, 2), round(val_acc, 2)])
            csf.flush()

            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Acc/train", train_acc, epoch)
            writer.add_scalar("Acc/val", val_acc, epoch)
            writer.add_scalar("LR", lr, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pth"))

            log = (f"Epoch {epoch+1:2d}/{args.epochs}"
                   f" | Loss {train_loss:.4f} | TrainAcc {train_acc:.2f}%"
                   f" | ValAcc {val_acc:.2f}% | LR {lr:.5f}")
            if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == args.epochs - 1:
                print(log)

    writer.close()

    # ========== 跨域评测 ==========
    print("\n" + "=" * 60)
    print("跨域评测 (Zero-shot)")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(run_dir, "best_model.pth")))
    results = evaluate(model, test_loaders, device)
    avg_acc = sum(r["acc"] for r in results.values()) / len(results)
    avg_ap  = sum(r["ap"]  for r in results.values()) / len(results)

    summary = {
        "best_val_acc": round(best_val_acc, 2),
        "avg_acc": round(avg_acc, 2),
        "avg_ap": round(avg_ap, 2),
        "per_generator": results
    }
    with open(os.path.join(run_dir, "final_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Avg Acc = {avg_acc:.2f}% | Avg AP = {avg_ap:.2f}%")
    print(f"  ✅ 训练完成 | 日志 → {run_dir}")
    print(f"  查看曲线: tensorboard --logdir runs --port 6006")


if __name__ == "__main__":
    import numpy as np
    main()
