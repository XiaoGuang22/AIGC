#!/usr/bin/env python3
"""
AIGC Experiment Tracker
封装训练 + 指标记录 + 配置快照 + TensorBoard + 实验对比

用法:
  python tools/train_tracker.py \
    --exp-name mmcdn_baseline \
    --lr 0.01 --batch-size 256 --epochs 30 \
    --json data/dataset_sdv4_inblip.json \
    --text-model-path weights/bert-base-uncased \
    --vit-path weights/clip-vit-large-patch14
"""
import argparse, json, os, sys, time, glob
from pathlib import Path
from datetime import datetime

# 确保能找到上级包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RUNS_DIR = "runs"


def save_config(args: argparse.Namespace, run_dir: str):
    """保存完整实验配置"""
    cfg = vars(args)
    cfg["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cfg["hostname"] = os.uname().nodename
    cfg["cuda_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    path = os.path.join(run_dir, "config.json")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"  [Tracker] 配置已保存 → {path}")


def list_experiments() -> list:
    """列出所有已完成的实验及其最佳指标"""
    if not os.path.isdir(RUNS_DIR):
        return []
    exps = []
    for d in sorted(glob.glob(f"{RUNS_DIR}/*/")):
        name = os.path.basename(os.path.normpath(d))
        cfg_file = os.path.join(d, "config.json")
        metrics_file = os.path.join(d, "metrics.csv")
        summary_file = os.path.join(d, "final_summary.json")
        cfg = {}
        if os.path.isfile(cfg_file):
            with open(cfg_file) as f:
                cfg = json.load(f)
        summary = {}
        if os.path.isfile(summary_file):
            with open(summary_file) as f:
                summary = json.load(f)
        exps.append({"name": name, "config": cfg, "summary": summary, "dir": d})
    return exps


def diff_experiments(exp_a: str, exp_b: str):
    """对比两次实验的配置差异和最终指标"""
    exps = list_experiments()
    names = {e["name"]: e for e in exps}
    for name in (exp_a, exp_b):
        if name not in names:
            print(f"  ⚠️ 实验 '{name}' 不存在")
            return
    a, b = names[exp_a], names[exp_b]

    print(f"\n{'=' * 60}")
    print(f"实验对比: {exp_a} vs {exp_b}")
    print(f"{'=' * 60}")

    # 配置差异
    print("\n--- 配置差异 ---")
    ca, cb = a["config"], b["config"]
    all_keys = set(ca.keys()) | set(cb.keys())
    diffs = 0
    for k in sorted(all_keys):
        va = ca.get(k, "<MISSING>")
        vb = cb.get(k, "<MISSING>")
        if str(va) != str(vb):
            print(f"  {k:25s}: {va}  →  {vb}")
            diffs += 1
    if diffs == 0:
        print("  (无差异，配置相同)")

    # 指标对比
    print("\n--- 最终指标 ---")
    sa, sb = a["summary"], b["summary"]
    for met in ["avg_acc", "avg_ap"]:
        va = sa.get(met, "?")
        vb = sb.get(met, "?")
        if va != "?" and vb != "?":
            delta = float(vb) - float(va)
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {met:15s}: {va:>8}  →  {vb:>8}  ({arrow}{delta:+.2f})")
        else:
            print(f"  {met:15s}: {va:>8}  →  {vb:>8}")

    # TensorBoard 链接提示
    print(f"\n  查看完整曲线: tensorboard --logdir {RUNS_DIR} --port 6006")
    print(f"  {exp_a} log: {a['dir']}")
    print(f"  {exp_b} log: {b['dir']}")


def main():
    parser = argparse.ArgumentParser(description="AIGC Experiment Tracker")
    sub = parser.add_subparsers(dest="command")

    # --- train 子命令 ---
    train_p = sub.add_parser("train", help="训练 + 自动追踪")
    train_p.add_argument("--exp-name", required=True, help="实验名称")
    train_p.add_argument("--lr", type=float, default=0.01)
    train_p.add_argument("--batch-size", type=int, default=256)
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--json", default="data/dataset_sdv4_inblip.json")
    train_p.add_argument("--text-model-path", default="weights/bert-base-uncased")
    train_p.add_argument("--vit-path", default="weights/clip-vit-large-patch14")
    train_p.add_argument("--tokenizer-type", default="bert")
    train_p.add_argument("--lambda-weight", type=float, default=0.5)
    train_p.add_argument("--margin", type=float, default=1.5)
    train_p.add_argument("--label-smooth", type=float, default=0.1)
    train_p.add_argument("--weight-decay", type=float, default=0.05)
    train_p.add_argument("--num-workers", type=int, default=8)
    train_p.add_argument("--model-type", default="mmcdn",
                        choices=["mmcdn", "specxnet", "vision_only"])

    # --- list 子命令 ---
    list_p = sub.add_parser("list", help="列出所有实验")
    # --- diff 子命令 ---
    diff_p = sub.add_parser("diff", help="对比两次实验")
    diff_p.add_argument("exp_a", help="实验 A 名称")
    diff_p.add_argument("exp_b", help="实验 B 名称")

    args = parser.parse_args()
    if args.command == "list":
        exps = list_experiments()
        if not exps:
            print(f"  (runs/ 空，尚无实验)")
            return
        print(f"{'名称':30s} {'时间':20s} {'Avg Acc':>8} {'Avg AP':>8}")
        print("-" * 70)
        for e in exps:
            ts = e["config"].get("timestamp", "")[5:19]
            acc = e["summary"].get("avg_acc", "?")
            ap = e["summary"].get("avg_ap", "?")
            print(f"{e['name']:30s} {ts:20s} {str(acc):>8} {str(ap):>8}")
        return
    elif args.command == "diff":
        diff_experiments(args.exp_a, args.exp_b)
        return

    # ========== train 模式 ==========
    exp_name = args.exp_name
    run_dir = os.path.join(RUNS_DIR, exp_name)
    os.makedirs(run_dir, exist_ok=True)

    save_config(args, run_dir)

    print(f"\n{'=' * 60}")
    print(f"开始训练: {exp_name}")
    print(f"{'=' * 60}")

    # 配置显式可见
    cfg = vars(args)
    for k, v in cfg.items():
        if k == "command":
            continue
        print(f"  {k:25s} = {v}")
    print(f"  日志目录: {run_dir}")

    # 动态选择入口
    if args.model_type == "specxnet":
        run_specxnet(args, run_dir)
    elif args.model_type == "mmcdn":
        run_mmcdn(args, run_dir)
    else:
        print(f"  ⚠️ 模型类型 '{args.model_type}' 训练入口待实现")

    print(f"\n  ✅ 训练完成 | 日志 → {run_dir}")
    print(f"  查看曲线: tensorboard --logdir {RUNS_DIR} --port 6006")


def run_mmcdn(args, run_dir):
    """用师兄的 train.py 训练 MMCDN，包装指标追踪"""
    import subprocess, csv
    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=run_dir)

    # 启动训练（直接用师兄的 train.py）
    cmd = [
        sys.executable, "train.py",
        "--json-path", args.json,
        "--text-model-path", args.text_model_path,
        "--vit-path", args.vit_path,
        "--tokenizer-type", args.tokenizer_type,
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--lambda-weight", str(args.lambda_weight),
        "--margin", str(args.margin),
        "--label-smooth", str(args.label_smooth),
        "--weight-decay", str(args.weight_decay),
        "--num-workers", str(args.num_workers),
        "--save-dir", os.path.join(run_dir, "checkpoints"),
        "--log-dir", run_dir,
    ]

    print(f"\n  [CMD] {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, bufsize=1)

    # 实时打印并抓指标
    metrics_csv = os.path.join(run_dir, "metrics.csv")
    with open(metrics_csv, "w", newline="") as csvf:
        cw = csv.writer(csvf)
        cw.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_ap"])

        for line in proc.stdout:
            print(line, end="")
            # 解析训练输出中的指标行（格式需适配师兄 train.py 的输出）
            # 例如: "Epoch 3 | Train Loss: 0.245 | Train Acc: 91.2 | Val Acc: 85.3"
            if "Val Acc" in line and "Epoch" in line:
                try:
                    ep = int(line.split("Epoch")[1].split("|")[0].strip())
                    tl = float(line.split("Train Loss:")[1].split("|")[0].strip())
                    ta = float(line.split("Train Acc:")[1].split("|")[0].strip())
                    va = float(line.split("Val Acc:")[1].split("|")[0].strip())
                    cw.writerow([ep, tl, ta, "", va, ""])
                    csvf.flush()
                    writer.add_scalar("Loss/train", tl, ep)
                    writer.add_scalar("Acc/val", va, ep)
                    writer.flush()
                except (IndexError, ValueError):
                    pass

    proc.wait()
    writer.close()

    # 训练完成后自动跑 evaluate_all.py
    print("\n  [Tracker] 训练完成，启动跨域评测...")
    import subprocess as sp
    eval_cmd = [
        sys.executable, "evaluate_all.py",
        "--model-path", os.path.join(run_dir, "checkpoints", "best_generalization_model.pth"),
        "--vit-path", args.vit_path,
        "--text-model-path", args.text_model_path,
        "--tokenizer-type", args.tokenizer_type,
    ]
    result = sp.run(eval_cmd, capture_output=True, text=True)
    print(result.stdout)

    # 解析评测结果（依 evaluate_all.py 输出格式）


def run_specxnet(args, run_dir):
    """训练 SpecXNet（需适配 DataLoader 到 JSON 格式）"""
    print("  [Tracker] SpecXNet 训练入口 — 将在 Step A 实现")
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=run_dir)
    writer.close()


if __name__ == "__main__":
    main()
