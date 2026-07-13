import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from transformers import get_cosine_schedule_with_warmup


def build_optimizer(model, config, steps_per_epoch):
    """
    封装了分层冻结策略、优化器构建和调度器设置
    适配 Phase 2.0 (跨模态语义冲突检测：Cosine Annealing + 1 Epoch Warmup)
    """
    # 1. 读取配置
    trainable_layers = config.get("TRAINABLE_LAYERS", 0)
    lr = config.get("LR", 1e-3)
    weight_decay = config.get("WEIGHT_DECAY", 0.01)
    accum_steps = config.get("ACCUMULATION_STEPS", 1)

    print(f"\n>>> ��️ [Optimizer Builder] Strategy: Train Last {trainable_layers} Layers | LR: {lr}")

    # ============================================================
    # 重点：冻结逻辑已经移交给了 Model 类！
    # 优化器现在只负责无脑接收 requires_grad==True 的参数
    # ============================================================
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())

    # ============================================================
    # B. 构建优化器与调度器 (�� 核心修改处)
    # ============================================================

    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # -------------------------------------------------------------
    # ✅ 修改为 Cosine Annealing with Warmup
    # -------------------------------------------------------------

    # 1. 计算总步数 (Step 而不是 Epoch)
    # effective_steps_per_epoch = steps_per_epoch // accum_steps
    # total_epochs = config.get("NUM_EPOCHS", 100)
    #
    # warmup_epochs = 1  # 保持 1 轮 Warmup
    #
    # warmup_steps = effective_steps_per_epoch * warmup_epochs
    # total_steps = effective_steps_per_epoch * total_epochs
    #
    # # 2. 构建余弦退火调度器
    # scheduler = get_cosine_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    #
    # print(f"�� Scheduler: Cosine with Warmup")
    # print(f"   - Warmup Steps: {warmup_steps} (approx. {warmup_epochs} epoch)")
    # print(f"   - Total Steps: {total_steps} ({total_epochs} epochs)")
    # print(f"   ⚠️ FATAL: Scheduler step() MUST be called per BATCH, not per epoch!")

    # # -------------------------------------------------------------
    # # ✅ 修改为 ReduceLROnPlateau
    # # -------------------------------------------------------------
    #
    # scheduler = ReduceLROnPlateau(
    #     optimizer,
    #     mode='max',  # 监控指标是 Accuracy (越大越好)
    #     factor=0.1,  # 每次衰减十倍，或0.5每次衰减 50% (减半)
    #     patience=200,  # 容忍 5 个 Epoch 指标不提升
    #     verbose=True,  # 触发衰减时打印日志
    #     threshold=1e-4,  # 只有提升超过这个阈值才算提升
    #     min_lr=1e-6  # 学习率下限
    # )
    # print(f"Scheduler: ReduceLROnPlateau (Mode='max', Patience=5, Factor=0.5)")
    # print(f"   Note: Scheduler step() must be called AFTER validation with the metric value.")

    # ============================================================
    # �� 修改为：固定学习率 (Constant LR)
    # ============================================================

    # 定义一个永远返回 1.0 的匿名函数
    # LambdaLR 会用初始学习率 (lr) 乘以这个返回值，1.0 代表永远不衰减
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    print(f"�� Scheduler: Constant LR (Fixed at {lr})")
    print(f"   - No Warmup, No Decay.")

    scaler = GradScaler()
    return optimizer, scheduler, scaler