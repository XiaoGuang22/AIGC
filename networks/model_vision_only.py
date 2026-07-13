import torch
import torch.nn as nn
from transformers import CLIPVisionModel


class VisionOnlyDetector(nn.Module):
    def __init__(self,
                 vit_path,
                 num_classes=2,
                 freeze_backbone=True):
        super(VisionOnlyDetector, self).__init__()

        # 1. 视觉编码器 (完全一样)
        print(f"Loading CLIP-ViT (Patch16) from: {vit_path}")
        self.vision_encoder = CLIPVisionModel.from_pretrained(
            vit_path,
            output_hidden_states=True  # 必须开启
        )

        # ❌ 砍掉 BERT
        # ❌ 砍掉 Cross-Attention

        # 2. 融合维度计算
        # 我们依然融合 Layer 12, 11, 10
        # 但 Layer 12 不再经过 Attention，而是直接取 CLS
        fusion_dim = 768 * 3

        # 3. 分类头 (保持完全一致，控制变量)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(512, num_classes)
        )

        # 4. 冻结策略
        if freeze_backbone:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
            print("Backbone status: FROZEN")

    def forward(self, images, input_ids=None, attention_mask=None):
        # input_ids 和 attention_mask 即使传进来也不用

        # 1. 视觉特征提取
        vision_outputs = self.vision_encoder(pixel_values=images)
        hidden_states = vision_outputs.hidden_states

        # 2. 特征融合 (关键消融点)

        # Layer 12 CLS (原先是拿去做 Attention 的，现在直接取)
        feat_l12_cls = hidden_states[-1][:, 0, :]

        # Layer 11 CLS
        feat_l11_cls = hidden_states[-2][:, 0, :]

        # Layer 10 CLS
        feat_l10_cls = hidden_states[-3][:, 0, :]

        # 3. 拼接
        combined_features = torch.cat([feat_l12_cls, feat_l11_cls, feat_l10_cls], dim=1)

        # 4. 分类
        logits = self.classifier(combined_features)

        return logits