import os
import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel

"""该版本为原版，提取L10 L11 L12 三层视觉和 text与L12融合的特征，四层再通过torch.cat拼接"""

class CrossModalDetector(nn.Module):
    def __init__(self,
                 bert_path,
                 vit_path,
                 num_classes=2,
                 freeze_backbone=True):
        super(CrossModalDetector, self).__init__()

        # --- 1. 视觉编码器 ---
        print(f"Loading CLIP-ViT (Patch16) from: {vit_path}")
        try:
            # 关键修改：我们需要访问所有层的输出，不只是最后一层
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vit_path,
                output_hidden_states=True  # 开启隐藏层输出
            )
            print("✅ CLIP-ViT-Patch16 Loaded Successfully")
        except Exception as e:
            raise FileNotFoundError(f"❌ Failed to load CLIP-ViT: {e}")

        # --- 2. 文本编码器 ---
        print(f"Loading BERT from: {bert_path}")
        if os.path.exists(bert_path):
            self.text_encoder = BertModel.from_pretrained(bert_path)
            print("✅ BERT Loaded Successfully")
        else:
            raise FileNotFoundError(f"❌ BERT path not found")

        # --- 3. 融合层与分类头 (架构大改) ---
        self.embed_dim = 768

        # Cross-Attention: 依然只对最后一层做 (语义层对齐文本)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True
        )

        #  关键修改：输入维度变大了，改为 4 层融合
        # 我们融合了:
        # 0. Layer 12 CLS Token (原始视觉特征，保留Wukong/SD能力) -> 768 [新增]
        # 1. Attention Output (Layer 12 + Text) -> 768
        # 2. Layer 11 CLS Token (纹理特征) -> 768
        # 3. Layer 10 CLS Token (浅层特征) -> 768
        fusion_dim = 768 * 4

        #  关键修改：更强的分类头 (MLP + High Dropout)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),  # 加 BN 稳定分布
            nn.ReLU(),
            nn.Dropout(0.5),  # 0.5 Dropout 防止过拟合

            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),   # 自己新增，测试test
            nn.ReLU(),
            nn.Dropout(0.5),  # 第二层也加 Dropout

            nn.Linear(512, num_classes)
        )

        # --- 4. 初始化冻结 ---
        if freeze_backbone:
            self._set_freeze_status(True)
        else:
            self._set_freeze_status(False)

    def _set_freeze_status(self, freeze: bool):
        for param in self.vision_encoder.parameters():
            param.requires_grad = not freeze
        for param in self.text_encoder.parameters():
            param.requires_grad = not freeze
        print(f"Backbone status: {'FROZEN' if freeze else 'UNFROZEN'}")

    def forward(self, images, input_ids, attention_mask):
        # --- 1. 视觉特征 (多层提取) ---
        # output_hidden_states=True 后，返回的是 tuple
        vision_outputs = self.vision_encoder(pixel_values=images)
        hidden_states = vision_outputs.hidden_states
        # hidden_states 是一个 tuple，包含 (embeddings, layer_0, ..., layer_11)
        # 最后一层是 hidden_states[-1]

        # A. 语义特征 (Layer 12): [Batch, 197, 768] - 用于做 Attention
        feat_l12 = hidden_states[-1]

        # [新增] 提取原始 L12 CLS 用于直接融合
        feat_l12_cls = feat_l12[:, 0, :]

        # B. 纹理/结构特征 (Layer 11 & 10): 取 CLS Token [Batch, 768]
        # 直接拿来拼接，补充丢失的伪影信息
        feat_l11_cls = hidden_states[-2][:, 0, :]
        feat_l10_cls = hidden_states[-3][:, 0, :]

        # --- 2. 文本特征 ---
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [Batch, Seq, 768]

        # --- 3. 跨模态注意力 (只针对 Layer 12) ---
        # Query: Image L12
        # Key/Val: Text
        attn_output, _ = self.cross_attention(
            query=feat_l12,
            key=text_features,
            value=text_features,
            key_padding_mask=~attention_mask.bool()
        )
        # 取 Attention 后的 CLS [Batch, 768]
        attn_cls = attn_output[:, 0, :]

        # --- 4. 特征融合 (Concatenation) ---
        # [修改] 4路融合：原始L12(保下限) + Attn(提上限) + L11 + L10
        # 形状: [Batch, 768*4] = [Batch, 3072]
        combined_features = torch.cat([feat_l12_cls, attn_cls, feat_l11_cls, feat_l10_cls], dim=1)

        # --- 5. 分类 ---
        logits = self.classifier(combined_features)

        return logits