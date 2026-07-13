import torch
import torch.nn as nn
# [修改] 导入 CLIPTextModel
from transformers import CLIPModel, CLIPTextModel, BertModel

"""该版本使用CLIP-L/14这种更大的模型作为视觉编码器，文本编码器也使用CLIP原生"""

class CrossModalDetector(nn.Module):
    def __init__(self,
                 vit_path,
                 bert_path=None,
                 num_classes=2,
                 freeze_backbone=True,
                 text_encoder_type='clip'):
        super(CrossModalDetector, self).__init__()

        # --- 1. 加载 CLIP Vision (L/14) ---
        print(f"Loading CLIP (L/14) from: {vit_path}")
        try:
            # 加载完整模型，这样才有 visual_projection
            self.clip_model = CLIPModel.from_pretrained(vit_path)
            print("✅ CLIPModel Loaded Successfully")
        except:
            print("⚠️ Local path failed, downloading openai/clip-vit-large-patch14...")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

        # 拆解 CLIP，拿到我们需要的组件
        self.vision_encoder = self.clip_model.vision_model  # 输出 1024
        self.visual_proj = self.clip_model.visual_projection  # Linear(1024 -> 768) [预训练好的!]

        # --- 2. 动态加载 Text Encoder ---
        self.text_encoder_type = str(text_encoder_type).lower().strip()
        print(f"Loading Text Encoder ({self.text_encoder_type.upper()})...")

        if self.text_encoder_type == 'clip':
            # 直接复用 clip_model 里的文本塔
            self.text_encoder = self.clip_model.text_model
            self.text_proj = self.clip_model.text_projection  # CLIP文本也有个投影
            print(f"✅ CLIPTextModel loaded from {vit_path}")

        elif self.text_encoder_type == 'bert':
            if bert_path is None: bert_path = "bert-base-uncased"
            self.text_encoder = BertModel.from_pretrained(bert_path)
            self.text_proj = nn.Identity()  # BERT 不需要投影，原生就是 768
            print(f"✅ BertModel loaded from {bert_path}")


        # --- 3. 维度确认 ---
        if hasattr(self.visual_proj, 'out_features'):
            # 如果是 Linear 层 (如 L/14)
            self.embed_dim = self.visual_proj.out_features
        else:
            # 如果是 Identity (某些 Base 模型实现)
            self.embed_dim = self.vision_encoder.config.hidden_size
        print(f"\n[Dimension Check]")
        print(f">>> Unified Dimension for Cross-Attention: {self.embed_dim}")
        print(f"Vision Raw Dim: {self.vision_encoder.config.hidden_size}")  # L/14 应为 1024
        print(f"Vision Dim: {self.visual_proj.out_features}")  # L/14 应为 768
        print(f"Text Raw Dim:   {self.text_encoder.config.hidden_size}")  # BERT/CLIP 应为 768

        assert self.embed_dim == self.text_encoder.config.hidden_size, \
            f"Dimension Mismatch: Vision {self.embed_dim} != Text {self.text_encoder.config.hidden_size}"

        print(f"Final Fused Dim:   {self.embed_dim}")
        print("-" * 20)


        # --- 4. 核心组件 (保持不变) ---
        num_heads = self.embed_dim // 64

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ln_visual = nn.LayerNorm(self.embed_dim)
        self.ln_attn = nn.LayerNorm(self.embed_dim)
        # self.ln_l11 = nn.LayerNorm(self.embed_dim)
        # self.ln_l10 = nn.LayerNorm(self.embed_dim)

        fusion_dim = self.embed_dim * 1

        self.classifier = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(fusion_dim, num_classes)
        )

        if freeze_backbone:
            self._set_freeze_status(True)

    def _set_freeze_status(self, freeze: bool):
        # 1. 冻结所有骨干
        # for param in self.vision_encoder.parameters(): param.requires_grad = False
        # for param in self.visual_proj.parameters(): param.requires_grad = False
        # for param in self.text_encoder.parameters(): param.requires_grad = False
        # if isinstance(self.text_proj, nn.Linear):
        #     for param in self.text_proj.parameters(): param.requires_grad = False

        for param in self.parameters():
            param.requires_grad = False

        # 2. 只训练我们添加的交互层
        trainable = [
            self.cross_attention,
            self.classifier,
            self.ln_attn
        ]
        for module in trainable:
            for param in module.parameters(): param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        print(f"�� [Model Freeze Status] Controlled by Model internally.")
        print(f"   �� Trainable Params: {trainable_params:,} / {all_params:,} ({trainable_params / all_params:.2%})")

    def forward(self, images, input_ids, attention_mask, return_features = False):
        #with torch.no_grad():
        # Vision (L/14)
        vision_outputs = self.vision_encoder(pixel_values=images)
        last_hidden = vision_outputs.last_hidden_state  # [B, 257, 1024]
        last_hidden_norm = self.clip_model.vision_model.post_layernorm(last_hidden)
        feat_768 = self.visual_proj(last_hidden_norm)
        cls_token = last_hidden[:, 0, :]  # [B, 1024]
        # 关键一步：必须经过最后的 LayerNorm (HuggingFace 默认不包含在 projection 里)
        # 在 CLIP 官方代码中，投影前必须过这层 LN
        cls_token_norm = self.clip_model.vision_model.post_layernorm(cls_token)
        # 投影到 768 维 (UFD 最终使用的特征)
        feat_cls_768 = self.visual_proj(cls_token_norm)  # [B, 768]

        # all_hidden = vision_outputs.hidden_states
        # feat_l11_raw = all_hidden[-2][:, 0, :]
        # feat_l10_raw = all_hidden[-3][:, 0, :]

        # Text (CLIP Text Encoder)
        # [修改] CLIPTextModel 的输出和 BERT 略有不同，但也有 last_hidden_state
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_raw = text_outputs.last_hidden_state  # [B, Seq, 768]

        # 如果是 CLIP Text，也需要投影；BERT 不需要
        if self.text_encoder_type == 'clip':
            text_features = self.text_proj(text_raw)
        else:
            text_features = text_raw
        # Projection: 768 -> 1024
        # text_features = self.text_proj(text_raw)

        # Normalization
        # feat_l11_norm = self.ln_l11(feat_l11_raw)
        # feat_l10_norm = self.ln_l10(feat_l10_raw)
        #feat_l12_norm = self.ln_visual(feat_l12_cls)

        #Cross-Attention
        attn_output, _ = self.cross_attention(
            query=feat_768,
            key=text_features,
            value=text_features,
            key_padding_mask=~attention_mask.bool()
        )
        attn_cls = attn_output[:, 0, :]
        attn_norm = self.ln_attn(attn_cls)
        feat_cls_768_norm = self.ln_visual(feat_cls_768)
        conflict_vec = feat_cls_768_norm - attn_norm

        # Fusion
        combined_features = torch.cat([
            #feat_cls_768,
            #attn_norm,
            conflict_vec,
            #feat_l11_norm,
            #feat_l10_norm
        ], dim=1)

        logits = self.classifier(combined_features)

        if return_features:
            return combined_features, logits

        # logits = self.classifier(combined_features)
        return logits