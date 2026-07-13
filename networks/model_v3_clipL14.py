import torch
import torch.nn as nn
# [修改] 导入 CLIPTextModel
from transformers import CLIPVisionModel, CLIPTextModel, BertModel

"""该版本使用CLIP-L/14这种更大的模型作为视觉编码器，文本编码器也使用CLIP原生"""

class CrossModalDetector(nn.Module):
    def __init__(self,
                 vit_path,  # 只需要这一个路径了
                 bert_path=None,  # 为了兼容接口保留，但实际不用了
                 num_classes=2,
                 freeze_backbone=True,
                 text_encoder_type='clip'):
        super(CrossModalDetector, self).__init__()

        # --- 1. 加载 CLIP Vision (L/14) ---
        print(f"Loading CLIP-ViT (Large) from: {vit_path}")
        try:
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vit_path,
                output_hidden_states=True
            )
            print("✅ CLIP-ViT-Patch14 Loaded Successfully")
        except:
            print("⚠️ Local path not found, downloading openai/clip-vit-large-patch14...")
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14",
                                                                  output_hidden_states=True)

        # --- 2. 动态加载 Text Encoder ---
        self.text_encoder_type = text_encoder_type.lower()
        print(f"Loading Text Encoder ({self.text_encoder_type.upper()})...")

        if self.text_encoder_type == 'clip':
            # 加载 CLIP Text Encoder (通常和 Vision 在同一个路径)
            try:
                self.text_encoder = CLIPTextModel.from_pretrained(vit_path)
                print(f"✅ CLIPTextModel loaded from {vit_path}")
            except:
                print("⚠️ Local path failed, downloading openai/clip-vit-large-patch14...")
                self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

        elif self.text_encoder_type == 'bert':
            # 加载 BERT
            if bert_path is None:
                bert_path = "bert-base-uncased"  # 默认值
            try:
                self.text_encoder = BertModel.from_pretrained(bert_path)
                print(f"✅ BertModel loaded from {bert_path}")
            except:
                print("⚠️ Local path failed, downloading bert-base-uncased...")
                self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        else:
            raise ValueError(f"❌ Unsupported text_encoder_type: {text_encoder_type}")

        # --- 3. 维度对齐 (Projection) ---
        self.vis_dim = self.vision_encoder.config.hidden_size  # 1024
        self.text_dim = self.text_encoder.config.hidden_size  # 768 (L14的文本层比较小)

        print(f">>> Dimensions - Vision: {self.vis_dim} | Text: {self.text_dim}")

        # [保持] 即使是原生 CLIP，L/14 的图文维度也不一样，必须投影
        if self.vis_dim != self.text_dim:
            self.text_proj = nn.Linear(self.text_dim, self.vis_dim)
            print(f">>> Added Projection Layer: {self.text_dim} -> {self.vis_dim}")
        else:
            self.text_proj = nn.Identity()

        self.embed_dim = self.vis_dim

        # --- 4. 核心组件 (保持不变) ---

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=16,
            batch_first=True
        )

        self.ln_visual = nn.LayerNorm(self.embed_dim)
        self.ln_attn = nn.LayerNorm(self.embed_dim)
        self.ln_l11 = nn.LayerNorm(self.embed_dim)
        self.ln_l10 = nn.LayerNorm(self.embed_dim)

        fusion_dim = self.embed_dim * 2

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

        if freeze_backbone:
            self._set_freeze_status(True)

    def _set_freeze_status(self, freeze: bool):
        for param in self.vision_encoder.parameters(): param.requires_grad = not freeze
        for param in self.text_encoder.parameters(): param.requires_grad = not freeze

        trainable = [
            self.cross_attention, self.classifier,
            self.ln_visual, self.ln_attn, self.ln_l11, self.ln_l10,
            self.text_proj  # 别忘了这个
        ]
        for module in trainable:
            for param in module.parameters(): param.requires_grad = True

    def forward(self, images, input_ids, attention_mask):
        with torch.no_grad():
            # Vision (L/14)
            vision_outputs = self.vision_encoder(pixel_values=images)
            feat_l12_full = vision_outputs.last_hidden_state  # [B, 257, 1024],已经归一化
            feat_l12_cls = feat_l12_full[:, 0, :]

            all_hidden = vision_outputs.hidden_states
            feat_l11_raw = all_hidden[-2][:, 0, :]
            feat_l10_raw = all_hidden[-3][:, 0, :]

            # Text (CLIP Text Encoder)
            # [修改] CLIPTextModel 的输出和 BERT 略有不同，但也有 last_hidden_state
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_raw = text_outputs.last_hidden_state  # [B, Seq, 768]

        # Projection: 768 -> 1024
        text_features = self.text_proj(text_raw)

        # Normalization
        feat_l11_norm = self.ln_l11(feat_l11_raw)
        feat_l10_norm = self.ln_l10(feat_l10_raw)
        feat_l12_norm = self.ln_visual(feat_l12_cls)

        # Cross-Attention
        attn_output, _ = self.cross_attention(
            query=feat_l12_full,
            key=text_features,
            value=text_features,
            key_padding_mask=~attention_mask.bool()
        )
        attn_cls = attn_output[:, 0, :]
        attn_norm = self.ln_attn(attn_cls)

        conflict_vec = feat_l12_norm - attn_norm

        # Fusion
        combined_features = torch.cat([
            #feat_l12_norm,
            attn_norm,
            conflict_vec,
            #feat_l11_norm,
            #feat_l10_norm
        ], dim=1)

        logits = self.classifier(combined_features)
        return logits