import torch
import torch.nn as nn
# [修改] 导入 CLIPTextModel
from transformers import CLIPModel, AutoModel, BertModel
import torch.nn.functional as F  # 确保在文件开头导入了 F

"""该版本使用CLIP-L/14这种更大的模型作为视觉编码器，文本编码器也使用CLIP原生"""

class CrossModalDetector(nn.Module):
    def __init__(self,
                 vit_path,
                 text_model_path=None,
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


        elif self.text_encoder_type in ['bert', 'roberta', 'deberta']:
            if text_model_path is None:
                print("ERROR")
            self.text_encoder = AutoModel.from_pretrained(text_model_path)
            self.text_proj = nn.Identity()
            print(f"✅ AutoModel loaded from {text_model_path}")
        else:
            raise ValueError(f"Unsupported text_encoder_type: {self.text_encoder_type}")
        # elif self.text_encoder_type == 'bert':
        #     if bert_path is None: bert_path = "bert-base-uncased"
        #     self.text_encoder = BertModel.from_pretrained(bert_path)
        #     self.text_proj = nn.Identity()  # BERT 不需要投影，原生就是 768
        #     print(f"✅ BertModel loaded from {bert_path}")


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

        fusion_dim = self.embed_dim * 4
        hidden_dim = self.embed_dim

        # 构建非线性 MLP 大脑
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),  # 第一步：将 3072 维的高维线索压缩提炼
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(fusion_dim, num_classes)
        # )

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
            self.ln_attn,
            self.ln_visual
        ]
        for module in trainable:
            for param in module.parameters(): param.requires_grad = True

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in self.parameters())
        print(f"�� [Model Freeze Status] Controlled by Model internally.")
        print(f"   �� Trainable Params: {trainable_params:,} / {all_params:,} ({trainable_params / all_params:.2%})")

    def forward(self, images, input_ids, attention_mask):
        # ==========================================
        # 1. 视觉特征提取
        # ==========================================
        with torch.no_grad():
            vision_outputs = self.vision_encoder(pixel_values=images)
            last_hidden = vision_outputs.last_hidden_state  # [B, 257, 1024]

            # 对全局进行 LN 和 Projection
            last_hidden_norm = self.clip_model.vision_model.post_layernorm(last_hidden)
            feat_768 = self.visual_proj(last_hidden_norm)  # [B, 257, 768]

            feat_cls_768 = feat_768[:, 0, :]  # [B, 768]

        # ==========================================
        # 2. 文本特征提取
        # ==========================================
        with torch.no_grad():
            text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            text_raw = text_outputs.last_hidden_state  # [B, Seq, 768]

            if self.text_encoder_type == 'clip':
                text_features = self.text_proj(text_raw)
            else:
                text_features = text_raw

        # ==========================================
        # 3. 跨模态交互 (Cross-Attention)
        # ==========================================
        attn_output, _ = self.cross_attention(
            query=feat_768,
            key=text_features,
            value=text_features,
            key_padding_mask=~attention_mask.bool()
        )
        attn_cls = attn_output[:, 0, :]

        # ==========================================
        # 4. 非线性 NLI 启发式融合
        # ==========================================
        V_ln = self.ln_visual(feat_cls_768)
        A_ln = self.ln_attn(attn_cls)

        #diff = torch.abs(V_ln - A_ln)
        diff = V_ln - A_ln

        mult = V_ln * A_ln

        # 3. 全量拼接
        # 维度变成 [B, 768 * 4] = [B, 3072]
        combined_features = torch.cat([V_ln, A_ln, diff, mult], dim=-1)

        # 传入带有 GELU 的非线性分类器
        logits = self.classifier(combined_features)
        return logits, V_ln, A_ln