import os
import torch
import torch.nn as nn
from transformers import BertModel, CLIPVisionModel

"""该版本融入了冲突残差，text_features = text_outputs.last_hidden_state，主题Q是文本"""

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
            self.vision_encoder = CLIPVisionModel.from_pretrained(
                vit_path,
                output_hidden_states=True  # 开启隐藏层输出
            )
            print("✅ CLIP-ViT-Patch16 Loaded Successfully")
        except Exception as e:
            # 为了防止你本地路径报错，加个简单的兼容逻辑
            print(f"⚠️ Local path failed, trying HuggingFace default...")
            self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16",
                                                                  output_hidden_states=True)

        # --- 2. 文本编码器 ---
        print(f"Loading BERT from: {bert_path}")
        try:
            if os.path.exists(bert_path):
                self.text_encoder = BertModel.from_pretrained(bert_path)
                print("✅ BERT Loaded Successfully")
            else:
                # 同样做个兼容
                self.text_encoder = BertModel.from_pretrained("bert-base-uncased")
        except:
            self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        # [新增] 维度检查：确保 CLIP 和 BERT 都是 768 维，否则减法没法做
        if self.vision_encoder.config.hidden_size != self.text_encoder.config.hidden_size:
            raise ValueError(f"❌ Dimension Mismatch! CLIP: {self.vision_encoder.config.hidden_size}, BERT: {self.text_encoder.config.hidden_size}")

        self.embed_dim = self.vision_encoder.config.hidden_size  # 通常是 768

        # --- 3. 融合层与分类头 (关键修改区) ---
        #self.embed_dim = 768

        # Cross-Attention:
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=8,
            batch_first=True
        )

        # [新增] 专门为 L11 和 L10 准备的归一化层
        # 既然 L12 自带了 LayerNorm，为了公平竞争，L11/L10 也要过一遍 LayerNorm
        self.ln_l11 = nn.LayerNorm(self.embed_dim)
        self.ln_l10 = nn.LayerNorm(self.embed_dim)
        self.ln_attn = nn.LayerNorm(self.embed_dim)

        # [修改] 输入维度改为 5 层融合 (768 * 5)
        # 1. feat_l12_cls (原始视觉)
        # 2. attn_vec (文本预期的视觉)
        # 3. conflict_vec (冲突残差) <-- 新增核心
        # 4. feat_l11_cls (纹理)
        # 5. feat_l10_cls (浅层)
        fusion_dim = 768 * 1

        # [保持] 分类头结构不变，依然是 MLP + High Dropout
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),   # 为了泛化性能不加更好
            nn.ReLU(),
            nn.Dropout(0.5),

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

        # [关键] 必须解冻所有我们自己添加的层 (包括新的 LayerNorm)
        trainable_modules = [
            self.cross_attention,
            self.classifier,
            self.ln_l11,
            self.ln_l10,
            self.ln_attn
        ]

        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

        print(f"Backbone status: {'FROZEN' if freeze else 'UNFROZEN'}")

    def forward(self, images, input_ids, attention_mask):
        # --- 1. 视觉特征 (保持原逻辑) ---
        vision_outputs = self.vision_encoder(pixel_values=images)

        # [统一 L12]: 使用 last_hidden_state，这是经过 CLIP 内部 Final LayerNorm 的
        # [Batch, 197, 768]
        feat_l12_full_norm = vision_outputs.last_hidden_state

        # 提取 L12 CLS (已归一化)
        feat_l12_cls_norm = feat_l12_full_norm[:, 0, :]

        # [提取 L11, L10]: 从 hidden_states 提取 (这是原始值，未归一化)
        all_hidden_states = vision_outputs.hidden_states
        feat_l11_raw = all_hidden_states[-2][:, 0, :]
        feat_l10_raw = all_hidden_states[-3][:, 0, :]
        # 既然 feat_l12_cls_norm 是归一化的，那么 L11 和 L10 也必须归一化
        # 否则拼接在一起时，数值大的特征会主导，数值小的特征会被忽略
        feat_l11_norm = self.ln_l11(feat_l11_raw)  # Trainable LayerNorm
        feat_l10_norm = self.ln_l10(feat_l10_raw)  # Trainable LayerNorm

        # --- 2. 文本特征提取 (Frozen) ---
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # last_hidden_state表示已经归一化过了

        # --- 3. 跨模态注意力 (逻辑反转) ---
        # [修改] Query = Text, Key/Value = Image
        # 含义：用文本去检索图像中对应的部分
        attn_output, _ = self.cross_attention(
            query=feat_l12_full_norm,  # Q: 文本
            key=text_features,  # K: 图像 L12
            value=text_features,  # V: 图像 L12
            key_padding_mask=~attention_mask.bool() # Mask掉 Padding 的文本
        )
        # [取 CLS] 因为 Q 是 Image，输出也是 Image 序列结构，第 0 个是 CLS
        attn_cls_raw = attn_output[:, 0, :]

        # 对 Attention 输出也做归一化，以便和 L12 做减法
        attn_cls_norm = self.ln_attn(attn_cls_raw)

        # --- 4. 计算冲突残差 ---
        # 现在的减法是公平的：Normed Visual - Normed Text_Explanation
        conflict_vec = feat_l12_cls_norm - attn_cls_norm

        # --- 5. 特征融合 (全归一化拼接) ---
        combined_features = torch.cat([
            feat_l12_cls_norm,  # 1. L12 (Normed)
            #attn_cls_norm,  # 2. Attn (Normed)
            #conflict_vec,  # 3. Conflict
            #feat_l11_norm,  # 4. L11 (Normed!)
            #feat_l10_norm  # 5. L10 (Normed!)
        ], dim=1)

        # --- 6. 分类 ---
        logits = self.classifier(combined_features)

        return logits