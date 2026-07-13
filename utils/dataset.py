import os
import json
import torch
import random
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import BertTokenizer
from transformers import CLIPTokenizer
from transformers import AutoTokenizer
from utils.cnn_aug import CNNAugmentations


class GenImageDataset(Dataset):
    def __init__(self, json_file, tokenizer_path, tokenizer_type='bert', split='train', max_len=150, transform=None):
        # 路径检查
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"❌ Index file not found: {json_file}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"❌ Tokenizer path not found: {tokenizer_path}")

        # 加载数据
        print(f"[{split.upper()}] Loading index from {json_file}...")
        with open(json_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        # 过滤 split
        self.data = [item for item in all_data if item['split'] == split]
        if len(self.data) == 0:
            print(f"Warning: No data found for split '{split}'.")

        self.max_len = max_len
        self.split = split  # <--- 记录当前是 train 还是 val

        # 4. 初始化 Tokenizer,选择bert还是clip
        self.tokenizer_type = tokenizer_type.lower()
        print(f"[{split.upper()}] Initializing {self.tokenizer_type.upper()} Tokenizer...")

        # if self.tokenizer_type == 'bert':
        #     self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path, do_lower_case=True)
        if self.tokenizer_type == 'clip':
            try:
                self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
            except:
                print(
                    f"Local CLIP tokenizer not found at {tokenizer_path}, trying default 'openai/clip-vit-large-patch14'...")
                self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        elif self.tokenizer_type in ['bert', 'roberta', 'deberta']:
            # 使用 AutoTokenizer 兼容所有高级模型
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
            self.max_len = max_len  # 放心用你的 150
        else:
            raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}. Choose 'bert' or 'clip'.")




        # 图像增强策略
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_cnn_detection_transform(split)

    def _get_cnn_detection_transform(self, split):
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

        if split == 'LYK':

            return transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.Lambda(lambda img: CNNAugmentations(p_blur=0.5, p_jpeg=0.5)(img)),
                transforms.RandomCrop(224),

                transforms.RandomHorizontalFlip(p=0.5),

                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            # 验证集：保持纯净，裁减保持一致
            return transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = item['image_path']

        # --- 图像加载 ---
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            # 遇到坏图，随机换一张，防止训练中断
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        if self.transform:
            image = self.transform(image)

        # --- 文本处理  ---

        # 获取 JSON 里的真实描述 (例如: "a man standing next to a lawn mower")
        raw_caption = str(item.get('caption', ""))

        caption = raw_caption  # 默认使用真实描述

        # 训练时的文本增强策略
        # if self.split == 'train':
        #     if random.random() < 0.1:
        #         caption = ""

        # 验证集/测试集：永远使用原始文本 (raw_caption)

        # 编码
        encoding = self.tokenizer(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        label = torch.tensor(item['label'], dtype=torch.long)

        return image, input_ids, attention_mask, label