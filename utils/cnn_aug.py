import random
import cv2
import numpy as np
import io
from PIL import Image
from scipy.ndimage import gaussian_filter


# ================= CNNDetection / UFD 核心增强逻辑 =================

def apply_gaussian_blur(img_np, sigma):
    """
    复刻 UFD 源码：直接对 Numpy 数组的三个通道分别进行高斯滤波
    """
    # UFD 源码：gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    # 注意：这会直接在原数组上修改
    for i in range(3):
        gaussian_filter(img_np[:, :, i], output=img_np[:, :, i], sigma=sigma)
    return img_np


def cv2_jpg(img_np, compress_val):
    """使用 OpenCV 引擎进行 JPEG 压缩"""
    # RGB 转 BGR (OpenCV 标准)
    img_cv2 = img_np[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    _, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    # BGR 转回 RGB
    return decimg[:, :, ::-1]


def pil_jpg(img_np, compress_val):
    """使用 PIL 引擎进行 JPEG 压缩"""
    img_pil = Image.fromarray(img_np)
    out = io.BytesIO()
    img_pil.save(out, format='jpeg', quality=compress_val)
    out.seek(0)
    return np.array(Image.open(out))


class CNNAugmentations:
    def __init__(self, p_blur=0.5, p_jpeg=0.5):
        self.p_blur = p_blur
        self.p_jpeg = p_jpeg
        # UFD 默认参数映射
        self.blur_sig = [0.0, 3.0]
        self.jpg_qual = [30, 100]
        self.jpg_method = ['cv2', 'pil']

    def __call__(self, img_pil):
        # 1. 转换为 Numpy 数组进行处理 (UFD 特色)
        img_np = np.array(img_pil)

        # 2. 高斯模糊 (Gaussian Blur)
        if random.random() < self.p_blur:
            # 采样 sigma
            sig = random.uniform(self.blur_sig[0], self.blur_sig[1])
            img_np = apply_gaussian_blur(img_np, sig)

        # 3. JPEG 压缩 (JPEG Compression)
        if random.random() < self.p_jpeg:
            # 随机采样质量和压缩引擎
            method = random.choice(self.jpg_method)
            qual = int(random.uniform(self.jpg_qual[0], self.jpg_qual[1]))

            if method == 'cv2':
                img_np = cv2_jpg(img_np, qual)
            else:
                img_np = pil_jpg(img_np, qual)

        # 4. 转回 PIL Image 给后续的 transforms 使用
        return Image.fromarray(img_np)