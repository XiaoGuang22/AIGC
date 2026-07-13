import os
from huggingface_hub import snapshot_download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ================= 配置区域 =================
# 方案 A (7B 旗舰版): "Salesforce/instructblip-vicuna-7b"
# 方案 B (3B 性能版): "Salesforce/instructblip-flan-t5-xl"
repo_id = "Salesforce/instructblip-flan-t5-xl"

# 本地保存路径（建议改写为你服务器上的绝对路径）
local_dir = "/home/liangpeng/LYK/AIGCdetection/weights/instructblip-flan-t5-xl"


# ============================================

def download_model():
    print(f"�� 开始下载权重: {repo_id}")
    print(f"�� 目标路径: {local_dir}")

    try:
        # 使用 snapshot_download 下载完整模型库
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # 禁用符号链接，直接存储文件
            resume_download=True,  # 开启断点续传
            # ignore_patterns=["*.msgpack", "*.h5", "*.ot"], # 忽略非 PyTorch 权重以节省空间
            token=None  # 如果是私有模型需要填写 HF Token
        )
        print(f"✅ 下载完成！请在生成脚本中将 MODEL_PATH 修改为: {local_dir}")
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        print("�� 提示：如果网络不稳定，请尝试设置环境变量: export HF_ENDPOINT=https://hf-mirror.com")


if __name__ == "__main__":
    download_model()