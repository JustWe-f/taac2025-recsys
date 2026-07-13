"""
下载 TAAC2025 TencentGR-1M 数据集（仅下载必要的 5 个子集，约 1.9GB）
使用 hf-mirror.com 镜像，无需代理
用法：python scripts/download_data.py --output_dir D:/taac2025-data
"""

import argparse
import os

# 设置 HuggingFace 镜像（必须在 import huggingface_hub 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

# 只需要这 5 个子集，跳过高维多模态 embedding（82-86，共约 128GB）
NEEDED_PATTERNS = [
    "candidate/*",
    "item_feat/*",
    "seq/*",
    "user_feat/*",
    "mm_emb/emb_81_32_parquet/*",
    "indexer.pkl",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="D:/taac2025-data",
                        help="数据保存路径")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"开始下载数据到: {args.output_dir}")
    print(f"使用镜像: https://hf-mirror.com")
    print(f"仅下载: {NEEDED_PATTERNS}")
    print(f"预计大小: ~1.9 GB")

    snapshot_download(
        repo_id="TAAC2025/TencentGR-1M",
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=NEEDED_PATTERNS,
        resume_download=True,
    )

    print(f"\n下载完成！数据保存在: {args.output_dir}")
    print("目录结构:")
    for root, dirs, files in os.walk(args.output_dir):
        level = root.replace(args.output_dir, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        if level < 2:
            for f in files[:5]:
                size = os.path.getsize(os.path.join(root, f))
                print(f"{indent}  {f} ({size / 1024 / 1024:.1f} MB)")
            if len(files) > 5:
                print(f"{indent}  ... 共 {len(files)} 个文件")

if __name__ == "__main__":
    main()
