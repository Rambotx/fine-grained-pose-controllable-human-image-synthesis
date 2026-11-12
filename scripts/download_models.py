"""预先下载本项目需要的全部模型权重（可选，第一次运行 app.py 时也会自动下载）。

用法：
  python scripts/download_models.py            # 下载全部
  python scripts/download_models.py --annotators
  python scripts/download_models.py --controlnet
  python scripts/download_models.py --base

国内网络可以先设置镜像：
  PowerShell : $env:HF_ENDPOINT = "https://hf-mirror.com"
  bash       : export HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download

# 让脚本可以直接 import 仓库根目录的 config.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

# 跳过 fp32 之外的大文件变体，只留推理够用的部分
IGNORE = ["*.onnx", "*.msgpack", "*.h5", "*.tflite", "*.ckpt"]


def _download(repo_id: str, label: str) -> str:
    print(f"-> 下载 {label}: {repo_id}")
    path = snapshot_download(repo_id=repo_id, ignore_patterns=IGNORE)
    print(f"   完成: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="预下载模型权重")
    parser.add_argument("--annotators", action="store_true", help="OpenPose 注解器")
    parser.add_argument("--controlnet", action="store_true", help="ControlNet OpenPose")
    parser.add_argument("--base", action="store_true", help="SD1.5 底模")
    args = parser.parse_args()

    # 不指定任何参数时全部下载
    all_models = not (args.annotators or args.controlnet or args.base)

    if config.ANNOTATOR_DIR:
        print(f"-> ANNOTATOR_DIR 已指定本地目录，跳过 OpenPose 注解器: {config.ANNOTATOR_DIR}")
    elif all_models or args.annotators:
        _download(config.HF_ANNOTATOR_ID, "OpenPose 注解器")

    if os.path.isfile(config.BASE_MODEL):
        print(f"-> BASE_MODEL 是本地权重文件，跳过底模下载: {config.BASE_MODEL}")
    elif all_models or args.base:
        _download(config.BASE_MODEL, "SD1.5 底模")

    if all_models or args.controlnet:
        _download(config.CONTROLNET_ID, "ControlNet OpenPose")

    print("-> 全部完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
