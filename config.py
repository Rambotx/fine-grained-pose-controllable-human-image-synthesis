"""集中管理环境变量配置。

本模块刻意不 import torch / diffusers 等重依赖，因此可以被测试直接导入。
"""

import os

# ---------- 模型来源 ----------
# OpenPose 注解器：本地目录优先，否则用 HuggingFace 仓库 id
ANNOTATOR_DIR = os.environ.get("ANNOTATOR_DIR", "").strip()
HF_ANNOTATOR_ID = "lllyasviel/Annotators"

CONTROLNET_ID = os.environ.get(
    "CONTROLNET_ID", "lllyasviel/control_v11p_sd15_openpose"
).strip()

# SD1.5 底模：可以是 HuggingFace 仓库 id，也可以是本地的 .safetensors / .ckpt
BASE_MODEL = os.environ.get(
    "BASE_MODEL", "stable-diffusion-v1-5/stable-diffusion-v1-5"
).strip()

# ---------- Gradio 前端 ----------
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1").strip()
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
GRADIO_SHARE = os.environ.get("GRADIO_SHARE", "0") == "1"

# ---------- FastAPI 分离式后端 ----------
API_HOST = os.environ.get("API_HOST", "127.0.0.1").strip()
API_PORT = int(os.environ.get("API_PORT", "8000"))
API_ALLOW_ORIGINS = [
    origin.strip()
    for origin in os.environ.get("API_ALLOW_ORIGINS", "*").split(",")
    if origin.strip()
]

# 单张图片上传上限（MB -> 字节）
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)

# ---------- 生成参数 ----------
# 生成分辨率相对骨架图的放大倍数，以及宽高需要向下对齐的步长
OUTPUT_SCALE = float(os.environ.get("OUTPUT_SCALE", "1.5"))
OUTPUT_ALIGN = 8
