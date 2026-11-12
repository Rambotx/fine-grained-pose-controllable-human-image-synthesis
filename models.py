"""模型加载器：懒加载 + 进程内缓存。

设计要点：
- 只在这里 import torch / controlnet_aux / diffusers 的重依赖，且尽量延迟到真正使用时；
- 因此界面可以先启动，没有权重、甚至预处理库有问题时，也能给出可读的错误而不是起不来。
"""

import threading

import torch

import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

_lock = threading.Lock()
_detector = None
_pipeline = None


def get_detector():
    """OpenPose 关键点检测器（懒加载）。"""
    global _detector
    with _lock:
        if _detector is None:
            from controlnet_aux import OpenposeDetector

            print(f"-> 加载 OpenPose 提取器（{DEVICE}）")
            _detector = OpenposeDetector.from_pretrained(
                config.ANNOTATOR_DIR or config.HF_ANNOTATOR_ID
            ).to(DEVICE)
        return _detector


def _build_pipeline():
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    controlnet = ControlNetModel.from_pretrained(
        config.CONTROLNET_ID, torch_dtype=DTYPE
    )
    base_model = config.BASE_MODEL
    if base_model.lower().endswith((".safetensors", ".ckpt")):
        pipe = StableDiffusionControlNetPipeline.from_single_file(
            base_model,
            controlnet=controlnet,
            torch_dtype=DTYPE,
            use_safetensors=base_model.lower().endswith(".safetensors"),
        )
    else:
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model, controlnet=controlnet, torch_dtype=DTYPE
        )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # CPU offload 仅在 CUDA 下启用，否则 diffusers 会直接报错
    if DEVICE == "cuda":
        pipe.enable_model_cpu_offload()
    return pipe


def get_pipeline():
    """ControlNet + SD1.5 生成管线（懒加载，失败会抛异常）。"""
    global _pipeline
    with _lock:
        if _pipeline is None:
            print(f"-> 加载 ControlNet + Stable Diffusion（{DEVICE}）")
            _pipeline = _build_pipeline()
        return _pipeline


def status() -> dict:
    """当前模型加载状态，供健康检查接口使用。"""
    return {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "detector_loaded": _detector is not None,
        "pipeline_loaded": _pipeline is not None,
        "annotator": config.ANNOTATOR_DIR or config.HF_ANNOTATOR_ID,
        "controlnet": config.CONTROLNET_ID,
        "base_model": config.BASE_MODEL,
    }
