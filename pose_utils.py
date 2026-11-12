"""姿态相关的纯函数工具：不依赖 torch / diffusers，方便单测。

- COCO-18 关键点归一化
- 生成分辨率计算
- base64 / data URL 编解码
"""

import base64
import io
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

# 与编辑器默认 T-pose 一致（x、y 均为 0~1 的相对坐标）
DEFAULT_POSE_NORMALIZED: list[tuple[float, float]] = [
    (0.500, 0.136), (0.500, 0.234), (0.350, 0.234), (0.310, 0.429),
    (0.270, 0.605), (0.640, 0.234), (0.680, 0.429), (0.720, 0.605),
    (0.420, 0.527), (0.390, 0.742), (0.350, 0.937), (0.570, 0.527),
    (0.600, 0.742), (0.640, 0.937), (0.460, 0.117), (0.530, 0.117),
    (0.420, 0.126), (0.570, 0.126),
]

NUM_BODY_KEYPOINTS = 18


def keypoints_from_body_estimation(
    candidate: Iterable[Sequence[float]],
    subset: Iterable[Sequence[float]],
    width: int,
    height: int,
    default_pose: Sequence[tuple[float, float]] = DEFAULT_POSE_NORMALIZED,
) -> list[dict]:
    """把 OpenPose 的 (candidate, subset) 转成 18 个归一化关键点。

    candidate 是 (N, 3) 的像素坐标，subset 第一行是 18 个点在 candidate 中的索引
    （-1 表示该点未检出）。未检出的点用默认 T-pose 的归一化坐标兜底，
    保证前端拿到的永远是 18 个有效坐标。
    """
    if len(subset) == 0:
        raise ValueError("未检测到人体，请换一张人物更清晰的图片")

    candidate = np.asarray(candidate, dtype=float)
    indices = list(subset[0])[:NUM_BODY_KEYPOINTS]
    if len(indices) < NUM_BODY_KEYPOINTS:
        indices += [-1] * (NUM_BODY_KEYPOINTS - len(indices))

    points: list[dict] = []
    for i, raw_index in enumerate(indices):
        index = int(raw_index)
        if index < 0 or index >= len(candidate):
            x, y = default_pose[i]
        else:
            x = float(candidate[index][0]) / width
            y = float(candidate[index][1]) / height
        points.append({"x": round(x, 5), "y": round(y, 5)})
    return points


def to_canvas_points(
    normalized_points: Sequence[dict], width: int, height: int
) -> list[dict]:
    """归一化坐标 -> 画布像素坐标（供前端/调试使用）。"""
    return [
        {"x": round(pt["x"] * width, 2), "y": round(pt["y"] * height, 2)}
        for pt in normalized_points
    ]


def build_target_size(
    width: int, height: int, scale: float = 1.5, align: int = 8
) -> tuple[int, int]:
    """按放大倍数计算生成分辨率，并向下对齐到 align 的整数倍。"""
    if width <= 0 or height <= 0:
        raise ValueError("图片宽高必须为正数")
    target_w = max(align, (int(width * scale) // align) * align)
    target_h = max(align, (int(height * scale) // align) * align)
    return target_w, target_h


def encode_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    """PIL 图像 -> data URL。"""
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_data_url(data_url: str) -> Image.Image:
    """data URL -> RGB 的 PIL 图像。"""
    if not data_url or not data_url.startswith("data:image"):
        raise ValueError("无效的 data URL")
    _header, _, encoded = data_url.partition(",")
    if not encoded:
        raise ValueError("data URL 中缺少 base64 数据")
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
