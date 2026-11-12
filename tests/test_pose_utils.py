"""pose_utils 的单元测试（不依赖 torch / 模型权重）。"""

import numpy as np
import pytest
from PIL import Image

from pose_utils import (
    DEFAULT_POSE_NORMALIZED,
    build_target_size,
    decode_data_url,
    encode_data_url,
    keypoints_from_body_estimation,
    to_canvas_points,
)


def _fake_body_estimation():
    # 18 个点全部检出：第 i 个点放在 (10*i, 20*i) 像素处
    candidate = np.array([[10.0 * i, 20.0 * i, 1.0] for i in range(18)])
    subset = np.array([[float(i) for i in range(18)] + [0.0]])
    return candidate, subset


def test_keypoints_shape_and_normalization():
    candidate, subset = _fake_body_estimation()
    points = keypoints_from_body_estimation(candidate, subset, width=200, height=400)

    assert len(points) == 18
    assert points[0] == {"x": 0.0, "y": 0.0}
    assert points[1] == {"x": 0.05, "y": 0.05}
    assert points[17]["x"] == pytest.approx(170 / 200)


def test_missing_keypoints_fall_back_to_default_pose():
    candidate, subset = _fake_body_estimation()
    subset = subset.copy()
    subset[0][3] = -1
    points = keypoints_from_body_estimation(candidate, subset, width=200, height=400)

    assert points[3] == {
        "x": round(DEFAULT_POSE_NORMALIZED[3][0], 5),
        "y": round(DEFAULT_POSE_NORMALIZED[3][1], 5),
    }


def test_no_person_raises():
    candidate, subset = _fake_body_estimation()
    with pytest.raises(ValueError):
        keypoints_from_body_estimation(candidate, np.empty((0, 18)), 100, 100)


def test_short_subset_is_padded():
    candidate, subset = _fake_body_estimation()
    points = keypoints_from_body_estimation(candidate, subset[:, :5], 100, 100)
    assert len(points) == 18
    assert points[10] == {
        "x": round(DEFAULT_POSE_NORMALIZED[10][0], 5),
        "y": round(DEFAULT_POSE_NORMALIZED[10][1], 5),
    }


def test_to_canvas_points():
    canvas = to_canvas_points([{"x": 0.5, "y": 0.25}], 512, 512)
    assert canvas == [{"x": 256.0, "y": 128.0}]


def test_build_target_size_aligns_to_8():
    assert build_target_size(368, 512) == (552, 768)
    assert build_target_size(100, 100) == (144, 144)


def test_build_target_size_rejects_bad_input():
    with pytest.raises(ValueError):
        build_target_size(0, 100)


def test_data_url_roundtrip():
    image = Image.new("RGB", (12, 20), (10, 20, 30))
    data_url = encode_data_url(image)
    assert data_url.startswith("data:image/png;base64,")
    restored = decode_data_url(data_url)
    assert restored.size == (12, 20)
    assert restored.getpixel((0, 0)) == (10, 20, 30)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode_data_url("not-a-data-url")
