"""config 的单元测试：环境变量解析。"""

import importlib

import config


def _reload_with(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config)


def test_defaults(monkeypatch):
    for key in (
        "ANNOTATOR_DIR",
        "CONTROLNET_ID",
        "BASE_MODEL",
        "GRADIO_SERVER_PORT",
        "API_PORT",
        "API_ALLOW_ORIGINS",
        "MAX_UPLOAD_MB",
    ):
        monkeypatch.delenv(key, raising=False)
    cfg = importlib.reload(config)
    assert cfg.ANNOTATOR_DIR == ""
    assert cfg.CONTROLNET_ID == "lllyasviel/control_v11p_sd15_openpose"
    assert cfg.BASE_MODEL == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert cfg.GRADIO_SERVER_PORT == 7860
    assert cfg.API_PORT == 8000
    assert cfg.MAX_UPLOAD_BYTES == 10 * 1024 * 1024


def test_env_overrides(monkeypatch):
    cfg = _reload_with(
        monkeypatch,
        ANNOTATOR_DIR="/models/annotators",
        BASE_MODEL="/models/majicmix.safetensors",
        API_PORT="9000",
        API_ALLOW_ORIGINS="http://localhost:3000, http://127.0.0.1:5173",
        MAX_UPLOAD_MB="4",
    )
    assert cfg.ANNOTATOR_DIR == "/models/annotators"
    assert cfg.BASE_MODEL == "/models/majicmix.safetensors"
    assert cfg.API_PORT == 9000
    assert cfg.API_ALLOW_ORIGINS == [
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]
    assert cfg.MAX_UPLOAD_BYTES == 4 * 1024 * 1024
