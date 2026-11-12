"""分离式后端：为 pose_editor_webui.html 提供 HTTP API。

前端（pose_editor_webui.html）用到的接口：
  POST /api/extract_pose   multipart 上传 file -> {"status":"success","keypoints":[{x,y}...18]}
  POST /api/generate       JSON {prompt, pose_base64} -> {"status":"success","image_url":"..."}

本文件同时把前端页面挂在根路径：启动后直接访问 http://127.0.0.1:8000 即可，
既不用单独打开 HTML，也不会遇到跨域问题。模型为懒加载，首次请求时才下载/加载。

环境变量见 config.py 与 README。
运行：python server_api.py
"""
import base64
import io
import os

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel

import config
import models
from pose_utils import build_target_size, decode_data_url, encode_data_url, keypoints_from_body_estimation

HERE = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Fine-grained Pose-Controllable API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.API_ALLOW_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = ""
    negative_prompt: str = "lowres, bad anatomy, bad hands, worst quality"
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = -1
    pose_base64: str


@app.get("/")
def index():
    """直接托管前端页面，免去跨域和单独开文件。"""
    return FileResponse(os.path.join(HERE, "pose_editor_webui.html"))


@app.get("/api/health")
def health():
    return {"status": "success", **models.status()}


@app.post("/api/extract_pose")
async def extract_pose(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if len(raw) > config.MAX_UPLOAD_BYTES:
            return {
                "status": "error",
                "message": f"图片过大，上限 {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
            }
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        detector = models.get_detector()
        img_bgr = np.array(image)[:, :, ::-1].copy()
        candidate, subset = detector.body_estimation(img_bgr)
        points = keypoints_from_body_estimation(
            candidate, subset, image.width, image.height
        )
        return {"status": "success", "keypoints": points}
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "message": f"骨架提取失败: {exc}"}


@app.post("/api/generate")
def generate(req: GenerateRequest):
    try:
        pose = decode_data_url(req.pose_base64)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"骨架数据不合法: {exc}") from exc

    try:
        pipe = models.get_pipeline()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"生图模型加载失败: {exc}") from exc

    target_w, target_h = build_target_size(
        pose.width, pose.height, config.OUTPUT_SCALE, config.OUTPUT_ALIGN
    )
    pose = pose.resize((target_w, target_h), Image.Resampling.LANCZOS)

    generator = None
    if int(req.seed) != -1:
        generator = torch.Generator(device="cpu").manual_seed(int(req.seed))

    try:
        result = pipe(
            prompt=req.prompt,
            image=pose,
            width=target_w,
            height=target_h,
            negative_prompt=req.negative_prompt,
            num_inference_steps=int(req.num_inference_steps),
            guidance_scale=float(req.guidance_scale),
            generator=generator,
        ).images[0]
        return {"status": "success", "image_url": encode_data_url(result)}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"生成失败: {exc}") from exc


if __name__ == "__main__":
    print(f"-> 服务地址：http://{config.API_HOST}:{config.API_PORT}")
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)