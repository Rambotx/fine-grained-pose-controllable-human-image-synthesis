"""细粒度姿态可控的人体图像生成 —— 一体化 Web 应用

三段式闭环：
  1. 上传照片 -> OpenPose 骨架提取（controlnet_aux）；
  2. 骨架与 18 个关键点坐标注入内置 HTML Canvas 编辑器，拖拽微调姿态；
  3. 编辑器点「导出并回传」后，骨架通过 /api/pose 自动回传后端；点「开始生成」
     直接以它作为 ControlNet 条件图，不再需要手动下载再上传。

服务结构：FastAPI 提供 /api/pose（骨架回传）与 /api/health，Gradio 页面挂在 /。
模型为懒加载，没有权重时界面也能启动，用到时才会下载（见 README）。

环境变量见 config.py 与 README。
运行：python app.py  ->  http://127.0.0.1:7860
"""
import base64
import html
import json
from io import BytesIO

import gradio as gr
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

import config
import models
from pose_utils import build_target_size, decode_data_url, keypoints_from_body_estimation

EDITOR_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <style>
        body { 
            background: #1e1e2e; color: #cdd6f4; 
            display: flex; flex-direction: column; align-items: center; 
            font-family: 'Segoe UI', Tahoma, sans-serif; 
            margin: 0; padding: 15px; overflow: hidden;
        }
        canvas { 
            border: 2px solid #45475a; background: black; 
            margin-top: 5px; cursor: crosshair; border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        }
        .controls-row { 
            margin-top: 15px; display: flex; gap: 10px; align-items: center; justify-content: center; width: 100%;
        }
        button { 
            padding: 8px 15px; font-size: 13px; font-weight: bold; cursor: pointer; 
            background: #89b4fa; color: #1e1e2e; border: none; border-radius: 5px; transition: 0.2s;
        }
        button:hover { filter: brightness(1.1); }
        button.action-btn { background: #a6e3a1; }
        input { 
            padding: 7px; border-radius: 5px; border: 1px solid #585b70; 
            background: #313244; color: white; outline: none; width: 140px;
        }
        .hint {
            text-align: center; font-size: 13px; margin-bottom: 5px; color:#a6adc8;
        }
    </style>
</head>
<body>
    <div class="hint">
        拖拽节点调姿势 → 点「导出并回传」，骨架会直接交给后端生成
    </div>

    <canvas id="poseCanvas" width="__CANVAS_WIDTH__" height="__CANVAS_HEIGHT__"></canvas>
    
    <div id="autoStatus" class="hint" style="min-height:18px;"></div>
    <div class="controls-row">
        <button onclick="resetPose()">重置节点</button>
        <span style="color: #bac2de; font-size: 14px;">保存为:</span>
        <input type="text" id="filenameInput" value="my_custom_pose">
        <span style="color: #bac2de; font-size: 14px;">.png</span>
        <button class="action-btn" onclick="saveImage()">导出并回传</button>
    </div>

    <script>
        const canvas = document.getElementById('poseCanvas');
        const ctx = canvas.getContext('2d');
        const w = canvas.width;
        const h = canvas.height;
        let bgImage = null;
        let isExporting = false;

        const defaultPoints = [
            {x: w*0.5, y: h*0.136}, {x: w*0.5, y: h*0.234}, {x: w*0.35, y: h*0.234}, {x: w*0.31, y: h*0.429}, 
            {x: w*0.27, y: h*0.605}, {x: w*0.64, y: h*0.234}, {x: w*0.68, y: h*0.429}, {x: w*0.72, y: h*0.605}, 
            {x: w*0.42, y: h*0.527}, {x: w*0.39, y: h*0.742}, {x: w*0.35, y: h*0.937}, {x: w*0.57, y: h*0.527}, 
            {x: w*0.6, y: h*0.742}, {x: w*0.64, y: h*0.937}, {x: w*0.46, y: h*0.117}, {x: w*0.53, y: h*0.117},  
            {x: w*0.42, y: h*0.126}, {x: w*0.57, y: h*0.126}
        ];
        
        let points = JSON.parse(JSON.stringify(defaultPoints));

        const injectedPoints = __INJECTED_POINTS__;
        if (injectedPoints && injectedPoints.length === 18) {
            for(let i = 0; i < 18; i++) {
                if (injectedPoints[i].x !== -1 && injectedPoints[i].y !== -1) {
                    points[i].x = injectedPoints[i].x;
                    points[i].y = injectedPoints[i].y;
                }
            }
        }

        const limbs = [
            [1, 2, '#ff0000'], [1, 5, '#ff5500'], [2, 3, '#ffaa00'], [3, 4, '#ffff00'],
            [5, 6, '#aaff00'], [6, 7, '#55ff00'], [1, 8, '#00ff00'], [8, 9, '#00ff55'],
            [9, 10, '#00ffaa'], [1, 11, '#00ffff'], [11, 12, '#00aaff'], [12, 13, '#0055ff'],
            [1, 0, '#0000ff'], [0, 14, '#5500ff'], [14, 16, '#aa00ff'], [0, 15, '#ff00ff'], [15, 17, '#ff00aa']
        ];
        let draggingPoint = null;

        function draw() {
            if (isExporting) {
                ctx.fillStyle = 'black'; ctx.fillRect(0, 0, w, h);
            } else {
                ctx.fillStyle = '#1e1e2e'; ctx.fillRect(0, 0, w, h);
                if (bgImage) {
                    ctx.globalAlpha = 0.5;
                    ctx.drawImage(bgImage, 0, 0, w, h);
                    ctx.globalAlpha = 1.0;
                }
            }

            ctx.lineWidth = Math.max(4, Math.min(w, h) * 0.025); 
            ctx.lineCap = 'round';
            limbs.forEach(limb => {
                const p1 = points[limb[0]]; const p2 = points[limb[1]];
                ctx.strokeStyle = limb[2]; ctx.beginPath();
                ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
            });

            points.forEach((p, index) => {
                ctx.fillStyle = '#ffffff'; ctx.beginPath();
                const radius = index > 13 ? Math.max(2, w*0.008) : Math.max(3, w*0.012);
                ctx.arc(p.x, p.y, radius, 0, Math.PI * 2); ctx.fill();
            });
        }

        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left; const mouseY = e.clientY - rect.top;
            const hitRadiusSq = Math.pow(Math.max(10, w*0.03), 2); 
            for (let i = 0; i < points.length; i++) {
                const dx = points[i].x - mouseX; const dy = points[i].y - mouseY;
                if (dx * dx + dy * dy < hitRadiusSq) { draggingPoint = i; break; }
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            if (draggingPoint !== null) {
                const rect = canvas.getBoundingClientRect();
                points[draggingPoint].x = e.clientX - rect.left; points[draggingPoint].y = e.clientY - rect.top;
                draw();
            }
        });
        canvas.addEventListener('mouseup', () => { draggingPoint = null; });
        canvas.addEventListener('mouseleave', () => { draggingPoint = null; });

        async function saveImage() {
            let filename = document.getElementById('filenameInput').value.trim();
            if (!filename) filename = 'my_custom_pose';
            if (!filename.toLowerCase().endsWith('.png')) filename += '.png';

            isExporting = true; draw();
            const dataUrl = canvas.toDataURL('image/png');
            const link = document.createElement('a');
            link.download = filename; link.href = dataUrl; link.click();
            isExporting = false; draw();

            // 自动回传给后端：省掉「下载 -> 再上传」这一步
            const statusEl = document.getElementById('autoStatus');
            try {
                if (statusEl) statusEl.textContent = '正在回传骨架…';
                const resp = await fetch('/api/pose', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl, filename: filename })
                });
                const data = await resp.json();
                if (statusEl) {
                    statusEl.textContent = data.ok
                        ? '已回传后端（' + filename + '），可在右侧直接点「开始生成」'
                        : '回传失败：' + (data.message || '未知错误');
                }
            } catch (err) {
                if (statusEl) statusEl.textContent = '回传失败（后端未启动？）仍可手动上传导出的 PNG';
            }
        }
        
        function resetPose() { points = JSON.parse(JSON.stringify(defaultPoints)); draw(); }

        const injectedBg = "__BASE64_IMAGE_DATA__";
        if (injectedBg && injectedBg.startsWith("data:image")) {
            const img = new Image();
            img.onload = function() { bgImage = img; draw(); }
            img.src = injectedBg;
        } else {
            draw();
        }
    </script>
</body>
</html>
"""

# 最近一次由编辑器回传的骨架（data URL）
_last_pose = {"data_url": None}


def generate_iframe_html(base64_str="", width=512, height=512, points_json="null"):
    safe_html = EDITOR_HTML_TEMPLATE.replace("__BASE64_IMAGE_DATA__", base64_str)
    safe_html = safe_html.replace("__CANVAS_WIDTH__", str(width))
    safe_html = safe_html.replace("__CANVAS_HEIGHT__", str(height))
    safe_html = safe_html.replace("__INJECTED_POINTS__", points_json)
    escaped_html = html.escape(safe_html)
    return (
        f'<iframe srcdoc="{escaped_html}" width="100%" height="{height + 170}px" '
        'style="border:none; border-radius:12px;"></iframe>'
    )


def extract_pose_and_inject(input_image, include_body, include_hand, include_face):
    """提取骨架，并把检测到的关键点坐标注入编辑器。"""
    if input_image is None:
        return generate_iframe_html("", 512, 512, "null")
    try:
        w, h = input_image.size
        if w > h:
            new_w, new_h = 512, max(1, int(h * (512 / w)))
        else:
            new_h, new_w = 512, max(1, int(w * (512 / h)))

        detector = models.get_detector()
        pose_image = detector(
            input_image,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
        )

        buffered = BytesIO()
        pose_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"

        points_json = "null"
        try:
            # 调用检测器内部的身体评估模块拿到 COCO-18 像素坐标
            img_bgr = np.array(input_image.convert("RGB"))[:, :, ::-1].copy()
            candidate, subset = detector.body_estimation(img_bgr)
            points = keypoints_from_body_estimation(candidate, subset, w, h)
            # 画布按原图比例缩放到 512，坐标同步换算
            canvas_points = [
                {"x": round(pt["x"] * new_w, 2), "y": round(pt["y"] * new_h, 2)}
                for pt in points
            ]
            points_json = json.dumps(canvas_points)
            print("-> 已把检测到的关键点注入到编辑器")
        except Exception as exc:  # noqa: BLE001
            print(f"-> 关键点注入失败，前端退回默认 T-pose：{exc}")
            points_json = "null"

        return generate_iframe_html(data_url, new_w, new_h, points_json)
    except Exception as exc:  # noqa: BLE001
        print(f"骨架提取出错: {exc}")
        return generate_iframe_html("", 512, 512, "null")


def _resolve_pose_source(use_editor_pose, final_pose_image):
    """优先用编辑器回传的骨架，其次用手动上传的骨架。"""
    if use_editor_pose and _last_pose["data_url"]:
        return decode_data_url(_last_pose["data_url"])
    if final_pose_image is not None:
        return final_pose_image.convert("RGB")
    return None


def generate_image(
    use_editor_pose,
    final_pose_image,
    prompt,
    negative_prompt,
    num_steps,
    guidance_scale,
    seed,
):
    pose = _resolve_pose_source(use_editor_pose, final_pose_image)
    if pose is None:
        raise gr.Error("没有可用的骨架：请先在编辑器里点「导出并回传」，或手动上传导出的 PNG。")
    if not (prompt or "").strip():
        raise gr.Error("请填写正向提示词。")

    try:
        pipe = models.get_pipeline()
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"生图模型加载失败：{exc}") from exc

    target_w, target_h = build_target_size(
        pose.width, pose.height, config.OUTPUT_SCALE, config.OUTPUT_ALIGN
    )
    highres_pose = pose.resize((target_w, target_h), Image.Resampling.LANCZOS)
    print(f"-> 生成分辨率: {target_w} x {target_h}")

    generator = None
    if int(seed) != -1:
        generator = torch.Generator(device="cpu").manual_seed(int(seed))

    try:
        output = pipe(
            prompt=prompt,
            image=highres_pose,
            width=target_w,
            height=target_h,
            negative_prompt=negative_prompt,
            num_inference_steps=int(num_steps),
            guidance_scale=float(guidance_scale),
            generator=generator,
        ).images[0]
        print("-> 生成完毕")
        return output
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"生成失败: {exc}") from exc


with gr.Blocks(theme=gr.themes.Soft(), title="细粒度姿态可控的人体图像生成") as demo:
    gr.Markdown("# 细粒度姿态可控的人体图像生成")
    gr.Markdown(
        "上传照片 → 提取骨架 → 拖拽微调 → 导出并回传 → 一键生成。"
        "（模型首次使用时会自动下载，约 5GB）"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 第一步：提取骨架")
            img_input = gr.Image(type="pil", label="上传原图")
            with gr.Row():
                cb_body = gr.Checkbox(label="身体", value=True, interactive=False)
                cb_hand = gr.Checkbox(label="手部", value=True)
                cb_face = gr.Checkbox(label="面部", value=False)
            btn_extract = gr.Button("提取骨架", variant="secondary")

        with gr.Column(scale=2):
            gr.Markdown("### 第二步：拖拽微调")
            gr.Markdown("调整完点「导出并回传」，骨架会直接交给后端，无需手动上传。")
            editor_html = gr.HTML(value=generate_iframe_html("", 512, 512, "null"))

        with gr.Column(scale=1):
            gr.Markdown("### 第三步：生成新图像")
            use_editor_pose = gr.Checkbox(
                label="使用编辑器最新回传的骨架", value=True
            )
            final_pose_input = gr.Image(
                type="pil", label="或手动上传骨架 PNG（可选）", height=160
            )

            prompt = gr.Textbox(label="正向提示词", lines=2)
            neg_prompt = gr.Textbox(
                label="反向提示词",
                value=(
                    "ng_deepnegative_v1_75t, EasyNegative, bad-hands-5, "
                    "(worst quality:2), (low quality:2), (normal quality:2), lowres, "
                    "bad anatomy, bad face, poorly drawn face, mutation, deformed"
                ),
                lines=2,
            )
            with gr.Row():
                steps = gr.Slider(label="迭代步数", minimum=10, maximum=50, value=20, step=1)
                cfg_scale = gr.Slider(
                    label="引导系数", minimum=1.0, maximum=15.0, value=7.5, step=0.5
                )
            seed = gr.Number(label="随机种子 (-1 随机)", value=-1, precision=0)
            btn_generate = gr.Button("开始生成", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 最终生成结果")
            final_output = gr.Image(type="pil", label="生成图像")

    btn_extract.click(
        fn=extract_pose_and_inject,
        inputs=[img_input, cb_body, cb_hand, cb_face],
        outputs=editor_html,
    )
    btn_generate.click(
        fn=generate_image,
        inputs=[
            use_editor_pose,
            final_pose_input,
            prompt,
            neg_prompt,
            steps,
            cfg_scale,
            seed,
        ],
        outputs=final_output,
    )


# ---------------- FastAPI：骨架回传接口 + 挂载 Gradio ----------------
api = FastAPI(title="Fine-grained Pose-Controllable Human Image Synthesis")


@api.post("/api/pose")
async def receive_pose(request: Request):
    """接收编辑器回传的骨架图（data URL）。"""
    try:
        payload = await request.json()
    except Exception:  # noqa: BLE001
        return JSONResponse({"ok": False, "message": "请求体不是合法 JSON"}, status_code=400)

    data_url = (payload or {}).get("image", "")
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        return JSONResponse(
            {"ok": False, "message": "缺少 data:image 骨架数据"}, status_code=400
        )
    if len(data_url) > 8 * 1024 * 1024:
        return JSONResponse(
            {"ok": False, "message": "骨架图过大"}, status_code=413
        )
    _last_pose["data_url"] = data_url
    return {"ok": True, "message": "骨架已回传"}


@api.get("/api/health")
def health():
    return {"ok": True, **models.status()}


app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    print(f"-> 服务地址：http://{config.GRADIO_SERVER_NAME}:{config.GRADIO_SERVER_PORT}")
    uvicorn.run(app, host=config.GRADIO_SERVER_NAME, port=config.GRADIO_SERVER_PORT)