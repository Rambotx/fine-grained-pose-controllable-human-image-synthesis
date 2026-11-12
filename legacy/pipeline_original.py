import gradio as gr
from PIL import Image
import torch
import numpy as np
import base64
from io import BytesIO
import html
import json
import cv2

from controlnet_aux import OpenposeDetector

from diffusers import (
    StableDiffusionControlNetPipeline, 
    ControlNetModel, 
    UniPCMultistepScheduler
)

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
        拖拽节点进行修改，完成后输入文件名导出
    </div>

    <canvas id="poseCanvas" width="__CANVAS_WIDTH__" height="__CANVAS_HEIGHT__"></canvas>
    
    <div class="controls-row">
        <button onclick="resetPose()">重置节点</button>
        <span style="color: #bac2de; font-size: 14px;">保存为:</span>
        <input type="text" id="filenameInput" value="my_custom_pose">
        <span style="color: #bac2de; font-size: 14px;">.png</span>
        <button class="action-btn" onclick="saveImage()">导出骨架</button>
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

        function saveImage() {
            let filename = document.getElementById('filenameInput').value.trim();
            if (!filename) filename = 'my_custom_pose';
            if (!filename.toLowerCase().endsWith('.png')) filename += '.png';
            
            isExporting = true; draw(); 
            const link = document.createElement('a');
            link.download = filename; link.href = canvas.toDataURL('image/png'); link.click();
            isExporting = false; draw(); 
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


def generate_iframe_html(base64_str="", width=512, height=512, points_json="null"):
    safe_html = EDITOR_HTML_TEMPLATE.replace("__BASE64_IMAGE_DATA__", base64_str)
    safe_html = safe_html.replace("__CANVAS_WIDTH__", str(width))
    safe_html = safe_html.replace("__CANVAS_HEIGHT__", str(height))
    safe_html = safe_html.replace("__INJECTED_POINTS__", points_json)
    escaped_html = html.escape(safe_html)
    return f'<iframe srcdoc="{escaped_html}" width="100%" height="{height + 150}px" style="border:none; border-radius:12px;"></iframe>'

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("-> 加载 OpenPose 提取器")
annotator_path = "D:/Programs/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators"
pose_detector = OpenposeDetector.from_pretrained(annotator_path)
pose_detector = pose_detector.to(device)

print("-> 加载 ControlNet 和 Stable Diffusion 管道")
controlnet_id = "lllyasviel/control_v11p_sd15_openpose"
base_model_path = "D:/Programs/ComfyUI_windows_portable/ComfyUI/models/checkpoints/majicmixRealistic_v7.safetensors"

try:
    controlnet = ControlNetModel.from_pretrained(controlnet_id, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_single_file(
        base_model_path, 
        controlnet=controlnet, 
        torch_dtype=dtype,
        use_safetensors=True
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    print("-> 模型加载完成")
except Exception as e:
    print(f"-> 生图模型加载失败，错误: {e}")
    pipe = None


def extract_pose_and_inject(input_image, include_body, include_hand, include_face):
    if input_image is None:
        return generate_iframe_html("", 512, 512, "null")
    try:
        w, h = input_image.size
        if w > h:
            new_w = 512
            new_h = int(h * (512 / w))
        else:
            new_h = 512
            new_w = int(w * (512 / h))

        pose_image = pose_detector(
            input_image, 
            include_body=include_body,
            include_hand=include_hand, 
            include_face=include_face
        )
        
        buffered = BytesIO()
        pose_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        data_url = f"data:image/png;base64,{img_str}"
        
        points_json = "null"
        try:
            #转换为底层的numpy/BGR格式
            img_np = np.array(input_image)
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_bgr = img_np[:, :, ::-1].copy()
            else:
                img_bgr = img_np
                
            # 调用 pose_detector 内部的身体评估模块
            candidate, subset = pose_detector.body_estimation(img_bgr)
            
            # 如果成功识别出人体
            if len(subset) > 0:
                person_indices = subset[0][:18] # 取画面中第一个人的18个关键点索引
                detected_points = []
                
                for i in range(18):
                    idx = int(person_indices[i])
                    if idx == -1:
                        detected_points.append({"x": -1, "y": -1})
                    else:
                        x_orig = float(candidate[idx][0])
                        y_orig = float(candidate[idx][1])
                        x_canvas = (x_orig / w) * new_w
                        y_canvas = (y_orig / h) * new_h
                        
                        detected_points.append({"x": round(x_canvas, 2), "y": round(y_canvas, 2)})
                        
                points_json = json.dumps(detected_points)
                print("-> 成功将识别坐标注入到编辑器！")
        except Exception as e:
            print(f"-> 节点坐标底层提取失败，前端将降级使用默认T-pose。原因: {e}")
            points_json = "null"
        
        return generate_iframe_html(data_url, new_w, new_h, points_json)
    except Exception as e:
        print(f"骨架提取出错: {e}")
        return generate_iframe_html("", 512, 512, "null")


def generate_image(final_pose_image, prompt, negative_prompt, num_steps, guidance_scale, seed):
    if final_pose_image is None:
        raise gr.Error("请先在骨架编辑器中上传微调后的骨架图")
    if pipe is None:
        raise gr.Error("生图模型未成功加载")

    final_pose_image = final_pose_image.convert("RGB")

    try:
        generator = torch.Generator(device="cpu").manual_seed(seed) if seed != -1 else None
        
        print("-> 计算并放大生成分辨率")
        target_w = int(final_pose_image.width * 1.5)
        target_h = int(final_pose_image.height * 1.5)
        
        target_w = (target_w // 8) * 8
        target_h = (target_h // 8) * 8
        
        highres_pose = final_pose_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        print(f"-> 目标分辨率: {target_w} x {target_h}")
        
        output = pipe(
            prompt=prompt, 
            image=highres_pose,  
            width=target_w,      
            height=target_h,     
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps, 
            guidance_scale=guidance_scale, 
            generator=generator,
        ).images[0]
        
        print("-> 生成完毕")
        return output
    except Exception as e:
        raise gr.Error(f"生成失败: {str(e)}")


with gr.Blocks(theme=gr.themes.Soft(), title="细粒度姿态可控的人体图像生成") as demo:
    gr.Markdown("# 细粒度姿态可控的人体图像生成")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 提取骨架")
            img_input = gr.Image(type="pil", label="上传原图")
            with gr.Row():
                cb_body = gr.Checkbox(label="身体", value=True, interactive=False)
                cb_hand = gr.Checkbox(label="手部", value=True)
                cb_face = gr.Checkbox(label="面部", value=False)
            btn_extract = gr.Button("提取骨架", variant="secondary")
            
        with gr.Column(scale=1):
            gr.Markdown("### 骨架编辑器")
            editor_html = gr.HTML(value=generate_iframe_html("", 512, 512, "null"))
            
        with gr.Column(scale=1):
            gr.Markdown("### 生成新图像")
            gr.Markdown("将导出的图片拖入下方：")
            
            final_pose_input = gr.Image(type="pil", label="上传 my_custom_pose.png", height=200)
            
            prompt = gr.Textbox(label="正向提示词", lines=2)
            neg_prompt = gr.Textbox(
                label="反向提示词", 
                value="ng_deepnegative_v1_75t, EasyNegative, bad-hands-5, (worst quality:2), (low quality:2), (normal quality:2), lowres, bad anatomy, bad face, poorly drawn face, mutation, deformed", 
                lines=2
            )
            with gr.Row():
                steps = gr.Slider(label="迭代步数", minimum=10, maximum=50, value=20, step=1)
                cfg_scale = gr.Slider(label="引导系数", minimum=1.0, maximum=15.0, value=7.5, step=0.5)
            seed = gr.Number(label="随机种子 (-1随机)", value=-1, precision=0)
            btn_generate = gr.Button("开始生成", variant="primary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 最终生成结果")
            final_output = gr.Image(type="pil", label="生成图像")

    btn_extract.click(
        fn=extract_pose_and_inject,
        inputs=[img_input, cb_body, cb_hand, cb_face],
        outputs=editor_html
    )
    
    btn_generate.click(
        fn=generate_image,
        inputs=[final_pose_input, prompt, neg_prompt, steps, cfg_scale, seed],
        outputs=final_output
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)