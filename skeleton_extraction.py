"""只有「脖子以上」的第一步：OpenPose 骨架提取 WebUI。

上传一张人物照片，得到 ControlNet 可直接使用的 OpenPose 骨架图。

环境变量（均可选）：
  ANNOTATOR_DIR 本地注解器权重目录，不设置则从 HuggingFace 下载 lllyasviel/Annotators
  GRADIO_SERVER_NAME / GRADIO_SERVER_PORT / GRADIO_SHARE  启动参数

运行：python skeleton_extraction.py
"""
import os

import gradio as gr
from controlnet_aux import OpenposeDetector
import torch

# --- 1. 模型加载与初始化 ---
print("正在加载 OpenPose 模型...")
annotator_dir = os.environ.get("ANNOTATOR_DIR", "").strip()
detector = OpenposeDetector.from_pretrained(annotator_dir or "lllyasviel/Annotators")
pose_detector = detector

# 有 GPU 时自动使用 GPU 加速
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = detector.to(device)
print(f"模型加载完成！正在使用 {device} 运行。")


# --- 2. 核心检测处理函数 ---
def process_image(input_image, include_body, include_hand, include_face):
    """处理输入图像并返回骨架图。"""
    if input_image is None:
        return None
    try:
        pose_image = detector(
            input_image,
            include_body=include_body,
            include_hand=include_hand,
            include_face=include_face,
        )
        return pose_image
    except Exception as e:  # noqa: BLE001
        print(f"检测出错: {e}")
        return None


# --- 3. Gradio WebUI 前端界面 ---
with gr.Blocks(theme=gr.themes.Soft(), title="骨架检测 WebUI") as demo:
    gr.Markdown(
        """
        # 细粒度姿态可控 - 本地骨架检测 WebUI
        请在左侧上传包含人物的图像，右侧将输出 ControlNet 标准的骨架特征图（OpenPose 格式）。
        """
    )

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="上传原图 (Input Image)")

            gr.Markdown("### 检测选项")
            with gr.Row():
                cb_body = gr.Checkbox(label="身体检测 (Body)", value=True, interactive=False)
                cb_hand = gr.Checkbox(label="手部检测 (Hand)", value=False)
                cb_face = gr.Checkbox(label="面部检测 (Face)", value=False)

            btn_detect = gr.Button("开始提取骨架", variant="primary")

        with gr.Column():
            img_output = gr.Image(type="pil", label="骨架生成结果 (Pose Result)")

    btn_detect.click(
        fn=process_image,
        inputs=[img_input, cb_body, cb_hand, cb_face],
        outputs=img_output,
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        share=os.environ.get("GRADIO_SHARE", "0") == "1",
    )
