import gradio as gr
from PIL import Image
import torch

# 导入 ControlNet 预处理器库中的 OpenPose 提取器
from controlnet_aux import OpenposeDetector

# --- 1. 模型加载与初始化 ---
print("正在加载本地 OpenPose 模型...")
# from_pretrained 默认会从 HuggingFace 下载模型并缓存到本地。
# 如果你已经下载了本地模型（例如 body_pose_model.pth），可以直接将其所在文件夹路径填入此处，例如：
# detector = OpenposeDetector.from_pretrained("/path/to/your/local/models")
detector = OpenposeDetector.from_pretrained("D:/Programs/ComfyUI_windows_portable/ComfyUI/custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators")

# 如果你有 GPU，可以加速处理
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = detector.to(device)
print(f"模型加载完成！正在使用 {device} 运行。")

# --- 2. 核心检测处理函数 ---
def process_image(input_image, include_body, include_hand, include_face):
    """
    处理输入图像并返回骨架图
    """
    if input_image is None:
        return None
        
    try:
        # 调用检测器生成骨架图
        # include_hand 和 include_face 可以控制是否精细检测手部和脸部特征点
        pose_image = detector(
            input_image, 
            include_body=include_body,
            include_hand=include_hand, 
            include_face=include_face
        )
        return pose_image
    except Exception as e:
        print(f"检测出错: {e}")
        return None

# --- 3. Gradio WebUI 前端界面搭建 ---
with gr.Blocks(theme=gr.themes.Soft(), title="骨架检测 WebUI") as demo:
    gr.Markdown(
        """
        # 🏃‍♂️ 细粒度姿态可控 - 本地骨架检测 WebUI
        请在左侧上传包含人物的图像，右侧将实时输出 ControlNet 标准的骨架特征图（OpenPose 格式）。
        """
    )
    
    with gr.Row():
        # 左侧：输入区域
        with gr.Column():
            img_input = gr.Image(type="pil", label="上传原图 (Input Image)")
            
            gr.Markdown("### ⚙️ 检测选项")
            with gr.Row():
                cb_body = gr.Checkbox(label="身体检测 (Body)", value=True, interactive=False)
                cb_hand = gr.Checkbox(label="手部检测 (Hand)", value=False)
                cb_face = gr.Checkbox(label="面部检测 (Face)", value=False)
                
            btn_detect = gr.Button("🔍 开始提取骨架", variant="primary")
            
        # 右侧：输出区域
        with gr.Column():
            img_output = gr.Image(type="pil", label="骨架生成结果 (Pose Result)")

    # 绑定按钮点击事件，连接输入和输出
    btn_detect.click(
        fn=process_image,
        inputs=[img_input, cb_body, cb_hand, cb_face],
        outputs=img_output
    )

    # 添加示例图片（可选，需本地有这些图片才能生效）
    # gr.Examples(
    #     examples=[["example.jpg"]],
    #     inputs=img_input
    # )

if __name__ == "__main__":
    # 启动 Web 服务
    # share=False 表示仅本地网络可访问。如果你想生成公网链接，可以设置 share=True
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)