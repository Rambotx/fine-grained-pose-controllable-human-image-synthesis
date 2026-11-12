# legacy —— 历史版本（不维护）

这里保留的是开发过程中产生的旧入口，逻辑已被仓库根目录的新入口取代。
**这些脚本仍硬编码了作者本机的 ComfyUI 模型路径，换机器不能直接跑**，
仅作为版本对照留档；要看当前实现请回到仓库根目录。

| 文件 | 原来是什么 | 现在对应 |
| --- | --- | --- |
| `pipeline_original.py` | 最早的「一体化 Gradio」脚本，含内嵌 HTML 骨架编辑器，但模型路径写死在本机 | `app.py`（已改为环境变量配置） |
| `backend_api_v1.py` | 第一版 Gradio 后端（调用 `runwayml/stable-diffusion-v1-5` 仓库，路径写死） | `app.py` |
| `backend_api_v2.py` | 第二版 Gradio 后端（在 v1 基础上加了坐标注入的前置逻辑雏形） | `app.py` |
| `skeleton_extraction_only.py` | 只做骨架提取的最小 WebUI（路径写死） | `skeleton_extraction.py` |
