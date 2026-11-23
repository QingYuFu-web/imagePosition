# OmniParser V2 界面控件检测模型

`omniparser-icon-v2.onnx` 由 Microsoft 发布的 [OmniParser V2 `icon_detect/model.pt`](https://huggingface.co/microsoft/OmniParser-v2.0/tree/f55d0750e5b94db2125ef0b45b0fa4a85ddc59b4/icon_detect) 导出。该权重是单类别 YOLO11m 检测器，由原发布方以 AGPL-3.0 授权，具体见 `omniparser-icon-v2-LICENSE.txt`。

- 原始 PyTorch 检查点 SHA-256：`dab3d4351ad00b035db829909a4db98354d5a90f6990e4ac00222a9a95d4bf57`
- 本地 ONNX 文件 SHA-256：`63efb8551358f0e01661e9e0987d0e5ce10140787001dcd7109eff038d47f8e8`
- 导出环境：Python 3.10、PyTorch 2.6.0 CPU、Ultralytics 8.3.79、ONNX 1.17.0。
- 导出调用：`YOLO("model.pt", task="detect").export(format="onnx", imgsz=640, opset=17, dynamic=True, simplify=False, nms=False, half=False, device="cpu", batch=1)`

ONNX 使用动态 NCHW float32 图像输入，预测单类控件框，NMS 在应用中完成。运行时使用 `onnxruntime`，无需加载 PyTorch 检查点。
