"""Local YOLO proposals for screen controls, independent of screen layout."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2
import numpy as np


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "omniparser-icon-v2.onnx"


class VisualControlDetector:
    """Run an unmodified, one-class YOLO UI detection head exported to ONNX.

    The detector estimates location and appearance confidence. It does not infer
    the purpose of a control or whether a control is currently enabled.
    """

    name = "YOLO / OmniParser V2"

    def __init__(self, model_path: str | Path | None = None, *, confidence: float = 0.20, iou: float = 0.50):
        model_path = Path(model_path or os.environ.get("SCREEN_YOLO_MODEL") or DEFAULT_MODEL)
        if not model_path.is_file():
            raise RuntimeError(f"未找到 YOLO 界面检测模型：{model_path}。请配置 SCREEN_YOLO_MODEL 指向单类别 UI YOLO ONNX 模型。")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("YOLO 需要 onnxruntime，请先运行 python -m pip install onnxruntime。") from error
        if not 0.0 < confidence < 1.0 or not 0.0 < iou < 1.0:
            raise ValueError("confidence 和 iou 必须在 0 与 1 之间。")

        options = ort.SessionOptions()
        options.intra_op_num_threads = min(4, os.cpu_count() or 1)
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        try:
            self._session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        except Exception as error:
            raise RuntimeError(f"无法载入 YOLO 模型 {model_path}：{error}") from error
        self._input = self._session.get_inputs()[0]
        self._output = self._session.get_outputs()[0].name
        if self._input.type != "tensor(float)" or len(self._input.shape) != 4 or self._input.shape[1] != 3:
            raise RuntimeError("YOLO 模型须使用 NCHW float32 图像输入。")
        batch, _, height, width = self._input.shape
        if isinstance(batch, int) and batch != 1:
            raise RuntimeError("YOLO 模型须支持单张图像推理。")
        self._size = (width, height) if isinstance(width, int) and isinstance(height, int) else None
        self._confidence = confidence
        self._iou = iou

    def detect(self, image_bgr: np.ndarray, *, pixel_budget: int = 640) -> list[dict]:
        """Return candidate polygons in the original input image pixel coordinates."""
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or image_bgr.size == 0:
            raise ValueError("YOLO 需要非空的 BGR 彩色图像。")
        if pixel_budget < 32:
            raise ValueError("YOLO 图像边长预算至少为 32。")
        src_h, src_w = image_bgr.shape[:2]
        if self._size is None:
            # Redistribute the square pixel budget to fit the source aspect
            # ratio. This preserves finer controls on tall phone screenshots.
            budget_scale = min(1.5 * pixel_budget / max(src_w, src_h), math.sqrt(pixel_budget**2 / (src_w * src_h)))
            width, height = (max(32, math.ceil(side * budget_scale / 32) * 32) for side in (src_w, src_h))
        else:
            width, height = self._size
        scale = min(width / src_w, height / src_h)
        scaled_w, scaled_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
        left, top = (width - scaled_w) // 2, (height - scaled_h) // 2
        image = np.full((height, width, 3), 114, np.uint8)
        image[top:top + scaled_h, left:left + scaled_w] = cv2.resize(
            image_bgr, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR,
        )
        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, swapRB=True)
        try:
            output = self._session.run([self._output], {self._input.name: blob})[0]
        except Exception as error:
            raise RuntimeError(f"YOLO 推理失败：{error}") from error
        return self._postprocess(output, (left, top), (scaled_w / src_w, scaled_h / src_h), (src_w, src_h))

    def _postprocess(self, output, offset, scale, size):
        if output.ndim != 3 or output.shape[0] != 1 or 5 not in output.shape[1:]:
            raise RuntimeError(f"YOLO 输出尺寸 {output.shape} 不符合单类别原始检测头（1×5×N）。")
        predictions = output[0].T if output.shape[1] == 5 else output[0]
        predictions = predictions[np.isfinite(predictions).all(axis=1) & (predictions[:, 4] >= self._confidence)]
        if not len(predictions):
            return []

        center = predictions[:, :2]
        half = predictions[:, 2:4] / 2
        lower = ((center - half) - offset) / scale
        upper = ((center + half) - offset) / scale
        lower = np.clip(lower, (0, 0), np.array(size) - 1)
        upper = np.clip(upper, (0, 0), np.array(size) - 1)
        extents = upper - lower
        valid = (extents >= 4).all(axis=1)
        if not valid.any():
            return []
        lower, upper = lower[valid], upper[valid]
        scores = predictions[valid, 4]
        boxes = np.column_stack((lower, extents[valid])).tolist()
        selected = cv2.dnn.NMSBoxes(boxes, scores.tolist(), self._confidence, self._iou, top_k=250)
        result = []
        for index in np.asarray(selected).ravel():
            x0, y0 = np.round(lower[index], 2).tolist()
            x1, y1 = np.round(upper[index], 2).tolist()
            result.append({
                "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                "score": round(float(scores[index]), 4),
                "kind": "control_candidate",
                "source": "yolo",
                "provenance": "YOLO",
            })
        return result
