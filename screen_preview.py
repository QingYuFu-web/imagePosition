"""Interactive preview of identified touch targets on the source photograph."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable

import cv2
import numpy as np
from PIL import Image, ImageTk


class ScreenPreview(tk.Canvas):
    """Keep the photograph and detection polygons aligned while it is resized."""

    def __init__(self, parent, on_select: Callable[[int], None], *, empty_text="打开照片以预览识别目标",
                 empty_color="#AAB9CA", image_border="#34445B"):
        super().__init__(parent, background="#111B2C", highlightthickness=0)
        self._on_select = on_select
        self._empty_text, self._empty_color, self._image_border = empty_text, empty_color, image_border
        self._source = None
        self._photo = None
        self._points = []
        self._polygons = []
        self._label_rectangles = {}
        self._selected = None
        self._hover = None
        self._marker = None
        self._action_labels = False
        self._image_offset = (0, 0)
        self._image_scale = (1, 1)
        self._boxes_visible = True
        self._details_visible = True
        self._resize_job = None
        self.bind("<Configure>", self._schedule_resize)
        self.bind("<Motion>", self._move)
        self.bind("<Leave>", self._leave)
        self.bind("<Button-1>", self._click)

    def set_image(self, image_bgr: np.ndarray | None, points: list[dict]) -> None:
        self._reset_points(points, action_labels=False)
        self.set_raster(image_bgr)

    def set_raster(self, image_bgr: np.ndarray | None) -> None:
        self._source = (
            Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
            if image_bgr is not None else None
        )
        self._render()

    def _reset_points(self, points, *, action_labels):
        self._points = list(points)
        self._action_labels = action_labels
        self._selected = None
        self._hover = None
        self._marker = None
        self.configure(cursor="")

    def set_points(self, points: list[dict], *, action_labels: bool = False) -> None:
        self._reset_points(points, action_labels=action_labels)
        self._project_points()
        self._draw_boxes()

    def select(self, index: int | None) -> None:
        if index is not None and not 0 <= index < len(self._points):
            return
        if index != self._selected:
            self._selected = index
            self._draw_boxes()

    def set_boxes_visible(self, visible: bool) -> None:
        self._boxes_visible = bool(visible)
        if not self._boxes_visible:
            self._hover = None
            self.configure(cursor="")
        self._draw_boxes()

    def set_marker(self, point: list[float] | None) -> None:
        self._marker = point
        self._draw_boxes()

    def set_details_visible(self, visible: bool) -> None:
        self._details_visible = visible
        self._hover = None
        self.configure(cursor="")
        self._draw_boxes()

    def _box_visible(self, index: int) -> bool:
        return self._details_visible or self._points[index]["type"] not in {"text", "status_indicator", "region"}

    def _schedule_resize(self, _event) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(30, self._resize)

    def _resize(self) -> None:
        self._resize_job = None
        self._render()

    def destroy(self) -> None:
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
            self._resize_job = None
        super().destroy()

    def _render(self) -> None:
        self.delete("all")
        self._polygons.clear()
        self._label_rectangles.clear()
        width, height = self.winfo_width(), self.winfo_height()
        if width < 4 or height < 4:
            return
        if self._source is None:
            self._photo = None
            self.create_text(
                width / 2, height / 2, text=self._empty_text,
                fill=self._empty_color, font=("Microsoft YaHei UI", 12),
            )
            return

        src_width, src_height = self._source.size
        margin = max(0, min(18, (min(width, height) - 4) // 4))
        scale = min((width - 2 * margin) / src_width, (height - 2 * margin) / src_height)
        render_width, render_height = max(1, round(src_width * scale)), max(1, round(src_height * scale))
        left, top = (width - render_width) // 2, (height - render_height) // 2
        self._photo = ImageTk.PhotoImage(
            self._source.resize((render_width, render_height), Image.Resampling.LANCZOS), master=self,
        )
        self.create_rectangle(left - 1, top - 1, left + render_width + 1, top + render_height + 1,
                              outline=self._image_border)
        self.create_image(left, top, image=self._photo, anchor="nw")
        offset = np.array((left, top), dtype=np.float32)
        scale_xy = np.array((render_width / src_width, render_height / src_height), dtype=np.float32)
        self._image_offset, self._image_scale = offset, scale_xy
        self._project_points()
        self._draw_boxes()

    def _project_points(self) -> None:
        self._polygons = [
            np.float32(point["photo_outline_px"]) * self._image_scale + self._image_offset
            for point in self._points
        ]

    def _draw_boxes(self) -> None:
        self.delete("overlay")
        self._label_rectangles.clear()
        marker = self._marker
        if self._boxes_visible:
            for index, polygon in enumerate(self._polygons):
                if index != self._selected and self._box_visible(index):
                    self._draw_box(index, polygon)
            if (self._selected is not None and self._selected < len(self._polygons)
                    and self._box_visible(self._selected)):
                self._draw_box(self._selected, self._polygons[self._selected])
            if self._action_labels and self._selected is not None:
                marker = self._points[self._selected]["photo_center_px"]
        if marker is not None:
            x, y = np.asarray(marker) * self._image_scale + self._image_offset
            self.create_line(x - 8, y, x + 8, y, fill="#172132", width=5, tags="overlay")
            self.create_line(x, y - 8, x, y + 8, fill="#172132", width=5, tags="overlay")
            self.create_line(x - 8, y, x + 8, y, fill="#FFBE40", width=3, tags="overlay")
            self.create_line(x, y - 8, x, y + 8, fill="#FFBE40", width=3, tags="overlay")

    def _draw_box(self, index: int, polygon: np.ndarray) -> None:
        selected = index == self._selected
        hovered = index == self._hover
        if (outline := self._points[index].get("display_outline_px")) is not None:
            polygon = np.float32(outline) * self._image_scale + self._image_offset
        default = {"text": "#67B4FC", "status_indicator": "#A8B4BF", "region": "#A8B4BF"}.get(self._points[index]["type"], "#40DDBD")
        if self._points[index].get("review_only"):
            default = "#F2BA58"
        elif "parent_id" in self._points[index]:
            default = "#FFA869"
        border = "#F8CB72" if selected else "#C1EEFF" if hovered else default
        stroke_width = 3 if selected else 2
        self.create_polygon(*polygon.ravel(), fill="", outline="#102236", width=stroke_width + 2, tags="overlay")
        self.create_polygon(*polygon.ravel(), fill="", outline=border, width=stroke_width, tags="overlay")

        if self._action_labels and not selected:
            return
        point = self._points[index]
        name = point["function"] if self._action_labels else point["id"]
        caption = name if len(name) <= 34 else f"{name[:33]}…"
        left = float(polygon[:, 0].min())
        top = max(0.0, float(polygon[:, 1].min()) - 16)
        label = self.create_text(
            left + 4, top + 6, text=caption, fill="#FFF2DB" if selected else "#E2EEF7",
            anchor="w", font=("Segoe UI", -11 if selected else -10, "bold"), tags="overlay",
        )
        x0, y0, x1, y1 = self.bbox(label)
        right = self.winfo_width() - 4
        if x1 > right:
            shift = right - x1
            self.move(label, shift, 0)
            x0, x1 = x0 + shift, x1 + shift
        rect = (x0 - 2, y0 - 1, x1 + 2, y1 + 1)
        self._label_rectangles[index] = rect
        background = self.create_rectangle(
            *rect, fill="#432F18" if selected else "#122634", outline="", tags="overlay",
        )
        self.tag_lower(background, label)

    def _hit(self, x: int, y: int) -> int | None:
        if not self._boxes_visible:
            return None
        for index, (x0, y0, x1, y1) in reversed(self._label_rectangles.items()):
            if x0 <= x <= x1 and y0 <= y <= y1:
                return index
        hits = [index for index, polygon in enumerate(self._polygons)
                if self._box_visible(index) and cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0]
        if self._action_labels:
            return min(hits, key=lambda index: cv2.contourArea(self._polygons[index]), default=None)
        if not hits:
            return None
        foreground = [index for index in hits if self._points[index].get("floating_owner") is not None]
        hits = foreground or hits
        children = {}
        for index in hits:
            if parent := self._points[index].get("parent_id"):
                children.setdefault(parent, index)
        winner = hits[0]
        visited = {winner}
        while (child := children.get(self._points[winner]["id"])) is not None and child not in visited:
            winner = child
            visited.add(winner)
        return winner

    def _move(self, event) -> None:
        index = self._hit(event.x, event.y)
        if index != self._hover:
            self._hover = index
            self.configure(cursor="hand2" if index is not None else "")
            self._draw_boxes()

    def _leave(self, _event) -> None:
        if self._hover is not None:
            self._hover = None
            self.configure(cursor="")
            self._draw_boxes()

    def _click(self, event) -> None:
        index = self._hit(event.x, event.y)
        if index is not None:
            self.select(index)
            self._on_select(index)
