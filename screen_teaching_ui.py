"""Review visible controls and confirm reusable examples on the source image."""

from math import ceil, floor, isfinite
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk


_RESIZE_CURSORS = {
    "n": "sb_v_double_arrow", "s": "sb_v_double_arrow",
    "w": "sb_h_double_arrow", "e": "sb_h_double_arrow",
    "nw": "top_left_corner", "ne": "top_right_corner",
    "sw": "bottom_left_corner", "se": "bottom_right_corner",
}


def _inside_point(box, point):
    left, top, right, bottom = box
    x, y = point
    return [max(left, min(right - .01, round(x, 2))), max(top, min(bottom - .01, round(y, 2)))]


class _ReviewImage(tk.Canvas):
    def __init__(self, parent, frame, *, on_pick=None, on_draw=None, on_change=None, on_hover=None):
        super().__init__(parent, background="#182231", highlightthickness=0,
                         cursor="crosshair" if on_pick is not None else "")
        self._source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._photo = None
        self._scale = (1, 1)
        self._offset = (0, 0)
        self._drag = None
        self._hide_point = False
        self._box = self._point = None
        self._on_pick, self._on_draw, self._on_change, self._on_hover = on_pick, on_draw, on_change, on_hover
        self.bind("<Configure>", self._render)
        if on_pick is not None:
            self.bind("<Button-1>", self._press)
            self.bind("<B1-Motion>", self._move)
            self.bind("<ButtonRelease-1>", self._release)
        if on_hover is not None:
            self.bind("<Motion>", self._hover)
            self.bind("<Leave>", lambda _event: self.delete("hover"))

    def display(self, box=None, point=None, *, hide_point=False):
        self._box, self._point, self._hide_point = box, point, hide_point
        self._highlight()

    def _render(self, event):
        self._drag = None
        width, height = self._source.size
        ratio = min(max(1, event.width - 20) / width, max(1, event.height - 20) / height)
        size = max(1, round(width * ratio)), max(1, round(height * ratio))
        self._scale = size[0] / width, size[1] / height
        self._offset = (event.width - size[0]) // 2, (event.height - size[1]) // 2
        self._photo = ImageTk.PhotoImage(self._source.resize(size, Image.Resampling.LANCZOS), master=self)
        self.delete("all")
        self.create_image(*self._offset, image=self._photo, anchor="nw")
        self._highlight()

    def _frame_handles(self, start, end):
        left, top = start
        right, bottom = end
        for name, x, y in (("nw", left - 2, top - 2), ("ne", right + 2, top - 2),
                           ("sw", left - 2, bottom + 2), ("se", right + 2, bottom + 2)):
            yield name, x, y

    def _move_handle(self, start, end):
        left, top = start
        right, bottom = end
        if min(right - left, bottom - top) < 16:
            return (left + right) / 2, top - 11 if top >= 16 else bottom + 11

    def _point_radius(self, start, end):
        return min(5, max(2.5, .23 * min(end[0] - start[0], end[1] - start[1])))

    def _highlight(self, geometry=None):
        self.delete("highlight")
        self.delete("hover")
        box, point = (self._box, self._point) if geometry is None else geometry
        if box is None:
            return
        color = "#78e9d2" if geometry is not None else "#ffce69"
        start = self._screen(box[:2])
        end = self._screen(box[2:])
        self.create_rectangle(*start, *end, outline=color, width=2, tags="highlight")
        if self._on_change is not None:
            for _, x, y in self._frame_handles(start, end):
                self.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill=color, outline="#202936", tags="highlight")
            if handle := self._move_handle(start, end):
                x, y = handle
                self.create_rectangle(x - 5, y - 2, x + 5, y + 2, fill=color, outline="#202936", tags="highlight")
        if point is not None and not self._hide_point:
            x, y = self._screen(point)
            radius = self._point_radius(start, end) if self._on_change is not None else 3
            self.create_oval(x - radius, y - radius, x + radius, y + radius,
                             fill=color, outline="#202936", width=2, tags="highlight")

    def _hover(self, event):
        self.delete("hover")
        if self._drag is not None:
            return
        mode = self._drag_hit(event.x, event.y)
        if mode != "draw":
            self.configure(cursor=_RESIZE_CURSORS.get(mode, "fleur"))
            return
        point = self._image_position(event.x, event.y)
        box = self._on_hover(point) if point is not None else None
        self.configure(cursor="hand2" if box is not None else "crosshair")
        if box is not None and box != self._box:
            self.create_rectangle(*self._screen(box[:2]), *self._screen(box[2:]),
                                  outline="#78ddbf", width=2, tags="hover")

    def _screen(self, point):
        x, y = point
        ox, oy = self._offset
        sx, sy = self._scale
        return ox + x * sx, oy + y * sy

    def _image_position(self, x, y, *, clamp=False):
        if self._photo is None:
            return None
        ox, oy = self._offset
        sx, sy = self._scale
        width, height = self._source.size
        x, y = (x - ox) / sx, (y - oy) / sy
        if clamp:
            return min(width, max(0, x)), min(height, max(0, y))
        return (x, y) if 0 <= x < width and 0 <= y < height else None

    def _drag_hit(self, x, y):
        if self._box is None:
            return "draw"
        start, end = self._screen(self._box[:2]), self._screen(self._box[2:])
        left, top = start
        right, bottom = end
        if self._point is not None and not self._hide_point:
            px, py = self._screen(self._point)
            radius = self._point_radius(start, end) + 1
            if (x - px) ** 2 + (y - py) ** 2 <= radius ** 2:
                return "point"
        if handle := self._move_handle(start, end):
            hx, hy = handle
            if abs(x - hx) <= 6 and abs(y - hy) <= 4:
                return "move"

        corners = [(max(abs(x - hx), abs(y - hy)), name) for name, hx, hy in self._frame_handles(start, end)]
        distance, corner = min(corners)
        if distance <= 5 and not (left + 2 < x < right - 2 and top + 2 < y < bottom - 2):
            return corner
        if left <= x <= right:
            inset = min(5, max(1, .16 * (bottom - top)))
            if top - 5 <= y <= top + inset:
                return "n"
            if bottom - inset <= y <= bottom + 5:
                return "s"
        if top <= y <= bottom:
            inset = min(5, max(1, .16 * (right - left)))
            if left - 5 <= x <= left + inset:
                return "w"
            if right - inset <= x <= right + 5:
                return "e"
        return "move" if left <= x <= right and top <= y <= bottom else "draw"

    def _press(self, event):
        mode = "draw" if event.state & 1 else self._drag_hit(event.x, event.y)
        start = self._image_position(event.x, event.y, clamp=mode != "draw")
        if start is not None:
            self.focus_set()
            self._drag = mode, (event.x, event.y), start

    def _dragged_geometry(self, event):
        mode, _, (x0, y0) = self._drag
        x, y = self._image_position(event.x, event.y, clamp=True)
        if mode == "draw":
            box = [floor(min(x0, x)), floor(min(y0, y)), ceil(max(x0, x)), ceil(max(y0, y))]
            return box, None
        box, point = list(self._box), self._point
        if mode == "point":
            return box, _inside_point(box, (x, y))
        left, top, right, bottom = box
        width, height = self._source.size
        dx, dy = round(x - x0), round(y - y0)
        if mode == "move":
            dx, dy = max(-left, min(width - right, dx)), max(-top, min(height - bottom, dy))
            return [left + dx, top + dy, right + dx, bottom + dy], [point[0] + dx, point[1] + dy]
        if "w" in mode:
            box[0] = max(0, min(right - 8, left + dx))
        if "e" in mode:
            box[2] = min(width, max(left + 8, right + dx))
        if "n" in mode:
            box[1] = max(0, min(bottom - 8, top + dy))
        if "s" in mode:
            box[3] = min(height, max(top + 8, bottom + dy))
        return box, _inside_point(box, point)

    def _moved(self, event):
        _, (x, y), _ = self._drag
        return max(abs(event.x - x), abs(event.y - y)) >= 3

    def _move(self, event):
        if self._drag is not None and self._moved(event):
            self._highlight(self._dragged_geometry(event))

    def _release(self, event):
        if self._drag is None:
            return
        mode, _, _ = self._drag
        changed = self._moved(event)
        geometry = self._dragged_geometry(event) if changed else None
        self.cancel_drag()
        if mode == "draw":
            if changed:
                self._on_draw(geometry[0])
            else:
                self._on_pick(self._image_position(event.x, event.y, clamp=True))
        elif changed:
            self._on_change(*geometry)

    def cancel_drag(self):
        if self._drag is None:
            return False
        self._drag = None
        self._highlight()
        return True


def _control_option(row, width, height):
    outline = row["photo_outline_px"]
    xs, ys = zip(*outline, strict=True)
    box = [max(0, floor(min(xs))), max(0, floor(min(ys))),
           min(width, ceil(max(xs))), min(height, ceil(max(ys)))]
    point = list(row["photo_center_px"])
    name = row.get("function")
    if not name and row.get("label", "").startswith("待复核 · "):
        name = row["label"].removeprefix("待复核 · ")
    return {"box": box, "click_point": point, "function": name,
            **({"known_unclickable": True} if row.get("known_unclickable") else {})}


class AIModelSettingsDialog(simpledialog.Dialog):
    def __init__(self, parent, config, api_key, *, notice="", run=False):
        self._config = config
        self._api_key = api_key
        self._notice = notice
        self._save_text = "保存并补标" if run else "保存设置"
        super().__init__(parent, title="AI 模型设置")

    def body(self, parent):
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text="使用支持图像输入的 OpenAI 兼容接口。补标时会向该模型发送当前局部图。",
                  wraplength=450).grid(row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(4, 10))
        fields = (("服务地址", self._config.endpoint if self._config else "", ""),
                  ("模型名称", self._config.model if self._config else "", ""),
                  ("API 密钥", self._api_key or "", "*"))
        self._inputs = []
        for row, (label, value, mask) in enumerate(fields, 1):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(8, 12), pady=4)
            entry = ttk.Entry(parent, width=49, show=mask)
            entry.insert(0, value)
            entry.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=4)
            self._inputs.append(entry)
        ttk.Label(parent, text="密钥仅用于本次运行。更换地址时请填写对应密钥。").grid(
            row=4, column=1, sticky="w", pady=(2, 6))
        self._status = tk.StringVar(self, value=self._notice)
        ttk.Label(parent, textvariable=self._status, wraplength=450).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=8)
        return self._inputs[0]

    def buttonbox(self):
        bar = ttk.Frame(self, padding=(8, 0, 8, 9))
        bar.pack(fill="x")
        ttk.Button(bar, text="取消", command=self.cancel).pack(side="right")
        ttk.Button(bar, text=self._save_text, style="Primary.TButton", command=self.ok).pack(side="right", padx=8)
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def validate(self):
        from screen_teaching_ai import AIModelConfig, save_ai_model_config

        endpoint, model, api_key = (field.get().strip() for field in self._inputs)
        try:
            config = AIModelConfig(endpoint, model, self._config.timeout if self._config else 45)
            save_ai_model_config(config)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            return False
        if self._config is not None and config.endpoint != self._config.endpoint and api_key == self._api_key:
            api_key = None
        self.result = config, api_key or None
        return True


class ReviewExamplesDialog(tk.Toplevel):
    def __init__(self, parent, frame, actions, points, selected, teach, forget, changed,
                 *, initial_annotation=None, point_to_mm=None, run_ai=None, configure_ai=None):
        super().__init__(parent)
        self.title("学习")
        self.transient(parent)
        self.configure(bg="#f3f5f8")
        self.geometry(f"{min(1020, self.winfo_screenwidth() - 48)}x{min(720, self.winfo_screenheight() - 92)}")
        self.minsize(680, 450)
        self._height, self._width = frame.shape[:2]
        self._teach, self._forget, self._changed = teach, forget, changed
        self._point_to_mm = point_to_mm
        self._run_ai, self._configure_ai = run_ai, configure_ai
        self._previous_grab = parent.grab_current()
        self._candidates = self._collect_options(actions, points)
        self._draft = self._initial = None
        self._undo_geometry = None
        self._seeded = initial_annotation is not None
        self._function = tk.StringVar(self)
        self._exclude = tk.BooleanVar(self)
        self._status = tk.StringVar(self)
        self._position = tk.StringVar(self)
        self._instruction = tk.StringVar(self, value="拖动画框。拖框内移动，拖边角缩放，拖圆点改落点。")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.image = _ReviewImage(self, frame, on_pick=self._pick_at, on_draw=self._draw_box,
                                  on_change=self._change_geometry, on_hover=self._hover_box)
        self.image.grid(row=1, column=0, sticky="nsew", padx=(16, 12), pady=(0, 16))
        self.image.bind("<Control-z>", self._undo_drag)
        self._build_controls()
        self.bind("<Escape>", self._escape)
        self.bind("<Destroy>", self._restore_grab, add="+")
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.grab_set()
        if initial_annotation is not None:
            self._edit(initial_annotation)
        elif selected is not None:
            self._edit(_control_option(selected, self._width, self._height))

    def _restore_grab(self, event):
        if event.widget is self and self._previous_grab is not None:
            try:
                if self._previous_grab.winfo_exists():
                    self._previous_grab.grab_set()
            except tk.TclError:
                return

    def _collect_options(self, actions, points):
        options = []
        seen = set()
        for row in (*actions, *points):
            if row["type"] != "control_candidate":
                continue
            annotation = _control_option(row, self._width, self._height)
            box = tuple(annotation["box"])
            if box not in seen:
                seen.add(box)
                options.append(annotation)
        return options

    def _build_controls(self):
        bar = tk.Frame(self, bg="#f3f5f8")
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(12, 10))
        tk.Label(bar, textvariable=self._instruction, bg="#f3f5f8", fg="#192431",
                 font=("Microsoft YaHei UI", 10)).pack(side="left")
        self._clear_button = ttk.Button(bar, text="清除本图样本…", command=self._delete_saved)
        self._ai_button = ttk.Button(bar, text="AI 补标", style="Primary.TButton", command=self._open_ai)
        if not self._seeded:
            self._clear_button.pack(side="right")
            if self._configure_ai is not None:
                ttk.Button(bar, text="AI 设置…", command=self._open_ai_settings).pack(side="right", padx=6)
            if self._run_ai is not None:
                self._ai_button.pack(side="right")

        panel = tk.Frame(self, bg="#ffffff", highlightbackground="#dde3ea", highlightthickness=1, padx=16, pady=16)
        panel.grid(row=1, column=1, sticky="nsew", padx=(0, 16), pady=(0, 16))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(5, weight=1)
        self._detail_photo = None
        self._detail = tk.Label(panel, bg="#f3f5f8", borderwidth=0, padx=10, pady=8)
        self._detail.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        self._detail.grid_remove()
        tk.Label(panel, text="用途", bg="#ffffff", fg="#192431",
                 font=("Microsoft YaHei UI", 12, "bold")).grid(row=1, column=0, sticky="w")
        self._entry = ttk.Entry(panel, textvariable=self._function, width=24, font=("Microsoft YaHei UI", 11))
        self._entry.grid(row=2, column=0, sticky="ew", ipady=7, pady=(9, 5))
        self._entry.bind("<Return>", lambda _event: self._save())
        self._example = tk.Label(panel, text="例如「打开评论」", bg="#ffffff", fg="#586679",
                                 font=("Microsoft YaHei UI", 9))
        self._example.grid(row=3, column=0, sticky="w", pady=(0, 12))
        self._exclude_button = ttk.Checkbutton(panel, text="标为不可点击", variable=self._exclude,
                                               command=self._toggle_exclude)
        self._exclude_button.grid(row=4, column=0, sticky="w")
        position = tk.Frame(panel, bg="#ffffff")
        position.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        tk.Label(position, textvariable=self._position, bg="#ffffff", fg="#586679",
                 font=("Microsoft YaHei UI", 9), anchor="w").pack(anchor="w")
        tk.Label(panel, textvariable=self._status, wraplength=220, bg="#ffffff", fg="#586679",
                 font=("Microsoft YaHei UI", 9), justify="left").grid(row=6, column=0, sticky="w", pady=(0, 9))
        self._save_button = ttk.Button(panel, text="保存", style="Primary.TButton", command=self._save)
        self._save_button.grid(row=7, column=0, sticky="ew")
        self._cancel_button = ttk.Button(panel, text="取消", command=self._close)
        self._cancel_button.grid(row=8, column=0, sticky="ew", pady=(6, 0))
        self._update_controls()

    def _candidate_at(self, point):
        x, y = point
        hits = []
        for annotation in self._candidates:
            l, t, r, b = annotation["box"]
            if l <= x < r and t <= y < b:
                hits.append((not annotation["function"], (r - l) * (b - t), annotation))
        winner = min(hits, key=lambda item: item[:2], default=None)
        return winner[2] if winner is not None else None

    def _hover_box(self, point):
        candidate = self._candidate_at(point) if not self._seeded else None
        return candidate["box"] if candidate is not None else None

    def _pick_at(self, point):
        if self._seeded:
            return
        candidate = self._candidate_at(point)
        if candidate is not None and candidate != self._initial and self._discard_ok():
            self._edit(candidate)

    def _draw_box(self, box):
        l, t, r, b = box
        if r - l < 8 or b - t < 8:
            self._status.set("请框住完整控件，至少宽高各 8 像素。")
            return
        point = [(l + r) / 2, (t + b) / 2]
        if self._seeded:
            self._change_geometry(box, point)
        elif self._discard_ok():
            self._edit({"function": "", "box": box, "click_point": point})

    def _change_geometry(self, box, point):
        previous = self._draft["box"], self._draft["click_point"]
        if previous == (box, point):
            return
        self._undo_geometry = previous
        self._update_geometry(box, point)

    def _update_geometry(self, box, point):
        changed_box = self._draft["box"] != box
        self._draft.update(box=box, click_point=point)
        self._status.set("")
        if changed_box:
            self._refresh_detail()
        self._refresh_image()

    def _undo_drag(self, _event=None):
        if self.image.cancel_drag():
            return "break"
        if self._undo_geometry is not None:
            previous, self._undo_geometry = self._undo_geometry, None
            self._update_geometry(*previous)
        return "break"

    def _edit(self, annotation):
        left, top, right, bottom = annotation["box"]
        box = [floor(left), floor(top), ceil(right), ceil(bottom)]
        self._draft = {"function": annotation["function"], "box": box, "click_point": list(annotation["click_point"])}
        self._function.set(annotation["function"] or "")
        self._exclude.set(bool(annotation.get("known_unclickable")
                               or self._seeded and annotation["function"] is None))
        self._initial = self._current()
        self._undo_geometry = None
        self._status.set("")
        self._update_controls()
        self._refresh_detail()
        if not self._exclude.get():
            self._entry.focus_set()
            self._entry.selection_range(0, "end")
        self._refresh_image()

    def _refresh_detail(self):
        crop = self.image._source.crop(self._draft["box"])
        if 0 < min(crop.size) <= .07 * max(self._width, self._height):
            ratio = min(210 / crop.width, 68 / crop.height, 3)
            size = tuple(max(1, round(side * ratio)) for side in crop.size)
            self._detail_photo = ImageTk.PhotoImage(crop.resize(size, Image.Resampling.LANCZOS), master=self)
            self._detail.configure(image=self._detail_photo)
            self._detail.grid()
        else:
            self._detail.grid_remove()
            self._detail_photo = None

    def _toggle_exclude(self):
        self._status.set("")
        self._update_controls()
        self._refresh_image()

    def _update_controls(self):
        selected = self._draft is not None
        editable = selected and not self._exclude.get()
        self._entry.state(["!disabled"] if editable else ["disabled"])
        self._exclude_button.state(["!disabled"] if selected else ["disabled"])
        self._save_button.state(["!disabled"] if selected else ["disabled"])
        if editable and not self._function.get():
            self._example.grid()
        else:
            self._example.grid_remove()

    def _refresh_image(self):
        if self._draft is not None:
            box, point = self._draft["box"], self._draft["click_point"]
            self.image.display(box, point, hide_point=self._exclude.get())
            if self._exclude.get() or self._point_to_mm is None:
                self._position.set("")
            else:
                x, y = self._point_to_mm(point)
                self._position.set(f"点击位置 {x:g}, {y:g} mm")

    def _current(self):
        return {**self._draft, "function": None if self._exclude.get() else self._function.get().strip()}

    def _discard_ok(self):
        if self._draft is None:
            return True
        current, initial = self._current(), self._initial
        return current == initial or messagebox.askyesno("放弃这次修改？", "这处修改还没有保存。", parent=self)

    def _close(self):
        if self._discard_ok():
            self.destroy()

    def _escape(self, _event=None):
        if not self.image.cancel_drag():
            self._close()
        return "break"

    def _open_ai(self):
        if not self._discard_ok():
            return
        try:
            started = self._run_ai(self)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            return
        if started is not None:
            self.destroy()

    def _open_ai_settings(self):
        try:
            saved = self._configure_ai(self)
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            return
        if saved is not None:
            self._status.set("AI 模型设置已保存")

    def _save(self):
        if self._draft is None:
            return
        annotation = self._current()
        function = annotation["function"]
        if function is not None and (not 1 <= len(function) <= 50 or not function.isprintable()):
            self._status.set("请填写一个简短的用途。")
            self._entry.focus_set()
            return
        try:
            count = self._teach([annotation])
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            return
        self.destroy()
        if count:
            self._changed("已保存修改")

    def _delete_saved(self):
        if not messagebox.askyesno("删除本地样本", "确定删除从当前截图保存的样本？", parent=self):
            return
        try:
            count = self._forget()
        except (OSError, ValueError) as error:
            self._status.set(str(error))
            return
        if count:
            self.destroy()
            self._changed(f"已删除 {count} 项本地样本")
        else:
            self._status.set("这张截图尚无已保存的样本。")


_MATCH_LABELS = {"exact": "原图匹配", "near": "近似画面", "relocated": "重新定位"}


def review_example_label(example):
    source = example["source"]
    match = _MATCH_LABELS[example["match"]]
    origin = {"manual": "手动", "ai": "AI", "detector": "本地识别"}.get(example.get("origin"), "手动")
    return f"{origin} · {source} · {match}"


class AppliedSamplesDialog(tk.Toplevel):
    """Show which saved decisions were used on this screenshot."""

    def __init__(self, parent, frame, examples, selected_id, forget_sample, changed):
        super().__init__(parent)
        self.title("本次用到的学习样本")
        self.transient(parent)
        self.configure(bg="#f3f5f8")
        self.geometry(f"{min(1040, self.winfo_screenwidth() - 60)}x{min(720, self.winfo_screenheight() - 110)}")
        self.minsize(730, 430)
        self._examples = examples
        self._forget_sample, self._changed = forget_sample, changed
        self._detail = tk.StringVar(self)
        self._error = tk.StringVar(self)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.image = _ReviewImage(self, frame)
        self.image.grid(row=0, column=0, sticky="nsew", padx=(10, 4), pady=10)

        panel = ttk.Frame(self, padding=(8, 10, 10, 10))
        panel.grid(row=0, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)
        table_frame = ttk.Frame(panel)
        table_frame.grid(row=0, column=0, sticky="nsew")
        self._table = ttk.Treeview(table_frame, columns=("decision", "source"), show="headings", selectmode="browse")
        self._table.heading("decision", text="用途", anchor="w")
        self._table.heading("source", text="来源图片", anchor="w")
        self._table.column("decision", width=125, minwidth=100, anchor="w")
        self._table.column("source", width=200, minwidth=140, anchor="w")
        self._table.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self._table.yview)
        scroll.pack(side="right", fill="y")
        self._table.configure(yscrollcommand=scroll.set)
        self._table.bind("<<TreeviewSelect>>", self._select)
        for index, example in enumerate(examples):
            self._table.insert("", "end", iid=str(index),
                               values=(example["function"] or "不可点击", example["source"]))

        ttk.Label(panel, textvariable=self._detail, wraplength=320, justify="left").grid(
            row=1, column=0, sticky="ew", pady=(9, 3))
        ttk.Label(panel, textvariable=self._error, wraplength=320).grid(row=2, column=0, sticky="ew")
        buttons = ttk.Frame(panel)
        buttons.grid(row=3, column=0, sticky="ew", pady=(5, 0))
        ttk.Button(buttons, text="撤销这条样本…", command=self._delete).pack(side="left")
        ttk.Button(buttons, text="关闭", command=self.destroy).pack(side="right")

        initial = next((i for i, item in enumerate(examples) if item["id"] == selected_id), 0)
        self._table.selection_set(str(initial))
        self._table.see(str(initial))
        self._select()
        self.bind("<Escape>", lambda _event: self.destroy())
        self.grab_set()

    def _selected_example(self):
        selection = self._table.selection()
        return self._examples[int(selection[0])] if selection else None

    def _select(self, _event=None):
        example = self._selected_example()
        if example is None:
            return
        self._error.set("")
        point = example["click_point"]
        self.image.display(example["box"], point if example["function"] is not None else None)
        kind = "核对位置" if example["function"] is None else "点击位置"
        units = "px"
        center_mm = example.get("center_mm")
        if (isinstance(center_mm, (list, tuple)) and len(center_mm) == 2
                and all(type(value) in (int, float) and isfinite(value) for value in center_mm)):
            point, units = center_mm, "mm"
        self._detail.set(f"{review_example_label(example)}\n{kind} {point[0]:.1f}, {point[1]:.1f} {units}")

    def _delete(self):
        example = self._selected_example()
        if example is None:
            return
        decision = example["function"] or "不可点击"
        note = f"删除来自「{example['source']}」的「{decision}」样本？当前画面会重新分析。"
        if not messagebox.askyesno("删除选中样本", note, parent=self):
            return
        try:
            count = self._forget_sample(example["id"])
        except (OSError, ValueError) as error:
            self._error.set(str(error))
            return
        self.destroy()
        self._changed("已删除 1 项本地样本" if count else "已按当前样本刷新")
