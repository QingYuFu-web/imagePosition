"""Recognize a screen, teach corrections, and export numbered AI crops."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter, time_ns
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageGrab

from screen_preview import ScreenPreview
from screen_registration import validate_screen_size_mm


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
BG, WHITE = "#F3F5F8", "#FFFFFF"
INK, MUTED = "#192431", "#586679"
ACCENT, SELECTED, BORDER = "#127763", "#E7F3EE", "#DDE3EA"
FONT = "Microsoft YaHei UI"
WORK_ERRORS = (ImportError, OSError, RuntimeError, ValueError, TypeError, KeyError, tk.TclError, cv2.error)


def _label(parent, text="", *, size=10, bold=False, color=INK, background=WHITE, **kwargs):
    return tk.Label(parent, text=text, font=(FONT, size, "bold" if bold else "normal"),
                    fg=color, bg=background, **kwargs)


def _control_points(page):
    from screen_ai_visibility import visible_point

    words = {row["id"]: row["label"] for row in page["elements"] if row["type"] == "text"}
    controls = [row for row in page["elements"] if row["type"] == "control_candidate"]
    ids = {row["id"] for row in controls}
    floating = page["layer"].get("floating_boxes", ())
    if floating:
        from screen_ai_handoff import _owner

    points = []
    for row in controls:
        l, t, r, b = row["box"]
        text = list(dict.fromkeys(words[key] for key in row.get("text_ids", ()) if key in words))
        label = "、".join(text[:2]) or "候选控件"
        if marker := row.get("selection_marker"):
            label = "已选照片" if marker["state"] == "selected" else "未选照片"
        sample = row.get("reviewed_sample")
        if sample is not None:
            label = sample.get("purpose") or "不可点击"
        disabled = bool(row.get("disabled") or (sample is not None and sample.get("purpose") is None))
        review = bool(row.get("review_only"))
        point = {"id": row["id"], "type": row["type"], "label": label, "geometry_box": row["box"],
                 "photo_outline_px": row.get("polygon", [[l, t], [r, t], [r, b], [l, b]]),
                 "photo_center_px": row["center"], "known_unclickable": disabled,
                 "review_only": review, "learned": sample is not None}
        if "center_mm" in row:
            point["center_mm"] = row["center_mm"]
        if sample is not None:
            point["function"] = sample.get("purpose")
            point["clickable"] = bool(point["function"] and visible_point(page, row, row["center"]))
        if row.get("parent_id") in ids and row["parent_id"] != row["id"]:
            point["parent_id"] = row["parent_id"]
        if (extent := row.get("component_box")) is not None and extent != row["box"]:
            x0, y0, x1, y1 = extent
            point["display_outline_px"] = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        if floating and (owner := _owner(row, floating)) is not None:
            point["floating_owner"] = owner
        points.append(point)
    return points


class ScreenRecognitionApp:
    def __init__(self, root: tk.Tk, *, regions_only=True, screen_size_mm=None, examples_path=None,
                 phone_photo=False, screen_corners=None):
        self.root = root
        self.source = self.image = self.result = self._view = None
        self.points = []
        self._selected = None
        self._analyzer = None
        self._examples_path = examples_path
        self._regions_only = regions_only
        self._calibration = {"screen_size_mm": screen_size_mm, "phone_photo": phone_photo,
                             "screen_corners": screen_corners}
        self._pending = None
        self._complete = None
        self._ai_credentials = (None, None)
        self._copied_handoff = None
        self._last_export = ROOT / "output"
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="screen-workbench")

        cv2.setNumThreads(2 if regions_only else 1)
        self._configure_window()
        self._build()
        self._sync_buttons()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-o>", lambda _: self.open_image())
        root.bind("<Control-v>", lambda _: self.paste_screenshot())
        root.bind("<F5>", lambda _: self.run_analysis())
        self._poll_job = root.after(35, self._poll)
        if regions_only:
            self._worker.submit(self._get_analyzer)

    def _configure_window(self):
        self.root.title("屏幕识别工作台")
        self.root.configure(bg=BG)
        width = min(1180, self.root.winfo_screenwidth() - 48)
        height = min(840, self.root.winfo_screenheight() - 96)
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(800, 510)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TButton", background=WHITE, foreground=INK, font=(FONT, 10),
                        padding=(10, 6), borderwidth=1, bordercolor=BORDER,
                        lightcolor=BORDER, darkcolor=BORDER, relief="flat")
        style.map("TButton", background=[("disabled", BG), ("active", SELECTED)],
                  foreground=[("disabled", "#8792A2")])
        style.configure("Primary.TButton", background=ACCENT, foreground=WHITE,
                        bordercolor=ACCENT, lightcolor=ACCENT, darkcolor=ACCENT)
        style.map("Primary.TButton", background=[("disabled", SELECTED), ("active", "#0B6553")],
                  foreground=[("disabled", MUTED), ("active", WHITE)],
                  bordercolor=[("disabled", SELECTED)], lightcolor=[("disabled", SELECTED)],
                  darkcolor=[("disabled", SELECTED)])
        style.configure("Treeview", background=WHITE, fieldbackground=WHITE, foreground=INK,
                        font=(FONT, 10), rowheight=31, borderwidth=0)
        style.configure("Treeview.Heading", background=BG, foreground=MUTED, font=(FONT, 9, "bold"),
                        padding=(7, 6), relief="flat")
        style.map("Treeview", background=[("selected", SELECTED)], foreground=[("selected", INK)])

    def _build(self):
        header = tk.Frame(self.root, bg=WHITE, padx=16, pady=11)
        header.pack(fill="x")
        _label(header, "屏幕识别工作台", size=14, bold=True).pack(side="left", padx=(0, 16))
        self.open_button = ttk.Button(header, text="打开图片…", command=self.open_image)
        self.open_button.pack(side="left")
        self.paste_button = ttk.Button(header, text="粘贴截图", command=self.paste_screenshot)
        self.paste_button.pack(side="left", padx=(6, 0))
        self.handoff_button = ttk.Button(header, text="交付 AI…", command=self._open_ai_handoff, style="Primary.TButton")
        self.handoff_button.pack(side="right")
        self.learning_button = ttk.Button(header, text="学习…", command=self._open_learning)
        self.learning_button.pack(side="right", padx=6)
        self.run_button = ttk.Button(header, text="重新识别", command=self.run_analysis)
        self.run_button.pack(side="right")

        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(12, 0))
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=1)

        image_panel = tk.Frame(body, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        image_panel.grid(row=0, column=0, sticky="nsew")
        self.image_name = _label(image_panel, "尚未打开图片", bold=True, anchor="w")
        self.image_name.pack(fill="x", padx=12, pady=(9, 8))
        self.preview = ScreenPreview(image_panel, self._select_preview_point, empty_text="打开图片或粘贴截图")
        self.preview.pack(fill="both", expand=True, padx=7, pady=(0, 7))

        results = tk.Frame(body, width=318, bg=WHITE, highlightbackground=BORDER, highlightthickness=1)
        results.grid(row=0, column=1, sticky="ns", padx=(10, 0))
        results.pack_propagate(False)
        result_header = tk.Frame(results, bg=WHITE)
        result_header.pack(fill="x", padx=12, pady=10)
        _label(result_header, "识别结果", size=11, bold=True).pack(side="left")
        self.count = _label(result_header, "", color=MUTED)
        self.count.pack(side="right")

        controls = tk.Frame(results, bg=WHITE)
        controls.pack(side="bottom", fill="x", padx=10, pady=(0, 10))
        self.copy_button = ttk.Button(controls, text="复制 JSON", command=self.copy_json)
        self.copy_button.pack(side="left")
        self.export_button = ttk.Button(controls, text="保存结果…", command=self.export_files)
        self.export_button.pack(side="right")
        detail = tk.Frame(results, bg=BG, padx=9, pady=8)
        detail.pack(side="bottom", fill="x", padx=10, pady=8)
        self.position = _label(detail, "点选控件查看位置", background=BG, color=MUTED, anchor="w", justify="left")
        self.position.pack(fill="x")

        listing = tk.Frame(results, bg=WHITE)
        listing.pack(fill="both", expand=True, padx=10)
        self.table = ttk.Treeview(listing, columns=("id", "label", "state"),
                                  show="headings", selectmode="browse")
        for key, label, width in (("id", "编号", 46), ("label", "控件", 154), ("state", "状态", 62)):
            self.table.heading(key, text=label, anchor="w")
            self.table.column(key, width=width, minwidth=width, stretch=key == "label", anchor="w")
        scroll = ttk.Scrollbar(listing, orient="vertical", command=self.table.yview)
        scroll.pack(side="right", fill="y")
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._table_select)

        self.status = _label(self.root, "就绪", size=9, color=MUTED, background=BG, anchor="w")
        self.status.pack(fill="x", padx=16, pady=(5, 8))

    def _sync_buttons(self):
        available = self._pending is None
        ready = available and self.result is not None
        for button in (self.open_button, self.paste_button):
            button.state(["!disabled"] if available else ["disabled"])
        self.run_button.state(["!disabled"] if available and self.image is not None else ["disabled"])
        for button in (self.learning_button, self.handoff_button, self.export_button, self.copy_button):
            button.state(["!disabled"] if ready else ["disabled"])

    def open_image(self):
        if self._pending is not None:
            return
        name = filedialog.askopenfilename(parent=self.root, title="打开图片", initialdir=ROOT / "img",
                                          filetypes=[("图片", "*.jpg *.jpeg *.png *.bmp *.webp"), ("所有文件", "*.*")])
        if name:
            self.load_image(Path(name))

    def load_image(self, path):
        from screen_boxes import read_frame

        if self._pending is not None:
            return False
        path = path.resolve()
        try:
            frame = read_frame(path)
        except WORK_ERRORS as error:
            messagebox.showerror("无法打开图片", str(error), parent=self.root)
            return False
        self._load_frame(path, frame, path.name)
        return True

    def paste_screenshot(self):
        if self._pending is not None:
            return
        try:
            content = ImageGrab.grabclipboard()
            if isinstance(content, Image.Image):
                frame = cv2.cvtColor(np.asarray(content.convert("RGB")), cv2.COLOR_RGB2BGR)
                self._load_frame(ROOT / f"clipboard-{time_ns()}.png", frame, "剪贴板截图")
                return
            if isinstance(content, list):
                candidate = next((Path(name) for name in content if Path(name).suffix.lower() in IMAGE_EXTENSIONS), None)
                if candidate is not None:
                    self.load_image(candidate)
                    return
        except (OSError, ValueError, cv2.error) as error:
            self.status.configure(text=f"无法读取剪贴板图片 · {error}")
            return
        self.status.configure(text="剪贴板里没有图片")

    def _load_frame(self, path, frame, label):
        self.image, self.source = frame, path
        self.result = self._view = None
        self.preview.set_image(frame, [])
        self._set_points([])
        h, w = frame.shape[:2]
        title = f"{label[:35]}…" if len(label) > 36 else label
        self.image_name.configure(text=f"{title}  ·  {w} × {h} px")
        self.run_analysis()

    def _get_analyzer(self):
        if self._analyzer is None:
            from screen_box_workbench import LocalBoxWorkbench

            analyzer = LocalBoxWorkbench(examples_path=self._examples_path, **self._calibration)
            if self._regions_only:
                analyzer.inspector.font_index.prepare()
            self._analyzer = analyzer
        return self._analyzer

    def _submit(self, work, complete, message):
        self._complete = complete
        self._pending = self._worker.submit(work)
        self.status.configure(text=message)
        self._sync_buttons()
        return self._pending

    def _poll(self):
        if self._pending is not None and self._pending.done():
            pending, self._pending = self._pending, None
            complete, self._complete = self._complete, None
            try:
                complete(pending.result())
            except WORK_ERRORS as error:
                self.status.configure(text=str(error))
                messagebox.showerror("处理失败", str(error), parent=self.root)
            self._sync_buttons()
        self._poll_job = self.root.after(35, self._poll)

    def run_analysis(self, *, learning_status=None):
        if self.image is None or self._pending is not None:
            return None
        frame, source = self.image, self.source

        def analyze():
            analyzer = self._get_analyzer()
            started = perf_counter()
            result, view, _ = analyzer.analyze(frame, source=source.name,
                                              learn_detector=not self._regions_only, regions_only=self._regions_only)
            return result, view, perf_counter() - started

        def show(outcome):
            self.result, self._view, duration = outcome
            self._set_points(_control_points(self.result["page_analysis"]))
            text = [learning_status] if learning_status else []
            applied = len(self.result.get("review_examples", ()))
            if applied:
                text.append(f"已应用 {applied} 条学习样本")
            unavailable = [key.upper() for key, state in self.result.get("engines", {}).items()
                           if state.get("status") != "ok"]
            if unavailable:
                text.append(f"{'、'.join(unavailable)} 不可用")
            text.append(f"识别耗时 {duration:.2f} 秒")
            self.status.configure(text="  ·  ".join(text))

        return self._submit(analyze, show, "正在本机识别…")

    def _set_points(self, points):
        self.points = points
        self._selected = None
        self.preview.set_points(points)
        self.position.configure(text="点选控件查看位置")
        self.table.delete(*self.table.get_children())
        for index, point in enumerate(points):
            state = ("不可点" if point["known_unclickable"] else "待核对" if point["review_only"]
                     else "已核对" if point["learned"] else "")
            self.table.insert("", "end", iid=str(index), values=(point["id"], point["label"], state))
        self.count.configure(text=f"{len(points)} 个控件" if self.result is not None else "")

    def _select_preview_point(self, index):
        self.table.selection_set(str(index))
        self.table.see(str(index))
        self._select(index)

    def _table_select(self, _event):
        if selected := self.table.selection():
            self._select(int(selected[0]))

    def _select(self, index):
        self._selected = index
        point = self.points[index]
        self.preview.select(index)
        x, y = point.get("center_mm", point["photo_center_px"])
        unit = "mm" if "center_mm" in point else "px"
        l, t, r, b = point["geometry_box"]
        self.position.configure(text=f"中心  {x:g}, {y:g} {unit}\n边框  {l:g}, {t:g} – {r:g}, {b:g} px")

    def _open_learning(self):
        if self.result is None or self._pending is not None:
            return None
        from screen_teaching_ai import classify_for_teaching, load_ai_model_config
        from screen_teaching_ui import AIModelSettingsDialog, ReviewExamplesDialog
        from screen_user_examples import _to_mm

        result, frame, source = self.result, self.image, self.source
        page = result["page_analysis"]
        revision = self._view.calibration_revision
        selected = self.points[self._selected] if self._selected is not None else None
        transform = _to_mm({"calibration": page.get("calibration", {}),
                            "screen_corners_px": page.get("screen_corners")}, frame)

        def check_current():
            if self.result is not result or self.image is not frame or self._pending is not None:
                raise ValueError("画面已变化，请重新打开学习窗口。")

        def teach(rows):
            check_current()
            return self._analyzer.teach(frame, rows, source=source.name, calibration_revision=revision)

        def forget():
            check_current()
            return self._analyzer.forget(frame)

        def to_mm(point):
            return cv2.perspectiveTransform(np.float64([[point]]), transform)[0, 0].round(1).tolist()

        def configure_ai(parent, *, run=False):
            check_current()
            notice = ""
            try:
                config = load_ai_model_config()
            except ValueError as error:
                config, notice = None, str(error)
            saved_endpoint, saved_key = self._ai_credentials
            matching_key = saved_key if config is not None and config.endpoint == saved_endpoint else None
            settings = AIModelSettingsDialog(parent, config, matching_key, notice=notice, run=run)
            parent.grab_set()
            if settings.result is None:
                return None
            config, key = settings.result
            self._ai_credentials = (config.endpoint, key)
            return config

        def run_ai(parent):
            check_current()
            try:
                config = load_ai_model_config()
            except ValueError:
                config = None
            if config is None and (config := configure_ai(parent, run=True)) is None:
                return None
            saved_endpoint, saved_key = self._ai_credentials
            api_key = saved_key if config.endpoint == saved_endpoint else None

            def apply_ai(rows):
                check_current()
                if rows:
                    count = self._analyzer.teach(frame, rows, source=source.name, origin="ai",
                                                 calibration_revision=revision)
                    if count:
                        self.run_analysis(learning_status=f"AI 补标已完成 · 更新 {count} 条学习记录")
                        return
                self.status.configure(text="AI 补标已完成 · 没有新增样本")

            return self._submit(lambda: classify_for_teaching(config, source, frame, page, api_key=api_key),
                                apply_ai, "AI 正在补标…")

        return ReviewExamplesDialog(self.root, frame, [], self.points, selected, teach, forget,
                                    lambda text: self.run_analysis(learning_status=text),
                                    point_to_mm=to_mm if transform is not None else None,
                                    run_ai=run_ai, configure_ai=configure_ai)

    def _choose_folder(self):
        directory = filedialog.askdirectory(parent=self.root, title="选择保存目录", initialdir=self._last_export)
        if directory:
            self._last_export = Path(directory)
            return self._last_export
        return None

    def _open_ai_handoff(self):
        if self.result is None or self._pending is not None:
            return None
        frame, page, source = self.image, self.result["page_analysis"], self.source

        def prepare():
            from screen_ai_exchange import write_ai_handoff_preview
            from screen_ai_handoff import save_ai_inputs

            temporary = TemporaryDirectory(prefix="imageposition-handoff-", ignore_cleanup_errors=True)
            try:
                request = save_ai_inputs(Path(temporary.name), source, frame, page, plain_crops=True)[0]
                index = write_ai_handoff_preview(request, page, frame)
            except BaseException:
                temporary.cleanup()
                raise
            return temporary, index

        def show(prepared):
            from screen_handoff_clipboard import copy_handoff_files
            from screen_handoff_ui import AIHandoffPreview

            temporary, index = prepared

            def copy(paths):
                copy_handoff_files(self.root.winfo_id(), paths)
                # Windows file paste reads these paths later, including after the preview closes.
                previous, self._copied_handoff = self._copied_handoff, temporary
                if previous is not None and previous is not temporary:
                    previous.cleanup()

            def close_preview():
                if self._copied_handoff is not temporary:
                    temporary.cleanup()

            try:
                AIHandoffPreview(self.root, index, copy_package=copy, on_close=close_preview)
            except WORK_ERRORS:
                temporary.cleanup()
                raise
            self.status.configure(text="AI 交付预览已就绪")

        return self._submit(prepare, show, "正在生成交付预览…")

    def copy_json(self):
        if self.result is not None and self._pending is None:
            self.root.clipboard_clear()
            self.root.clipboard_append(json.dumps(self.result["page_analysis"], ensure_ascii=False, indent=2))
            self.status.configure(text="识别 JSON 已复制")

    def export_files(self):
        if self.result is None or self._pending is not None or (destination := self._choose_folder()) is None:
            return None
        result, frame, source = self.result, self.image, self.source

        def save():
            from screen_page_handoff import save_page_handoff

            return save_page_handoff(destination, source, frame, result, include_review_images=False)

        return self._submit(save, lambda paths: self.status.configure(text=f"JSON 和标注图已保存 · {paths[0].parent}"),
                            "正在保存识别结果…")

    def close(self):
        self.root.after_cancel(self._poll_job)
        self._worker.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()
        if self._copied_handoff is not None:
            self._copied_handoff.cleanup()
            self._copied_handoff = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path)
    parser.add_argument("--examples", type=Path, help="使用指定的本地学习库。")
    parser.add_argument("--full-analysis", action="store_true", help="同时识别画面可见文字。")
    parser.add_argument("--phone-photo", action="store_true", help="在手机实拍中定位屏幕。")
    parser.add_argument("--screen-size-mm", nargs=2, type=float, metavar=("WIDTH", "HEIGHT"),
                        help="实际显示区域的实测宽高，单位为毫米。")
    parser.add_argument("--screen-corners", nargs=8, type=float, metavar="PX",
                        help="左上、右上、右下、左下四组原图像素坐标。")
    parser.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        screen_size_mm = validate_screen_size_mm(args.screen_size_mm)
    except ValueError as error:
        parser.error(str(error))
    root = tk.Tk()
    app = ScreenRecognitionApp(root, regions_only=not args.full_analysis, examples_path=args.examples,
                               phone_photo=args.phone_photo, screen_corners=args.screen_corners,
                               screen_size_mm=screen_size_mm)
    if args.image is not None:
        root.after_idle(app.load_image, args.image)
    root.mainloop()


if __name__ == "__main__":
    main()
