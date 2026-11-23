"""Preview the exact manifest and individual crops in a local AI handoff."""

import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


_THUMBNAIL = (156, 124)
_TILE_WIDTH = _THUMBNAIL[0] + 20


def _package_preview(index_path):
    path = Path(index_path)
    text = path.read_bytes().decode("utf-8-sig")
    package = json.loads(text)
    try:
        owners = {row["file"]: row["element_id"] for row in package["components"] if "file" in row}
        filenames = package["attachment_files"]
        if not isinstance(filenames, list):
            raise ValueError("交付包的图片清单格式无效。")
        thumbnails = []
        for name in filenames:
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("交付包的裁图文件名无效。")
            element_id = owners[name]
            with Image.open(path.parent / name) as original:
                original.thumbnail(_THUMBNAIL, Image.Resampling.LANCZOS)
                thumbnails.append((element_id, original.convert("RGB")))
    except (KeyError, TypeError) as error:
        raise ValueError("交付包的图片清单缺少对应的控件编号。") from error
    return text, thumbnails, (path, *(path.parent / name for name in filenames))


class AIHandoffPreview(tk.Toplevel):
    """Show the same evidence included in the clipboard attachments."""

    def __init__(self, parent, index_path, *, copy_package, on_close):
        text, images, attachments = _package_preview(index_path)
        super().__init__(parent)
        self.json_text = text
        self.thumbnails = tuple(identity for identity, _ in images)
        self._attachments = attachments
        self._copy_package = copy_package
        self._on_close = on_close
        self._status = tk.StringVar(self)
        self._photos = []
        self._tiles = []
        self._columns = 0

        self.title("AI 交付预览")
        self.transient(parent)
        self.minsize(700, 450)
        width = min(1120, self.winfo_screenwidth() - 70)
        height = min(760, self.winfo_screenheight() - 110)
        self.geometry(f"{width}x{height}")
        self.configure(background="#F3F5F8")
        self._build(images)
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.bind("<Escape>", self.close)
        self.bind("<Destroy>", self._closed, add="+")
        self.bind("<MouseWheel>", self._scroll_images)
        self.grab_set()
        self.after_idle(self.copy_button.focus_set)

    def _build(self, images):
        panes = ttk.Panedwindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=14, pady=(14, 0))
        json_pane = ttk.Frame(panes, padding=(10, 8))
        crop_pane = ttk.Frame(panes, padding=(10, 8))
        panes.add(json_pane, weight=1)
        panes.add(crop_pane, weight=1)
        self._build_json(json_pane)
        self._build_crops(crop_pane, images)

        footer = ttk.Frame(self, padding=(14, 10))
        footer.pack(fill="x")
        footer.columnconfigure(0, weight=1)
        status = ttk.Label(footer, textvariable=self._status, foreground="#526174", wraplength=600)
        status.grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Button(footer, text="关闭", command=self.close).grid(row=0, column=1, padx=(0, 8))
        self.copy_button = ttk.Button(footer, text="复制附件", style="Primary.TButton", command=self.copy_attachments)
        self.copy_button.grid(row=0, column=2)

    def _build_json(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        heading = ttk.Frame(parent)
        heading.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))
        ttk.Label(heading, text="交付 JSON", font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")
        ttk.Button(heading, text="复制 JSON", command=self.copy_json).pack(side="right")

        view = tk.Text(parent, width=45, height=10, wrap="char", font=("Consolas", 10),
                       background="#FFFFFF", foreground="#192431", padx=10, pady=8, relief="flat")
        vertical = ttk.Scrollbar(parent, orient="vertical", command=view.yview)
        view.configure(yscrollcommand=vertical.set)
        view.grid(row=1, column=0, sticky="nsew")
        vertical.grid(row=1, column=1, sticky="ns")
        view.insert("1.0", self.json_text)
        view.configure(state="disabled")

    def _build_crops(self, parent, images):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)
        ttk.Label(parent, text=f"编号局部图（{len(images)} 张）", font=("Microsoft YaHei UI", 11, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        self._canvas = tk.Canvas(parent, background="#FFFFFF", borderwidth=0, highlightthickness=0, width=360)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scrollbar.set)
        self._canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        content = tk.Frame(self._canvas, background="#FFFFFF")
        self._crop_window = self._canvas.create_window(0, 0, window=content, anchor="nw")
        content.bind("<Configure>", lambda _event: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>", self._layout_crops)

        for identity, image in images:
            tile = tk.Frame(content, background="#F6F8FA", highlightbackground="#DDE3EA", highlightthickness=1)
            tk.Label(tile, text=identity, background="#F6F8FA", foreground="#192431",
                     font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", padx=8, pady=(5, 0))
            well = tk.Frame(tile, width=_THUMBNAIL[0], height=_THUMBNAIL[1], background="#F6F8FA")
            well.pack(padx=5, pady=(3, 6))
            photo = ImageTk.PhotoImage(image, master=self)
            tk.Label(well, image=photo, background="#F6F8FA", borderwidth=0).place(relx=.5, rely=.5, anchor="center")
            self._photos.append(photo)
            self._tiles.append(tile)

    def _layout_crops(self, event):
        self._canvas.itemconfigure(self._crop_window, width=event.width)
        columns = max(1, event.width // _TILE_WIDTH)
        if columns != self._columns:
            self._columns = columns
            for index, tile in enumerate(self._tiles):
                row, column = divmod(index, columns)
                tile.grid(row=row, column=column, padx=4, pady=4, sticky="n")

    def _scroll_images(self, event):
        if event.widget is self._canvas or str(event.widget).startswith(f"{self._canvas}."):
            if event.delta:
                self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
                return "break"

    def copy_json(self):
        self.clipboard_clear()
        self.clipboard_append(self.json_text)
        self._status.set("已复制交付 JSON")

    def copy_attachments(self):
        try:
            self._copy_package(self._attachments)
        except (OSError, ValueError, RuntimeError, tk.TclError) as error:
            self._status.set(f"复制附件失败 · {error}")
        else:
            self._status.set(f"已复制 JSON 和 {len(self.thumbnails)} 张裁图，可粘贴到支持文件附件的输入框。")

    def close(self, _event=None):
        self.destroy()
        return "break"

    def _closed(self, event):
        if event.widget is self and self._on_close is not None:
            callback, self._on_close = self._on_close, None
            callback()
