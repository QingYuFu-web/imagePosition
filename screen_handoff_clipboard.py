"""Copy staged handoff files to the Windows clipboard as a native file list."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from functools import cache
import operator
import os
from pathlib import Path
import struct
from typing import Iterable


_CF_HDROP = 15
_GMEM_MOVEABLE = 0x0002
_DROPFILES = struct.Struct("<IiiII")


def _drop_payload(paths: Iterable[str | os.PathLike[str]]) -> bytes:
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise ValueError("需要提供待复制的文件列表")
    try:
        files = iter(paths)
    except TypeError as error:
        raise ValueError("需要提供待复制的文件列表") from error

    absolute_paths = []
    for path in files:
        try:
            absolute_path = Path(path).resolve(strict=True)
            if not absolute_path.is_file():
                raise ValueError("待复制路径必须是文件")
        except (TypeError, OSError, RuntimeError) as error:
            raise ValueError("待复制的文件不存在或无法访问") from error
        absolute_paths.append(str(absolute_path))

    if not absolute_paths:
        raise ValueError("没有待复制的文件")

    header = _DROPFILES.pack(_DROPFILES.size, 0, 0, 0, 1)
    file_list = "\0".join(absolute_paths) + "\0\0"
    return header + file_list.encode("utf-16-le", errors="surrogatepass")


@cache
def _windows_api():
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    for function, argtypes, restype in (
        (user32.IsWindow, (wintypes.HWND,), wintypes.BOOL),
        (user32.OpenClipboard, (wintypes.HWND,), wintypes.BOOL),
        (user32.EmptyClipboard, (), wintypes.BOOL),
        (user32.SetClipboardData, (wintypes.UINT, wintypes.HANDLE), wintypes.HANDLE),
        (user32.CloseClipboard, (), wintypes.BOOL),
        (kernel32.GlobalAlloc, (wintypes.UINT, ctypes.c_size_t), wintypes.HANDLE),
        (kernel32.GlobalLock, (wintypes.HANDLE,), ctypes.c_void_p),
        (kernel32.GlobalUnlock, (wintypes.HANDLE,), wintypes.BOOL),
        (kernel32.GlobalFree, (wintypes.HANDLE,), wintypes.HANDLE),
    ):
        function.argtypes = argtypes
        function.restype = restype
    return user32, kernel32


def _clipboard_error(message: str) -> OSError:
    code = ctypes.get_last_error()
    if code:
        return ctypes.WinError(code, message)
    return OSError(message)


def copy_handoff_files(owner_hwnd: int, paths: Iterable[str | os.PathLike[str]]) -> None:
    """Copy file references, keeping all staged files alive until the recipient reads them."""
    if os.name != "nt":
        raise OSError("复制文件到剪贴板仅在 Windows 上可用")

    try:
        hwnd = operator.index(owner_hwnd)
    except TypeError as error:
        raise ValueError("复制文件需要有效的窗口句柄") from error
    if isinstance(owner_hwnd, bool) or not 0 < hwnd < 1 << (8 * ctypes.sizeof(wintypes.HWND)):
        raise ValueError("复制文件需要有效的窗口句柄")

    payload = _drop_payload(paths)
    user32, kernel32 = _windows_api()
    if not user32.IsWindow(hwnd):
        raise ValueError("复制文件所用的窗口已经关闭")

    memory = kernel32.GlobalAlloc(_GMEM_MOVEABLE, len(payload))
    if not memory:
        raise _clipboard_error("无法为剪贴板文件列表分配内存")

    opened = False
    failed = False
    try:
        destination = kernel32.GlobalLock(memory)
        if not destination:
            raise _clipboard_error("无法准备剪贴板文件列表")
        try:
            ctypes.memmove(destination, payload, len(payload))
        finally:
            # GlobalUnlock returns zero when unlocking the last lock successfully.
            ctypes.set_last_error(0)
            if not kernel32.GlobalUnlock(memory) and ctypes.get_last_error():
                raise _clipboard_error("无法解锁剪贴板文件列表")

        if not user32.OpenClipboard(hwnd):
            raise _clipboard_error("无法打开剪贴板，请稍后重试")
        opened = True
        if not user32.EmptyClipboard():
            raise _clipboard_error("无法清空剪贴板")
        if not user32.SetClipboardData(_CF_HDROP, memory):
            raise _clipboard_error("无法复制文件到剪贴板")
        memory = None  # Windows owns the memory after SetClipboardData succeeds.
    except BaseException:
        failed = True
        raise
    finally:
        cleanup_error = None
        if opened and not user32.CloseClipboard():
            cleanup_error = _clipboard_error("无法关闭剪贴板")
        if memory and kernel32.GlobalFree(memory) and cleanup_error is None:
            cleanup_error = _clipboard_error("无法释放剪贴板文件列表内存")
        if cleanup_error is not None and not failed:
            raise cleanup_error
