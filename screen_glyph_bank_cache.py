"""Persist the deterministic system-font glyph banks between processes."""

from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
from typing import NamedTuple
from zipfile import BadZipFile, ZIP_STORED, ZipFile

import cv2
import numpy as np
import PIL
from PIL import features


# Bump when the rendered glyph bank's rasterization or ordering changes.
_RECIPE_VERSION = 1
_MAX_BANK_BYTES = 192 * 1024**2
_DISK_BUDGET = 512 * 1024**2
_HEADER_LIMIT = 4096
_PREFIX = "glyph-bank-"
_CACHE_ERRORS = (OSError, ValueError, EOFError, BadZipFile, OverflowError, RuntimeError)


class GlyphBankPayload(NamedTuple):
    rasters: np.ndarray
    coarse: np.ndarray
    ratios: np.ndarray
    chars: tuple[str, ...]
    fonts: tuple[str, ...]


def _payload(bank):
    if bank is None:
        return None
    if hasattr(bank, "rasters"):
        return GlyphBankPayload(bank.rasters, bank.coarse, bank.ratios, tuple(bank.chars), tuple(bank.fonts))
    rasters, coarse, ratios, chars, fonts = bank
    return GlyphBankPayload(rasters, coarse, ratios, tuple(chars), tuple(fonts))


def _directory():
    override = os.environ.get("SCREEN_GLYPH_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "ImagePosition" / "glyph-banks"


def _fingerprint(paths, characters, raster_size, coarse_size):
    fonts = []
    for path in paths:
        digest = sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024**2), b""):
                digest.update(chunk)
        fonts.append((path.name, digest.hexdigest()))
    recipe = (_RECIPE_VERSION, raster_size, coarse_size, cv2.__version__, PIL.__version__,
              features.version_module("freetype2"), np.__version__, sys.byteorder)
    return sha256(json.dumps((recipe, fonts, characters), separators=(",", ":")).encode("ascii")).hexdigest()


def _specification(count, raster_size, coarse_size, name_length):
    return {"rasters": ((count, raster_size, raster_size), np.dtype("uint8")),
            "coarse": ((count, coarse_size * coarse_size), np.dtype("uint8")),
            "ratios": ((count,), np.dtype("float32")),
            "chars": ((count,), np.dtype("U1")),
            "fonts": ((count,), np.dtype(f"U{name_length}")),
            "key": ((), np.dtype("S64"))}


def _array_info(source, member):
    with source.open(member) as entry:
        version = np.lib.format.read_magic(entry)
        if version == (1, 0):
            read_header, length_size = np.lib.format.read_array_header_1_0, 2
        elif version == (2, 0):
            read_header, length_size = np.lib.format.read_array_header_2_0, 4
        else:
            raise ValueError("Unsupported glyph cache array header")
        before_length = entry.tell()
        length = entry.read(length_size)
        if len(length) != length_size or int.from_bytes(length, "little") > _HEADER_LIMIT:
            raise ValueError("Invalid glyph cache array header length")
        entry.seek(before_length)
        shape, fortran, dtype = read_header(entry, max_header_size=_HEADER_LIMIT)
        if (fortran or dtype.hasobject or not all(type(size) is int and size >= 0 for size in shape)
                or entry.tell() + prod(shape) * dtype.itemsize != member.file_size):
            raise ValueError("Invalid glyph cache array")
        return shape, dtype


def _read(path, fingerprint, characters, names, raster_size, coarse_size):
    try:
        with path.open("rb") as binary:
            if os.fstat(binary.fileno()).st_size > _MAX_BANK_BYTES:
                return None
            with ZipFile(binary) as archive:
                members = archive.infolist()
                name_length = max(map(len, names))
                expected = _specification(0, raster_size, coarse_size, name_length)
                if (len(members) != len(expected) or {item.filename for item in members} !=
                        {f"{name}.npy" for name in expected}
                        or sum(item.file_size for item in members) > _MAX_BANK_BYTES
                        or any(item.compress_type != ZIP_STORED for item in members)):
                    return None
                actual = {item.filename.removesuffix(".npy"): _array_info(archive, item) for item in members}
                shape, _ = actual["rasters"]
                if (not shape or not 1 <= shape[0] <= len(names) * len(characters)
                        or actual != _specification(shape[0], raster_size, coarse_size, name_length)):
                    return None
            binary.seek(0)
            with np.load(binary, allow_pickle=False, max_header_size=_HEADER_LIMIT) as arrays:
                if arrays["key"].item() != fingerprint.encode("ascii"):
                    return None
                chars, fonts = tuple(arrays["chars"].tolist()), tuple(arrays["fonts"].tolist())
                if not set(chars) <= set(characters) or not set(fonts) <= set(names):
                    return None
                ratios = arrays["ratios"]
                if not np.isfinite(ratios).all():
                    return None
                return GlyphBankPayload(arrays["rasters"], arrays["coarse"], ratios, chars, fonts)
    except _CACHE_ERRORS:
        return None


def _lock(handle):
    # Closing the handle releases the lock even when a writer exits early.
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _write(path, bank, fingerprint, characters, names, raster_size, coarse_size):
    count = len(bank.chars)
    if (not 1 <= count <= len(characters) * len(names) or len(bank.fonts) != count
            or not set(bank.chars) <= set(characters) or not set(bank.fonts) <= set(names)):
        return
    spec = _specification(count, raster_size, coarse_size, max(map(len, names)))
    arrays = {"rasters": bank.rasters, "coarse": bank.coarse, "ratios": bank.ratios}
    if any((value.shape, value.dtype) != spec[name] for name, value in arrays.items()):
        return
    predicted_size = sum(prod(shape) * dtype.itemsize for shape, dtype in spec.values()) + 32768
    if predicted_size > _MAX_BANK_BYTES or not np.isfinite(bank.ratios).all():
        return
    arrays.update(chars=np.asarray(bank.chars, dtype=spec["chars"][1]),
                  fonts=np.asarray(bank.fonts, dtype=spec["fonts"][1]), key=np.asarray(fingerprint, dtype="S64"))
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with (path.parent / ".write-lock").open("a+b") as lock:
            _lock(lock)
            current = sum(item.stat().st_size for item in path.parent.glob(f"{_PREFIX}*")
                          if item.suffix in {".npz", ".tmp"})
            if current + predicted_size > _DISK_BUDGET:
                return
            with NamedTemporaryFile(dir=path.parent, prefix=_PREFIX, suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                np.savez(output, **{name: np.require(value, requirements="C") for name, value in arrays.items()})
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            temporary = None
    except _CACHE_ERRORS:
        pass
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def load_glyph_bank(paths, characters, *, raster_size, coarse_size, build):
    """Load a validated local bank, or return the result of calling ``build()``."""
    disabled = os.environ.get("SCREEN_GLYPH_CACHE_DISABLE", "").strip().lower() in {"1", "true", "yes", "on"}
    paths = tuple(Path(path) for path in paths)
    if disabled or not paths or not characters:
        return _payload(build())
    try:
        names = tuple(path.name for path in paths)
        fingerprint = _fingerprint(paths, characters, raster_size, coarse_size)
        destination = _directory() / f"{_PREFIX}{fingerprint}.npz"
    except _CACHE_ERRORS:
        return _payload(build())
    bank = _read(destination, fingerprint, characters, names, raster_size, coarse_size)
    if bank is not None:
        return bank
    bank = _payload(build())
    if bank is not None:
        try:
            unchanged = _fingerprint(paths, characters, raster_size, coarse_size) == fingerprint
        except _CACHE_ERRORS:
            unchanged = False
        if unchanged:
            _write(destination, bank, fingerprint, characters, names, raster_size, coarse_size)
    return bank
