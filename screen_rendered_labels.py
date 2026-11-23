"""Match complete command captions rendered from local fonts."""

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COMMANDS = (
    "Cancel", "Go to login", "Log in", "Login", "Sign in", "Register", "Sign up with email",
    "Continue with Apple", "Continue with Google", "Continue with Facebook", "Continue with Microsoft",
    "Allow", "Deny", "Deny access", "Don't allow", "Once", "Allow once", "Only this time",
    "Allow only while using the app", "While using the app", "OK", "Continue", "Done", "Save",
    "Confirm", "Submit", "Agree", "Next", "Start", "Apply", "Accept", "Back", "Close",
)


def _mask_data(mask):
    x, y, w, h = cv2.boundingRect(mask)
    if not w or not h:
        return None
    ratio = w / h
    out = cv2.resize(np.float32(mask[y:y + h, x:x + w] > 0),
                     (max(8, round(24 * ratio)), 24), interpolation=cv2.INTER_AREA)
    return out, ratio


@lru_cache(maxsize=2000)
def _font_label(path, name):
    font = _system_font(path)
    l, t, r, b = _font_bounds(path, name)
    image = Image.new("L", (r - l, b - t))
    ImageDraw.Draw(image).text((-l, -t), name, font=font, fill=255)
    return _mask_data(np.uint8(np.asarray(image) >= 156))


@lru_cache(maxsize=12)
def _system_font(path):
    return ImageFont.truetype(str(path), 59)


@lru_cache(maxsize=2000)
def _font_bounds(path, name):
    return _system_font(path).getbbox(name)


def rank_rendered_label(mask, index, *, limit=4):
    if (data := _mask_data(mask)) is None:
        return []
    image, ratio = data
    variants = {}
    directory = index.choice_latin_sources[0].parent if index.choice_latin_sources else None
    fonts = tuple(path for name in ("arial.ttf", "arialbd.ttf", "segoeui.ttf", "segoeuib.ttf", "calibri.ttf",
                                     "calibrib.ttf", "tahoma.ttf", "tahomabd.ttf")
                  if directory is not None and (path := directory / name).is_file())
    for name in COMMANDS:
        for written in (name, name.upper()):
            for path in fonts:
                l, t, r, b = _font_bounds(path, written)
                if abs(np.log((r - l) / ((b - t) * ratio))) > .30:
                    continue
                if (template_info := _font_label(path, written)) is None:
                    continue
                template, expected = template_info
                ratio_error = abs(np.log(expected / ratio))
                if ratio_error > .19:
                    continue
                resized = cv2.resize(template, image.shape[::-1], interpolation=cv2.INTER_LINEAR)
                error = np.abs(image - resized).sum() / max(1, (image + resized).sum()) + .19 * ratio_error
                if error < variants.get(name, 1):
                    variants[name] = float(error)
    return sorted(variants.items(), key=lambda pair: pair[1])[:limit]
