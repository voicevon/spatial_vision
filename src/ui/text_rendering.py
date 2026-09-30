#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通用文字渲染工具: TrueType 中文字体缓存 + OpenCV BGR 图像上的 PIL 高质量抗锯齿文本贴图。

下沉自 tools/gui_launcher (消除 src→tools 逆向依赖):
src 库层 (报告可视化) 与全部 GUI 工具统一从本模块导入, 依赖方向保持 tools→src 单向。

== 中文渲染规范 (全仓唯一文本管线) ==
禁止在图像绘制路径使用 cv2.putText / cv2.getTextSize (Hershey 字体仅含 ASCII,
非 ASCII 字符一律渲染为 '?')。统一使用本模块:
  - put_text     : 等价 cv2.putText, 参数顺序完全一致, drop-in 替换
  - measure_text : 等价 cv2.getTextSize, 返回结构相同 ((w, h), base_line)
  - fit_font_size: 按最大可用宽度求最大字号 (控件防溢出)
  - scale_to_font_size / resolve_bold : cv2 参数 → TTF 换算
  - draw_text    : 左上角定位的高层绘制接口 (GUI 主力)
绘制采用"文字覆盖率掩膜缓存 + 颜色注入合成": 同文本多次绘制只光栅化一次。
"""

import os
from collections import OrderedDict
from typing import Dict, Optional, Tuple

import cv2  # noqa: F401  (仅文档兼容性引用; 图像绘制路径禁止使用 cv2.putText)
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 字体对象缓存: (font_size, bold) -> FreeTypeFont
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}

# 文字覆盖率掩膜缓存: (text, font_size, bold) -> (mask uint8, ink_dx, ink_dy)
# ink_dx/dy 为墨迹左上角相对 PIL 绘制原点的偏移 (含 2px 抗锯齿边距)
_MASK_CACHE: "OrderedDict[Tuple[str, int, bool], Tuple[np.ndarray, int, int]]" = OrderedDict()
_MASK_CACHE_MAX = 1500


def get_cached_font(font_size: int = 16, bold: bool = False) -> ImageFont.FreeTypeFont:
    """获取缓存的 TrueType 中文字体"""
    key = (font_size, bold)
    if key not in _FONT_CACHE:
        font_paths = [
            "C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc",
            "C:/Windows/Fonts/simheittc.ttc" if bold else "C:/Windows/Fonts/simhei.ttf",
            "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()
        _FONT_CACHE[key] = font
    return _FONT_CACHE[key]


# ---------------------------------------------------------------------------
# cv2 兼容换算 API
# ---------------------------------------------------------------------------
def scale_to_font_size(font_scale: float) -> int:
    """cv2 font_scale → TTF 像素字号 (0.42→12, 0.52→15, 0.58→16, 与既有界面字号体系一致)"""
    return max(9, int(round(font_scale * 28)))


def resolve_bold(thickness: int, bold: Optional[bool] = None) -> bool:
    """cv2 thickness>=2 视为视觉加粗; 显式 bold 优先"""
    if bold is not None:
        return bool(bold)
    return thickness >= 2


def fit_font_size(text: str, max_width: int, start_size: int = 16, min_size: int = 9,
                  bold: bool = False) -> int:
    """给定最大可用宽度, 从 start_size 逐级递减求最大可用字号 (控件内文本防溢出)"""
    size = start_size
    while size > min_size:
        font = get_cached_font(size, bold)
        bbox = font.getbbox(text)
        if bbox[2] - bbox[0] <= max_width:
            return size
        size -= 1
    return min_size


def measure_text(text: str, font_face=None, font_scale: float = 0.5, thickness: int = 1,
                 font_size: Optional[int] = None, bold: Optional[bool] = None):
    """等价 cv2.getTextSize: 返回 ((宽度, 字高), 基线偏移)。

    字高按 ascent (基线以上高度) 给出, 与 cv2 同义; 基线偏移为 descent。
    """
    if bold is None:
        bold = resolve_bold(thickness)
    if font_size is None:
        font_size = scale_to_font_size(font_scale)
    font = get_cached_font(font_size, bold)
    bbox = font.getbbox(text)
    ascent, descent = font.getmetrics()
    return ((int(bbox[2] - bbox[0]), int(ascent)), int(descent))


# ---------------------------------------------------------------------------
# 掩膜缓存核心
# ---------------------------------------------------------------------------
def _get_text_mask(text: str, font_size: int, bold: bool) -> Tuple[np.ndarray, int, int]:
    """获取文字覆盖率掩膜 (uint8 0~255) 及墨迹相对绘制原点的偏移"""
    key = (text, font_size, bold)
    hit = _MASK_CACHE.get(key)
    if hit is not None:
        _MASK_CACHE.move_to_end(key)
        return hit

    font = get_cached_font(font_size, bold)
    bbox = font.getbbox(text)
    x0, y0 = int(bbox[0]), int(bbox[1])
    tw, th = int(bbox[2] - x0), int(bbox[3] - y0)
    pad = 2
    if tw <= 0 or th <= 0:
        mask = np.zeros((1, 1), np.uint8)
    else:
        pil = Image.new("L", (tw + pad * 2, th + pad * 2), 0)
        ImageDraw.Draw(pil).text((pad - x0, pad - y0), text, font=font, fill=255)
        mask = np.asarray(pil, dtype=np.uint8)
    ink_dx, ink_dy = x0 - pad, y0 - pad

    if len(_MASK_CACHE) >= _MASK_CACHE_MAX:
        _MASK_CACHE.popitem(last=False)
    _MASK_CACHE[key] = (mask, ink_dx, ink_dy)
    return mask, ink_dx, ink_dy


def _blit_mask(img: np.ndarray, text: str, origin: Tuple[int, int], font_size: int,
               color: Tuple[int, int, int], bold: bool) -> None:
    """将文本掩膜以指定 BGR 颜色合成到图像 (origin 为 PIL 绘制原点)"""
    mask, ink_dx, ink_dy = _get_text_mask(text, font_size, bold)
    mh, mw = mask.shape[:2]
    if mh <= 1 and mw <= 1 and mask[0, 0] == 0:
        return

    ix, iy = origin[0] + ink_dx, origin[1] + ink_dy
    h, w = img.shape[:2]
    # 掩膜侧裁剪 (墨迹可部分越界)
    mx0, my0 = max(0, -ix), max(0, -iy)
    mx1, my1 = mw - max(0, ix + mw - w), mh - max(0, iy + mh - h)
    if mx0 >= mx1 or my0 >= my1:
        return
    dx0, dy0 = max(0, ix), max(0, iy)
    dx1, dy1 = min(w, ix + mw), min(h, iy + mh)

    sub = img[dy0:dy1, dx0:dx1]
    m = mask[my0:my1, mx0:mx1].astype(np.float32) * (1.0 / 255.0)
    m3 = m[:, :, None]
    col = np.array([color[2], color[1], color[0]], dtype=np.float32)  # RGB→BGR
    blended = sub.astype(np.float32) * (1.0 - m3) + col * m3
    sub[:] = blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# 高层绘制 API
# ---------------------------------------------------------------------------
def draw_text(img: np.ndarray, text: str, pos: Tuple[int, int], font_size: int = 16,
              color: Tuple[int, int, int] = (240, 240, 240), bold: bool = False):
    """在 OpenCV BGR 图像上绘制高质量抗锯齿矢量文本 (支持中文)。pos 为文本左上角。"""
    if not text:
        return
    _blit_mask(img, text, (int(pos[0]), int(pos[1])), int(font_size), color, bool(bold))


def put_text(img, text: str, org, font_face=None, font_scale: float = 0.5, color=(255, 255, 255),
             thickness: int = 1, line_type=None, *, font_size: Optional[int] = None,
             bold: Optional[bool] = None, max_width: Optional[int] = None):
    """等价 cv2.putText 的中文安全绘制: 参数顺序完全一致, drop-in 替换。

    org 语义与 cv2 相同 (文本基线左端); thickness>=2 视觉加粗; line_type 忽略 (恒为抗锯齿)。
    可选 font_size 直接指定像素字号; 可选 max_width 自动缩字防溢出。
    """
    if not text:
        return
    if bold is None:
        bold = resolve_bold(thickness)
    if font_size is None:
        font_size = scale_to_font_size(font_scale)
    if max_width is not None:
        font_size = min(font_size, fit_font_size(text, max_width, font_size, bold=bold))
    font = get_cached_font(font_size, bold)
    ascent, _ = font.getmetrics()
    # cv2 org 为基线左端 → PIL 绘制原点 = 基线 - ascent
    _blit_mask(img, text, (int(org[0]), int(org[1]) - int(ascent)), font_size, color, bold)
