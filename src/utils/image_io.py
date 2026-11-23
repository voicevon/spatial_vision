# -*- coding: utf-8 -*-
"""
图像读写工具模块 (Image IO)
===========================
封装 Windows 平台兼容中文及特殊字符物理路径的 Unicode 安全图像读写函数。
利用 np.fromfile + cv2.imdecode 与 cv2.imencode + tofile 规避 OpenCV 原生
cv2.imread / cv2.imwrite 在 Windows 非 ASCII 路径下的读写崩溃/静默失败问题。
"""

import os
from typing import Optional
import cv2
import numpy as np


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """兼容 Windows 中文/特殊字符物理路径的鲁棒图像读取 (np.fromfile + cv2.imdecode)"""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        if data is None or len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(filepath: str, img: np.ndarray) -> bool:
    """兼容 Windows 中文/特殊字符物理路径的鲁棒图像写入 (cv2.imencode + tofile)"""
    try:
        ext = os.path.splitext(filepath)[1]
        if not ext:
            ext = ".png"
        ok, buf = cv2.imencode(ext, img)
        if ok and buf is not None:
            buf.tofile(filepath)
            return True
        return False
    except Exception:
        return False
