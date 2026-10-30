# -*- coding: utf-8 -*-
"""
多视角采图向导工具 (Image Capture Suite)
单一职责: 纯预览 + 保存 (Tag 识别/解算由离线建图管线完成, 先采后验)
"""

from tools.capture.app import ImageCaptureApp, main

__all__ = ["ImageCaptureApp", "main"]
