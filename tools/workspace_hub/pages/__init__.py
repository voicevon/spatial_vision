# -*- coding: utf-8 -*-
"""
Workspace Hub 页面渲染组件模块 (Pages Package)
==============================================
对大型页面实施组件化拆分：
- DashboardPageRenderer: 工位大盘体检看板 (元数据/高精质检/观测有效性/3D拓扑概览)
- WhitelistPageRenderer: AprilTag 放行矩阵与锚点管理
- GalleryPageRenderer: 标定/生产相册卡片网格与全宽大图沉浸式预览
- FramesRoisPageRenderer: 机构坐标系树、Tag 专属分段、3D ROI 空间物件列表
"""

from tools.workspace_hub.pages.dashboard_page import DashboardPageRenderer
from tools.workspace_hub.pages.gallery_page import GalleryPageRenderer
from tools.workspace_hub.pages.frames_rois_page import FramesRoisPageRenderer

__all__ = [
    "DashboardPageRenderer",
    "GalleryPageRenderer",
    "FramesRoisPageRenderer",
]
