# -*- coding: utf-8 -*-
"""
Workspace Hub 相册状态机 (GalleryState)
=====================================
管理工位标定相册与生产相册的数据流转：
1. 标定/生产相册图片文件扫描与异步载入
2. 网格卡片分页偏移、滚动与选中索引对齐
3. 高性能内存 LRU 缩略图与单帧高清预览图缓存
4. 物理照片文件删除与安全自适应定位
5. 实时相机帧归档抓拍与白闪动效触发
6. 沉浸式全宽大图视图模式流转
"""

import os
import glob
import time
from collections import OrderedDict
from typing import Any, Optional
import cv2
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
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
        ok, buf = cv2.imencode(ext, img)
        if ok and buf is not None:
            buf.tofile(filepath)
            return True
        return False
    except Exception:
        return False


class GalleryState:
    """标定与生产相册状态机"""

    # 视图模式 (全宽大图沉浸预览, 仅在标定相册页签下双击卡片展开)
    VIEW_STANDARD = "standard"    # 标准: 左栏 + 右侧页签内容
    VIEW_EXPANDED = "expanded"    # 全宽大图: 右侧区域整体铺满单帧大图

    # 相册卡片网格规格 (与渲染器保持一致): 3 列 x 3 行 = 每页 9 张大卡片
    GRID_COLS = 3
    GRID_ROWS = 3
    GRID_PAGE = 9

    def __init__(self, parent_hub_state: Any):
        self.hub = parent_hub_state

        # 标定相册状态
        self.current_images: list[str] = []
        self.selected_image_idx = 0
        self.image_grid_offset = 0   # 卡片网格当前页起始索引 (按整行对齐)

        # 生产相册状态
        self.prod_images: list[str] = []
        self.selected_prod_image_idx = 0
        self.prod_grid_offset = 0

        # 视图模式
        self.view_mode = self.VIEW_STANDARD

        # 缩略图与单帧大图缓存 (LRU OrderedDict)
        self.thumbnail_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_cache_size = 120

        # 快门白闪动效截止时间戳
        self.flash_timer = 0.0

    @property
    def expanded_preview_mode(self) -> bool:
        """当处于全宽大图模式时返回 True"""
        return self.view_mode == self.VIEW_EXPANDED

    @expanded_preview_mode.setter
    def expanded_preview_mode(self, val: bool):
        self.view_mode = self.VIEW_EXPANDED if val else self.VIEW_STANDARD

    def set_view_mode(self, mode: str):
        """显式设定视图模式 (标准页签看板 / 全宽大图沉浸)"""
        if mode in (self.VIEW_STANDARD, self.VIEW_EXPANDED):
            self.view_mode = mode
            names = {
                self.VIEW_STANDARD: "标准页签看板",
                self.VIEW_EXPANDED: "全宽大图沉浸",
            }
            self.hub.set_toast(f"已切换视图模式: 【{names[mode]}】")

    def cycle_view_mode(self):
        """切换视图模式: 标准页签 <-> 全宽大图 (双击卡片触发)"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.set_view_mode(self.VIEW_EXPANDED)

    def toggle_expanded_preview(self):
        """切换全宽大图模式与标准看板模式 (全宽大图仅作用于标定相册页签)"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.hub.active_tab = self.hub.TAB_CALIB_IMAGES
            self.set_view_mode(self.VIEW_EXPANDED)

    def load_current_workspace_images(self):
        """扫描并加载当前选中工位 raw_images 目录中的所有图片"""
        ws = self.hub.get_selected_workspace()
        if not ws or not os.path.exists(ws.calib_raw_images_dir):
            self.current_images = []
            self.selected_image_idx = 0
            self.image_grid_offset = 0
            return

        imgs = sorted(glob.glob(os.path.join(ws.calib_raw_images_dir, "*.png")))
        self.current_images = imgs
        if self.current_images:
            self.selected_image_idx = max(0, min(self.selected_image_idx, len(self.current_images) - 1))
        else:
            self.selected_image_idx = 0
        self.image_grid_offset = self._clamp_grid_offset(self.image_grid_offset, len(imgs))

    def _clamp_grid_offset(self, offset: int, total: int) -> int:
        """限制偏移量在合理范围内，并按行对齐 (3的倍数)"""
        if total <= self.GRID_PAGE:
            return 0
        max_offset = ((total - 1) // self.GRID_COLS) * self.GRID_COLS - (self.GRID_PAGE - self.GRID_COLS)
        max_offset = max(0, max_offset)
        clamped = max(0, min(offset, max_offset))
        return (clamped // self.GRID_COLS) * self.GRID_COLS

    def scroll_image_grid(self, delta_rows: int):
        """按行滚动标定卡片网格 (每次滚动 delta_rows 行 = delta_rows * 3 张)"""
        new_offset = self.image_grid_offset + delta_rows * self.GRID_COLS
        self.image_grid_offset = self._clamp_grid_offset(new_offset, len(self.current_images))

    def select_image_at_index(self, idx: int):
        """选中标定相册指定索引的照片，并自动调整网格偏移使其可见"""
        if 0 <= idx < len(self.current_images):
            self.selected_image_idx = idx
            self._ensure_image_visible()

    def _ensure_image_visible(self):
        """确保当前选中的图片在标定相册网格视口内可见"""
        idx = self.selected_image_idx
        if idx < self.image_grid_offset:
            self.image_grid_offset = (idx // self.GRID_COLS) * self.GRID_COLS
        elif idx >= self.image_grid_offset + self.GRID_PAGE:
            target_row = idx // self.GRID_COLS
            self.image_grid_offset = (target_row - self.GRID_ROWS + 1) * self.GRID_COLS
        self.image_grid_offset = self._clamp_grid_offset(self.image_grid_offset, len(self.current_images))

    def select_image_by_offset(self, delta: int):
        """相对移动标定相册照片选择 (上一张/下一张)"""
        if not self.current_images:
            return
        new_idx = max(0, min(len(self.current_images) - 1, self.selected_image_idx + delta))
        self.select_image_at_index(new_idx)

    def load_prod_images(self):
        """扫描并加载生产基准工位 raw_images 目录中的所有图片"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            self.prod_images = []
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            return

        prod_dir = ws.calib_raw_images_dir
        if not os.path.isdir(prod_dir):
            self.prod_images = []
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(prod_dir, ext)))
        imgs.sort(key=lambda f: os.path.getmtime(f) if os.path.exists(f) else 0)
        self.prod_images = imgs
        if self.selected_prod_image_idx >= len(imgs):
            self.selected_prod_image_idx = max(0, len(imgs) - 1)
        self.prod_grid_offset = self._clamp_grid_offset(self.prod_grid_offset, len(imgs))

    def scroll_prod_grid(self, delta_rows: int):
        """按行滚动生产相册卡片网格"""
        new_offset = self.prod_grid_offset + delta_rows * self.GRID_COLS
        self.prod_grid_offset = self._clamp_grid_offset(new_offset, len(self.prod_images))

    def select_prod_image_at_index(self, idx: int):
        """选中生产相册指定索引的照片，并自动调整网格偏移使其可见"""
        if 0 <= idx < len(self.prod_images):
            self.selected_prod_image_idx = idx
            self._ensure_prod_visible()

    def _ensure_prod_visible(self):
        """确保当前选中的图片在生产相册网格视口内可见"""
        idx = self.selected_prod_image_idx
        if idx < self.prod_grid_offset:
            self.prod_grid_offset = (idx // self.GRID_COLS) * self.GRID_COLS
        elif idx >= self.prod_grid_offset + self.GRID_PAGE:
            target_row = idx // self.GRID_COLS
            self.prod_grid_offset = (target_row - self.GRID_ROWS + 1) * self.GRID_COLS
        self.prod_grid_offset = self._clamp_grid_offset(self.prod_grid_offset, len(self.prod_images))

    def select_prod_image_by_offset(self, delta: int):
        """相对移动生产相册照片选择"""
        if not self.prod_images:
            return
        new_idx = max(0, min(len(self.prod_images) - 1, self.selected_prod_image_idx + delta))
        self.select_prod_image_at_index(new_idx)

    def delete_selected_image(self) -> bool:
        """删除当前选中的照片帧（物理安全移除、清理缓存，并自适应指向相邻帧）"""
        if not self.current_images:
            self.hub.set_toast("当前工位相册为空，无照片可删除。")
            return False

        idx = self.selected_image_idx
        if idx < 0 or idx >= len(self.current_images):
            return False

        img_path = self.current_images[idx]
        file_name = os.path.basename(img_path)

        try:
            if os.path.exists(img_path):
                os.remove(img_path)

            # 清理缩略图与预览图缓存
            keys_to_del = [k for k in self.thumbnail_cache if k.startswith(img_path)]
            for k in keys_to_del:
                self.thumbnail_cache.pop(k, None)
            keys_to_del_prev = [k for k in self.preview_cache if k.startswith(img_path)]
            for k in keys_to_del_prev:
                self.preview_cache.pop(k, None)

            # 重新载入相册列表
            self.load_current_workspace_images()

            # 自适应定位相邻图片
            if self.current_images:
                self.selected_image_idx = min(idx, len(self.current_images) - 1)
            else:
                self.selected_image_idx = 0
            self._ensure_image_visible()

            # 同步更新工位对象的 image_count
            ws = self.hub.get_selected_workspace()
            if ws:
                ws.image_count = len(self.current_images)

            self.hub.set_toast(f"已删除照片: {file_name}")
            return True
        except Exception as e:
            self.hub.set_toast(f"删除照片失败: {e}")
            return False

    def get_thumbnail(self, img_path: str, tw: int = 110, th: int = 70) -> np.ndarray | None:
        """获取缩略图 (带 LRU 内存缓存)"""
        if not os.path.exists(img_path):
            return None
        key = f"{img_path}_{tw}_{th}"
        if key in self.thumbnail_cache:
            self.thumbnail_cache.move_to_end(key)
            return self.thumbnail_cache[key]

        bgr = imread_unicode(img_path)
        if bgr is None:
            return None
        thumb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

        if len(self.thumbnail_cache) >= self.max_cache_size:
            self.thumbnail_cache.popitem(last=False)
        self.thumbnail_cache[key] = thumb
        return thumb

    def get_preview(self, img_path: str, max_w: int = 440, max_h: int = 280) -> np.ndarray | None:
        """获取单帧高清预览图 (等比例缩放)"""
        if not os.path.exists(img_path):
            return None
        key = f"{img_path}_{max_w}_{max_h}"
        if key in self.preview_cache:
            self.preview_cache.move_to_end(key)
            return self.preview_cache[key]

        bgr = imread_unicode(img_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        prev = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        if len(self.preview_cache) >= self.max_cache_size:
            self.preview_cache.popitem(last=False)
        self.preview_cache[key] = prev
        return prev

    def save_capture_frame(self, raw_frame: np.ndarray) -> str:
        """将当前相机帧归档至选中的工位沙盒 raw_images 目录"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return ""

        os.makedirs(ws.calib_raw_images_dir, exist_ok=True)
        existing = glob.glob(os.path.join(ws.calib_raw_images_dir, "view_*.png"))
        max_idx = 0
        for f in existing:
            base = os.path.basename(f)
            num_part = base.replace("view_", "").replace(".png", "")
            if num_part.isdigit():
                max_idx = max(max_idx, int(num_part))

        new_idx = max_idx + 1
        filename = f"view_{new_idx:04d}.png"
        filepath = os.path.join(ws.calib_raw_images_dir, filename)
        imwrite_unicode(filepath, raw_frame)

        # 触发白闪动效
        self.flash_timer = time.time() + 0.08

        # 刷新工位状态
        ws.refresh_stats()
        ws.save_meta()
        self.load_current_workspace_images()
        self.selected_image_idx = len(self.current_images) - 1
        self.hub.set_toast(f"快照保存成功: {filename} (工位累计 {ws.image_count} 帧)")
        return filepath
