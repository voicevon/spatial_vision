# -*- coding: utf-8 -*-
"""
Workspace Hub 相册状态机 (GalleryState)
=====================================
管理工位三大专属图集与生产运行相册的数据流转：
1. 相机内参标定图集 (intrinsics): 棋盘格/网格标定板图片，专门用于求解物理焦距与畸变
2. 外参建图标定图集 (calibration): AprilTag 空间标靶照片，用于 BA 全局平差建图与世界对齐
3. 生产工件采样图集 (production): 生产现场工件检测/在席质检采样照片
4. 网格卡片分页偏移、滚动与选中索引对齐
5. 高性能内存 LRU 缩略图与单帧高清预览图缓存
6. 物理照片文件删除与安全自适应定位
7. 实时相机帧归档抓拍与白闪动效触发
8. 沉浸式全宽大图视图模式流转
"""

import os
import glob
import time
from collections import OrderedDict
from typing import Any, Optional, Tuple, List
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
    """三大图集与大图预览状态机"""

    # 视图模式 (全宽大图沉浸预览, 在相册页签下双击卡片展开)
    VIEW_STANDARD = "standard"    # 标准: 左栏 + 右侧页签内容
    VIEW_EXPANDED = "expanded"    # 全宽大图: 右侧区域整体铺满单帧大图

    # 相册卡片网格规格 (与渲染器保持一致): 3 列 x 3 行 = 每页 9 张大卡片
    GRID_COLS = 3
    GRID_ROWS = 3
    GRID_PAGE = 9

    def __init__(self, parent_hub_state: Any):
        self.hub = parent_hub_state

        # 1. 内参标定图集 (棋盘格/网格标定板)
        self.intrinsics_images: List[str] = []
        self.selected_intrinsics_image_idx = 0
        self.intrinsics_grid_offset = 0

        # 2. 外参建图图集 (Tag 空间位姿标靶)
        self.current_images: List[str] = []
        self.selected_image_idx = 0
        self.image_grid_offset = 0

        # 3. 生产采样图集 (工件检测采样)
        self.prod_images: List[str] = []
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
        """切换全宽大图模式与标准看板模式"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            allowed_tabs = (
                getattr(self.hub, "TAB_INTRINSICS_IMAGES", "tab_intrinsics_images"),
                self.hub.TAB_CALIB_IMAGES,
                self.hub.TAB_PROD_IMAGES,
            )
            if self.hub.active_tab not in allowed_tabs:
                self.hub.active_tab = self.hub.TAB_CALIB_IMAGES
            self.set_view_mode(self.VIEW_EXPANDED)

    # ------------------------------ 1. 内参标定图集 ------------------------------
    def load_intrinsics_images(self):
        """扫描并加载当前工位 intrinsics/raw_images 目录中的所有图片"""
        ws = self.hub.get_selected_workspace()
        if not ws or not os.path.exists(ws.intrinsics_raw_images_dir):
            self.intrinsics_images = []
            self.selected_intrinsics_image_idx = 0
            self.intrinsics_grid_offset = 0
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(ws.intrinsics_raw_images_dir, ext)))
        imgs.sort(key=lambda f: os.path.basename(f))
        self.intrinsics_images = imgs

        if self.intrinsics_images:
            self.selected_intrinsics_image_idx = max(0, min(self.selected_intrinsics_image_idx, len(self.intrinsics_images) - 1))
        else:
            self.selected_intrinsics_image_idx = 0
        self.intrinsics_grid_offset = self._clamp_grid_offset(self.intrinsics_grid_offset, len(imgs))

    def scroll_intrinsics_grid(self, delta_rows: int):
        """按行滚动内参卡片网格"""
        new_offset = self.intrinsics_grid_offset + delta_rows * self.GRID_COLS
        self.intrinsics_grid_offset = self._clamp_grid_offset(new_offset, len(self.intrinsics_images))

    def select_intrinsics_image_at_index(self, idx: int):
        """选中内参图集指定索引的照片"""
        if 0 <= idx < len(self.intrinsics_images):
            self.selected_intrinsics_image_idx = idx
            self._ensure_intrinsics_visible()

    def _ensure_intrinsics_visible(self):
        """确保选中的内参图片在网格内可见"""
        idx = self.selected_intrinsics_image_idx
        if idx < self.intrinsics_grid_offset:
            self.intrinsics_grid_offset = (idx // self.GRID_COLS) * self.GRID_COLS
        elif idx >= self.intrinsics_grid_offset + self.GRID_PAGE:
            target_row = idx // self.GRID_COLS
            self.intrinsics_grid_offset = (target_row - self.GRID_ROWS + 1) * self.GRID_COLS
        self.intrinsics_grid_offset = self._clamp_grid_offset(self.intrinsics_grid_offset, len(self.intrinsics_images))

    def select_intrinsics_image_by_offset(self, delta: int):
        """相对移动内参照片选择"""
        if not self.intrinsics_images:
            return
        new_idx = max(0, min(len(self.intrinsics_images) - 1, self.selected_intrinsics_image_idx + delta))
        self.select_intrinsics_image_at_index(new_idx)

    # ------------------------------ 2. 外参建图图集 ------------------------------
    def load_current_workspace_images(self):
        """扫描并加载当前选中工位 calibration/raw_images 目录中的所有图片"""
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
        """按行滚动外参卡片网格"""
        new_offset = self.image_grid_offset + delta_rows * self.GRID_COLS
        self.image_grid_offset = self._clamp_grid_offset(new_offset, len(self.current_images))

    def select_image_at_index(self, idx: int):
        """选中外参相册指定索引的照片"""
        if 0 <= idx < len(self.current_images):
            self.selected_image_idx = idx
            self._ensure_image_visible()

    def _ensure_image_visible(self):
        """确保当前选中的图片在外参相册网格视口内可见"""
        idx = self.selected_image_idx
        if idx < self.image_grid_offset:
            self.image_grid_offset = (idx // self.GRID_COLS) * self.GRID_COLS
        elif idx >= self.image_grid_offset + self.GRID_PAGE:
            target_row = idx // self.GRID_COLS
            self.image_grid_offset = (target_row - self.GRID_ROWS + 1) * self.GRID_COLS
        self.image_grid_offset = self._clamp_grid_offset(self.image_grid_offset, len(self.current_images))

    def select_image_by_offset(self, delta: int):
        """相对移动外参相册照片选择 (上一张/下一张)"""
        if not self.current_images:
            return
        new_idx = max(0, min(len(self.current_images) - 1, self.selected_image_idx + delta))
        self.select_image_at_index(new_idx)

    # ------------------------------ 3. 生产采样图集 ------------------------------
    def load_prod_images(self):
        """扫描并加载生产工位 production/raw_images 目录中的所有图片"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            self.prod_images = []
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            return

        prod_dir = ws.prod_raw_images_dir
        if not os.path.isdir(prod_dir):
            self.prod_images = []
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(prod_dir, ext)))
        imgs.sort(key=lambda f: os.path.basename(f))
        self.prod_images = imgs
        if self.selected_prod_image_idx >= len(imgs):
            self.selected_prod_image_idx = max(0, len(imgs) - 1)
        self.prod_grid_offset = self._clamp_grid_offset(self.prod_grid_offset, len(imgs))

    def scroll_prod_grid(self, delta_rows: int):
        """按行滚动生产相册卡片网格"""
        new_offset = self.prod_grid_offset + delta_rows * self.GRID_COLS
        self.prod_grid_offset = self._clamp_grid_offset(new_offset, len(self.prod_images))

    def select_prod_image_at_index(self, idx: int):
        """选中生产相册指定索引的照片"""
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

    # ------------------------------ 通用图集删除与查询 ------------------------------
    def get_active_images_and_index(self) -> Tuple[List[str], int, str]:
        """根据当前活跃页签返回对应的 (图片列表, 当前选中索引, 相册名称)"""
        tab_intrinsics = getattr(self.hub, "TAB_INTRINSICS_IMAGES", "tab_intrinsics_images")
        if self.hub.active_tab == tab_intrinsics:
            return self.intrinsics_images, self.selected_intrinsics_image_idx, "内参图集"
        elif self.hub.active_tab == self.hub.TAB_PROD_IMAGES:
            return self.prod_images, self.selected_prod_image_idx, "生产图集"
        else:
            return self.current_images, self.selected_image_idx, "外参图集"

    def delete_selected_image(self) -> bool:
        """删除当前选中的照片帧（自动根据当前页签分流）"""
        images, idx, album_name = self.get_active_images_and_index()

        if not images:
            self.hub.set_toast(f"当前工位【{album_name}】为空，无照片可删除。")
            return False

        if idx < 0 or idx >= len(images):
            return False

        img_path = images[idx]
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

            # 重新载入对应相册
            tab_intrinsics = getattr(self.hub, "TAB_INTRINSICS_IMAGES", "tab_intrinsics_images")
            if self.hub.active_tab == tab_intrinsics:
                self.load_intrinsics_images()
                if self.intrinsics_images:
                    self.selected_intrinsics_image_idx = min(idx, len(self.intrinsics_images) - 1)
                else:
                    self.selected_intrinsics_image_idx = 0
                self._ensure_intrinsics_visible()
            elif self.hub.active_tab == self.hub.TAB_PROD_IMAGES:
                self.load_prod_images()
                if self.prod_images:
                    self.selected_prod_image_idx = min(idx, len(self.prod_images) - 1)
                else:
                    self.selected_prod_image_idx = 0
                self._ensure_prod_visible()
            else:
                self.load_current_workspace_images()
                if self.current_images:
                    self.selected_image_idx = min(idx, len(self.current_images) - 1)
                else:
                    self.selected_image_idx = 0
                self._ensure_image_visible()

            # 同步更新工位对象的统计
            ws = self.hub.get_selected_workspace()
            if ws:
                ws.refresh_stats()

            self.hub.set_toast(f"已删除【{album_name}】照片: {file_name}")
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

    def save_capture_frame(self, raw_frame: np.ndarray, purpose: str = "calibration") -> str:
        """
        将当前相机帧归档至选中的工位沙盒指定的图集目录。
        :param raw_frame: 图像像素数组
        :param purpose: 'intrinsics' | 'calibration' | 'production'
        :return: 保存的绝对路径
        """
        ws = self.hub.get_selected_workspace()
        if not ws:
            return ""

        target_dir = ws.get_raw_images_dir(purpose)
        os.makedirs(target_dir, exist_ok=True)
        prefix = "intr_" if purpose == "intrinsics" else ("prod_" if purpose == "production" else "view_")
        existing = glob.glob(os.path.join(target_dir, f"{prefix}*.png"))
        max_idx = 0
        for f in existing:
            base = os.path.basename(f)
            num_part = base.replace(prefix, "").replace(".png", "")
            if num_part.isdigit():
                max_idx = max(max_idx, int(num_part))

        new_idx = max_idx + 1
        filename = f"{prefix}{new_idx:04d}.png"
        filepath = os.path.join(target_dir, filename)
        imwrite_unicode(filepath, raw_frame)

        # 触发白闪动效
        self.flash_timer = time.time() + 0.08

        # 刷新工位状态与相册加载
        ws.refresh_stats()
        ws.save_meta()

        if purpose == "intrinsics":
            self.load_intrinsics_images()
            self.selected_intrinsics_image_idx = len(self.intrinsics_images) - 1
            self.hub.set_toast(f"内参标定照片已保存: {filename} (图集累计 {len(self.intrinsics_images)} 帧)")
        elif purpose == "production":
            self.load_prod_images()
            self.selected_prod_image_idx = len(self.prod_images) - 1
            self.hub.set_toast(f"生产采样照片已保存: {filename} (图集累计 {len(self.prod_images)} 帧)")
        else:
            self.load_current_workspace_images()
            self.selected_image_idx = len(self.current_images) - 1
            self.hub.set_toast(f"外参建图快照已保存: {filename} (图集累计 {len(self.current_images)} 帧)")

        return filepath
