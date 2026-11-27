# -*- coding: utf-8 -*-
"""
Camera Hub 状态管理 (CameraHubState)
====================================
管理多相机资产列表、当前选中的相机与分辨率档案、标定板参数、
实时预览 / 图集审核双模式切换以及缩略图与逐帧重投影误差缓存。
"""

import os
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.cameras.camera_entity import CameraProfile, IntrinsicsRecord, parse_resolution
from src.cameras.camera_registry import CameraRegistry
from src.utils.image_io import imread_unicode


class CameraHubState:
    """Camera Hub 核心状态机"""

    MODE_LIVE = "live"        # 实时预览 + 棋盘格角点检测引导采图
    MODE_GALLERY = "gallery"  # 图集审核 + 逐帧误差 + 一键求解

    PRESET_RESOLUTIONS = ("1920x1080", "1280x720", "640x480")
    GRID_COLS = 3
    GRID_ROWS = 3
    GRID_PAGE = 9

    def __init__(self, registry: Optional[CameraRegistry] = None):
        self.registry = registry or CameraRegistry()
        self.cameras: List[CameraProfile] = []
        self.selected_camera_id: str = ""
        self.selected_resolution: str = "1920x1080"
        self.active_mode: str = self.MODE_LIVE

        # 棋盘格标定板参数（默认 9x6 内角点, 25.0 mm）
        self.pattern_cols: int = 9
        self.pattern_rows: int = 6
        self.square_size_mm: float = 25.0

        # 当前分辨率下的图集与标定结果状态
        self.images: List[str] = []
        self.selected_image_idx: int = 0
        self.grid_offset: int = 0
        self.current_intrinsics: Optional[IntrinsicsRecord] = None
        self.per_image_errors: Dict[str, Optional[float]] = {}  # filename -> error_px (None 表示未检出)
        self.per_image_detected: Dict[str, bool] = {}

        # 在线探测标记 (camera_id -> True)
        self.online_camera_ids: set[str] = set()

        # 实时预览与角点检测缓存
        self.pipeline_running: bool = False
        self.stream_desc: str = "相机未开启"
        self.live_frame: Optional[np.ndarray] = None
        self.live_corners_found: bool = False
        self.live_corners: Optional[np.ndarray] = None
        self.flash_timer: float = 0.0

        # 缩略图 LRU 缓存
        self._thumb_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._max_thumb_cache = 90

        self.refresh_cameras()

    @property
    def pattern_size(self) -> Tuple[int, int]:
        return (self.pattern_cols, self.pattern_rows)

    def get_selected_camera(self) -> Optional[CameraProfile]:
        for c in self.cameras:
            if c.camera_id == self.selected_camera_id:
                return c
        return None

    def get_available_resolutions(self) -> List[str]:
        res_list = list(self.PRESET_RESOLUTIONS)
        cam = self.get_selected_camera()
        if cam is not None:
            for r in cam.list_resolutions():
                if r not in res_list:
                    res_list.append(r)
        return res_list

    def refresh_cameras(self, select_id: Optional[str] = None):
        prev_id = select_id or self.selected_camera_id
        self.cameras = self.registry.list_cameras()
        if not self.cameras:
            self.selected_camera_id = ""
            self.images = []
            self.current_intrinsics = None
            self.per_image_errors.clear()
            self.per_image_detected.clear()
            return

        if prev_id and any(c.camera_id == prev_id for c in self.cameras):
            self.selected_camera_id = prev_id
        else:
            self.selected_camera_id = self.cameras[0].camera_id

        self.refresh_profile_data()

    def select_camera(self, camera_id: str):
        if any(c.camera_id == camera_id for c in self.cameras):
            self.selected_camera_id = camera_id
            cam = self.get_selected_camera()
            if cam is not None:
                existing_res = cam.list_resolutions()
                if existing_res and self.selected_resolution not in existing_res:
                    self.selected_resolution = existing_res[0]
            self.selected_image_idx = 0
            self.grid_offset = 0
            self.refresh_profile_data()

    def select_resolution(self, resolution: str):
        if parse_resolution(resolution) is None:
            return
        self.selected_resolution = resolution
        self.selected_image_idx = 0
        self.grid_offset = 0
        self.refresh_profile_data()

    def refresh_profile_data(self):
        cam = self.get_selected_camera()
        if cam is None:
            self.images = []
            self.current_intrinsics = None
            self.per_image_errors.clear()
            self.per_image_detected.clear()
            return

        self.images = cam.list_images(self.selected_resolution)
        if self.images:
            self.selected_image_idx = max(0, min(self.selected_image_idx, len(self.images) - 1))
        else:
            self.selected_image_idx = 0
        self.grid_offset = self._clamp_grid_offset(self.grid_offset, len(self.images))

        rec = cam.load_intrinsics(self.selected_resolution)
        self.current_intrinsics = rec
        self.per_image_errors.clear()
        self.per_image_detected.clear()
        if rec is not None:
            if len(rec.pattern_size) >= 2:
                self.pattern_cols = int(rec.pattern_size[0])
                self.pattern_rows = int(rec.pattern_size[1])
            if rec.square_size_mm > 0:
                self.square_size_mm = float(rec.square_size_mm)
            for item in rec.per_image_errors:
                fn = str(item.get("filename") or "")
                if fn:
                    self.per_image_detected[fn] = bool(item.get("detected", False))
                    self.per_image_errors[fn] = item.get("error_px")

    def _clamp_grid_offset(self, offset: int, total: int) -> int:
        if total <= self.GRID_PAGE:
            return 0
        max_offset = ((total - 1) // self.GRID_COLS) * self.GRID_COLS - (self.GRID_PAGE - self.GRID_COLS)
        max_offset = max(0, max_offset)
        clamped = max(0, min(offset, max_offset))
        return (clamped // self.GRID_COLS) * self.GRID_COLS

    def scroll_grid(self, delta_rows: int):
        self.grid_offset = self._clamp_grid_offset(
            self.grid_offset + delta_rows * self.GRID_COLS, len(self.images)
        )

    def adjust_pattern(self, d_cols: int = 0, d_rows: int = 0, d_square_mm: float = 0.0):
        self.pattern_cols = max(3, min(20, self.pattern_cols + d_cols))
        self.pattern_rows = max(3, min(20, self.pattern_rows + d_rows))
        self.square_size_mm = round(max(1.0, min(200.0, self.square_size_mm + d_square_mm)), 1)

    def get_thumbnail(self, img_path: str, tw: int, th: int) -> Optional[np.ndarray]:
        if not os.path.isfile(img_path):
            return None
        key = f"{img_path}_{tw}_{th}"
        if key in self._thumb_cache:
            self._thumb_cache.move_to_end(key)
            return self._thumb_cache[key]
        img = imread_unicode(img_path)
        if img is None:
            return None
        thumb = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        if len(self._thumb_cache) >= self._max_thumb_cache:
            self._thumb_cache.popitem(last=False)
        self._thumb_cache[key] = thumb
        return thumb

    def invalidate_thumb(self, img_path: str):
        for k in [k for k in self._thumb_cache if k.startswith(img_path)]:
            self._thumb_cache.pop(k, None)
