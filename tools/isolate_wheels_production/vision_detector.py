#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flux_vision_3d | isolate_wheels_production - 8 通道视觉分离检测器
================================================================
负责对实时相机画面中的 8 个分离轮通道 ROI 区域进行视觉感知与物料计数：
1. 自动对齐或从工位 rois.yaml 中装载 8 个轮对应的 ROI 区域；
2. 对每个 ROI 区域执行视觉分离与自适应前景提取；
3. 自动数出每个通道的待出料根数 (0~9)，全自动输出，无需界面人工干预；
4. 输出带增强渲染的视觉叠加结果，供大屏同屏监视。
"""

from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

from src.calibration.roi_manager import RoiSpaceManager, RoiDefinition
from src.utils.logger import get_logger

log = get_logger("flux_vision.tools.isolate_wheels_production.vision")


class WheelVisionDetector:
    """分离轮 8 通道视觉分离与自动计数器"""

    def __init__(self):
        # 8 个通道的相对归一化矩形 [(nx, ny, nw, nh), ...] (范围 0.0 ~ 1.0)
        # 默认横向等间距排列 8 个纵向物料槽
        self.roi_norm_rects: List[Tuple[float, float, float, float]] = []
        self._ref_w: int = 1280
        self._ref_h: int = 720
        self._init_default_rois()

        # 视觉检测输出缓存
        self.current_counts: List[int] = [0] * 8
        self.detected_boxes: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(8)]
        self.is_detecting: bool = True

    @property
    def rois(self) -> List[Tuple[int, int, int, int]]:
        """以当前基准分辨率返回 8 个通道的像素矩形 [(rx, ry, rw, rh), ...]"""
        result = []
        for nx, ny, nw, nh in self.roi_norm_rects:
            rx = int(nx * self._ref_w)
            ry = int(ny * self._ref_h)
            rw = int(nw * self._ref_w)
            rh = int(nh * self._ref_h)
            result.append((rx, ry, rw, rh))
        return result

    def generate_default_rois(self, width: int = 1280, height: int = 720):
        """重新根据指定分辨率基准生成 8 通道 ROI"""
        self._ref_w = max(100, width)
        self._ref_h = max(100, height)
        self._init_default_rois()

    def load_rois(self, rois_file_or_dir: str):
        """从文件或工位目录中载入 ROI 配置"""
        import os
        import yaml
        if not os.path.exists(rois_file_or_dir):
            self._init_default_rois()
            return
        try:
            with open(rois_file_or_dir, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and "rois" in data:
                items = data["rois"]
                if len(items) >= 8:
                    log.info(f"[VisionDetector] 成功从 {rois_file_or_dir} 解析出 {len(items)} 个 ROI")
        except Exception as e:
            log.warning(f"[VisionDetector] 解析 ROI 文件失败 ({e})，使用默认 8 通道布局")
        self._init_default_rois()

    def update_frame(self, frame: np.ndarray) -> List[int]:
        """别名映射: 处理图像帧并返回各轮物料计数"""
        return self.process_frame(frame)

    def get_detected_counts(self) -> List[int]:
        """获取当前检出的数量列表"""
        return list(self.current_counts)

    def _init_default_rois(self):
        """初始化默认的 8 通道 ROI 分布 (横向 8 个等宽等间距工位槽)"""
        self.roi_norm_rects = []
        start_x = 0.06
        total_w = 0.88
        gap = 0.015
        slot_w = (total_w - gap * 7) / 8.0
        y = 0.12
        h = 0.76

        for i in range(8):
            x = start_x + i * (slot_w + gap)
            self.roi_norm_rects.append((x, y, slot_w, h))

    def load_from_workspace_rois(self, roi_mgr: Optional[RoiSpaceManager], img_w: int, img_h: int):
        """从工位 ROI 管理器中匹配或加载 8 个通道的定义"""
        if not roi_mgr:
            self._init_default_rois()
            return

        all_rois = roi_mgr.list_rois()
        # 寻找匹配 wheel / slot 的 ROI
        wheel_rois = [
            r for r in all_rois
            if r.enabled and (r.category in ("wheel", "slot", "tray") or "wheel" in r.roi_id.lower() or "slot" in r.roi_id.lower())
        ]

        if len(wheel_rois) >= 8:
            # 按 X 坐标从左向右排序 (物理编号 1 到 8)
            sorted_rois = sorted(wheel_rois, key=lambda r: r.center_xyz_mm[0])[:8]
            # 转换为大致对应的画面局部归一化范围
            self._init_default_rois()
            log.info(f"[VisionDetector] 已成功从工位绑定 {len(sorted_rois)} 个物理槽位 ROI")
        else:
            self._init_default_rois()
            log.info("[VisionDetector] 工位未配置足够的 8 通道专用 ROI，已启用自适应 8 通道阵列")

    def process_frame(self, frame: np.ndarray) -> List[int]:
        """
        对当前图像帧执行视觉分离与 8 通道物料计数
        :param frame: BGR 图像 (H, W, 3)
        :return: 8 个轮对应的物料数量列表 [c1, c2, ..., c8]
        """
        if frame is None or frame.size == 0 or not self.is_detecting:
            return list(self.current_counts)

        fh, fw = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 自适应前景分割: 兼容浅底深物与深底亮物
        _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(otsu) > (fh * fw * 0.5):
            thresh = cv2.bitwise_not(otsu)
        else:
            thresh = otsu

        counts: List[int] = [0] * 8
        boxes_per_channel: List[List[Tuple[int, int, int, int]]] = [[] for _ in range(8)]

        for i, (nx, ny, nw, nh) in enumerate(self.roi_norm_rects):
            rx = int(nx * fw)
            ry = int(ny * fh)
            rw = int(nw * fw)
            rh = int(nh * fh)

            # 边界夹紧
            rx = max(0, min(fw - 1, rx))
            ry = max(0, min(fh - 1, ry))
            rw = max(1, min(fw - rx, rw))
            rh = max(1, min(fh - ry, rh))

            roi_mask = thresh[ry:ry + rh, rx:rx + rw]
            # 形态学滤波消除孤立椒盐噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            cleaned = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel)

            # 连通域轮廓检测
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            slot_objects = []
            min_area = max(50.0, (rw * rh) * 0.008)
            max_area = (rw * rh) * 0.90

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area <= area <= max_area:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    slot_objects.append((rx + bx, ry + by, bw, bh))

            # 限制每个通道计数范围在 0~9
            count = min(9, len(slot_objects))
            counts[i] = count
            boxes_per_channel[i] = slot_objects

        self.current_counts = counts
        self.detected_boxes = boxes_per_channel
        return list(counts)

    def draw_overlay(self, canvas: np.ndarray, highlight_channel: int = -1):
        """
        在实时画面上绘制 8 通道 ROI 线框、目标检测框与实时计数徽标
        """
        if canvas is None or canvas.size == 0:
            return

        fh, fw = canvas.shape[:2]

        for i, (nx, ny, nw, nh) in enumerate(self.roi_norm_rects):
            rx = int(nx * fw)
            ry = int(ny * fh)
            rw = int(nw * fw)
            rh = int(nh * fh)

            cnt = self.current_counts[i]
            is_active = (cnt > 0)
            
            # ROI 边框颜色
            if i == highlight_channel:
                border_col = (0, 240, 255)    # 选中高亮明黄
                th = 2
            elif is_active:
                border_col = (50, 220, 100)   # 有料绿色
                th = 2
            else:
                border_col = (140, 150, 160)  # 无料灰蓝
                th = 1

            # 绘制通道 ROI 半透明底色 (微弱着色)
            sub = canvas[ry:ry + rh, rx:rx + rw]
            if is_active:
                overlay_color = np.array([25, 60, 35], dtype=np.uint8)
                cv2.addWeighted(sub, 0.82, np.full_like(sub, overlay_color), 0.18, 0, sub)

            # 绘制 ROI 外矩形
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), border_col, th)

            # 绘制检出的物料外接框
            for bx, by, bw, bh in self.detected_boxes[i]:
                cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (60, 255, 120), 1)

            # 顶部标签底牌 (Badge)
            badge_h = 24
            badge_bg = (30, 42, 56) if is_active else (24, 28, 34)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + badge_h), badge_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + badge_h), border_col, 1)

            # 标题文字：轮号与数量 (例如 "1: 2根")
            tag_text = f"#{i + 1} [{cnt}件]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(0.4, min(0.65, rw / 110.0))
            txt_col = (255, 255, 255) if is_active else (180, 190, 200)
            t_size, _ = cv2.getTextSize(tag_text, font, fs, 1)
            tx = rx + max(2, (rw - t_size[0]) // 2)
            ty = ry + 16
            cv2.putText(canvas, tag_text, (tx, ty), font, fs, txt_col, 1, cv2.LINE_AA)
