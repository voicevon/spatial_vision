"""
空间建图工作站 - 视口几何变换与鼠标交互层 (CanvasViewportInteractor)
================================================================================
负责工作站中栏核心视口的：
1. 缩放 (Zoom In / Zoom Out: 0.4x ~ 15.0x, 保持以光标为中心自适应无级缩放)
2. 平移 (Pan: 鼠标中键 / 右键拖拽平移)
3. 视口与图像坐标正逆映射 (Canvas 屏幕坐标 ⇋ 原始图像像素坐标)
4. 画布内标靶多边形点击命中测试 (Hit-Test Tag)
5. 视口渲染 ROI 裁剪与自适应求交计算
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


class CanvasViewportInteractor:
    """视口几何变换与鼠标交互控制器"""

    MIN_ZOOM: float = 0.4
    MAX_ZOOM: float = 15.0
    ZOOM_STEP_FACTOR: float = 1.15

    top_bar_h: int = 44
    bottom_bar_h: int = 52

    def __init__(self, top_bar_h: int = 44, bottom_bar_h: int = 52, win_w: int = 1920, win_h: int = 1080):
        self.top_bar_h = top_bar_h
        self.bottom_bar_h = bottom_bar_h
        self.win_w = win_w
        self.win_h = win_h
        self.zoom_level: float = 1.0
        self.pan_offset_x: float = 0.0
        self.pan_offset_y: float = 0.0
        self.is_panning: bool = False
        self.pan_start_pos: Tuple[int, int] = (0, 0)

    def sync_window_size(self, window_name: str) -> bool:
        """同步窗口大小变化（若 OpenCV 窗口被用户拖拽改变尺寸）"""
        try:
            rect = cv2.getWindowImageRect(window_name)
            if rect and len(rect) >= 4:
                _, _, w, h = rect
                if w > 100 and h > 100 and (w != self.win_w or h != self.win_h):
                    self.win_w = w
                    self.win_h = h
                    return True
        except Exception:
            pass  # GUI 可选功能：窗口矩形查询失败按尺寸未变处理
        return False

    def reset(self):
        """重置视口缩放与平移为适应屏幕初始状态 (1.0x)"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.is_panning = False
        self.pan_start_pos = (0, 0)

    def zoom_at(self, mx: int, my: int, wheel_up: bool, viewport_rect: Tuple[int, int, int, int]):
        """
        以当前鼠标屏幕坐标为固定锚点，对视口进行无级缩放
        :param mx: 鼠标在画布上的 X 像素坐标
        :param my: 鼠标在画布上的 Y 像素坐标
        :param wheel_up: True 为滚轮向上放大，False 为滚轮向下缩小
        :param viewport_rect: 视口边界矩形 (x, y, w, h)
        """
        vx, vy, vw, vh = viewport_rect
        factor = self.ZOOM_STEP_FACTOR if wheel_up else (1.0 / self.ZOOM_STEP_FACTOR)
        new_zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, self.zoom_level * factor))
        if abs(new_zoom - self.zoom_level) < 1e-6:
            return

        ratio = new_zoom / self.zoom_level
        curr_center_x = vx + vw / 2.0 + self.pan_offset_x
        curr_center_y = vy + vh / 2.0 + self.pan_offset_y

        dx = mx - curr_center_x
        dy = my - curr_center_y
        new_center_x = mx - dx * ratio
        new_center_y = my - dy * ratio

        self.pan_offset_x = new_center_x - (vx + vw / 2.0)
        self.pan_offset_y = new_center_y - (vy + vh / 2.0)
        self.zoom_level = new_zoom

    def start_pan(self, mx: int, my: int):
        """开始拖拽平移"""
        self.is_panning = True
        self.pan_start_pos = (mx, my)

    def update_pan(self, mx: int, my: int) -> bool:
        """
        更新拖拽平移位置
        :return: 是否发生了实际平移
        """
        if not self.is_panning:
            return False
        dx = mx - self.pan_start_pos[0]
        dy = my - self.pan_start_pos[1]
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        self.pan_start_pos = (mx, my)
        return True

    def end_pan(self):
        """结束拖拽平移"""
        self.is_panning = False

    def compute_image_rect(self, viewport_rect: Tuple[int, int, int, int], 
                           frame_w: int, frame_h: int) -> Tuple[int, int, int, int, float]:
        """
        计算图像在视口世界中经缩放与平移后的虚拟显示矩形与当前比例尺
        :return: (img_x1, img_y1, img_x2, img_y2, curr_scale)
        """
        vx, vy, vw, vh = viewport_rect
        base_scale = min(vw / frame_w, vh / frame_h)
        curr_scale = base_scale * self.zoom_level
        target_w = int(round(frame_w * curr_scale))
        target_h = int(round(frame_h * curr_scale))

        center_x = vx + vw / 2.0 + self.pan_offset_x
        center_y = vy + vh / 2.0 + self.pan_offset_y

        img_x1 = int(round(center_x - target_w / 2.0))
        img_y1 = int(round(center_y - target_h / 2.0))
        img_x2 = img_x1 + target_w
        img_y2 = img_y1 + target_h
        return img_x1, img_y1, img_x2, img_y2, curr_scale

    def screen_to_image_coords(self, mx: int, my: int, 
                               viewport_rect: Tuple[int, int, int, int], 
                               frame_w: int, frame_h: int) -> Optional[Tuple[float, float]]:
        """
        将画布屏幕坐标反向投影转换为原始图像像素坐标
        :return: (img_px, img_py) 或 None (若超出图像范围)
        """
        img_x1, img_y1, img_x2, img_y2, curr_scale = self.compute_image_rect(viewport_rect, frame_w, frame_h)
        if not (img_x1 <= mx <= img_x2 and img_y1 <= my <= img_y2):
            return None
        if curr_scale <= 1e-6:
            return None
        img_px = (mx - img_x1) / float(curr_scale)
        img_py = (my - img_y1) / float(curr_scale)
        return img_px, img_py

    def hit_test_tag(self, mx: int, my: int, observations: List[Dict[str, Any]], 
                     viewport_rect: Tuple[int, int, int, int], 
                     frame_w: int, frame_h: int, 
                     tolerance_px: float = 8.0) -> Optional[int]:
        """
        在视口中点击测试是否命中了某个标靶
        :param mx: 鼠标点击 X
        :param my: 鼠标点击 Y
        :param observations: 当前帧的观测列表
        :param viewport_rect: 视口矩形 (x, y, w, h)
        :param frame_w: 原始图宽
        :param frame_h: 原始图高
        :param tolerance_px: 多边形外边缘容差 (px)
        :return: 命中标靶 tag_id 或 None
        """
        pt = self.screen_to_image_coords(mx, my, viewport_rect, frame_w, frame_h)
        if pt is None:
            return None
        img_px, img_py = pt

        for obs in observations:
            corners = obs.get("corners")
            if corners is None:
                continue
            c_pts = np.array(corners, dtype=np.float32).reshape((4, 2))
            dist = cv2.pointPolygonTest(c_pts, (img_px, img_py), measureDist=True)
            if dist >= -tolerance_px:
                return obs.get("tag_id")
        return None

    def compute_viewport_render_rois(self, viewport_rect: Tuple[int, int, int, int], 
                                    frame_w: int, frame_h: int) -> Optional[Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]]:
        """
        计算视口渲染时，原始图像源切片 (src_roi: x1, y1, x2, y2) 与画布目标切片 (dst_roi: x1, y1, x2, y2) 的求交映射
        :return: (src_roi, dst_roi) 或 None (若完全不可见)
        """
        vx, vy, vw, vh = viewport_rect
        img_x1, img_y1, img_x2, img_y2, _ = self.compute_image_rect(viewport_rect, frame_w, frame_h)
        target_w = img_x2 - img_x1
        target_h = img_y2 - img_y1

        dst_x1 = max(vx, img_x1)
        dst_y1 = max(vy, img_y1)
        dst_x2 = min(vx + vw, img_x2)
        dst_y2 = min(vy + vh, img_y2)

        if dst_x2 <= dst_x1 or dst_y2 <= dst_y1 or target_w <= 0 or target_h <= 0:
            return None

        rel_x1 = (dst_x1 - img_x1) / float(target_w)
        rel_y1 = (dst_y1 - img_y1) / float(target_h)
        rel_x2 = (dst_x2 - img_x1) / float(target_w)
        rel_y2 = (dst_y2 - img_y1) / float(target_h)

        src_x1 = max(0, min(frame_w - 1, int(round(rel_x1 * frame_w))))
        src_y1 = max(0, min(frame_h - 1, int(round(rel_y1 * frame_h))))
        src_x2 = max(src_x1 + 1, min(frame_w, int(round(rel_x2 * frame_w))))
        src_y2 = max(src_y1 + 1, min(frame_h, int(round(rel_y2 * frame_h))))

        return (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2)
