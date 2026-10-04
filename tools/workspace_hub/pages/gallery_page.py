# -*- coding: utf-8 -*-
"""
Workspace Hub 相册与大图预览页面渲染器 (GalleryPageRenderer)
=========================================================
负责：
1. 相机内参标定图集 (棋盘格/圆形标定板，工位沙盒内参看板，一键相机采集向导与内参求解)
2. 外参建图相册网格 (AprilTag 空间建图图集，快捷采图入口)
3. 生产采样相册网格 (现场工件与在席质检图集，快捷采图入口)
4. 全宽自适应大图视口 (双击卡片放大，上张/下张/删帧/返回)
"""

import os
import cv2
import numpy as np
from typing import Any

from src.ui.gui_components import draw_text, put_text
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.states.gallery_state import GalleryState


class GalleryPageRenderer:
    """相册与大图预览页面渲染器"""

    def __init__(self, parent_renderer: Any):
        self.r = parent_renderer

    def render_intrinsics_images(self, canvas: np.ndarray, state: HubState, ws: Any):
        """页签: 相机内参标定图集与工位内参参数看板 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, self.r.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.r.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 顶部工位专属相机内参看板条 (y: 56~122)
        bar_x = box_x + 12
        bar_y = box_y + 8
        bar_w = box_w - 24
        bar_h = 68
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (22, 27, 36), -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (38, 48, 64), 1)

        intr_data = ws.load_camera_intrinsics() if ws else None
        has_custom_intr = bool(intr_data and intr_data.get("calibrated", False))

        # 状态标牌
        status_bg = (18, 55, 36) if has_custom_intr else (55, 42, 18)
        status_border = (0, 230, 80) if has_custom_intr else (0, 180, 240)
        cv2.rectangle(canvas, (bar_x + 10, bar_y + 10), (bar_x + 115, bar_y + 34), status_bg, -1)
        cv2.rectangle(canvas, (bar_x + 10, bar_y + 10), (bar_x + 115, bar_y + 34), status_border, 1)
        status_txt = "已工位标定" if has_custom_intr else "硬件缺省/名义"
        status_col = (0, 255, 180) if has_custom_intr else (0, 200, 240)
        draw_text(canvas, status_txt, (bar_x + 16, bar_y + 15), font_size=12, color=status_col, bold=True)

        # 参数文本渲染
        if intr_data:
            fx = intr_data.get("fx", 0.0)
            fy = intr_data.get("fy", 0.0)
            cx = intr_data.get("cx", 0.0)
            cy = intr_data.get("cy", 0.0)
            w = intr_data.get("width", 0)
            h = intr_data.get("height", 0)
            rmse = intr_data.get("rmse", 0.0)
            param_str = f"fx: {fx:.1f}  fy: {fy:.1f}  cx: {cx:.1f}  cy: {cy:.1f} | 规格: {w}x{h}"
            if rmse > 0:
                param_str += f" | RMSE: {rmse:.3f} px"
        else:
            param_str = "当前工位尚未写入独立标定内参，系统正回退使用硬件名义内参"

        draw_text(canvas, param_str, (bar_x + 125, bar_y + 15), font_size=12, color=(200, 215, 230))
        tip_str = f"工位图集照片 ({len(state.gallery.intrinsics_images)} 张) | 建议采集 15~25 张不同角度距离的棋盘格以求解高精度物理焦距"
        draw_text(canvas, tip_str, (bar_x + 12, bar_y + 44), font_size=11, color=(140, 155, 175))

        # 右侧操作按钮组: [+ 采集相片] [求解内参]
        btn_capture_rect = (bar_x + bar_w - 230, bar_y + 16, 105, 36)
        btn_calib_rect = (bar_x + bar_w - 115, bar_y + 16, 105, 36)
        self.r._draw_button(canvas, btn_capture_rect, "+ 采集相片", mpos, theme_color=(0, 220, 160))
        self.r._draw_button(canvas, btn_calib_rect, "求解内参", mpos, theme_color=(0, 200, 240))

        # 2. 下方相册网格卡片墙 (从 y=134 开始)
        self.render_gallery_page(
            canvas, state, "",
            state.gallery.intrinsics_images,
            state.gallery.selected_intrinsics_image_idx,
            state.gallery.intrinsics_grid_offset,
            empty_hint="当前工位尚未采集棋盘格内参标定图片！",
            is_calib=True,
            y_offset=74,
            purpose="intrinsics",
        )

    def render_calib_images(self, canvas: np.ndarray, state: HubState, sc: Any):
        """页签: 外参建图图集 - 当前选中工位的 AprilTag 采样图集 (x: 340~960, y: 50~670)"""
        self.render_gallery_page(
            canvas, state, "",
            state.gallery.current_images, state.gallery.selected_image_idx, state.gallery.image_grid_offset,
            empty_hint="当前工位尚未采集外参建图照片！",
            is_calib=True,
            purpose="calibration",
        )

    def render_prod_images(self, canvas: np.ndarray, state: HubState, sc: Any):
        """页签: 生产采样图集 - 生产现场工件与在席质检采样 (x: 340~960, y: 50~670)"""
        if not state.gallery.prod_images and sc and os.path.isdir(sc.prod_raw_images_dir):
            state.gallery.load_prod_images()
        prod_images = state.gallery.prod_images
        prod_idx = state.gallery.selected_prod_image_idx
        prod_offset = state.gallery.prod_grid_offset
        self.render_gallery_page(
            canvas, state, "",
            prod_images, prod_idx, prod_offset,
            empty_hint="生产基准工位尚未采集任何照片！",
            is_calib=False,
            purpose="production",
        )

    def render_gallery_page(self, canvas: np.ndarray, state: HubState, title: str,
                            images: list[str], sel_idx: int, grid_offset: int,
                            empty_hint: str, is_calib: bool, y_offset: int = 0,
                            purpose: str = "calibration"):
        """渲染通用图片卡片网格墙: 3列x3行大卡片网格 (双击卡片放大)"""
        box_x = 340
        box_y = 50 + y_offset
        box_w = self.r.canvas_w - 340
        box_h = 620 - y_offset
        if y_offset == 0:
            cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.r.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 外参建图与生产图集：在顶栏右侧常驻 [+ 采集相片] 按钮 (y: 56~88)
        if y_offset == 0:
            btn_top_capture = (box_x + box_w - 125, box_y + 6, 110, 32)
            self.r._draw_button(canvas, btn_top_capture, "+ 采集相片", mpos, theme_color=(0, 220, 160))

        # 空状态
        if not images:
            empty_box_y = box_y + (100 if y_offset == 0 else 60)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 120), (22, 26, 36), -1)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 120), self.r.COLOR_BORDER, 1)
            approx_w = sum(18 if ord(c) > 127 else 10 for c in empty_hint)
            hint_x = box_x + max(20, (box_w - approx_w) // 2)
            draw_text(canvas, empty_hint, (hint_x, empty_box_y + 26), font_size=16, color=(0, 200, 240), bold=True)

            # 居中快捷采图按钮
            btn_w = 160
            btn_x = box_x + (box_w - btn_w) // 2
            btn_y = empty_box_y + 66
            self.r._draw_button(canvas, (btn_x, btn_y, btn_w, 36), "+ 打开相机采集", mpos, theme_color=(0, 220, 160))
            return

        # 卡片网格
        grid_y0 = self.r.GRID_Y0 + y_offset
        visible_imgs = images[grid_offset: grid_offset + GalleryState.GRID_PAGE]

        for i, img_path in enumerate(visible_imgs):
            real_idx = grid_offset + i
            row, col = divmod(i, GalleryState.GRID_COLS)
            x = self.r.GRID_X0 + col * (self.r.GRID_CELL_W + self.r.GRID_GAP_X)
            y = grid_y0 + row * (self.r.GRID_CELL_H + self.r.GRID_GAP_Y)
            if y + self.r.GRID_CELL_H > box_y + box_h + 10:
                continue

            is_cur = (real_idx == sel_idx)
            is_hover = (x <= mpos[0] <= x + self.r.GRID_CELL_W and y <= mpos[1] <= y + self.r.GRID_CELL_H)

            card_bg = (30, 40, 36) if is_cur else ((28, 33, 41) if is_hover else (22, 26, 36))
            card_border = (0, 255, 180) if is_cur else ((0, 200, 240) if is_hover else (38, 44, 58))
            cv2.rectangle(canvas, (x, y), (x + self.r.GRID_CELL_W, y + self.r.GRID_CELL_H), card_bg, -1)
            cv2.rectangle(canvas, (x, y), (x + self.r.GRID_CELL_W, y + self.r.GRID_CELL_H), card_border, 2 if is_cur else 1)

            # 缩略图
            tw, th = self.r.GRID_CELL_W - 8, self.r.GRID_THUMB_H
            thumb = state.gallery.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y + 4:y + 4 + th, x + 4:x + 4 + tw] = thumb
            else:
                draw_text(canvas, "读取失败", (x + 60, y + 74), font_size=13, color=(120, 120, 140))

            # 左上角序号徽章
            cv2.rectangle(canvas, (x + 4, y + 4), (x + 56, y + 26), (8, 12, 16), -1)
            put_text(canvas, f"#{real_idx + 1:02d}", (x + 11, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                     0.42, (0, 255, 180) if is_cur else (170, 185, 205), 1, cv2.LINE_AA)

            # 底部文件名信息条
            bar_y = y + self.r.GRID_CELL_H - 22
            cv2.rectangle(canvas, (x + 1, bar_y), (x + self.r.GRID_CELL_W - 1, y + self.r.GRID_CELL_H - 1), (10, 12, 18), -1)
            put_text(canvas, os.path.basename(img_path)[:30], (x + 6, y + self.r.GRID_CELL_H - 7),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                     (0, 255, 200) if is_cur else self.r.COLOR_GRAY, 1, cv2.LINE_AA)

            # 选中卡片高亮指示条
            if is_cur:
                cv2.rectangle(canvas, (x, y), (x + 4, y + self.r.GRID_CELL_H), (0, 255, 180), -1)

    def render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc: Any):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)
        mpos = (state.mouse_x, state.mouse_y)

        images, sel_idx, album_name = state.gallery.get_active_images_and_index()

        if not images:
            hint = f"当前{album_name}无图片"
            draw_text(canvas, hint, (box_x + 220, box_y + 280), font_size=20, color=self.r.COLOR_DARK_GRAY)
            self.r._draw_button(canvas, (box_x + box_w - 140, box_y + 10, 120, 32), "返回网格", mpos)
            return

        sel_idx = max(0, min(sel_idx, len(images) - 1))
        cur_img = images[sel_idx]
        prev = state.gallery.get_preview(cur_img, max_w=590, max_h=520)

        img_title = f"[{album_name}] {os.path.basename(cur_img)} ({sel_idx + 1}/{len(images)})"
        draw_text(canvas, img_title, (box_x + 16, box_y + 14), font_size=14, color=(0, 255, 200), bold=True)

        # 右上角实体按钮组: [上张] [下张] [删帧] [返回] (x: 680~950)
        self.r._draw_button(canvas, (680, box_y + 10, 60, 30), "上张", mpos)
        self.r._draw_button(canvas, (746, box_y + 10, 60, 30), "下张", mpos)
        self.r._draw_button(canvas, (812, box_y + 10, 68, 30), "删帧", mpos, theme_color=(180, 60, 60))
        self.r._draw_button(canvas, (886, box_y + 10, 64, 30), "返回", mpos)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 48 + (520 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)
