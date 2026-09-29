# -*- coding: utf-8 -*-
"""
Workspace Hub 相册与大图预览页面渲染器 (GalleryPageRenderer)
=========================================================
负责：
1. 标定相册与生产相册卡片网格墙 (4列x3行大卡片网格，缩略图缓存，序号与文件名条)
2. 全宽自适应大图视口 (按 F 键展开，双击卡片放大，上张/下张/删帧/返回)
"""

import os
import cv2
import numpy as np
from typing import Any

from src.utils.gui_components import draw_text, put_text
from tools.workspace_hub.hub_state import HubState


class GalleryPageRenderer:
    """相册与大图预览页面渲染器"""

    def __init__(self, parent_renderer: Any):
        self.r = parent_renderer

    def render_calib_images(self, canvas: np.ndarray, state: HubState, sc: Any):
        """页签2: 标定相册 - 当前选中工位的采样相册卡片网格墙 (x: 340~960, y: 50~670)"""
        self.render_gallery_page(
            canvas, state, "",
            state.gallery.current_images, state.gallery.selected_image_idx, state.gallery.image_grid_offset,
            empty_hint="当前场景尚未采集任何照片！",
            is_calib=True,
        )

    def render_prod_images(self, canvas: np.ndarray, state: HubState, sc: Any):
        """页签4: 生产相册 - 生产运行基准工位的采样相册 (x: 340~960, y: 50~670)"""
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
        )

    def render_gallery_page(self, canvas: np.ndarray, state: HubState, title: str,
                            images: list[str], sel_idx: int, grid_offset: int,
                            empty_hint: str, is_calib: bool):
        """渲染通用图片卡片网格墙: 4列x3行大卡片网格 (双击卡片放大)
        布局: 面板 (340, 50, 620, 620); 网格 x: 356~944, y: 86~648; 卡片 184x168
        """
        box_x, box_y, box_w, box_h = 340, 50, self.r.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.r.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 空状态：纯净单个提示，删除所有冗余副提示红字
        if not images:
            empty_box_y = box_y + 140
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 90), (22, 26, 36), -1)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 90), self.r.COLOR_BORDER, 1)
            # 水平居中渲染
            approx_w = sum(18 if ord(c) > 127 else 10 for c in empty_hint)
            hint_x = box_x + max(20, (box_w - approx_w) // 2)
            draw_text(canvas, empty_hint, (hint_x, empty_box_y + 32), font_size=17, color=(0, 200, 240), bold=True)
            return

        # 3. 图片卡片网格墙 (4 列 x 3 行, 每页 12 张大卡片, 铺满面板底部空白)
        visible_imgs = images[grid_offset: grid_offset + HubState.GRID_PAGE]

        for i, img_path in enumerate(visible_imgs):
            real_idx = grid_offset + i
            row, col = divmod(i, HubState.GRID_COLS)
            x = self.r.GRID_X0 + col * (self.r.GRID_CELL_W + self.r.GRID_GAP_X)
            y = self.r.GRID_Y0 + row * (self.r.GRID_CELL_H + self.r.GRID_GAP_Y)

            is_cur = (real_idx == sel_idx)
            is_hover = (x <= mpos[0] <= x + self.r.GRID_CELL_W and y <= mpos[1] <= y + self.r.GRID_CELL_H)

            card_bg = (30, 40, 36) if is_cur else ((28, 33, 41) if is_hover else (22, 26, 36))
            card_border = (0, 255, 180) if is_cur else ((0, 200, 240) if is_hover else (38, 44, 58))
            cv2.rectangle(canvas, (x, y), (x + self.r.GRID_CELL_W, y + self.r.GRID_CELL_H), card_bg, -1)
            cv2.rectangle(canvas, (x, y), (x + self.r.GRID_CELL_W, y + self.r.GRID_CELL_H), card_border, 2 if is_cur else 1)

            # 3.1 卡片主体: 接近 4:3 的大缩略图
            tw, th = self.r.GRID_CELL_W - 8, self.r.GRID_THUMB_H
            thumb = state.gallery.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y + 4:y + 4 + th, x + 4:x + 4 + tw] = thumb
            else:
                draw_text(canvas, "读取失败", (x + 60, y + 74), font_size=13, color=(120, 120, 140))

            # 3.2 左上角序号徽章
            cv2.rectangle(canvas, (x + 4, y + 4), (x + 56, y + 26), (8, 12, 16), -1)
            put_text(canvas, f"#{real_idx + 1:02d}", (x + 11, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                     0.42, (0, 255, 180) if is_cur else (170, 185, 205), 1, cv2.LINE_AA)

            # 3.3 底部文件名信息条
            bar_y = y + self.r.GRID_CELL_H - 22
            cv2.rectangle(canvas, (x + 1, bar_y), (x + self.r.GRID_CELL_W - 1, y + self.r.GRID_CELL_H - 1), (10, 12, 18), -1)
            put_text(canvas, os.path.basename(img_path)[:30], (x + 6, y + self.r.GRID_CELL_H - 7),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                     (0, 255, 200) if is_cur else self.r.COLOR_GRAY, 1, cv2.LINE_AA)

            # 3.4 选中卡片左侧高亮指示条
            if is_cur:
                cv2.rectangle(canvas, (x, y), (x + 4, y + self.r.GRID_CELL_H), (0, 255, 180), -1)

    def render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc: Any):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)
        mpos = (state.mouse_x, state.mouse_y)

        is_prod = (state.active_tab == HubState.TAB_PROD_IMAGES)
        images = state.gallery.prod_images if is_prod else state.gallery.current_images
        sel_idx = state.gallery.selected_prod_image_idx if is_prod else state.gallery.selected_image_idx

        # 标题与右上角实体按钮
        if not images:
            hint = "当前生产相册无图片" if is_prod else "当前标定相册无图片"
            draw_text(canvas, hint, (box_x + 220, box_y + 280), font_size=20, color=self.r.COLOR_DARK_GRAY)
            self.r._draw_button(canvas, (box_x + box_w - 140, box_y + 10, 120, 32), "返回网格", mpos)
            return

        sel_idx = max(0, min(sel_idx, len(images) - 1))
        cur_img = images[sel_idx]
        prev = state.gallery.get_preview(cur_img, max_w=590, max_h=520)

        img_title = f"{os.path.basename(cur_img)} ({sel_idx + 1}/{len(images)})"
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
