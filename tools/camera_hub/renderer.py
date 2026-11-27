# -*- coding: utf-8 -*-
"""
Camera Hub 视觉渲染器 (CameraHubRenderer)
========================================
严格遵循 gui-tool-development 只读分离规范：
- 仅持有 CameraHubState 只读引用
- 统一使用 GuiTheme 暗色工业配色与矢量中文绘制
- 每帧返回按钮命中表 [(btn_id, (x1, y1, x2, y2), payload)]
"""

import os
import time
from typing import Any, List, Tuple
import cv2
import numpy as np

from src.calibration.solvers.chessboard_guide import project_guide_mesh
from src.ui.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text
from tools.camera_hub.state import CameraHubState


def _draw_dashed_line(
    canvas: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 12,
    gap_len: int = 8,
):
    """在 OpenCV 画布上绘制虚线（用于标识 3D 引导框的远边）"""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = float(np.hypot(x2 - x1, y2 - y1))
    if dist < 1.0:
        return
    dx = (x2 - x1) / dist
    dy = (y2 - y1) / dist
    pos = 0.0
    step = float(dash_len + gap_len)
    while pos < dist:
        end_pos = min(dist, pos + dash_len)
        p_start = (int(round(x1 + dx * pos)), int(round(y1 + dy * pos)))
        p_end = (int(round(x1 + dx * end_pos)), int(round(y1 + dy * end_pos)))
        cv2.line(canvas, p_start, p_end, color, thickness, cv2.LINE_AA)
        pos += step


class CameraHubRenderer:
    """Camera Hub 纯只读画布渲染器"""

    LEFT_W = 310
    HEADER_H = 52
    FOOTER_H = 38

    def __init__(self, state: CameraHubState):
        self.state = state
        self.buttons: List[Tuple[str, Tuple[int, int, int, int], Any]] = []

    def _btn(
        self,
        canvas: np.ndarray,
        btn_id: str,
        rect: Tuple[int, int, int, int],
        label: str,
        mouse_pos: Tuple[int, int],
        accent: Tuple[int, int, int] = GuiTheme.ACCENT,
        active: bool = False,
        payload: Any = None,
        font_size: int = 12,
    ):
        x, y, w, h = rect
        x2, y2 = x + w, y + h
        self.buttons.append((btn_id, (x, y, x2, y2), payload))
        mx, my = mouse_pos
        hover = (x <= mx <= x2 and y <= my <= y2)

        if active:
            bg = (int(accent[0] * 0.30), int(accent[1] * 0.30), int(accent[2] * 0.30))
            border = accent
            text_col = GuiTheme.TEXT
        elif hover:
            bg = GuiTheme.CARD_HOVER
            border = accent
            text_col = GuiTheme.TEXT
        else:
            bg = GuiTheme.CARD_BG
            border = GuiTheme.BORDER
            text_col = GuiTheme.TEXT_SUB

        cv2.rectangle(canvas, (x, y), (x2, y2), bg, -1)
        cv2.rectangle(canvas, (x, y), (x2, y2), border, 1)
        tw = sum(font_size if ord(c) > 127 else int(font_size * 0.58) for c in label)
        tx = x + max(6, (w - tw) // 2)
        ty = y + max(3, (h - font_size) // 2 - 1)
        draw_text(canvas, label, (tx, ty), font_size=font_size, color=text_col, bold=(active or hover))

    def render(self, canvas_w: int, canvas_h: int, mouse_pos: Tuple[int, int], toast_msg: str = "") -> np.ndarray:
        self.buttons = []
        canvas = np.full((canvas_h, canvas_w, 3), GuiTheme.BG, dtype=np.uint8)

        self._render_header(canvas, canvas_w, mouse_pos)
        self._render_left_panel(canvas, canvas_h, mouse_pos)
        self._render_right_panel(canvas, canvas_w, canvas_h, mouse_pos)
        self._render_footer(canvas, canvas_w, canvas_h, toast_msg)
        return canvas

    def _render_header(self, canvas: np.ndarray, cw: int, mpos: Tuple[int, int]):
        cv2.rectangle(canvas, (0, 0), (cw, self.HEADER_H), GuiTheme.CARD_BG, -1)
        cv2.line(canvas, (0, self.HEADER_H), (cw, self.HEADER_H), GuiTheme.BORDER, 1)
        draw_text(canvas, "Camera Hub", (16, 10), font_size=20, color=GuiTheme.GOLD, bold=True)
        draw_text(canvas, "多相机资产库 & 棋盘格内参标定中枢", (142, 16), font_size=13, color=GuiTheme.TEXT_SUB)

        # 右侧模式切换 Tab
        tx = self.LEFT_W + 16
        self._btn(
            canvas, "mode_live", (tx, 10, 136, 32), "实时采图 (棋盘格)", mpos,
            accent=GuiTheme.ACCENT, active=(self.state.active_mode == CameraHubState.MODE_LIVE),
        )
        self._btn(
            canvas, "mode_gallery", (tx + 146, 10, 146, 32),
            f"图集与求解 ({len(self.state.images)})", mpos,
            accent=GuiTheme.GOLD, active=(self.state.active_mode == CameraHubState.MODE_GALLERY),
        )

        # 退出按钮
        self._btn(canvas, "btn_exit", (cw - 86, 10, 72, 32), "退出", mpos, accent=GuiTheme.ERR)

    def _render_left_panel(self, canvas: np.ndarray, ch: int, mpos: Tuple[int, int]):
        lw = self.LEFT_W
        y0 = self.HEADER_H
        y1 = ch - self.FOOTER_H
        cv2.rectangle(canvas, (0, y0), (lw, y1), GuiTheme.BG, -1)
        cv2.line(canvas, (lw, y0), (lw, y1), GuiTheme.BORDER, 1)

        # 顶部工具按钮：探测本机相机 / 新增USB档案
        self._btn(canvas, "btn_probe", (12, y0 + 10, 140, 30), "探测本机相机", mpos, accent=GuiTheme.ACCENT)
        self._btn(canvas, "btn_add_usb", (160, y0 + 10, 138, 30), "+ 手动登记相机", mpos, accent=GuiTheme.OK)

        draw_text(canvas, f"已登记相机 ({len(self.state.cameras)})", (14, y0 + 50), font_size=13, color=GuiTheme.TEXT_SUB, bold=True)

        # 相机列表卡片
        cy = y0 + 74
        list_bottom = y1 - 148
        for cam in self.state.cameras:
            card_h = 62
            if cy + card_h > list_bottom:
                break
            is_sel = (cam.camera_id == self.state.selected_camera_id)
            is_online = (cam.camera_id in self.state.online_camera_ids)
            rect = (12, cy, lw - 24, card_h)
            self.buttons.append(("select_cam", (rect[0], rect[1], rect[0] + rect[2], rect[1] + rect[3]), cam.camera_id))

            bg = GuiTheme.CARD_SEL if is_sel else GuiTheme.CARD_BG
            border = GuiTheme.ACCENT if is_sel else GuiTheme.BORDER
            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), bg, -1)
            cv2.rectangle(canvas, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), border, 1)

            dot_col = GuiTheme.OK if is_online else GuiTheme.TEXT_MUTED
            facing_tag = "前置" if cam.is_front_facing else "后置"
            draw_text(canvas, f"● {cam.name}", (22, cy + 8), font_size=13, color=GuiTheme.TEXT if is_sel else GuiTheme.TEXT_SUB, bold=is_sel)
            draw_text(canvas, f"ID: {cam.camera_id} · {cam.camera_type} · {facing_tag}", (22, cy + 28), font_size=11, color=GuiTheme.TEXT_MUTED)

            # 显示当前分辨率是否已标定
            rec = cam.load_intrinsics(self.state.selected_resolution)
            if rec is not None:
                st_txt = f"✓ {rec.resolution_str} RMSE {rec.rmse:.2f}px"
                st_col = GuiTheme.OK
            else:
                res_cnt = len([r for r in cam.list_resolutions() if cam.load_intrinsics(r) is not None])
                st_txt = f"已标定 {res_cnt} 档分辨率" if res_cnt > 0 else "○ 尚未标定内参"
                st_col = GuiTheme.GOLD if res_cnt > 0 else dot_col
            draw_text(canvas, st_txt, (22, cy + 44), font_size=11, color=st_col)
            cy += card_h + 8

        # 底部：选中相机操作 + 分辨率选择
        panel_y = y1 - 140
        cv2.line(canvas, (12, panel_y), (lw - 12, panel_y), GuiTheme.BORDER, 1)
        draw_text(canvas, "目标分辨率档案", (14, panel_y + 8), font_size=12, color=GuiTheme.TEXT_SUB, bold=True)

        res_list = self.state.get_available_resolutions()[:3]
        rx = 12
        for res in res_list:
            self._btn(
                canvas, "select_res", (rx, panel_y + 30, 92, 28), res, mpos,
                accent=GuiTheme.ACCENT, active=(res == self.state.selected_resolution),
                payload=res, font_size=11,
            )
            rx += 97

        draw_text(canvas, "相机档案管理", (14, panel_y + 68), font_size=12, color=GuiTheme.TEXT_SUB, bold=True)
        self._btn(canvas, "btn_rename_cam", (12, panel_y + 92, 138, 30), "重命名别名", mpos, accent=GuiTheme.GOLD)
        self._btn(canvas, "btn_delete_cam", (158, panel_y + 92, 138, 30), "删除相机档案", mpos, accent=GuiTheme.ERR)

    def _render_right_panel(self, canvas: np.ndarray, cw: int, ch: int, mpos: Tuple[int, int]):
        rx0 = self.LEFT_W + 14
        ry0 = self.HEADER_H + 10
        rw = cw - rx0 - 14
        rh = ch - self.FOOTER_H - ry0 - 10

        cam = self.state.get_selected_camera()
        if cam is None:
            draw_text(canvas, "暂无已登记相机，请点击左侧 [探测本机相机] 或 [+ 手动登记相机]",
                      (rx0 + 40, ry0 + rh // 2 - 12), font_size=16, color=GuiTheme.TEXT_SUB)
            return

        # 顶部棋盘格规格控制条 (双模式共用)
        bar_h = 44
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + bar_h), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (rx0, ry0), (rx0 + rw, ry0 + bar_h), GuiTheme.BORDER, 1)

        draw_text(canvas, f"棋盘格内角点: {self.state.pattern_cols}×{self.state.pattern_rows}",
                  (rx0 + 12, ry0 + 13), font_size=13, color=GuiTheme.TEXT, bold=True)
        self._btn(canvas, "pat_col_dec", (rx0 + 168, ry0 + 8, 28, 28), "C-", mpos, font_size=11)
        self._btn(canvas, "pat_col_inc", (rx0 + 200, ry0 + 8, 28, 28), "C+", mpos, font_size=11)
        self._btn(canvas, "pat_row_dec", (rx0 + 236, ry0 + 8, 28, 28), "R-", mpos, font_size=11)
        self._btn(canvas, "pat_row_inc", (rx0 + 268, ry0 + 8, 28, 28), "R+", mpos, font_size=11)

        draw_text(canvas, f"方格边长: {self.state.square_size_mm:.1f} mm",
                  (rx0 + 312, ry0 + 13), font_size=13, color=GuiTheme.GOLD, bold=True)
        self._btn(canvas, "pat_sq_dec", (rx0 + 452, ry0 + 8, 32, 28), "-1", mpos, font_size=11)
        self._btn(canvas, "pat_sq_inc", (rx0 + 488, ry0 + 8, 32, 28), "+1", mpos, font_size=11)

        # 右侧当前档案概览
        rec = self.state.current_intrinsics
        summary = f"{cam.camera_id} @ {self.state.selected_resolution}" + (
            f" | RMSE {rec.rmse:.3f}px" if rec else " | 未标定"
        )
        draw_text(canvas, summary, (rx0 + 536, ry0 + 14), font_size=12, color=GuiTheme.OK if rec else GuiTheme.TEXT_MUTED)

        body_y = ry0 + bar_h + 10
        body_h = rh - bar_h - 10

        if self.state.active_mode == CameraHubState.MODE_LIVE:
            self._render_live_mode(canvas, rx0, body_y, rw, body_h, mpos)
        else:
            self._render_gallery_mode(canvas, rx0, body_y, rw, body_h, mpos)

    def _render_live_mode(self, canvas: np.ndarray, x0: int, y0: int, w: int, h: int, mpos: Tuple[int, int]):
        ctrl_h = 42
        is_front = self.state.is_mirror_preview
        self._btn(
            canvas, "btn_toggle_cam", (x0, y0, 108, 34),
            "关闭相机" if self.state.pipeline_running else "开启预览",
            mpos, accent=GuiTheme.ERR if self.state.pipeline_running else GuiTheme.OK,
            active=self.state.pipeline_running,
        )
        self._btn(
            canvas, "btn_toggle_facing", (x0 + 114, y0, 108, 34),
            "前置:镜像 [M]" if is_front else "后置:正常 [M]",
            mpos, accent=GuiTheme.ACCENT if is_front else GuiTheme.TEXT_SUB,
            active=is_front, font_size=11,
        )
        self._btn(
            canvas, "btn_toggle_auto", (x0 + 228, y0, 104, 34),
            f"自动抓拍:{'开' if self.state.auto_capture_enabled else '关'}",
            mpos, accent=GuiTheme.OK if self.state.auto_capture_enabled else GuiTheme.TEXT_MUTED,
            active=self.state.auto_capture_enabled, font_size=11,
        )
        self._btn(
            canvas, "btn_capture", (x0 + 338, y0, 112, 34),
            f"抓拍 ({len(self.state.images)})",
            mpos, accent=GuiTheme.ACCENT, font_size=11,
        )
        self._btn(canvas, "btn_skip_pose", (x0 + 456, y0, 68, 34), "跳过 [N]", mpos, accent=GuiTheme.TEXT_SUB, font_size=11)
        self._btn(canvas, "btn_undo_capture", (x0 + 530, y0, 68, 34), "撤销 [Z]", mpos, accent=GuiTheme.WARN, font_size=11)
        self._btn(canvas, "btn_reset_guide", (x0 + 604, y0, 50, 34), "重置", mpos, accent=GuiTheme.TEXT_SUB, font_size=11)
        self._btn(canvas, "btn_solve", (x0 + w - 118, y0, 118, 34), "立即标定求解", mpos, accent=GuiTheme.GOLD, font_size=11)

        # 视频视口
        vx, vy, vw, vh = x0, y0 + ctrl_h, w, h - ctrl_h
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), (12, 15, 20), -1)

        fb = self.state.last_feedback
        is_aligned = bool(fb and fb.aligned)
        border_col = GuiTheme.OK if (self.state.pipeline_running and is_aligned) else GuiTheme.BORDER
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), border_col, 2 if is_aligned else 1)

        if self.state.live_frame is not None and self.state.pipeline_running:
            frame = self.state.live_frame
            display_frame = cv2.flip(frame, 1) if self.state.is_mirror_preview else frame
            fh, fw = display_frame.shape[:2]
            scale = min((vw - 4) / max(1, fw), (vh - 4) / max(1, fh))
            dw, dh = max(1, int(fw * scale)), max(1, int(fh * scale))
            resized = cv2.resize(display_frame, (dw, dh), interpolation=cv2.INTER_AREA)
            ox = vx + (vw - dw) // 2
            oy = vy + (vh - dh) // 2
            canvas[oy:oy + dh, ox:ox + dw] = resized

            # 叠加 flux_android 风格 3D 透视引导轮廓与稳定保持进度环
            self._render_3d_guide_overlay(canvas, ox, oy, dw, dh, float(fw) / float(max(1, fh)))

            # 顶部半透明引导横幅（步骤、姿态标题、倾角提示、实时方向指引与进度条）
            self._render_guide_banner(canvas, vx + 4, vy + 4, vw - 8)

            # 快门白闪反馈
            if time.time() < self.state.flash_timer:
                overlay = np.full((dh, dw, 3), 255, dtype=np.uint8)
                canvas[oy:oy + dh, ox:ox + dw] = cv2.addWeighted(canvas[oy:oy + dh, ox:ox + dw], 0.45, overlay, 0.55, 0)
        else:
            pose = self.state.get_current_pose()
            draw_text(
                canvas,
                f"点击左上 [开启相机预览] — 3D 透视引导共 {len(self.state.guide_poses)} 个姿态（当前: {pose.title}）",
                (vx + max(20, vw // 2 - 290), vy + vh // 2 - 22), font_size=15, color=GuiTheme.TEXT_SUB,
            )
            draw_text(
                canvas,
                "开启自动抓拍后，只需将棋盘格对准画面中的 3D 轮廓并稳住 0.7 秒即可自动抓拍保存",
                (vx + max(20, vw // 2 - 300), vy + vh // 2 + 8), font_size=13, color=GuiTheme.TEXT_MUTED,
            )

    def _render_3d_guide_overlay(
        self,
        canvas: np.ndarray,
        ox: int,
        oy: int,
        dw: int,
        dh: int,
        viewport_aspect: float,
    ):
        """绘制 3D 投影目标棋盘格轮廓、近/远边深度编码、实时角点与中心自动抓拍进度环"""
        pose = self.state.get_current_pose()
        fb = self.state.last_feedback
        mesh = project_guide_mesh(pose, self.state.pattern_size, viewport_aspect=viewport_aspect)

        def to_px(pt: Tuple[float, float]) -> Tuple[int, int]:
            return (int(round(ox + pt[0] * dw)), int(round(oy + pt[1] * dh)))

        # BGR 颜色语义：绿=已对准，黄=已检出待微调，青=搜寻中
        if fb and fb.aligned:
            guide_col = (94, 197, 34)    # #22C55E
        elif fb and fb.detected:
            guide_col = (21, 204, 250)   # #FACC15
        else:
            guide_col = (248, 189, 56)   # #38BDF8

        quad_px = [to_px(p) for p in mesh.outer_quad]

        # 1. 半透明目标多边形填充
        if len(quad_px) == 4:
            overlay = canvas.copy()
            poly = np.array(quad_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly], guide_col)
            alpha = 0.22 if (fb and fb.aligned) else 0.12
            cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)

        # 2. 内部 3D 透视网格线
        grid_col = (int(guide_col[0] * 0.75), int(guide_col[1] * 0.75), int(guide_col[2] * 0.75))
        for a, b in mesh.grid_lines:
            cv2.line(canvas, to_px(a), to_px(b), grid_col, 1, cv2.LINE_AA)

        # 3. 深度编码外框边（近边粗实线，远边虚线）
        is_tilted = (abs(pose.pitch_deg) > 4.0 or abs(pose.yaw_deg) > 4.0)
        min_z = min(mesh.corner_depths) if mesh.corner_depths else 2.6
        max_z = max(mesh.corner_depths) if mesh.corner_depths else 2.6
        z_span = max(1e-4, max_z - min_z)

        closest_edge, closest_z = -1, float("inf")
        farthest_edge, farthest_z = -1, -float("inf")
        for i in range(4):
            nxt = (i + 1) % 4
            avg_z = 0.5 * (mesh.corner_depths[i] + mesh.corner_depths[nxt])
            if avg_z < closest_z:
                closest_z, closest_edge = avg_z, i
            if avg_z > farthest_z:
                farthest_z, farthest_edge = avg_z, i

        for i in range(4):
            nxt = (i + 1) % 4
            avg_z = 0.5 * (mesh.corner_depths[i] + mesh.corner_depths[nxt])
            depth_norm = max(0.0, min(1.0, (avg_z - min_z) / z_span)) if is_tilted else 0.5
            is_far = is_tilted and (depth_norm > 0.65)
            thick = max(2, int(round(5.0 - depth_norm * 3.0))) if is_tilted else 3
            if is_far:
                _draw_dashed_line(canvas, quad_px[i], quad_px[nxt], guide_col, thickness=thick)
            else:
                cv2.line(canvas, quad_px[i], quad_px[nxt], guide_col, thick, cv2.LINE_AA)

        # 4. 四角锚点圆
        for p in quad_px:
            cv2.circle(canvas, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 4, guide_col, -1, cv2.LINE_AA)

        # 5. 倾斜姿态近边/远边中文标签
        if is_tilted and closest_edge >= 0 and farthest_edge >= 0:
            na, nb = quad_px[closest_edge], quad_px[(closest_edge + 1) % 4]
            fa, fb_pt = quad_px[farthest_edge], quad_px[(farthest_edge + 1) % 4]
            near_mid = ((na[0] + nb[0]) // 2 - 14, (na[1] + nb[1]) // 2 - 10)
            far_mid = ((fa[0] + fb_pt[0]) // 2 - 14, (fa[1] + fb_pt[1]) // 2 - 10)
            draw_text(canvas, "近边", near_mid, font_size=13, color=(255, 255, 255), bold=True)
            draw_text(canvas, "远边", far_mid, font_size=13, color=(220, 230, 240), bold=True)

        # 6. 实时检测到的棋盘格外框与角点
        if fb and fb.detected and len(fb.detected_quad_norm) == 4:
            det_px = [to_px(p) for p in fb.detected_quad_norm]
            det_col = (128, 222, 74) if fb.aligned else (71, 224, 253)
            poly_det = np.array(det_px, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [poly_det], True, det_col, 2, cv2.LINE_AA)
            for pt in fb.detected_corners_norm:
                cv2.circle(canvas, to_px(pt), 3, det_col, -1, cv2.LINE_AA)

        # 7. 目标中心锚点与自动抓拍稳定保持进度环
        center_px = to_px((pose.center_x, pose.center_y))
        cv2.circle(canvas, center_px, 6, guide_col, 2, cv2.LINE_AA)
        if self.state.hold_progress > 0.01:
            sweep = 360.0 * min(1.0, max(0.0, self.state.hold_progress))
            cv2.ellipse(
                canvas,
                center_px,
                (28, 28),
                0.0,
                -90.0,
                -90.0 + sweep,
                (94, 197, 34),
                5,
                cv2.LINE_AA,
            )

    def _render_guide_banner(self, canvas: np.ndarray, bx: int, by: int, bw: int):
        """视口顶部半透明引导横幅：步骤进度、姿态名称、倾角说明与实时方向指引"""
        bh = 56
        overlay = canvas.copy()
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (18, 24, 34), -1)
        cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

        pose = self.state.get_current_pose()
        total = len(self.state.guide_poses)
        step_txt = f"步骤 {self.state.current_pose_idx + 1}/{total} · {pose.title}  [{pose.tilt_hint}]"
        draw_text(canvas, step_txt, (bx + 12, by + 6), font_size=13, color=GuiTheme.TEXT, bold=True)

        fb = self.state.last_feedback
        guide_msg = fb.guidance_text if fb else pose.instruction
        if fb and fb.aligned and fb.stable:
            msg_col = GuiTheme.OK
        elif fb and fb.detected:
            msg_col = GuiTheme.GOLD
        else:
            msg_col = GuiTheme.ACCENT
        draw_text(canvas, guide_msg, (bx + 12, by + 28), font_size=13, color=msg_col, bold=True)

        # 右侧进度条
        bar_w = 180
        bar_x = bx + bw - bar_w - 14
        bar_y = by + 22
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10), (40, 48, 62), -1)
        prog = float(self.state.current_pose_idx) / float(max(1, total))
        fill_w = int(round(bar_w * min(1.0, max(0.0, prog))))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + 10), GuiTheme.OK, -1)
        draw_text(
            canvas,
            f"会话已拍 {len(self.state.session_captured_paths)}/{total}",
            (bar_x, by + 4), font_size=11, color=GuiTheme.TEXT_SUB,
        )

    def _render_gallery_mode(self, canvas: np.ndarray, x0: int, y0: int, w: int, h: int, mpos: Tuple[int, int]):
        # 顶部内参指标卡 + 操作按钮
        card_h = 86
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + card_h), GuiTheme.CARD_BG, -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + card_h), GuiTheme.BORDER, 1)

        rec = self.state.current_intrinsics
        if rec is not None:
            draw_text(
                canvas,
                f"内参矩阵 K:  fx={rec.fx:.2f}  fy={rec.fy:.2f}  cx={rec.cx:.2f}  cy={rec.cy:.2f}  ({rec.width}×{rec.height})",
                (x0 + 12, y0 + 10), font_size=13, color=GuiTheme.OK, bold=True,
            )
            d = rec.dist_coeffs
            draw_text(
                canvas,
                f"畸变系数 D:  k1={d[0]:.4f}  k2={d[1]:.4f}  p1={d[2]:.4f}  p2={d[3]:.4f}  k3={d[4]:.4f}",
                (x0 + 12, y0 + 34), font_size=12, color=GuiTheme.TEXT_SUB,
            )
            draw_text(
                canvas,
                f"全局 RMSE: {rec.rmse:.4f} px  |  有效帧: {rec.valid_images_count}/{rec.total_images_count}  |  标定时间: {rec.calibrated_at}",
                (x0 + 12, y0 + 58), font_size=12, color=GuiTheme.GOLD,
            )
        else:
            draw_text(canvas, "尚未完成内参求解 — 请采集至少 3 张不同视角棋盘格照片后点击 [开始棋盘格标定]",
                      (x0 + 12, y0 + 28), font_size=13, color=GuiTheme.TEXT_SUB)

        # 右侧按钮列
        self._btn(canvas, "btn_solve", (x0 + w - 152, y0 + 8, 140, 32), "开始棋盘格标定", mpos, accent=GuiTheme.GOLD)
        self._btn(canvas, "btn_prune_bad", (x0 + w - 302, y0 + 8, 142, 32), "剔除无效/高误差帧", mpos, accent=GuiTheme.WARN)
        self._btn(canvas, "btn_del_img", (x0 + w - 152, y0 + 46, 140, 32), "删除选中帧 [Del]", mpos, accent=GuiTheme.ERR)

        # 3x3 缩略图网格
        gy0 = y0 + card_h + 10
        gh = h - card_h - 10
        if not self.state.images:
            draw_text(canvas, "当前分辨率下尚无棋盘格照片，请切换到 [实时采图] 模式拍摄。",
                      (x0 + max(20, w // 2 - 220), gy0 + gh // 2 - 10), font_size=15, color=GuiTheme.TEXT_MUTED)
            return

        cols, rows = CameraHubState.GRID_COLS, CameraHubState.GRID_ROWS
        gap = 10
        cell_w = (w - gap * (cols - 1)) // cols
        cell_h = (gh - gap * (rows - 1)) // rows
        visible = self.state.images[self.state.grid_offset: self.state.grid_offset + CameraHubState.GRID_PAGE]

        for i, img_path in enumerate(visible):
            real_idx = self.state.grid_offset + i
            r, c = divmod(i, cols)
            cx = x0 + c * (cell_w + gap)
            cy = gy0 + r * (cell_h + gap)
            is_sel = (real_idx == self.state.selected_image_idx)
            self.buttons.append(("select_img", (cx, cy, cx + cell_w, cy + cell_h), real_idx))

            cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), GuiTheme.CARD_BG, -1)
            thumb_h = max(40, cell_h - 26)
            thumb = self.state.get_thumbnail(img_path, max(40, cell_w - 4), thumb_h)
            if thumb is not None:
                canvas[cy + 2:cy + 2 + thumb_h, cx + 2:cx + 2 + (cell_w - 4)] = thumb

            fname = os.path.basename(img_path)
            draw_text(canvas, fname, (cx + 6, cy + cell_h - 20), font_size=11, color=GuiTheme.TEXT)

            # 逐帧重投影误差徽章
            if fname in self.state.per_image_detected:
                if not self.state.per_image_detected[fname]:
                    badge_txt, badge_col = "未检出", GuiTheme.ERR
                else:
                    err = self.state.per_image_errors.get(fname)
                    badge_txt = f"{err:.2f}px" if err is not None else "OK"
                    badge_col = GuiTheme.OK if (err is not None and err < 0.8) else GuiTheme.WARN
                draw_text(canvas, badge_txt, (cx + cell_w - 58, cy + cell_h - 20), font_size=11, color=badge_col, bold=True)

            border = GuiTheme.ACCENT if is_sel else GuiTheme.BORDER
            cv2.rectangle(canvas, (cx, cy), (cx + cell_w, cy + cell_h), border, 2 if is_sel else 1)

    def _render_footer(self, canvas: np.ndarray, cw: int, ch: int, toast_msg: str):
        fy = ch - self.FOOTER_H
        cv2.rectangle(canvas, (0, fy), (cw, ch), GuiTheme.CARD_BG, -1)
        cv2.line(canvas, (0, fy), (cw, fy), GuiTheme.BORDER, 1)
        tip = toast_msg or "[自动抓拍] 对准 3D 轮廓稳住 0.7s 自动保存  |  [Space] 手动抓拍  |  [N] 跳过  |  [Z] 撤销  |  [B] 求解  |  [ESC] 退出"
        draw_text(canvas, tip, (14, fy + 10), font_size=12, color=GuiTheme.TEXT_SUB)
