# -*- coding: utf-8 -*-
"""
标定精度验证 3D/2D 视觉渲染器 (VerificationVisualizer)
===================================================
单一职责设计：
  1. 3D 双四棱柱虚实位姿对比立体渲染 (BA 理论真值 vs 实测抓取)；
  2. 2D 角点残差矢量放大图绘制与误差标牌合成；
  3. 纯图像处理与 OpenCV 画布合成，绝不参与 PnP 解算逻辑，亦不管理窗口生命周期。
"""

import os
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

from src.ui.text_rendering import draw_text, get_cached_font, measure_text, put_text
from src.utils.logger import get_logger
from src.calibration.verification.prism_renderer import draw_prism, COLORS_THEORY, COLORS_OBSERVED

log = get_logger(__name__)


class VerificationVisualizer:
    """标定验证视觉呈现与双模态渲染器"""

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        初始化渲染器
        :param camera_matrix: 3x3 相机内参
        :param dist_coeffs: 畸变系数
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)

    def render_tag_dual_prisms(
        self,
        img: np.ndarray,
        ba_rvec: Optional[np.ndarray],
        ba_tvec: Optional[np.ndarray],
        obs_rvec: Optional[np.ndarray],
        obs_tvec: Optional[np.ndarray],
        tag_id: int,
        err_px: float,
        err_mm: float,
        observed_corners: Optional[np.ndarray] = None,
        tag_status_hint: Optional[str] = None,
        world_position_mm: Optional[List[float]] = None,
        world_rpy_deg: Optional[List[float]] = None,
        ba_center_xyz: Optional[List[float]] = None,
        obs_center_xyz: Optional[List[float]] = None,
        hovered: bool = True,
        ref_frame_id: Optional[str] = None,
        ref_position_mm: Optional[List[float]] = None,
        nominal_local_xyz: Optional[List[float]] = None,
        nominal_error_mm: Optional[float] = None,
    ):

        """
        绘制全局 BA 平差理论位姿 (纯正翠绿) 与单帧本地实测抓取位姿 (科技天蓝) 的 3D 双四棱柱立体对比
        (黑色提示框: 默认仅 Tag 编号省空间; 鼠标悬停展开六维详情:
         世界系 XYZ/RPY + BA 理论与单帧实测的相机系中心 XYZ, 绿/蓝着色区分)
        """
        try:
            hw = 15.0   # 截面半宽 15mm，整体截面 30.0mm x 30.0mm
            L = 75.0    # 柱体高度 75mm (与实际尺寸协调)

            # 统一 PrismRenderer: 绿色 BA 理论棱柱 + 蓝色实测棱柱 (半透明填充 + 棱线 + 顶面中心)
            proj_ba = None
            if ba_rvec is not None and ba_tvec is not None:
                proj_ba = draw_prism(img, self.camera_matrix, self.dist_coeffs,
                                     ba_rvec, ba_tvec, half_w=hw, height=L,
                                     colors=COLORS_THEORY, alpha=0.35)
            proj_obs = None
            if obs_rvec is not None and obs_tvec is not None:
                proj_obs = draw_prism(img, self.camera_matrix, self.dist_coeffs,
                                      obs_rvec, obs_tvec, half_w=hw, height=L,
                                      colors=COLORS_OBSERVED, alpha=0.35)
            if proj_ba is None and proj_obs is None:
                return

            # C. 顶面中心标牌 (BA 翠绿标注 vs OBS 亮蓝标注)
            ba_c = None
            if proj_ba is not None:
                ba_c = tuple(proj_ba["top_center"])
                put_text(img, "BA(绿)", (ba_c[0] + 6, ba_c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 1, cv2.LINE_AA)

            obs_c = None
            if proj_obs is not None:
                obs_c = tuple(proj_obs["top_center"])
                put_text(img, "实测(蓝)", (obs_c[0] + 6, obs_c[1] + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 210, 80), 1, cv2.LINE_AA)

            # D. 两者顶面中心空间错位拉扯连线 (橙黄连线与红绿小圆点)
            if ba_c is not None and obs_c is not None:
                cv2.line(img, obs_c, ba_c, (0, 80, 255), 2, cv2.LINE_AA)
                cv2.circle(img, obs_c, 5, (255, 180, 0), -1, cv2.LINE_AA)
                cv2.circle(img, ba_c, 5, (0, 255, 100), -1, cv2.LINE_AA)

            # E. 悬浮状态标签 (默认仅 Tag 编号避免高密度遮挡; 鼠标悬停展开六维详情)
            anchor = proj_ba if proj_ba is not None else proj_obs
            anchor_pts = np.vstack([anchor["bottom"], anchor["top"], anchor["top_center"]])
            min_x = np.min(anchor_pts[:, 0])
            min_y = np.min(anchor_pts[:, 1])
            bx = max(10, int(min_x - 10))
            by = max(40, int(min_y - 14))

            if tag_status_hint is not None:
                status_badge = tag_status_hint
                border_c = (0, 220, 100)
                label = f"Tag#{tag_id} {status_badge}"
            elif obs_rvec is None:
                status_badge = "[理论位姿(BA)]"
                border_c = (0, 220, 100)
                label = f"Tag#{tag_id} {status_badge}"
            else:
                is_good = (err_px <= 0.6 and err_mm <= 1.0)
                status_badge = "[吻合良好]" if is_good else f"[空间偏差 {err_mm:.2f}mm]"
                border_c = (0, 240, 90) if is_good else (0, 180, 255)
                label = f"Tag#{tag_id} {status_badge} ({err_px:.2f}px)"

            if not hovered:
                # 高密度场景: 仅显示紧凑编号, 状态语义由边框颜色承载
                info_lines = [(f"Tag#{tag_id}", (255, 255, 255), 14)]
            else:
                # FR-9.7 排版: 三组坐标各占一行 (数值前无 X/Y/Z 字母, 纯列位辨识), 一位小数定宽右对齐,
                # 标签 "世界/理论/实测" 均为双字宽, 保证三行 X/Y/Z 列小数点垂直对齐
                def _fmt(v: float) -> str:
                    return f"{v:>7.1f}"

                info_lines = [(label, (255, 255, 255), 15)]
                # 若指定了非世界基准参考坐标系，优先展示局部坐标与名义偏差
                if ref_frame_id and ref_frame_id != "world" and ref_position_mm is not None and len(ref_position_mm) >= 3:
                    info_lines.append((f"基准[{ref_frame_id}] {_fmt(ref_position_mm[0])}, {_fmt(ref_position_mm[1])}, "
                                       f"{_fmt(ref_position_mm[2])} mm", (0, 240, 220), 14))
                    if nominal_local_xyz is not None and len(nominal_local_xyz) >= 3:
                        err_str = f" (偏差: {nominal_error_mm:.2f}mm)" if nominal_error_mm is not None else ""
                        info_lines.append((f"名义{_fmt(nominal_local_xyz[0])}, {_fmt(nominal_local_xyz[1])}, "
                                           f"{_fmt(nominal_local_xyz[2])} mm{err_str}", (255, 200, 80), 13))
                elif world_position_mm is not None and len(world_position_mm) >= 3:
                    info_lines.append((f"世界{_fmt(world_position_mm[0])}, {_fmt(world_position_mm[1])}, "
                                       f"{_fmt(world_position_mm[2])} mm", (150, 220, 255), 14))

                if ba_center_xyz is not None and len(ba_center_xyz) >= 3:
                    info_lines.append((f"理论{_fmt(ba_center_xyz[0])}, {_fmt(ba_center_xyz[1])}, "
                                       f"{_fmt(ba_center_xyz[2])} mm", (0, 255, 100), 14))
                if obs_center_xyz is not None and len(obs_center_xyz) >= 3:
                    info_lines.append((f"实测{_fmt(obs_center_xyz[0])}, {_fmt(obs_center_xyz[1])}, "
                                       f"{_fmt(obs_center_xyz[2])} mm", (255, 195, 70), 14))
                if world_rpy_deg is not None and len(world_rpy_deg) >= 3:
                    info_lines.append((f"RPY: {world_rpy_deg[0]:>7.1f}, {world_rpy_deg[1]:>7.1f}, "
                                       f"{world_rpy_deg[2]:>7.1f} deg", (150, 255, 200), 13))

            box_w = 0
            line_steps = []
            for txt, _c, fs in info_lines:
                bb = get_cached_font(fs).getbbox(txt)
                box_w = max(box_w, bb[2] - bb[0])
                line_steps.append(fs + 5)
            box_h = sum(line_steps) + 8
            by = max(by, box_h + 10)
            box_top = by - box_h
            cv2.rectangle(img, (bx - 6, box_top), (bx + box_w + 10, by + 4), (16, 22, 28), -1)
            cv2.rectangle(img, (bx - 6, box_top), (bx + box_w + 10, by + 4), border_c, 1)
            ly_ = box_top + 6
            for (txt, col, fs), step in zip(info_lines, line_steps):
                draw_text(img, txt, (bx, ly_), font_size=fs, color=col)
                ly_ += step
        except Exception as e:
            log.warning(f"[Visualizer] 棱柱标注文本绘制异常 (已忽略): {e}")


    def render_verification_frame(
        self,
        img_path: str,
        frame_results: List[Dict],
        view_mode_3d: bool = True,
        manifest_data: Optional[Dict] = None,
        solve_single_tag_pnp_func = None,
        out_path: Optional[str] = None
    ) -> np.ndarray:
        """
        渲染单帧 LOO 盲测可视化图 (支持 3D 双棱柱空间对比视图 与 2D 角点残差矢量视图)
        """
        img = cv2.imread(img_path)
        if img is None:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
        disp = img.copy()
        h, w = disp.shape[:2]

        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        # 若当前帧 frame_results 为空（如孤立单靶帧），依然尽可能从清单提取有效样本渲染实测 3D 棱柱
        if not frame_results and manifest_data and solve_single_tag_pnp_func:
            fname = os.path.basename(img_path)
            img_entry = manifest_data.get("images", {}).get(fname, {})
            for obs in img_entry.get("observations", []):
                if obs.get("keep", True):
                    tid = int(obs.get("tag_id", -1))
                    c = np.array(obs.get("corners", []), dtype=np.float64)
                    if len(c) == 4:
                        ok_pnp, r_obs, t_obs = solve_single_tag_pnp_func(c)
                        if ok_pnp:
                            self.render_tag_dual_prisms(
                                disp, ba_rvec=None, ba_tvec=None,
                                obs_rvec=r_obs, obs_tvec=t_obs,
                                tag_id=tid, err_px=0.0, err_mm=0.0,
                                observed_corners=c
                            )

        if view_mode_3d:
            # 模式 A: 3D 双四棱柱虚实位姿对比
            for r in frame_results:
                self.render_tag_dual_prisms(
                    disp, ba_rvec=r.get("ba_rvec"), ba_tvec=r.get("ba_tvec"),
                    obs_rvec=r.get("obs_rvec"), obs_tvec=r.get("obs_tvec"),
                    tag_id=r["blind_tag_id"], err_px=r["err_px"], err_mm=r["err_mm"],
                    observed_corners=r.get("obs_corners")
                )

            mode_badge = "[ 视图: 3D双四棱柱空间对比模式 (按 T 键切换 2D 残差矢量) ]"
            (mw, mh), _ = measure_text(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (18, 22, 30), -1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (0, 220, 180), 1)
            put_text(disp, mode_badge, (22, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 200), 1, cv2.LINE_AA)

        else:
            # 模式 B: 2D 角点残差矢量模式
            for r in frame_results:
                obs = r["obs_corners"].astype(np.int32)
                proj = r["proj_corners"].astype(np.int32)
                tid = r["blind_tag_id"]
                err_px = r["err_px"]
                err_mm = r["err_mm"]

                is_good = (err_px <= 1.5)
                is_moderate = (err_px <= 3.0)

                obs_line_color = (0, 230, 80) if is_good else ((180, 180, 180) if is_moderate else (120, 120, 120))
                cv2.polylines(disp, [obs], True, obs_line_color, 2 if is_good else 1, cv2.LINE_AA)

                proj_color = (0, 255, 100) if is_good else ((0, 180, 255) if is_moderate else (0, 0, 255))
                cv2.polylines(disp, [proj], True, proj_color, 2, cv2.LINE_AA)

                scale = 15.0
                for i in range(4):
                    p_obs = r["obs_corners"][i]
                    p_proj = r["proj_corners"][i]
                    dx = (p_proj[0] - p_obs[0]) * scale
                    dy = (p_proj[1] - p_obs[1]) * scale
                    pt_s = (int(p_obs[0]), int(p_obs[1]))
                    pt_e = (int(p_obs[0] + dx), int(p_obs[1] + dy))
                    cv2.circle(disp, pt_s, 4, dot_colors[i], -1)
                    cv2.arrowedLine(disp, pt_s, pt_e, proj_color, 2, tipLength=0.25)

                cx = int(np.mean(obs[:, 0]))
                min_y = int(np.min(obs[:, 1]))
                badge_y = max(75, min_y - 14)
                badge_x = max(10, cx - 85)

                status_text = "[PASS]" if is_good else ("[WARN]" if is_moderate else "[FAIL]")
                label = f"Tag#{tid} {status_text} {err_px:.2f}px / {err_mm:.2f}mm"
                badge_bg = (12, 42, 16) if is_good else ((15, 35, 75) if is_moderate else (15, 15, 75))
                (tw, th), _ = measure_text(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 18), (badge_x + tw + 8, badge_y + 4), badge_bg, -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 18), (badge_x + tw + 8, badge_y + 4), proj_color, 1 if is_good else 2)
                put_text(disp, label, (badge_x, badge_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            mode_badge = "[ 视图: 2D角点残差矢量模式 (按 T 键切换 3D 双棱柱) ]"
            (mw, mh), _ = measure_text(mode_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (18, 22, 30), -1)
            cv2.rectangle(disp, (14, h - 35), (14 + mw + 16, h - 8), (0, 200, 255), 1)
            put_text(disp, mode_badge, (22, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 230, 255), 1, cv2.LINE_AA)

        if out_path is not None:
            cv2.imwrite(out_path, disp)

        return disp

    def draw_reprojection_vectors(
        self,
        img: np.ndarray,
        observed_pts: np.ndarray,
        projected_pts: np.ndarray,
        scale_factor: float = 40.0,
        color: Tuple[int, int, int] = (0, 0, 255),
        thickness: int = 2
    ):
        """
        在目标图像上绘制角点重投影残差放大矢量箭头 (实测角点 -> 理论投影角点)
        :param img: 目标画布图像 (原地修改)
        :param observed_pts: (N, 2) 实测角点
        :param projected_pts: (N, 2) 理论投影角点
        :param scale_factor: 残差放大倍数 (如 40.0x 用于清晰放大亚像素误差)
        :param color: 箭头颜色，默认红色 (0, 0, 255)
        :param thickness: 线条粗细
        """
        if observed_pts is None or projected_pts is None:
            return
        try:
            obs = np.asarray(observed_pts, dtype=np.float64).reshape((-1, 2))
            proj = np.asarray(projected_pts, dtype=np.float64).reshape((-1, 2))
            n = min(len(obs), len(proj))
            for i in range(n):
                p_obs = obs[i]
                p_proj = proj[i]
                dx = (p_proj[0] - p_obs[0]) * scale_factor
                dy = (p_proj[1] - p_obs[1]) * scale_factor
                pt_s = (int(round(p_obs[0])), int(round(p_obs[1])))
                pt_e = (int(round(p_obs[0] + dx)), int(round(p_obs[1] + dy)))

                # 绘制实测观测角点黄色圆点
                cv2.circle(img, pt_s, 3, (0, 255, 255), -1, cv2.LINE_AA)
                # 绘制放大误差矢量箭头
                if abs(dx) > 1e-2 or abs(dy) > 1e-2:
                    cv2.arrowedLine(img, pt_s, pt_e, color, thickness, tipLength=0.25)
        except Exception as e:
            log.warning(f"[Visualizer] 残差矢量箭头绘制异常 (已忽略): {e}")

