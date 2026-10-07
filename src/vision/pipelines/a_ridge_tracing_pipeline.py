"""
技术路线 A：基于欧氏距离变换与垂向峰脊跟踪的感知流水线 (Ridge Tracing Pipeline)
=============================================================================
核心原理：
  1. 传送带作业 ROI 约束 + ExG 超绿指数前景掩膜提取
  2. 欧氏距离变换 (Distance Transform)，前景像素距离边界距离表征内切截面半径
  3. 沿垂向跨度方向做非极大值抑制 (Transverse NMS)，在并排贴合接缝处天然形成凹陷谷底，只在物料中轴取峰
  4. 沿长轴方向形态学闭运算桥接连通成中轴脊线 (Ridge Lines)
  5. PCA 主轴拟合 + 多点沿轴距离场半径采样，精确解算物理直径 (支持大到 48mm 粗笋)
  6. 综合相对台面凸起高度与几何质量锁定 Top 3 推荐抓取目标
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry


@PipelineRegistry.register("ridge_tracing", "算法 A: 距离场脊线法 (Ridge Tracing)")
class RidgeTracingPipeline(BaseAsparagusPipeline):
    """算法 A：距离场脊线感知流水线"""

    name = "算法 A: 距离场脊线法 (Ridge Tracing)"
    description = "通过欧氏距离变换与垂向极大值峰脊追踪，利用能量鞍部天然解耦并排并贴合的芦笋"

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        super().__init__(fx, fy, cx, cy)
        self.min_length_mm = 80.0
        self.max_length_mm = 600.0
        self.min_diam_mm = 6.0
        self.max_diam_mm = 48.0

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep(
                "stage1_fg", "1.前景",
                "ExG 超绿指数与传送带物理 ROI 作业区约束",
                details="计算 2G - R - B 超绿特征，将绿色芦笋从黑色/深灰色输送带背景中无阈值剥离",
                parameters="ExG 阈值=12.0; ROI=[0.35W, 0.81W]",
                pros_cons="优点: 极其契合新鲜蔬菜色彩特征; 缺点: 面对黄化或泥沙包裹芦笋需配合亮度补偿"
            ),
            PipelineStep(
                "stage1_dist", "距离场",
                "欧氏距离变换场 (Distance Transform 半径能量分布)",
                details="计算前景每个像素到最近背景的欧氏距离，中轴处数值最大，鞍部形成天然分割谷底",
                parameters="cv2.DIST_L2, 算子大小=5x5",
                pros_cons="优点: 赋予并排贴合芦笋天然解耦鞍部; 缺点: 边缘锯齿会带来距离场毛刺"
            ),
            PipelineStep(
                "stage2_peaks", "峰脊线",
                "垂向局部极大值抑制峰脊点阵 (解耦并排贴合缝隙)",
                details="沿垂直截面提取距离场局域最大值，抑制平坦区域，只提取中心中轴关键脊点",
                parameters="min_peak_radius=3.0px; nms_window=5px",
                pros_cons="优点: 离散化极快，完全消除并排芦笋粘连; 缺点: 严重空洞内部可能产生假极大值"
            ),
            PipelineStep(
                "stage2_spines", "2.骨架",
                "横向桥接中轴脊线、采样截面半径圆与紧凑多边形",
                details="将脊点按物理间距约束桥接成连续骨架线，沿线膨胀真实半径圆构造带方向轮廓",
                parameters="bridge_dx=18px, bridge_dy=8px; min_spine_len=50px",
                pros_cons="优点: 中轴点与直径完全同步解算; 缺点: 严重断续时需依赖外推拟合"
            ),
            PipelineStep(
                "stage3_poses", "3.位姿",
                "三维空间位姿解算、SCARA抓取坐标与Top 3顶层锁定",
                details="结合相机外参及桌平面模型，反投影计算吸盘抓取点 (X, Y, Z, R) 并按高度仲裁顶层",
                parameters="safe_z=80.0mm; 吸盘相对抓取沉降=-12mm",
                pros_cons="优点: 直接驱动工业 SCARA 执行抓取; 缺点: 需事先完成手眼与工作区标定"
            )
        ]

    def run(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated",
        nominal_z_mm: float = 640.0
    ) -> PipelineResult:
        t0 = time.perf_counter()
        h, w = color_bgr.shape[:2]
        step_images: Dict[str, Optional[np.ndarray]] = {}

        # ---------------- 步骤 1: 前景提取 ----------------
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
        b, g, r = roi_bgr[:, :, 0], roi_bgr[:, :, 1], roi_bgr[:, :, 2]
        exg = 2.0 * g - r - b
        gray = cv2.cvtColor(color_bgr[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
        fg_roi = ((exg > 10.0) | ((g > b * 1.05) & (gray > 42))).astype(np.uint8) * 255
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_OPEN, k_open)

        if depth_mm is not None:
            depth_roi = depth_mm[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depth = (depth_roi >= 350) & (depth_roi <= 780)
            fg_roi = np.where((depth_roi > 0) & (~valid_depth), 0, fg_roi)

        # 步骤 1 渲染
        vis_1 = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
        green_layer = vis_1.copy()
        fg_full = np.zeros((h, w), dtype=np.uint8)
        fg_full[roi_y1:roi_y2, roi_x1:roi_x2] = fg_roi
        green_layer[fg_full > 0] = (40, 235, 90)
        vis_1 = cv2.addWeighted(green_layer, 0.65, vis_1, 0.35, 0)
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_1, (12, 12), (540, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (540, 48), (0, 235, 90), 2)
        put_text(vis_1, f"STAGE 1: FOREGROUND (ExG+ROI) | Pixels: {int(np.count_nonzero(fg_full))}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 120), 2)
        step_images["stage1_fg"] = vis_1

        # ---------------- 步骤 2: 欧氏距离变换 ----------------
        dist = cv2.distanceTransform(fg_roi, cv2.DIST_L2, 5)
        vis_dist = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        dist_norm = np.clip(dist / 28.0 * 255.0, 0, 255).astype(np.uint8)
        dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        roi_patch = vis_dist[roi_y1:roi_y2, roi_x1:roi_x2]
        fg_bool = fg_roi > 0
        roi_patch[fg_bool] = dist_color[fg_bool]
        vis_dist[roi_y1:roi_y2, roi_x1:roi_x2] = roi_patch
        cv2.rectangle(vis_dist, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_dist, (12, 12), (560, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_dist, (12, 12), (560, 48), (40, 230, 240), 2)
        put_text(vis_dist, "CV: DISTANCE TRANSFORM (Radius Field & Seam Valleys)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage1_dist"] = vis_dist

        # ---------------- 步骤 3: 垂向极大值峰脊线提取 ----------------
        kernel_v = np.ones((7, 1), np.uint8)
        dist_dil_v = cv2.dilate(dist, kernel_v)
        peaks = (dist == dist_dil_v) & (dist >= 4.0)

        vis_peaks = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        cnts_fg, _ = cv2.findContours(fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cf in cnts_fg:
            cv2.drawContours(vis_peaks, [cf + np.array([roi_x1, roi_y1])], -1, (60, 110, 75), 1)
        py, px = np.where(peaks)
        for y_pt, x_pt in zip(py, px):
            gx, gy = x_pt + roi_x1, y_pt + roi_y1
            cv2.drawMarker(vis_peaks, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 4, 1)
        cv2.rectangle(vis_peaks, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_peaks, (12, 12), (580, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_peaks, (12, 12), (580, 48), (0, 255, 255), 2)
        put_text(vis_peaks, "CV: TRANSVERSE NMS RIDGE PEAKS (De-coupling Seams)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        step_images["stage2_peaks"] = vis_peaks

        # ---------------- 步骤 4: 横向桥接与单体中轴骨架 ----------------
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        peaks_connected = cv2.morphologyEx(peaks.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_h)
        cnts, _ = cv2.findContours(peaks_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = (nominal_z_mm / self.fx) if depth_mm is None else None
        candidates = []
        vis_2 = color_bgr.copy()
        overlay = vis_2.copy()
        palette = [
            (255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120),
            (255, 210, 40), (120, 160, 255), (200, 255, 80), (255, 80, 140)
        ]

        cand_id = 1
        for c in cnts:
            pts = c.reshape(-1, 2)
            if len(pts) < 8:
                continue
            rect = cv2.minAreaRect(c)
            (rcx, rcy), (rw, rh), _ = rect
            if max(rw, rh) < 80:
                continue

            [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx[0]), float(vy[0])
            if abs(vx) < 0.45:
                continue
            if vx < 0:
                vx, vy = -vx, -vy

            proj = np.dot(pts - np.array([rcx, rcy]), np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            if len_px < 100:
                continue

            sampled_radii = []
            spine_pts_img = []
            for s in np.linspace(min_p * 0.15, max_p * 0.85, 16):
                sx = int(round(rcx + s * vx))
                sy = int(round(rcy + s * vy))
                if 0 <= sx < fg_roi.shape[1] and 0 <= sy < fg_roi.shape[0]:
                    sampled_radii.append(float(dist[sy, sx]))
                    spine_pts_img.append((sx + roi_x1, sy + roi_y1))

            if not sampled_radii:
                continue

            avg_rad = float(np.median(sampled_radii))
            diam_px = float(avg_rad * 2.0)
            global_cx = float(rcx + roi_x1)
            global_cy = float(rcy + roi_y1)

            if depth_mm is not None:
                sample_depths = [depth_mm[py, px] for px, py in spine_pts_img if 0 <= px < w and 0 <= py < h and 350 <= depth_mm[py, px] <= 780]
                z_ref = float(np.median(sample_depths)) if len(sample_depths) >= 4 else nominal_z_mm
                local_scale = z_ref / self.fx
            else:
                z_ref = nominal_z_mm
                local_scale = scale

            l_mm = float(len_px * local_scale)
            d_mm = float(diam_px * local_scale)

            if not (self.min_diam_mm <= d_mm <= self.max_diam_mm and self.min_length_mm <= l_mm <= self.max_length_mm):
                continue

            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0

            u_vec = np.array([vx, vy])
            v_vec = np.array([-vy, vx])
            half_l = len_px * 0.5
            half_w = max(diam_px * 0.5, 4.0)

            c_pt = np.array([global_cx, global_cy])
            p1 = c_pt - half_l * u_vec - half_w * v_vec
            p2 = c_pt + half_l * u_vec - half_w * v_vec
            p3 = c_pt + half_l * u_vec + half_w * v_vec
            p4 = c_pt - half_l * u_vec + half_w * v_vec
            box_corners = np.array([p1, p2, p3, p4], dtype=np.int32)

            color_theme = palette[(cand_id - 1) % len(palette)]
            cv2.fillPoly(overlay, [box_corners], color_theme)
            cv2.polylines(vis_2, [box_corners], True, color_theme, 2)
            cv2.line(vis_2, (int(global_cx - half_l * vx), int(global_cy - half_l * vy)),
                     (int(global_cx + half_l * vx), int(global_cy + half_l * vy)), (255, 255, 255), 2)
            for (px_x, px_y) in spine_pts_img[::4]:
                cv2.circle(vis_2, (px_x, px_y), max(2, int(diam_px * 0.5)), (255, 255, 200), 1)

            put_text(vis_2, f"#{cand_id} D:{d_mm:.1f} L:{l_mm:.0f}",
                     (int(global_cx - 30), int(global_cy - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            candidates.append({
                'id': cand_id, 'center_px': (global_cx, global_cy),
                'length_px': len_px, 'diam_px': diam_px,
                'length_mm': round(l_mm, 1), 'diam_mm': round(d_mm, 1),
                'yaw_deg': round(yaw_deg, 1), 'axis_vector': (vx, vy),
                'box_corners': box_corners, 'z_ref': z_ref, 'spine_pts': spine_pts_img
            })
            cand_id += 1

        vis_2 = cv2.addWeighted(overlay, 0.25, vis_2, 0.75, 0)
        cv2.rectangle(vis_2, (12, 12), (540, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (540, 48), (40, 230, 240), 2)
        put_text(vis_2, f"STAGE 2: SPINES & RIDGES | Instances: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage2_spines"] = vis_2

        # ---------------- 步骤 5: 位姿解算与顶层锁定 (Top 3) ----------------
        from src.vision.asparagus_analyzer import AsparagusTarget, AsparagusAnalyzer
        targets: List[AsparagusTarget] = []

        for sp in candidates:
            cx_val, cy_val = sp['center_px']
            vx_val, vy_val = sp['axis_vector']
            box_corners = sp['box_corners']
            z_ref = sp['z_ref']

            if depth_mm is not None:
                spine_pts = sp.get('spine_pts', [])
                valid_ds = [depth_mm[py, px] for px, py in spine_pts if 0 <= px < w and 0 <= py < h and 350 <= depth_mm[py, px] <= 780]
                if len(valid_ds) >= 3:
                    z_top = float(np.percentile(valid_ds, 15))
                    z_med = float(np.median(valid_ds))
                else:
                    z_top = z_ref
                    z_med = z_ref

                grip_x = float((cx_val - self.cx) * z_med / self.fx)
                grip_y = float((cy_val - self.cy) * z_med / self.fy)
                grip_z = float(z_top)

                if plane_coeff is not None:
                    table_z_local = plane_coeff[0] * cx_val + plane_coeff[1] * cy_val + plane_coeff[2]
                    rel_height_mm = float(table_z_local - z_top)
                else:
                    rel_height_mm = float(640.0 - z_top)
            else:
                z_top = 0.0
                z_med = z_ref
                grip_x = float((cx_val - self.cx) * z_ref / self.fx)
                grip_y = float((cy_val - self.cy) * z_ref / self.fy)
                grip_z = 0.0
                rel_height_mm = 0.0

            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])

                r_mat = frame_transform[:3, :3]
                v_cam = np.array([vx_val, vy_val, 0.0])
                v_robot = r_mat @ v_cam
                r_rad = np.arctan2(v_robot[1], v_robot[0])
                robot_r = float(np.degrees(r_rad))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                robot_x = float(grip_x)
                robot_y = float(grip_y)
                robot_z = float(rel_height_mm)
                robot_r = float(sp['yaw_deg'])

            target = AsparagusTarget(
                id=sp['id'], center_px=(cx_val, cy_val),
                length_px=sp['length_px'], diam_px=sp['diam_px'],
                yaw_deg=sp['yaw_deg'], axis_vector=(vx_val, vy_val),
                box_corners=box_corners, contour=box_corners,
                length_mm=sp['length_mm'], diam_mm=sp['diam_mm'],
                grip_x=round(grip_x, 1), grip_y=round(grip_y, 1), grip_z=round(grip_z, 1),
                z_top=round(z_top, 1), rel_height_mm=round(rel_height_mm, 1),
                robot_x=round(robot_x, 1), robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1), robot_r=round(robot_r, 1),
                is_topmost=False, calibration_source=frame_calib_source
            )
            targets.append(target)

        if len(targets) > 0:
            if depth_mm is not None:
                targets.sort(key=lambda t: t.rel_height_mm, reverse=True)
            else:
                targets.sort(key=lambda t: (t.length_px * t.diam_px), reverse=True)
            targets = targets[:3]
            for rank_i, t in enumerate(targets):
                t.id = rank_i + 1
                t.is_topmost = (rank_i == 0)

        # 渲染位姿图 (调用现有 draw_detections 工具)
        dummy_analyzer = AsparagusAnalyzer(self.fx, self.fy, self.cx, self.cy)
        vis_3 = dummy_analyzer.draw_detections(color_bgr, targets, sel_target_idx=0)
        step_images["stage3_poses"] = vis_3

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=targets,
            elapsed_ms=round(elapsed_ms, 1),
            step_snapshots=step_images,
            extra_metrics={"candidate_count": len(candidates), "roi_pixels": int(np.count_nonzero(fg_full))}
        )
