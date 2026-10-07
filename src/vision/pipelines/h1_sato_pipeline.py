"""
算法 H1：基于 Sato 多尺度 Vesselness 管状能量的感知流水线
============================================================
核心原理:
  1. 传送带核心 ROI 裁切 + 双边滤波保边去噪
  2. Sato 多尺度 Hessian 特征值管状滤波：利用两个负特征值乘积激发柱状结构
  3. 能量脊线极大值抑制提取中心轴点阵
  4. 点阵聚类与 RANSAC 鲁棒主轴直线拟合
  5. 纯 2D 叠压拓扑剥层 (Occlusion Peeling)，锁定最顶层可抓取物料
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.occlusion_peeler import CandidateSpine, OcclusionPeeler
from src.vision.pipelines.registry import PipelineRegistry
from src.vision.pipelines.h1_sato_vesselness import compute_sato_vesselness
from src.vision.pipelines.multi_object_separator import dbscan_cluster_ridge_points


@PipelineRegistry.register("sato_vesselness", "算法 H1: Sato管状滤波法 (Sato Vesselness)")
class SatoVesselnessPipeline(BaseAsparagusPipeline):
    """算法 H1：基于多尺度 Sato Vesselness 管状滤波的纯 2D 感知流水线"""

    name = "算法 H1: Sato管状滤波法 (Sato Vesselness)"
    description = "多尺度 Hessian 特征值乘积 + Sato 管状能量激发 + 能量脊线极大值跟踪"

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        super().__init__(fx, fy, cx, cy)
        self.min_length_mm = 60.0
        self.max_length_mm = 600.0
        self.min_diam_mm = 6.0
        self.max_diam_mm = 45.0
        self.peeler = OcclusionPeeler(t_junction_radius=18.0)

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep(
                "stage1_prep", "1.预处理",
                "ROI 区域裁切与双边滤波保边去噪",
                details="双边滤波在保留芦笋边缘轮廓的同时平滑表面纹理与水珠反光",
                parameters="d=5, sigmaColor=35, sigmaSpace=35",
                pros_cons="优点: 保边效果好; 缺点: 计算量中等"
            ),
            PipelineStep(
                "stage2_sato", "2.Sato管状能量",
                "多尺度 Sato Vesselness 管状响应计算",
                details="在多个高斯尺度下计算 Hessian 矩阵特征值，利用两个负特征值的绝对值乘积作为管状响应。"
                        "与 Frangi 的区别：Sato 不使用各向异性比率 Rb 滤除斑点，而是直接以特征值符号筛选柱状结构",
                parameters="scales=(1, 2, 3); 特征值 lambda1, lambda2 均需 < 0",
                pros_cons="优点: 公式简洁，对亮柱状结构响应纯净; 缺点: 无各向异性比率门限，圆斑抑制弱于 Frangi"
            ),
            PipelineStep(
                "stage3_ridges", "3.能量脊线",
                "垂向局部极大值抑制提取中心轴点阵",
                details="沿垂直截面扫描 Sato Vesselness 响应图，锁定局部能量最高峰，估算粗细",
                parameters="scan_step_x=6px, thresh_val=25",
                pros_cons="优点: 脊点定位精准; 缺点: 极暗物料能量可能不足"
            ),
            PipelineStep(
                "stage4_spines", "4.主干拟合",
                "横向点阵聚类与主轴鲁棒直线拟合",
                details="将离散能量峰值聚合为连通主轴簇，利用 fitLine 拟合主干方向与长度",
                parameters="max_dx=22px, max_dy=12px; min_pts=12",
                pros_cons="优点: 跨越断隙拟合; 缺点: 并排紧贴时需分离两簇能量峰"
            ),
            PipelineStep(
                "stage5_poses", "5.顶层位姿",
                "纯 2D 叠压拓扑剥层与 Top 3 抓取位姿输出",
                details="以 Sato 能量掩膜运行有向拓扑剥层，仲裁交叉遮挡关系，锁定 Layer 0 最顶层芦笋",
                parameters="t_junction_radius=18px; 顶层抓取高度=35mm",
                pros_cons="优点: 位姿稳定; 缺点: 计算整体耗时约 25~40ms"
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

        # ---------------- 步骤 1: 预处理 ----------------
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        filtered = cv2.bilateralFilter(roi_bgr, d=5, sigmaColor=35, sigmaSpace=35)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        vis_1 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_1[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (40, 230, 240), 2)
        cv2.rectangle(vis_1, (12, 12), (580, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (580, 48), (40, 230, 240), 2)
        put_text(vis_1, "STAGE 1: PREPROCESSING (ROI Bilateral Filter)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 230, 240), 2)
        step_images["stage1_prep"] = vis_1

        # ---------------- 步骤 2: Sato Vesselness ----------------
        vessel_u8 = compute_sato_vesselness(gray, scales=[1, 2, 3])
        v_color = cv2.applyColorMap(vessel_u8, cv2.COLORMAP_VIRIDIS)

        vis_2 = (color_bgr.astype(np.float32) * 0.20).astype(np.uint8)
        vis_2[roi_y1:roi_y2, roi_x1:roi_x2] = v_color
        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 240, 140), 2)
        cv2.rectangle(vis_2, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (640, 48), (0, 240, 140), 2)
        put_text(vis_2, "STAGE 2: SATO VESSELNESS (Multi-Scale Tubular Filter)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 140), 2)
        step_images["stage2_sato"] = vis_2

        # ---------------- 步骤 3: 能量脊线极大值提取 ----------------
        roi_h, roi_w = gray.shape
        ridge_points: List[Tuple[float, float, float]] = []
        scale_2d = nominal_z_mm / self.fx
        vis_3 = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)

        scan_step_x = 8
        thresh_val = 25
        for sx in range(12, roi_w - 12, scan_step_x):
            col_v = vessel_u8[:, sx]
            above = col_v[4:-4] > thresh_val
            if not np.any(above):
                continue
            peaks = np.where(
                above &
                (col_v[4:-4] >= col_v[3:-5]) &
                (col_v[4:-4] >= col_v[5:-3]) &
                (col_v[4:-4] > col_v[1:-7]) &
                (col_v[4:-4] > col_v[7:-1])
            )[0] + 4
            for y in peaks:
                y_up = max(0, y - 20)
                y_down = min(roi_h - 1, y + 20)
                diam_est = float(max(10, (y_down - y_up) * 0.7))

                gx = float(sx + roi_x1)
                gy = float(y + roi_y1)
                ridge_points.append((gx, gy, diam_est))
                cv2.circle(vis_3, (int(gx), int(gy)), 2, (0, 255, 255), -1)

        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (80, 240, 120), 2)
        put_text(vis_3, f"STAGE 3: RIDGE PEAKS (Sato NMS | Points: {len(ridge_points)})",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 240, 120), 2)
        step_images["stage3_ridges"] = vis_3

        # ---------------- 步骤 4: DBSCAN 聚类与主干拟合 (多根分离) ----------------
        clusters = dbscan_cluster_ridge_points(ridge_points, eps=0.0, min_samples=10)
        candidates: List[CandidateSpine] = []
        vis_4 = color_bgr.copy()
        cand_id = 1
        palette = [(255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120), (255, 210, 40)]

        for cluster_pts in clusters:
            if len(cluster_pts) < 12:
                continue

            pts_arr = np.array([[p[0], p[1]] for p in cluster_pts], dtype=np.float32)
            diam_vals = [p[2] for p in cluster_pts]

            [vx_v, vy_v, x0_v, y0_v] = cv2.fitLine(pts_arr, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx_v[0]), float(vy_v[0])
            if abs(vx) < 0.45:
                continue
            if vx < 0:
                vx, vy = -vx, -vy

            mean_pt = np.mean(pts_arr, axis=0)
            proj = np.dot(pts_arr - mean_pt, np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            len_mm = len_px * scale_2d
            diam_px = float(np.median(diam_vals))
            diam_mm = diam_px * scale_2d

            if not (self.min_length_mm <= len_mm <= self.max_length_mm and
                    self.min_diam_mm <= diam_mm <= self.max_diam_mm):
                continue

            cx_val = float(mean_pt[0] + (min_p + max_p) * 0.5 * vx)
            cy_val = float(mean_pt[1] + (min_p + max_p) * 0.5 * vy)
            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0

            u_vec = np.array([vx, vy])
            v_vec = np.array([-vy, vx])
            half_l = len_px * 0.5
            half_w = diam_px * 0.5
            c_pt = np.array([cx_val, cy_val])
            p1 = c_pt - half_l * u_vec - half_w * v_vec
            p2 = c_pt + half_l * u_vec - half_w * v_vec
            p3 = c_pt + half_l * u_vec + half_w * v_vec
            p4 = c_pt - half_l * u_vec + half_w * v_vec
            box_corners = np.array([p1, p2, p3, p4], dtype=np.int32)

            cand = CandidateSpine(
                id=cand_id,
                center_px=(cx_val, cy_val),
                length_px=len_px,
                diam_px=diam_px,
                yaw_deg=yaw_deg,
                axis_vector=(vx, vy),
                box_corners=box_corners,
                extra_data={'l_mm': len_mm, 'd_mm': diam_mm, 'scale': scale_2d}
            )
            candidates.append(cand)

            col = palette[(cand_id - 1) % len(palette)]
            cv2.polylines(vis_4, [box_corners], True, col, 2)
            cv2.line(vis_4,
                     (int(cx_val - half_l * vx), int(cy_val - half_l * vy)),
                     (int(cx_val + half_l * vx), int(cy_val + half_l * vy)),
                     (255, 255, 255), 2)
            put_text(vis_4, f"#{cand_id} D:{diam_mm:.1f} L:{len_mm:.0f}",
                     (int(cx_val - 35), int(cy_val - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cand_id += 1

        cv2.rectangle(vis_4, (12, 12), (600, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_4, (12, 12), (600, 48), (40, 230, 240), 2)
        put_text(vis_4, f"STAGE 4: SATO SPINES FITTED | Candidates: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 230, 240), 2)
        step_images["stage4_spines"] = vis_4

        # ---------------- 步骤 5: 叠压拓扑剥层与 Top 3 输出 ----------------
        full_vessel_mask = np.zeros((h, w), dtype=np.uint8)
        full_vessel_mask[roi_y1:roi_y2, roi_x1:roi_x2] = (vessel_u8 > 25).astype(np.uint8) * 255
        peeled_layers = self.peeler.peel_layers(candidates, edge_image=full_vessel_mask)

        ordered_candidates: List[Tuple[CandidateSpine, int]] = []
        for layer_idx, layer_cands in enumerate(peeled_layers):
            for c_item in layer_cands:
                ordered_candidates.append((c_item, layer_idx))

        from src.vision.asparagus_analyzer import AsparagusTarget, AsparagusAnalyzer
        targets: List[AsparagusTarget] = []

        for rank_idx, (cand, layer_level) in enumerate(ordered_candidates):
            cx_val, cy_val = cand.center_px
            vx, vy = cand.axis_vector
            len_mm = cand.extra_data['l_mm']
            diam_mm = cand.extra_data['d_mm']

            rel_h = max(10.0, 35.0 - layer_level * 15.0)
            z_nominal = nominal_z_mm - rel_h

            grip_x = float((cx_val - self.cx) * z_nominal / self.fx)
            grip_y = float((cy_val - self.cy) * z_nominal / self.fy)
            grip_z = float(z_nominal)

            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])
                r_mat = frame_transform[:3, :3]
                v_robot = r_mat @ np.array([vx, vy, 0.0])
                robot_r = float(np.degrees(np.arctan2(v_robot[1], v_robot[0])))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                robot_x, robot_y, robot_z = grip_x, grip_y, rel_h
                robot_r = cand.yaw_deg

            targets.append(AsparagusTarget(
                id=rank_idx + 1,
                center_px=(cx_val, cy_val),
                length_px=cand.length_px,
                diam_px=cand.diam_px,
                yaw_deg=round(cand.yaw_deg, 1),
                axis_vector=(vx, vy),
                box_corners=cand.box_corners,
                contour=cand.box_corners,
                length_mm=round(len_mm, 1),
                diam_mm=round(diam_mm, 1),
                grip_x=round(grip_x, 1),
                grip_y=round(grip_y, 1),
                grip_z=round(grip_z, 1),
                z_top=round(z_nominal, 1),
                rel_height_mm=round(rel_h, 1),
                robot_x=round(robot_x, 1),
                robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1),
                robot_r=round(robot_r, 1),
                is_topmost=(rank_idx == 0),
                calibration_source=frame_calib_source
            ))

        top_targets = targets[:3]
        dummy_analyzer = AsparagusAnalyzer(self.fx, self.fy, self.cx, self.cy)
        vis_5 = dummy_analyzer.draw_detections(color_bgr, top_targets, sel_target_idx=0)
        cv2.rectangle(vis_5, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_5, (12, 12), (640, 48), (0, 255, 120), 2)
        put_text(vis_5, f"STAGE 5: SATO POSES | Top {len(top_targets)} Targets",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 2)
        step_images["stage5_poses"] = vis_5

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=top_targets,
            elapsed_ms=elapsed_ms,
            step_snapshots=step_images,
            extra_metrics={"candidates_found": len(candidates), "ridge_points": len(ridge_points)}
        )


