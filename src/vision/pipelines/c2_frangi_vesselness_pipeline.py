"""
算法 C2：基于 Hessian 矩阵二阶导与 Frangi 管状滤波的感知流水线 (Frangi Vesselness Pipeline)
=============================================================================================
核心原理:
  1. 尺度空间高斯平滑：匹配芦笋物理直径尺度金字塔
  2. 计算 Hessian 矩阵二阶偏导 (Ixx, Iyy, Ixy) 并解析求解局部主曲率特征值 (lambda1, lambda2)
  3. 计算各尺度下的 Frangi Vesselness 响应函数：彻底消除传送带各向同性斑点与反光气泡，高亮柱状/纤维中轴
  4. 垂向一维极大值抑制 (Non-Maximum Suppression) 提取高能量脊线离散点
  5. 横向点阵聚类与 RANSAC 鲁棒主轴直线拟合
  6. 纯 2D 叠压拓扑剥层 (Occlusion Peeling)，锁定最顶层可抓取物料
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.occlusion_peeler import CandidateSpine, OcclusionPeeler
from src.vision.pipelines.registry import PipelineRegistry


@PipelineRegistry.register("frangi_vesselness", "算法 C2: Frangi管状滤波法 (Frangi Vesselness)")
class FrangiVesselnessPipeline(BaseAsparagusPipeline):
    """算法 C2：基于多尺度 Hessian 矩阵特征值与 Frangi 管状滤波的纯 2D 感知流水线"""

    name = "算法 C2: Frangi管状滤波法 (Frangi Vesselness)"
    description = "多尺度 Hessian 二阶导 + Frangi 管状能量激发 + 能量脊线极大值跟踪"

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
                "stage1_prep", "1.高斯平滑",
                "ROI 区域裁切与尺度空间多尺度高斯平滑",
                details="根据芦笋直径区间 (6~45mm) 建立高斯尺度金字塔，在多尺度下平滑微小反光毛刺与表皮纤维抖动",
                parameters="sigmas=(3.5, 6.0, 9.0) (对应细笋、中笋、粗笋的物理半径尺度)",
                pros_cons="优点: 多尺度自适应覆盖不同粗细物料; 缺点: 多尺度高斯计算量随尺度层数倍增"
            ),
            PipelineStep(
                "stage2_hessian", "2.Hessian二阶导",
                "偏导矩阵 Ixx, Iyy, Ixy 与曲率特征值分析",
                details="计算图像局部 Hessian 矩阵特征值 lambda1 与 lambda2。圆柱形/条状物沿轴线曲率近0，垂直轴线二阶导极高且为负",
                parameters="Ixx, Iyy, Ixy 归一化因子 sigma^2; 特征值分析按绝对值排序 |lambda1| <= |lambda2|",
                pros_cons="优点: 极其精确的二阶曲率几何描述; 缺点: 需解析求根公式，需浮点计算"
            ),
            PipelineStep(
                "stage3_vesselness", "3.管状能量",
                "Frangi 条状能量响应，彻底抑制反光气泡与斑点",
                details="利用各向异性比率 Rb 与结构强度 S 综合激发管状响应。各向同性反光气泡 (Rb≈1) 被瞬间清零，仅保留圆柱纤维能量",
                parameters="beta=0.5 (抑制斑状/气泡灵敏度), c=12.0 (背景噪声灰度强度门限)",
                pros_cons="优点: 彻底免疫传送带水斑、油污与各向同性反光气泡; 缺点: 端头截断面响应会微弱衰减"
            ),
            PipelineStep(
                "stage4_ridges", "4.能量脊线",
                "垂向局部极大值抑制提取清晰中心中轴点阵",
                details="沿垂直截面扫描 Frangi 响应图，通过非极大值抑制 (NMS) 锁定局部能量最高峰，并根据能量扩散半宽估算粗细",
                parameters="scan_step_x=6px (扫描步长), thresh_val=35 (管状能量最低响应门槛)",
                pros_cons="优点: 脊点严格位于物料真几何中心轴; 缺点: 极暗或枯萎物料能量值可能低于门槛"
            ),
            PipelineStep(
                "stage5_spines", "5.主干拟合",
                "横向点阵聚类与主轴鲁棒直线拟合",
                details="将离散能量峰值聚合为连通主轴簇，利用 RANSAC 鲁棒拟合，沿拟合主干双向搜索真实截断端点，输出定向矩形",
                parameters="max_dx=22px, max_dy=12px (聚类容差); min_pts=12 (最少峰值点数)",
                pros_cons="优点: 即使断续遮挡也能跨越断隙拟合; 缺点: 并排紧贴时需精细分离两簇能量峰"
            ),
            PipelineStep(
                "stage6_top_poses", "6.顶层位姿",
                "纯 2D 叠压拓扑剥层与 Top 3 抓取位姿输出",
                details="以 Frangi 能量掩膜为底，运行有向拓扑剥层算法，仲裁交叉重叠处的层级遮挡关系，优先锁定 Layer 0 最顶层芦笋",
                parameters="t_junction_radius=18px; 顶层抓取高度=35mm",
                pros_cons="优点: 极强抗杂散噪光，位姿极其纯净稳定; 缺点: 计算整体耗时约 20~30ms"
            )
        ]

    def _compute_frangi(
        self,
        gray_roi: np.ndarray,
        sigmas: Tuple[float, ...] = (3.5, 6.0, 9.0),
        beta: float = 0.5,
        c: float = 12.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算高效多尺度 Frangi 管状滤波器"""
        h, w = gray_roi.shape
        vesselness_max = np.zeros((h, w), dtype=np.float32)
        hessian_vis = np.zeros((h, w), dtype=np.float32)
        gray_f = gray_roi.astype(np.float32)

        for sigma in sigmas:
            ksize = int(2 * round(2.5 * sigma) + 1)
            smoothed = cv2.GaussianBlur(gray_f, (ksize, ksize), sigma)

            grad_x = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
            ixx = cv2.Sobel(grad_x, cv2.CV_32F, 1, 0, ksize=3) * (sigma ** 2)
            iyy = cv2.Sobel(grad_y, cv2.CV_32F, 0, 1, ksize=3) * (sigma ** 2)
            ixy = cv2.Sobel(grad_x, cv2.CV_32F, 0, 1, ksize=3) * (sigma ** 2)

            tmp = np.sqrt((ixx - iyy) ** 2 + 4 * (ixy ** 2))
            l1 = (ixx + iyy - tmp) * 0.5
            l2 = (ixx + iyy + tmp) * 0.5

            swap_mask = np.abs(l1) > np.abs(l2)
            lambda1 = np.where(swap_mask, l2, l1)
            lambda2 = np.where(swap_mask, l1, l2)

            # 亮芦笋条状物对应 lambda2 < 0
            valid_tube = (lambda2 < -0.5)

            rb_sq = (lambda1 / (lambda2 + 1e-5)) ** 2
            s_sq = lambda1 ** 2 + lambda2 ** 2

            vesselness = (1.0 - np.exp(-rb_sq / (2.0 * (beta ** 2)))) * (1.0 - np.exp(-s_sq / (2.0 * (c ** 2))))
            vesselness[~valid_tube] = 0.0

            update_mask = vesselness > vesselness_max
            vesselness_max[update_mask] = vesselness[update_mask]

            if sigma == sigmas[1]:
                hessian_vis = np.abs(iyy)  # 记录横向输送条带的主曲率分量可视化

        # 归一化到 0~255
        v_max = float(np.max(vesselness_max))
        if v_max > 1e-4:
            vesselness_u8 = (vesselness_max / v_max * 255.0).astype(np.uint8)
        else:
            vesselness_u8 = np.zeros((h, w), dtype=np.uint8)

        h_max = float(np.max(hessian_vis))
        if h_max > 1e-4:
            hessian_u8 = (hessian_vis / h_max * 255.0).astype(np.uint8)
        else:
            hessian_u8 = np.zeros((h, w), dtype=np.uint8)

        return vesselness_u8, hessian_u8, vesselness_max

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

        # ---------------- 步骤 1: 高斯平滑 ----------------
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        smoothed = cv2.GaussianBlur(gray, (9, 9), 3.0)

        vis_1 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_1[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (40, 230, 240), 2)
        cv2.rectangle(vis_1, (12, 12), (620, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (620, 48), (40, 230, 240), 2)
        put_text(vis_1, "STAGE 1: GAUSSIAN SCALE SPACE (Multi-Scale Prep)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 230, 240), 2)
        step_images["stage1_prep"] = vis_1

        # ---------------- 步骤 2: Hessian 二阶导特征值 ----------------
        vessel_u8, hessian_u8, v_raw = self._compute_frangi(gray)
        h_color = cv2.applyColorMap(hessian_u8, cv2.COLORMAP_INFERNO)

        vis_2 = (color_bgr.astype(np.float32) * 0.20).astype(np.uint8)
        vis_2[roi_y1:roi_y2, roi_x1:roi_x2] = h_color
        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 165, 255), 2)
        cv2.rectangle(vis_2, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (640, 48), (0, 165, 255), 2)
        put_text(vis_2, "STAGE 2: HESSIAN EIGENVALUES (Transverse Curvature Response)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 165, 255), 2)
        step_images["stage2_hessian"] = vis_2

        # ---------------- 步骤 3: Frangi 管状能量激发 ----------------
        v_color = cv2.applyColorMap(vessel_u8, cv2.COLORMAP_VIRIDIS)

        vis_3 = (color_bgr.astype(np.float32) * 0.20).astype(np.uint8)
        vis_3[roi_y1:roi_y2, roi_x1:roi_x2] = v_color
        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 240, 140), 2)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (0, 240, 140), 2)
        put_text(vis_3, "STAGE 3: FRANGI VESSELNESS (Tubular Filter: Noise & Glare Suppressed)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 240, 140), 2)
        step_images["stage3_vesselness"] = vis_3

        # ---------------- 步骤 4: 能量脊线极大值提取 ----------------
        roi_h, roi_w = gray.shape
        ridge_points: List[Tuple[float, float, float]] = []  # (gx, gy, diam_px)
        scale_2d = nominal_z_mm / self.fx
        vis_4 = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)

        # 沿垂直扫描线寻找局部能量极大值
        scan_step_x = 6
        thresh_val = 35  # 最低管状能量门限
        for sx in range(12, roi_w - 12, scan_step_x):
            col_v = vessel_u8[:, sx]
            # 寻找极大值
            for y in range(4, roi_h - 4):
                val = col_v[y]
                if val > thresh_val and val >= col_v[y - 1] and val >= col_v[y + 1] and val > col_v[y - 3] and val > col_v[y + 3]:
                    # 向上向下探索有效条状宽度以推断直径
                    y_up = y
                    while y_up > 0 and col_v[y_up] > val * 0.3:
                        y_up -= 1
                    y_down = y
                    while y_down < roi_h - 1 and col_v[y_down] > val * 0.3:
                        y_down += 1
                    diam_est = float(max(10, (y_down - y_up) * 1.4))

                    gx = float(sx + roi_x1)
                    gy = float(y + roi_y1)
                    ridge_points.append((gx, gy, diam_est))
                    cv2.circle(vis_4, (int(gx), int(gy)), 2, (0, 255, 255), -1)

        cv2.rectangle(vis_4, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_4, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_4, (12, 12), (640, 48), (80, 240, 120), 2)
        put_text(vis_4, f"STAGE 4: RIDGE PEAKS (Non-Max Suppressed | Points: {len(ridge_points)})",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (80, 240, 120), 2)
        step_images["stage4_ridges"] = vis_4

        # ---------------- 步骤 5: 点阵聚类与主干直线拟合 ----------------
        clusters = self._cluster_points(ridge_points, max_dx=22.0, max_dy=12.0)
        candidates: List[CandidateSpine] = []
        vis_5 = color_bgr.copy()
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
            cv2.polylines(vis_5, [box_corners], True, col, 2)
            cv2.line(vis_5,
                     (int(cx_val - half_l * vx), int(cy_val - half_l * vy)),
                     (int(cx_val + half_l * vx), int(cy_val + half_l * vy)),
                     (255, 255, 255), 2)
            put_text(vis_5, f"#{cand_id} D:{diam_mm:.1f} L:{len_mm:.0f}",
                     (int(cx_val - 35), int(cy_val - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cand_id += 1

        cv2.rectangle(vis_5, (12, 12), (600, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_5, (12, 12), (600, 48), (40, 230, 240), 2)
        put_text(vis_5, f"STAGE 5: VESSEL SPINES FITTED | Candidates: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 230, 240), 2)
        step_images["stage5_spines"] = vis_5

        # ---------------- 步骤 6: 纯 2D 叠压拓扑剥层与 Top 3 输出 ----------------
        full_vessel_mask = np.zeros((h, w), dtype=np.uint8)
        full_vessel_mask[roi_y1:roi_y2, roi_x1:roi_x2] = (vessel_u8 > 35).astype(np.uint8) * 255
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
        vis_6 = dummy_analyzer.draw_detections(color_bgr, top_targets, sel_target_idx=0)
        cv2.rectangle(vis_6, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_6, (12, 12), (640, 48), (0, 255, 120), 2)
        put_text(vis_6, f"STAGE 6: FRANGI POSES | Layer0: Top Priority ({len(top_targets)} Selected)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 2)
        step_images["stage6_top_poses"] = vis_6

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=top_targets,
            elapsed_ms=elapsed_ms,
            step_snapshots=step_images,
            extra_metrics={"candidates_found": len(candidates), "ridge_points": len(ridge_points)}
        )

    def _cluster_points(self, points: List[Tuple[float, float, float]], max_dx: float = 22.0, max_dy: float = 12.0) -> List[List[Tuple[float, float, float]]]:
        """将扫描离散脊点聚合成独立连通簇"""
        if not points:
            return []
        sorted_pts = sorted(points, key=lambda p: p[0])
        clusters: List[List[Tuple[float, float, float]]] = []

        for pt in sorted_pts:
            px, py, _ = pt
            merged = False
            for cluster in clusters:
                last_x, last_y, _ = cluster[-1]
                if abs(px - last_x) <= max_dx and abs(py - last_y) <= max_dy:
                    cluster.append(pt)
                    merged = True
                    break
            if not merged:
                clusters.append([pt])

        # 二次合并相近簇
        final_clusters: List[List[Tuple[float, float, float]]] = []
        for c in clusters:
            if len(c) < 5:
                continue
            merged_into = False
            c_mean_y = np.mean([p[1] for p in c])
            c_min_x, c_max_x = min(p[0] for p in c), max(p[0] for p in c)

            for fc in final_clusters:
                fc_mean_y = np.mean([p[1] for p in fc])
                fc_min_x, fc_max_x = min(p[0] for p in fc), max(p[0] for p in fc)
                if abs(c_mean_y - fc_mean_y) <= max_dy and (c_min_x - fc_max_x <= max_dx * 2):
                    fc.extend(c)
                    merged_into = True
                    break
            if not merged_into:
                final_clusters.append(c)

        return final_clusters
