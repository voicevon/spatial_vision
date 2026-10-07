"""
算法 H2：基于距离变换与形态学骨架化的感知流水线
============================================================
核心原理:
  1. 传送带核心 ROI 裁切 + 双边滤波保边去噪
  2. 自适应阈值二值化获取前景掩膜
  3. 距离变换 + skimage 骨架化提取单像素中心线
  4. 骨架连通分量聚类与主轴方向拟合
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
from src.vision.pipelines.h2_distance_skeleton import distance_transform_skeleton
from src.vision.pipelines.multi_object_separator import (
    watershed_presegment,
    cut_skeleton_junctions,
    apply_per_label_skeleton,
)


@PipelineRegistry.register("distance_skeleton", "算法 H2: 距离变换骨架法 (Distance Skeleton)")
class DistanceSkeletonPipeline(BaseAsparagusPipeline):
    """算法 H2：基于距离变换与形态学骨架化的纯 2D 感知流水线"""

    name = "算法 H2: 距离变换骨架法 (Distance Skeleton)"
    description = "自适应二值化 + 距离变换 + 形态学骨架化 + 连通分量主轴拟合"

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        super().__init__(fx, fy, cx, cy)
        self.min_length_mm = 60.0
        self.max_length_mm = 600.0
        self.min_diam_mm = 2.0
        self.max_diam_mm = 45.0
        self.peeler = OcclusionPeeler(t_junction_radius=18.0)

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep(
                "stage1_prep", "1.预处理",
                "ROI 区域裁切与双边滤波保边去噪",
                details="双边滤波保边去噪后进行自适应阈值二值化分割前景",
                parameters="d=5, sigmaColor=35, sigmaSpace=35; adaptiveThreshold blockSize=25, C=8",
                pros_cons="优点: 自适应阈值应对不均匀光照; 缺点: 阈值参数需针对实际场景调优"
            ),
            PipelineStep(
                "stage2_binary", "2.二值分割",
                "自适应阈值二值化 + 形态学闭运算填平沟壑",
                details="自适应高斯阈值提取前景掩膜，形态学闭运算连接断裂区域",
                parameters="blockSize=25, C=8; 闭运算核 (7,7)",
                pros_cons="优点: 对传送带背景分离效果好; 缺点: 粘连物料可能合并"
            ),
            PipelineStep(
                "stage3_skeleton", "3.骨架化",
                "距离变换 + skimage 形态学骨架提取",
                details="对二值掩膜执行距离变换，再使用 skimage.morphology.skeletonize 提取单像素宽中轴骨架",
                parameters="min_length=30 (过滤短分支)",
                pros_cons="优点: 单像素骨架准确表达中心线; 缺点: 分支点可能产生伪枝"
            ),
            PipelineStep(
                "stage4_spines", "4.主干拟合",
                "骨架连通分量聚类与主轴方向拟合",
                details="对骨架执行连通分量分析，对每个连通分量进行 fitLine 拟合主干方向与长度",
                parameters="min_area=40px; 距离变换中值估算直径",
                pros_cons="优点: 拟合直接来自骨架; 缺点: 交叉处需后续拓扑处理"
            ),
            PipelineStep(
                "stage5_poses", "5.顶层位姿",
                "纯 2D 叠压拓扑剥层与 Top 3 抓取位姿输出",
                details="运行有向拓扑剥层算法，仲裁交叉遮挡关系，锁定最顶层芦笋",
                parameters="t_junction_radius=18px; 顶层抓取高度=35mm",
                pros_cons="优点: 位姿稳定; 缺点: 严重粘连时骨架可能合并"
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

        # ---------------- 步骤 2: 二值分割与 Watershed 预分割 ----------------
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 25, 8)
        # 反转 (使物料为白色)
        binary = cv2.bitwise_not(binary)
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)

        # 多目标 Watershed 预分割
        ws_labels = watershed_presegment(binary, min_area=60)
        num_ws = int(ws_labels.max())

        vis_2 = (color_bgr.astype(np.float32) * 0.20).astype(np.uint8)
        vis_2[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 165, 255), 2)
        cv2.rectangle(vis_2, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (640, 48), (0, 165, 255), 2)
        put_text(vis_2, f"STAGE 2: BINARY & WATERSHED (Separated Regions: {num_ws})",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 165, 255), 2)
        step_images["stage2_binary"] = vis_2

        # ---------------- 步骤 3: 骨架化与分叉切割 ----------------
        if num_ws > 1:
            skeleton = apply_per_label_skeleton(ws_labels, binary)
        else:
            skeleton = distance_transform_skeleton(binary, min_length=25)

        # 切断分叉点，使粘连或交叉芦笋拆为独立路径
        skeleton = cut_skeleton_junctions(skeleton, radius=2)

        # 计算距离变换以估算各骨架点处的直径
        dist_map = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

        skel_color = np.zeros((skeleton.shape[0], skeleton.shape[1], 3), dtype=np.uint8)
        skel_color[skeleton > 0] = (0, 255, 255)

        vis_3 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_3[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.addWeighted(
            vis_3[roi_y1:roi_y2, roi_x1:roi_x2], 0.5, skel_color, 0.5, 0
        )
        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (0, 255, 255), 2)
        skel_px = int(np.sum(skeleton > 0))
        put_text(vis_3, f"STAGE 3: SKELETON (Junction-Cut | Pixels: {skel_px})",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 2)
        step_images["stage3_skeleton"] = vis_3

        # ---------------- 步骤 4: 连通分量主干拟合 ----------------
        scale_2d = nominal_z_mm / self.fx
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(skeleton, connectivity=8)
        candidates: List[CandidateSpine] = []
        vis_4 = color_bgr.copy()
        cand_id = 1
        palette = [(255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120), (255, 210, 40)]

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 40:
                continue
            pts = np.column_stack(np.where(labels == i))  # (N, 2): (y, x)
            pts_xy = pts[:, ::-1].astype(np.float32)  # (N, 2): (x, y) in ROI coords

            [vx_v, vy_v, x0_v, y0_v] = cv2.fitLine(pts_xy, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx_v[0]), float(vy_v[0])
            if vy < 0:
                vx, vy = -vx, -vy

            mean_pt = np.mean(pts_xy, axis=0)
            proj = np.dot(pts_xy - mean_pt, np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            len_mm = len_px * scale_2d

            # 从距离变换估算直径
            diam_px = float(np.median([dist_map[pt[0], pt[1]] for pt in pts]) * 2.0)
            diam_mm = diam_px * scale_2d

            if not (self.min_length_mm <= len_mm <= self.max_length_mm and
                    self.min_diam_mm <= diam_mm <= self.max_diam_mm):
                continue

            # 转全局坐标
            cx_val = float(mean_pt[0] + (min_p + max_p) * 0.5 * vx + roi_x1)
            cy_val = float(mean_pt[1] + (min_p + max_p) * 0.5 * vy + roi_y1)
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
        put_text(vis_4, f"STAGE 4: SKELETON SPINES | Candidates: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (40, 230, 240), 2)
        step_images["stage4_spines"] = vis_4

        # ---------------- 步骤 5: 叠压拓扑剥层与 Top 3 输出 ----------------
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = binary
        peeled_layers = self.peeler.peel_layers(candidates, edge_image=full_mask)

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
        put_text(vis_5, f"STAGE 5: SKELETON POSES | Top {len(top_targets)} Targets",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 120), 2)
        step_images["stage5_poses"] = vis_5

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=top_targets,
            elapsed_ms=elapsed_ms,
            step_snapshots=step_images,
            extra_metrics={"candidates_found": len(candidates), "skeleton_pixels": skel_px}
        )
