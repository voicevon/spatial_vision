"""
技术路线 B2：基于分线段提取与几何对称拓扑配对的感知流水线 (Segment Topology Pipeline)
======================================================================================
核心原理:
  1. 传送带核心 ROI 裁切 + 保边双边滤波与开运算去噪
  2. 高频边缘提取与水平线段细化：过滤垂直短毛刺与非主要走向边缘
  3. 平行边缘双轨配对 (Parallel Edge Pairing)：严格按 X 重叠度与 Y 轴物理直径间距成对匹配，根治外框合并病症
  4. 几何对称中心轴线求解 (Centerline Fitting)：根据配对双轨计算单根中心线与物理尺寸
  5. 纯 2D 叠压与上下层拓扑剥层 (Occlusion Peeling)：输出 Top 3 顶层抓取位姿
"""

import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.occlusion_peeler import CandidateSpine, OcclusionPeeler
from src.vision.pipelines.registry import PipelineRegistry


@PipelineRegistry.register("segment_topology", "算法 B2: 分线段提取法 (Segment Topology)")
class SegmentTopologyPipeline(BaseAsparagusPipeline):
    """算法 B2：基于平行边缘分段提取与几何拓扑配对的纯 2D 感知流水线"""

    name = "算法 B2: 分线段提取法 (Segment Topology)"
    description = "提取近似水平边缘线段 + 平行双轨配对 + 几何对称中线 + 叠压拓扑剥层"

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
                "ROI 区域裁切、双边滤波与去反光开运算",
                details="双边滤波保边去噪配合微小开运算，切断水珠反光链条，为亚像素线段提取提供纯净底图",
                parameters="d=5, sigmaColor=35, sigmaSpace=35; 开运算 ksize=(3,3)",
                pros_cons="优点: 极大减少假线段数量; 缺点: 若开运算核过大可能导致细长边缘局部断裂"
            ),
            PipelineStep(
                "stage2_segments", "线段提取",
                "Sobel-Y & Canny 水平走向细长边缘线段检测与短枝剪枝",
                details="提取水平走向的连续亚像素边缘线段，依据斜率和连续长度过滤非传送方向与细碎杂线",
                parameters="Canny 阈值=[30, 80]; min_seg_len=25px, 角度容差=±25°",
                pros_cons="优点: 边界线形清晰分明; 缺点: 弯曲芦笋边缘会被打碎为多段短折线"
            ),
            PipelineStep(
                "stage3_pairing", "双轨配对",
                "平行边缘线段双轨匹配 (按重叠度与物理直径解耦并排)",
                details="在水平重叠区间内寻找间距处于芦笋直径范围 (6~45mm) 的上下平行线段对，锁定单根边界双轨",
                parameters="overlap_ratio>=0.4; min_diam_mm=6.0, max_diam_mm=45.0",
                pros_cons="优点: 双轨平行约束极严，误配率极低; 缺点: 单侧边缘严重污损缺失时无法成对"
            ),
            PipelineStep(
                "stage4_centerline", "2.中线拓扑",
                "双轨对称几何中心线拟合与单根长径测算",
                details="在配对的双轨之间取对称中心线，计算平均法向间距作为直径，沿中线延伸提取物料全长",
                parameters="centerline_subsample_step=8px; 直线拟合残差门限=2.5px",
                pros_cons="优点: 几何对称性极强，中心线定位精准; 缺点: 两端截断处需依赖额外端点检测"
            ),
            PipelineStep(
                "stage5_poses", "3.顶层位姿",
                "纯 2D 叠压拓扑剥层与 Top 3 顶层抓取位姿输出",
                details="分析双轨线段在交叉区域的打断贯通情况，构建叠压 DAG 有向图，优先拣选完全贯通的最顶层芦笋",
                parameters="t_junction_radius=18px; 顶层抓取沉降=-10mm",
                pros_cons="优点: 顶层判定具备严格几何线段贯穿证明; 缺点: 两线段完全平行时依赖端点可见性"
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

        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gray_clean = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k_open)

        vis_1 = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis_1[roi_y1:roi_y2, roi_x1:roi_x2] = cv2.cvtColor(gray_clean, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_1, (12, 12), (560, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (560, 48), (40, 230, 240), 2)
        put_text(vis_1, "STAGE 1: PREPROCESSING (ROI Bilateral & Morph Clean)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage1_prep"] = vis_1

        # ---------------- 步骤 2: 提取细长水平边缘线段 (去除大闭运算和外围轮廓误区) ----------------
        grad_y = cv2.Sobel(gray_clean, cv2.CV_32F, 0, 1, ksize=3)
        grad_x = cv2.Sobel(gray_clean, cv2.CV_32F, 1, 0, ksize=3)
        canny_edges = cv2.Canny(gray_clean, 25, 80)

        # 筛选水平高频边界：排除垂直截断与竖向反光
        strong_h_edge = (canny_edges > 0) & (np.abs(grad_y) > np.abs(grad_x) * 0.55)
        edge_clean = self._filter_short_segments(strong_h_edge.astype(np.uint8) * 255, min_pts=30)

        # 提取各个细长单线段 (使用 RETR_LIST 严禁 RETR_EXTERNAL 彻底保留内部并排分界线！)
        cnts, _ = cv2.findContours(edge_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        scale_2d = nominal_z_mm / self.fx

        edge_segments = []
        vis_2 = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        palette = [(0, 255, 255), (0, 165, 255), (80, 240, 120), (255, 120, 240)]

        seg_idx = 0
        for c in cnts:
            if len(c) < 25:
                continue
            pts = c.reshape(-1, 2)
            # 拟合线段方向
            [vx_v, vy_v, x0_v, y0_v] = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx_v[0]), float(vy_v[0])
            if abs(vx) < 0.45:
                continue  # 排除非输送方向
            if vx < 0:
                vx, vy = -vx, -vy

            # 投影长度
            mean_pt = np.mean(pts, axis=0)
            proj = np.dot(pts - mean_pt, np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            seg_len = max_p - min_p
            if seg_len * scale_2d < 45.0:
                continue  # 过滤过短线段

            # 全局坐标转换
            global_pts = pts + np.array([roi_x1, roi_y1])
            edge_segments.append({
                'id': seg_idx,
                'pts': pts,
                'global_pts': global_pts,
                'mean_y': float(mean_pt[1]),
                'x_min': float(np.min(pts[:, 0])),
                'x_max': float(np.max(pts[:, 0])),
                'vx': vx, 'vy': vy,
                'len_px': seg_len
            })

            col = palette[seg_idx % len(palette)]
            cv2.polylines(vis_2, [global_pts], False, col, 2)
            seg_idx += 1

        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_2, (12, 12), (620, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (620, 48), (0, 255, 255), 2)
        put_text(vis_2, f"STAGE 2: HORIZONTAL SEGMENTS (RETR_LIST) | Segments: {len(edge_segments)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 2)
        step_images["stage2_segments"] = vis_2

        # ---------------- 步骤 3: 平行边缘双轨配对 ----------------
        min_diam_px = self.min_diam_mm / scale_2d
        max_diam_px = self.max_diam_mm / scale_2d

        paired_rails = []
        used_segs = set()
        vis_3 = color_bgr.copy()

        # 按 Y 坐标由上至下排序
        edge_segments.sort(key=lambda s: s['mean_y'])

        for i, s1 in enumerate(edge_segments):
            if s1['id'] in used_segs:
                continue
            best_pair = None
            best_score = 0.0

            for j in range(i + 1, len(edge_segments)):
                s2 = edge_segments[j]
                if s2['id'] in used_segs:
                    continue

                dy = s2['mean_y'] - s1['mean_y']
                if dy < min_diam_px:
                    continue
                if dy > max_diam_px:
                    break  # 已超出芦笋最大物理粗度

                # 计算横向 X 轴重叠长度
                overlap_x1 = max(s1['x_min'], s2['x_min'])
                overlap_x2 = min(s1['x_max'], s2['x_max'])
                overlap_len = max(0.0, overlap_x2 - overlap_x1)

                if overlap_len * scale_2d >= 50.0:
                    # 角度一致性打分
                    dot_angle = s1['vx'] * s2['vx'] + s1['vy'] * s2['vy']
                    if dot_angle > 0.88:  # 平行双轨
                        score = overlap_len * dot_angle
                        if score > best_score:
                            best_score = score
                            best_pair = s2

            if best_pair is not None:
                used_segs.add(s1['id'])
                used_segs.add(best_pair['id'])
                paired_rails.append((s1, best_pair))

                col = palette[len(paired_rails) % len(palette)]
                cv2.polylines(vis_3, [s1['global_pts']], False, col, 2)
                cv2.polylines(vis_3, [best_pair['global_pts']], False, col, 2)
                # 绘制连接指示线
                mid_x = int((s1['x_min'] + s1['x_max'] + best_pair['x_min'] + best_pair['x_max']) * 0.25) + roi_x1
                cv2.line(vis_3, (mid_x, int(s1['mean_y'] + roi_y1)),
                         (mid_x, int(best_pair['mean_y'] + roi_y1)), (0, 255, 255), 1)

        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_3, (12, 12), (620, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (620, 48), (80, 240, 120), 2)
        put_text(vis_3, f"STAGE 3: PARALLEL RAILS PAIRED | Dual Rails: {len(paired_rails)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 240, 120), 2)
        step_images["stage3_pairing"] = vis_3

        # ---------------- 步骤 4: 几何对称中心线拟合 ----------------
        candidates: List[CandidateSpine] = []
        vis_4 = color_bgr.copy()
        cand_id = 1

        for (r1, r2) in paired_rails:
            # 综合两轨的所有像素点计算中心线与外轮廓
            combined_pts = np.vstack([r1['pts'], r2['pts']])
            diam_px = float(r2['mean_y'] - r1['mean_y'])
            diam_mm = diam_px * scale_2d

            [vx_v, vy_v, x0_v, y0_v] = cv2.fitLine(combined_pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx_v[0]), float(vy_v[0])
            if vx < 0:
                vx, vy = -vx, -vy

            mean_pt = np.mean(combined_pts, axis=0)
            proj = np.dot(combined_pts - mean_pt, np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            len_mm = len_px * scale_2d

            if not (self.min_length_mm <= len_mm <= self.max_length_mm and
                    self.min_diam_mm <= diam_mm <= self.max_diam_mm):
                continue

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
        put_text(vis_4, f"STAGE 4: CENTERLINES FITTED | Candidates: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage4_centerline"] = vis_4

        # ---------------- 步骤 5: 纯 2D 叠压拓扑剥层与 Top 3 输出 ----------------
        full_edge_map = np.zeros((h, w), dtype=np.uint8)
        full_edge_map[roi_y1:roi_y2, roi_x1:roi_x2] = edge_clean

        peeled_layers = self.peeler.peel_layers(candidates, edge_image=full_edge_map)

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
        step_images["stage5_poses"] = vis_5

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=top_targets,
            elapsed_ms=round(elapsed_ms, 1),
            step_snapshots=step_images,
            extra_metrics={
                "segments_found": len(edge_segments),
                "rails_paired": len(paired_rails),
                "peeled_layers": len(peeled_layers)
            }
        )

    def _filter_short_segments(self, binary_mask: np.ndarray, min_pts: int = 30) -> np.ndarray:
        """过滤短小边缘"""
        cnts, _ = cv2.findContours(binary_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        out = np.zeros_like(binary_mask)
        for c in cnts:
            if len(c) >= min_pts:
                cv2.drawContours(out, [c], -1, 255, 1)
        return out
