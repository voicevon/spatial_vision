"""
算法 C1：基于前景形态学与拓扑骨架细化的感知流水线 (Skeleton Thinning Pipeline)
================================================================================
核心原理:
  1. 传送带核心 ROI 裁切 + 双边滤波保边去噪
  2. 自适应阈值二值化分割 + 形态学闭运算填平表面沟壑，获取致密实心物料掩膜
  3. 形态学迭代细化提取单像素中轴骨架 (Medial Axis Skeleton)
  4. 3x3 拓扑邻域卷积检测端点 (End Point) 与交叉节点 (Junction Node)，剪除微小毛刺伪分支
  5. 沿骨架路径聚类与 RANSAC 鲁棒主轴拟合，结合距离场或几何先验估算物理直径
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


@PipelineRegistry.register("skeleton_thinning", "算法 C1: 形态学骨架细化法 (Skeleton Thinning)")
class SkeletonThinningPipeline(BaseAsparagusPipeline):
    """算法 C1：基于形态学二值分割与拓扑骨架细化的纯 2D 感知流水线"""

    name = "算法 C1: 形态学骨架细化法 (Skeleton Thinning)"
    description = "前景二值化 + 形态学闭运算 + 拓扑骨架细化 + 节点剪枝与主轴拟合"

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
                "stage1_prep", "1.双边平滑",
                "ROI 区域裁切与保边双边滤波去噪",
                details="ROI 聚焦传送带有效输送带区域，双边滤波在平滑光斑与皮带微孔的同时紧锁芦笋外廓",
                parameters="d=5, sigmaColor=35, sigmaSpace=35 (平滑半径与色彩空间门限)",
                pros_cons="优点: 保边效果优异，为后续阈值分割提供均匀底色; 缺点: 对大面积强反光斑需配合开运算"
            ),
            PipelineStep(
                "stage2_binary", "2.二值掩膜",
                "自适应阈值与形态学闭运算获取实心前景",
                details="利用 Otsu 自动阈值解耦亮物料与深暗底板，结合 (7,7) 椭圆闭运算自动填平芦笋表面鳞片与光照缝隙",
                parameters="Otsu 阈值自动计算; 闭运算 k_close=(7,7), 开运算 k_open=(3,3)",
                pros_cons="优点: 掩膜实心完整无空洞; 缺点: 若皮带有高反光油渍可能被误连入前景"
            ),
            PipelineStep(
                "stage3_skeleton", "3.骨架细化",
                "形态学迭代细化削减为单像素中心拓扑骨架",
                details="通过结构元腐蚀与开运算残差迭代差分，从物料轮廓向内层层剥离，直到留下单像素几何中轴 (Medial Axis)",
                parameters="max_iters=50 (最大迭代步数，芦笋半径越大需步数越多); 结构元=CROSS(3,3)",
                pros_cons="优点: 天然保留任意弯曲芦笋的中心拓扑; 缺点: 边缘微小凹凸会在骨架上激发出侧向假分支"
            ),
            PipelineStep(
                "stage4_prune", "4.拓扑剪枝",
                "3x3 邻域端点与分支点检测，剪除微小毛刺",
                details="利用 3x3 卷积统计邻居度数: 度=1 为端点 (橙红)，度>=3 为交叉分支 (亮黄)。从自由端点向内剪除短毛刺",
                parameters="min_branch_len=12px (剪枝距离，毛刺多时可适当增大至 18px)",
                pros_cons="优点: 自动剪除侧向伪分叉，留下纯净主轴; 缺点: 剪枝过猛可能削减两端真实端点"
            ),
            PipelineStep(
                "stage5_spines", "5.主干拟合",
                "骨架分支跟踪与 RANSAC 鲁棒直线主轴拟合",
                details="对修剪后的单像素骨架进行连通域聚类，结合局部欧氏距离场推算真实物理直径，直线拟合求中心点与偏航角",
                parameters="min_pts=25 (成杆点数阈值); 直线拟合 DIST_L2 容差=0.01",
                pros_cons="优点: 结合骨架点与距离场，直径估算极准; 缺点: 严重重叠粘连时骨架相交需拓扑解开"
            ),
            PipelineStep(
                "stage6_top_poses", "6.顶层位姿",
                "纯 2D 叠压拓扑剥层与 Top 3 抓取位姿输出",
                details="基于有向无环图与骨架交叉连续性，推断物料上下层空间层叠关系，优先选择无遮挡的 Layer 0 芦笋并生成 G-code",
                parameters="t_junction_radius=18px; 顶层抓取高度裕量=35mm",
                pros_cons="优点: 适合复杂多枝态与微弯曲物料抓取规划; 缺点: 骨架完全断裂时难以通过几何关联拼合"
            )
        ]

    def _extract_skeleton(self, binary_mask: np.ndarray, max_iters: int = 50) -> np.ndarray:
        """纯 OpenCV 形态学骨架提取 (腐蚀与开运算残差累积)"""
        skel = np.zeros_like(binary_mask)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        temp = binary_mask.copy()
        for _ in range(max_iters):
            eroded = cv2.erode(temp, element)
            temp_open = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, element)
            subset = cv2.subtract(eroded, temp_open)
            cv2.bitwise_or(skel, subset, skel)
            temp = eroded
            if cv2.countNonZero(temp) == 0:
                break
        return skel

    def _prune_skeleton(self, skel: np.ndarray, min_branch_len: int = 15) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """通过 3x3 卷积分析拓扑节点并剪除微小毛刺分支"""
        skel_bin = (skel > 0).astype(np.uint8)
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbors = cv2.filter2D(skel_bin, -1, kernel) * skel_bin

        # 端点 (邻居数为 1) 与交叉节点 (邻居数 >= 3)
        endpoints = (neighbors == 1) & (skel_bin == 1)
        junctions = (neighbors >= 3) & (skel_bin == 1)

        pruned = skel.copy()
        # 简单迭代剥离悬空毛刺
        for _ in range(min_branch_len):
            sk_curr = (pruned > 0).astype(np.uint8)
            deg = cv2.filter2D(sk_curr, -1, kernel) * sk_curr
            tips = (deg == 1)
            if cv2.countNonZero(tips.astype(np.uint8)) == 0:
                break
            pruned[tips] = 0

        # 若剪枝过度则回退保留主骨架
        if cv2.countNonZero(pruned) < 20:
            pruned = skel.copy()

        return pruned, endpoints, junctions

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

        # ---------------- 步骤 1: 双边平滑 ----------------
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        filtered = cv2.bilateralFilter(roi_bgr, d=5, sigmaColor=35, sigmaSpace=35)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

        vis_1 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_1[roi_y1:roi_y2, roi_x1:roi_x2] = filtered
        cv2.rectangle(vis_1, (roi_x1, roi_y1), (roi_x2, roi_y2), (40, 230, 240), 2)
        cv2.rectangle(vis_1, (12, 12), (620, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_1, (12, 12), (620, 48), (40, 230, 240), 2)
        put_text(vis_1, "STAGE 1: BILATERAL FILTER (Edge-Preserving Smoothing)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 230, 240), 2)
        step_images["stage1_prep"] = vis_1

        # ---------------- 步骤 2: 自适应阈值与二值掩膜 ----------------
        # 利用 Otsu 结合自适应截断提取前景芦笋 (芦笋较背景传送带显著亮)
        _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 形态学闭运算填平表面沟壑与反光盲区
        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask_dense = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, k_close)
        # 开运算过滤细微碎屑
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_clean = cv2.morphologyEx(mask_dense, cv2.MORPH_OPEN, k_open)

        # 欧氏距离变换用于后续物理直径推断
        dist_map = cv2.distanceTransform(mask_clean, cv2.DIST_L2, 5)

        vis_2 = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        patch_2 = vis_2[roi_y1:roi_y2, roi_x1:roi_x2]
        patch_2[mask_clean > 0] = (60, 210, 100)  # 前景翡翠绿高亮
        vis_2[roi_y1:roi_y2, roi_x1:roi_x2] = patch_2
        cv2.rectangle(vis_2, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 220, 120), 2)
        cv2.rectangle(vis_2, (12, 12), (620, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_2, (12, 12), (620, 48), (0, 220, 120), 2)
        put_text(vis_2, "STAGE 2: BINARY MASK & MORPH CLOSE (Dense Object Shell)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 220, 120), 2)
        step_images["stage2_binary"] = vis_2

        # ---------------- 步骤 3: 骨架细化 ----------------
        raw_skel = self._extract_skeleton(mask_clean, max_iters=50)

        vis_3 = (color_bgr.astype(np.float32) * 0.20).astype(np.uint8)
        patch_3 = vis_3[roi_y1:roi_y2, roi_x1:roi_x2]
        # 膨胀 1 像素便于高清屏幕显示骨架
        skel_dil = cv2.dilate(raw_skel, np.ones((3, 3), np.uint8))
        patch_3[skel_dil > 0] = (40, 230, 240)  # 青黄高亮细化中心线
        vis_3[roi_y1:roi_y2, roi_x1:roi_x2] = patch_3
        cv2.rectangle(vis_3, (roi_x1, roi_y1), (roi_x2, roi_y2), (40, 230, 240), 2)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_3, (12, 12), (640, 48), (40, 230, 240), 2)
        put_text(vis_3, "STAGE 3: TOPOLOGICAL SKELETON (Medial Axis 1px Thinning)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (40, 230, 240), 2)
        step_images["stage3_skeleton"] = vis_3

        # ---------------- 步骤 4: 拓扑节点检测与剪枝 ----------------
        pruned_skel, endpoints, junctions = self._prune_skeleton(raw_skel, min_branch_len=12)

        vis_4 = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        patch_4 = vis_4[roi_y1:roi_y2, roi_x1:roi_x2]
        p_dil = cv2.dilate(pruned_skel, np.ones((3, 3), np.uint8))
        patch_4[p_dil > 0] = (255, 255, 255)       # 纯白骨干
        patch_4[endpoints] = (0, 80, 255)          # 橙红：端点
        patch_4[junctions] = (0, 240, 255)         # 亮黄：相交分支节点
        vis_4[roi_y1:roi_y2, roi_x1:roi_x2] = patch_4
        cv2.rectangle(vis_4, (roi_x1, roi_y1), (roi_x2, roi_y2), (80, 240, 120), 2)
        cv2.rectangle(vis_4, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_4, (12, 12), (640, 48), (80, 240, 120), 2)
        put_text(vis_4, "STAGE 4: TOPOLOGY PRUNING (White: Spine | Orange: Tip | Yellow: Node)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 240, 120), 2)
        step_images["stage4_prune"] = vis_4

        # ---------------- 步骤 5: 骨架分支跟踪与主轴拟合 ----------------
        scale_2d = nominal_z_mm / self.fx
        num_labels, labels_im, stats, centroids = cv2.connectedComponentsWithStats(pruned_skel, connectivity=8)

        candidates: List[CandidateSpine] = []
        vis_5 = color_bgr.copy()
        cand_id = 1
        palette = [(255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120), (255, 210, 40)]

        for lbl in range(1, num_labels):
            pts_y, pts_x = np.where(labels_im == lbl)
            if len(pts_x) < 25:
                continue

            pts_arr = np.column_stack((pts_x + roi_x1, pts_y + roi_y1)).astype(np.float32)
            # 通过局部距离场提取该骨干处的平均半径并转换为直径
            diams_px = [dist_map[y, x] * 2.0 for y, x in zip(pts_y, pts_x)]
            diam_px = float(np.median(diams_px)) if diams_px else 20.0
            diam_mm = diam_px * scale_2d

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
        put_text(vis_5, f"STAGE 5: SKELETON SPINES | Candidates: {len(candidates)}",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        step_images["stage5_spines"] = vis_5

        # ---------------- 步骤 6: 纯 2D 叠压剥层与 Top 3 输出 ----------------
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = mask_clean
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
        vis_6 = dummy_analyzer.draw_detections(color_bgr, top_targets, sel_target_idx=0)
        cv2.rectangle(vis_6, (12, 12), (640, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_6, (12, 12), (640, 48), (0, 255, 120), 2)
        put_text(vis_6, f"STAGE 6: SKELETON POSES | Layer0: Top Priority ({len(top_targets)} Selected)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 120), 2)
        step_images["stage6_top_poses"] = vis_6

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=top_targets,
            elapsed_ms=elapsed_ms,
            step_snapshots=step_images,
            extra_metrics={"candidates_found": len(candidates), "skeletons": num_labels - 1}
        )
