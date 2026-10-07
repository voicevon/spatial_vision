#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶 (AprilTag 16h5) 空间地图建图与全局平差求解工具 (Tag Map Builder)
- 基于多视角重叠图像构建标靶共视连通图 (Co-visibility Graph)
- 基于 Bundle Adjustment (BA) 联合优化静止标靶 3D 空间位姿与相机位姿
- 支持 Tag 0 (SCARA J1 旋转中心原点) 与 Tag 1 (世界 +X 轴基准) 的刚体对齐闭环
- 导出 config/tags_map.yaml 供运行时毫秒级在线相机定位
"""

import os
import sys
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any, Set

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
try:
    from src.utils.config_guard import resolve_camera_intrinsics, load_raw_config
except ImportError:
    resolve_camera_intrinsics = None
    load_raw_config = None

from src.calibration.solvers.covisibility_graph import (
    CovisibilityGraphAnalyzer,
)
from src.calibration.manifest_repository import ManifestRepository
from src.calibration.solvers.ba_optimizer import BundleAdjustmentOptimizer
from src.vision.tag_detector import TagDetector
from src.calibration.verification.prism_renderer import draw_prism, COLORS_MAPPING
from src.utils.text_rendering import measure_text, put_text
from src.utils.logger import get_logger

log = get_logger(__name__)


class TagMapBuilder:
    def __init__(self,
                 marker_size_mm: float,
                 tag_family: int = cv2.aruco.DICT_APRILTAG_16h5,
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None):
        """
        初始化建图求解器
        :param marker_size_mm: 标靶黑白边框物理边长 (毫米)
        :param tag_family: OpenCV ArUco 字典枚举 (默认 AprilTag 16h5)
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数
        """
        self.marker_size_mm = float(marker_size_mm)
        self.dictionary = cv2.aruco.getPredefinedDictionary(tag_family)

        # 读取 config.yaml 的最小周长门限 (统一走 config_guard, 仅 tag_detection.min_perimeter_rate 是全局图像处理参数)
        min_perim = 0.006
        c = load_raw_config() if load_raw_config else {}
        min_perim = float(c.get("calibration", {}).get("tag_detection", {}).get("min_perimeter_rate", min_perim))

        # 工位物理白名单 (恒启用): allowed_ids 非空 → 权威, 名单内容即行为 (Tag ID 已 100% 下沉至工位沙盒)
        valid_tag_ids: List[int] = []
        try:
            from src.workspace.workspace_manager import WorkspaceManager, load_workspace_tag_whitelist
            wl = load_workspace_tag_whitelist(WorkspaceManager().get_current_workspace().workspace_dir)
            if wl:
                log.info(f"[BUILDER] 工位白名单已生效: {wl}")
                valid_tag_ids = wl
            else:
                log.info("[BUILDER] 工位白名单为空: 全 ID 自由通行 (由 tag_anchors 显式锚定约束)")
        except Exception:
            log.info("[BUILDER] 工位上下文不可用 (单测/独立调用), 不强制白名单")

        self.valid_tag_ids = valid_tag_ids

        # 统一双路互补检测器 (高光路 C=5.5/0.55 + 暗部拉伸路 C=2.5/0.45, 含亚像素精修)
        self.tag_detector = TagDetector(
            valid_tag_ids=valid_tag_ids,
            min_perimeter_rate=max(0.008, min_perim)
        )
        self.detector_bright = self.tag_detector.detector_bright
        self.detector_dark = self.tag_detector.detector_dark
        self.refine_corners_subpix = self.tag_detector.refine_corners_subpix

        # 相机内参与畸变 (若未指定，优先从 config_guard 加载并自适应)
        if camera_matrix is None:
            if resolve_camera_intrinsics is not None:
                K, dist, _ = resolve_camera_intrinsics()
                self.camera_matrix = K
                self.dist_coeffs = dist
            else:
                self.camera_matrix = np.array([
                    [1363.68, 0.0, 971.19],
                    [0.0, 1361.19, 566.26],
                    [0.0, 0.0, 1.0]
                ], dtype=np.float64)
                self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)

        # 标靶局部坐标系下的 4 个角点物理坐标 (逆时针, Z=0)
        s = self.marker_size_mm / 2.0
        self.obj_points = np.array([
            [-s,  s, 0.0],
            [ s,  s, 0.0],
            [ s, -s, 0.0],
            [-s, -s, 0.0]
        ], dtype=np.float64)

        # 清单与地图仓储管理器
        self.repository = ManifestRepository(builder=self)

        # 专业 BA 平差优化求解器
        self.ba_optimizer = BundleAdjustmentOptimizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm,
            obj_points=self.obj_points,
            builder=self
        )

    # 亚像素精修由实例属性 self.refine_corners_subpix 委托统一 TagDetector
    # (manifest_repository 经 hasattr 探测调用, 外部行为不变)

    def compute_observation_weight(self, corners: np.ndarray, 
                                   rvec: Optional[np.ndarray] = None, 
                                   tvec: Optional[np.ndarray] = None) -> float:
        """
        计算单次观测标靶的信息矩阵综合权重 w in [0.0, 1.0]：
        1. 像面有效像素面积权重 (远景小标靶像素信噪比低，适度衰减)
        2. 正对入射夹角权重 (大倾角透视掠射视角不确定度剧增，强力衰减)
        3. 径向畸变边缘衰减权重 (图像四周边缘相差区域适度衰减)
        4. 极小噪点/虚警直接置 0 (拒绝面积 < 100px^2 或 Z > 2500mm 的离群杂波)
        """
        pts = corners.reshape((4, 2)).astype(np.float64)
        area = float(cv2.contourArea(pts.astype(np.float32)))
        
        # 守门员：面积微小严重退化
        if area < 100.0:
            return 0.0

        if rvec is None or tvec is None:
            succ, rvec, tvec = self.solve_single_tag_pnp(pts)
            if not succ:
                return 0.1

        z = float(tvec[2][0])
        # 守门员：单靶解算深度异常（超过 2.5 米或小于 150mm）
        if z > 2500.0 or z < 150.0:
            return 0.0

        # 1. 面积权重 (以 3600 px^2 约 60x60px 为标准)
        w_area = float(np.clip(area / 3600.0, 0.25, 1.0))

        # 2. 正对夹角权重 (法向量与光轴夹角)
        R, _ = cv2.Rodrigues(rvec)
        # 标靶法向量在相机系下的朝向为 R[:, 2]
        # 当正对相机时，其 Z 分量朝向相机光心即 R[2, 2] 接近 -1
        cos_theta = abs(float(R[2, 2]))
        if cos_theta >= 0.85:
            w_angle = 1.0
        elif cos_theta >= 0.50:
            w_angle = 0.5 + 0.5 * (cos_theta - 0.50) / 0.35
        else:
            w_angle = max(0.15, cos_theta / 0.50 * 0.5)

        # 3. 径向边缘衰减 (根据中心距离)
        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        center = np.mean(pts, axis=0)
        dist_from_center = np.linalg.norm(center - np.array([cx, cy]))
        max_radius = np.sqrt(cx**2 + cy**2)
        r_norm = dist_from_center / max_radius
        if r_norm <= 0.65:
            w_radial = 1.0
        else:
            w_radial = float(np.clip(1.0 - (r_norm - 0.65) * 1.2, 0.4, 1.0))

        return float(np.clip(w_area * w_angle * w_radial, 0.1, 1.0))

    def solve_single_tag_pnp(self, corners: np.ndarray,
                             expected_z_cam: Optional[np.ndarray] = None) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        对单标靶执行 PnP 获得其在相机系下的位姿 (rvec, tvec)
        集成 IPPE_SQUARE 翻转二义性智能消歧 (Planar Ambiguity Disambiguation):
        当由于图像噪点导致对称翻转的伪解重投影误差极小时，
        基于物理几何先验（相机俯视拍摄工作台，标靶法向量 Z 轴必须向上立起即 R[1, 2] < 0）
        精准筛选物理真实解，彻底杜绝“Z 轴倒栽葱”或“反向刺入工作台”。

        expected_z_cam: 标靶法向 (Z 轴) 在相机系下的显式先验方向 (与引擎版同接口)。
        提供后按法向同半球 (dot>0) 过滤; 未提供时使用内置"朝天"先验 (R[1, 2] < 0)。
        先验下无同向合格解时拒绝输出 (防错优先, 宁缺勿反)。
        """
        corners_2d = corners.reshape((4, 2)).astype(np.float64)

        def _prior_ok(R_c: np.ndarray) -> bool:
            """候选解法向是否与先验同向: 显式先验用 dot>0, 内置先验用 R[1,2]<0 (朝天)"""
            if expected_z_cam is not None:
                z_exp = np.asarray(expected_z_cam, dtype=np.float64).reshape(3)
                n = float(np.linalg.norm(z_exp))
                if n <= 1e-9:
                    return True
                return float(R_c[:, 2] @ (z_exp / n)) > 0.0
            return bool(R_c[1, 2] < 0.0)

        retval, rvecs, tvecs, reprojErrors = cv2.solvePnPGeneric(
            self.obj_points,
            corners_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if retval and len(rvecs) > 0:
            best_idx, best_err = -1, float("inf")
            prior_idx, prior_err = -1, float("inf")
            for k, (r_k, t_k) in enumerate(zip(rvecs, tvecs)):
                if float(t_k[2, 0]) <= 0:      # 深度非法 (相机后方), 直接剔除
                    continue
                err_k = float(reprojErrors[k][0]) if reprojErrors is not None else 0.0
                if err_k < best_err:
                    best_err, best_idx = err_k, k
                if _prior_ok(cv2.Rodrigues(r_k)[0]) and err_k < prior_err:
                    prior_err, prior_idx = err_k, k
            if prior_idx >= 0:
                return True, rvecs[prior_idx], tvecs[prior_idx]
            if expected_z_cam is None and best_idx >= 0:
                # 无显式先验且无朝天合格解: 退回纯误差择优 (保持旧行为)
                return True, rvecs[best_idx], tvecs[best_idx]
            # 有显式先验但无同向解: 落入下方兜底 (兜底同样做先验校验)

        # 降级尝试 ITERATIVE
        success, rvec, tvec = cv2.solvePnP(
            self.obj_points,
            corners_2d,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if success and float(tvec[2, 0]) > 0:
            if _prior_ok(cv2.Rodrigues(rvec)[0]):
                return True, rvec, tvec
            return False, None, None           # 与先验反向的翻转解, 拒绝输出
        return False, None, None

    def render_tag_3d_axes(self, img: np.ndarray, corners: np.ndarray, tag_id: int):
        """
        在图像上绘制标靶 3D 空间坐标系与实心正四棱柱：
        - X 轴 (红色, 25mm), Y 轴 (绿色, 25mm)
        - Z 轴指示: 边长 20.0mm x 20.0mm (原始标靶 1/2)、长 120.0mm 的实心正四棱柱 (半透明实心柱体 + 12条高亮棱线 + 顶盖透视截面)
        """
        try:
            corners_2d = corners.reshape((4, 2)).astype(np.float64)
            ok, rvec, tvec = self.solve_single_tag_pnp(corners_2d)
            if not ok:
                return

            # 统一 PrismRenderer: 实心正四棱柱 (半透明 + 12 棱线 + 顶盖) + XYZ 坐标轴
            draw_prism(img, self.camera_matrix, self.dist_coeffs, rvec, tvec,
                       half_w=15.0, height=80.0, colors=COLORS_MAPPING,
                       alpha=0.42, draw_axes=True, axis_len=25.0)
        except Exception as e:
            log.warning(f"[Builder] 3D 坐标轴棱柱绘制失败 (已跳过): {e}")

    @staticmethod
    def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """旋转向量与平移向量转 4x4 齐次矩阵"""
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return T

    @staticmethod
    def matrix_to_rvec_tvec(T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """4x4 齐次矩阵转旋转向量与平移向量"""
        rvec, _ = cv2.Rodrigues(T[:3, :3])
        tvec = T[:3, 3].reshape((3, 1))
        return rvec, tvec

    def compute_tag_metrics(self, corners: np.ndarray) -> Dict[str, Any]:
        """
        根据标靶 4 个角点物理投影计算空间像元分辨率及几何指标
        """
        pts = corners.reshape((4, 2)).astype(np.float64)
        l01 = float(np.linalg.norm(pts[1] - pts[0]))
        l12 = float(np.linalg.norm(pts[2] - pts[1]))
        l23 = float(np.linalg.norm(pts[3] - pts[2]))
        l30 = float(np.linalg.norm(pts[0] - pts[3]))
        cell_w = int(round((l01 + l23) / 12.0))
        cell_h = int(round((l30 + l12) / 12.0))
        center_x = float(np.mean(pts[:, 0]))
        center_y = float(np.mean(pts[:, 1]))
        area = float(cv2.contourArea(pts.astype(np.float32)))
        return {
            "cell_size_px": [cell_w, cell_h],
            "center_px": [round(center_x, 1), round(center_y, 1)],
            "area_px": round(area, 1),
            "edge_lengths_px": [round(l01, 1), round(l12, 1), round(l23, 1), round(l30, 1)]
        }

    def render_annotated_frame(self, image: np.ndarray, detected_tags: Dict[int, np.ndarray]) -> np.ndarray:
        """
        渲染完整图示化分析图像 (带绿色轮廓线、彩色角点、像元分辨率标牌与 3D 正四棱柱)
        """
        disp = image.copy()
        dot_colors = [
            (0, 0, 255),    # 角点 0: 红色
            (0, 255, 0),    # 角点 1: 绿色
            (255, 0, 0),    # 角点 2: 蓝色
            (0, 255, 255)   # 角点 3: 黄色
        ]
        for tag_id, corners in detected_tags.items():
            pts = corners.reshape((4, 2)).astype(np.int32)
            # 绘制绿色轮廓线
            cv2.polylines(disp, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
            # 绘制 4 个角点彩色圆点
            for pt_idx, pt in enumerate(pts):
                cv2.circle(disp, tuple(pt), 5, dot_colors[pt_idx], -1)

            # 计算机械标靶单元方格像素尺寸 (AprilTag 16h5 为 6x6 网格)
            metrics = self.compute_tag_metrics(corners)
            cell_w, cell_h = metrics["cell_size_px"]

            # 绘制 ID 与最小单元方格像素分辨率标牌
            cx = int(metrics["center_px"][0])
            min_y = int(np.min(pts[:, 1]))
            tag_text = f"Tag {tag_id}" + (" [ORIGIN]" if tag_id == 0 else "")
            cell_text = f"Cell: {cell_w}x{cell_h}px"
            
            badge_x = cx - 50
            badge_y = max(42, min_y - 12)
            cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 106, badge_y + 8), (20, 20, 20), -1)
            cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + 106, badge_y + 8), (0, 255, 255), 1)
            put_text(disp, tag_text, (badge_x, badge_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            put_text(disp, cell_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            # 绘制 3D 空间坐标系 (实心正四棱柱)
            self.render_tag_3d_axes(disp, corners, tag_id)

        return disp

    def optimize_bundle_adjustment(self, 
                                   frame_detections: List[Dict[int, np.ndarray]], 
                                   active_frame_names: Optional[List[str]] = None,
                                   origin_tag_id: int = 0,
                                   x_align_tag_id: int = 1,
                                   baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        基于非线性最小二乘 (Bundle Adjustment) 联合优化所有标靶位姿与相机位姿
        委托至独立专业求解器 BundleAdjustmentOptimizer 处理
        """
        self.ba_optimizer.camera_matrix = self.camera_matrix
        self.ba_optimizer.dist_coeffs = self.dist_coeffs
        self.ba_optimizer.marker_size_mm = self.marker_size_mm
        self.ba_optimizer.obj_points = self.obj_points

        result = self.ba_optimizer.optimize(
            frame_detections=frame_detections,
            active_frame_names=active_frame_names,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )
        self.marker_size_mm = self.ba_optimizer.marker_size_mm
        return result

    def build_map_from_manifest(self, 
                                manifest_path: Optional[str] = None,
                                origin_tag_id: int = 0,
                                x_align_tag_id: int = 1,
                                baseline_pair: Optional[Tuple[int, int, float]] = None) -> Dict[str, Any]:
        """
        从审核清单直接加载过滤后的观测数据并执行 BA 优化建图
        """
        frame_detections, valid_frames, stats = self.repository.load_manifest(manifest_path=manifest_path)
        log.info(f"[+] 从审核清单成功载入: {len(valid_frames)} 张有效图像，保留观测 {stats['total_kept']} 次，排除观测 {stats['total_excluded']} 次")
        if stats["total_excluded"] > 0:
            log.info(f"    【已人工剔除的坏样本】:")
            for exc in stats["excluded_items"]:
                note_str = f" (备注: {exc['note']})" if exc['note'] else ""
                log.info(f"      - [{exc['image']}] Tag #{exc['tag_id']}{note_str}")
        if stats["dropped_single_tag_frames"]:
            for d_name, d_cnt in stats["dropped_single_tag_frames"]:
                log.warning(f"图像 {d_name} 有效标靶少于 2 个 (实际={d_cnt})，已自动不参与相对刚体平差约束")

        return self.optimize_bundle_adjustment(
            frame_detections=frame_detections,
            active_frame_names=valid_frames,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )

    def build_map_from_images(self, 
                              image_paths: List[str],
                              origin_tag_id: int = 0,
                              x_align_tag_id: int = 1,
                              baseline_pair: Optional[Tuple[int, int, float]] = None,
                              manifest_path: Optional[str] = None,
                              use_manifest: bool = True) -> Dict[str, Any]:
        """
        从一组多视角图像构建标靶全局地图并进行 BA 全局平差优化
        自动联动审核清单流水线 (若清单不存在则先导出，若存在则使用清单的过滤规则)
        """
        log.info(f"[*] 开始处理 {len(image_paths)} 张多视角标定图像...")
        
        # 探测输入图像真实分辨率，执行内参动态自适应与防呆校验
        for path in image_paths:
            probe_img = cv2.imread(path)
            if probe_img is not None:
                if resolve_camera_intrinsics is not None:
                    self.camera_matrix, self.dist_coeffs, _ = resolve_camera_intrinsics(
                        actual_image_shape=probe_img.shape[:2]
                    )
                break

        if use_manifest:
            if not os.path.exists(manifest_path):
                log.info(f"[*] 未检测到观测清单，正在生成初始观测数据清单: {manifest_path} ...")
                self.repository.export_manifest(image_paths=image_paths, manifest_path=manifest_path)
            return self.build_map_from_manifest(
                manifest_path=manifest_path,
                origin_tag_id=origin_tag_id,
                x_align_tag_id=x_align_tag_id,
                baseline_pair=baseline_pair
            )

        # 直接全量检测（备用快速通道）
        frame_detections = []
        valid_frames = []
        for path in image_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            tags = self.tag_detector.detect_tags(img, refine=True)
            if len(tags) >= 2:
                frame_detections.append(tags)
                valid_frames.append(os.path.basename(path))

        return self.optimize_bundle_adjustment(
            frame_detections=frame_detections,
            active_frame_names=valid_frames,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            baseline_pair=baseline_pair
        )

    def apply_baseline_scale(self, 
                             tag_poses: Dict[int, np.ndarray],
                             tag_id_a: int, 
                             tag_id_b: int, 
                             real_distance_mm: float) -> Tuple[Dict[int, np.ndarray], float, float]:
        """利用双标靶距离锁定绝对尺度（委托专职求解器）"""
        self.ba_optimizer.marker_size_mm = self.marker_size_mm
        scaled_poses, scale_factor, real_marker_size = self.ba_optimizer.apply_baseline_scale(
            tag_poses=tag_poses,
            tag_id_a=tag_id_a,
            tag_id_b=tag_id_b,
            real_distance_mm=real_distance_mm
        )
        self.marker_size_mm = real_marker_size
        return scaled_poses, scale_factor, real_marker_size


    def save_map(self, map_data: Dict, output_path: str):
        """保存标靶地图至 YAML 文件（委托专职仓储处理, 必须显式传入路径 — 默认值已废弃）"""
        if not output_path:
            raise ValueError("save_map: output_path 不能为空 (Tag 地图必须写入工位沙盒, 不再回退全局 config/tags_map.yaml)")
        return self.repository.save_map(map_data=map_data, output_path=output_path)


def main():
    """CLI 入口（流程编排已拆分至 builder_workflow.py，纯几何与 BA 求解保留在本类内）"""
    from tools.calibration.builder_workflow import main as _workflow_main
    _workflow_main()


if __name__ == "__main__":
    main()
