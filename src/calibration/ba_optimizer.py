#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多标靶空间 Bundle Adjustment (BA) 全局平差优化求解器 (Bundle Adjustment Optimizer)
- 纯面向对象单一职责设计，专注于空间静止标靶位姿与多视角相机位姿的非线性联合平差优化
- 阶段一：基于 Cauchy 鲁棒核函数的粗差自动识别与清洗 (MAD 鲁棒离群统计)
- 阶段二：微容差极致深层收敛求解
- 标靶物理间距先验约束惩罚项 (Metric Baseline Gauge)
- 基于雅可比矩阵逆的一阶 3D 空间置信度 (Uncertainty Estimation) 分析
- 世界坐标系对齐闭环 (FR-9.6 绝对坐标锚定 / Origin 锚定与 X 轴水平对齐)
- 2D 像面 Quiver Plot 残差矢量场与 Markdown 诊断报告输出
"""

import os
import math
from typing import Dict, List, Tuple, Optional, Any, Set, Callable
import numpy as np
import cv2
from scipy.optimize import least_squares

from src.calibration.covisibility_graph import CovisibilityGraphAnalyzer, CovisibilityGraphError
from src.calibration.ba_report import compute_3d_uncertainties, export_diagnostic_report
from src.calibration.world_datum_aligner import WorldDatumAligner

from src.utils.logger import get_logger

log = get_logger(__name__)

# 两阶段 BA 最小二乘统一收敛容差 (ftol/xtol/gtol 三项同值)
_BA_CONVERGE_TOL = 1e-5

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


class BundleAdjustmentOptimizer:
    """
    BA 联合平差优化求解器
    """

    def __init__(self, 
                 camera_matrix: np.ndarray,
                 dist_coeffs: np.ndarray,
                 marker_size_mm: float,
                 obj_points: Optional[np.ndarray] = None,
                 builder: Optional[Any] = None):
        """
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数
        :param marker_size_mm: 标靶物理边长 (mm)
        :param obj_points: 标靶局部坐标系 4 角点物理坐标 (4, 3)
        :param builder: 宿主 TagMapBuilder 实例（用于辅助观测加权等接口）
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
        self.marker_size_mm = float(marker_size_mm)
        self.builder = builder

        if obj_points is not None:
            self.obj_points = np.array(obj_points, dtype=np.float64)
        else:
            s = self.marker_size_mm / 2.0
            self.obj_points = np.array([
                [-s,  s, 0.0],
                [ s,  s, 0.0],
                [ s, -s, 0.0],
                [-s, -s, 0.0]
            ], dtype=np.float64)

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

    def compute_observation_weight(self, corners: np.ndarray) -> float:
        """计算角点观测权重 (如果宿主 builder 有实现则委托，否则使用内置几何加权)"""
        if self.builder is not None and hasattr(self.builder, "compute_observation_weight"):
            return float(self.builder.compute_observation_weight(corners))

        pts = corners.reshape((4, 2)).astype(np.float64)
        area = float(cv2.contourArea(pts.astype(np.float32)))
        w_area = float(np.clip(area / 1200.0, 0.2, 1.0))

        cx = self.camera_matrix[0, 2]
        cy = self.camera_matrix[1, 2]
        center = np.mean(pts, axis=0)
        dist_from_center = np.linalg.norm(center - np.array([cx, cy]))
        max_radius = np.sqrt(cx**2 + cy**2)
        r_norm = dist_from_center / max_radius
        w_radial = 1.0 if r_norm <= 0.65 else float(np.clip(1.0 - (r_norm - 0.65) * 1.2, 0.4, 1.0))

        return float(np.clip(w_area * w_radial, 0.1, 1.0))

    def optimize(self, 
                 frame_detections: List[Dict[int, np.ndarray]], 
                 active_frame_names: Optional[List[str]] = None,
                 origin_tag_id: int = 0,
                 x_align_tag_id: int = 1,
                 baseline_pair: Optional[Tuple[int, int, float]] = None,
                 anchor_tags: Optional[Any] = None,
                 callback: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """
        基于非线性最小二乘 (Bundle Adjustment) 联合优化所有标靶位姿与相机位姿
        平差前严格调用共视连通性安全守门员校验，杜绝奇异矩阵。
        """
        if active_frame_names is None:
            active_frame_names = [f"frame_{i:04d}" for i in range(len(frame_detections))]

        # 1. 守门员审查
        report = CovisibilityGraphAnalyzer.analyze(frame_detections, active_frame_names, origin_tag_id, x_align_tag_id)
        if not report["is_valid"]:
            raise CovisibilityGraphError(report["message"])

        all_detected_tags = set(report["all_tags"])
        if report["critical_bridges"]:
            log.info(f"[NOTE] 提示：发现 {len(report['critical_bridges'])} 对标靶仅由单张图共视支撑 (关键桥梁): {report['critical_bridges']}")

        # 2. 生成高质量初值 (多标靶联合超定 PnP 初值传递，杜绝单链累积误差与翻转)
        if origin_tag_id in all_detected_tags:
            base_static_id = origin_tag_id
        elif x_align_tag_id in all_detected_tags:
            base_static_id = x_align_tag_id
        else:
            base_static_id = min(all_detected_tags)
        tag_poses_init = {base_static_id: np.eye(4, dtype=np.float64)}
        camera_poses_init = {}  # {frame_idx: T_w_cam}

        changed = True
        while changed:
            changed = False
            # 步骤 2.1: 优先利用该帧可见的所有已知标靶进行联合超定 PnP 求解相机位姿
            for f_idx, tags in enumerate(frame_detections):
                if f_idx not in camera_poses_init:
                    obj_pts_list = []
                    img_pts_list = []
                    for t_id, corners in tags.items():
                        if t_id in tag_poses_init:
                            T_w_t = tag_poses_init[t_id]
                            for pt3d in self.obj_points:
                                pt_h = np.append(pt3d, 1.0)
                                w_pt = (T_w_t @ pt_h)[:3]
                                obj_pts_list.append(w_pt)
                            img_pts_list.extend(corners.reshape(4, 2))

                    if len(obj_pts_list) >= 4:
                        obj_pts_arr = np.array(obj_pts_list, dtype=np.float64)
                        img_pts_arr = np.array(img_pts_list, dtype=np.float64)
                        succ, rvec, tvec = cv2.solvePnP(
                            obj_pts_arr, img_pts_arr, 
                            self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_ITERATIVE
                        )
                        if succ:
                            T_c_w = self.rvec_tvec_to_matrix(rvec, tvec)
                            camera_poses_init[f_idx] = np.linalg.inv(T_c_w)
                            changed = True

            # 步骤 2.2: 利用已定位的相机推算未知标靶在世界系下的绝对位姿
            for f_idx in list(camera_poses_init.keys()):
                T_w_c = camera_poses_init[f_idx]
                T_c_w = np.linalg.inv(T_w_c)
                tags = frame_detections[f_idx]
                for t_id, corners in tags.items():
                    if t_id not in tag_poses_init:
                        corners_2d = corners.reshape(4, 2).astype(np.float64)
                        succ, rvec, tvec = cv2.solvePnP(
                            self.obj_points, corners_2d,
                            self.camera_matrix, self.dist_coeffs,
                            flags=cv2.SOLVEPNP_IPPE_SQUARE
                        )
                        if succ:
                            T_c_t = self.rvec_tvec_to_matrix(rvec, tvec)
                            T_w_t = T_w_c @ T_c_t
                            tag_poses_init[t_id] = T_w_t
                            changed = True

        missing_tags = all_detected_tags - set(tag_poses_init.keys())
        if missing_tags:
            raise RuntimeError(f"以下标靶未能完成初值初始化: {missing_tags}")

        active_frames = sorted(list(camera_poses_init.keys()))
        log.info(f"[+] 初值推导完成: 成功初始化 {len(tag_poses_init)} 个标靶位姿，{len(active_frames)} 个采图机位位姿")

        # 3. 计算每个观测点的初始权重 (观测加权)
        obs_weights = {}
        for f in active_frames:
            tags = frame_detections[f]
            for t_id, corners in tags.items():
                w = self.compute_observation_weight(corners)
                obs_weights[(f, t_id)] = w

        # 4. 构建两阶段非线性优化变量 (固定 base_static_id 位姿作为 Gauge Freedom 锚点)
        static_tags_to_opt = sorted([t for t in tag_poses_init.keys() if t != base_static_id])

        def pack_params(tags_dict, cams_dict):
            params = []
            for t in static_tags_to_opt:
                rv, tv = self.matrix_to_rvec_tvec(tags_dict[t])
                params.extend(rv.flatten())
                params.extend(tv.flatten())
            for f in active_frames:
                rv, tv = self.matrix_to_rvec_tvec(cams_dict[f])
                params.extend(rv.flatten())
                params.extend(tv.flatten())
            return np.array(params, dtype=np.float64)

        def unpack_params(x):
            tags_pose = {base_static_id: tag_poses_init[base_static_id].copy()}
            cams_pose = {}
            offset = 0
            for t in static_tags_to_opt:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                tags_pose[t] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6
            for f in active_frames:
                rv = x[offset:offset + 3]
                tv = x[offset + 3:offset + 6]
                cams_pose[f] = self.rvec_tvec_to_matrix(rv, tv)
                offset += 6
            return tags_pose, cams_pose

        def residuals_func(x, weights_dict, active_outliers=None):
            tags_pose, cams_pose = unpack_params(x)
            residuals = []
            for f in active_frames:
                T_w_c = cams_pose[f]
                T_c_w = np.linalg.inv(T_w_c)
                tags = frame_detections[f]
                for t_id, corners_img in tags.items():
                    if t_id in tags_pose:
                        w = weights_dict.get((f, t_id), 1.0)
                        if active_outliers and (f, t_id) in active_outliers:
                            w *= 0.01  # 离群项强力压制

                        T_w_t = tags_pose[t_id]
                        T_c_t = T_c_w @ T_w_t
                        rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                        proj_pts, _ = cv2.projectPoints(
                            self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs
                        )
                        proj_pts = proj_pts.reshape((4, 2))
                        diff = (proj_pts - corners_img).flatten()
                        residuals.extend(diff * np.sqrt(max(1e-4, w)))

            # 若配置了物理标靶间距先验约束，作为硬约束惩罚项联合求解
            if baseline_pair is not None:
                id_a, id_b, real_dist_mm = baseline_pair
                if id_a in tags_pose and id_b in tags_pose:
                    t_a = tags_pose[id_a][:3, 3]
                    t_b = tags_pose[id_b][:3, 3]
                    dist_est = float(np.linalg.norm(t_a - t_b))
                    residuals.append((dist_est - real_dist_mm) * 5.0)

            return np.array(residuals, dtype=np.float64)

        # 内部轻量迭代步进与残差监控器
        class _OptimizationMonitor:
            def __init__(self, stage: int, stage_name: str, max_iters: int, cb=None):
                self.stage = stage
                self.stage_name = stage_name
                self.max_iters = max(max_iters, 50)
                self.cb = cb
                self.call_count = 0
                self.iter_count = 0
                self.last_x = None

            def wrap_residuals(self, base_func, weights_dict, active_outliers):
                def _wrapped(x):
                    self.call_count += 1
                    res = base_func(x, weights_dict, active_outliers)

                    # 判断是否为新的优化主步 (过滤雅可比差分时的微摄动)
                    is_new_step = False
                    if self.last_x is None:
                        is_new_step = True
                        self.last_x = x.copy()
                    else:
                        diff = np.linalg.norm(x - self.last_x)
                        if diff > 1e-4:
                            is_new_step = True
                            self.last_x = x.copy()

                    if is_new_step:
                        self.iter_count += 1
                        # 动态自适应调整最大轮次：分母永不小于分子，若超过预设则自适应平滑扩充
                        if self.iter_count > self.max_iters:
                            import math
                            self.max_iters = int(math.ceil(self.iter_count / 10.0) * 10)

                        rmse = float(np.sqrt(np.mean(res ** 2))) if len(res) > 0 else 0.0
                        # 迭代运行中进度条最高逼近 95%，收敛完成时由回调置 100%
                        sub_pct = min(0.95, self.iter_count / float(max(1, self.max_iters)))
                        if self.cb:
                            try:
                                self.cb({
                                    "stage": self.stage,
                                    "stage_name": self.stage_name,
                                    "iter": self.iter_count,
                                    "max_iter": self.max_iters,
                                    "rmse": rmse,
                                    "sub_progress": sub_pct,
                                    "call_count": self.call_count
                                })
                            except Exception as e:
                                log.warning(f"[BA] 优化进度回调异常 (已忽略): {e}")
                    return res
                return _wrapped

        x0 = pack_params(tag_poses_init, camera_poses_init)

        stage1_est_max = 60
        if callback:
            callback({
                "stage": 1,
                "stage_name": "粗差清洗与收敛",
                "iter": 0,
                "max_iter": stage1_est_max,
                "rmse": 1.0,
                "sub_progress": 0.0,
                "call_count": 0
            })

        log.info("[*] 正在执行 Phase 1 阶段一：基于 Cauchy 鲁棒核的粗差清洗与全局收敛...")
        monitor1 = _OptimizationMonitor(stage=1, stage_name="粗差清洗收敛", max_iters=stage1_est_max, cb=callback)
        res_stage1 = least_squares(
            monitor1.wrap_residuals(residuals_func, obs_weights, None), x0,
            method='trf',
            loss='cauchy',
            f_scale=1.5,
            x_scale='jac',
            ftol=_BA_CONVERGE_TOL,
            xtol=_BA_CONVERGE_TOL,
            gtol=_BA_CONVERGE_TOL,
            max_nfev=200,
            verbose=0
        )

        final_iter1 = max(1, monitor1.iter_count)
        final_max1 = max(monitor1.max_iters, final_iter1)
        if callback:
            callback({
                "stage": 1,
                "stage_name": "粗差清洗收敛",
                "iter": final_iter1,
                "max_iter": final_max1,
                "rmse": float(np.sqrt(np.mean(res_stage1.fun ** 2))) if len(res_stage1.fun) > 0 else 0.0,
                "sub_progress": 1.0,
                "call_count": monitor1.call_count
            })

        # 统计阶段一未加权像元残差，自动识别标准化残差 > 3.0 sigma 的粗差观测
        tags_p1, cams_p1 = unpack_params(res_stage1.x)
        raw_errors = []
        obs_map = []
        for f in active_frames:
            T_c_w = np.linalg.inv(cams_p1[f])
            for t_id, corners_img in frame_detections[f].items():
                if t_id in tags_p1:
                    T_c_t = T_c_w @ tags_p1[t_id]
                    rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                    proj, _ = cv2.projectPoints(self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs)
                    e = float(np.mean(np.linalg.norm(proj.reshape(4, 2) - corners_img, axis=1)))
                    raw_errors.append(e)
                    obs_map.append((f, t_id, e))

        med_e = float(np.median(raw_errors))
        mad_e = float(np.median(np.abs(np.array(raw_errors) - med_e)))
        sigma_robust = max(0.5, 1.4826 * mad_e)
        outlier_thresh = max(4.0, med_e + 2.8 * sigma_robust)
        outliers_detected = set()
        for f, t_id, e in obs_map:
            if e > outlier_thresh or obs_weights.get((f, t_id), 1.0) <= 0.05:
                outliers_detected.add((f, t_id))

        if outliers_detected:
            log.info(f"[CLEAN] 自动清洗识别出 {len(outliers_detected)} 个潜在粗差/远景噪点观测 (MAD 门限 > {outlier_thresh:.2f}px, 中位数={med_e:.2f}px):")
            for f, t_id in sorted(list(outliers_detected)):
                f_name = active_frame_names[f] if f < len(active_frame_names) else f"frame_{f}"
                log.info(f"        - [{f_name}] Tag #{t_id}")

        max_iters_p2 = 60
        if callback:
            callback({
                "stage": 2,
                "stage_name": "微容差深度平差",
                "iter": 0,
                "max_iter": max_iters_p2,
                "rmse": float(np.sqrt(np.mean(res_stage1.fun ** 2))) if len(res_stage1.fun) > 0 else 0.0,
                "sub_progress": 0.0,
                "call_count": 0
            })

        log.info("[*] 正在执行 Phase 1 阶段二：微容差 (ftol=1e-5) 极致深层平差收敛...")
        monitor2 = _OptimizationMonitor(stage=2, stage_name="微容差深度平差", max_iters=max_iters_p2, cb=callback)
        res_stage2 = least_squares(
            monitor2.wrap_residuals(residuals_func, obs_weights, outliers_detected), res_stage1.x,
            method='trf',
            loss='cauchy',
            f_scale=1.0,
            x_scale='jac',
            ftol=_BA_CONVERGE_TOL,
            xtol=_BA_CONVERGE_TOL,
            gtol=_BA_CONVERGE_TOL,
            max_nfev=200,
            verbose=0
        )

        final_iter2 = max(1, monitor2.iter_count)
        final_max2 = max(monitor2.max_iters, final_iter2)
        if callback:
            callback({
                "stage": 2,
                "stage_name": "微容差深度平差",
                "iter": final_iter2,
                "max_iter": final_max2,
                "rmse": float(np.sqrt(np.mean(res_stage2.fun ** 2))) if len(res_stage2.fun) > 0 else 0.0,
                "sub_progress": 1.0,
                "call_count": monitor2.call_count
            })

        optimized_tags_pose, optimized_cams_pose = unpack_params(res_stage2.x)

        clean_residuals = []
        detailed_obs_res = []
        for f in active_frames:
            T_c_w = np.linalg.inv(optimized_cams_pose[f])
            for t_id, corners_img in frame_detections[f].items():
                if t_id in optimized_tags_pose:
                    T_c_t = T_c_w @ optimized_tags_pose[t_id]
                    rv, tv = self.matrix_to_rvec_tvec(T_c_t)
                    proj, _ = cv2.projectPoints(self.obj_points, rv, tv, self.camera_matrix, self.dist_coeffs)
                    proj_2d = proj.reshape(4, 2)
                    diff = proj_2d - corners_img
                    e_pt = np.linalg.norm(diff, axis=1)
                    detailed_obs_res.append({
                        "frame_idx": f,
                        "frame_name": active_frame_names[f] if f < len(active_frame_names) else f"frame_{f}",
                        "tag_id": t_id,
                        "corners_obs": corners_img,
                        "corners_proj": proj_2d,
                        "rmse_px": float(np.sqrt(np.mean(e_pt ** 2))),
                        "is_outlier": (f, t_id) in outliers_detected
                    })
                    if (f, t_id) not in outliers_detected:
                        clean_residuals.extend(diff.flatten())

        rmse_px = float(np.sqrt(np.mean(np.array(clean_residuals) ** 2)))
        log.info(f"[OK] 两阶段 BA 极限优化完成！有效观测像面 RMSE: {rmse_px:.3f} 像素 (迭代次数: {res_stage2.nfev})")

        # 5. 计算 3D 标靶空间坐标一阶协方差置信区间 (Uncertainty Estimation)
        tag_uncertainties = self.compute_3d_uncertainties(
            res_stage2.jac, static_tags_to_opt, base_static_id, rmse_px
        )

        # 6. 双标靶中心基线测距尺度修正 (Metric Baseline Gauge)
        scale_factor = 1.0
        real_marker_size = self.marker_size_mm
        baseline_info = None

        if baseline_pair is not None:
            id_a, id_b, real_dist_mm = baseline_pair
            optimized_tags_pose, scale_factor, real_marker_size = self.apply_baseline_scale(
                optimized_tags_pose, id_a, id_b, real_dist_mm
            )
            baseline_info = {
                "tag_a": int(id_a),
                "tag_b": int(id_b),
                "measured_dist_mm": float(real_dist_mm),
                "scale_factor": round(float(scale_factor), 6)
            }
            self.marker_size_mm = real_marker_size

        # 7. 坐标系解耦逻辑:
        #    - 若未指定 anchor_tags: 纯自由平差 (阶段一), 输出纯视觉相对几何地图 (以 base_static_id 为相对原点)
        #    - 若指定了 anchor_tags: 世界绝对锚定 (阶段二), 求解 3D 相似变换变换至世界系
        if anchor_tags:
            final_tags_map = self.anchor_to_absolute_world(
                optimized_tags_pose, anchor_tags,
                origin_tag_id=origin_tag_id, x_align_tag_id=x_align_tag_id,
                strict=True
            )
        else:
            # 阶段一: 纯自由平差相对地图
            tags_dict = {}
            for tid, T in optimized_tags_pose.items():
                pos = T[:3, 3]
                roll, pitch, yaw = self._rotation_to_rpy_deg(T[:3, :3])
                tags_dict[tid] = {
                    "position_mm": [round(float(v), 2) for v in pos],
                    "rpy_deg": [round(float(math.degrees(v)), 2) for v in [roll, pitch, yaw]],
                    "transform_matrix": [[round(float(val), 5) for val in row] for row in T],
                    "is_origin": bool(tid == base_static_id),
                    "is_dynamic_yaw": bool(tid == base_static_id)
                }
            final_tags_map = {
                "origin_tag_id": base_static_id,
                "x_axis_align_tag_id": x_align_tag_id,
                "anchor_mode": "unaligned",
                "tags": tags_dict
            }
            log.info(f"[+] [FREE_BA] 阶段一纯视觉自由平差完成: 基准标靶 Tag #{base_static_id}, 相对构型已固化 (尚未校准世界系)")

        # 始终保存一份未经世界变换的纯相对矩阵 (供后续随时独立做世界系校准)
        final_tags_map["raw_relative_poses"] = {
            int(tid): [[round(float(val), 5) for val in row] for row in T]
            for tid, T in optimized_tags_pose.items()
        }

        final_tags_map["rmse_reprojection_px"] = rmse_px
        final_tags_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        final_tags_map["tag_family"] = "DICT_APRILTAG_16h5"
        final_tags_map["calibrated_images_count"] = len(active_frames)
        final_tags_map["cleaned_outliers_count"] = len(outliers_detected)
        final_tags_map["final_rmse"] = rmse_px
        final_tags_map["final_tag_poses_aligned"] = {
            tid: np.array(t_info["transform_matrix"], dtype=np.float64)
            for tid, t_info in final_tags_map.get("tags", {}).items()
        }
        if baseline_info:
            final_tags_map["baseline_gauge"] = baseline_info
        if tag_uncertainties:
            final_tags_map["uncertainties_mm"] = tag_uncertainties

        # 8. 生成 2D 像面 Quiver Plot 残差矢量场与详细 Markdown 诊断报告
        try:
            self.export_diagnostic_report(
                final_tags_map=final_tags_map,
                detailed_obs_res=detailed_obs_res,
                active_frame_names=active_frame_names,
                tag_uncertainties=tag_uncertainties,
                outliers_detected=outliers_detected,
                rmse_px=rmse_px
            )
        except Exception as e:
            log.warning(f"[WARN] 导出深度诊断报告异常 (已安全忽略): {e}")

        return final_tags_map

    def compute_3d_uncertainties(self, jacobian, static_tags, base_id, sigma_res_px) -> Dict[int, Dict[str, float]]:
        """标靶 3D 置信区间计算（委托 ba_report 纯函数实现）"""
        return compute_3d_uncertainties(jacobian, static_tags, base_id, sigma_res_px)

    def export_diagnostic_report(self,
                                 final_tags_map: Dict[str, Any],
                                 detailed_obs_res: List[Dict[str, Any]],
                                 active_frame_names: List[str],
                                 tag_uncertainties: Dict[int, Dict[str, float]],
                                 outliers_detected: Set[Tuple[int, int]],
                                 rmse_px: float,
                                 report_dir: Optional[str] = None) -> str:
        """Quiver 残差矢量场与 Markdown 精度体检报告生成（委托 ba_report 纯函数实现）"""
        return export_diagnostic_report(
            final_tags_map=final_tags_map,
            detailed_obs_res=detailed_obs_res,
            active_frame_names=active_frame_names,
            tag_uncertainties=tag_uncertainties,
            outliers_detected=outliers_detected,
            rmse_px=rmse_px,
            report_dir=report_dir
        )

    def apply_baseline_scale(self, 
                             tag_poses: Dict[int, np.ndarray],
                             tag_id_a: int, 
                             tag_id_b: int, 
                             real_distance_mm: float) -> Tuple[Dict[int, np.ndarray], float, float]:
        """
        利用两个标靶中心物理测量距离锁定绝对尺度 (Metric Baseline Gauge)
        :param tag_poses: 各标靶 4x4 位姿矩阵字典
        :param tag_id_a: 标靶 A 的 ID
        :param tag_id_b: 标靶 B 的 ID
        :param real_distance_mm: 现场实际测量的中心物理直线距离 (mm)
        :return: (scaled_tag_poses, scale_factor, real_marker_size_mm)
        """
        if tag_id_a not in tag_poses or tag_id_b not in tag_poses:
            log.warning(f"[WARN] 尺度标定失败：标靶 {tag_id_a} 或 {tag_id_b} 未在重构地图中！保持名义尺度。")
            return tag_poses, 1.0, self.marker_size_mm

        p_a = tag_poses[tag_id_a][:3, 3]
        p_b = tag_poses[tag_id_b][:3, 3]
        nominal_dist = float(np.linalg.norm(p_a - p_b))

        if nominal_dist < 1e-4:
            log.warning(f"[WARN] 标靶 {tag_id_a} 与 {tag_id_b} 距离过近，无法用作尺度基线！")
            return tag_poses, 1.0, self.marker_size_mm

        scale_factor = float(real_distance_mm) / nominal_dist
        real_marker_size = self.marker_size_mm * scale_factor

        log.info(f"\n[+] ====== 双标靶中心基线绝对尺度校准 (Metric Baseline Gauge) ======")
        log.info(f"  -> 基准标靶对: Tag #{tag_id_a} <---> Tag #{tag_id_b}")
        log.info(f"  -> 当前名义欧氏距离: {nominal_dist:.2f} mm")
        log.info(f"  -> 现场测量实际距离: {real_distance_mm:.2f} mm")
        log.info(f"  -> 尺度修正系数 (Scale): {scale_factor:.6f}")
        log.info(f"  -> 反算单个 Tag 真实物理边长: {real_marker_size:.2f} mm (名义初值: {self.marker_size_mm:.2f} mm)")
        log.info(f"===================================================================\n")

        scaled_poses = {}
        for t_id, T in tag_poses.items():
            T_scaled = T.copy()
            T_scaled[:3, 3] = T[:3, 3] * scale_factor
            scaled_poses[t_id] = T_scaled

        return scaled_poses, scale_factor, real_marker_size

    @staticmethod
    def _rotation_to_rpy_deg(R: np.ndarray) -> Tuple[float, float, float]:
        """3x3 旋转矩阵 -> (roll, pitch, yaw) 弧度 (含万向锁奇异保护)"""
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:
            roll = math.atan2(R[2, 1], R[2, 2])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = math.atan2(R[1, 0], R[0, 0])
        else:
            roll = math.atan2(-R[1, 2], R[1, 1])
            pitch = math.atan2(-R[2, 0], sy)
            yaw = 0.0
        return roll, pitch, yaw

    # ------------------------------------------------------------------
    # 世界系绝对锚定与基准对齐 (统一委托给独立的 WorldDatumAligner 模块实现)
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_anchor_tags(anchor_input: Any) -> Optional[Dict[int, Dict[str, Any]]]:
        """锚点配置格式归一化 (委托 WorldDatumAligner 实现)"""
        return WorldDatumAligner.normalize_anchor_tags(anchor_input)

    @staticmethod
    def evaluate_anchor_dof(anchor_tags: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """锚点自由度记账 (委托 WorldDatumAligner 实现)"""
        return WorldDatumAligner.evaluate_anchor_dof(anchor_tags)

    @staticmethod
    def _umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """Umeyama 3D 相似变换 (委托 WorldDatumAligner 实现)"""
        return WorldDatumAligner.umeyama_alignment(src_pts, dst_pts)

    @classmethod
    def solve_similarity_from_anchors(cls,
                                      tag_poses: Dict[int, np.ndarray],
                                      anchor_tags: Dict[int, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """自适应求解相似变换 (委托 WorldDatumAligner 实现)"""
        return WorldDatumAligner.solve_similarity_from_anchors(tag_poses, anchor_tags)

    def anchor_to_absolute_world(self,
                                 tag_poses: Dict[int, np.ndarray],
                                 anchor_input: Any,
                                 origin_tag_id: int = 0,
                                 x_align_tag_id: int = 1,
                                 strict: bool = False) -> Dict[str, Any]:
        """FR-9.6 世界坐标系绝对锚定 (委托 WorldDatumAligner 实现)"""
        aligner = WorldDatumAligner(marker_size_mm=self.marker_size_mm)
        res = aligner.anchor_to_absolute_world(
            tag_poses, anchor_input,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            strict=strict
        )
        self.marker_size_mm = aligner.marker_size_mm
        return res

    def align_relative_map_to_world(self,
                                    relative_map: Dict[str, Any],
                                    anchor_tags: Dict[int, Dict[str, Any]],
                                    origin_tag_id: int = 0,
                                    x_align_tag_id: int = 1,
                                    strict: bool = True) -> Dict[str, Any]:
        """【阶段二独立解算核心】将自由平差相对底图对齐至世界系 (委托 WorldDatumAligner 实现)"""
        aligner = WorldDatumAligner(marker_size_mm=self.marker_size_mm)
        res = aligner.align_relative_map_to_world(
            relative_map, anchor_tags,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            strict=strict
        )
        self.marker_size_mm = aligner.marker_size_mm
        return res

    def align_to_scara_world(self,
                             tag_poses: Dict[int, np.ndarray], 
                             origin_tag_id: int, 
                             x_align_tag_id: int) -> Dict[str, Any]:
        """相对单靶原点退化对齐 (委托 WorldDatumAligner 实现)"""
        aligner = WorldDatumAligner(marker_size_mm=self.marker_size_mm)
        return aligner.align_to_scara_world(tag_poses, origin_tag_id, x_align_tag_id)

