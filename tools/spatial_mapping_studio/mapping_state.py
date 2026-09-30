#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MappingDataManager - 空间建图工作站数据管理与指标缓存层
======================================================
单一职责：
1. 采图资产扫描与当前选定帧维护；
2. 观测清单 (Manifest YAML) 与标靶立体几何地图 (tags_map.yaml) 的持久化与同步；
3. 全景/单帧重投影误差 (RMSE) 精度体检缓存与实时计算；
4. 左栏序列过滤筛选 (All / Warning / Excluded) 与四种排序规则。
变更型动作 (超精提取/剪枝/快照/复位/翻转) 拆分至 mapping_data_actions.py (MappingDataActionsMixin)。
"""

import os
import glob
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2
import yaml

from src.calibration.manifest_repository import ManifestRepository
from src.vision.pnp_solver import PnpSolver
from src.vision.tag_detector import TagDetector
from tools.spatial_mapping_studio.mapping_data_actions import MappingDataActionsMixin
from src.utils.logger import get_logger

log = get_logger(__name__)


class MappingDataManager(MappingDataActionsMixin):
    """空间建图工作站 (Spatial Mapping Studio) 领域模型与数据状态管理器 (变更型动作见 MappingDataActionsMixin)"""

    def __init__(
        self,
        map_path: str,
        image_dir: str,
        manifest_path: str,
        pnp_solver: PnpSolver,
        marker_size_mm: float,
        tag_detector: Optional[TagDetector] = None
    ):
        self.map_path = map_path
        self.image_dir = image_dir
        self.manifest_path = manifest_path
        self.pnp_solver = pnp_solver
        self.marker_size_mm = marker_size_mm
        self.tag_detector = tag_detector or TagDetector()

        # 1. 资产与清单
        self.manifest_repo = ManifestRepository()
        self.manifest_data: Dict[str, Any] = {}
        self._load_manifest()

        self.tags_map_data: Dict[str, Any] = {}
        self._load_tags_map()

        self.image_files: List[str] = []
        self._scan_images()
        self.current_img_idx: int = 0
        self.scroll_offset: int = 0

        # 2. 筛选与排序模式
        self.filter_mode: str = "all"
        self.sort_mode: str = "name_asc"

        # 3. 体检指标缓存
        self.frame_metrics_cache: Dict[str, Dict[str, Any]] = {}
        self.global_rmse: float = 0.0
        self.global_median_mm: float = 0.0
        self.global_mean_mm: float = 0.0
        self.gate_status: str = "REVIEW"
        self.topology_status: Dict[str, Any] = {
            "is_valid": True,
            "components_count": 1,
            "critical_bridges": [],
            "unconnected_tags": [],
            "connected_tags_count": 0,
            "message": "就绪"
        }
        self.current_diagnostics: Dict[str, Any] = {}

        # 4. 智能剪枝平差逐帧多轮残差收敛矩阵
        # frame_convergence_matrix: { "view_0001.png": [582.13, 39.81, 25.30, ...] }
        # convergence_headers: ["R0(基准)", "R1", "R2", ...]
        self.frame_convergence_matrix: Dict[str, List[Optional[float]]] = {}
        self.convergence_headers: List[str] = []

        # 5. 超精重提取引擎 (惰性装载)
        self._super_extractor = None

        # 6. 世界 XY 透视网格与 Z 轴特殊点观察状态 (移植自在线跟踪)
        self.show_xy_plane_on: bool = False
        self.plane_z: float = 0.0

        # 首次加载全集残差指标
        self.refresh_all_frame_metrics()

    PLANE_EXTENT_MM: int = 600       # XY 平面网格半宽 (mm)
    PLANE_STEP_MM: int = 100         # XY 平面网格间距 (mm)
    PLANE_Z_MM: int = 600            # Z 轴绘制最大高度 (mm)
    PLANE_Z_BASE_CHOICES = (350, 300, 250, 200, 150, 100, 50, 0)
    PLANE_Z_STATIC_LABELS = {
        350: "", 300: "", 250: "", 200: "", 150: "", 100: "", 50: "", 0: " (地面)"
    }

    def get_plane_z_options(self) -> List[Tuple[Optional[float], str]]:
        """从世界坐标地图动态提取所有已知标靶的中心 Z 坐标作为特殊点，并与基础梯度合并降序排列"""
        anchor_z_labels = {}
        if self.tags_map_data and "tags" in self.tags_map_data:
            for tid, t_info in self.tags_map_data["tags"].items():
                mat = t_info.get("transform_matrix")
                if mat and len(mat) == 4:
                    z_val = float(mat[2][3])
                    z_key = int(round(z_val))
                    if z_key in anchor_z_labels:
                        anchor_z_labels[z_key] += f"/Tag {tid}"
                    else:
                        anchor_z_labels[z_key] = f" (Tag {tid})"

        combined_labels = {**self.PLANE_Z_STATIC_LABELS, **anchor_z_labels}
        all_choices = sorted(set(self.PLANE_Z_BASE_CHOICES) | set(anchor_z_labels.keys()), reverse=True)

        options: List[Tuple[Optional[float], str]] = [
            (None, "不绘制 XY 平面")
        ]
        for z in all_choices:
            options.append((float(z), f"Z {z} mm" + combined_labels.get(z, "")))
        return options

    def get_current_plane_z_label(self) -> str:
        """获取当前选中的 Z 高度或特殊点简要标签 (用于顶栏按钮紧凑呈现)"""
        z_int = int(round(self.plane_z))
        options = self.get_plane_z_options()
        for opt_val, opt_lbl in options:
            if opt_val is not None and abs(opt_val - self.plane_z) < 1.0:
                compact_lbl = opt_lbl.replace(" mm", "mm")
                return f"Z: {compact_lbl[2:]}"  # 如 "Z: 196mm (Tag 1)" 或 "Z: 0mm (地面)"
        return f"Z: {z_int}mm"

    def step_plane_z(self, direction: int) -> None:
        """快捷键 [ / ] 升降切换 Z 轴特殊点 (direction: +1 上一档/升高, -1 下一档/降低)"""
        opts = [opt_val for opt_val, _ in self.get_plane_z_options() if opt_val is not None]
        if not opts:
            return
        cur_idx = 0
        min_diff = 1e9
        for i, val in enumerate(opts):
            diff = abs(val - self.plane_z)
            if diff < min_diff:
                min_diff = diff
                cur_idx = i

        # direction > 0: 升高 -> 对应降序列表中索引减小
        if direction > 0:
            new_idx = max(0, cur_idx - 1)
        else:
            new_idx = min(len(opts) - 1, cur_idx + 1)

        self.plane_z = opts[new_idx]
        self.show_xy_plane_on = True

    def reload_dataset(self, map_path: str, image_dir: str, manifest_path: str):
        """场景切换时整套重载数据集：路径更新、清单/地图重载、图片扫描与指标重算"""
        self.map_path = map_path
        self.image_dir = image_dir
        self.manifest_path = manifest_path
        self._super_extractor = None

        self._load_manifest()
        self._load_tags_map()
        self._scan_images()
        self.current_img_idx = 0
        self.scroll_offset = 0

        self.frame_metrics_cache.clear()
        self.frame_convergence_matrix.clear()
        self.convergence_headers.clear()
        self.refresh_all_frame_metrics()

    @property
    def super_extractor(self):
        """惰性装载工序 3 工业级超精重提取引擎"""
        if self._super_extractor is None:
            from tools.calibration.tag_super_extractor import TagSuperExtractor
            self._super_extractor = TagSuperExtractor(
                image_dir=self.image_dir,
                manifest_path=self.manifest_path,
                marker_size_mm=self.marker_size_mm
            )
        return self._super_extractor

    def _sync_manifest_summary(self):
        """同步更新 manifest 顶层 summary 统计指标"""
        total_obs = 0
        total_kept = 0
        total_excl = 0
        total_enabled = 0
        images_dict = self.manifest_data.get("images", {})
        for img_info in images_dict.values():
            if img_info.get("enabled", True) and not img_info.get("excluded", False):
                total_enabled += 1
            for obs in img_info.get("observations", []):
                total_obs += 1
                if obs.get("keep", True):
                    total_kept += 1
                else:
                    total_excl += 1
        if "summary" not in self.manifest_data or not isinstance(self.manifest_data["summary"], dict):
            self.manifest_data["summary"] = {}
        self.manifest_data["summary"].update({
            "total_images": len(images_dict),
            "total_enabled_images": total_enabled,
            "total_observations": total_obs,
            "total_kept": total_kept,
            "total_excluded": total_excl
        })

    def _scan_images(self):
        """扫描采图资产目录"""
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir, exist_ok=True)
            self.image_files = []
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        self.image_files = sorted(files)

    def _load_manifest(self):
        """加载观测清单文件"""
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    self.manifest_data = yaml.safe_load(f) or {}
            except Exception:
                self.manifest_data = {"images": {}}
        else:
            self.manifest_data = {"images": {}}

    def _save_manifest(self):
        """保存观测清单文件"""
        os.makedirs(os.path.dirname(self.manifest_path), exist_ok=True)
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                yaml.dump(self.manifest_data, f, allow_unicode=True, sort_keys=False)
        except Exception as e:
            log.warning(f"保存 Manifest 异常: {e}")

    def _load_tags_map(self):
        """加载空间立体地图数据"""
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.tags_map_data = yaml.safe_load(f) or {}
                if self.tags_map_data and "tags" in self.tags_map_data:
                    log.info(f"[OK] Studio 成功装载地图: {self.map_path} (共 {len(self.tags_map_data['tags'])} 个标靶)")
                    # 优先采用地图 BA 反算的真实边长, 保证绿(BA理论)/蓝(实测)棱柱比例与偏差解算一致
                    self.set_marker_size_mm(self.tags_map_data.get("marker_size_mm"))
            except Exception as e:
                log.warning(f"无法读取地图: {e}")
                self.tags_map_data = {}
        else:
            self.tags_map_data = {}
        if self.pnp_solver:
            self.pnp_solver.tags_map = self.tags_map_data

    def set_marker_size_mm(self, size_mm) -> None:
        """同步标靶物理边长到全局状态与引擎单靶 PnP 模型"""
        try:
            size_mm = float(size_mm)
        except (TypeError, ValueError):
            return
        if size_mm <= 0:
            return
        self.marker_size_mm = size_mm
        if getattr(self, "pnp_solver", None):
            self.pnp_solver.set_marker_size_mm(size_mm)

    def get_tag_transform(self, tag_id: int) -> Optional[np.ndarray]:
        """获取已知标靶在世界系下的 4x4 位姿变换矩阵"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return None
        t_info = self.tags_map_data["tags"].get(tag_id)
        if not t_info or "transform_matrix" not in t_info:
            return None
        return np.array(t_info["transform_matrix"], dtype=np.float64)

    def get_tag_world_corners(self, tag_id: int) -> Optional[np.ndarray]:
        """获取标靶 4 角点在世界坐标系下的 3D 物理坐标 (4, 3)"""
        T = self.get_tag_transform(tag_id)
        if T is None:
            return None
        s = self.marker_size_mm / 2.0
        local_corners = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
        world_corners = (T @ local_corners.T).T
        return world_corners[:, :3]

    def get_observations_for_image(self, base_name: str) -> List[Dict[str, Any]]:
        """获取某张采图下的所有标靶观测记录"""
        images_dict = self.manifest_data.get("images", {})
        info = images_dict.get(base_name, {})
        return info.get("observations", [])

    def is_image_excluded(self, base_name: str) -> bool:
        """判断某帧是否被整帧标记为剔除 (兼容 excluded 与 not enabled)"""
        images_dict = self.manifest_data.get("images", {})
        info = images_dict.get(base_name, {})
        if "excluded" in info:
            return bool(info["excluded"])
        if "enabled" in info:
            return not bool(info["enabled"])
        return False

    def _evaluate_frame_reprojection(
        self,
        observations: List[Dict[str, Any]]
    ) -> Tuple[float, float, Dict[int, float], Optional[np.ndarray], Optional[np.ndarray], float, Dict[int, float]]:
        """计算单帧中所有有效标靶的像素残差与空间毫米偏差"""
        if not self.tags_map_data or "tags" not in self.tags_map_data:
            return 0.0, 0.0, {}, None, None, 0.0, {}

        obj_pts = []
        img_pts = []
        valid_tids = []

        for obs in observations:
            if not obs.get("keep", True):
                continue
            tid = obs["tag_id"]
            w_corners = self.get_tag_world_corners(tid)
            if w_corners is not None:
                c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                obj_pts.append(w_corners)
                img_pts.append(c_arr)
                valid_tids.append(tid)

        if not obj_pts:
            return 0.0, 0.0, {}, None, None, 0.0, {}

        obj_flat = np.concatenate(obj_pts, axis=0)
        img_flat = np.concatenate(img_pts, axis=0)

        rvec, tvec, success = self.pnp_solver.solve_pnp(obj_flat, img_flat)
        if not success:
            return 0.0, 0.0, {}, None, None, 0.0, {}

        proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, self.pnp_solver.camera_matrix, self.pnp_solver.dist_coeffs)
        dists = np.linalg.norm(img_flat - proj_pts.reshape((-1, 2)), axis=1)

        tz = float(tvec[2, 0]) if tvec is not None else 800.0
        fx = float(self.pnp_solver.camera_matrix[0, 0]) if (self.pnp_solver and self.pnp_solver.camera_matrix is not None) else 1363.0
        scale_mm_per_px = abs(tz) / fx if fx > 0 else 0.0

        errors_dict = {}
        errors_dict_mm = {}
        for idx, tid in enumerate(valid_tids):
            err_px = float(np.mean(dists[idx * 4:(idx + 1) * 4]))
            errors_dict[tid] = err_px
            errors_dict_mm[tid] = float(err_px * scale_mm_per_px)

        mean_val = float(np.mean(dists))
        max_val = float(np.max(dists))
        mean_val_mm = float(mean_val * scale_mm_per_px)
        return mean_val, max_val, errors_dict, rvec, tvec, mean_val_mm, errors_dict_mm

    def refresh_all_frame_metrics(self):
        """全量预热并刷新所有采图帧的精度体检残差指标、空间毫米偏差与共视拓扑健康度"""
        self.frame_metrics_cache.clear()
        all_reproj_errors = []
        all_mm_errors = []
        valid_frame_detections = []
        valid_frame_names = []

        for p in self.image_files:
            base_name = os.path.basename(p)
            obs_list = self.get_observations_for_image(base_name)
            is_excl = self.is_image_excluded(base_name)

            mean_err, max_err, errors_dict, rvec, tvec, mean_mm, errs_mm = self._evaluate_frame_reprojection(obs_list)
            if not is_excl and obs_list:
                all_reproj_errors.extend(list(errors_dict.values()))
                all_mm_errors.extend(list(errs_mm.values()))
                tag_dict = {}
                for obs in obs_list:
                    if obs.get("keep", True):
                        tag_dict[int(obs["tag_id"])] = np.array(obs["corners"], dtype=np.float64)
                if len(tag_dict) >= 1:
                    valid_frame_detections.append(tag_dict)
                    valid_frame_names.append(base_name)

            self.frame_metrics_cache[base_name] = {
                "tag_count": len(obs_list),
                "mean_err": mean_err,
                "max_err": max_err,
                "mean_err_mm": mean_mm,
                "is_excluded": is_excl,
                "observations": obs_list,
                "tag_errors": errors_dict,
                "tag_errors_mm": errs_mm,
                "rvec": rvec,
                "tvec": tvec
            }

        if all_reproj_errors:
            self.global_rmse = float(np.sqrt(np.mean(np.array(all_reproj_errors) ** 2)))
        else:
            self.global_rmse = 0.0

        if all_mm_errors:
            self.global_median_mm = float(np.median(np.array(all_mm_errors)))
            self.global_mean_mm = float(np.mean(np.array(all_mm_errors)))
        else:
            self.global_median_mm = 0.0
            self.global_mean_mm = 0.0

        # 共视拓扑连通度检查
        if valid_frame_detections:
            try:
                from src.calibration.solvers.covisibility_graph import CovisibilityGraphAnalyzer
                topo = CovisibilityGraphAnalyzer.analyze(valid_frame_detections, valid_frame_names)
                self.topology_status = {
                    "is_valid": bool(topo.get("is_valid", False)),
                    "components_count": len(topo.get("components", [])),
                    "critical_bridges": topo.get("critical_bridges", []),
                    "unconnected_tags": topo.get("unconnected_tags", []),
                    "connected_tags_count": len(topo.get("connected_tags", [])),
                    "message": str(topo.get("message", "拓扑分析完成"))
                }
            except Exception as e:
                self.topology_status = {
                    "is_valid": True,
                    "components_count": 1,
                    "critical_bridges": [],
                    "unconnected_tags": [],
                    "connected_tags_count": 0,
                    "message": f"拓扑分析异常: {e}"
                }
        else:
            self.topology_status = {
                "is_valid": False,
                "components_count": 0,
                "critical_bridges": [],
                "unconnected_tags": [],
                "connected_tags_count": 0,
                "message": "暂无参与解算的有效采图帧"
            }

        # 放行门限评定
        if self.global_rmse > 0 and self.global_rmse <= 0.50 and self.global_median_mm <= 1.50 and self.topology_status.get("is_valid", False):
            self.gate_status = "PASS"
        elif self.global_rmse > 0 and self.global_rmse <= 1.00:
            self.gate_status = "ACCEPTABLE"
        else:
            self.gate_status = "REVIEW"

    def diagnose_frame(self, img_idx: Optional[int] = None) -> Dict[str, Any]:
        """
        对指定帧或当前选定帧执行工序 3/4 深度图像质量与漏检病因切片诊断
        """
        if img_idx is None:
            img_idx = self.current_img_idx
        if not self.image_files or img_idx >= len(self.image_files):
            return {}

        cur_file = self.image_files[img_idx]
        bname = os.path.basename(cur_file)
        bgr = cv2.imread(cur_file)
        if bgr is None:
            return {}

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        contrast = float(np.std(gray))
        brightness = float(np.mean(gray))
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 候选多边形与拒检分析
        detector_bright = self.tag_detector.detector_bright
        c_raw, ids_raw, rejected = detector_bright.detectMarkers(gray)
        detected_tids = set(ids_raw.flatten().tolist()) if ids_raw is not None else set()
        rej_count = len(rejected) if rejected is not None else 0

        rej_small = 0
        rej_aspect = 0
        if rejected is not None:
            for r in rejected:
                pts = r.reshape((4, 2))
                area = cv2.contourArea(pts)
                if area < 200:
                    rej_small += 1
                side_a = np.linalg.norm(pts[0] - pts[1])
                side_b = np.linalg.norm(pts[1] - pts[2])
                if side_b > 0 and (side_a / side_b > 3.0 or side_b / side_a > 3.0):
                    rej_aspect += 1

        # 理论漏检分析
        missing_tags = []
        meta = self.frame_metrics_cache.get(bname, {})
        rvec = meta.get("rvec")
        tvec = meta.get("tvec")
        if rvec is not None and tvec is not None:
            mapped_tids = self.tags_map_data.get("tags", {})
            for tid_str in mapped_tids.keys():
                t_int = int(tid_str)
                if t_int not in detected_tids:
                    wc = self.get_tag_world_corners(t_int)
                    if wc is not None:
                        proj, _ = cv2.projectPoints(wc, rvec, tvec, self.pnp_solver.camera_matrix, self.pnp_solver.dist_coeffs)
                        p2 = proj.reshape((4, 2))
                        if np.all(p2[:, 0] >= -20) and np.all(p2[:, 0] < w + 20) and np.all(p2[:, 1] >= -20) and np.all(p2[:, 1] < h + 20):
                            missing_tags.append({
                                "tag_id": t_int,
                                "predicted_corners": p2.tolist(),
                                "reason": "视场内但未被快速检出 (建议按 E 超精提取)"
                            })

        diag_res = {
            "image": bname,
            "contrast": contrast,
            "contrast_rms": contrast,
            "contrast_grade": "良" if contrast >= 35 else ("偏低" if contrast >= 20 else "极差"),
            "brightness": brightness,
            "mean_intensity": brightness,
            "brightness_grade": "正常" if 60 <= brightness <= 190 else ("偏暗" if brightness < 60 else "过曝"),
            "sharpness": sharpness,
            "laplacian_var": sharpness,
            "sharpness_grade": "清晰" if sharpness >= 100 else ("轻微模糊" if sharpness >= 50 else "严重虚焦"),
            "detected_count": len(detected_tids),
            "rejected_quads_count": rej_count,
            "false_rejections_count": rej_count,
            "rej_small": rej_small,
            "rej_aspect": rej_aspect,
            "missing_theoretical_tags": missing_tags,
            "missing_projected_tags": missing_tags
        }
        self.current_diagnostics = diag_res
        return diag_res

    def get_filtered_indices(self) -> List[int]:
        """依据当前的过滤模式与排序模式获取最终展示的帧索引列表"""
        matched = []
        for idx, p in enumerate(self.image_files):
            base_name = os.path.basename(p)
            meta = self.frame_metrics_cache.get(base_name, {})
            if self.filter_mode == "all":
                matched.append(idx)
            elif self.filter_mode == "warning":
                if meta.get("mean_err", 0.0) > 0.5 and not meta.get("is_excluded", False):
                    matched.append(idx)
            elif self.filter_mode == "excluded":
                if meta.get("is_excluded", False):
                    matched.append(idx)

        # 排序规则处理
        if self.sort_mode == "name_asc":
            matched.sort(key=lambda i: os.path.basename(self.image_files[i]))
        elif self.sort_mode == "err_desc":
            matched.sort(
                key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("mean_err", 0.0),
                reverse=True
            )
        elif self.sort_mode == "err_asc":
            matched.sort(
                key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("mean_err", 0.0)
            )
        elif self.sort_mode == "tags_desc":
            matched.sort(
                key=lambda i: self.frame_metrics_cache.get(os.path.basename(self.image_files[i]), {}).get("tag_count", 0),
                reverse=True
            )

        return matched
