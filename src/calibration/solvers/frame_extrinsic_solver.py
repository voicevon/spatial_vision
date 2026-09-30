#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
子坐标系外参反推求解器 (Frame Extrinsic Solver)
=============================================
核心职责:
1. 建立子坐标系从 "unknown" (未知) 到 "calibrated" (已平差解算) 的数学反推管道；
2. 依据子坐标系中声明的 calibration_spec (标定规范) 与 BA 输出的世界系标靶地图 (tags_map)，
   自动反推子坐标系到其父级坐标系的 6DoF 刚体外参 T_parent_from_child；
3. 支持三种工程反推求解模式:
   - "anchor_tags_registration": 多标靶 3D 刚体点云配准 (Umeyama SVD 闭式解析, scale=1.0)；
   - "axis_align": 双标靶基准定轴模式 (原点靶 + X轴对准靶)；
   - "single_tag_offset": 单标靶已知刚体偏移模式；
4. 输出逐标靶物理拟合残差与全局 RMSE，提供装配精度与形变评估。
"""

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.workspace.coordinate_manager import (
    CoordinateTreeManager,
    FrameDefinition,
    make_transform_matrix,
    rot_mat_to_rpy_deg,
    rpy_deg_to_rot_mat,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


def rigid_transform_3d(pts_from: np.ndarray, pts_to: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    闭式求解 3D 刚体变换: P_to = R @ P_from + t (尺度严格固定为 1.0, 纯刚体)
    :param pts_from: (N, 3) 局部名义标靶坐标集 (子坐标系)
    :param pts_to: (N, 3) 目标标靶坐标集 (父坐标系或世界系实测)
    :return: (R 3x3, t 3, rmse_mm)
    """
    assert pts_from.shape == pts_to.shape, "点集维度必须一致"
    n = pts_from.shape[0]
    if n < 3:
        raise ValueError("刚体 3D 配准至少需要 3 个非共线点对")

    centroid_from = np.mean(pts_from, axis=0)
    centroid_to = np.mean(pts_to, axis=0)

    X = pts_from - centroid_from
    Y = pts_to - centroid_to

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T

    # 反射矩阵修正 (保证 det(R) == +1, 杜绝镜像手性翻转)
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T

    t_vec = centroid_to - R_mat @ centroid_from

    # 计算逐点残差与 RMSE
    pts_pred = (R_mat @ pts_from.T).T + t_vec
    residuals = np.linalg.norm(pts_pred - pts_to, axis=1)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return R_mat, t_vec, rmse


class FrameExtrinsicSolver:
    """工位子坐标系外参反推求解器"""

    def __init__(self, tags_map: Union[Dict[int, np.ndarray], Dict[str, Any]]):
        """
        :param tags_map: 当前工位 BA 平差解算出的标靶地图，支持以下格式:
                         - {tag_id: 4x4 T_world_from_tag}
                         - tags_map.yaml 字典 (含 "tags" 节点)
        """
        self._tag_poses_world: Dict[int, np.ndarray] = {}
        self._tag_centers_world: Dict[int, np.ndarray] = {}
        self._parse_tags_map(tags_map)

    def _parse_tags_map(self, tags_map: Union[Dict[int, np.ndarray], Dict[str, Any]]):
        """解析并统一归一化标靶位姿与中心点"""
        if not tags_map:
            return

        # 检查是否为 tags_map.yaml 导出的字典格式
        raw_tags = tags_map.get("tags") if isinstance(tags_map, dict) and "tags" in tags_map else tags_map

        for k, v in raw_tags.items():
            try:
                tid = int(k)
            except (ValueError, TypeError):
                continue

            T = None
            if isinstance(v, np.ndarray) and v.shape == (4, 4):
                T = v.astype(np.float64)
            elif isinstance(v, dict):
                if "transform_matrix" in v:
                    T = np.array(v["transform_matrix"], dtype=np.float64)
                elif "center" in v or "position_mm" in v:
                    pos = v.get("position_mm", v.get("center"))
                    T = np.eye(4, dtype=np.float64)
                    T[:3, 3] = np.array(pos[:3], dtype=np.float64)

            if T is not None and T.shape == (4, 4):
                self._tag_poses_world[tid] = T
                self._tag_centers_world[tid] = T[:3, 3].copy()

    def get_tag_world_center(self, tag_id: int) -> Optional[np.ndarray]:
        """获取指定标靶在世界系下的物理中心坐标 (3,)"""
        return self._tag_centers_world.get(tag_id)

    def get_tag_world_pose(self, tag_id: int) -> Optional[np.ndarray]:
        """获取指定标靶在世界系下的 4x4 位姿变换矩阵"""
        return self._tag_poses_world.get(tag_id)

    def solve_frame_extrinsic(
        self,
        frame: FrameDefinition,
        coord_mgr: Optional[CoordinateTreeManager] = None,
    ) -> Tuple[bool, Optional[List[float]], Optional[List[float]], Optional[float], str]:
        """
        反推指定子坐标系相对于其父坐标系的外参 (translation_xyz_mm, rotation_rpy_deg)。
        :param frame: 待解算的子坐标系定义
        :param coord_mgr: 坐标系树管理器 (用于求解父坐标系到世界系的变换)
        :return: (success, translation_xyz_mm, rotation_rpy_deg, rmse_mm, message)
        """
        if frame.type == "world":
            return True, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "世界坐标系为绝对基准"

        if frame.type == "tag_bound":
            return True, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, "动标绑定坐标系无需反推固定外参"

        spec = frame.calibration_spec
        if not spec or not isinstance(spec, dict):
            return False, None, None, None, f"坐标系 [{frame.name or frame.frame_id}] 未配置 calibration_spec 标定规范"

        method = spec.get("method", "anchor_tags_registration")

        # 确定父坐标系到位姿转换关系 T_world_from_parent
        parent_id = frame.parent_frame_id or "world"
        T_world_from_parent = np.eye(4, dtype=np.float64)
        if parent_id != "world":
            if coord_mgr is None:
                return False, None, None, None, f"解算非 world 父级坐标系 [{parent_id}] 时必须传入 coord_mgr"
            T_w_p, is_p_res = coord_mgr.get_frame_to_world(parent_id)
            if not is_p_res:
                return False, None, None, None, f"父坐标系 [{parent_id}] 尚未解算，无法反推子坐标系"
            T_world_from_parent = T_w_p

        T_parent_from_world = np.linalg.inv(T_world_from_parent)

        if method in ("anchor_tags_registration", "multi_tag_registration", "registration_3d"):
            return self._solve_by_registration(frame, spec, T_parent_from_world)
        elif method in ("axis_align", "two_tag_alignment"):
            return self._solve_by_axis_align(frame, spec, T_parent_from_world)
        elif method in ("single_tag_offset", "single_tag"):
            return self._solve_by_single_tag(frame, spec, T_parent_from_world)
        else:
            return False, None, None, None, f"不支持的反推标定方法: {method}"

    def _solve_by_registration(
        self,
        frame: FrameDefinition,
        spec: Dict[str, Any],
        T_parent_from_world: np.ndarray,
    ) -> Tuple[bool, Optional[List[float]], Optional[List[float]], Optional[float], str]:
        """模式 1: 多标靶 3D 点云刚体配准反推"""
        ref_tags = spec.get("reference_tags", {})
        if not ref_tags:
            return False, None, None, None, "未指定 reference_tags 标靶名义坐标表"

        pts_child_list = []
        pts_parent_list = []
        matched_tids = []

        for tid_raw, pos_local in ref_tags.items():
            try:
                tid = int(tid_raw)
            except (ValueError, TypeError):
                continue

            pw = self.get_tag_world_center(tid)
            if pw is None:
                continue

            # 变换到父坐标系下
            pw_homo = np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float64)
            p_parent = (T_parent_from_world @ pw_homo)[:3]

            pts_child_list.append(np.array(pos_local[:3], dtype=np.float64))
            pts_parent_list.append(p_parent)
            matched_tids.append(tid)

        if len(matched_tids) < 3:
            # 若恰好匹配 2 枚标靶，智能降级为双标靶定轴模式
            if len(matched_tids) == 2:
                log.info(f"[FrameExtrinsic] 坐标系 [{frame.frame_id}] 仅匹配 2 枚标靶 (Tag {matched_tids})，降级为定轴模式")
                spec_axis = {
                    "origin_tag_id": matched_tids[0],
                    "x_axis_tag_id": matched_tids[1],
                    "origin_local_xyz_mm": pts_child_list[0].tolist(),
                    "x_axis_local_xyz_mm": pts_child_list[1].tolist(),
                }
                return self._solve_by_axis_align(frame, spec_axis, T_parent_from_world)
            return False, None, None, None, f"有效匹配标靶数量不足 ({len(matched_tids)} < 3)，无法执行 3D 刚体配准"

        pts_child = np.array(pts_child_list, dtype=np.float64)
        pts_parent = np.array(pts_parent_list, dtype=np.float64)

        try:
            # 求解刚体变换: P_parent = R @ P_child + t
            R_mat, t_vec, rmse = rigid_transform_3d(pts_child, pts_parent)
            rpy = rot_mat_to_rpy_deg(R_mat)
            t_xyz = [float(x) for x in t_vec]

            msg = f"多标靶刚体配准成功: 匹配 {len(matched_tids)} 枚标靶 {matched_tids}，残差 RMSE: {rmse:.3f} mm"
            log.info(f"[FrameExtrinsic] 坐标系 [{frame.frame_id}] {msg}")
            return True, t_xyz, rpy, rmse, msg
        except Exception as e:
            return False, None, None, None, f"3D 刚体配准解算异常: {e}"

    def _solve_by_axis_align(
        self,
        frame: FrameDefinition,
        spec: Dict[str, Any],
        T_parent_from_world: np.ndarray,
    ) -> Tuple[bool, Optional[List[float]], Optional[List[float]], Optional[float], str]:
        """模式 2: 双标靶基准定轴模式 (原点靶 + X轴靶)"""
        origin_tid = spec.get("origin_tag_id")
        x_tid = spec.get("x_axis_tag_id")
        if origin_tid is None or x_tid is None:
            return False, None, None, None, "定轴模式必须提供 origin_tag_id 与 x_axis_tag_id"

        pw_o = self.get_tag_world_center(int(origin_tid))
        pw_x = self.get_tag_world_center(int(x_tid))
        if pw_o is None or pw_x is None:
            missing = []
            if pw_o is None:
                missing.append(int(origin_tid))
            if pw_x is None:
                missing.append(int(x_tid))
            return False, None, None, None, f"定轴标靶在当前地图中未找到: Tag {missing}"

        # 映射至父坐标系
        p_parent_o = (T_parent_from_world @ np.array([pw_o[0], pw_o[1], pw_o[2], 1.0]))[:3]
        p_parent_x = (T_parent_from_world @ np.array([pw_x[0], pw_x[1], pw_x[2], 1.0]))[:3]

        # 计算 X 轴向量
        vx = p_parent_x - p_parent_o
        dist_meas = float(np.linalg.norm(vx))
        if dist_meas < 1e-3:
            return False, None, None, None, "原点标靶与 X 轴标靶在物理空间重合，无法定轴"

        vx_unit = vx / dist_meas

        # 计算 Y/Z 轴: 优先保持父坐标系水平/垂直基准
        vz_candidate = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(vx_unit, vz_candidate))) > 0.95:
            # 若 X 轴几乎垂直，改用 Y 轴为辅助
            vz_candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        vy = np.cross(vz_candidate, vx_unit)
        vy_unit = vy / np.linalg.norm(vy)
        vz_unit = np.cross(vx_unit, vy_unit)

        R_mat = np.column_stack([vx_unit, vy_unit, vz_unit])
        rpy = rot_mat_to_rpy_deg(R_mat)

        # 考虑局部名义偏移 (若标靶并不在 child 的 [0,0,0])
        origin_local = spec.get("origin_local_xyz_mm", [0.0, 0.0, 0.0])
        t_vec = p_parent_o - R_mat @ np.array(origin_local[:3], dtype=np.float64)
        t_xyz = [float(x) for x in t_vec]

        # 若提供了理论标称距离，评估尺差
        rmse = 0.0
        x_local = spec.get("x_axis_local_xyz_mm")
        if x_local:
            dist_nom = float(np.linalg.norm(np.array(x_local[:3]) - np.array(origin_local[:3])))
            rmse = abs(dist_meas - dist_nom)

        msg = f"双标靶定轴对齐成功: Tag {origin_tid} -> Tag {x_tid}，实测间距: {dist_meas:.1f} mm"
        log.info(f"[FrameExtrinsic] 坐标系 [{frame.frame_id}] {msg}")
        return True, t_xyz, rpy, rmse, msg

    def _solve_by_single_tag(
        self,
        frame: FrameDefinition,
        spec: Dict[str, Any],
        T_parent_from_world: np.ndarray,
    ) -> Tuple[bool, Optional[List[float]], Optional[List[float]], Optional[float], str]:
        """模式 3: 单标靶已知偏移推导"""
        tag_id = spec.get("tag_id")
        if tag_id is None:
            return False, None, None, None, "单标靶模式未提供 tag_id"

        T_w_t = self.get_tag_world_pose(int(tag_id))
        if T_w_t is None:
            return False, None, None, None, f"标靶 Tag {tag_id} 在地图中不存在"

        off_xyz = spec.get("offset_xyz_mm", [0.0, 0.0, 0.0])
        off_rpy = spec.get("offset_rpy_deg", [0.0, 0.0, 0.0])

        T_t_c = make_transform_matrix(rpy_deg_to_rot_mat(off_rpy), off_xyz)
        T_w_c = T_w_t @ T_t_c
        T_parent_from_child = T_parent_from_world @ T_w_c

        t_xyz = [float(x) for x in T_parent_from_child[:3, 3]]
        rpy = rot_mat_to_rpy_deg(T_parent_from_child[:3, :3])

        msg = f"单标靶刚体偏移反推成功: Tag {tag_id}"
        log.info(f"[FrameExtrinsic] 坐标系 [{frame.frame_id}] {msg}")
        return True, t_xyz, rpy, 0.0, msg

    def solve_all_unknown_frames(
        self,
        coord_mgr: CoordinateTreeManager,
    ) -> Dict[str, Dict[str, Any]]:
        """
        批量求解坐标系树中所有声明了标定规范的 unknown 坐标系 (多轮拓扑递推，支持多级父子继承)
        :param coord_mgr: 工位坐标系树管理器
        :return: {frame_id: {"success": bool, "translation_xyz_mm": ..., "rotation_rpy_deg": ..., "rmse_mm": ..., "message": ...}}
        """
        results: Dict[str, Dict[str, Any]] = {}
        all_frames = coord_mgr.list_frames()
        targets = [f for f in all_frames if f.type == "fixed_transform" and f.status in ("unknown", "partial") and f.calibration_spec]

        if not targets:
            log.info("[FrameExtrinsic] 无待反推的 unknown 坐标系")
            return results

        # 最多进行 N 轮迭代解析 (解决深层父级依赖关系)
        remaining = list(targets)
        for _ in range(len(targets) + 1):
            if not remaining:
                break
            progress = False
            next_remaining = []

            for frame in remaining:
                succ, t_xyz, rpy, rmse, msg = self.solve_frame_extrinsic(frame, coord_mgr)
                if succ and t_xyz is not None and rpy is not None:
                    coord_mgr.update_frame_solved_extrinsic(
                        frame_id=frame.frame_id,
                        translation_xyz_mm=t_xyz,
                        rotation_rpy_deg=rpy,
                        rmse_mm=rmse,
                        method=frame.calibration_spec.get("method", "registration")
                    )
                    results[frame.frame_id] = {
                        "success": True,
                        "translation_xyz_mm": t_xyz,
                        "rotation_rpy_deg": rpy,
                        "rmse_mm": rmse,
                        "message": msg,
                    }
                    progress = True
                else:
                    next_remaining.append(frame)
                    results[frame.frame_id] = {
                        "success": False,
                        "message": msg,
                    }

            remaining = next_remaining
            if not progress:
                # 无法继续推导更深层节点 (可能依赖未解算的父系或标靶缺失)
                break

        return results
