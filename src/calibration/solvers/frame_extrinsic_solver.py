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

import math
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


def align_vectors_3d(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    计算将 3D 空间向量 v_from 严格旋转对齐至 v_to 的最小旋转矩阵 R (SO(3))
    满足: R @ (v_from / ||v_from||) = v_to / ||v_to||
    基于 Rodrigues 旋转公式 (轴角最小测地线旋转，无多余绕轴自旋)
    """
    norm_from = float(np.linalg.norm(v_from))
    norm_to = float(np.linalg.norm(v_to))
    if norm_from < 1e-9 or norm_to < 1e-9:
        return np.eye(3, dtype=np.float64)

    u_from = v_from / norm_from
    u_to = v_to / norm_to

    v_cross = np.cross(u_from, u_to)
    s = float(np.linalg.norm(v_cross))
    c = float(np.dot(u_from, u_to))

    # 两向量同向共线
    if s < 1e-7:
        if c > 0:
            return np.eye(3, dtype=np.float64)
        else:
            # 180 度反向共线翻转: 选取一个与 u_from 正交的单位轴进行 180 度旋转
            if abs(u_from[0]) < 0.9:
                ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = np.cross(u_from, ortho)
            axis = axis / np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    # Rodrigues 旋转公式
    vx = np.array([
        [0.0, -v_cross[2], v_cross[1]],
        [v_cross[2], 0.0, -v_cross[0]],
        [-v_cross[1], v_cross[0], 0.0]
    ], dtype=np.float64)
    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c) / (s ** 2))
    return R


def two_points_rigid_align_3d(
    p_from_a: np.ndarray,
    p_from_b: np.ndarray,
    p_to_a: np.ndarray,
    p_to_b: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    两点 3D 空间刚体对齐 (保距刚体变换, scale=1.0)
    数学原理与设计哲学:
    1. 已知点是坐标系建立的第一真理源和最大约束；
    2. 提取两标靶在源空间的名义向量 v_from 与在目标空间的实测向量 v_to；
    3. 通过 align_vectors_3d 求解将单位向量 u_from 旋转至 u_to 的测地线最小旋转矩阵 R in SO(3)；
    4. 质心对齐平移 t = centroid_to - R @ centroid_from；
    5. 对齐后两点在垂直于连线方向的残差分量严格为 0 (完全消除 Z 轴等法向假超差)，
       残差仅由两点标称间距与实测间距的尺度差 (欧氏距离差的一半) 决定。
    :return: (R 3x3, t 3, rmse_mm)
    """
    p_from_a = np.asarray(p_from_a, dtype=np.float64)
    p_from_b = np.asarray(p_from_b, dtype=np.float64)
    p_to_a = np.asarray(p_to_a, dtype=np.float64)
    p_to_b = np.asarray(p_to_b, dtype=np.float64)

    v_from = p_from_b - p_from_a
    v_to = p_to_b - p_to_a

    dist_from = float(np.linalg.norm(v_from))
    dist_to = float(np.linalg.norm(v_to))
    if dist_from < 1e-4 or dist_to < 1e-4:
        raise ValueError("两标靶空间距离过小 (<0.1mm)，无法唯一定向坐标轴")

    R_mat = align_vectors_3d(v_from, v_to)

    centroid_from = 0.5 * (p_from_a + p_from_b)
    centroid_to = 0.5 * (p_to_a + p_to_b)
    t_vec = centroid_to - R_mat @ centroid_from

    # 计算逐点残差与 RMSE
    err_a = p_to_a - (R_mat @ p_from_a + t_vec)
    err_b = p_to_b - (R_mat @ p_from_b + t_vec)
    rmse = float(np.sqrt(0.5 * (np.sum(err_a ** 2) + np.sum(err_b ** 2))))

    return R_mat, t_vec, rmse


def rigid_transform_3d(pts_from: np.ndarray, pts_to: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    闭式求解 3D 刚体变换: P_to = R @ P_from + t (尺度严格固定为 1.0, 纯刚体)
    :param pts_from: (N, 3) 局部名义标靶坐标集 (子坐标系)
    :param pts_to: (N, 3) 目标标靶坐标集 (父坐标系或世界系实测)
    :return: (R 3x3, t 3, rmse_mm)
    """
    assert pts_from.shape == pts_to.shape, "点集维度必须一致"
    n = pts_from.shape[0]
    if n < 2:
        raise ValueError("刚体 3D 配准至少需要 2 个对应点")
    if n == 2:
        return two_points_rigid_align_3d(pts_from[0], pts_from[1], pts_to[0], pts_to[1])

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

    def __init__(
        self,
        tags_map: Union[Dict[int, np.ndarray], Dict[str, Any]],
        whitelist_anchors: Optional[Dict[int, Dict]] = None,
    ):
        """
        :param tags_map: 当前工位 BA 平差解算出的标靶地图，支持以下格式:
                         - {tag_id: 4x4 T_world_from_tag}
                         - tags_map.yaml 字典 (含 "tags" 节点)
        :param whitelist_anchors: tag_whitelist.yaml 解析后的锚点字典 {tag_id: {"xyz_mm": [x,y,z], "known": [b,b,b]}}
                                  为子坐标系外参反推提供唯一权威的局部名义坐标真理源。
        """
        self._tag_poses_world: Dict[int, np.ndarray] = {}
        self._tag_centers_world: Dict[int, np.ndarray] = {}
        # whitelist_anchors 中的 xyz_mm 是 Tag 在其所属子坐标系中的名义局部坐标
        self._whitelist_anchors: Dict[int, Dict] = whitelist_anchors or {}
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
            elif isinstance(v, dict) and "transform_matrix" in v:
                T = np.array(v["transform_matrix"], dtype=np.float64)

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

        if method == "anchor_tags_registration":
            return self._solve_by_registration(frame, spec, T_parent_from_world)
        elif method == "axis_align":
            return self._solve_by_axis_align(frame, spec, T_parent_from_world)
        elif method == "single_tag_offset":
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
        # 从 reference_tag_ids 读取参与配准的 Tag ID 列表 (新规范)
        ref_ids_raw = spec.get("reference_tag_ids")
        if not ref_ids_raw:
            return False, None, None, None, (
                f"calibration_spec 未指定 reference_tag_ids，"
                f"请在 tag_whitelist.yaml 录入 Tag 局部坐标并在 calibration_spec 填写 reference_tag_ids: [11, 14]"
            )
        if not self._whitelist_anchors:
            return False, None, None, None, "whitelist_anchors 未传入，无法读取 Tag 局部名义坐标"

        pts_child_list = []
        pts_parent_list = []
        matched_tids = []
        missing_tids = []

        for tid_raw in ref_ids_raw:
            try:
                tid = int(tid_raw)
            except (ValueError, TypeError):
                continue

            anchor = self._whitelist_anchors.get(tid)
            if anchor is None:
                missing_tids.append(tid)
                continue
            pos_local = anchor.get("xyz_mm")
            if pos_local is None or len(pos_local) < 3 or None in pos_local:
                missing_tids.append(tid)
                continue

            pw = self.get_tag_world_center(tid)
            if pw is None:
                continue

            pw_homo = np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float64)
            p_parent = (T_parent_from_world @ pw_homo)[:3]

            pts_child_list.append(np.array(pos_local[:3], dtype=np.float64))
            pts_parent_list.append(p_parent)
            matched_tids.append(tid)

        if missing_tids:
            log.warning(f"[FrameExtrinsic] Tag {missing_tids} 在 whitelist_anchors 中无完整 xyz_mm，已跳过")

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
        """
        模式 2: 双标靶 3D 刚体空间对齐模式 (通用双标靶基准或原点靶+定轴靶)
        数学原理与设计哲学:
        1. 已知标靶是该坐标系的第一真理源和最大约束，坐标系建立必须严格服从已知点的几何定义；
        2. 提取两标靶在子坐标系中的名义向量 v_child 与在父坐标系中的实测向量 v_parent；
        3. 通过 align_vectors_3d 求解最优 3D 空间刚体旋转矩阵 R in SO(3) (Rodrigues 最小测地线旋转)；
        4. 质心对齐平移 t = centroid_parent - R @ centroid_child；
        5. 垂直于标靶连线方向的残差 (包含局部 Z 轴方向) 严格为 0.00 mm，彻底消灭人为假超差。
        """
        origin_tid = spec.get("origin_tag_id")
        target_tid = spec.get("x_axis_tag_id")
        if origin_tid is None or target_tid is None:
            return False, None, None, None, "双标靶对齐模式必须提供 origin_tag_id 与 x_axis_tag_id"

        pw_o = self.get_tag_world_center(int(origin_tid))
        pw_t = self.get_tag_world_center(int(target_tid))
        if pw_o is None or pw_t is None:
            missing = []
            if pw_o is None:
                missing.append(int(origin_tid))
            if pw_t is None:
                missing.append(int(target_tid))
            return False, None, None, None, f"双标靶在当前地图中未找到: Tag {missing}"

        # 映射至父坐标系
        p_parent_o = (T_parent_from_world @ np.array([pw_o[0], pw_o[1], pw_o[2], 1.0], dtype=np.float64))[:3]
        p_parent_t = (T_parent_from_world @ np.array([pw_t[0], pw_t[1], pw_t[2], 1.0], dtype=np.float64))[:3]

        # 计算父坐标系下的实测向量
        v_parent = p_parent_t - p_parent_o
        dist_meas_3d = float(np.linalg.norm(v_parent))
        if dist_meas_3d < 1e-3:
            return False, None, None, None, "两枚标靶空间实测间距过小 (< 1mm)，无法唯一定向坐标轴"

        # 获取子坐标系下的名义坐标与名义向量 (严格要求规范字段，杜绝隐式实测伪造)
        origin_local = spec.get("origin_local_xyz_mm")
        if origin_local is None or len(origin_local) < 3:
            origin_local = [0.0, 0.0, 0.0]
        p_child_o = np.array(origin_local[:3], dtype=np.float64)

        target_local = spec.get("x_axis_local_xyz_mm")
        if target_local is None or len(target_local) < 3:
            return False, None, None, None, "双标靶外参对齐未提供目标标靶局部标称坐标 (x_axis_local_xyz_mm)，严禁以实测值伪造质检"
        p_child_t = np.array(target_local[:3], dtype=np.float64)
        v_child = p_child_t - p_child_o

        dist_nom_3d = float(np.linalg.norm(v_child))
        if dist_nom_3d < 1e-3:
            return False, None, None, None, "两枚标靶在子系名义坐标下的空间距离过小 (< 1mm)，无法唯一定向坐标轴"

        # 执行 3D 两点刚体空间对齐
        R_mat, t_vec, rmse = two_points_rigid_align_3d(p_child_o, p_child_t, p_parent_o, p_parent_t)
        rpy = rot_mat_to_rpy_deg(R_mat)
        t_xyz = [float(x) for x in t_vec]

        scale_err = abs(dist_meas_3d - dist_nom_3d)

        msg = (
            f"双标靶 3D 空间刚体对齐成功: Tag {origin_tid} 与 Tag {target_tid}，"
            f"实测间距: {dist_meas_3d:.1f} mm (标称: {dist_nom_3d:.1f} mm, 差值: {scale_err:.2f} mm)，"
            f"残差 RMSE: {rmse:.3f} mm, 姿态 RPY: [{rpy[0]:.2f}°, {rpy[1]:.2f}°, {rpy[2]:.2f}°] "
            f"(提示: 绕两靶中心连线轴向之旋转自由度按最小测地线先验约束)"
        )
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
        批量求解坐标系树中所有声明了标定规范的 unknown 坐标系 (树拓扑排序单趟求解，天然支持多级父子继承)
        :param coord_mgr: 工位坐标系树管理器
        :return: {frame_id: {"success": bool, "translation_xyz_mm": ..., "rotation_rpy_deg": ..., "rmse_mm": ..., "message": ...}}
        """
        results: Dict[str, Dict[str, Any]] = {}
        all_frames = coord_mgr.list_frames()
        targets = [f for f in all_frames if f.type == "fixed_transform" and f.status in ("unknown", "partial") and f.calibration_spec]

        if not targets:
            log.info("[FrameExtrinsic] 无待反推的 unknown 坐标系")
            return results

        # 计算每个待求解坐标系到 world 的依赖深度，实现父在前、子在后的拓扑遍历
        def _get_depth(f: FrameDefinition) -> int:
            depth = 0
            cur = f.parent_frame_id
            visited = set()
            while cur and cur != "world" and cur not in visited:
                visited.add(cur)
                p_frame = coord_mgr.get_frame(cur)
                if not p_frame:
                    break
                cur = p_frame.parent_frame_id
                depth += 1
            return depth

        # 按依赖深度由浅入深拓扑排序
        sorted_targets = sorted(targets, key=_get_depth)

        for frame in sorted_targets:
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
            else:
                results[frame.frame_id] = {
                    "success": False,
                    "message": msg,
                }

        return results
