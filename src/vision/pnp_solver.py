# -*- coding: utf-8 -*-
"""
标靶 PnP 空间几何与位姿求解器 (PnpSolver)
========================================
单一职责设计：
  1. 纯几何与位姿求解：多标靶 SQPNP -> ITERATIVE 稳健求解；
  2. 单标靶二义性消歧：IPPE_SQUARE 求解与 180° 法向翻转先验纠偏 (expected_z_cam)；
  3. 提供标靶局部物理角点模型生成与 tags_map 世界变换查询函数；
  4. 绝不掺杂图像检测、GUI 交互或相机设备硬件操作。
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


def make_tag_local_corners(marker_size_mm: float, with_homogeneous: bool = False) -> np.ndarray:
    """
    生成单标靶局部坐标系 4 角点 (顺序: 左上, 右上, 右下, 左下, 逆时针，Z=0)
    :param marker_size_mm: 标靶物理边长 (mm)
    :param with_homogeneous: 是否返回齐次坐标 (4, 4)，默认返回 (4, 3)
    """
    s = float(marker_size_mm) / 2.0
    if with_homogeneous:
        return np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)
    return np.array([
        [-s,  s, 0.0],
        [ s,  s, 0.0],
        [ s, -s, 0.0],
        [-s, -s, 0.0]
    ], dtype=np.float64)


def get_tag_world_transform(tags_map: Dict[str, Any], tag_id: int | str) -> Optional[np.ndarray]:
    """
    获取标靶在世界坐标系下的 4x4 变换矩阵
    :param tags_map: tags_map.yaml 字典数据
    :param tag_id: 标靶 ID (支持 int 或 str)
    :return: 4x4 变换矩阵，若不存在返回 None
    """
    if not tags_map or "tags" not in tags_map:
        return None
    tags_dict = tags_map.get("tags", {})
    tag_data = tags_dict.get(tag_id)
    if tag_data is None:
        tag_data = tags_dict.get(str(tag_id))
    if tag_data is None:
        return None
    if "transform_matrix" in tag_data:
        return np.array(tag_data["transform_matrix"], dtype=np.float64)
    return None


def get_tag_world_corners(
    tags_map: Dict[str, Any],
    tag_id: int | str,
    marker_size_mm: float
) -> Optional[np.ndarray]:
    """
    获取标靶 4 个角点在世界坐标系下的 3D 物理坐标 (4, 3)
    :param tags_map: tags_map.yaml 字典数据
    :param tag_id: 标靶 ID
    :param marker_size_mm: 标靶物理边长 (mm)
    :return: 4x3 角点世界坐标矩阵，若不存在返回 None
    """
    T = get_tag_world_transform(tags_map, tag_id)
    if T is None:
        return None
    local_homo = make_tag_local_corners(marker_size_mm, with_homogeneous=True)
    return (T @ local_homo.T).T[:, :3]


class PnpSolver:
    """标靶空间位姿稳健 PnP 求解器"""

    def __init__(
        self,
        camera_matrix: np.ndarray,
        dist_coeffs: Optional[np.ndarray] = None,
        marker_size_mm: float = 40.0,
        tags_map: Optional[Dict[str, Any]] = None
    ):
        """
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数向量 (若为 None 则默认为 0 畸变)
        :param marker_size_mm: 标靶物理边长 (mm)
        :param tags_map: 可选的 tags_map.yaml 字典 (用于便捷世界坐标查询)
        """
        self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
        self.dist_coeffs = (
            np.zeros((5, 1), dtype=np.float64)
            if dist_coeffs is None
            else np.array(dist_coeffs, dtype=np.float64)
        )
        self.marker_size_mm = float(marker_size_mm)
        self.tags_map = tags_map or {}

    def set_marker_size_mm(self, size_mm: float) -> None:
        """更新标靶物理边长"""
        if size_mm and float(size_mm) > 0:
            self.marker_size_mm = float(size_mm)

    @property
    def obj_points(self) -> np.ndarray:
        """获取当前边长下的标靶局部 4 角点 (4, 3)"""
        return make_tag_local_corners(self.marker_size_mm)

    def get_tag_world_transform(self, tag_id: int | str) -> Optional[np.ndarray]:
        """查询关联地图中指定标靶的 4x4 世界变换矩阵"""
        return get_tag_world_transform(self.tags_map, tag_id)

    def get_tag_world_corners(self, tag_id: int | str) -> Optional[np.ndarray]:
        """查询关联地图中指定标靶的 4 角点世界 3D 坐标 (4, 3)"""
        return get_tag_world_corners(self.tags_map, tag_id, self.marker_size_mm)

    def solve_pnp(
        self,
        obj_flat: np.ndarray,
        img_flat: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], bool]:
        """
        标准两阶段 SQPNP -> ITERATIVE 稳健 PnP 求解
        :param obj_flat: (N, 3) 3D 空间点
        :param img_flat: (N, 2) 2D 像面像点
        :return: (rvec, tvec, success)
        """
        try:
            succ, rvec, tvec = cv2.solvePnP(
                obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP
            )
            if succ:
                succ, rvec, tvec = cv2.solvePnP(
                    obj_flat, img_flat, self.camera_matrix, self.dist_coeffs,
                    rvec=rvec, tvec=tvec, useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
            return rvec, tvec, succ
        except Exception:
            return None, None, False

    def solve_single_tag_pnp(
        self,
        corners_2d: np.ndarray,
        expected_z_cam: Optional[np.ndarray] = None,
        marker_size_mm: Optional[float] = None
    ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        根据单帧检出的 4 个 2D 角点解算单标靶实测相机外参位姿 (优先 IPPE_SQUARE，兜底 ITERATIVE)

        平面靶 PnP 天然存在二义性: IPPE 返回的两个候选解沿靶面内一轴相差约 180°
        (标靶法向/Z 轴翻转), 斜视 (约 45°) 时两解重投影误差之差缩小到像素噪声量级,
        仅按误差择优会间歇性选中翻转解。expected_z_cam 为标靶法向 (Z 轴) 在相机系下的
        先验方向 (来自地图理论位姿或"标靶朝向天空"先验), 提供后剔除法向与先验反向
        (dot<=0) 的翻转解, 再按重投影误差择优; 先验下无同向合格解时拒绝输出 (防错优先)。
        :param corners_2d: (4, 2) 图像角点
        :param expected_z_cam: 标靶法向在相机系下的先验单位方向
        :param marker_size_mm: 可选指定标靶边长，为空时使用实例属性 self.marker_size_mm
        :return: (success, rvec, tvec)
        """
        obj_pts = (
            make_tag_local_corners(marker_size_mm)
            if marker_size_mm and marker_size_mm > 0
            else self.obj_points
        )

        z_exp = None
        if expected_z_cam is not None:
            z_exp = np.asarray(expected_z_cam, dtype=np.float64).reshape(3)
            n = float(np.linalg.norm(z_exp))
            z_exp = z_exp / n if n > 1e-9 else None

        try:
            c = corners_2d.reshape((4, 2)).astype(np.float64)
            succ, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                obj_pts, c, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if succ and len(rvecs) > 0:
                best_r, best_t, min_err = None, None, float("inf")
                prior_r, prior_t, min_err_prior = None, None, float("inf")
                for r, t in zip(rvecs, tvecs):
                    if t[2, 0] <= 0:
                        continue
                    proj, _ = cv2.projectPoints(
                        obj_pts, r, t, self.camera_matrix, self.dist_coeffs
                    )
                    err = np.mean(np.linalg.norm(proj.reshape(-1, 2) - c, axis=1))
                    if err < min_err:
                        min_err = err
                        best_r, best_t = r, t
                    if z_exp is not None:
                        R_c, _ = cv2.Rodrigues(r)
                        if float(R_c[:, 2] @ z_exp) > 0.0 and err < min_err_prior:
                            min_err_prior = err
                            prior_r, prior_t = r, t
                if prior_r is not None:
                    return True, prior_r, prior_t
                if z_exp is None and best_r is not None:
                    return True, best_r, best_t
                # 有先验但候选解全部反向/深度非法: 落入兜底 (兜底同样做先验校验)
        except Exception:
            pass

        try:
            c = corners_2d.reshape((4, 2)).astype(np.float64)
            succ, r, t = cv2.solvePnP(
                obj_pts, c, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if succ and t[2, 0] > 0:
                if z_exp is not None:
                    R_c, _ = cv2.Rodrigues(r)
                    if float(R_c[:, 2] @ z_exp) <= 0.0:
                        return False, None, None   # 与先验反向的翻转解, 拒绝输出
                return True, r, t
        except Exception:
            pass

        return False, None, None
