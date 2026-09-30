#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 16h5 在线实时相机外参定位器 (TagLocalizer)
- 加载已标定建图的 config/tags_map.yaml
- 单帧图像毫秒级提取视野内的 Tag 角点
- 利用 3D-2D 对应执行 cv2.solvePnPRansac() 求解相机在 SCARA 世界系下的 6DoF 位姿
- 支持 Tag 0 (SCARA 旋转中心) 的动态朝向跟踪与角度监控
"""

import os
import yaml
import math
import cv2
import numpy as np
from typing import Dict, Tuple, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)


class TagLocalizer:
    def __init__(self,
                 tags_map_path: str = "config/tags_map.yaml",
                 camera_matrix: Optional[np.ndarray] = None,
                 dist_coeffs: Optional[np.ndarray] = None,
                 *,
                 marker_size_mm: float):
        """
        初始化定位器
        :param tags_map_path: 标靶空间地图配置文件路径
        :param camera_matrix: 3x3 相机内参矩阵
        :param dist_coeffs: 畸变系数
        :param marker_size_mm: 标靶物理边长 (mm)
        """
        self.tags_map_path = tags_map_path
        self.tags_map: Optional[Dict] = None
        self.marker_size_mm = marker_size_mm

        # 相机内参与畸变
        if camera_matrix is None:
            self.camera_matrix = np.array([
                [615.0, 0.0, 320.0],
                [0.0, 615.0, 240.0],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        else:
            self.camera_matrix = np.array(camera_matrix, dtype=np.float64)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float64) if dist_coeffs is None else np.array(dist_coeffs, dtype=np.float64)

        # 加载地图
        self.load_map(tags_map_path)

        # 初始化检测器 (AprilTag 16h5: 统一采用 CONTOUR 消除梯度吸附滑移)
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
        self.detector_params = cv2.aruco.DetectorParameters()
        self.detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.detector_params)

        # 单标靶局部 4 角点物理坐标 (逆时针, Z=0)
        s = self.marker_size_mm / 2.0
        self.local_corners = np.array([
            [-s,  s, 0.0, 1.0],
            [ s,  s, 0.0, 1.0],
            [ s, -s, 0.0, 1.0],
            [-s, -s, 0.0, 1.0]
        ], dtype=np.float64)

    def load_map(self, path: str):
        """加载已建好的标靶地图配置并执行 Schema 安全守门校验"""
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.tags_map = yaml.safe_load(f) or {}
            self.marker_size_mm = float(self.tags_map.get("marker_size_mm", 40.0))
            
            # 严格校验 schema 与 is_dynamic_yaw 标记 (安全守门)
            tags_dict = self.tags_map.get("tags", {})
            origin_id = int(self.tags_map.get("origin_tag_id", 0))
            missing_dynamic_flag = False
            for tid, t_info in list(tags_dict.items()):
                if int(tid) == origin_id:
                    if not t_info.get("is_dynamic_yaw", False):
                        t_info["is_dynamic_yaw"] = True
                        missing_dynamic_flag = True
            if missing_dynamic_flag:
                log.warning(f"[TagLocalizer] [WARN] 地图缺少 is_dynamic_yaw 标志，已自动强制将原点标靶 Tag {origin_id} 标记为动态 (禁止参与 PnP 外参求解)！")
            # 锚定模式溯源: partial 模式地图的 Z 轴为相对坐标, 运行时必须守门
            self.anchor_mode = str(self.tags_map.get("anchor_mode", "") or "")
            if self.anchor_mode:
                log.info(f"[TagLocalizer] 地图锚定模式: {self.anchor_mode}")
                if self.anchor_mode == "partial":
                    log.warning("[TagLocalizer] [WARN] 地图为 partial 锚定 (XY 绝对/Z 相对), 世界系位姿 Z 分量不可信！")
            log.info(f"[TagLocalizer] 成功加载标靶立体地图: {path} (包含 {len(tags_dict)} 个标靶)")
        else:
            self.tags_map = None

    def localize_camera(self, image: np.ndarray) -> Tuple[bool, Optional[np.ndarray], Dict]:
        """
        根据当前帧图像中的标靶解算相机在 SCARA 世界系下的 6DoF 外参
        :param image: BGR 或灰度图像
        :return: (success, T_cam_to_world, info_dict)
        """
        info = {
            "detected_tag_ids": [],
            "static_tags_count": 0,
            "reprojection_error_px": 0.0,
            "inlier_count": 0,
            "tag0_dynamic_yaw_deg": None
        }

        if self.tags_map is None:
            return False, None, info

        # 下游守门: partial 锚定地图 (XY 绝对/Z 相对) 解算的世界系位姿 Z 分量不可信, 拒绝输出
        # (每次调用实时读取, 覆盖 tags_map 就地热更新场景)
        anchor_mode = (self.tags_map.get("anchor_mode")
                       or (self.tags_map.get("world_anchor") or {}).get("anchor_mode")
                       or "")
        if anchor_mode == "partial":
            info["anchor_mode"] = "partial"
            info["reject_reason"] = "partial_anchor_map_z_relative"
            log.warning("[TagLocalizer] [WARN] partial 锚定地图 (Z 相对), 拒绝输出世界系相机位姿！请补全锚点 Z 约束后重建地图。")
            return False, None, info

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return False, None, info

        detected_ids = [int(i) for i in ids.flatten()]
        info["detected_tag_ids"] = detected_ids

        # 收集 3D-2D 对应点对 (严格使用静止标靶解算相机绝对外参)
        object_points_3d = []
        image_points_2d = []
        static_tags_map = self.tags_map.get("tags", {})
        origin_tid = int(self.tags_map.get("origin_tag_id", 0))

        for idx, tag_id in enumerate(detected_ids):
            # 核心安全守门：原点标靶 (Tag 0) 随机械臂旋转，双重铁律排除在静态 PnP 之外
            if tag_id == origin_tid:
                continue
            # 如果是已知静止标靶 (必须确认 is_dynamic_yaw 不为 True)
            if tag_id in static_tags_map and not static_tags_map[tag_id].get("is_dynamic_yaw", False):
                T_w_t = np.array(static_tags_map[tag_id]["transform_matrix"], dtype=np.float64)
                # 计算 4 个角点在世界系下的绝对 3D 坐标: P_w = T_w_t @ P_local
                for k in range(4):
                    p_local = self.local_corners[k]
                    p_w = (T_w_t @ p_local)[:3]
                    object_points_3d.append(p_w)
                    image_points_2d.append(corners[idx][0, k])
                info["static_tags_count"] += 1

        # 至少需要 2 个静止 Tag (8 个空间点对) 才能获得极高精度的非共面 PnP 解
        if len(object_points_3d) < 8:
            return False, None, info

        obj_pts = np.array(object_points_3d, dtype=np.float64)
        img_pts = np.array(image_points_2d, dtype=np.float64)

        # 运行 RANSAC-PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
            reprojectionError=2.0, flags=cv2.SOLVEPNP_SQPNP
        )

        if not success or rvec is None:
            # 降级尝试常规 PnP
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, self.camera_matrix, self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

        if not success:
            return False, None, info

        # T_c_w: 世界点到相机系
        R_c_w, _ = cv2.Rodrigues(rvec)
        T_c_w = np.eye(4, dtype=np.float64)
        T_c_w[:3, :3] = R_c_w
        T_c_w[:3, 3] = tvec.flatten()

        # T_w_c: 相机在世界坐标系下的绝对位姿 (相机外参)
        T_w_c = np.linalg.inv(T_c_w)
        # T_cam_to_world: 即相机系下的点变换到 SCARA 世界坐标系下的齐次矩阵
        T_cam_to_world = T_w_c

        # 计算残差
        proj_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs)
        proj_pts = proj_pts.reshape((-1, 2))
        err = np.sqrt(np.mean((proj_pts - img_pts) ** 2))
        info["reprojection_error_px"] = float(err)
        info["inlier_count"] = len(inliers) if inliers is not None else len(obj_pts)

        # 若当前视野也看到了 Tag 0，计算 Tag 0 当前的绕 Z 轴动态转角 (Yaw)
        if 0 in detected_ids:
            t0_idx = detected_ids.index(0)
            t0_corners = corners[t0_idx].reshape((4, 2))
            s_half = self.marker_size_mm / 2.0
            t0_local = np.array([[-s_half, s_half, 0], [s_half, s_half, 0], [s_half, -s_half, 0], [-s_half, -s_half, 0]], dtype=np.float64)
            succ0, rv0, tv0 = cv2.solvePnP(t0_local, t0_corners, self.camera_matrix, self.dist_coeffs)
            if succ0:
                R0_c, _ = cv2.Rodrigues(rv0)
                T_c_t0 = np.eye(4)
                T_c_t0[:3, :3] = R0_c
                T_c_t0[:3, 3] = tv0.flatten()
                # T_w_t0 = T_w_c @ T_c_t0
                T_w_t0 = T_w_c @ T_c_t0
                R_w_t0 = T_w_t0[:3, :3]
                yaw0 = math.atan2(R_w_t0[1, 0], R_w_t0[0, 0])
                info["tag0_dynamic_yaw_deg"] = round(float(math.degrees(yaw0)), 2)

        return True, T_cam_to_world, info


if __name__ == "__main__":
    localizer = TagLocalizer(marker_size_mm=50.0)
    log.info("[OK] TagLocalizer 模块语法与加载测试通过！")
