"""
工位 ROI 空间物件集合管理模块 (ROI Space Manager)
=================================================
负责管理工位内的 3D ROI 空间物件 (有向长方体 OBB 集合)：
1. 每个 ROI 挂载在指定的坐标系 (world 或任意相对坐标系) 下；
2. 描述物理部件 (同步带、输送带、驱动轮、托盘检测区等)；
3. 依托 CoordinateTreeManager 求解世界坐标系下的 8 顶点 OBB 与姿态；
4. 提供工业级高效率点云空间裁剪与过滤接口。
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

from src.calibration.coordinate_manager import CoordinateTreeManager, rpy_deg_to_rot_mat
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class RoiDefinition:
    """单个 3D ROI 物件数据模型"""
    roi_id: str                                  # 唯一标识，如 roi_sync_belt_01, roi_drive_wheel
    name: str                                    # 友好名称，如 1号同步带工作面
    frame_id: str = "world"                      # 依附的坐标系 ID
    category: str = "general"                    # 类别: belt(皮带), wheel(轮子), tray(托盘), general(通用)
    enabled: bool = True                         # 是否使能
    description: str = ""                        # 说明备注

    # 3D 几何长方体参数 (在其依附的 frame 局部系内定义)
    center_xyz_mm: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    size_xyz_mm: List[float] = field(default_factory=lambda: [100.0, 100.0, 50.0])   # dx, dy, dz
    rotation_rpy_deg: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 局部微调欧拉角

    # 3D 渲染颜色 [R, G, B] 0~255
    visual_color_rgb: List[int] = field(default_factory=lambda: [0, 255, 128])

    # Smart ROI 生产意图与工艺角色扩展
    role: str = "general"                        # 角色: source(来料抓取源), destination(落料目标), keepout(防撞禁区), general(通用)
    target_intent: str = "general"               # 感知意图: pose_pick(位姿抓取), piece_count(根数统计), occupancy(有无检测), general(常规)
    binding: Dict[str, Any] = field(default_factory=dict)  # 通信映射: 如 {"channel": "mqtt_loader", "slot_index": 0, "capacity_max": 5}
    pipeline_override: str = ""                  # 定制视觉流水线算法 key (若为空则沿用工位全局默认流水线)
    min_confidence: float = 0.5                  # 检出置信度门限

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典供 YAML 存储"""
        d = {
            "roi_id": self.roi_id,
            "name": self.name,
            "frame_id": self.frame_id,
            "category": self.category,
            "enabled": self.enabled,
            "description": self.description,
            "geometry": {
                "type": "box",
                "center_xyz_mm": [float(x) for x in self.center_xyz_mm],
                "size_xyz_mm": [float(x) for x in self.size_xyz_mm],
                "rotation_rpy_deg": [float(x) for x in self.rotation_rpy_deg],
            },
            "visual_color_rgb": [int(c) for c in self.visual_color_rgb],
            "role": self.role,
            "target_intent": self.target_intent,
        }
        if self.binding:
            d["binding"] = dict(self.binding)
        if self.pipeline_override:
            d["pipeline_override"] = self.pipeline_override
        if self.min_confidence != 0.5:
            d["min_confidence"] = float(self.min_confidence)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoiDefinition":
        """从字典反序列化"""
        geom = data.get("geometry", {})
        c_xyz = geom.get("center_xyz_mm", [0.0, 0.0, 0.0])
        s_xyz = geom.get("size_xyz_mm", [100.0, 100.0, 50.0])
        r_rpy = geom.get("rotation_rpy_deg", [0.0, 0.0, 0.0])

        color = data.get("visual_color_rgb", [0, 255, 128])

        return cls(
            roi_id=data.get("roi_id", ""),
            name=data.get("name", ""),
            frame_id=data.get("frame_id", "world"),
            category=data.get("category", "general"),
            enabled=data.get("enabled", True),
            description=data.get("description", ""),
            center_xyz_mm=c_xyz,
            size_xyz_mm=s_xyz,
            rotation_rpy_deg=r_rpy,
            visual_color_rgb=color,
            role=data.get("role", "general"),
            target_intent=data.get("target_intent", "general"),
            binding=data.get("binding", {}),
            pipeline_override=data.get("pipeline_override", ""),
            min_confidence=float(data.get("min_confidence", 0.5)),
        )


class RoiSpaceManager:
    """工位 ROI 空间物件集合管理器"""

    def __init__(self, workspace_id: str = "", spatial_scene_path: Optional[str] = None):
        self.workspace_id = workspace_id
        self.spatial_scene_path = spatial_scene_path
        self._rois: Dict[str, RoiDefinition] = {}

        if spatial_scene_path and os.path.exists(spatial_scene_path):
            self.load()

    def add_roi(self, roi: RoiDefinition):
        """添加或更新 ROI 物件"""
        self._rois[roi.roi_id] = roi

    def remove_roi(self, roi_id: str) -> bool:
        """删除指定 ROI 物件"""
        if roi_id in self._rois:
            del self._rois[roi_id]
            return True
        return False

    def rename_roi(self, old_roi_id: str, new_roi_id: str) -> bool:
        """重命名指定 ROI 物件"""
        if not new_roi_id or not new_roi_id.strip():
            log.warning("[RoiSpace] 新 ROI ID 不能为空！")
            return False
        new_roi_id = new_roi_id.strip()
        if old_roi_id == new_roi_id:
            return True
        if old_roi_id not in self._rois:
            log.warning(f"[RoiSpace] 原 ROI {old_roi_id} 不存在！")
            return False
        if new_roi_id in self._rois:
            log.warning(f"[RoiSpace] 目标 ROI ID {new_roi_id} 已存在，不可重复！")
            return False

        roi = self._rois.pop(old_roi_id)
        roi.roi_id = new_roi_id
        self._rois[new_roi_id] = roi
        return True

    def get_roi(self, roi_id: str) -> Optional[RoiDefinition]:
        return self._rois.get(roi_id)

    def list_rois(self) -> List[RoiDefinition]:
        return list(self._rois.values())

    def get_roi_local_corners(self, roi: RoiDefinition) -> np.ndarray:
        """
        求解 ROI 在其所属局部坐标系 (frame) 下的 8 个角点坐标 (8, 3)。
        角点顺序:
          0: [-dx/2, -dy/2, -dz/2]
          1: [+dx/2, -dy/2, -dz/2]
          2: [+dx/2, +dy/2, -dz/2]
          3: [-dx/2, +dy/2, -dz/2]
          4: [-dx/2, -dy/2, +dz/2]
          5: [+dx/2, -dy/2, +dz/2]
          6: [+dx/2, +dy/2, +dz/2]
          7: [-dx/2, +dy/2, +dz/2]
        """
        dx, dy, dz = roi.size_xyz_mm[0] / 2.0, roi.size_xyz_mm[1] / 2.0, roi.size_xyz_mm[2] / 2.0
        base_corners = np.array([
            [-dx, -dy, -dz],
            [ dx, -dy, -dz],
            [ dx,  dy, -dz],
            [-dx,  dy, -dz],
            [-dx, -dy,  dz],
            [ dx, -dy,  dz],
            [ dx,  dy,  dz],
            [-dx,  dy,  dz],
        ], dtype=np.float64)

        # 局部微调旋转
        R_local = rpy_deg_to_rot_mat(roi.rotation_rpy_deg)
        rotated = (R_local @ base_corners.T).T
        center = np.array(roi.center_xyz_mm, dtype=np.float64)
        return rotated + center

    def get_roi_world_obb(self, roi_id: str, coord_mgr: CoordinateTreeManager) -> Optional[Dict[str, Any]]:
        """
        求解该 ROI 在世界坐标系下的有向包围盒 (OBB):
        返回字典:
          - corners_8x3: 世界坐标系下的 8 个角点 (8, 3) np.ndarray
          - center_world: 空间中心点世界坐标 (3,)
          - axes_world: 3 个主轴世界方向向量 (3, 3) [axis_x, axis_y, axis_z]
          - size_xyz_mm: 尺寸 [dx, dy, dz]
          - is_resolved: 依附坐标系是否有效就绪
        """
        roi = self._rois.get(roi_id)
        if not roi:
            return None

        # 1. 求解在所属 frame 下的 8 角点
        local_corners = self.get_roi_local_corners(roi)

        # 2. 变换到世界系
        T_world_from_frame, is_resolved = coord_mgr.get_frame_to_world(roi.frame_id)

        # 转换 8 角点到世界系
        homo_corners = np.hstack([local_corners, np.ones((8, 1), dtype=np.float64)])
        world_corners = (T_world_from_frame @ homo_corners.T).T[:, :3]

        # 转换中心点
        local_center = np.array(roi.center_xyz_mm, dtype=np.float64)
        center_homo = np.array([local_center[0], local_center[1], local_center[2], 1.0])
        world_center = (T_world_from_frame @ center_homo)[:3]

        # 转换主轴向量 (旋转矩阵 R_world_from_frame @ R_local)
        R_local = rpy_deg_to_rot_mat(roi.rotation_rpy_deg)
        R_world_axes = T_world_from_frame[:3, :3] @ R_local

        return {
            "roi_id": roi.roi_id,
            "name": roi.name,
            "category": roi.category,
            "enabled": roi.enabled,
            "corners_8x3": world_corners,
            "center_world": world_center,
            "axes_world": R_world_axes,
            "size_xyz_mm": roi.size_xyz_mm,
            "visual_color_rgb": roi.visual_color_rgb,
            "is_resolved": is_resolved,
        }

    def get_point_cloud_mask(
        self,
        points_world: np.ndarray,
        roi_id: str,
        coord_mgr: CoordinateTreeManager
    ) -> np.ndarray:
        """
        判断世界坐标系点云 points_world (N, 3) 是否位于该 ROI 长方体内部。
        返回布尔掩码 (N,)，True 表示落入该 ROI。
        """
        roi = self._rois.get(roi_id)
        if roi is None or points_world is None or len(points_world) == 0:
            return np.zeros(0 if points_world is None else len(points_world), dtype=bool)

        # 1. 将世界坐标系点云先变换到该 ROI 所属的 frame 局部系
        points_frame = coord_mgr.transform_points(points_world, "world", roi.frame_id)

        # 2. 将点减去 ROI 中心，并逆旋转消除 ROI 本身的微调姿态
        center = np.array(roi.center_xyz_mm, dtype=np.float64)
        pts_centered = points_frame - center

        R_local = rpy_deg_to_rot_mat(roi.rotation_rpy_deg)
        # 逆旋转: R.T @ pts.T
        pts_canonical = (R_local.T @ pts_centered.T).T

        # 3. 标准 AABB 区间判定: [-dx/2, dx/2], [-dy/2, dy/2], [-dz/2, dz/2]
        dx, dy, dz = roi.size_xyz_mm[0] / 2.0, roi.size_xyz_mm[1] / 2.0, roi.size_xyz_mm[2] / 2.0
        mask_x = (pts_canonical[:, 0] >= -dx) & (pts_canonical[:, 0] <= dx)
        mask_y = (pts_canonical[:, 1] >= -dy) & (pts_canonical[:, 1] <= dy)
        mask_z = (pts_canonical[:, 2] >= -dz) & (pts_canonical[:, 2] <= dz)

        return mask_x & mask_y & mask_z

    def filter_point_cloud(
        self,
        points_world: np.ndarray,
        roi_id: str,
        coord_mgr: CoordinateTreeManager,
        keep_inside: bool = True
    ) -> np.ndarray:
        """
        依据 ROI 空间裁剪世界坐标点云:
        - keep_inside=True: 保留落在 ROI 内的点 (裁剪提取特定部件表面点)
        - keep_inside=False: 排除落在 ROI 内的点 (剔除背景/机架干涉点)
        """
        if points_world is None or len(points_world) == 0:
            return points_world

        mask = self.get_point_cloud_mask(points_world, roi_id, coord_mgr)
        if keep_inside:
            return points_world[mask]
        else:
            return points_world[~mask]

    def load(self, filepath: Optional[str] = None) -> bool:
        """从 spatial_scene.yaml 文件加载 ROI 集合"""
        path = filepath or self.spatial_scene_path
        if not path or not os.path.exists(path):
            self._rois = {}
            return False

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            self.workspace_id = data.get("workspace_id", self.workspace_id)
            raw_rois = data.get("rois", [])

            loaded = {}
            for item in raw_rois:
                roi = RoiDefinition.from_dict(item)
                loaded[roi.roi_id] = roi

            self._rois = loaded
            log.debug(f"[RoiSpace] 成功从空间场景加载 {len(self._rois)} 个 ROI 物件: {path}")
            return True
        except Exception as e:
            log.error(f"[RoiSpace] 加载空间场景 ROI 集合失败 ({path}): {e}")
            self._rois = {}
            return False

    def save(self, filepath: Optional[str] = None) -> bool:
        """持久化保存至 spatial_scene.yaml 文件 (原子保留 frames 节)"""
        path = filepath or self.spatial_scene_path
        if not path:
            return False

        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            existing_data = {}
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        existing_data = yaml.safe_load(f) or {}
                except Exception:
                    existing_data = {}

            existing_data["version"] = "2.0"
            existing_data["workspace_id"] = self.workspace_id
            if "active_frame_id" not in existing_data:
                existing_data["active_frame_id"] = "world"
            if "frames" not in existing_data:
                existing_data["frames"] = []
            existing_data["rois"] = [r.to_dict() for r in self.list_rois()]

            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(existing_data, f, allow_unicode=True, sort_keys=False, indent=2)
            log.info(f"[RoiSpace] 成功写穿 ROI 集合 ({len(self._rois)} 个) 至空间场景: {path}")
            return True
        except Exception as e:
            log.error(f"[RoiSpace] 保存空间场景 ROI 集合失败 ({path}): {e}")
            return False
