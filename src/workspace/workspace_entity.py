# -*- coding: utf-8 -*-
"""
工位自包含领域实体模型 (Workspace Entity)
========================================
定义单一工位 (Workspace) 的领域模型、全量物理资产路径契约、
相机硬件与分辨率真理源规格、指标统计与持久化。
"""

import os
import glob
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple, TYPE_CHECKING
import yaml
import numpy as np

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.workspace.coordinate_manager import CoordinateTreeManager
    from src.workspace.roi_manager import RoiSpaceManager

log = get_logger(__name__)


@dataclass
class Workspace:
    """单个工位工作空间自包含数据模型"""
    workspace_id: str                      # 工位唯一标识 (目录名，如 20260915_bench_default)
    name: str                              # 友好别名 (如 bench_default, 现场工位A)
    workspace_dir: str                     # 工位物理根目录绝对路径
    description: str = ""                  # 工位说明备注
    created_at: str = ""                   # 创建时间
    updated_at: str = ""                   # 最后更新时间
    camera_type: str = "realsense"         # 绑定相机类型: 'realsense' (默认), 'usb', 'mock'
    camera_serial: str = ""                # 采集相机硬件序列号或设备索引
    valid_tag_ids: List[int] = field(default_factory=list)
    origin_tag_id: int = 0
    x_axis_tag_id: int = 28
    camera_resolution: List[int] = field(default_factory=lambda: [1920, 1080])  # 工位锁定的标准相机物理分辨率 [宽, 高]

    # 状态与指标
    extrinsic_calib_image_count: int = 0   # 外参建图图片数
    prod_image_count: int = 0              # 生产采图数
    intrinsics_image_count: int = 0        # 相机内参标定图片数
    active_image_count: int = 0            # 标定有效帧数
    ba_solved: bool = False
    global_rmse_px: float = 0.0

    # 全局生产工作流与驱动规范
    production: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------ 顶层核心资产路径 ------------------------------
    @property
    def map_path(self) -> str:
        """【工位核心资产】空间立体几何地图路径 (工位顶层，供生产/标定消费)"""
        return os.path.join(self.workspace_dir, "tags_map.yaml")

    @property
    def whitelist_path(self) -> str:
        """【工位核心资产】本工位合法 AprilTag 白名单文件"""
        return os.path.join(self.workspace_dir, "tag_whitelist.yaml")

    @property
    def spatial_scene_path(self) -> str:
        """【工位核心资产】工位空间几何场景与 3D 物件大一统配置文件 (spatial_scene.yaml)"""
        return os.path.join(self.workspace_dir, "spatial_scene.yaml")

    @property
    def meta_path(self) -> str:
        """工位自描述元数据路径"""
        return os.path.join(self.workspace_dir, "workspace_meta.yaml")

    # ------------------------------ 工位分辨率与内参规范 (SSOT) ------------------------------
    @property
    def resolution(self) -> Tuple[int, int]:
        """【工位核心契约】相机标准物理分辨率 (宽, 高)，优先内参规格，其次元数据，缺省 1920x1080"""
        intr = self.load_camera_intrinsics()
        if intr and intr.get("width") and intr.get("height"):
            return int(intr["width"]), int(intr["height"])
        if self.camera_resolution and len(self.camera_resolution) >= 2:
            return int(self.camera_resolution[0]), int(self.camera_resolution[1])
        return 1920, 1080

    @property
    def resolution_str(self) -> str:
        """以标准字符串返回工位锁定分辨率，如 '1920x1080'"""
        w, h = self.resolution
        return f"{w}x{h}"

    def validate_image_resolution(self, img_w: int, img_h: int) -> Tuple[bool, str]:
        """防呆校验传入图像是否与工位锁定的物理分辨率一致"""
        target_w, target_h = self.resolution
        if (img_w, img_h) != (target_w, target_h):
            return False, f"图像物理分辨率 ({img_w}x{img_h}) 与工位锁定规格 ({target_w}x{target_h}) 严重失配，已被系统拒绝！"
        return True, ""

    @property
    def total_image_count(self) -> int:
        """工位全量业务照片总数 (基于内存缓存指标: 内参 + 外参 + 生产)"""
        return self.intrinsics_image_count + self.extrinsic_calib_image_count + self.prod_image_count

    @property
    def is_hardware_locked(self) -> bool:
        """工位是否已锁定硬件配置 (已有历史照片保护，杜绝跨分辨率或换机污染)"""
        return self.total_image_count > 0

    # ------------------------------ 相机内参标定专区 (intrinsics/) ------------------------------
    @property
    def intrinsics_dir(self) -> str:
        """【图集一】相机内参标定业务专区根目录"""
        return os.path.join(self.workspace_dir, "intrinsics")

    @property
    def intrinsics_raw_images_dir(self) -> str:
        """【图集一】相机内参标定原始照片 (棋盘格/圆形标定板) 存储目录"""
        return os.path.join(self.intrinsics_dir, "raw_images")

    @property
    def intrinsics_reports_dir(self) -> str:
        """【图集一】相机内参标定质量评估与残差报告目录"""
        return os.path.join(self.intrinsics_dir, "reports")

    @property
    def camera_intrinsics_path(self) -> str:
        """【工位专属】相机内参文件路径 (单一真理源: intrinsics/camera_intrinsics.yaml)"""
        return os.path.join(self.intrinsics_dir, "camera_intrinsics.yaml")

    def load_camera_intrinsics(self) -> Optional[Dict[str, Any]]:
        """加载本工位专属相机内参配置字典，不存在则返回 None"""
        p = self.camera_intrinsics_path
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                log.warning(f"[Workspace] 读取工位内参异常 ({p}): {e}")
        return None

    def save_camera_intrinsics(
        self,
        data: Optional[Dict[str, Any]] = None,
        *,
        fx: Optional[float] = None,
        fy: Optional[float] = None,
        cx: Optional[float] = None,
        cy: Optional[float] = None,
        dist_coeffs: Optional[List[float]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        rmse: Optional[float] = None,
    ) -> bool:
        """将高精相机内参写穿持久化至本工位沙盒中"""
        payload = dict(data) if data else {}
        if fx is not None:
            payload["fx"] = float(fx)
        if fy is not None:
            payload["fy"] = float(fy)
        if cx is not None:
            payload["cx"] = float(cx)
        if cy is not None:
            payload["cy"] = float(cy)
        if dist_coeffs is not None:
            payload["dist_coeffs"] = [float(x) for x in dist_coeffs]
        if image_size is not None:
            payload["width"] = int(image_size[0])
            payload["height"] = int(image_size[1])
        if rmse is not None:
            payload["rmse"] = float(rmse)
        payload["calibrated"] = True
        payload["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        p = self.camera_intrinsics_path
        os.makedirs(os.path.dirname(p), exist_ok=True)
        try:
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
            log.info(f"[Workspace] 工位专属相机内参已写穿保存: {p}")
            return True
        except Exception as e:
            log.error(f"[Workspace] 保存工位内参失败 ({p}): {e}")
            return False

    # ------------------------------ 外参建图标定专区 (calibration/) ------------------------------
    @property
    def calibration_dir(self) -> str:
        """【图集二】外参建图与空间标定业务专区根目录"""
        return os.path.join(self.workspace_dir, "calibration")

    @property
    def calib_raw_images_dir(self) -> str:
        """【图集二】外参建图原始图像存储目录 (Tag 空间标靶照片)"""
        return os.path.join(self.calibration_dir, "raw_images")

    @property
    def calib_manifest_path(self) -> str:
        """标定观测清单 (tag_observations.yaml) 路径"""
        return os.path.join(self.calibration_dir, "tag_observations.yaml")

    @property
    def calib_reports_dir(self) -> str:
        """标定质检单与盲测体检报告目录"""
        return os.path.join(self.calibration_dir, "reports")

    @property
    def calib_visualized_dir(self) -> str:
        """标定残差场与特征图示化标注目录"""
        return os.path.join(self.calibration_dir, "visualized")

    @property
    def raw_map_path(self) -> str:
        """【标定业务资产】阶段一自由平差相对几何底图缓存 (位于 calibration/ 下)"""
        return os.path.join(self.calibration_dir, "tags_map_raw.yaml")

    # ------------------------------ 生产业务专区 (production/) ------------------------------
    @property
    def production_dir(self) -> str:
        """【图集三】生产业务专区根目录"""
        return os.path.join(self.workspace_dir, "production")

    @property
    def prod_raw_images_dir(self) -> str:
        """【图集三】生产现场原始采样图像存储目录"""
        return os.path.join(self.production_dir, "raw_images")

    @property
    def prod_reports_dir(self) -> str:
        """生产执行质检单与在线盲测报告目录"""
        return os.path.join(self.production_dir, "reports")

    @property
    def prod_results_dir(self) -> str:
        """生产分拣位姿、点云或中间诊断数据目录"""
        return os.path.join(self.production_dir, "results")

    # ------------------------------ 多用途通用辅助 ------------------------------
    def get_raw_images_dir(self, purpose: str = "calibration") -> str:
        """根据用途获取对应原始相册目录 ('intrinsics' | 'calibration' | 'production')"""
        if purpose == "intrinsics":
            return self.intrinsics_raw_images_dir
        elif purpose == "production":
            return self.prod_raw_images_dir
        return self.calib_raw_images_dir

    def get_image_count(self, purpose: str = "calibration") -> int:
        """统计指定用途图集的图片数量"""
        d = self.get_raw_images_dir(purpose)
        if not os.path.isdir(d):
            return 0
        img_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        count = 0
        for ext in img_exts:
            count += len(glob.glob(os.path.join(d, ext)))
        return count

    def ensure_directories(self):
        """确保工位顶层及三大图集核心子目录就绪"""
        os.makedirs(self.workspace_dir, exist_ok=True)
        os.makedirs(self.intrinsics_raw_images_dir, exist_ok=True)
        os.makedirs(self.intrinsics_reports_dir, exist_ok=True)
        os.makedirs(self.calib_raw_images_dir, exist_ok=True)
        os.makedirs(self.calib_reports_dir, exist_ok=True)
        os.makedirs(self.calib_visualized_dir, exist_ok=True)
        os.makedirs(self.prod_raw_images_dir, exist_ok=True)

        # 确保工位核心资产文件就绪 (单一真理源自愈)
        if not os.path.exists(self.whitelist_path):
            try:
                with open(self.whitelist_path, "w", encoding="utf-8") as f:
                    yaml.dump({
                        "workspace_id": self.workspace_id,
                        "workspace_name": self.name,
                        "tag_default_size_mm": 40.0,
                        "tags": {},
                        "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义): tags 字典留空 = 探索模式放行所有检测标靶；非空时仅放行字典内标靶",
                    }, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            except Exception as e:
                log.warning(f"创建默认 tag_whitelist.yaml 失败: {e}")

        if not os.path.exists(self.spatial_scene_path):
            try:
                from src.workspace.coordinate_manager import CoordinateTreeManager
                coord_mgr = CoordinateTreeManager(workspace_id=self.workspace_id, spatial_scene_path=self.spatial_scene_path)
                coord_mgr.save()
            except Exception as e:
                log.warning(f"创建默认 spatial_scene.yaml 失败: {e}")

    def refresh_stats(self):
        """快速刷新物理磁盘状态并持久化至元数据缓存"""
        self.ensure_directories()

        img_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        # 1. 标定原始图片数
        calib_files = []
        for ext in img_exts:
            calib_files.extend(glob.glob(os.path.join(self.calib_raw_images_dir, ext)))
        self.extrinsic_calib_image_count = len(calib_files)

        # 2. 生产采样图片数
        prod_files = []
        for ext in img_exts:
            prod_files.extend(glob.glob(os.path.join(self.prod_raw_images_dir, ext)))
        self.prod_image_count = len(prod_files)

        # 3. 相机内参标定图片数
        intr_files = []
        for ext in img_exts:
            intr_files.extend(glob.glob(os.path.join(self.intrinsics_raw_images_dir, ext)))
        self.intrinsics_image_count = len(intr_files)

        # 4. 标定观测清单解析
        if os.path.exists(self.calib_manifest_path):
            try:
                with open(self.calib_manifest_path, "r", encoding="utf-8") as f:
                    obs_data = yaml.safe_load(f) or {}
                images_dict = obs_data.get("images", {})
                active_count = 0
                seen_tags = set()
                for img in images_dict.values():
                    tags_list = img.get("observations") or []
                    if tags_list:
                        active_count += 1
                        for t in tags_list:
                            tag_id = t.get("tag_id")
                            if tag_id is not None:
                                seen_tags.add(int(tag_id))
                self.active_image_count = active_count
                self.valid_tag_ids = sorted(list(seen_tags))
            except Exception as e:
                log.debug(f"[WS] 解析标定清单失败 {self.workspace_id}: {e}")
                self.active_image_count = 0
        else:
            self.active_image_count = 0
            self.valid_tag_ids = []

        # 5. BA 平差解算状态及指标
        if os.path.exists(self.map_path) and os.path.getsize(self.map_path) > 50:
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    map_data = yaml.safe_load(f) or {}
                meta = map_data.get("meta", {})
                tags_dict = map_data.get("tags", {})

                rmse = (
                    map_data.get("rmse_reprojection_px")
                    or meta.get("rmse_reprojection_px")
                    or 0.0
                )
                self.ba_solved = bool(len(tags_dict) > 0)
                self.global_rmse_px = float(rmse) if rmse is not None else 0.0

                if self.ba_solved and not self.valid_tag_ids:
                    self.valid_tag_ids = sorted([int(tid) for tid in tags_dict.keys()])
            except Exception as e:
                log.debug(f"[WS] 解析 tags_map 失败 {self.workspace_id}: {e}")
                self.ba_solved = False
                self.global_rmse_px = 0.0
        else:
            self.ba_solved = False
            self.global_rmse_px = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """将工位实体转换为序列化字典"""
        return {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": self.updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_type": self.camera_type,
            "camera_serial": self.camera_serial,
            "camera_resolution": list(self.resolution),
            "status": {
                "extrinsic_calib_image_count": self.extrinsic_calib_image_count,
                "prod_image_count": self.prod_image_count,
                "intrinsics_image_count": self.intrinsics_image_count,
                "active_image_count": self.active_image_count,
                "ba_solved": self.ba_solved,
                "global_rmse_px": self.global_rmse_px,
                "valid_tag_ids": self.valid_tag_ids,
                "origin_tag_id": self.origin_tag_id,
                "x_axis_tag_id": self.x_axis_tag_id,
            },
            "production": self.production,
        }

    def save_meta(self):
        """保存元数据到工位根目录"""
        self.ensure_directories()
        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                yaml.dump(self.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            log.error(f"保存工位元数据失败: {self.meta_path}, err: {e}")

    @classmethod
    def load(cls, workspace_dir: str, force_refresh: bool = False) -> Optional["Workspace"]:
        """从指定物理路径加载工位实体"""
        ws_id = os.path.basename(os.path.normpath(workspace_dir))
        meta_path = os.path.join(workspace_dir, "workspace_meta.yaml")
        manifest_path = os.path.join(workspace_dir, "calibration", "tag_observations.yaml")

        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
            except Exception as e:
                log.warning(f"读取工位元数据异常: {meta_path}, err: {e}")

        alias = meta.get("name", ws_id)
        desc = meta.get("description", "")
        created_at = meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        updated_at = meta.get("updated_at", "")
        camera_type = meta.get("camera_type", "realsense")
        camera_serial = meta.get("camera_serial", "")
        camera_resolution = meta.get("camera_resolution", [1920, 1080])
        production = meta.get("production", {})

        status = meta.get("status", {})
        extrinsic_calib_image_count = status.get("extrinsic_calib_image_count", 0)
        prod_image_count = status.get("prod_image_count", 0)
        intrinsics_image_count = status.get("intrinsics_image_count", 0)
        active_image_count = status.get("active_image_count", 0)
        ba_solved = status.get("ba_solved", False)
        global_rmse_px = status.get("global_rmse_px", 0.0)
        valid_tag_ids = status.get("valid_tag_ids", [])
        origin_tag_id = status.get("origin_tag_id", 0)
        x_axis_tag_id = status.get("x_axis_tag_id", 28)

        ws = cls(
            workspace_id=ws_id,
            name=alias,
            workspace_dir=workspace_dir,
            description=desc,
            created_at=created_at,
            updated_at=updated_at,
            camera_type=camera_type,
            camera_serial=camera_serial,
            camera_resolution=camera_resolution,
            extrinsic_calib_image_count=extrinsic_calib_image_count,
            prod_image_count=prod_image_count,
            intrinsics_image_count=intrinsics_image_count,
            active_image_count=active_image_count,
            ba_solved=ba_solved,
            global_rmse_px=global_rmse_px,
            valid_tag_ids=valid_tag_ids,
            origin_tag_id=origin_tag_id,
            x_axis_tag_id=x_axis_tag_id,
            production=production,
        )

        needs_heal = (
            force_refresh
            or not status
            or (ba_solved and global_rmse_px <= 1e-6)
            or (active_image_count == 0 and os.path.exists(manifest_path))
        )
        if needs_heal:
            ws.refresh_stats()
            ws.save_meta()

        return ws

    # ------------------------------ 领域组件对象访问器 (Domain Accessors) ------------------------------
    def get_coordinate_manager(self) -> "CoordinateTreeManager":
        """
        获取本工位的多坐标系管理器 (CoordinateTreeManager)。
        自动绑定工位空间场景文件与 tags_map.yaml 姿态真值，场景缺失自动自愈。
        """
        from src.workspace.coordinate_manager import CoordinateTreeManager
        mgr = CoordinateTreeManager(workspace_id=self.workspace_id, spatial_scene_path=self.spatial_scene_path)
        if not os.path.exists(self.spatial_scene_path):
            mgr.save()

        # 若工位已有立体地图，挂载其 tag poses
        if os.path.exists(self.map_path):
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                tags = data.get("tags", {})
                tag_poses = {}
                for tid_str, tinfo in tags.items():
                    tid = int(tid_str)
                    t_mat = tinfo.get("transform_matrix")
                    if t_mat and len(t_mat) == 4:
                        tag_poses[tid] = np.array(t_mat, dtype=np.float64)
                if tag_poses:
                    mgr.set_tags_map(tag_poses)
            except Exception as e:
                log.warning(f"[WS] 加载工位地图 Tag 姿态失败: {e}")
        return mgr

    def get_roi_manager(self) -> "RoiSpaceManager":
        """
        获取本工位的 3D ROI 空间物件集合管理器 (RoiSpaceManager)。
        自动绑定工位空间场景文件，缺失自动自愈。
        """
        from src.workspace.roi_manager import RoiSpaceManager
        mgr = RoiSpaceManager(workspace_id=self.workspace_id, spatial_scene_path=self.spatial_scene_path)
        if not os.path.exists(self.spatial_scene_path):
            mgr.save()
        return mgr
