"""
工位工作空间 (Workspace) 与批次沙盒管理核心模块
=================================================
负责工业工位 (Workspace) 全局数据生命周期管理：
1. 物理沙盒隔离：每个工位自包含顶层资产 (tags_map.yaml, tag_whitelist.yaml, workspace_meta.yaml)
2. 双业务分支：
   - calibration/ (标定专区: raw_images/, tag_observations.yaml, reports/, visualized/)
   - production/  (生产与模拟生产专区: raw_images/, reports/, results/)
3. 命名与检索：YYYYMMDD_<alias> 规范化目录生成与元数据持久化
"""

import os
import shutil
import time
import glob
from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict, Tuple
import yaml
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)

# 项目根目录常量
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WORKSPACES_DIR = os.path.join(PROJECT_ROOT, "data", "workspaces")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


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
    image_count: int = 0                   # 外参建图图片数
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
        """【工位专属】相机内参文件路径 (优先存在 intrinsics/camera_intrinsics.yaml，兼容历史路径)"""
        p_intr = os.path.join(self.intrinsics_dir, "camera_intrinsics.yaml")
        if os.path.exists(p_intr):
            return p_intr
        p_calib = os.path.join(self.calibration_dir, "camera_intrinsics.yaml")
        if os.path.exists(p_calib):
            return p_calib
        p_root = os.path.join(self.workspace_dir, "camera_intrinsics.yaml")
        if os.path.exists(p_root):
            return p_root
        return p_intr

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
        """【图集三】生产与模拟生产业务专区根目录"""
        return os.path.join(self.workspace_dir, "production")

    @property
    def prod_raw_images_dir(self) -> str:
        """【图集三】生产与现场采样图像存储目录"""
        return os.path.join(self.production_dir, "raw_images")

    # ------------------------------ 业务方法 ------------------------------
    def get_raw_images_dir(self, purpose: str = "calibration") -> str:
        """按三组用途返回采图目录: 'intrinsics' -> 相机内参图集, 'calibration' -> 外参建图图集, 'production' -> 生产采样图集"""
        if purpose == "intrinsics":
            return self.intrinsics_raw_images_dir
        elif purpose == "production":
            return self.prod_raw_images_dir
        return self.calib_raw_images_dir

    def get_image_count(self, purpose: str = "calibration") -> int:
        """根据用途返回当前磁盘物理图片数量"""
        folder = self.get_raw_images_dir(purpose)
        if not os.path.isdir(folder):
            return 0
        calib_exts = ("*.png", "*.jpg", "*.jpeg")
        count = 0
        for ext in calib_exts:
            count += len(glob.glob(os.path.join(folder, ext)))
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

    def refresh_stats(self):
        """快速刷新物理磁盘状态并持久化至元数据缓存"""
        self.ensure_directories()

        # 1. 标定原始图片数
        img_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
        calib_files = []
        for ext in img_exts:
            calib_files.extend(glob.glob(os.path.join(self.calib_raw_images_dir, ext)))
        self.image_count = len(calib_files)

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

        # 3. 标定观测清单解析 (兼容 observations 与 tags 字段)
        if os.path.exists(self.calib_manifest_path):
            try:
                with open(self.calib_manifest_path, "r", encoding="utf-8") as f:
                    obs_data = yaml.safe_load(f) or {}
                images_dict = obs_data.get("images", {})
                active_count = 0
                seen_tags = set()
                for img in images_dict.values():
                    tags_list = img.get("observations") or img.get("tags") or []
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

        # 4. BA 平差解算状态及指标 (兼容 tags_map.yaml 顶层 rmse_reprojection_px 与 meta 字段)
        if os.path.exists(self.map_path) and os.path.getsize(self.map_path) > 50:
            try:
                with open(self.map_path, "r", encoding="utf-8") as f:
                    map_data = yaml.safe_load(f) or {}
                meta = map_data.get("meta", {})
                tags_dict = map_data.get("tags", {})

                rmse = (
                    map_data.get("rmse_reprojection_px")
                    or map_data.get("rmse_px")
                    or map_data.get("global_rmse_px")
                    or meta.get("global_rmse_px")
                    or meta.get("rmse_reprojection_px")
                    or meta.get("rmse_px")
                )

                if len(tags_dict) > 0:
                    self.ba_solved = True
                    self.global_rmse_px = float(rmse) if rmse is not None else 0.0
                    # 若观测清单未提取到 tags，从平差地图 tags 补齐
                    if not self.valid_tag_ids and tags_dict:
                        self.valid_tag_ids = sorted([int(k) for k in tags_dict.keys() if str(k).isdigit()])
                    # 若 active_image_count 为 0，尝试读取 calibrated_images_count
                    if self.active_image_count == 0 and map_data.get("calibrated_images_count"):
                        self.active_image_count = int(map_data["calibrated_images_count"])
                else:
                    self.ba_solved = False
                    self.global_rmse_px = 0.0
            except Exception:
                self.ba_solved = False
                self.global_rmse_px = 0.0
        else:
            self.ba_solved = False
            self.global_rmse_px = 0.0

        self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")

    def save_meta(self):
        """持久化元数据至 workspace_meta.yaml"""
        self.ensure_directories()
        data = {
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": self.updated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
            "camera_type": self.camera_type,
            "camera_serial": self.camera_serial,
            "camera_resolution": list(self.resolution),
            "status": {
                "image_count": self.image_count,
                "prod_image_count": self.prod_image_count,
                "intrinsics_image_count": self.intrinsics_image_count,
                "active_image_count": self.active_image_count,
                "ba_solved": self.ba_solved,
                "global_rmse_px": self.global_rmse_px,
            }
        }
        if self.production:
            data["production"] = dict(self.production)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, workspace_dir: str, force_refresh: bool = False) -> Optional["Workspace"]:
        """从已有物理目录加载 Workspace 对象 (唯一信任 workspace_meta.yaml)"""
        if not os.path.isdir(workspace_dir):
            return None
        ws_id = os.path.basename(os.path.normpath(workspace_dir))
        meta_path = os.path.join(workspace_dir, "workspace_meta.yaml")

        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = yaml.safe_load(f) or {}
            except Exception:
                meta = {}

        name = meta.get("name") or (ws_id.split("_", 1)[1] if "_" in ws_id else ws_id)
        desc = meta.get("description", "")
        created_at = meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
        updated_at = meta.get("updated_at", "")
        camera_type = meta.get("camera_type", "realsense")
        camera_serial = meta.get("camera_serial", "")
        camera_resolution = meta.get("camera_resolution", [1920, 1080])
        production = meta.get("production", {})

        status = meta.get("status", {})
        image_count = status.get("image_count", 0)
        prod_image_count = status.get("prod_image_count", 0)
        intrinsics_image_count = status.get("intrinsics_image_count", 0)
        active_image_count = status.get("active_image_count", 0)
        ba_solved = status.get("ba_solved", False)
        global_rmse_px = status.get("global_rmse_px", 0.0)

        ws = cls(
            workspace_id=ws_id,
            name=name,
            workspace_dir=os.path.abspath(workspace_dir),
            description=desc,
            created_at=created_at,
            updated_at=updated_at,
            camera_type=camera_type,
            camera_serial=camera_serial,
            camera_resolution=camera_resolution,
            image_count=image_count,
            prod_image_count=prod_image_count,
            intrinsics_image_count=intrinsics_image_count,
            active_image_count=active_image_count,
            ba_solved=ba_solved,
            global_rmse_px=global_rmse_px,
            production=production,
        )

        # 自动探测脏数据或未统计数据，自愈刷新并写回元数据
        manifest_path = os.path.join(workspace_dir, "calibration", "tag_observations.yaml")
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


class WorkspaceManager:
    """工位工作空间总库管理器"""

    # 当前工位 ID 为类级运行时状态 (进程内所有实例共享, 不落盘):
    # 由各 GUI 的显式选择驱动 (set_current_workspace), 兜底最新工位
    _current_ws_id: Optional[str] = None

    def __init__(
        self,
        workspaces_dir: str = DEFAULT_WORKSPACES_DIR,
        config_path: str = DEFAULT_CONFIG_PATH,
        prod_map_path: Optional[str] = None
    ):
        self.workspaces_dir = os.path.abspath(workspaces_dir)
        self.config_path = os.path.abspath(config_path)
        self._cached_workspaces: Dict[str, Workspace] = {}

        os.makedirs(self.workspaces_dir, exist_ok=True)

    def list_workspaces(self) -> List[Workspace]:
        """枚举所有有效工位，按创建时间降序"""
        workspaces = []
        if not os.path.isdir(self.workspaces_dir):
            return []

        for item in os.listdir(self.workspaces_dir):
            if item.startswith("."):
                continue
            item_path = os.path.join(self.workspaces_dir, item)
            if os.path.isdir(item_path):
                try:
                    ws = Workspace.load(item_path)
                    if ws:
                        workspaces.append(ws)
                except Exception as e:
                    log.warning(f"[WARN] 加载工位异常 {item}: {e}")

        workspaces.sort(key=lambda s: (s.created_at, s.workspace_id), reverse=True)
        return workspaces

    def get_workspace_by_id(self, ws_id: str, force_refresh: bool = False) -> Optional[Workspace]:
        """根据工位 ID 检索 Workspace 对象"""
        if not ws_id:
            return None
        target_dir = os.path.join(self.workspaces_dir, ws_id)
        if not os.path.isdir(target_dir):
            return None
        if not force_refresh and ws_id in self._cached_workspaces:
            return self._cached_workspaces[ws_id]
        ws = Workspace.load(target_dir, force_refresh=force_refresh)
        if ws:
            self._cached_workspaces[ws_id] = ws
        return ws

    def get_tag_whitelist_path(self, workspace_id: str) -> str:
        """获取指定工位的 tag_whitelist.yaml 绝对路径"""
        return os.path.join(self.workspaces_dir, workspace_id, "tag_whitelist.yaml")

    def ensure_tag_whitelist(self, workspace_id: str) -> str:
        """确保指定工位的 tag_whitelist.yaml 存在，若不存在则自愈生成规范模板"""
        path = self.get_tag_whitelist_path(workspace_id)
        if not os.path.exists(path):
            ws = self.get_workspace_by_id(workspace_id)
            map_path = os.path.join(self.workspaces_dir, workspace_id, "tags_map.yaml")
            default_marker_size = None
            if os.path.isfile(map_path):
                try:
                    with open(map_path, "r", encoding="utf-8") as mf:
                        mdata = yaml.safe_load(mf) or {}
                    if mdata.get("marker_size_mm"):
                        default_marker_size = float(mdata["marker_size_mm"])
                except Exception:
                    pass

            tags_dict = {int(tid): {} for tid in ws.valid_tag_ids} if (ws and ws.valid_tag_ids) else {}
            default_config = {
                "workspace_id": workspace_id,
                "workspace_name": ws.name if ws else workspace_id,
                "tag_default_size_mm": default_marker_size or 40.0,
                "tags": tags_dict,
                "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义): tags 字典留空 = 探索模式放行所有检测标靶；非空时仅放行字典内标靶",
            }
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(default_config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            except Exception as e:
                log.warning(f"创建默认 tag_whitelist.yaml 失败: {e}")
        return path

    def update_workspace_marker_size(self, workspace_id: str, marker_size_mm: float) -> Tuple[bool, str]:
        """显式更新当前工位的标靶物理边长 tag_default_size_mm (写穿 tag_whitelist.yaml)"""
        if marker_size_mm <= 0:
            return False, "标靶边长必须大于 0 mm"
        ws = self.get_workspace_by_id(workspace_id)
        if not ws:
            return False, f"未找到工位: {workspace_id}"
        ok = save_workspace_tag_config(ws, tag_default_size_mm=marker_size_mm)
        if ok:
            return True, f"标靶物理边长已显式更新为 {marker_size_mm:.3f} mm"
        return False, "写回标靶边长失败"

    def update_tag_anchor(self, workspace_id: str, tag_id: int, xyz: Optional[Any]) -> Tuple[bool, str]:
        """
        更新或清除工位标靶配置中的物理坐标标注 (统一收敛至 tags[tag_id]["xyz_mm"])
        - xyz is None: 清除该 tag 的坐标标注 (保留 tag 本身放行，仅置空 xyz_mm)
        - xyz: 字典 {"xyz_mm": [x, y, z], "known": [bool, bool, bool]} 或坐标列表 [x, y, z]
        """
        ws = self.get_workspace_by_id(workspace_id)
        if not ws:
            return False, f"未找到工位: {workspace_id}"
        cfg = load_workspace_tag_config(ws.workspace_dir)
        tags = cfg.get("tags", {})
        tid_int = int(tag_id)

        if xyz is None:
            if tid_int in tags:
                tags[tid_int].pop("xyz_mm", None)
            msg = f"已清除 Tag #{tid_int:02d} 的物理坐标标注。"
        else:
            raw_xyz = xyz.get("xyz_mm") if isinstance(xyz, dict) and "xyz_mm" in xyz else xyz
            raw_k = xyz.get("known") if isinstance(xyz, dict) else None
            xyz_f = []
            known_from_xyz = []
            for v in (raw_xyz or []):
                if v is None or (isinstance(v, str) and v.strip().lower() in ("null", "none", "~", "nan", ".nan")):
                    xyz_f.append(None)
                    known_from_xyz.append(False)
                else:
                    try:
                        xyz_f.append(float(v))
                        known_from_xyz.append(True)
                    except (TypeError, ValueError):
                        xyz_f.append(None)
                        known_from_xyz.append(False)

            while len(xyz_f) < 3:
                xyz_f.append(None)
                known_from_xyz.append(False)
            xyz_f = xyz_f[:3]
            known_from_xyz = known_from_xyz[:3]

            if isinstance(raw_k, (list, tuple)) and len(raw_k) == 3:
                for i in range(3):
                    if not raw_k[i]:
                        xyz_f[i] = None

            if tid_int not in tags:
                tags[tid_int] = {}
            if any(c is not None for c in xyz_f):
                tags[tid_int]["xyz_mm"] = xyz_f
            else:
                tags[tid_int].pop("xyz_mm", None)

            n_k = sum(1 for c in xyz_f if c is not None)
            coords_str = ", ".join(f"{c:.1f}" if c is not None else "null" for c in xyz_f)
            msg = f"已成功标注 Tag #{tid_int:02d} 坐标: ({coords_str}) mm ({n_k}/3 轴已知) 并自动放行"

        ok = save_workspace_tag_config(ws, tags=tags)
        if ok:
            return True, msg
        return False, "保存工位标靶配置失败"

    def get_current_workspace_id(self) -> str:
        """获取当前工位 ID (运行时显式选择优先, 兜底最新工位; 无任何落盘标记)"""
        cur_id = WorkspaceManager._current_ws_id
        if cur_id and os.path.isdir(os.path.join(self.workspaces_dir, cur_id)):
            return cur_id

        workspaces = self.list_workspaces()
        if workspaces:
            return workspaces[0].workspace_id
        return ""

    def get_current_workspace(self, force_refresh: bool = False) -> Workspace:
        """获取当前工位对象"""
        cur_id = self.get_current_workspace_id()
        if cur_id:
            ws = self.get_workspace_by_id(cur_id, force_refresh=force_refresh)
            if ws:
                return ws

        new_ws = self.create_workspace(alias="默认工位", description="系统自动初始化默认工位")
        return new_ws

    def set_current_workspace(self, ws_id: str) -> bool:
        """设置当前工位 (运行时内存态, 进程内所有 WorkspaceManager 实例共享, 不落盘)"""
        target_dir = os.path.join(self.workspaces_dir, ws_id)
        if not os.path.isdir(target_dir):
            return False
        WorkspaceManager._current_ws_id = ws_id.strip()
        self._cached_workspaces[ws_id] = Workspace.load(target_dir)
        return True

    def invalidate_cache(self):
        """显式使缓存失效"""
        self._cached_workspaces.clear()

    def create_workspace(
        self,
        alias: str,
        description: str = "",
        camera_type: str = "realsense",
        camera_serial: str = "",
        camera_resolution: Optional[List[int]] = None,
    ) -> Workspace:
        """创建新工位 (包含工位专属相机硬件与分辨率规格定义)"""
        display_name = alias.strip() if alias and alias.strip() else "新建工位"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ascii_suffix = "".join(c for c in alias if c.isascii() and (c.isalnum() or c in ("_", "-"))).strip("_")
        if ascii_suffix:
            ws_id = f"{timestamp}_{ascii_suffix}"
        else:
            ws_id = f"{timestamp}_workspace"

        counter = 1
        original_id = ws_id
        while os.path.exists(os.path.join(self.workspaces_dir, ws_id)):
            ws_id = f"{original_id}_{counter}"
            counter += 1

        ws_dir = os.path.join(self.workspaces_dir, ws_id)
        ws = Workspace(
            workspace_id=ws_id,
            name=display_name,
            workspace_dir=ws_dir,
            description=description,
            camera_type=camera_type or "realsense",
            camera_serial=camera_serial or "",
            camera_resolution=camera_resolution or [1920, 1080],
            created_at=time.strftime("%Y-%m-%d %H:%M:%S"),
            production={
                "mode": "scara_sorting",
                "name": "SCARA 智能分选生产线",
                "active_pipeline": "asparagus_studio"
            }
        )
        ws.ensure_directories()
        ws.save_meta()

        # 生成初始白名单 (tags 字典单一真理源)
        with open(ws.whitelist_path, "w", encoding="utf-8") as f:
            yaml.dump({
                "workspace_id": ws_id,
                "workspace_name": display_name,
                "tag_default_size_mm": 40.0,
                "tags": {},
                "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义): tags 字典留空 = 探索模式放行所有检测标靶；非空时仅放行字典内标靶",
            }, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

        # 生成初始空间几何场景 (包含 world 绝对世界坐标系与空 rois 列表)
        from src.workspace.coordinate_manager import CoordinateTreeManager
        coord_mgr = CoordinateTreeManager(workspace_id=ws_id, spatial_scene_path=ws.spatial_scene_path)
        coord_mgr.save()

        self._cached_workspaces[ws_id] = ws
        return ws

    def update_workspace_hardware(
        self,
        workspace_id: str,
        camera_type: str = "realsense",
        camera_serial: str = "",
        camera_resolution: Optional[List[int]] = None
    ) -> Tuple[bool, str]:
        """更新工位绑定的相机硬件类型、序列号及标定分辨率单一真理源"""
        ws = self.get_workspace_by_id(workspace_id)
        if not ws:
            return False, f"未找到工位: {workspace_id}"

        ws.camera_type = str(camera_type).strip().lower() or "realsense"
        ws.camera_serial = str(camera_serial).strip()
        if camera_resolution and len(camera_resolution) == 2:
            w, h = int(camera_resolution[0]), int(camera_resolution[1])
            if w > 0 and h > 0:
                ws.camera_resolution = [w, h]
        ws.save_meta()
        return True, f"工位【{ws.name}】硬件配置已更新: {ws.camera_type} ({ws.camera_serial or '默认'}) @ {ws.resolution_str}"

    def clone_workspace(self, src_ws_id: str, new_alias: str, description: str = "") -> Optional[Workspace]:
        """克隆已有工位数据至新工位"""
        src_dir = os.path.join(self.workspaces_dir, src_ws_id)
        if not os.path.isdir(src_dir):
            return None

        src_ws = Workspace.load(src_dir)
        if not src_ws:
            return None

        new_ws = self.create_workspace(alias=new_alias, description=description or f"克隆自 {src_ws_id}")

        # 1. 拷贝顶层地图与白名单
        if os.path.exists(src_ws.map_path):
            shutil.copy2(src_ws.map_path, new_ws.map_path)
        if os.path.exists(src_ws.whitelist_path):
            shutil.copy2(src_ws.whitelist_path, new_ws.whitelist_path)
        # 1.2 拷贝工位空间几何场景资产 (spatial_scene.yaml)
        if os.path.exists(src_ws.spatial_scene_path):
            shutil.copy2(src_ws.spatial_scene_path, new_ws.spatial_scene_path)

        # 2. 拷贝标定图片与清单
        if os.path.exists(src_ws.calib_raw_images_dir):
            for f in glob.glob(os.path.join(src_ws.calib_raw_images_dir, "*.png")):
                shutil.copy2(f, os.path.join(new_ws.calib_raw_images_dir, os.path.basename(f)))
        if os.path.exists(src_ws.calib_manifest_path):
            shutil.copy2(src_ws.calib_manifest_path, new_ws.calib_manifest_path)

        # 3. 拷贝生产图片
        if os.path.exists(src_ws.prod_raw_images_dir):
            for f in glob.glob(os.path.join(src_ws.prod_raw_images_dir, "*.png")):
                shutil.copy2(f, os.path.join(new_ws.prod_raw_images_dir, os.path.basename(f)))

        new_ws.refresh_stats()
        new_ws.save_meta()
        self._cached_workspaces[new_ws.workspace_id] = new_ws
        return new_ws

    def rename_workspace(self, ws_id: str, new_name: str, new_description: Optional[str] = None) -> bool:
        """修改工位别名与描述"""
        clean_name = new_name.strip()
        if not clean_name:
            return False

        target_dir = os.path.join(self.workspaces_dir, ws_id)
        ws = Workspace.load(target_dir)
        if not ws:
            return False

        ws.name = clean_name
        if new_description is not None:
            ws.description = new_description
        ws.save_meta()
        self._cached_workspaces[ws_id] = ws
        return True

    def update_workspace_description(self, ws_id: str, new_desc: str) -> bool:
        """修改指定工位的备注说明文本并持久化至元数据"""
        target_dir = os.path.join(self.workspaces_dir, ws_id)
        ws = Workspace.load(target_dir)
        if not ws:
            return False
        ws.description = new_desc.strip()
        ws.save_meta()
        self._cached_workspaces[ws_id] = ws
        return True

    def delete_workspace(self, ws_id: str) -> Tuple[bool, str]:
        """物理删除指定工位"""
        ws_dir = os.path.join(self.workspaces_dir, ws_id)
        if not os.path.isdir(ws_dir):
            return False, f"工位目录不存在: {ws_id}"

        try:
            shutil.rmtree(ws_dir)
            if ws_id in self._cached_workspaces:
                del self._cached_workspaces[ws_id]
            self.invalidate_cache()
            return True, f"已成功删除工位: {ws_id}"
        except Exception as e:
            return False, f"删除工位发生异常: {e}"


def load_workspace_tag_whitelist(workspace_dir: str) -> List[int]:
    """
    读取工位 tag_whitelist.yaml 的放行标靶 (工位级物理白名单, 恒启用 — 名单内容即行为):
    - tags 非空 → 返回 sorted(tags.keys()) (该工位检测过滤的权威约束)
    - tags 为空/文件缺失/解析失败 → 返回 [] (探索模式: 不加额外过滤)
    """
    cfg = load_workspace_tag_config(workspace_dir)
    tags = cfg.get("tags") or {}
    return sorted(list(tags.keys()))


def load_workspace_marker_size_mm(workspace_dir: str) -> Optional[float]:
    """
    读取工位 tag_whitelist.yaml 的 tag_default_size_mm (标靶物理边长, 唯一权威来源).

    规则 (FR-9.x):
      - 文件存在且 tag_default_size_mm 为正数 → 返回 float(mm)
      - 文件缺失 / 字段缺失 / 非正数 / 解析失败 → 返回 None
        (调用方需主动报错, 禁止默认 50.0 等兜底值)
    """
    path = os.path.join(workspace_dir, "tag_whitelist.yaml")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        log.warning(f"[WS] 解析工位标靶边长失败 ({path}): {e}")
        return None
    v = data.get("tag_default_size_mm")
    if v is None:
        # 自愈机制: 尝试从当前工位 tags_map.yaml 继承已标定的 marker_size_mm
        map_path = os.path.join(workspace_dir, "tags_map.yaml")
        if os.path.isfile(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as mf:
                    mdata = yaml.safe_load(mf) or {}
                mv = mdata.get("marker_size_mm")
                if mv is not None and float(mv) > 0:
                    f_val = float(mv)
                    data["tag_default_size_mm"] = f_val
                    try:
                        with open(path, "w", encoding="utf-8") as wf:
                            yaml.dump(data, wf, allow_unicode=True, default_flow_style=False, sort_keys=False)
                        log.info(f"[WS] 从 tags_map.yaml 自愈回填 tag_default_size_mm={f_val} mm 至 {path}")
                    except Exception as we:
                        log.warning(f"[WS] 自愈写回 tag_whitelist.yaml 失败: {we}")
                    return f_val
            except Exception as me:
                log.warning(f"[WS] 读取 tags_map.yaml 自愈标靶边长异常: {me}")
        return None
    try:
        f = float(v)
        if f <= 0:
            log.warning(f"[WS] 工位标靶边长非法 (≤0): {path} → {v}")
            return None
        return f
    except (ValueError, TypeError) as e:
        log.warning(f"[WS] 工位标靶边长字段非数值 ({path}): {v} ({e})")
        return None


def load_workspace_tag_config(workspace_dir: str) -> Dict[str, Any]:
    """
    【单一真理源】加载工位标靶综合先验配置 (收敛于 tag_whitelist.yaml.tags):
    - 返回标准 Schema 字典:
      {
          "workspace_id": str,
          "workspace_name": str,
          "tag_default_size_mm": float,
          "tags": Dict[int, Dict[str, Any]],
          "notes": str,
      }
    """
    ws_id = os.path.basename(os.path.normpath(workspace_dir))
    wl_path = os.path.join(workspace_dir, "tag_whitelist.yaml")

    wl_data: Dict[str, Any] = {}
    if os.path.exists(wl_path):
        try:
            with open(wl_path, "r", encoding="utf-8") as f:
                wl_data = yaml.safe_load(f) or {}
        except Exception as e:
            log.warning(f"[WS] 读取 tag_whitelist.yaml 异常 ({wl_path}): {e}")

    clean_tags: Dict[int, Dict[str, Any]] = {}

    if "tags" in wl_data and isinstance(wl_data["tags"], dict):
        # 1. 最新标准 Schema: 单一真理源 tags 字典
        for k, v in wl_data["tags"].items():
            try:
                tid = int(k)
                if not isinstance(v, dict):
                    v = {}
                clean_item = {}
                raw_xyz = v.get("xyz_mm") or v.get("coords") or v.get("position_mm")
                if raw_xyz is not None and isinstance(raw_xyz, (list, tuple)):
                    xyz_f = []
                    for val in raw_xyz[:3]:
                        if val is None or (isinstance(val, str) and val.strip().lower() in ("null", "none", "~", "nan", ".nan")):
                            xyz_f.append(None)
                        else:
                            try:
                                xyz_f.append(round(float(val), 4))
                            except (ValueError, TypeError):
                                xyz_f.append(None)
                    while len(xyz_f) < 3:
                        xyz_f.append(None)
                    if any(c is not None for c in xyz_f):
                        clean_item["xyz_mm"] = xyz_f
                clean_tags[tid] = clean_item
            except (ValueError, TypeError):
                continue


    # 3. 标靶边长解析
    size_val = wl_data.get("tag_default_size_mm")
    if size_val is not None:
        try:
            size_val = float(size_val)
        except Exception:
            size_val = 40.0
    else:
        size_val = 40.0

    return {
        "workspace_id": wl_data.get("workspace_id", ws_id),
        "workspace_name": wl_data.get("workspace_name", ws_id),
        "tag_default_size_mm": size_val,
        "tags": clean_tags,
        "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义)"
    }


def save_workspace_tag_config(
    workspace: Workspace,
    tags: Optional[Dict[int, Dict[str, Any]]] = None,
    tag_default_size_mm: Optional[float] = None
) -> bool:
    """
    【统一真理源唯一序列化出口】写穿工位标靶配置 (tag_whitelist.yaml):
    - 严格且仅输出 tags 单源字典，绝不接受 allowed_ids / anchor_tags 废弃胶水参数;
    - 未知轴在 YAML 中原生格式化为标准 null (跨语言/跨平台无歧义);
    - 物理自愈垃圾清除: 彻底删除孤立的旧 anchor_tags.yaml。
    """
    path = workspace.whitelist_path
    curr = load_workspace_tag_config(workspace.workspace_dir)

    # 1. 确定最终生效的 tags
    if tags is not None:
        curr_tags = dict(tags)
    else:
        curr_tags = dict(curr.get("tags") or {})

    # 2. 清洗规范化 tags
    clean_tags: Dict[int, Dict[str, Any]] = {}
    for tid, tcfg in curr_tags.items():
        try:
            t_int = int(tid)
            if not isinstance(tcfg, dict):
                tcfg = {}
            item = {}
            xyz = tcfg.get("xyz_mm")
            if xyz is not None and isinstance(xyz, (list, tuple)):
                xyz_out = [
                    round(float(c), 4) if (c is not None and str(c).strip().lower() not in ("null", "none", "~", "nan")) else None
                    for c in xyz[:3]
                ]
                while len(xyz_out) < 3:
                    xyz_out.append(None)
                if any(c is not None for c in xyz_out):
                    item["xyz_mm"] = xyz_out
            clean_tags[t_int] = item
        except (ValueError, TypeError):
            continue

    if tag_default_size_mm is not None and tag_default_size_mm > 0:
        curr["tag_default_size_mm"] = float(round(tag_default_size_mm, 3))

    # 3. 严格 Schema 序列化: 仅输出 tags 单一真理源
    clean_doc = {
        "workspace_id": workspace.workspace_id,
        "workspace_name": workspace.name or workspace.workspace_id,
        "tag_default_size_mm": curr["tag_default_size_mm"],
        "tags": {k: clean_tags[k] for k in sorted(clean_tags.keys())},
        "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义)"
    }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(clean_doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        log.info(f"[WS] 统一工位标靶配置已写穿: {path} (总标靶 {len(clean_tags)} 枚)")
    except Exception as e:
        log.error(f"[WS] 写穿工位标靶配置失败 ({path}): {e}")
        return False

    # 物理自愈垃圾清除: 彻底删除孤立的旧 anchor_tags.yaml
    old_anchor_file = os.path.join(workspace.workspace_dir, "anchor_tags.yaml")
    if os.path.isfile(old_anchor_file):
        try:
            os.remove(old_anchor_file)
            log.info(f"[WS] 已删除冗余独立锚点文件: {old_anchor_file}")
        except Exception as e:
            log.warning(f"[WS] 删除旧 anchor_tags.yaml 失败: {e}")

    return True


def load_workspace_anchor_tags(workspace_dir: str) -> Optional[Dict[int, Dict]]:
    """
    【统一真理源接口】读取工位世界坐标锚点 (收敛至 tag_whitelist.yaml 的 tags 字段):
    - 文件存在 → 返回所有具备非空 xyz_mm 的锚点字典 {int tag_id: {"xyz_mm": [f3], "known": [b3]}}
    - 若无任何锚点返回空字典 {}
    - 若整个文件不存在返回 None
    """
    wl_path = os.path.join(workspace_dir, "tag_whitelist.yaml")
    if not os.path.exists(wl_path):
        return None

    cfg = load_workspace_tag_config(workspace_dir)
    tags = cfg.get("tags") or {}
    anchors = {}
    for tid, tcfg in tags.items():
        xyz = tcfg.get("xyz_mm")
        if xyz is not None and isinstance(xyz, (list, tuple)):
            xyz_f = [float(c) if c is not None else 0.0 for c in xyz[:3]]
            known_b = [c is not None for c in xyz[:3]]
            while len(xyz_f) < 3:
                xyz_f.append(0.0)
                known_b.append(False)
            if any(known_b):
                anchors[int(tid)] = {
                    "xyz_mm": xyz_f[:3],
                    "known": known_b[:3]
                }
    return anchors





def load_workspace_coordinate_manager(workspace: Workspace) -> "CoordinateTreeManager":
    """获取指定工位的多坐标系管理器 (自动挂接 tags_map.yaml，空间场景缺失自动自愈模板)"""
    from src.workspace.coordinate_manager import CoordinateTreeManager
    mgr = CoordinateTreeManager(workspace_id=workspace.workspace_id, spatial_scene_path=workspace.spatial_scene_path)
    if not os.path.exists(workspace.spatial_scene_path):
        mgr.save()

    # 若工位已有立体地图，挂载其 tag poses
    if os.path.exists(workspace.map_path):
        try:
            with open(workspace.map_path, "r", encoding="utf-8") as f:
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


def load_workspace_roi_manager(workspace: Workspace) -> "RoiSpaceManager":
    """获取指定工位的 ROI 空间物件集合管理器 (空间场景缺失自动自愈模板)"""
    from src.workspace.roi_manager import RoiSpaceManager
    mgr = RoiSpaceManager(workspace_id=workspace.workspace_id, spatial_scene_path=workspace.spatial_scene_path)
    if not os.path.exists(workspace.spatial_scene_path):
        mgr.save()
    return mgr



