# -*- coding: utf-8 -*-
"""
工位生命周期与多沙盒服务管理器 (Workspace Manager)
=================================================
负责工位 (Workspace) 仓储级的生命周期管理：
1. 扫描与检索：list_workspaces, get_workspace_by_id, get_current_workspace
2. 生命周期服务：create_workspace, clone_workspace, rename_workspace, delete_workspace
3. 相机与标靶配置代理：bind_workspace_camera, update_workspace_marker_size, update_tag_anchor
"""

import os
import shutil
import time
import glob
from typing import Any, List, Optional, Dict, Tuple
import yaml

from src.utils.logger import get_logger
from src.workspace.workspace_entity import Workspace
from src.workspace.tag_whitelist_manager import (
    load_workspace_tag_whitelist,
    load_workspace_marker_size_mm,
    load_workspace_tag_config,
    save_workspace_tag_config,
    load_workspace_anchor_tags,
)

log = get_logger(__name__)

# 项目根目录常量
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_WORKSPACES_DIR = os.path.join(PROJECT_ROOT, "data", "workspaces")
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")


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

    def bind_workspace_camera(
        self,
        workspace_id: str,
        camera_type: str = "realsense",
        camera_serial: str = "",
        camera_resolution: Optional[List[int]] = None
    ) -> Tuple[bool, str]:
        """将物理相机硬件类型、序列号及标定基准分辨率原子写穿绑定至工位沙盒"""
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

        new_desc = description if description else f"克隆自工位【{src_ws.name}】({src_ws_id})"
        new_ws = self.create_workspace(
            alias=new_alias,
            description=new_desc,
            camera_type=src_ws.camera_type,
            camera_serial=src_ws.camera_serial,
            camera_resolution=list(src_ws.resolution),
        )

        # 1. 拷贝核心元数据与地图
        if os.path.exists(src_ws.map_path):
            shutil.copy2(src_ws.map_path, new_ws.map_path)
        if os.path.exists(src_ws.whitelist_path):
            shutil.copy2(src_ws.whitelist_path, new_ws.whitelist_path)
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


# ------------------------------ 领域组件对象工厂适配器 ------------------------------
def load_workspace_coordinate_manager(workspace: Workspace):
    """获取指定工位的多坐标系管理器 (委托至 workspace.get_coordinate_manager())"""
    return workspace.get_coordinate_manager()


def load_workspace_roi_manager(workspace: Workspace):
    """获取指定工位的 ROI 空间物件集合管理器 (委托至 workspace.get_roi_manager())"""
    return workspace.get_roi_manager()


__all__ = [
    "Workspace",
    "WorkspaceManager",
    "load_workspace_tag_whitelist",
    "load_workspace_marker_size_mm",
    "load_workspace_tag_config",
    "save_workspace_tag_config",
    "load_workspace_anchor_tags",
    "load_workspace_coordinate_manager",
    "load_workspace_roi_manager",
]
