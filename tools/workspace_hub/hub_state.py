# -*- coding: utf-8 -*-
"""
Workspace Hub 全局中枢状态机 (HubState)
=====================================
组织多 Workspace 列表、两级树形导航流转与全局交互通知，
并将其下领域状态分别内聚至三大专属子状态机：
- state.gallery (GalleryState): 标定/生产相册、网格分页与 LRU 高清预览图缓存
- state.whitelist (WhitelistState): 标靶白名单放行矩阵、物理名义边长与锚点世界坐标
- state.geometry (GeometryState): 机构相对坐标系树、3D ROI 空间物件与 6DoF 模态表单
"""

import os
import json
import time
from typing import Optional

from src.calibration.workspace_manager import WorkspaceManager, Workspace
from src.utils.logger import get_logger

from tools.workspace_hub.states.gallery_state import GalleryState, imread_unicode, imwrite_unicode
from tools.workspace_hub.states.whitelist_state import WhitelistState
from tools.workspace_hub.states.geometry_state import GeometryState

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "workspace_hub"


class HubState:
    """工作空间中枢 (Workspace Hub) 核心状态机"""

    # 1. 工位宏观视图页签 (选中工位根节点时激活)
    TAB_REPORT = "tab_report"               # 大盘看板 (体检报告/全局拓扑)
    TAB_CALIB_IMAGES = "tab_calib_images"   # 标定相册 (工位采样相册)
    TAB_PROD_IMAGES = "tab_prod_images"     # 生产相册 (生产基准相册)
    WS_TAB_ORDER = (TAB_REPORT, TAB_CALIB_IMAGES, TAB_PROD_IMAGES)

    # 2. 坐标系微观视图页签 (选中坐标系子节点时激活)
    TAB_FRAME_POSE_TAGS = "tab_frame_pose_tags" # 机构位姿与 10-Slot Tag 标靶
    TAB_FRAME_ROIS = "tab_frame_rois"           # 3D ROI 空间物件
    FRAME_TAB_ORDER = (TAB_FRAME_POSE_TAGS, TAB_FRAME_ROIS)

    # 视图模式与网格规格 (对齐子状态机定义)
    VIEW_STANDARD = GalleryState.VIEW_STANDARD
    VIEW_EXPANDED = GalleryState.VIEW_EXPANDED
    GRID_COLS = GalleryState.GRID_COLS
    GRID_ROWS = GalleryState.GRID_ROWS
    GRID_PAGE = GalleryState.GRID_PAGE
    ROI_VISIBLE_COUNT = GeometryState.ROI_VISIBLE_COUNT

    def __init__(self, workspace_mgr: WorkspaceManager = None, force_mock: bool = False):
        self.workspace_mgr = workspace_mgr or WorkspaceManager()

        self.workspaces: list[Workspace] = []
        self.selected_workspace_idx = 0

        # 左侧两层树导航状态: ("workspace", ws_idx, None) 或 ("frame", ws_idx, frame_id)
        self.selected_tree_item: tuple[str, int, str | None] = ("workspace", 0, None)
        self.expanded_workspaces: set[str] = set()

        # 右侧激活页签与业务说明弹窗
        self.active_tab = self.TAB_REPORT
        self.is_help_modal_open = False

        # 当前鼠标悬停坐标 (用于 Hover 动效)
        self.mouse_x = -1
        self.mouse_y = -1

        # 浮动提示通知
        self.toast_msg = ""
        self.toast_time = 0.0

        # 实例化领域子状态机
        self.gallery = GalleryState(self)
        self.whitelist = WhitelistState(self)
        self.geometry = GeometryState(self)

        # 扫描并恢复工位状态
        self.refresh_workspaces()
        if self.workspaces:
            saved_id = self._read_saved_workspace_id()
            matched_idx = 0
            if saved_id:
                for i, w in enumerate(self.workspaces):
                    if w.workspace_id == saved_id:
                        matched_idx = i
                        break
            self.select_tree_workspace(matched_idx)
            self.expanded_workspaces.add(self.workspaces[matched_idx].workspace_id)

    # ------------------------------ 工位生命周期与持久化 ------------------------------
    @staticmethod
    def _read_saved_workspace_id() -> str:
        """从 gui_settings.json 读取上次选中的工位 ID"""
        if not os.path.exists(GUI_SETTINGS_FILE):
            return ""
        try:
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return (data.get(APP_ID) or {}).get("selected_workspace_id", "")
        except Exception:
            return ""

    def save_selected_workspace(self):
        """保存当前选中的工位 ID 到 gui_settings.json"""
        ws = self.get_selected_workspace()
        if not ws:
            return
        os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
        data = {}
        if os.path.exists(GUI_SETTINGS_FILE):
            try:
                with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
        if APP_ID not in data:
            data[APP_ID] = {}
        data[APP_ID]["selected_workspace_id"] = ws.workspace_id
        try:
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            log.warning(f"保存 gui_settings.json 失败: {e}")

    def select_workspace_at_index(self, idx: int):
        """选中指定索引的 Workspace，并同步驱动各领域子状态机加载数据"""
        if 0 <= idx < len(self.workspaces):
            self.selected_workspace_idx = idx
            self.save_selected_workspace()
            self.gallery.selected_image_idx = 0
            self.gallery.image_grid_offset = 0
            self.gallery.selected_prod_image_idx = 0
            self.gallery.prod_grid_offset = 0
            self.geometry.roi_scroll_offset = 0
            self.geometry.load_geometry_managers()
            self.gallery.load_current_workspace_images()
            self.gallery.load_prod_images()
            self.whitelist.refresh_whitelist_cache()

    def refresh_workspaces(self):
        """扫描磁盘重新加载所有工位列表，保持原有选中项或自适应定位"""
        prev_ws_id = self.get_selected_workspace().workspace_id if self.workspaces else None
        self.workspaces = self.workspace_mgr.list_workspaces()
        if not self.workspaces:
            self.selected_workspace_idx = 0
            self.gallery.current_images = []
            self.geometry.coord_mgr = None
            self.geometry.roi_mgr = None
            return

        new_idx = 0
        if prev_ws_id:
            for i, w in enumerate(self.workspaces):
                if w.workspace_id == prev_ws_id:
                    new_idx = i
                    break
        self.select_workspace_at_index(new_idx)

    def get_selected_workspace(self) -> Workspace | None:
        """获取当前选中的 Workspace 对象"""
        if 0 <= self.selected_workspace_idx < len(self.workspaces):
            return self.workspaces[self.selected_workspace_idx]
        return None

    def select_workspace_by_offset(self, delta: int):
        """相对移动 Workspace 选择"""
        if not self.workspaces:
            return
        new_idx = (self.selected_workspace_idx + delta) % len(self.workspaces)
        self.select_workspace_at_index(new_idx)
        ws = self.get_selected_workspace()
        if ws:
            self.set_toast(f"已切换工位: 【{ws.name}】 ({ws.workspace_id})")

    # ------------------------------ 树形导航与视图切换 ------------------------------
    def select_tree_workspace(self, ws_idx: int):
        """激活左侧工位根节点 (切换至工位宏观视图: 大盘看板/标定相册/生产相册)"""
        if not self.workspaces or not (0 <= ws_idx < len(self.workspaces)):
            return
        self.select_workspace_at_index(ws_idx)
        self.selected_tree_item = ("workspace", ws_idx, None)
        self.geometry.roi_scroll_offset = 0
        ws = self.get_selected_workspace()
        if ws:
            self.expanded_workspaces.add(ws.workspace_id)
        if self.active_tab not in self.WS_TAB_ORDER:
            self.active_tab = self.TAB_REPORT

    def select_tree_frame(self, arg1, arg2: str = None):
        """激活左侧坐标系子节点 (切换至微观视图: 机构位姿与Tag/3D ROI空间)"""
        if arg2 is not None:
            ws_idx = int(arg1)
            frame_id = str(arg2)
        else:
            ws_idx = self.selected_workspace_idx
            frame_id = str(arg1)
        if not self.workspaces or not (0 <= ws_idx < len(self.workspaces)):
            return
        self.select_workspace_at_index(ws_idx)
        self.selected_tree_item = ("frame", ws_idx, frame_id)
        self.geometry.roi_scroll_offset = 0
        ws = self.get_selected_workspace()
        if ws:
            self.expanded_workspaces.add(ws.workspace_id)
        if self.active_tab not in self.FRAME_TAB_ORDER:
            self.active_tab = self.TAB_FRAME_POSE_TAGS

    def toggle_workspace_expanded(self, ws_id: str):
        """展开/折叠指定工位树节点"""
        if ws_id in self.expanded_workspaces:
            self.expanded_workspaces.remove(ws_id)
        else:
            self.expanded_workspaces.add(ws_id)

    def get_current_tabs(self) -> list[tuple[str, str]]:
        """根据当前左侧选中的树节点类型 (工位 vs 坐标系) 动态返回对应的右侧 Tab 列表"""
        item_type = self.selected_tree_item[0]
        if item_type == "frame":
            return [
                (self.TAB_FRAME_POSE_TAGS, "Tag"),
                (self.TAB_FRAME_ROIS, "ROI物件")
            ]
        else:
            return [
                (self.TAB_REPORT, "大盘看板"),
                (self.TAB_CALIB_IMAGES, "标定相册"),
                (self.TAB_PROD_IMAGES, "★ 生产相册")
            ]

    def get_current_frame_id(self) -> str | None:
        """获取当前激活的坐标系 ID (若当前选中工位根节点则返回 None)"""
        if self.selected_tree_item[0] == "frame":
            return self.selected_tree_item[2]
        return None

    def set_tab(self, tab: str):
        """切换右侧动态区页签 (若处于全宽大图沉浸模式则自动退出)"""
        self.active_tab = tab
        if self.gallery.view_mode == self.VIEW_EXPANDED:
            self.gallery.view_mode = self.VIEW_STANDARD

    def set_toast(self, msg: str, duration: float = 3.0):
        """设定底部浮动提示信息"""
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def toggle_help_modal(self):
        """打开或关闭 Workspace 工位与生产体系说明弹窗"""
        self.is_help_modal_open = not self.is_help_modal_open
        if self.is_help_modal_open:
            self.set_toast("已呼出【Workspace 工位与生产体系】业务说明窗")
        else:
            self.set_toast("已关闭说明窗。")

    def rename_current_workspace(self, new_name: str) -> bool:
        """重命名当前选中的工位显示名称"""
        ws = self.get_selected_workspace()
        if not ws:
            return False
        clean = new_name.strip()
        if not clean:
            return False
        ok = self.workspace_mgr.rename_workspace(ws.workspace_id, clean)
        if ok:
            ws.name = clean
            self.refresh_workspaces()
            self.set_toast(f"工位名称已成功修改为: 【{clean}】")
        return ok

    def update_current_workspace_description(self, new_desc: str) -> bool:
        """更新当前选中工位的备注说明文本"""
        ws = self.get_selected_workspace()
        if not ws:
            return False
        clean = str(new_desc).strip()
        ok = self.workspace_mgr.update_workspace_description(ws.workspace_id, clean)
        if ok:
            ws.description = clean
            self.refresh_workspaces()
            self.set_toast(f"工位备注已成功修改为: 【{clean or '无'}】")
        return ok

    def cycle_workspace_production_mode(self) -> str:
        """循环切换当前工位的生产工作流模式 (scara_sorting -> wheel_inspection -> none)"""
        ws = self.get_selected_workspace()
        if not ws:
            return ""
        modes = [
            ("scara_sorting", "SCARA 智能分选生产线", "asparagus_studio"),
            ("wheel_inspection", "分选轮在席质检生产线", "wheel_inspector"),
            ("none", "通用标定观察工位", "general_viewer"),
        ]
        curr_mode = ws.production.get("mode", "scara_sorting")
        cur_idx = 0
        for i, (m, _, _) in enumerate(modes):
            if m == curr_mode:
                cur_idx = i
                break
        next_idx = (cur_idx + 1) % len(modes)
        next_mode, next_name, next_pipe = modes[next_idx]
        ws.production["mode"] = next_mode
        ws.production["name"] = next_name
        ws.production["active_pipeline"] = next_pipe
        ws.save_meta()
        self.set_toast(f"工位生产工作流已切换为: 【{next_name}】 ({next_mode})")
        return next_mode
