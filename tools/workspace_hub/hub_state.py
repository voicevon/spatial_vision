"""
Workspace Hub 全局状态与数据模型 (HubState)
======================================
管理多 Workspace 列表、当前选定 Workspace 状态、采图连拍与模式流转
"""

import os
import glob
import json
import time
from collections import OrderedDict
from typing import Any, Optional
import cv2
import numpy as np

from src.calibration.workspace_manager import WorkspaceManager, Workspace
from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "workspace_hub"


def imread_unicode(filepath: str, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """兼容 Windows 中文/特殊字符物理路径的鲁棒图像读取 (np.fromfile + cv2.imdecode)"""
    if not os.path.exists(filepath):
        return None
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        if data is None or len(data) == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(filepath: str, img: np.ndarray) -> bool:
    """兼容 Windows 中文/特殊字符物理路径的鲁棒图像写入 (cv2.imencode + tofile)"""
    try:
        ext = os.path.splitext(filepath)[1]
        ok, buf = cv2.imencode(ext, img)
        if ok and buf is not None:
            buf.tofile(filepath)
            return True
        return False
    except Exception:
        return False


class HubState:
    """工作空间中枢 (Workspace Hub) 统一状态与缓存模型"""

    # 右侧动态区页签 (工位宏观视图三页签 + 坐标系微观视图两页签)
    TAB_CALIB_IMAGES = "tab_calib_images"   # 页签: 标定相册 (工位采样相册)
    TAB_PROD_IMAGES = "tab_prod_images"     # 页签: 生产相册 (生产基准工位相册)
    TAB_REPORT = "tab_report"               # 页签: 大盘看板 (体检报告/全局拓扑)
    TAB_WHITELIST = "tab_whitelist"         # 兼容旧页签: Tag 白名单
    TAB_FRAMES_ROIS = "tab_frames_rois"     # 兼容旧页签: 坐标系与 3D ROI 空间
    TAB_FRAME_POSE_TAGS = "tab_frame_pose_tags" # 坐标系页签: 机构位姿与 Tag 标靶
    TAB_FRAME_ROIS = "tab_frame_rois"           # 坐标系页签: 3D ROI 空间物件
    # 工位宏观视图页签顺序
    WS_TAB_ORDER = (TAB_REPORT, TAB_CALIB_IMAGES, TAB_PROD_IMAGES)
    # 坐标系微观视图页签顺序
    FRAME_TAB_ORDER = (TAB_FRAME_POSE_TAGS, TAB_FRAME_ROIS)
    TAB_ORDER = (TAB_REPORT, TAB_FRAMES_ROIS, TAB_WHITELIST, TAB_CALIB_IMAGES, TAB_PROD_IMAGES)

    # 视图模式 (全宽大图沉浸预览, 仅在标定相册页签下双击卡片展开)
    VIEW_STANDARD = "standard"    # 标准: 左栏 + 右侧页签内容
    VIEW_EXPANDED = "expanded"    # 全宽大图: 右侧区域整体铺满单帧大图

    # 相册卡片网格规格 (与渲染器保持一致): 3 列 x 3 行 = 每页 9 张大卡片
    GRID_COLS = 3
    GRID_ROWS = 3
    GRID_PAGE = 9

    # 3D ROI 空间物件列表单屏可见行数 (高 552px / 80px = 6.9 -> 7行完整铺满)
    ROI_VISIBLE_COUNT = 7

    def __init__(self, workspace_mgr: WorkspaceManager = None, force_mock: bool = False):
        self.workspace_mgr = workspace_mgr or WorkspaceManager()

        self.workspaces: list[Workspace] = []
        self.selected_workspace_idx = 0

        # 左侧两层树导航状态: ("workspace", ws_idx, None) 或 ("frame", ws_idx, frame_id)
        self.selected_tree_item: tuple[str, int, str | None] = ("workspace", 0, None)
        self.expanded_workspaces: set[str] = set()

        # 当前选中工位的照片列表与卡片网格选中项
        self.current_images: list[str] = []
        self.selected_image_idx = 0
        self.image_grid_offset = 0   # 卡片网格当前页起始索引 (按整行对齐)
        self.roi_scroll_offset = 0   # 3D ROI 物件列表滚动起始行偏移

        # 生产相册相关状态 (生产运行基准工位相册)
        self.prod_images: list[str] = []
        self.selected_prod_image_idx = 0
        self.prod_grid_offset = 0

        # 内存缩略图与预览图缓存 (有序字典实现 LRU，限制最大 200 张防内存溢出)
        self.thumbnail_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.preview_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.max_cache_size = 200

        # 状态 Toast 提示
        self.toast_msg = ""
        self.toast_time = 0.0

        # 当前视图模式 (默认标准; 按 F 键在标定相册页签内进入全宽大图)
        self.view_mode = self.VIEW_STANDARD

        # 右侧动态区当前激活页签 (默认: 大盘看板)
        self.active_tab = self.TAB_REPORT

        # 多坐标系与 3D ROI 空间管理器缓存
        self.coord_mgr = None
        self.roi_mgr = None
        self._coord_mgr_cache: dict[str, Any] = {}
        self._roi_mgr_cache: dict[str, Any] = {}

        # 结构化表单弹窗状态 (坐标系 / ROI)
        self.frame_modal_open: bool = False
        self.frame_modal_is_new: bool = False
        self.frame_modal_data: dict = {}

        self.roi_modal_open: bool = False
        self.roi_modal_is_new: bool = False
        self.roi_modal_data: dict = {}

        # 弹窗内下拉选择框展开状态 (如 "frame_type" / "frame_parent" / "roi_category" / "roi_frame" 或 None)
        self.active_dropdown: str | None = None
        self.frame_modal_orig_id: str | None = None
        self.roi_modal_orig_id: str | None = None

        # 6DoF 外参位姿与约束独立模态弹窗
        self.pose6d_modal_open: bool = False
        self.pose6d_modal_vals: list[float] = [0.0] * 6     # [X, Y, Z, Roll, Pitch, Yaw]
        self.pose6d_modal_known: list[bool] = [False] * 6  # 逐轴已知掩码
        self.pose6d_modal_axis_sel: int = 0          # 0~5
        self.pose6d_modal_axis_buf: str = ""


        # Tag 白名单缓存 (按文件 mtime 自动感知外部编辑并刷新)
        self._whitelist_cache: dict = {}
        self._whitelist_cache_mtime: float = -1.0
        self._whitelist_cache_ws: str = ""

        # 白名单页内编辑态 (方案A 芯片矩阵编辑器: 点击写穿保存, 不依赖外部编辑器)
        self.whitelist_edit_mode: bool = False
        self.whitelist_edit_ids: set = set()      # 编辑工作集合 (芯片即时反馈源)

        # 锚点坐标编辑 (编辑态子模式: 弹窗逐轴输入 xyz, 支持部分已知与清除)
        self.anchor_config_path: str = "config/config.yaml"   # 可注入临时路径供测试 (已废弃: 锚点全走工位沙盒, 仅供老测试兼容)
        self.anchor_mode: bool = False                 # 子模式: 芯片点击改为打开锚点弹窗
        self.anchor_map: dict = {}                     # {tag_id: {"xyz_mm","known"}} 缓存
        self.anchor_modal_open: bool = False
        self.anchor_modal_tag: int = -1
        self.anchor_modal_xyz: list = [0.0, 0.0, 0.0]  # 草稿值 (未知轴为占位)
        self.anchor_modal_known: list = [False, False, False]
        self.anchor_axis_sel: int = -1                 # 当前输入轴 0/1/2 (X/Y/Z)
        self.anchor_axis_buf: str = ""                 # 输入缓冲

        # 标靶物理边长 (Tag 公共属性) 专属编辑模态窗
        self.marker_size_modal_open: bool = False
        self.marker_size_buf: str = ""

        # 兼容性属性 (右键菜单已全量移除，此标记恒为 False)
        self.context_menu_open = False
        # 生产机制业务说明弹窗状态
        self.is_help_modal_open = False

        # 当前鼠标悬停坐标 (用于按钮 Hover 高亮效果)
        self.mouse_x = -1
        self.mouse_y = -1

        # 左侧两层树导航状态: ("workspace", ws_idx, None) 或 ("frame", ws_idx, frame_id)
        self.selected_tree_item: tuple[str, int, str | None] = ("workspace", 0, None)
        self.expanded_workspaces: set[str] = set()

        # 左侧激活卡片的持久化 (按 workspace_id 定位, 不受列表排序变化影响)
        self._saved_workspace_id = self._read_saved_workspace_id()
        self._selection_restored = False

        # 初始加载工位
        self.refresh_workspaces()

    # ------------------------------ 左侧激活卡片持久化 ------------------------------
    @staticmethod
    def _read_saved_workspace_id() -> str:
        """读取上次激活的工位 ID (config/gui_settings.json → workspace_hub.hub_state)"""
        try:
            if not os.path.exists(GUI_SETTINGS_FILE):
                return ""
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            return str(((root.get(APP_ID) or {}).get("hub_state") or {})
                       .get("selected_workspace_id") or "")
        except Exception as e:
            log.warning(f"读取 Workspace Hub 激活工位失败: {e}")
            return ""

    def save_selected_workspace(self):
        """持久化当前左侧激活的工位卡片 (退出后下次启动自动恢复高亮与相册)"""
        ws = self.get_selected_workspace()
        if ws is None:
            return
        try:
            root = {}
            if os.path.exists(GUI_SETTINGS_FILE):
                try:
                    with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                    if not isinstance(root, dict):
                        root = {}
                except Exception:
                    root = {}
            node = root.setdefault(APP_ID, {})
            state = node.get("hub_state")
            if not isinstance(state, dict):
                state = {}
            state["selected_workspace_id"] = str(ws.workspace_id)
            state["selected_workspace_name"] = str(ws.name)
            state["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            node["hub_state"] = state
            os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存 Workspace Hub 激活工位失败: {e}")

    def select_workspace_at_index(self, idx: int):
        """激活左侧第 idx 张工位卡片 (高亮 → 载入其相册 → 持久化选择)"""
        if not self.workspaces or not (0 <= idx < len(self.workspaces)):
            return
        if idx != self.selected_workspace_idx:
            self.selected_workspace_idx = idx
            self.selected_image_idx = 0
            self.image_grid_offset = 0
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
        self.load_current_workspace_images()
        self.load_prod_images()
        self.load_geometry_managers()
        self.save_selected_workspace()
        ws = self.get_selected_workspace()
        if ws:
            self.workspace_mgr.set_current_workspace(ws.workspace_id)

    def refresh_workspaces(self):
        """刷新工位列表"""
        self.workspaces = self.workspace_mgr.list_workspaces()
        # 清理已不存在工位的管理器缓存
        valid_ws_ids = {w.workspace_id for w in self.workspaces}
        for k in list(self._coord_mgr_cache.keys()):
            if k not in valid_ws_ids:
                self._coord_mgr_cache.pop(k, None)
                self._roi_mgr_cache.pop(k, None)

        # 首次加载: 恢复上次激活的工位卡片 (按 workspace_id 定位, 索引排序变化无影响)
        if not self._selection_restored:
            self._selection_restored = True
            if self._saved_workspace_id:
                for i, ws in enumerate(self.workspaces):
                    if ws.workspace_id == self._saved_workspace_id:
                        self.selected_workspace_idx = i
                        break
                # 恢复的高亮工位同步为运行时当前工位 (保证管道层白名单/锚点跟随)
                restored = self.get_selected_workspace()
                if restored and restored.workspace_id == self._saved_workspace_id:
                    self.workspace_mgr.set_current_workspace(restored.workspace_id)

        # 确保选中索引不越界
        if not self.workspaces:
            self.selected_workspace_idx = 0
        else:
            self.selected_workspace_idx = max(0, min(self.selected_workspace_idx, len(self.workspaces) - 1))

        self.load_current_workspace_images()
        self.load_prod_images()
        self.load_geometry_managers()
        # 列表变动 (新建/克隆/重命名/删除) 后同步落盘激活卡片
        self.save_selected_workspace()

    def get_selected_workspace(self) -> Workspace | None:
        """获取当前高亮选中的工位"""
        if not self.workspaces or self.selected_workspace_idx >= len(self.workspaces):
            return None
        return self.workspaces[self.selected_workspace_idx]

    def select_workspace_by_offset(self, delta: int):
        """按偏移量切换选中的工位卡片 (同步运行时当前工位, 保证管道层白名单/锚点跟随)"""
        if not self.workspaces:
            return
        new_idx = (self.selected_workspace_idx + delta) % len(self.workspaces)
        if new_idx != self.selected_workspace_idx:
            self.selected_workspace_idx = new_idx
            self.selected_image_idx = 0
            self.image_grid_offset = 0
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            self.load_current_workspace_images()
            self.load_prod_images()
            self.load_geometry_managers()
        self.save_selected_workspace()
        ws = self.get_selected_workspace()
        if ws:
            self.workspace_mgr.set_current_workspace(ws.workspace_id)

    def load_geometry_managers(self):
        """加载当前选中工位的多坐标系与 ROI 管理器 (利用内存缓存杜绝重复磁盘 I/O)"""
        ws = self.get_selected_workspace()
        if ws:
            ws_id = ws.workspace_id
            if ws_id in self._coord_mgr_cache and ws_id in self._roi_mgr_cache:
                self.coord_mgr = self._coord_mgr_cache[ws_id]
                self.roi_mgr = self._roi_mgr_cache[ws_id]
            else:
                from src.calibration.workspace_manager import (
                    load_workspace_coordinate_manager,
                    load_workspace_roi_manager
                )
                self.coord_mgr = load_workspace_coordinate_manager(ws)
                self.roi_mgr = load_workspace_roi_manager(ws)
                self._coord_mgr_cache[ws_id] = self.coord_mgr
                self._roi_mgr_cache[ws_id] = self.roi_mgr
        else:
            self.coord_mgr = None
            self.roi_mgr = None

    def get_workspace_coord_mgr(self, ws) -> Any:
        """获取指定工位的坐标系管理器（带内存缓存，避免渲染树形结构每帧重复加载磁盘 YAML）"""
        if not ws:
            return None
        ws_id = ws.workspace_id
        if ws_id in self._coord_mgr_cache:
            return self._coord_mgr_cache[ws_id]
        from src.calibration.workspace_manager import load_workspace_coordinate_manager
        mgr = load_workspace_coordinate_manager(ws)
        self._coord_mgr_cache[ws_id] = mgr
        return mgr

    def get_coordinate_frames(self):
        """获取当前工位的所有坐标系定义列表"""
        if self.coord_mgr:
            return self.coord_mgr.list_frames()
        return []

    def get_roi_spaces(self):
        """获取当前工位的所有 ROI 空间物件列表"""
        if self.roi_mgr:
            return self.roi_mgr.list_rois()
        return []

    # ------------------------------ 树形导航与 Tag 分段管理 ------------------------------
    def select_tree_workspace(self, ws_idx: int):
        """激活左侧工位根节点 (切换至工位宏观视图: 大盘看板/标定相册/生产相册)"""
        if not self.workspaces or not (0 <= ws_idx < len(self.workspaces)):
            return
        self.select_workspace_at_index(ws_idx)
        self.selected_tree_item = ("workspace", ws_idx, None)
        self.roi_scroll_offset = 0
        ws = self.get_selected_workspace()
        if ws:
            self.expanded_workspaces.add(ws.workspace_id)
        if self.active_tab not in (self.TAB_REPORT, self.TAB_CALIB_IMAGES, self.TAB_PROD_IMAGES):
            self.active_tab = self.TAB_REPORT

    def select_tree_frame(self, arg1, arg2: str = None):
        """激活左侧坐标系子节点 (支持 (ws_idx, frame_id) 或单传 (frame_id))"""
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
        self.roi_scroll_offset = 0
        ws = self.get_selected_workspace()
        if ws:
            self.expanded_workspaces.add(ws.workspace_id)
        if self.active_tab not in (self.TAB_FRAME_POSE_TAGS, self.TAB_FRAME_ROIS):
            self.active_tab = self.TAB_FRAME_POSE_TAGS

    def toggle_workspace_expanded(self, ws_id: str):
        """展开/折叠指定工位卡片"""
        if ws_id in self.expanded_workspaces:
            self.expanded_workspaces.remove(ws_id)
        else:
            self.expanded_workspaces.add(ws_id)

    def get_current_tabs(self) -> list[tuple[str, str]]:
        """根据当前左侧选中的树节点 (工位 vs 坐标系) 动态返回对应的右侧 Tab 列表"""
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

    def get_frame_tag_range(self, frame_id: str) -> list[int]:
        """获取坐标系分配的专属 Tag ID 命名空间区间 (0~9, 10~19, 20~29...)"""
        if frame_id == "world":
            return list(range(0, 10))
        frames = self.get_coordinate_frames()
        non_world_frames = [f.frame_id for f in frames if f.frame_id != "world"]
        if frame_id in non_world_frames:
            k = non_world_frames.index(frame_id) + 1
            start = k * 10
            return list(range(start, start + 10))
        return list(range(10, 20))

    def toggle_frame_tag_allowed(self, frame_id: str, tag_id: int) -> bool:
        """在当前坐标系专属区间内切换某 Tag 的放行状态，并原子写穿工位 tag_whitelist.yaml"""
        ws = self.get_selected_workspace()
        if not ws:
            return False
        import yaml
        wl_path = os.path.join(ws.workspace_dir, "tag_whitelist.yaml")
        curr_cfg = {}
        if os.path.isfile(wl_path):
            try:
                with open(wl_path, "r", encoding="utf-8") as f:
                    curr_cfg = yaml.safe_load(f) or {}
            except Exception:
                curr_cfg = {}
        allowed = set(curr_cfg.get("allowed_ids", []))
        if tag_id in allowed:
            allowed.remove(tag_id)
            now_allowed = False
        else:
            allowed.add(tag_id)
            now_allowed = True
        curr_cfg["allowed_ids"] = sorted(list(allowed))
        with open(wl_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(curr_cfg, f, allow_unicode=True)
        self.refresh_whitelist_cache()
        action_desc = "放行" if now_allowed else "禁行"
        self.set_toast(f"坐标系 [{frame_id}] 标靶 Tag #{tag_id} 已{action_desc} (已同步工位白名单)")
        return now_allowed

    def get_frame_tags_status(self, frame_id: str) -> list[int]:
        """获取当前坐标系在工位白名单中属于其分配区间的已放行 Tag ID 列表"""
        tag_range = set(self.get_frame_tag_range(frame_id))
        wl = self.get_tag_whitelist() or {}
        allowed_set = set(wl.get("allowed_ids") or [])
        return sorted(list(tag_range.intersection(allowed_set)))

    def get_frame_rois(self, frame_id: str):
        """获取专属归属于当前坐标系的 3D ROI 空间物件"""
        return [r for r in self.get_roi_spaces() if r.frame_id == frame_id]

    def scroll_roi_list(self, delta_items: int):
        """3D ROI 物件列表滚动 (滚轮/翻页按钮驱动)"""
        cur_frame = self.get_selected_frame()
        if not cur_frame:
            self.roi_scroll_offset = 0
            return
        rois = self.get_frame_rois(cur_frame.frame_id)
        max_offset = max(0, len(rois) - self.ROI_VISIBLE_COUNT)
        self.roi_scroll_offset = max(0, min(self.roi_scroll_offset + delta_items, max_offset))

    def jump_roi_scroll_by_y(self, click_y: int, track_y: int = 100, track_h: int = 552):
        """点击滚动条轨道快速跳转 ROI 视口"""
        cur_frame = self.get_selected_frame()
        if not cur_frame:
            return
        rois = self.get_frame_rois(cur_frame.frame_id)
        max_offset = max(0, len(rois) - self.ROI_VISIBLE_COUNT)
        if max_offset <= 0:
            self.roi_scroll_offset = 0
            return
        ratio = max(0.0, min(1.0, (click_y - track_y) / float(track_h)))
        self.roi_scroll_offset = int(round(ratio * max_offset))

    def load_current_workspace_images(self):
        """载入当前选中工位的照片列表"""
        ws = self.get_selected_workspace()
        if not ws or not os.path.exists(ws.calib_raw_images_dir):
            self.current_images = []
            return

        imgs = sorted(glob.glob(os.path.join(ws.calib_raw_images_dir, "*.png")))
        self.current_images = imgs
        if self.current_images:
            self.selected_image_idx = max(0, min(self.selected_image_idx, len(self.current_images) - 1))
        else:
            self.selected_image_idx = 0

    def _clamp_grid_offset(self, offset: int, total: int) -> int:
        """将网格分页起始索引按整行对齐并夹紧到合法范围"""
        aligned = max(0, (offset // self.GRID_COLS) * self.GRID_COLS)
        max_offset = max(0, ((max(total, 1) - 1) // self.GRID_PAGE) * self.GRID_PAGE)
        return max(0, min(aligned, max_offset))

    def scroll_image_grid(self, delta_rows: int):
        """卡片网格按行滚动 (滚轮/翻页按钮驱动)"""
        self.image_grid_offset = self._clamp_grid_offset(
            self.image_grid_offset + delta_rows * self.GRID_COLS, len(self.current_images))

    def select_image_at_index(self, idx: int):
        """直接选中第 idx 张卡片 (自动翻页使其可见)"""
        if 0 <= idx < len(self.current_images):
            self.selected_image_idx = idx
            self._ensure_image_visible()

    def _ensure_image_visible(self):
        """确保当前选中卡片处于可见页范围内 (自动翻页)"""
        if not self.current_images:
            self.image_grid_offset = 0
            return
        idx = self.selected_image_idx
        start = self.image_grid_offset
        if idx < start:
            self.image_grid_offset = self._clamp_grid_offset(
                (idx // self.GRID_COLS) * self.GRID_COLS, len(self.current_images))
        elif idx >= start + self.GRID_PAGE:
            target_row = max(0, (idx // self.GRID_COLS) - self.GRID_ROWS + 1)
            self.image_grid_offset = self._clamp_grid_offset(
                target_row * self.GRID_COLS, len(self.current_images))

    def select_image_by_offset(self, delta: int):
        """在卡片网格中前后切换选中的单帧图片 (自动翻页跟随)"""
        if not self.current_images:
            return
        new_idx = max(0, min(self.selected_image_idx + delta, len(self.current_images) - 1))
        self.selected_image_idx = new_idx
        self._ensure_image_visible()

    def load_prod_images(self):
        """载入当前选中工位的生产采图相册 (生产相册页签数据源)"""
        ws = self.get_selected_workspace()
        if not ws or not os.path.exists(ws.prod_raw_images_dir):
            self.prod_images = []
            self.selected_prod_image_idx = 0
            self.prod_grid_offset = 0
            return

        exts = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
        imgs = []
        for ext in exts:
            imgs.extend(glob.glob(os.path.join(ws.prod_raw_images_dir, ext)))
        self.prod_images = sorted(list(set(imgs)))
        if self.prod_images:
            self.selected_prod_image_idx = max(0, min(self.selected_prod_image_idx, len(self.prod_images) - 1))
        else:
            self.selected_prod_image_idx = 0

    def scroll_prod_grid(self, delta_rows: int):
        """生产相册卡片网格按行滚动"""
        self.prod_grid_offset = self._clamp_grid_offset(
            self.prod_grid_offset + delta_rows * self.GRID_COLS, len(self.prod_images))

    def select_prod_image_at_index(self, idx: int):
        """直接选中生产相册第 idx 张卡片 (自动翻页使其可见)"""
        if 0 <= idx < len(self.prod_images):
            self.selected_prod_image_idx = idx
            self._ensure_prod_visible()

    def _ensure_prod_visible(self):
        """确保当前选中的生产相册卡片处于可见页范围内"""
        if not self.prod_images:
            self.prod_grid_offset = 0
            return
        idx = self.selected_prod_image_idx
        start = self.prod_grid_offset
        if idx < start:
            self.prod_grid_offset = self._clamp_grid_offset(
                (idx // self.GRID_COLS) * self.GRID_COLS, len(self.prod_images))
        elif idx >= start + self.GRID_PAGE:
            target_row = max(0, (idx // self.GRID_COLS) - self.GRID_ROWS + 1)
            self.prod_grid_offset = self._clamp_grid_offset(
                target_row * self.GRID_COLS, len(self.prod_images))

    def select_prod_image_by_offset(self, delta: int):
        """在生产相册卡片网格中前后切换选中的单帧图片"""
        if not self.prod_images:
            return
        new_idx = max(0, min(self.selected_prod_image_idx + delta, len(self.prod_images) - 1))
        self.selected_prod_image_idx = new_idx
        self._ensure_prod_visible()

    def get_whitelist_data(self) -> dict:
        """获取当前工位的白名单与锚点数据 (get_tag_whitelist 的别名)"""
        return self.get_tag_whitelist()

    def get_tag_whitelist(self) -> dict:
        """读取当前选中工位的 tag_whitelist.yaml (基于 mtime 自动感知外部编辑并刷新缓存)"""
        ws = self.get_selected_workspace()
        if not ws:
            return {}
        path = self.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        if not os.path.exists(path):
            return {}
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return {}

        if (self._whitelist_cache_ws == ws.workspace_id
                and self._whitelist_cache
                and abs(mtime - self._whitelist_cache_mtime) < 1e-6):
            return self._whitelist_cache

        try:
            import yaml
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}

        self._whitelist_cache = data
        self._whitelist_cache_mtime = mtime
        self._whitelist_cache_ws = ws.workspace_id
        return data

    def refresh_whitelist_cache(self):
        """强制失效白名单缓存 (外部编辑器保存返回后立即刷新)"""
        self._whitelist_cache = {}
        self._whitelist_cache_mtime = -1.0
        self._whitelist_cache_ws = ""

    def ensure_tag_whitelist_file(self) -> str:
        """确保当前选中工位的白名单文件存在并有效"""
        ws = self.get_selected_workspace()
        if not ws:
            return ""
        return self.workspace_mgr.ensure_tag_whitelist(ws.workspace_id)

    def update_tag_anchor(self, tag_id: int, anchor_data: Any) -> tuple[bool, str]:
        """更新或清除当前工位中的 tag_anchors 物理坐标标注 (统一标准结构)"""
        ws = self.get_selected_workspace()
        if not ws:
            return False, "未选择工位"
        ok, msg = self.workspace_mgr.update_tag_anchor(ws.workspace_id, tag_id, anchor_data)
        if ok:
            self._reload_anchor_map()
            self.refresh_whitelist_cache()
        return ok, msg

    # ==================== 白名单页内编辑态 (芯片矩阵写穿保存) ====================

    def enter_whitelist_edit(self):
        """进入编辑模式: 以当前 yaml 的 allowed_ids 为初始工作集合"""
        wl = self.get_tag_whitelist() or {}
        ids = set()
        for x in (wl.get("allowed_ids") or []):
            try:
                ids.add(int(x))
            except (TypeError, ValueError):
                continue
        self.whitelist_edit_ids = ids
        self.whitelist_edit_mode = True
        self.anchor_mode = False
        self._close_anchor_modal()

    def exit_whitelist_edit(self):
        """退出编辑模式 (写穿式保存, 无未保存残留)"""
        self.whitelist_edit_mode = False
        self.anchor_mode = False
        self._close_anchor_modal()

    def _save_whitelist_yaml(self):
        """写穿当前编辑集合到 tag_whitelist.yaml (保留其余字段, 更新 mtime 联动全链路缓存)"""
        import yaml
        ws = self.get_selected_workspace()
        if not ws:
            return
        path = self.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        doc = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
            except Exception:
                doc = {}
        doc["workspace_id"] = ws.workspace_id
        doc["workspace_name"] = ws.name
        doc["allowed_ids"] = sorted(self.whitelist_edit_ids)
        if "tag_default_size_mm" not in doc:
            size_val = 35.5
            if os.path.isfile(ws.map_path):
                try:
                    with open(ws.map_path, "r", encoding="utf-8") as mf:
                        mdata = yaml.safe_load(mf) or {}
                    if mdata.get("marker_size_mm"):
                        size_val = float(mdata["marker_size_mm"])
                except Exception:
                    pass
            doc["tag_default_size_mm"] = size_val
        doc.setdefault("description", f"Workspace {ws.name} 标靶白名单配置")
        doc.setdefault("notes", "工位物理白名单恒启用 (名单内容即行为): allowed_ids 非空时仅放行名单内标靶 (权威约束)；留空 = 探索模式放行所有检测标靶")
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            log.warning(f"写回 tag_whitelist.yaml 失败: {e}")
            self.set_toast(f"写回白名单失败: {e}")
            return
        self.refresh_whitelist_cache()

    def toggle_whitelist_id(self, tag_id: int) -> int:
        """切换芯片放行/拦截并写穿保存, 返回当前放行数"""
        if tag_id in self.whitelist_edit_ids:
            self.whitelist_edit_ids.discard(tag_id)
        else:
            self.whitelist_edit_ids.add(tag_id)
        self._save_whitelist_yaml()
        return len(self.whitelist_edit_ids)

    def whitelist_batch(self, action: str) -> int:
        """批量操作: 'all'=全量放行 0~29, 'clear'=清空 (探索模式); 返回当前放行数"""
        if action == "all":
            self.whitelist_edit_ids = set(range(30))
        elif action == "clear":
            self.whitelist_edit_ids = set()
        self._save_whitelist_yaml()
        return len(self.whitelist_edit_ids)

    # ==================== 锚点坐标编辑 (编辑态子模式, 弹窗写回工位锚点文件) ====================

    def enter_anchor_mode(self):
        """进入锚点子模式: 芯片点击改为打开锚点弹窗"""
        self._reload_anchor_map()
        self.anchor_mode = True

    def exit_anchor_mode(self):
        self.anchor_mode = False
        self._close_anchor_modal()

    def _reload_anchor_map(self):
        """载入当前工位锚点 (Tag 锚点统一为工位独立数据, 未写穿前以当前标定为基础)"""
        from src.calibration.workspace_manager import load_workspace_anchor_tags, load_workspace_tag_anchors
        from src.utils.config_guard import load_anchor_tags
        ws = self.get_selected_workspace()
        if not ws:
            self.anchor_map = {}
            return
        m = load_workspace_anchor_tags(ws.workspace_dir)
        if m is None:
            m = load_workspace_tag_anchors(ws.workspace_dir)
        if m is None and getattr(self, "anchor_config_path", None) and os.path.isfile(self.anchor_config_path):
            m = load_anchor_tags(self.anchor_config_path)
        self.anchor_map = m or {}

    def get_anchor_map(self) -> dict:
        """获取当前工位完整的 AprilTag 物理锚点映射表 (优先取 anchor_tags.yaml，回退 tag_whitelist.yaml)"""
        self._reload_anchor_map()
        return self.anchor_map or {}

    def _persist_anchor_map(self) -> bool:
        """写穿当前工位锚点文件并同步写穿 tag_whitelist.yaml (统一采用标准结构)"""
        from src.calibration.workspace_manager import save_workspace_anchor_tags
        ws = self.get_selected_workspace()
        if ws is None:
            return False
        ok = save_workspace_anchor_tags(ws, self.anchor_map)
        path = self.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        if os.path.isfile(path):
            try:
                import yaml
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg["tag_anchors"] = {
                    tid: {"xyz_mm": [float(v) for v in a["xyz_mm"]], "known": [bool(b) for b in a.get("known", [True, True, True])]}
                    for tid, a in sorted(self.anchor_map.items())
                }
                # 自动将具有已知轴约束的 tag 加入 allowed_ids
                allowed_set = set(cfg.get("allowed_ids", []))
                for tid, a in self.anchor_map.items():
                    if any(a.get("known", [])):
                        allowed_set.add(tid)
                cfg["allowed_ids"] = sorted(list(allowed_set))
                if "tag_default_size_mm" not in cfg:
                    size_val = 35.5
                    if os.path.isfile(ws.map_path):
                        try:
                            with open(ws.map_path, "r", encoding="utf-8") as mf:
                                mdata = yaml.safe_load(mf) or {}
                            if mdata.get("marker_size_mm"):
                                size_val = float(mdata["marker_size_mm"])
                        except Exception:
                            pass
                    cfg["tag_default_size_mm"] = size_val
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                self.refresh_whitelist_cache()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"同步 tag_whitelist.yaml tag_anchors 异常: {e}")
        return ok

    def _close_anchor_modal(self):
        self.anchor_modal_open = False
        self.anchor_modal_tag = -1
        self.anchor_axis_sel = -1
        self.anchor_axis_buf = ""

    def open_anchor_editor(self, tag_id: int):
        """打开 Tag 专属坐标编辑弹窗: 以工位统一锚点结构为草稿"""
        self._reload_anchor_map()
        entry = self.anchor_map.get(tag_id) if self.anchor_map else None
        if not entry:
            wl = self.get_tag_whitelist()
            anchors = wl.get("tag_anchors", {}) if isinstance(wl, dict) else {}
            cand = anchors.get(tag_id) or anchors.get(str(tag_id))
            if isinstance(cand, dict) and "xyz_mm" in cand:
                entry = cand
        if entry:
            self.anchor_modal_xyz = [float(v) for v in entry["xyz_mm"]]
            self.anchor_modal_known = [bool(b) for b in entry.get("known", [True, True, True])]
        else:
            self.anchor_modal_xyz = [0.0, 0.0, 0.0]
            self.anchor_modal_known = [False, False, False]
        self.anchor_modal_tag = tag_id
        self.anchor_modal_open = True
        self.anchor_axis_sel = 0   # 默认聚焦 X 轴，方便快速录入
        self.anchor_axis_buf = ""

    def anchor_axis_select(self, axis: int):
        """选中输入轴 (自动提交上一轴缓冲)"""
        self._commit_axis_buffer()
        self.anchor_axis_sel = axis
        self.anchor_axis_buf = ""

    def _commit_axis_buffer(self):
        buf = self.anchor_axis_buf.strip()
        if self.anchor_axis_sel < 0 or not buf:
            self.anchor_axis_buf = ""
            return
        try:
            self.anchor_modal_xyz[self.anchor_axis_sel] = float(buf)
            self.anchor_modal_known[self.anchor_axis_sel] = True
        except ValueError:
            self.set_toast(f"轴 {'XYZ'[self.anchor_axis_sel]} 数值无效: {buf}")
        self.anchor_axis_buf = ""

    def anchor_pad_key(self, label: str):
        """锚点键盘: 数字/小数点入缓冲, '-/+' 切换符号, '清空'/'退格' 编辑, '确认' 提交当前轴"""
        if label == "退格":
            self.anchor_axis_buf = self.anchor_axis_buf[:-1]
            return
        if label == "清空":
            self.anchor_axis_buf = ""
            return
        if label == "确认":
            self._commit_axis_buffer()
            return
        if label == "-/+":
            if self.anchor_axis_buf.startswith("-"):
                self.anchor_axis_buf = self.anchor_axis_buf[1:]
            else:
                self.anchor_axis_buf = "-" + self.anchor_axis_buf
            return
        if label == "." and "." in self.anchor_axis_buf:
            return
        self.anchor_axis_buf = (self.anchor_axis_buf + label)[-12:]

    def anchor_axis_clear(self, axis: int):
        """清除单轴已知标记 (支持部分已知)"""
        self.anchor_modal_known[axis] = False

    def anchor_known_count(self) -> int:
        return sum(1 for b in self.anchor_modal_known if b)

    def save_anchor_modal(self) -> tuple[bool, str]:
        """保存弹窗: 已知轴 ≥1 写穿工位锚点文件；全未知 = 从工位锚点中删除该条目"""
        self._commit_axis_buffer()
        tid = self.anchor_modal_tag
        n = self.anchor_known_count()
        self._reload_anchor_map()
        if n == 0:
            self.anchor_map.pop(tid, None)
            ok = self._persist_anchor_map()
            self._close_anchor_modal()
            return (ok, f"Tag #{tid} 锚点已清除") if ok else (False, "锚点写回失败")
        self.anchor_map[tid] = {"xyz_mm": [float(v) for v in self.anchor_modal_xyz],
                                "known": [bool(b) for b in self.anchor_modal_known]}
        ok = self._persist_anchor_map()
        self._close_anchor_modal()
        if ok:
            return True, f"Tag #{tid} 锚点已保存到本工位 ({n}/3 轴已知)"
        return False, "锚点写回失败"

    def clear_anchor_modal(self) -> tuple[bool, str]:
        """删除当前 Tag 的锚点并写穿工位锚点文件"""
        tid = self.anchor_modal_tag
        self._reload_anchor_map()
        self.anchor_map.pop(tid, None)
        ok = self._persist_anchor_map()
        self._close_anchor_modal()
        return (True, f"Tag #{tid} 锚点已清除") if ok else (False, "锚点写回失败")

    def cancel_anchor_modal(self):
        self._close_anchor_modal()

    # ==================== Tag 公共属性: 标靶物理边长 (marker_size_mm) ====================

    def get_workspace_marker_size(self) -> Optional[float]:
        """获取当前工位显式配置的标靶物理边长 (mm)"""
        ws = self.get_selected_workspace()
        if not ws:
            return None
        from src.calibration.workspace_manager import load_workspace_marker_size_mm
        return load_workspace_marker_size_mm(ws.workspace_dir)

    def open_marker_size_editor(self):
        """打开标靶物理边长专属编辑模态窗"""
        cur_sz = self.get_workspace_marker_size()
        self.marker_size_buf = f"{cur_sz:.3f}" if cur_sz and cur_sz > 0 else ""
        self.marker_size_modal_open = True

    def save_marker_size_modal(self) -> tuple[bool, str]:
        """提交保存标靶物理边长 (写穿工位 tag_whitelist.yaml)"""
        buf = self.marker_size_buf.strip()
        if not buf:
            return False, "标靶边长不能为空"
        try:
            val = float(buf)
        except ValueError:
            return False, f"非法数值: {buf}"
        if val <= 0:
            return False, "标靶边长必须大于 0 mm"
        ws = self.get_selected_workspace()
        if not ws:
            return False, "未选择工位"
        ok, msg = self.workspace_mgr.update_workspace_marker_size(ws.workspace_id, val)
        if ok:
            self.refresh_whitelist_cache()
            self.marker_size_modal_open = False
            self.marker_size_buf = ""
        return ok, msg

    def cancel_marker_size_modal(self):
        """取消标靶物理边长编辑"""
        self.marker_size_modal_open = False
        self.marker_size_buf = ""

    def marker_size_pad_key(self, label: str):
        """标靶边长键盘输入分发"""
        if label == "退格":
            self.marker_size_buf = self.marker_size_buf[:-1]
            return
        if label == "清空":
            self.marker_size_buf = ""
            return
        if label == "确认":
            ok, msg = self.save_marker_size_modal()
            self.set_toast(msg)
            return
        if label == "." and "." in self.marker_size_buf:
            return
        if label in "0123456789.":
            self.marker_size_buf = (self.marker_size_buf + label)[-8:]

    def delete_selected_image(self) -> bool:
        """删除当前选中的照片帧（物理安全移除、清理缓存，并自适应指向相邻帧）"""
        if not self.current_images:
            self.set_toast("当前工位相册为空，无照片可删除。")
            return False

        idx = self.selected_image_idx
        if idx < 0 or idx >= len(self.current_images):
            return False

        img_path = self.current_images[idx]
        file_name = os.path.basename(img_path)

        try:
            if os.path.exists(img_path):
                os.remove(img_path)

            # 清理缩略图与预览图缓存
            keys_to_del = [k for k in self.thumbnail_cache if k.startswith(img_path)]
            for k in keys_to_del:
                self.thumbnail_cache.pop(k, None)
            keys_to_del_prev = [k for k in self.preview_cache if k.startswith(img_path)]
            for k in keys_to_del_prev:
                self.preview_cache.pop(k, None)

            # 重新载入相册列表
            self.load_current_workspace_images()

            # 自适应定位相邻图片
            if self.current_images:
                self.selected_image_idx = min(idx, len(self.current_images) - 1)
            else:
                self.selected_image_idx = 0
            self._ensure_image_visible()

            # 同步更新工位对象的 image_count
            ws = self.get_selected_workspace()
            if ws:
                ws.image_count = len(self.current_images)

            self.set_toast(f"已删除照片: {file_name}")
            return True
        except Exception as e:
            self.set_toast(f"删除照片失败: {e}")
            return False

    def get_thumbnail(self, img_path: str, tw: int = 110, th: int = 70) -> np.ndarray | None:
        """获取缩略图 (带 LRU 内存缓存)"""
        if not os.path.exists(img_path):
            return None
        key = f"{img_path}_{tw}_{th}"
        if key in self.thumbnail_cache:
            self.thumbnail_cache.move_to_end(key)
            return self.thumbnail_cache[key]

        bgr = imread_unicode(img_path)
        if bgr is None:
            return None
        thumb = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)

        # 缓存大小控制
        if len(self.thumbnail_cache) >= self.max_cache_size:
            self.thumbnail_cache.popitem(last=False)
        self.thumbnail_cache[key] = thumb
        return thumb

    def get_preview(self, img_path: str, max_w: int = 440, max_h: int = 280) -> np.ndarray | None:
        """获取单帧高清预览图 (等比例缩放)"""
        if not os.path.exists(img_path):
            return None
        key = f"{img_path}_{max_w}_{max_h}"
        if key in self.preview_cache:
            self.preview_cache.move_to_end(key)
            return self.preview_cache[key]

        bgr = imread_unicode(img_path)
        if bgr is None:
            return None
        h, w = bgr.shape[:2]
        scale = min(max_w / w, max_h / h)
        nw, nh = int(w * scale), int(h * scale)
        prev = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)

        if len(self.preview_cache) >= self.max_cache_size:
            self.preview_cache.popitem(last=False)
        self.preview_cache[key] = prev
        return prev

    def save_capture_frame(self, raw_frame: np.ndarray) -> str:
        """将当前相机帧归档至选中的工位沙盒 raw_images 目录"""
        ws = self.get_selected_workspace()
        if not ws:
            return ""

        os.makedirs(ws.calib_raw_images_dir, exist_ok=True)
        # 获取现有帧的最大序号
        existing = glob.glob(os.path.join(ws.calib_raw_images_dir, "view_*.png"))
        max_idx = 0
        for f in existing:
            base = os.path.basename(f)
            num_part = base.replace("view_", "").replace(".png", "")
            if num_part.isdigit():
                max_idx = max(max_idx, int(num_part))

        new_idx = max_idx + 1
        filename = f"view_{new_idx:04d}.png"
        filepath = os.path.join(ws.calib_raw_images_dir, filename)
        imwrite_unicode(filepath, raw_frame)

        # 触发白闪动效
        self.flash_timer = time.time() + 0.08

        # 刷新工位状态
        ws.refresh_stats()
        ws.save_meta()
        self.load_current_workspace_images()
        self.selected_image_idx = len(self.current_images) - 1
        self.set_toast(f"快照保存成功: {filename} (工位累计 {ws.image_count} 帧)")
        return filepath

    def set_toast(self, msg: str, duration: float = 3.0):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    @property
    def expanded_preview_mode(self) -> bool:
        """当处于全宽大图模式时返回 True"""
        return self.view_mode == self.VIEW_EXPANDED

    @expanded_preview_mode.setter
    def expanded_preview_mode(self, val: bool):
        self.view_mode = self.VIEW_EXPANDED if val else self.VIEW_STANDARD

    def set_view_mode(self, mode: str):
        """显式设定视图模式 (标准页签看板 / 全宽大图沉浸)"""
        if mode in (self.VIEW_STANDARD, self.VIEW_EXPANDED):
            self.view_mode = mode
            names = {
                self.VIEW_STANDARD: "标准页签看板",
                self.VIEW_EXPANDED: "全宽大图沉浸",
            }
            self.set_toast(f"已切换视图模式: 【{names[mode]}】")

    def cycle_view_mode(self):
        """切换视图模式: 标准页签 <-> 全宽大图 (双击卡片触发)"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.set_view_mode(self.VIEW_EXPANDED)

    def toggle_expanded_preview(self):
        """切换全宽大图模式与标准看板模式 (全宽大图仅作用于标定相册页签)"""
        if self.view_mode == self.VIEW_EXPANDED:
            self.set_view_mode(self.VIEW_STANDARD)
        else:
            self.active_tab = self.TAB_CALIB_IMAGES
            self.set_view_mode(self.VIEW_EXPANDED)

    def set_tab(self, tab: str):
        """切换右侧动态区页签 (左栏保持稳定，仅右栏内容动态更新)"""
        if tab not in self.TAB_ORDER:
            return
        if tab == self.active_tab and self.view_mode == self.VIEW_STANDARD:
            return
        # 离开全宽大图沉浸模式
        self.view_mode = self.VIEW_STANDARD
        if tab != self.active_tab:
            self.active_tab = tab
            if tab == self.TAB_PROD_IMAGES:
                self.load_prod_images()
            elif tab == self.TAB_CALIB_IMAGES:
                self.load_current_workspace_images()
            names = {
                self.TAB_CALIB_IMAGES: "标定相册",
                self.TAB_PROD_IMAGES: "生产相册",
                self.TAB_REPORT: "体检报告",
                self.TAB_WHITELIST: "Tag 白名单",
                self.TAB_FRAMES_ROIS: "坐标系&ROI",
            }
            self.set_toast(f"已切换页签: 【{names.get(tab, tab)}】")

    def rename_current_workspace(self, new_name: str) -> bool:
        """重命名当前选中的工位显示名称 (支持中文)"""
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
        """更新当前选中工位的备注说明文本 (支持中文单行文本)"""
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

    def toggle_help_modal(self):
        """打开或关闭 Workspace 工位与生产体系说明弹窗"""
        self.is_help_modal_open = not self.is_help_modal_open
        if self.is_help_modal_open:
            self.set_toast("已呼出【Workspace 工位与生产体系】业务说明窗")
        else:
            self.set_toast("已关闭说明窗。")

    # ---------------- 结构化坐标系表单弹窗方法 ----------------
    def open_frame_modal(self, frame_id: str | None = None):
        if not self.coord_mgr:
            self.load_geometry_managers()
        if not self.coord_mgr:
            self.set_toast("未选择任何工位，无法配置坐标系！")
            return

        if frame_id and frame_id in self.coord_mgr._frames:
            f = self.coord_mgr.get_frame(frame_id)
            self.frame_modal_is_new = False
            self.frame_modal_orig_id = f.frame_id
            b_tags = f.get_tag_ids() if hasattr(f, "get_tag_ids") else ([f.tag_id] if f.tag_id is not None else [])
            tag_str = ",".join(str(x) for x in b_tags) if b_tags else str(f.tag_id or 0)
            
            st = getattr(f, "status", "unknown")
            k_dof = getattr(f, "known_dof", None)
            if k_dof is None:
                if st in ("manual", "calibrated"):
                    k_dof = [True] * 6
                else:
                    k_dof = [False] * 6

            t_val = [float(x) for x in f.translation_xyz_mm] if f.translation_xyz_mm is not None else [0.0, 0.0, 0.0]
            r_val = [float(x) for x in f.rotation_rpy_deg] if f.rotation_rpy_deg is not None else [0.0, 0.0, 0.0]

            self.frame_modal_data = {
                "frame_id": f.frame_id,
                "name": f.name,
                "parent_frame_id": f.parent_frame_id or "world",
                "type": f.type,
                "status": st,
                "known_dof": list(k_dof),
                "translation_xyz_mm": t_val,
                "rotation_rpy_deg": r_val,
                "tag_id": tag_str,
                "offset_xyz_mm": [float(x) for x in f.offset_xyz_mm],
                "offset_rpy_deg": [float(x) for x in f.offset_rpy_deg],
            }
        else:
            self.frame_modal_is_new = True
            self.frame_modal_orig_id = None
            idx = len(self.coord_mgr.list_frames())
            self.frame_modal_data = {
                "frame_id": f"frame_sub_{idx}",
                "name": f"{idx}号机构坐标系",
                "parent_frame_id": "world",
                "type": "fixed_transform",
                "status": "unknown",
                "known_dof": [False] * 6,
                "translation_xyz_mm": [0.0, 0.0, 0.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
                "tag_id": 0,
                "offset_xyz_mm": [0.0, 0.0, 0.0],
                "offset_rpy_deg": [0.0, 0.0, 0.0],
            }
        self.frame_modal_open = True
        self.roi_modal_open = False
        self.active_dropdown = None

    def close_frame_modal(self):
        self.frame_modal_open = False
        self.frame_modal_data = {}
        self.frame_modal_orig_id = None
        self.active_dropdown = None
        self.close_pose6d_modal()

    def save_frame_modal(self) -> tuple[bool, str]:
        if not self.coord_mgr or not self.frame_modal_data:
            return False, "无有效坐标系数据"
        from src.calibration.coordinate_manager import FrameDefinition
        d = self.frame_modal_data
        fid = str(d.get("frame_id", "")).strip()
        if not fid:
            return False, "坐标系 ID 不能为空"

        orig_fid = self.frame_modal_orig_id
        if not self.frame_modal_is_new and orig_fid and orig_fid != fid:
            if orig_fid == "world":
                return False, "绝对世界坐标系禁止修改 ID"
            ok_rename = self.coord_mgr.rename_frame(orig_fid, fid)
            if not ok_rename:
                return False, f"重命名坐标系 ID 失败 (ID '{fid}' 可能已被占用)"
            if self.roi_mgr:
                roi_modified = False
                for r in self.roi_mgr.list_rois():
                    if r.frame_id == orig_fid:
                        r.frame_id = fid
                        roi_modified = True
                if roi_modified:
                    self.roi_mgr.save()
        
        ftype = d.get("type", "fixed_transform")
        parent = None if fid == "world" else d.get("parent_frame_id", "world")
        raw_tid = str(d.get("tag_id", "0")).replace("，", ",").strip()
        parsed_tag_ids = []
        for part in raw_tid.split(","):
            part_s = part.strip()
            if part_s.isdigit():
                parsed_tag_ids.append(int(part_s))
        primary_tid = parsed_tag_ids[0] if parsed_tag_ids else 0

        existing_frame = self.coord_mgr.get_frame(orig_fid) if (not self.frame_modal_is_new and orig_fid) else None
        calib_spec = existing_frame.calibration_spec if existing_frame else None
        calib_metrics = existing_frame.calibration_metrics if existing_frame else None

        st = d.get("status", "unknown")
        k_dof = d.get("known_dof", [False]*6)
        n_known = sum(1 for b in k_dof if b)

        if n_known == 0 and st != "calibrated":
            final_status = "unknown"
            t_xyz = None
            r_rpy = None
        elif n_known == 6:
            final_status = "manual" if st != "calibrated" else "calibrated"
            t_xyz = [float(x) for x in d.get("translation_xyz_mm", [0, 0, 0])]
            r_rpy = [float(x) for x in d.get("rotation_rpy_deg", [0, 0, 0])]
        else:
            final_status = "partial" if st != "calibrated" else "calibrated"
            t_xyz = [float(x) for x in d.get("translation_xyz_mm", [0, 0, 0])]
            r_rpy = [float(x) for x in d.get("rotation_rpy_deg", [0, 0, 0])]

        frame = FrameDefinition(
            frame_id=fid,
            name=str(d.get("name", fid)).strip(),
            parent_frame_id=parent,
            type=ftype,
            status=final_status,
            known_dof=k_dof,
            translation_xyz_mm=t_xyz,
            rotation_rpy_deg=r_rpy,
            calibration_spec=calib_spec,
            calibration_metrics=calib_metrics,
            tag_id=primary_tid,
            tag_ids=parsed_tag_ids,
            offset_xyz_mm=[float(x) for x in d.get("offset_xyz_mm", [0, 0, 0])],
            offset_rpy_deg=[float(x) for x in d.get("offset_rpy_deg", [0, 0, 0])],
        )
        ok = self.coord_mgr.add_frame(frame)
        if not ok:
            return False, "保存坐标系失败 (可能导致拓扑环路或父级不存在)"
        self.coord_mgr.save()
        ws = self.get_selected_workspace()
        if ws:
            self.expanded_workspaces.add(ws.workspace_id)
        self.close_frame_modal()
        self.set_toast(f"已成功保存坐标系: 【{frame.name}】")
        return True, "保存成功"

    # ---------------- 6DoF 外参位姿与约束独立模态弹窗方法 ----------------
    def open_pose6d_modal(self, axis_idx: int = 0):
        """打开 6DoF 外参位姿与约束编辑模态窗"""
        d = self.frame_modal_data
        t = d.get("translation_xyz_mm", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        r = d.get("rotation_rpy_deg", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        st = d.get("status", "unknown")
        default_known = [True]*6 if st in ("manual", "calibrated") else [False]*6
        k_dof = list(d.get("known_dof", default_known))
        if len(k_dof) < 6:
            k_dof = k_dof + [False]*(6 - len(k_dof))

        self.pose6d_modal_vals = [float(t[0]), float(t[1]), float(t[2]), float(r[0]), float(r[1]), float(r[2])]
        self.pose6d_modal_known = [bool(b) for b in k_dof]
        self.pose6d_modal_axis_sel = max(0, min(5, axis_idx))
        self.pose6d_modal_axis_buf = ""
        self.pose6d_modal_open = True

    def close_pose6d_modal(self):
        """关闭 6DoF 模态窗"""
        self.pose6d_modal_open = False
        self.pose6d_modal_axis_buf = ""

    def pose6d_select_axis(self, axis_idx: int):
        """选中指定轴并提交之前轴的输入缓冲区"""
        if self.pose6d_modal_axis_buf:
            try:
                v = float(self.pose6d_modal_axis_buf)
                self.pose6d_modal_vals[self.pose6d_modal_axis_sel] = v
                self.pose6d_modal_known[self.pose6d_modal_axis_sel] = True
            except ValueError:
                pass
            self.pose6d_modal_axis_buf = ""
        self.pose6d_modal_axis_sel = max(0, min(5, axis_idx))

    def pose6d_clear_axis(self, axis_idx: int):
        """将指定轴标记为未知"""
        if self.pose6d_modal_axis_sel == axis_idx:
            self.pose6d_modal_axis_buf = ""
        self.pose6d_modal_known[axis_idx] = False

    def pose6d_toggle_axis(self, axis_idx: int):
        """切换指定轴的已知/未知状态"""
        self.pose6d_modal_known[axis_idx] = not self.pose6d_modal_known[axis_idx]

    def pose6d_set_all_unknown(self):
        """一键设为全未知 (待BA平差反向求解)"""
        self.pose6d_modal_known = [False] * 6
        self.pose6d_modal_axis_buf = ""

    def pose6d_set_all_known(self):
        """一键设为全已知"""
        self.pose6d_modal_known = [True] * 6
        self.pose6d_modal_axis_buf = ""

    def pose6d_set_planar_preset(self):
        """快捷应用水平面运动约束: Roll=0.0°, Pitch=0.0° 已知，其余轴保持"""
        self.pose6d_modal_vals[3] = 0.0
        self.pose6d_modal_known[3] = True
        self.pose6d_modal_vals[4] = 0.0
        self.pose6d_modal_known[4] = True
        self.pose6d_modal_axis_buf = ""

    def pose6d_pad_key(self, label: str):
        """处理 15 键软键盘按键输入"""
        axis = self.pose6d_modal_axis_sel
        buf = self.pose6d_modal_axis_buf

        if label == "确认":
            if buf:
                try:
                    v = float(buf)
                    self.pose6d_modal_vals[axis] = v
                    self.pose6d_modal_known[axis] = True
                except ValueError:
                    pass
                self.pose6d_modal_axis_buf = ""
            self.pose6d_modal_axis_sel = (axis + 1) % 6
            return

        if label == "清空":
            self.pose6d_modal_axis_buf = ""
            return

        if label == "退格":
            if buf:
                self.pose6d_modal_axis_buf = buf[:-1]
            return

        if label == "-/+":
            if buf.startswith("-"):
                self.pose6d_modal_axis_buf = buf[1:]
            else:
                self.pose6d_modal_axis_buf = "-" + buf
            return

        if label == ".":
            if "." not in buf:
                self.pose6d_modal_axis_buf = (buf if buf else "0") + "."
            return

        if label.isdigit():
            if buf == "0":
                self.pose6d_modal_axis_buf = label
            else:
                self.pose6d_modal_axis_buf = buf + label
            try:
                self.pose6d_modal_vals[axis] = float(self.pose6d_modal_axis_buf)
                self.pose6d_modal_known[axis] = True
            except ValueError:
                pass

    def save_pose6d_modal(self):
        """保存 6DoF 位姿与约束回 frame_modal_data"""
        if self.pose6d_modal_axis_buf:
            try:
                v = float(self.pose6d_modal_axis_buf)
                self.pose6d_modal_vals[self.pose6d_modal_axis_sel] = v
                self.pose6d_modal_known[self.pose6d_modal_axis_sel] = True
            except ValueError:
                pass
            self.pose6d_modal_axis_buf = ""

        d = self.frame_modal_data
        d["translation_xyz_mm"] = [round(float(x), 2) for x in self.pose6d_modal_vals[:3]]
        d["rotation_rpy_deg"] = [round(float(x), 2) for x in self.pose6d_modal_vals[3:]]
        d["known_dof"] = list(self.pose6d_modal_known)

        n_known = sum(1 for b in self.pose6d_modal_known if b)
        if n_known == 0:
            d["status"] = "unknown"
            self.set_toast("已应用外参配置: 【全未知】(待 BA 平差反向求解)")
        elif n_known == 6:
            d["status"] = "manual"
            self.set_toast("已应用外参配置: 【全已知人工指定】")
        else:
            d["status"] = "partial"
            self.set_toast(f"已应用外参配置: 【部分已知先验约束】(已知 {n_known}/6 轴)")

        self.close_pose6d_modal()


    def delete_frame(self, frame_id: str) -> tuple[bool, str]:
        if not self.coord_mgr:
            return False, "坐标系管理器未就绪"
        if frame_id == "world":
            return False, "绝对世界坐标系禁止删除"
        ok = self.coord_mgr.remove_frame(frame_id)
        if ok:
            self.coord_mgr.save()
            self.set_toast(f"已成功删除坐标系: {frame_id}")
            return True, "删除成功"
        return False, "删除失败"

    # ---------------- 结构化 ROI 表单弹窗方法 ----------------
    def open_roi_modal(self, roi_id: str | None = None):
        if not self.roi_mgr:
            self.load_geometry_managers()
        if not self.roi_mgr:
            self.set_toast("未选择任何工位，无法配置 ROI！")
            return

        frames = self.get_coordinate_frames()
        default_frame = frames[1].frame_id if len(frames) > 1 else "world"

        if roi_id and roi_id in self.roi_mgr._rois:
            r = self.roi_mgr.get_roi(roi_id)
            self.roi_modal_is_new = False
            self.roi_modal_orig_id = r.roi_id
            self.roi_modal_data = {
                "roi_id": r.roi_id,
                "name": r.name,
                "frame_id": r.frame_id,
                "category": r.category,
                "center_xyz_mm": [float(x) for x in r.center_xyz_mm],
                "size_xyz_mm": [float(x) for x in r.size_xyz_mm],
                "rotation_rpy_deg": [float(x) for x in r.rotation_rpy_deg],
                "visual_color_rgb": [int(c) for c in (r.visual_color_rgb or [0, 255, 128])],
            }
        else:
            self.roi_modal_is_new = True
            self.roi_modal_orig_id = None
            idx = len(self.roi_mgr.list_rois()) + 1
            self.roi_modal_data = {
                "roi_id": f"roi_part_{idx}",
                "name": f"{idx}号机构部件空间",
                "frame_id": default_frame,
                "category": "belt",
                "center_xyz_mm": [0.0, 100.0, 20.0],
                "size_xyz_mm": [80.0, 200.0, 30.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
                "visual_color_rgb": [0, 255, 128],
            }
        self.roi_modal_open = True
        self.frame_modal_open = False
        self.active_dropdown = None

    def close_roi_modal(self):
        self.roi_modal_open = False
        self.roi_modal_data = {}
        self.roi_modal_orig_id = None
        self.active_dropdown = None

    def save_roi_modal(self) -> tuple[bool, str]:
        if not self.roi_mgr or not self.roi_modal_data:
            return False, "无有效 ROI 数据"
        from src.calibration.roi_manager import RoiDefinition
        d = self.roi_modal_data
        rid = str(d.get("roi_id", "")).strip()
        if not rid:
            return False, "ROI ID 不能为空"

        orig_rid = self.roi_modal_orig_id
        if not self.roi_modal_is_new and orig_rid and orig_rid != rid:
            ok_rename = self.roi_mgr.rename_roi(orig_rid, rid)
            if not ok_rename:
                return False, f"重命名 ROI ID 失败 (ID '{rid}' 可能已被占用)"
        
        # 强 Schema 尺寸校验: dx, dy, dz 必须 > 0
        sizes = [float(x) for x in d.get("size_xyz_mm", [10, 10, 10])]
        if any(s <= 0 for s in sizes):
            return False, "空间尺寸 (长宽高) 必须严格大于 0"

        roi = RoiDefinition(
            roi_id=rid,
            name=str(d.get("name", rid)).strip(),
            frame_id=str(d.get("frame_id", "world")),
            category=str(d.get("category", "general")),
            enabled=True,
            center_xyz_mm=[float(x) for x in d.get("center_xyz_mm", [0, 0, 0])],
            size_xyz_mm=sizes,
            rotation_rpy_deg=[float(x) for x in d.get("rotation_rpy_deg", [0, 0, 0])],
            visual_color_rgb=[int(c) for c in d.get("visual_color_rgb", [0, 255, 128])],
        )
        self.roi_mgr.add_roi(roi)
        self.roi_mgr.save()
        self.close_roi_modal()
        self.set_toast(f"已成功保存 3D ROI 物件: 【{roi.name}】")
        return True, "保存成功"

    def delete_roi(self, roi_id: str) -> tuple[bool, str]:
        if not self.roi_mgr:
            return False, "ROI 管理器未就绪"
        ok = self.roi_mgr.remove_roi(roi_id)
        if ok:
            self.roi_mgr.save()
            self.scroll_roi_list(0)
            self.set_toast(f"已成功删除 ROI 物件: {roi_id}")
            return True, "删除成功"
        return False, "删除失败"

    def get_selected_frame(self):
        """获取当前树导航选中的机构坐标系对象"""
        item_type, ws_idx, frame_id = self.selected_tree_item
        if item_type == "frame" and frame_id and self.coord_mgr:
            return self.coord_mgr.get_frame(frame_id)
        return None

    def set_tab(self, tab: str):
        """切换当前激活的页签 (若处于全屏大图预览则自动回退至标准视图)"""
        self.active_tab = tab
        if self.view_mode == self.VIEW_EXPANDED:
            self.view_mode = self.VIEW_STANDARD

