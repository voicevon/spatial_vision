#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 空间建图工作站 (Spatial Mapping Studio)
=====================================================
旗舰级资产中心化 (Asset-Centric) 离线标定工作台：
  - 100% 装配复用底层核心领域模型：
      - ManifestRepository (清单与地图仓储 / 剔除状态同步)
      - OfflineEngine (PnP 求解 / 重投影残差 / LOO 盲测)
      - BundleAdjustmentOptimizer (两阶段 Cauchy BA 全局平差)
      - VerificationReporter (残差分析 / MAD 粗差诊断 / 质检报告)
      - VerificationVisualizer (3D 双四棱柱立体对比 / 2D 残差矢量)
  - 1920x1080 工业级三栏排版 (左侧高密紧凑帧列表、中央高清视口、右侧属性诊断面板、底栏全局调度)
  - 异步 BA 全局平差，前台丝滑响应，求解完成后就地热重载地图并即时刷新全量帧残差数值。

模块拆分结构 (上帝文件拆分)：
  - 本模块 (app.py): SpatialMappingStudioApp 核心控制器 (装配 / 属性代理 / 数据委托 / 渲染委托 / 主循环)
  - mapping_events.py: MappingEventMixin (鼠标事件命中测试与 GUI 按钮分发)
  - mapping_workflows.py: MappingWorkflowMixin (异步超精提取 / 智能剪枝 / BA 平差 / 发布与质检报告)
  - mapping_app_meta.py: 跨模块共享常量 (PROJECT_ROOT)
"""

import os
import sys
import time
import json
import atexit
import threading
import argparse
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import cv2

# Windows 终端色彩与编码适配
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass  # 编码重配置失败无伤大雅，终端仍可正常运行

# 共享常量 PROJECT_ROOT 从 mapping_app_meta 导入 (Mixin 模块亦引用，避免双处定义不一致)
try:
    from tools.spatial_mapping_studio.mapping_app_meta import PROJECT_ROOT
except ImportError:
    # 以脚本方式直接运行本文件时的回退导入 (同目录)
    from mapping_app_meta import PROJECT_ROOT
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.manifest_repository import ManifestRepository
from src.vision.pnp_solver import PnpSolver
from src.vision.tag_detector import TagDetector
from src.calibration.solvers.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.verification.verification_reporter import VerificationReporter
from src.calibration.verification.verification_visualizer import VerificationVisualizer
from src.workspace.workspace_manager import (
    load_workspace_coordinate_manager,
    load_workspace_marker_size_mm,
    load_workspace_roi_manager,
)
from tools.spatial_mapping_studio.mapping_state import MappingDataManager
from tools.spatial_mapping_studio.mapping_viewport_interactor import MappingViewportInteractor
from tools.spatial_mapping_studio.mapping_ba_runner import MappingBARunner
from tools.spatial_mapping_studio.mapping_renderer import MappingRenderer
from src.ui.viewport_manager import (
    ViewportManager
)
from src.ui.gui_components import ScrollableListBox
from src.utils.logger import get_logger
from tools.spatial_mapping_studio.mapping_events import MappingEventMixin
from tools.spatial_mapping_studio.mapping_workflows import MappingWorkflowMixin
from src.workspace.workspace_manager import WorkspaceManager

try:
    from tools.window_helper import force_window_focus
except ImportError:
    force_window_focus = None


try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

log = get_logger(__name__)

_ws_fallback = os.path.join(PROJECT_ROOT, "data", "workspaces", "default", "calibration")
CALIB_IMAGES_DIR = os.path.join(_ws_fallback, "raw_images")
DEFAULT_MAP_PATH = os.path.join(PROJECT_ROOT, "data", "workspaces", "default", "tags_map.yaml")
MANIFEST_PATH = os.path.join(_ws_fallback, "tag_observations.yaml")

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
APP_ID = "spatial_mapping_studio"
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")


class SpatialMappingStudioApp(MappingEventMixin, MappingWorkflowMixin):

    """AprilTag 离线标定综合工作站 (Spatial Mapping Studio) 控制器
    (事件交互职责见 MappingEventMixin，异步工作流职责见 MappingWorkflowMixin)"""

    def __init__(
        self,
        map_path: str = None,
        image_dir: str = None,
        marker_size_mm: Optional[float] = None,
        win_w: int = 1920,
        win_h: int = 1080,
        manifest_path: Optional[str] = None,
        workspace_id: Optional[str] = None,
        settings_file: Optional[str] = None
    ):
        self.win_w = win_w
        self.win_h = win_h
        self.settings_file = settings_file or GUI_SETTINGS_FILE

        # 工位管理器感知与初始目标工位装配
        try:
            self.workspace_mgr = WorkspaceManager()
            target_ws_id = workspace_id
            if not target_ws_id and not image_dir:
                try:
                    if os.path.exists(self.settings_file):
                        with open(self.settings_file, "r", encoding="utf-8") as f:
                            root = json.load(f)
                        saved_ws_id = (root.get(APP_ID) or {}).get("dropdown_state", {}).get("workspace_id")
                        if saved_ws_id and self.workspace_mgr.get_workspace_by_id(saved_ws_id):
                            target_ws_id = saved_ws_id
                except Exception:
                    pass

            if target_ws_id:
                ws = self.workspace_mgr.get_workspace_by_id(target_ws_id)
            elif image_dir:
                norm_target = os.path.normpath(image_dir)
                ws = next((s for s in self.workspace_mgr.list_workspaces()
                           if os.path.normpath(s.calib_raw_images_dir) == norm_target or os.path.normpath(s.workspace_dir) == norm_target), None)
            else:
                ws = self.workspace_mgr.get_current_workspace()

            if not ws:
                ws = self.workspace_mgr.get_current_workspace()
            self.current_workspace = ws
            self.current_workspace_id = ws.workspace_id if ws else ""
            if ws:
                self.workspace_mgr.set_current_workspace(ws.workspace_id)
        except Exception:
            self.workspace_mgr = None
            self.current_workspace = None
            self.current_workspace_id = ""

        # 标靶物理边长解析 (FR-9.x: 硬编码默认值已废除, 唯一权威来源 = 当前工位 tag_whitelist.yaml.tag_default_size_mm)
        if marker_size_mm is None:
            ws_dir = self.current_workspace.workspace_dir if self.current_workspace else None
            loaded = load_workspace_marker_size_mm(ws_dir) if ws_dir else None
            if loaded is None:
                raise ValueError(
                    "未指定标靶物理边长 (marker_size_mm), 且当前工位 tag_whitelist.yaml.tag_default_size_mm 缺失或非法. "
                    "请在 tag_whitelist.yaml 显式录入 tag_default_size_mm (mm) 后重试, 或显式传入 marker_size_mm 参数."
                )
            marker_size_mm = loaded
        self.marker_size_mm = marker_size_mm

        self.map_path = map_path or (self.current_workspace.map_path if self.current_workspace else DEFAULT_MAP_PATH)
        self.image_dir = image_dir or (self.current_workspace.calib_raw_images_dir if self.current_workspace else CALIB_IMAGES_DIR)
        self.manifest_path = manifest_path or (self.current_workspace.calib_manifest_path if self.current_workspace else os.path.join(self.image_dir, "tag_observations.yaml"))
        self.manifest_repo = ManifestRepository()

        # 1. 初始化视口管理器与物理布局尺寸
        self.viewport = ViewportManager(win_w=win_w, win_h=win_h, top_bar_h=44, bottom_bar_h=52)
        self.left_bar_w = 340   # 左侧紧凑列表宽度
        self.right_bar_w = 180  # 右侧精简属性栏宽度 (瘦身至 180px，充分释放主视口空间)

        # 2. 相机内参与领域模型装配
        self.camera_matrix, self.dist_coeffs = self._load_camera_intrinsics()

        self.pnp_solver = PnpSolver(
            tags_map={},
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm
        )
        self.tag_detector = TagDetector()
        self.visualizer = VerificationVisualizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs
        )
        self.reporter = VerificationReporter()
        self.optimizer = BundleAdjustmentOptimizer(
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            marker_size_mm=self.marker_size_mm
        )

        # 3. 领域与数据状态管理器 (从主类中独立解耦)
        self.data_mgr = MappingDataManager(
            map_path=self.map_path,
            image_dir=self.image_dir,
            manifest_path=self.manifest_path,
            pnp_solver=self.pnp_solver,
            marker_size_mm=self.marker_size_mm,
            tag_detector=self.tag_detector
        )

        # 视口显示双独立正交模式 (用户指定默认 3d)
        self.ba_view_mode: str = "3d"    # BA 理论值: "3d" (翡翠绿棱柱), "2d" (投影框), "off" (隐藏)
        self.obs_view_mode: str = "3d"   # 实测识别值: "3d" (天蓝棱柱), "2d" (实测角点框), "off" (隐藏)

        # 下拉菜单展开态标识 ("FILTER_DROPDOWN", "SORT_DROPDOWN", "BA_VIEW_DROPDOWN", "OBS_VIEW_DROPDOWN" 或 None)
        self.active_dropdown: Optional[str] = None
        self.dropdown_boxes: Dict[str, Dict[str, Any]] = {}

        # 4. 文件列表多轮残差演进矩阵视图模式 (Matrix View)
        self.matrix_view_mode: bool = False

        # 4b. ROI 物件绘制模式 (子工具栏切换)
        self.draw_roi_mode: bool = False

        # 4d. 坐标系可见性字典 {frame_id: bool} —— 控制哪些坐标系显示 XY 平面和三轴正向
        # 默认: world 坐标系开启，其余隐藏
        self.coord_frame_visibility: dict = {"world": True}
        # 4e. 当前主参考基准坐标系 (默认 'world', 可切换为任意已解算子坐标系如 'frame_sub_1')
        self.active_reference_frame_id: str = "world"

        # 4c. 工位多坐标系树与 ROI 空间物件集合管理器 (供 ROI 绘制场景使用)
        self.coord_mgr = None
        self.roi_mgr = None
        if self.current_workspace:
            self._load_workspace_geometry()

        # 5. 异步 BA 全局平差任务调度器
        self.ba_runner = MappingBARunner(
            data_mgr=self.data_mgr,
            optimizer=self.optimizer,
            manifest_repo=self.manifest_repo,
            map_path=self.map_path,
            manifest_path=self.manifest_path,
            marker_size_mm=self.marker_size_mm,
            on_status_change=self.set_toast,
            workspace=self.current_workspace
        )

        # 6. 单帧深度病因切片诊断开关
        self.show_frame_diagnostics = False

        # 7. 浮层通知 (Toast): 持久显示直至用户手动点击 ❌ 关闭
        self.status_toast = "欢迎进入空间建图工作站 (Spatial Mapping Studio)"
        self.status_toast_time = time.time()
        self.toast_sticky = True  # True = 持久显示, 仅用户点击 ❌ 才关闭

        # 7b. 世界坐标系对齐质检单报告卡片数据
        self.alignment_report: Optional[Dict[str, Any]] = (
            (self.data_mgr.tags_map_data or {}).get("world_anchor", {}).get("alignment_report")
            if getattr(self, "data_mgr", None) else None
        )
        self.alignment_report_sort: str = "id"  # "id" 或 "err_desc"

        # 8. GUI 交互按钮注册表
        self.gui_buttons: List[Tuple[str, Tuple[int, int, int, int], Any]] = []
        self.mouse_pos = (-1, -1)
        self.is_running = True

        # 9. 视口几何变换与鼠标交互控制器 (Viewport Zoom & Pan)
        self.viewport = MappingViewportInteractor(win_w=self.win_w, win_h=self.win_h)

        # 10. UI 界面排版与视觉渲染器
        self.ui_renderer = MappingRenderer()

        # 11. 异步全量超精提取任务状态调度
        self.is_extracting_all: bool = False
        self.extract_progress: float = 0.0
        self.extract_stage_text: str = ""
        self.extract_thread: Optional[threading.Thread] = None
        self.extract_result_queue: Optional[Tuple[bool, str]] = None

        # 12. 工业级可滚动列表组件 (左栏帧列表 & 右栏标靶残差清单)
        self.frame_list_box = ScrollableListBox(
            item_height=36,
            item_gap=2,
            scrollbar_width=6,
            auto_hide_scrollbar=True,
            render_item_background=True,
            scroll_speed=2,
        )
        self.tag_list_box = ScrollableListBox(
            item_height=38,
            item_gap=2,
            scrollbar_width=6,
            auto_hide_scrollbar=True,
            render_item_background=True,
            scroll_speed=2,
        )

        # 首次预热并计算全集残差指标
        self.data_mgr.refresh_all_frame_metrics()

        # 13. 从持久化配置恢复各下拉选项状态
        self._load_dropdown_state()

        # 注册退出持久化钩子保底
        try:
            atexit.register(self.save_dropdown_state)
        except Exception:
            pass

    @property
    def image_files(self) -> List[str]:
        """代理获取当前工位所有采图路径列表"""
        return self.data_mgr.image_files if getattr(self, "data_mgr", None) else []

    # ------------------------------ 下拉选项持久化 ------------------------------
    def _load_dropdown_state(self):
        """从 settings_file (config/gui_settings.json) 恢复建图工作站下拉选择偏好"""
        try:
            if not os.path.exists(self.settings_file):
                return
            with open(self.settings_file, "r", encoding="utf-8") as f:
                root = json.load(f)
            if not isinstance(root, dict):
                return
            node = root.get(APP_ID, {})
            if not isinstance(node, dict):
                return
            state = node.get("dropdown_state") or node.get("viewer_state") or {}
            if not isinstance(state, dict):
                return

            from tools.spatial_mapping_studio.mapping_ui_common import (
                FILTER_MODE_OPTIONS,
                SORT_MODE_OPTIONS,
                BA_VIEW_OPTIONS,
                OBS_VIEW_OPTIONS,
            )

            # 1. 筛选范围下拉 (filter_mode)
            valid_filters = [k for k, _ in FILTER_MODE_OPTIONS]
            if state.get("filter_mode") in valid_filters:
                self.data_mgr.filter_mode = state["filter_mode"]

            # 2. 排序方式下拉 (sort_mode)
            valid_sorts = [k for k, _ in SORT_MODE_OPTIONS]
            if state.get("sort_mode") in valid_sorts:
                self.data_mgr.sort_mode = state["sort_mode"]

            # 3. BA 理论视口显示下拉 (ba_view_mode)
            valid_ba = [k for k, _ in BA_VIEW_OPTIONS]
            if state.get("ba_view_mode") in valid_ba:
                self.ba_view_mode = state["ba_view_mode"]

            # 4. 实测识别视口显示下拉 (obs_view_mode)
            valid_obs = [k for k, _ in OBS_VIEW_OPTIONS]
            if state.get("obs_view_mode") in valid_obs:
                self.obs_view_mode = state["obs_view_mode"]

            # 5. Z 轴特殊点 / XY 平面显示下拉 (plane_z, show_xy_plane)
            if "show_xy_plane" in state:
                self.data_mgr.show_xy_plane_on = bool(state["show_xy_plane"])
            if "plane_z" in state:
                try:
                    self.data_mgr.plane_z = float(state["plane_z"])
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            log.warning(f"恢复建图工作站下拉偏好设置失败，使用默认配置: {e}")

    def save_dropdown_state(self):
        """保存当前建图工作站下拉选择偏好至 settings_file (config/gui_settings.json)"""
        try:
            root = {}
            if os.path.exists(self.settings_file):
                try:
                    with open(self.settings_file, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        root = loaded
                except Exception:
                    root = {}

            node = root.setdefault(APP_ID, {})
            state_data = {
                "workspace_id": str(self.current_workspace_id or ""),
                "filter_mode": str(self.data_mgr.filter_mode),
                "sort_mode": str(self.data_mgr.sort_mode),
                "ba_view_mode": str(self.ba_view_mode),
                "obs_view_mode": str(self.obs_view_mode),
                "show_xy_plane": bool(self.data_mgr.show_xy_plane_on),
                "plane_z": float(self.data_mgr.plane_z),
            }
            node["dropdown_state"] = state_data
            node["viewer_state"] = state_data.copy()
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
            with open(self.settings_file, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存建图工作站下拉偏好设置失败: {e}")

    @property
    def workspace_options(self):
        """动态读取所有可用工位供顶栏下拉菜单展示"""
        if not self.workspace_mgr:
            return []
        opts = []
        for s in self.workspace_mgr.list_workspaces():
            opts.append((s.workspace_id, f"{s.name} ({s.image_count}帧)"))
        return opts

    @property
    def current_workspace_name(self):
        return self.current_workspace.name if self.current_workspace else "默认工位"

    def switch_workspace(self, workspace_id: str):
        """实时热切换工位：重新装载图像、清单与地图并复位视口与平差引擎"""
        if not self.workspace_mgr:
            return
        target_ws = self.workspace_mgr.get_workspace_by_id(workspace_id)
        if not target_ws:
            return

        # 1. 自动持久化当前工位已修改数据
        try:
            self.data_mgr.save_manifest()
        except Exception:
            pass

        # 2. 同步运行时当前工位 (进程内共享, 不落盘)
        self.workspace_mgr.set_current_workspace(target_ws.workspace_id)

        # 3. 重新指向新工位
        self.current_workspace = target_ws
        self.current_workspace_id = target_ws.workspace_id
        self.image_dir = target_ws.calib_raw_images_dir
        self.manifest_path = target_ws.calib_manifest_path
        self.map_path = target_ws.map_path

        # 3. 驱动 data_mgr 重载
        self.data_mgr.reload_dataset(
            map_path=self.map_path,
            image_dir=self.image_dir,
            manifest_path=self.manifest_path
        )

        # 4. 更新 BA 调度器中的路径与工位引用
        self.ba_runner.map_path = self.map_path
        self.ba_runner.manifest_path = self.manifest_path
        self.ba_runner.workspace = self.current_workspace
        self.ba_runner._load_alignment_config()  # 重新装载新工位的锚点配置

        # 5. 重新装载当前工位的多坐标系与 ROI 空间物件管理器
        self._load_workspace_geometry()

        # 6. 同步更新/重置世界坐标系对齐质检单（避免残留旧工位质检卡片）
        new_rep = (self.data_mgr.tags_map_data or {}).get("world_anchor", {}).get("alignment_report")
        self.alignment_report = new_rep if new_rep else None

        self.set_toast(f"已热重载切换至场景: 【{target_ws.name}】(共 {len(self.data_mgr.image_files)} 帧)")
        log.info(f"[SPATIAL_MAPPING] 成功切换场景至: {target_ws.name} ({target_ws.workspace_id})")
        self.save_dropdown_state()

    def _load_workspace_geometry(self):
        """从当前工位装载坐标系树与 ROI 空间物件集合管理器 (缺失则空)"""
        if not self.current_workspace:
            self.coord_mgr = None
            self.roi_mgr = None
            return
        try:
            self.coord_mgr = load_workspace_coordinate_manager(self.current_workspace)
            self.roi_mgr = load_workspace_roi_manager(self.current_workspace)
            if self.coord_mgr and getattr(self.coord_mgr, "active_frame_id", None):
                self.active_reference_frame_id = self.coord_mgr.active_frame_id
            else:
                self.active_reference_frame_id = "world"
        except Exception as e:
            log.warning(f"[SPATIAL_MAPPING] 装载工位 ROI/坐标系失败: {e}")
            self.coord_mgr = None
            self.roi_mgr = None

    def set_active_reference_frame(self, frame_id: str):
        """切换当前视口观察基准坐标系 (如 'world' 或 'frame_sub_1')"""
        if self.coord_mgr and frame_id in self.coord_mgr._frames:
            self.active_reference_frame_id = frame_id
            frame = self.coord_mgr.get_frame(frame_id)
            name = frame.name if frame else frame_id
            self.set_toast(f"基准参考坐标系已切换为: {name} ({frame_id})")
        else:
            self.active_reference_frame_id = "world"
            self.set_toast("基准参考坐标系已复位为: 绝对世界系 (world)")

    def reset_viewport_zoom(self):
        """重置中间视口缩放与平移状态为适应屏幕 (1.0x)"""
        self.viewport.reset()
        self.set_toast("视口缩放已重置 (1.0x 适应视口)")

    def set_toast(self, msg: str):
        self.status_toast = msg
        self.status_toast_time = time.time()
        self.toast_sticky = True  # 每次弹出新消息均持久显示, 等待用户手动关闭

    def dismiss_toast(self):
        """用户手动点击 ❌ 关闭当前 Toast 消息"""
        self.toast_sticky = False
        self.status_toast = ""

    def copy_toast(self):
        """将当前 Toast 消息文本复制到系统剪贴板"""
        text = self.status_toast
        if not text:
            return
        try:
            import subprocess
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
            process.communicate(text.encode('utf-16-le'))
            self.set_toast(f"已复制到剪贴板: {text}")
        except Exception as e:
            log.warning(f"[SPATIAL_MAPPING] 复制到剪贴板失败: {e}")

    def copy_alignment_report(self):
        """将当前世界坐标系对齐质检单复制为 Markdown 文本到系统剪贴板"""
        if not self.alignment_report:
            return
        rep = self.alignment_report
        lines = [
            "# 世界坐标系对齐质检单 (World Datum Alignment Report)",
            f"- **解算算法**: `{rep.get('solver_type', 'Umeyama 3D')}`",
            f"- **均值物理残差**: `{rep.get('mean_mm', 0.0):.2f} mm`",
            f"- **最大物理残差**: `{rep.get('max_mm', 0.0):.2f} mm`",
            f"- **整体质检结论**: {'⚠️ 存在超标标靶 (请核对输入坐标)' if rep.get('has_warn') else '🟢 全部标靶优良吻合'}",
            "",
            "| 标靶 ID | 设定世界坐标 (X, Y, Z) mm | 实测对齐坐标 (X, Y, Z) mm | 分轴偏差 (ΔX, ΔY, ΔZ) mm | 3D 绝对残差 | 质检状态 |",
            "| :---: | :--- | :--- | :--- | :---: | :---: |"
        ]
        rows = list(rep.get("rows", []))
        if self.alignment_report_sort == "err_desc":
            rows = sorted(rows, key=lambda r: r.get("dist_3d_mm", 0.0), reverse=True)
        else:
            rows = sorted(rows, key=lambda r: r.get("tag_id", 0))

        for row in rows:
            tag_str = f"Tag #{row['tag_id']}"
            t_vals = [f"{v:.1f}" if v is not None else "--" for v in row.get("target_xyz", [])]
            tgt = f"({', '.join(t_vals)})"
            f_vals = [f"{v:.1f}" if v is not None else "--" for v in row.get("fitted_xyz", [])]
            fit = f"({', '.join(f_vals)})"
            delta = f"({', '.join(str(v) for v in row.get('delta_xyz', []))})"
            dist = f"{row.get('dist_3d_mm', 0.0):.2f} mm"
            stat = "⚠️ 偏差过大" if row.get("is_warn") else "🟢 吻合"
            lines.append(f"| {tag_str} | {tgt} | {fit} | {delta} | {dist} | {stat} |")

        conflict_pairs = rep.get("conflict_pairs", [])
        if conflict_pairs:
            from src.calibration.solvers.world_datum_aligner import format_conflict_pairs_report
            conflict_diag = format_conflict_pairs_report(conflict_pairs)
            if conflict_diag:
                lines.append("")
                lines.append("## 锚点几何形变与测距冲突报告")
                lines.append(conflict_diag)

        text = "\n".join(lines)
        if getattr(self, "latest_milestone_markdown", None):
            text = f"{self.latest_milestone_markdown}\n\n---\n\n{text}"
        try:
            import subprocess
            process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, shell=True)
            process.communicate(text.encode('utf-16-le'))
            self.set_toast("已将世界系对齐质检单复制到剪贴板！")
        except Exception as e:
            log.warning(f"[SPATIAL_MAPPING] 复制质检单失败: {e}")


    @property
    def toast_msg(self) -> str:
        return self.status_toast

    def _load_camera_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载相机内参和畸变参数"""
        if resolve_camera_intrinsics is not None:
            K, dist, _ = resolve_camera_intrinsics(CONFIG_PATH)
            return K, dist
        else:
            K = np.array([
                [1363.68, 0.0, 971.19],
                [0.0, 1361.19, 566.26],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist = np.zeros(5, dtype=np.float64)
            return K, dist




    def select_frame(self, idx: int):
        """选定指定索引的图像帧并复位标靶列表滚动偏移"""
        self.data_mgr.current_img_idx = idx
        if hasattr(self, "tag_list_box"):
            self.tag_list_box.scroll_offset = 0

    def toggle_current_frame_exclusion(self):
        bname, is_excl = self.data_mgr.toggle_current_frame_exclusion()
        if bname:
            status_str = "已标记为 [剔除/EXCLUDED]" if is_excl else "已恢复为 [保留/ACTIVE]"
            self.set_toast(f"帧 {bname} {status_str}")
            log.info(f"[*] 帧状态翻转: {bname} -> {status_str}")

    def toggle_tag_exclusion_in_current_frame(self, target_tag_id: int):
        bname, is_kept = self.data_mgr.toggle_tag_exclusion_in_current_frame(target_tag_id)
        if bname:
            t_str = "已保留" if is_kept else "已剔除 (打叉)"
            self.set_toast(f"标靶 Tag #{target_tag_id} 在本帧中 {t_str}")


    @property
    def dynamic_left_bar_w(self) -> int:
        """根据当前是否处于矩阵视图动态计算左栏排版宽度"""
        if not self.matrix_view_mode:
            return 340
        headers = getattr(self.data_mgr, "convergence_headers", [])
        num_cols = max(1, len(headers))
        # 基础列宽 175px (状态点+文件名+标靶数) + 各轮次列 (56px/列) + 降幅列 (72px)
        calc_w = 175 + num_cols * 56 + 72
        return min(960, max(640, calc_w))

    def toggle_matrix_view_mode(self):
        """一键切换左栏文件列表的单列紧凑视图与多轮残差演进矩阵宽表大视图"""
        self.matrix_view_mode = not self.matrix_view_mode
        self.left_bar_w = self.dynamic_left_bar_w
        if self.matrix_view_mode:
            self.set_toast("已切换为: 逐帧多轮残差演进矩阵大表 (Matrix View)")
        else:
            self.set_toast("已切换为: 紧凑图像帧列表 (Compact View)")

    def toggle_draw_roi_mode(self):
        """一键切换 ROI 物件绘制模式 (供子工具栏 [绘制ROI物件] 按钮调用)"""
        self.draw_roi_mode = not self.draw_roi_mode
        if self.draw_roi_mode:
            self.set_toast("ROI 物件绘制模式：已开启 (下一步在中央视口框选区域)")
        else:
            self.set_toast("ROI 物件绘制模式：已关闭")

    def toggle_roi_enabled(self, roi_id: str):
        """切换指定 ROI 的 enabled 状态，并立即持久化写盘至 spatial_scene.yaml (方案 A)"""
        if not self.roi_mgr:
            return
        roi = self.roi_mgr.get_roi(roi_id)
        if roi is None:
            return
        roi.enabled = not roi.enabled
        ok = self.roi_mgr.save()
        state_str = "已启用 ✓" if roi.enabled else "已隐藏 ✗"
        name_str = roi.name or roi_id
        if ok:
            self.set_toast(f"ROI [{name_str}] {state_str} (已写盘)")
        else:
            self.set_toast(f"ROI [{name_str}] {state_str} (⚠ 写盘失败)")
        log.info(f"[SPATIAL_MAPPING] toggle_roi_enabled: {roi_id} → enabled={roi.enabled}")

    def toggle_coord_frame_visibility(self, frame_id: str):
        """切换指定坐标系的显示/隐藏（控制 XY 平面 + 三轴正向显示）"""
        current = self.coord_frame_visibility.get(frame_id, False)
        self.coord_frame_visibility[frame_id] = not current
        state_str = "已显示" if self.coord_frame_visibility[frame_id] else "已隐藏"
        self.set_toast(f"坐标系 [{frame_id}] {state_str}")
        log.info(f"[SPATIAL_MAPPING] toggle_coord_frame_visibility: {frame_id} → {self.coord_frame_visibility[frame_id]}")


    # ===================== 渲染管线 (三栏自适应排版) =====================

    def render(self, canvas: np.ndarray):
        """完整渲染 Spatial Mapping Studio 的顶栏、左栏列表、中间视口、右栏诊断与底栏"""
        self.ui_renderer.render(self, canvas)


    # ===================== 事件分发与主循环 =====================

    def toggle_frame_diagnostics(self):
        """唤起/关闭当前选定帧的漏检病因深度切片诊断视图"""
        self.show_frame_diagnostics = not self.show_frame_diagnostics
        if self.show_frame_diagnostics:
            diag = self.data_mgr.diagnose_frame(self.data_mgr.current_img_idx)
            bname = os.path.basename(self.data_mgr.image_files[self.data_mgr.current_img_idx])
            c_g = diag.get("contrast_grade", "")
            s_g = diag.get("sharpness_grade", "")
            rej_n = diag.get("rejected_quads_count", 0)
            miss_n = len(diag.get("missing_theoretical_tags", []))
            self.set_toast(f"[{bname}] 病因切片: 对比度 {c_g} | 清晰度 {s_g} | 拒检 {rej_n} | 理论漏检 {miss_n}")
            print("\n" + "=" * 70)
            print(f"[*] [SPATIAL_MAPPING DIAGNOSTICS] 图像深度病因切片: {bname}")
            print(f"    - 对比度 (灰度标准差): {diag.get('contrast', 0.0):.1f} ({c_g})")
            print(f"    - 亮度均值: {diag.get('brightness', 0.0):.1f} ({diag.get('brightness_grade', '')})")
            print(f"    - 图像清晰度 (拉普拉斯梯度): {diag.get('sharpness', 0.0):.1f} ({s_g})")
            print(f"    - 算法被拒候选四边形: {rej_n} 个 (过小: {diag.get('rej_small', 0)}, 长宽失真: {diag.get('rej_aspect', 0)})")
            if diag.get("missing_theoretical_tags"):
                print(f"    - 视场理论可见但漏检的标靶: {[m['tag_id'] for m in diag['missing_theoretical_tags']]}")
            print("=" * 70 + "\n")
        else:
            self.set_toast("已退出病因切片诊断模式，返回常规视口")

    def launch_robot_online_tracker(self):
        """一键跨工序启动 Robot 在线跟踪 (Tag 世界坐标实时解算 + 机械臂联动)"""
        log.info("\n[*] [SPATIAL_MAPPING] 正在启动 Robot 在线跟踪 (tools/tracker/app.py)...")
        self.set_toast("正在启动 Robot 在线跟踪...")
        import subprocess
        subprocess.Popen([sys.executable, "tools/tracker/app.py"])

    def run(self):
        """进入空间建图工作站主交互渲染循环"""
        window_name = "Spatial Mapping Studio - 空间建图工作站"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(window_name, self._on_mouse)

        if force_window_focus:
            force_window_focus(window_name)

        log.info(f"空间建图工作站已启动: {len(self.data_mgr.image_files)} 帧图像, 地图: {self.map_path}")

        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

        try:
            while self.is_running:
                # 视口自适应物理尺寸
                if self.viewport and self.viewport.sync_window_size(window_name):
                    self.win_w = self.viewport.win_w
                    self.win_h = self.viewport.win_h
                    canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)

                # 渲染整帧
                self.render(canvas)

                cv2.imshow(window_name, canvas)
                raw_key = cv2.waitKey(20)
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if raw_key == -1:
                    continue
                key = raw_key & 0xFF

                # 优先拦截结算确认卡片按键交互
                if self.ba_runner.prune_settlement_data is not None:
                    if key in (10, 13):  # Enter 键 -> 采纳结果
                        self.accept_prune_results()
                        continue
                    elif key == 27:      # ESC 键 -> 撤销还原
                        self.undo_prune_results()
                        continue

                # 优先拦截世界系对齐质检单浮层按键交互
                if getattr(self, "alignment_report", None) is not None:
                    if key in (27, 10, 13):  # ESC / Enter -> 关闭质检单
                        self.alignment_report = None
                        continue


                # 运行中支持空格急停
                if self.ba_runner.is_auto_pruning:
                    if key == 32:  # 空格键 -> 急停
                        self.ba_runner.request_stop_pruning()
                        continue

                if key in (ord('q'), ord('Q'), 27):
                    break
                elif key in (ord('a'), ord('A')):      # A 键 -> 智能迭代剪枝平差
                    self.start_auto_prune_ba()
                elif key in (ord('x'), ord('X')):      # X 键 -> 切换多轮残差矩阵视图
                    self.toggle_matrix_view_mode()
                elif key in (ord('w'), ord('W'), 82):  # 上一帧 (W / Up)
                    if self.data_mgr.image_files:
                        next_idx = (self.data_mgr.current_img_idx - 1) % len(self.data_mgr.image_files)
                        self.select_frame(next_idx)
                        self.set_toast(f"选定帧: {os.path.basename(self.data_mgr.image_files[next_idx])}")
                        if self.show_frame_diagnostics:
                            self.data_mgr.diagnose_frame(next_idx)
                elif key in (ord('s'), ord('S'), 84):  # 下一帧 (S / Down)
                    if self.data_mgr.image_files:
                        next_idx = (self.data_mgr.current_img_idx + 1) % len(self.data_mgr.image_files)
                        self.select_frame(next_idx)
                        self.set_toast(f"选定帧: {os.path.basename(self.data_mgr.image_files[next_idx])}")
                        if self.show_frame_diagnostics:
                            self.data_mgr.diagnose_frame(next_idx)
                elif key in (ord('e'), ord('E')):      # E 键 -> 单帧超精重提取
                    self.set_toast("正在执行单帧工业级超精重提取...")
                    bname, cnt = self.data_mgr.super_extract_current_frame(self.data_mgr.current_img_idx)
                    if bname:
                        self.set_toast(f"帧 {bname} 超精提取完成并永久持久化: 检出 {cnt} 个标靶")
                elif key in (ord('d'), ord('D')):      # D 键 -> 漏检病因切片诊断
                    self.toggle_frame_diagnostics()
                elif key in (ord('v'), ord('V')):      # V 键 -> 循环切换视口预设模式
                    presets = [
                        ("3d", "3d", "全 3D 双棱柱空间对比 (BA 3D + 实测 3D)"),
                        ("2d", "2d", "全 2D 重投影与残差矢量 (BA 2D + 实测 2D)"),
                        ("off", "2d", "仅单帧实测识别角点框"),
                        ("3d", "off", "仅 BA 空间理论 3D 棱柱"),
                        ("off", "off", "纯净原始采图 (全隐藏)")
                    ]
                    curr_idx = -1
                    for idx, (b_m, o_m, _) in enumerate(presets):
                        if self.ba_view_mode == b_m and self.obs_view_mode == o_m:
                            curr_idx = idx
                            break
                    next_idx = (curr_idx + 1) % len(presets)
                    self.ba_view_mode, self.obs_view_mode, desc = presets[next_idx]
                    self.set_toast(f"视口模式: {desc}")
                    self.save_dropdown_state()
                elif key in (ord('z'), ord('Z'), ord('0')):  # Z / 0 键 -> 重置缩放
                    self.reset_viewport_zoom()
                elif key in (ord('t'), ord('T'), 32):  # T 键或空格键 -> 翻转状态
                    self.toggle_current_frame_exclusion()
                elif key in (ord('b'), ord('B')):      # B 键 -> 阶段一: 自由平差
                    self.ba_runner.start()
                elif key in (ord('c'), ord('C')):      # C 键 -> 阶段二: 独立校准世界系
                    self.align_current_workspace_world_datum()
                elif key in (ord('p'), ord('P')):      # P 键 -> 重算体检
                    self.data_mgr.refresh_all_frame_metrics()
                    self.set_toast("已全量重算体检指标")
                elif key in (ord('r'), ord('R')):      # R 键 -> 导出报告
                    self.export_verification_report()
                elif key in (ord('m'), ord('M'), ord('u'), ord('U')):  # M/U 键 -> 保存工位地图
                    ManifestRepository.save_map(self.data_mgr.tags_map_data, self.map_path)
                    if self.current_workspace:
                        self.current_workspace.refresh_stats()
                        self.current_workspace.save_meta()
                    self.set_toast("空间立体地图已保存至当前工位沙盒 (tags_map.yaml)！")
                elif key in (ord('y'), ord('Y')):      # Y 键 -> 开关 XY 平面网格
                    self.data_mgr.show_xy_plane_on = not self.data_mgr.show_xy_plane_on
                    self.set_toast(f"XY 平面网格{'已开启' if self.data_mgr.show_xy_plane_on else '已关闭'} ({self.data_mgr.get_current_plane_z_label()})")
                    self.save_dropdown_state()
                elif key in (ord('['), 219):           # [ 键 -> XY 平面高度升档
                    self.data_mgr.step_plane_z(direction=+1)
                    self.set_toast(f"XY 平面高度升档: {self.data_mgr.get_current_plane_z_label()}")
                    self.save_dropdown_state()
                elif key in (ord(']'), 221):           # ] 键 -> XY 平面高度降档
                    self.data_mgr.step_plane_z(direction=-1)
                    self.set_toast(f"XY 平面高度降档: {self.data_mgr.get_current_plane_z_label()}")
                    self.save_dropdown_state()
                elif key in (8, 127):                  # Backspace 或 Delete (DEL) -> 一键复位地图
                    self.data_mgr.reset_map()
                    self.set_toast("立体地图已复位清空 (备份为 .bak)，恢复为纯观测模式")




        finally:
            self.save_dropdown_state()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="AprilTag 空间建图工作站 (Spatial Mapping Studio)")
    parser.add_argument("--workspace", type=str, default=None, help="目标工位 ID")
    parser.add_argument("--map", type=str, default=None, help="标靶空间立体地图路径")
    parser.add_argument("--images", type=str, default=None, help="标定采图目录")
    parser.add_argument("--marker_size", type=float, default=None, help="标靶物理边长 (mm), 缺省从工位 tag_whitelist.yaml.tag_default_size_mm 读取")
    args = parser.parse_args()

    app = SpatialMappingStudioApp(
        map_path=args.map,
        image_dir=args.images,
        marker_size_mm=args.marker_size,
        workspace_id=args.workspace
    )
    app.run()


if __name__ == "__main__":
    main()
