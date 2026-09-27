#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot 在线跟踪 (Robot Online Tracker) — FR-12
==============================================
Dashboard 第 5 张卡片「Robot 在线跟踪」的主工具：
  - GUI 先行启动 (不自动开相机)：顶部工具栏选相机类型 (RealSense D435 / USB 摄像头) →
    分辨率 → [开启] 乒乓开关 (布局与风格借鉴 d435_viewer 深度相机诊断工具)；
  - [一次性建立世界坐标系] 一键单帧闭环 (与 Offline Studio 单帧流程一致)：开相机 → 拍一张 → 关相机 →
    识别视野内已知 Tag (蓝色实测棱柱) → 单帧 PnP 确定世界坐标系零点 →
    地图白名单全部 Tag 以绿色理论棱柱叠加在静态照片上；
  - 真实相机实时取流，基于世界坐标地图 (FR-9.6, 地图世界系=机械臂坐标系) 实时检测标靶；
  - 用视野内非目标标靶的世界角点 PnP 解算相机世界系位姿，进而解出目标 Tag (默认 2 号) 的世界坐标实时显示；
  - 勾选 [√连续跟踪] 开启连续跟踪：经机械臂串口 (FR-7.1) 以"抬起→平移→下探"安全路径
    驱动末端自动跟随目标最新世界坐标；
  - 到位后 M114 回读末端实际坐标，与视觉解算世界坐标同屏对比偏差 (FR-12.4 相机位置校准)。

工具栏 (双排, 组间空白分隔):
  第一排: [工作空间 ▼] [相机类型 ▼] [分辨率 ▼] [开启/关闭] | [串口 ▼] [连接机械臂] [M84+G92] [Park] ... [退出 X]
          (工作空间下拉: 切换工位并加载其世界坐标地图, 选择持久化于 config/gui_settings.json)
  第二排 (第一组居左, 第二组跟踪居右):
    第一组: [一次性建立世界坐标系] [确定世界坐标系] | [XY平面 ▼] [√显示已知Tag] | [目标 ▼] [识别目标·单次] [√连续识别]
    第二组: [跟踪目标·单次] [√连续跟踪]
快捷键: [S] 一次性建立世界坐标系  [P] XY平面下拉  [L] 确定/解除世界坐标系  [A] 显示已知Tag  [R] 连续识别  [C] 连接/断开机械臂  [T] 勾选/取消连续跟踪  [X]/[ESC] 退出

机械臂消息面板: [跟踪目标·单次]/[M84]/[G92]/[Park] 按钮下方, 逐条显示 指令 G-code → 回读/偏差 (调试用)。
"""

import os
import sys
import json
import time
import argparse
import threading

import numpy as np
import cv2
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass  # 编码重配置失败无伤大雅，终端仍可正常运行

from src.calibration.pnp_solver import PnpSolver, get_tag_world_transform, get_tag_world_corners
from src.calibration.tag_detector import TagDetector
from src.control.robot_serial import RobotSerial
from src.utils.gui_window_manager import GuiWindowManager
from tools.tracker.camera_controller import CameraController
from tools.scara_debug.loader_core.config import LoaderConfig
from tools.tracker.common import (
    COLOR_ACCENT, COLOR_TEXT_SUB, COL_CYAN, COL_YELLOW,
    TOOLBAR_H, fmt_point, list_serial_ports)
from src.utils.text_rendering import draw_text
from tools.tracker.renderer import TrackerRenderer
from src.utils.logger import get_logger

try:
    from src.utils.config_guard import resolve_camera_intrinsics
except ImportError:
    resolve_camera_intrinsics = None

log = get_logger(__name__)

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")
APP_ID = "robot_online_tracker"


class RobotOnlineTracker:
    """Robot 在线跟踪控制器: GUI 先行 + Tag 世界坐标实时解算 + 机械臂联动跟踪 + 到位偏差对比"""

    LOCK_MIN_SAMPLES = 5     # 确定世界坐标系最少采样帧数 (保证滤波统计意义)
    LOCK_MAX_SAMPLES = 30    # 采样帧数上限
    LOCK_CONVERGE_MM = 2.0   # 提前收敛阈值: 帧位姿与累计均值的平移偏差 (mm)
    LOCK_CONVERGE_DEG = 0.5  # 提前收敛阈值: 旋转偏差 (deg)
    LOCK_CONVERGE_STREAK = 2  # 连续满足收敛条件的次数
    PLANE_EXTENT_MM = 600     # XY 平面网格半宽 (mm)
    PLANE_STEP_MM = 100       # XY 平面网格间距 (mm)
    PLANE_Z_MM = 600          # Z 轴长度 (mm)
    PLANE_Z_BASE_CHOICES = (350, 300, 250, 200, 150, 100, 50, 0)  # 基础平面高度档 (mm, 锚点高度从地图动态注入)
    PLANE_Z_STATIC_LABELS = {0: " (地面)"}  # 静态高度标注 (锚点标注按地图动态生成)
    HP_TARGET_SIDE_PX = 240   # 高精度模式: ROI 放大后目标 Tag 边长 (px)
    TRACK_RETRIGGER_MM = 3.0  # 持续跟踪: 目标位移超过该阈值才重新发起移动 (mm)
    TRACK_RETRIGGER_DEG = 3.0 # 持续跟踪: 目标角度偏转超过该阈值才重新发起旋转 (deg)
    TRACK_MIN_INTERVAL_S = 1.0  # 持续跟踪: 两次移动任务的最小间隔 (s)
    TRACK_FEEDRATE = 3000        # 跟踪水平平移进给率 (mm/min, F 参数)
    PARK_FEEDRATE = 5000         # Park 回放料位水平平移进给率 (mm/min, 高速)
    TRACK_LOG_MAX = 8         # 机械臂消息面板保留条数 (逐条显示 G-code 与回读, 调试用)

    def __init__(self, map_path=None, target_tag_id=2, port=None, baudrate=0):
        self.target_tag_id = int(target_tag_id)

        # 0. 工位工作空间下拉 (最左上角): 先恢复上次选择再加载对应地图 (持久化)
        self.workspace_id = ""         # 当前选中的工位 ID (空=全局回退地图)
        self.workspace_options = []    # 工位下拉选项 [(ws_id, label), ...] (打开下拉时刷新)
        self._load_workspace_selection()
        self.map_path = map_path or self._workspace_map_path()

        # 1. 世界坐标地图与几何求解器 (内参待相机开启后按实际分辨率刷新)
        self.pnp_solver = None
        self.tag_detector = None
        self.theoretical = None     # 地图中目标 Tag 的理论世界坐标 (_load_pnp_solver 填充)
        self.anchor_positions = {}  # 锚定标靶 BA 理论世界中心 {tag_id: np.array(3)} (不含 Tag 2)
        self._load_pnp_solver()

        # 2. 机械臂串口控制器
        self.robot = RobotSerial(port=port or "", baudrate=baudrate)
        self.port_options = list_serial_ports()   # 工具栏串口下拉选项 (打开下拉时刷新)
        self.robot_connecting = False             # 机械臂拨号中 (按钮三态: 连接→正在连接→断开)
        self.robot_cmd_busy = False               # M84/G92 即时指令执行中 (防重入)

        # 3. 相机硬件控制器 (类型/分辨率状态机 + 取流启停; GUI 先行, 不自动开相机)
        self.camera = CameraController(self.pnp_solver)

        # 4. 工具栏状态 (借鉴 d435_viewer: 相机类型 → 分辨率 → 开关; 绘制由 TrackerRenderer 负责)
        self.active_dropdown = None     # "WORKSPACE_DROPDOWN" | "CAMERA_TYPE_DROPDOWN" | "RES_DROPDOWN" | "PLANE_DROPDOWN" | "TARGET_DROPDOWN" | "PORT_DROPDOWN" | None
        self.plane_z = 0                # XY 平面绘制高度 (mm, 下拉框选择)
        # XY 平面高度选项与标注: 锚点档位/标注从世界坐标地图动态生成, 基础档位为固定梯度
        self.anchor_z_labels = {int(round(pos[2])): f" (Tag {tid})"
                                for tid, pos in self.anchor_positions.items()}
        self.plane_z_labels = {**self.PLANE_Z_STATIC_LABELS, **self.anchor_z_labels}
        self.plane_z_choices = tuple(sorted(
            set(self.PLANE_Z_BASE_CHOICES) | set(self.anchor_z_labels), reverse=True))
        self.plane_options = [
            (None, "不绘制 XY 平面")
        ] + [
            (z, f"Z {z} mm" + self.plane_z_labels.get(z, ""))
            for z in self.plane_z_choices
        ]
        self.renderer = TrackerRenderer(self)  # UI 渲染器 (工具栏/叠加层/按钮命中表)
        self.win_mgr = GuiWindowManager(app_id="robot_online_tracker")  # 窗口/缩放/偏好单源管理

        # 5. 识别与世界系锁定状态 (相机开启默认纯预览, FR-12.5/12.6)
        self.target_kind = "tag"       # 跟踪目标类型: "tag"=Tag 2号标靶 / "asparagus"=顶层芦笋
        self.target_options = [
            ("tag",       "Tag 2号"),
            ("asparagus", "顶层芦笋"),
        ]
        self.recog_tag2_on = False     # "识别 Tag 2" 乒乓开关 (默认关, 仅识别目标 Tag)
        self.show_anchors_on = False   # "显示已知 Tag" 乒乓开关 (绿=BA理论 / 蓝=实测)
        self.show_xy_plane_on = False  # "XY平面" 下拉选择状态 (False=不绘制, True=绘制 plane_z 高度平面)
        self.world_locked = False      # 世界坐标系锁定状态
        self.locked_rvec = None        # 锁定的相机世界位姿 (rvec)
        self.locked_tvec = None        # 锁定的相机世界位姿 (tvec)
        self.lock_info = ""            # 锁定信息 "30帧均值 | RMSE 0.31px"
        self.sampling = False          # 确定世界坐标系流程执行中
        self.sample_stage = ""
        self.recognizing = False       # "识别" 一键单帧闭环流程执行中
        self.recog_stage = ""          # 识别流程阶段提示 (按钮上显示)
        self.static_frame = None       # 识别保留的单帧照片 (相机已关闭, 静态显示)
        self.static_det = None         # 单帧照片的检测结果 {tag_id: corners(4,2)}

        # 6. 跟踪运行状态
        self.measured = None        # 目标 Tag 世界坐标实测 (EMA 平滑)
        self.measured_r = None      # 目标 Tag 局部 Y 轴偏航角/旋转角 R (度, 世界系水平投影, EMA 平滑)
        self.target_rvec = None     # 目标 Tag 相机系旋转向量 (供 3D 拟真芦笋投影)
        self.target_tvec = None     # 目标 Tag 相机系平移向量 (供 3D 拟真芦笋投影)
        self.measured_time = 0.0    # 最近一次目标成功解算的时刻 (单次识别结果判定)
        self.rmse = None            # 世界位姿 PnP 重投影 RMSE (px)
        self.support_ids = []       # 支撑世界位姿解算的标靶 ID
        self.tracking = False       # 跟踪任务执行中
        self.track_stage = ""
        self.track_thread = None
        self.track_armed = False    # [√连续跟踪] 勾选框: 勾选=末端自动跟随目标最新位置
        self.recog_once_requested = False  # [识别目标·单次] 请求标志 (主循环解算一帧后消费)
        self.last_dev = None        # 最近一次到位偏差 (dx, dy, dz)
        self._last_track_done = 0.0     # 上次跟踪任务完成时刻 (连续跟踪节流)
        self._last_track_target = None  # 上次跟踪目标点 (目标位移 < 阈值不重复触发)
        self._last_track_r = None       # 上次跟踪目标角度 (目标偏转 < 阈值不重复触发)
        self.robot_pos = None       # 最近一次 M114 末端坐标
        self.track_log = []         # 机械臂消息面板 [(time_str, msg, kind), ...] kind: info/cmd/ok/err
        self.toast = "选择相机类型与分辨率后点击 [开启]"
        self.toast_err = False
        self.toast_time = time.time()
        self._quit_requested = False

        # 7. 持久化恢复工具栏状态
        self._load_viewer_state()

    # ------------------------------ 工作空间选择 (最左上角下拉) ------------------------------
    def _load_workspace_selection(self):
        """初始化工作空间管理器并恢复上次选中的工位 (持久化于 config/gui_settings.json)"""
        try:
            from src.calibration.workspace_manager import WorkspaceManager
            self.ws_manager = WorkspaceManager()
        except Exception as e:
            log.warning(f"WorkspaceManager 初始化失败, 工作空间下拉不可用: {e}")
            self.ws_manager = None
            return
        # 恢复持久化选择; 无记录或已失效时回退当前默认工位
        saved = ""
        try:
            if os.path.exists(GUI_SETTINGS_FILE):
                with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                    root = json.load(f)
                saved = str(((root.get(APP_ID) or {}).get("viewer_state") or {})
                            .get("workspace_id") or "")
        except Exception:
            saved = ""
        ws_ids = [ws.workspace_id for ws in self.ws_manager.list_workspaces()]
        if saved and saved in ws_ids:
            self.workspace_id = saved
        else:
            try:
                self.workspace_id = self.ws_manager.get_current_workspace_id()
            except Exception:
                self.workspace_id = ws_ids[0] if ws_ids else ""

    def _workspace_map_path(self) -> str:
        """当前工位的地图路径"""
        if self.ws_manager is not None and self.workspace_id:
            ws = self.ws_manager.get_workspace_by_id(self.workspace_id)
            if ws and os.path.exists(ws.map_path) and os.path.getsize(ws.map_path) > 50:
                return ws.map_path
        try:
            from src.calibration.workspace_manager import WorkspaceManager
            return WorkspaceManager().get_current_workspace().map_path
        except Exception:
            return ""

    @property
    def workspace_label(self) -> str:
        """工具栏工作空间下拉的显示文本"""
        if self.ws_manager is not None and self.workspace_id:
            ws = self.ws_manager.get_workspace_by_id(self.workspace_id)
            if ws:
                return ws.name
        return self.workspace_id or "工作空间"

    def refresh_workspace_options(self):
        """刷新工位下拉选项 (打开下拉时调用, 保证新建/删除工位后列表最新)"""
        if self.ws_manager is None:
            self.workspace_options = []
            return
        opts = [(ws.workspace_id, ws.name)
                for ws in self.ws_manager.list_workspaces()]
        # 当前工位不在工位列表时追加占位项, 保证选中态可见
        if self.workspace_id and all(k != self.workspace_id for k, _ in opts):
            opts.insert(0, (self.workspace_id, self.workspace_id))
        self.workspace_options = opts

    def select_workspace(self, ws_id):
        """工作空间下拉选择: 切换工位 → 加载其世界坐标地图 → 旧解算状态全部作废 (持久化)"""
        if not ws_id or self.ws_manager is None or ws_id == self.workspace_id:
            return
        if self.sampling or self.tracking or self.recognizing:
            self.set_toast("任务执行中, 禁止切换工作空间", True)
            return
        ws = self.ws_manager.get_workspace_by_id(ws_id, force_refresh=True)
        if ws is None:
            self.set_toast(f"工位不存在: {ws_id}", True)
            return
        if not os.path.exists(ws.map_path) or os.path.getsize(ws.map_path) < 50:
            self.set_toast(f"工位 [{ws.name}] 尚无有效地图, 请先在标定流程中平差生成", True)
            return
        self.ws_manager.set_current_workspace(ws_id)
        old_map = self.map_path
        self.workspace_id = ws_id
        self.map_path = ws.map_path
        try:
            self._load_pnp_solver()
        except Exception as e:
            # 加载失败回滚到旧地图
            self.workspace_id = ""
            for w in self.ws_manager.list_workspaces():
                if os.path.normpath(w.map_path) == os.path.normpath(old_map):
                    self.workspace_id = w.workspace_id
                    break
            self.map_path = old_map
            try:
                self._load_pnp_solver()
            except Exception:
                pass
            self.set_toast(f"加载工位地图失败: {e}", True)
            return
        # 旧地图的相机取流/世界系锁定/解算结果全部作废
        if self.camera.pipeline_running:
            self._toggle_camera(force_off=True)
        self._release_world_lock(silent=True)
        self.static_frame = None
        self.static_det = None
        self.measured = None
        self.measured_r = None
        self.support_ids = []
        self.rmse = None
        # XY 平面高度档位/标注按新地图锚点重建
        self.anchor_z_labels = {int(round(pos[2])): f" (Tag {tid})"
                                for tid, pos in self.anchor_positions.items()}
        self.plane_z_labels = {**self.PLANE_Z_STATIC_LABELS, **self.anchor_z_labels}
        self.plane_z_choices = tuple(sorted(
            set(self.PLANE_Z_BASE_CHOICES) | set(self.anchor_z_labels), reverse=True))
        self.plane_options = [
            (None, "不绘制 XY 平面")
        ] + [
            (z, f"Z {z} mm" + self.plane_z_labels.get(z, ""))
            for z in self.plane_z_choices
        ]
        if self.show_xy_plane_on and self.plane_z not in self.plane_z_choices:
            self.show_xy_plane_on = False
            self.plane_z = 0
        self._save_viewer_state()
        self.set_toast(f"工作空间已切换: {ws.name} | 地图 {os.path.basename(ws.map_path)}")

    def _load_pnp_solver(self):
        """加载世界坐标地图并构建纯几何计算求解器 (相机内参以 config.yaml 默认值初始化)"""
        with open(self.map_path, "r", encoding="utf-8") as f:
            tags_map = yaml.safe_load(f) or {}
        marker_size = float(tags_map.get("marker_size_mm") or 40.0)

        if resolve_camera_intrinsics is not None:
            camera_matrix, dist_coeffs, _ = resolve_camera_intrinsics(CONFIG_PATH)
        else:
            camera_matrix = np.array([
                [1363.68, 0.0, 971.19],
                [0.0, 1361.19, 566.26],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((5, 1), dtype=np.float64)

        self.pnp_solver = PnpSolver(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            marker_size_mm=marker_size,
            tags_map=tags_map
        )
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.pnp_solver = self.pnp_solver
        self.tag_detector = TagDetector(valid_tag_ids=None)
        n_tags = len(tags_map.get("tags", {}))
        log.info(f"[OK] 世界坐标地图已加载: {self.map_path} | 标靶 {n_tags} 枚 | 边长 {marker_size:.1f}mm")
        T = self.pnp_solver.get_tag_world_transform(self.target_tag_id)
        self.theoretical = T[:3, 3].copy() if T is not None else None

        # 锚定标靶 (地图白名单内, 排除移动的 Tag 2) 的 BA 理论世界中心
        self.anchor_positions = {}
        for tid_s in tags_map.get("tags", {}):
            tid_i = int(tid_s)
            if tid_i == self.target_tag_id:
                continue
            T_i = self.pnp_solver.get_tag_world_transform(tid_i)
            if T_i is not None:
                self.anchor_positions[tid_i] = T_i[:3, 3].copy()

    # ------------------------------ 工具栏状态持久化 ------------------------------
    def _load_viewer_state(self):
        """从 config/gui_settings.json 恢复上次退出时的下拉选择 (工作空间/相机/分辨率/XY平面/目标类型/串口)"""
        try:
            if not os.path.exists(GUI_SETTINGS_FILE):
                return
            with open(GUI_SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            state = (root.get(APP_ID) or {}).get("viewer_state") or {}
            if state.get("camera_type") in ("realsense", "usb"):
                self.camera.camera_type = state["camera_type"]
            if any(k == state.get("resolution") for k, _ in self.camera.resolution_options):
                self.camera.resolution = state["resolution"]
            # XY 平面下拉: 高度档位须在当前地图可选档内, 否则保持默认 (不绘制)
            if state.get("show_plane"):
                pz = state.get("plane_z")
                if isinstance(pz, int) and pz in self.plane_z_choices:
                    self.show_xy_plane_on = True
                    self.plane_z = pz
            # 跟踪目标类型下拉: 仅接受合法选项
            if any(k == state.get("target_kind") for k, _ in self.target_options):
                self.target_kind = state["target_kind"]
            # 机械臂串口: 上次选择的 COM 口
            if state.get("robot_port"):
                self.robot.port = str(state["robot_port"])
        except Exception as e:
            log.warning(f"恢复相机查看器状态失败，使用默认配置: {e}")

    def _save_viewer_state(self):
        """保存下拉选择 (相机/分辨率/XY平面/目标类型/串口) 到 config/gui_settings.json"""
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
            node["viewer_state"] = {
                "workspace_id": str(self.workspace_id or ""),
                "camera_type": self.camera.camera_type,
                "resolution": self.camera.resolution,
                "show_plane": bool(self.show_xy_plane_on),
                "plane_z": int(self.plane_z),
                "target_kind": str(self.target_kind),
                "robot_port": str(self.robot.port or ""),
            }
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            os.makedirs(os.path.dirname(GUI_SETTINGS_FILE), exist_ok=True)
            with open(GUI_SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存相机查看器状态失败: {e}")

    # ------------------------------ 相机开关 (借鉴 d435_viewer) ------------------------------
    def _select_camera_type(self, cam_key):
        """切换相机类型：如果 pipeline 已运行则先停再切换"""
        if cam_key == self.camera.camera_type:
            return
        if self.camera.pipeline_running:
            self._toggle_camera(force_off=True)
        self.camera.camera_type = cam_key
        self._save_viewer_state()
        log.info(f"相机类型已切换为: {dict(self.camera.camera_options).get(cam_key, cam_key)}")

    def _change_resolution(self, res_key):
        """切换分辨率：运行中则先停再按新分辨率重启 (任务执行中禁止)"""
        if res_key == self.camera.resolution:
            return
        if self.sampling or self.tracking or self.recognizing:
            self.set_toast("任务执行中, 禁止切换分辨率", True)
            return
        was_running = self.camera.pipeline_running
        if was_running:
            self._toggle_camera(force_off=True)
        self.camera.set_resolution_key(res_key)
        self._save_viewer_state()
        if was_running:
            self._toggle_camera()
        log.info(f"分辨率已切换: {res_key}")

    def _toggle_camera(self, force_off=False, _internal=False):
        """开启或关闭相机取流 (采样/跟踪/识别任务执行中禁止手动开关; 内部调用与强制关闭除外)"""
        if not force_off and not _internal and (self.sampling or self.tracking or self.recognizing):
            self.set_toast("任务执行中, 禁止切换相机状态", True)
            return
        if force_off or self.camera.pipeline_running:
            self.camera.stop()
            self._release_world_lock(silent=True)
            self.recog_tag2_on = False
            self.set_toast("相机已关闭")
            log.info("相机已关闭")
        else:
            # 重新开启相机: 单帧识别静态结果让位于实时预览
            self.static_frame = None
            self.static_det = None
            try:
                self.camera.start()
                self.set_toast(f"相机已开启: {dict(self.camera.camera_options).get(self.camera.camera_type)}"
                               f" @ {self.camera.resolution}")
            except Exception as e:
                log.warning(f"相机开启失败: {e}")
                self.set_toast(f"相机开启失败: {e}", True)

    def _release_world_lock(self, silent=False):
        """解除世界系锁定 (相机关闭/切换时位姿失效)"""
        if self.world_locked or self.lock_info:
            self.world_locked = False
            self.locked_rvec = None
            self.locked_tvec = None
            self.lock_info = ""
            if not silent:
                self.set_toast("世界坐标系锁定已解除, 可重新确定")

    # ------------------------------ 几何解算 ------------------------------
    def solve_frame(self, frame):
        """单帧解算 (相机开启默认纯预览, 仅在下列情况执行检测):
        - "识别 Tag 2" 开关打开: 每帧仅识别 Tag 2, 已锁定时解算其世界坐标 (FR-12.6)
        - "确定世界坐标系" 流程采样在后台线程独立执行, 不走此函数
        """
        if not (self.recog_tag2_on or self.show_anchors_on or self.show_xy_plane_on):
            self.support_ids = []
            self.rmse = None
            return None

        det = self.tag_detector.detect_tags(frame)
        det = self._detect_high_precision(frame, det)  # 目标 Tag ROI 放大重检 (高精度)
        target_world = None
        target_r = None
        c2 = det.get(self.target_tag_id)

        if c2 is not None:
            # 目标 Tag 法向先验: 已锁定时用"朝向天空"先验消除 IPPE 平面二义性翻转
            R_lock = (cv2.Rodrigues(self.locked_rvec)[0]
                      if (self.world_locked and self.locked_rvec is not None) else None)
            z_exp = None if R_lock is None else R_lock @ np.array([0.0, 0.0, 1.0])
            ok2, rvec2, t2 = self.pnp_solver.solve_single_tag_pnp(c2, expected_z_cam=z_exp)
            if ok2:
                self.target_rvec = rvec2
                self.target_tvec = t2.reshape((3, 1))
                R_c_t2, _ = cv2.Rodrigues(rvec2)
                if R_lock is not None:
                    p_cam = t2.reshape(3)
                    target_world = R_lock.T @ (p_cam - self.locked_tvec.reshape(3))
                    self.support_ids = ["锁定"]
                    # 芦笋/工件长轴为 Tag 局部 Y 轴: 计算 Y 轴在世界系水平 XY 平面上的朝向角度 (度)
                    R_w_t2 = R_lock.T @ R_c_t2
                    v_w_y = R_w_t2[:, 1]  # Y 轴在世界系下的方向向量
                    target_r = float(np.degrees(np.arctan2(v_w_y[1], v_w_y[0])))
                else:
                    # 未锁定时: 尝试利用视野内已知锚定标靶临时解算世界系位姿
                    sol_dyn = self._solve_per_frame(det)
                    if sol_dyn["rvec"] is not None and sol_dyn["target_world"] is not None:
                        target_world = np.asarray(sol_dyn["target_world"], dtype=np.float64)
                        self.support_ids = sol_dyn["support"]
                        self.rmse = sol_dyn["rmse"]
                        target_r = sol_dyn["target_r"]
                    else:
                        self.support_ids = []
                        # 无已知标靶时退化为相机系 Y 轴方向角
                        v_c_y = R_c_t2[:, 1]
                        target_r = float(np.degrees(np.arctan2(v_c_y[1], v_c_y[0])))
            else:
                self.target_rvec = None
                self.target_tvec = None
        else:
            self.target_rvec = None
            self.target_tvec = None

        # 更新显示状态 (实测坐标与 R 轴旋转角 EMA 平滑抑制抖动)
        if target_world is not None:
            p = target_world
            self.measured = p if self.measured is None else 0.5 * self.measured + 0.5 * p
            if target_r is not None:
                if self.measured_r is None:
                    self.measured_r = target_r
                else:
                    diff = (target_r - self.measured_r + 180.0) % 360.0 - 180.0
                    self.measured_r = (self.measured_r + 0.4 * diff + 180.0) % 360.0 - 180.0
            self.measured_time = time.time()
        return det

    def _solve_per_frame(self, det):
        """确定世界坐标系流程用: 识别锚定标靶 (排除 Tag 2) -> 相机世界位姿 PnP"""
        sol = {"support": [], "rmse": None, "target_world": None, "target_r": None,
               "target_rvec": None, "target_tvec": None,
               "rvec": None, "tvec": None}

        obj_list, img_list, ids = [], [], []
        for tid, corners in det.items():
            if tid == self.target_tag_id:
                continue
            wc = self.pnp_solver.get_tag_world_corners(tid)
            if wc is None:
                continue
            obj_list.append(wc)
            img_list.append(corners.reshape((4, 2)))
            ids.append(tid)

        if obj_list:
            obj = np.vstack(obj_list).astype(np.float64)
            img = np.vstack(img_list).astype(np.float64)
            rvec, tvec, ok = self.pnp_solver.solve_pnp(obj, img)
            if ok:
                sol["rvec"] = rvec
                sol["tvec"] = tvec.reshape(3)
                proj, _ = cv2.projectPoints(obj, rvec, tvec,
                                            self.pnp_solver.camera_matrix, self.pnp_solver.dist_coeffs)
                sol["rmse"] = float(np.mean(np.linalg.norm(
                    proj.reshape(-1, 2) - img, axis=1)))
                sol["support"] = ids

                c2 = det.get(self.target_tag_id)
                if c2 is not None:
                    # 目标 Tag 法向"朝向天空"先验: 用锚定 PnP 旋转把世界 +Z 映到相机系
                    R_wc, _ = cv2.Rodrigues(rvec)
                    ok2, rvec2, t2 = self.pnp_solver.solve_single_tag_pnp(
                        c2, expected_z_cam=R_wc @ np.array([0.0, 0.0, 1.0]))
                    if ok2:
                        p_cam = t2.reshape(3)                      # 目标 Tag 中心 (相机系)
                        sol["target_world"] = R_wc.T @ (p_cam - tvec.reshape(3))  # -> 世界系
                        sol["target_rvec"] = rvec2
                        sol["target_tvec"] = t2
                        R_c_t2, _ = cv2.Rodrigues(rvec2)
                        R_w_t2 = R_wc.T @ R_c_t2
                        v_w_y = R_w_t2[:, 1]
                        sol["target_r"] = float(np.degrees(np.arctan2(v_w_y[1], v_w_y[0])))
        return sol

    def _detect_high_precision(self, frame, det):
        """目标 Tag 高精度二次识别: 全景粗检定位 -> ROI 裁剪双三次放大 ->
        双路参数重检 -> 亚像素精修 -> 角点映射回原图坐标。
        放大后标靶边缘像素数倍增, 亚像素角点精度显著提升 (目标边长 HP_TARGET_SIDE_PX)。
        """
        c2 = det.get(self.target_tag_id)
        if c2 is None:
            return det
        det_bright = self.tag_detector.detector_bright
        det_dark = self.tag_detector.detector_dark
        if det_bright is None:
            return det
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        h_f, w_f = gray.shape[:2]
        c = np.asarray(c2, dtype=np.float64).reshape(4, 2)
        side = float((np.linalg.norm(c[0] - c[1]) + np.linalg.norm(c[1] - c[2])) / 2.0)
        pad = max(side * 0.8, 30.0)                        # ROI 外扩 (含倾斜余量)
        x1 = int(max(0, c[:, 0].min() - pad))
        y1 = int(max(0, c[:, 1].min() - pad))
        x2 = int(min(w_f, c[:, 0].max() + pad))
        y2 = int(min(h_f, c[:, 1].max() + pad))
        if x2 - x1 < 24 or y2 - y1 < 24:
            return det
        roi = gray[y1:y2, x1:x2]
        scale = float(np.clip(self.HP_TARGET_SIDE_PX / max(side, 1.0), 2.0, 8.0))
        scale = min(scale, 800.0 / max(roi.shape[0], roi.shape[1], 1))  # 限制放大图尺寸
        if scale < 1.01:
            return det
        roi_up = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        refined = None
        for d in (det_bright, det_dark):                   # 双路重检 (与全景检测同参数)
            corners, ids, _ = d.detectMarkers(roi_up)
            if ids is not None and len(ids) > 0:
                for i, tid in enumerate(ids.flatten()):
                    if int(tid) == self.target_tag_id:
                        refined = corners[i].reshape(4, 2)
                        break
            if refined is not None:
                break
        if refined is None:
            return det
        refine = getattr(self.tag_detector, "refine_corners_subpix", None)
        if refine is not None:
            refined = refine(roi_up, refined)
        refined_orig = np.asarray(refined, dtype=np.float64) / scale + np.array([x1, y1])
        det = dict(det)
        det[self.target_tag_id] = refined_orig.reshape(4, 2)
        return det

    def get_world_pose(self, det):
        """当前相机世界位姿 (世界->相机): 优先锁定位姿, 未锁定用当帧锚定 PnP (渲染器叠加层使用)"""
        if self.world_locked and self.locked_rvec is not None:
            return self.locked_rvec, self.locked_tvec
        if det is None:
            return None, None
        sol = self._solve_per_frame(det)
        if sol["rvec"] is not None:
            return sol["rvec"], np.asarray(sol["tvec"]).reshape(3, 1)
        return None, None

    def toggle_world_lock(self):
        """触发 / 解除"确定世界坐标系"一键流程 (FR-12.5)"""
        if self.world_locked:
            self._release_world_lock()
            return
        if self.sampling:
            self.set_toast("世界坐标系确定流程进行中, 请稍候")
            return
        if self.recognizing:
            self.set_toast("识别流程进行中, 请稍候")
            return
        if not self.camera.pipeline_running:
            self.set_toast("相机未开启, 无法确定世界坐标系", True)
            return
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止采样", True)
            return
        self.sampling = True
        threading.Thread(target=self._lock_worker, daemon=True).start()

    def _lock_worker(self):
        """确定世界坐标系线程 (动态采样): 识别锚定标靶 (排除 Tag 2) ->
        最少 5 帧 / 上限 30 帧, 当帧位姿与累计均值偏差 < 2mm 且旋转 < 0.5deg
        连续 2 次时提前收敛, 否则采满上限; 最终均值滤波锁定相机世界位姿
        """
        try:
            rot_mats, tvecs, rmses = [], [], []
            n = self.LOCK_MAX_SAMPLES
            stable_streak = 0
            converged_at = None
            for i in range(1, n + 1):
                self.sample_stage = f"采样中 {i}/{n}"
                frame = self.camera.read_frame()
                if frame is None:
                    continue
                det = self.tag_detector.detect_tags(frame)
                sol = self._solve_per_frame(det)
                if sol["rvec"] is None or sol["rmse"] is None:
                    continue
                R_i = cv2.Rodrigues(sol["rvec"])[0]
                t_i = np.asarray(sol["tvec"], dtype=np.float64).reshape(3)
                rot_mats.append(R_i)
                tvecs.append(t_i)
                rmses.append(sol["rmse"])

                # 动态收敛判定: 新帧与此前累计均值的偏差足够小且连续稳定
                if len(tvecs) >= self.LOCK_MIN_SAMPLES:
                    t_mean = np.mean(tvecs, axis=0)
                    R_mean = self._average_rotation(rot_mats)
                    d_mm = float(np.linalg.norm(t_i - t_mean))
                    d_deg = self._rotation_angle_deg(R_i, R_mean)
                    if d_mm < self.LOCK_CONVERGE_MM and d_deg < self.LOCK_CONVERGE_DEG:
                        stable_streak += 1
                        if stable_streak >= self.LOCK_CONVERGE_STREAK:
                            converged_at = len(tvecs)
                            break
                    else:
                        stable_streak = 0

            if len(rmses) < self.LOCK_MIN_SAMPLES:
                self.set_toast(f"有效采样不足 ({len(rmses)} < 最少{self.LOCK_MIN_SAMPLES}帧), 世界坐标系确定失败", True)
                return
            # 均值滤波: 平移取均值, 旋转 SVD 投影到 SO(3)
            self.locked_rvec = cv2.Rodrigues(self._average_rotation(rot_mats))[0]
            self.locked_tvec = np.mean(tvecs, axis=0).reshape(3, 1)
            self.world_locked = True
            tip = (f" | 第{converged_at}帧提前收敛(<{self.LOCK_CONVERGE_MM:g}mm)"
                   if converged_at else "")
            self.lock_info = f"{len(rmses)}帧均值 | RMSE {np.mean(rmses):.2f}px{tip}"
            self.recog_tag2_on = True  # 锁定后自动开启"识别 Tag 2"
            self.set_toast(f"世界坐标系零点已确定并锁定 ({self.lock_info}), "
                           f"已自动开启识别 Tag {self.target_tag_id}")
        finally:
            self.sampling = False
            self.sample_stage = ""

    # ------------------------------ 一键识别 (单帧闭环) ------------------------------
    def trigger_recognize(self):
        """一键识别: 开相机 → 拍一张 → 关相机 → 识别已知 Tag (蓝棱柱) →
        单帧 PnP 确定世界坐标系零点 → 地图白名单全部 Tag 绿色理论棱柱"""
        if self.recognizing:
            self.set_toast("识别流程进行中, 请稍候")
            return
        if self.sampling or self.tracking:
            self.set_toast("任务执行中, 请稍候", True)
            return
        self.recognizing = True
        threading.Thread(target=self._recognize_worker, daemon=True).start()

    def _recognize_worker(self):
        """识别线程: 完整"开→拍→关"闭环, 结束后相机保持关闭 (与 Offline Studio 单帧流程一致)"""
        try:
            # 1. 开相机 (若未开启; 内部调用不受任务互斥限制)
            if not self.camera.pipeline_running:
                self.recog_stage = "开启相机..."
                self._toggle_camera(_internal=True)
                if not self.camera.pipeline_running:
                    return
            # 2. 采集一帧 (连读数帧待曝光稳定, 取最后一帧)
            self.recog_stage = "采集照片..."
            frame = None
            for _ in range(8):
                f = self.camera.read_frame()
                if f is not None:
                    frame = f
                time.sleep(0.05)
            # 3. 关相机 (拍完即关, 完整闭环)
            self.recog_stage = "关闭相机..."
            if self.camera.pipeline_running:
                self._toggle_camera(force_off=True)
            if frame is None:
                self.set_toast("采集照片失败, 识别中止", True)
                return
            # 4. 识别视野内已知的 Tag (目标 Tag 追加 ROI 放大重检, 高精度)
            self.recog_stage = "检测 Tag... (高精度)"
            raw_det = self.tag_detector.detect_tags(frame)
            det = self._detect_high_precision(frame, raw_det)
            self.static_frame = frame.copy()
            self.static_det = dict(det) if det else {}
            if not det:
                self.set_toast("识别完成: 未识别到任何 Tag", True)
                return
            # 5. 单帧 PnP 确定世界坐标系零点 (锚定标靶, 排除目标 Tag)
            sol = self._solve_per_frame(det)
            if sol["rvec"] is None:
                self.set_toast(f"识别到 {len(det)} 枚 Tag, 但无已知标靶入镜, 无法确定世界坐标系", True)
                return
            self.locked_rvec = sol["rvec"]
            self.locked_tvec = np.asarray(sol["tvec"], dtype=np.float64).reshape(3, 1)
            self.world_locked = True
            self.lock_info = f"单帧识别 | RMSE {sol['rmse']:.2f}px | 支撑 {sol['support']}"
            self.support_ids = sol["support"]
            self.rmse = sol["rmse"]
            if sol["target_world"] is not None:
                self.measured = np.asarray(sol["target_world"], dtype=np.float64)
                if sol.get("target_r") is not None:
                    self.measured_r = float(sol["target_r"])
                if sol.get("target_rvec") is not None:
                    self.target_rvec = sol["target_rvec"]
                if sol.get("target_tvec") is not None:
                    self.target_tvec = np.asarray(sol["target_tvec"]).reshape((3, 1))
                self.measured_time = time.time()
            n_map = len(self.anchor_positions) + (1 if self.theoretical is not None else 0)
            self.set_toast(f"识别完成: {len(det)} 枚 Tag(蓝) | 世界坐标系已确定 | "
                           f"地图 {n_map} 枚理论 Tag(绿) | RMSE {sol['rmse']:.2f}px")
        finally:
            self.recognizing = False
            self.recog_stage = ""

    @staticmethod
    def _average_rotation(rot_mats):
        """旋转均值滤波: 矩阵均值后 SVD 投影到 SO(3)"""
        U, _, Vt = np.linalg.svd(np.mean(rot_mats, axis=0))
        R = U @ Vt
        if np.linalg.det(R) < 0:
            R = U @ np.diag([1.0, 1.0, -1.0]) @ Vt
        return R

    @staticmethod
    def _rotation_angle_deg(R_a, R_b):
        """两个旋转矩阵之间的夹角 (deg)"""
        R_d = R_a @ R_b.T
        return float(np.degrees(np.arccos(np.clip((np.trace(R_d) - 1.0) / 2.0, -1.0, 1.0))))

    # ------------------------------ 机械臂联动 ------------------------------
    def toggle_robot(self):
        """连接 / 断开机械臂串口 (连接过程异步, 按钮显示 '正在连接...')"""
        if self.robot_connecting:
            return
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止断开", True)
            return
        if self.robot.is_connected:
            self.robot.close()
            self.robot_pos = None
            self.add_track_log("机械臂串口已断开")
            self.set_toast("机械臂串口已断开")
        else:
            if not self.robot.port:
                self.set_toast("请先在 [串口▼] 下拉中选择串口", True)
                return
            self.robot_connecting = True
            self.set_toast(f"连接中: {self.robot.port} @ {self.robot.baudrate} ...")
            threading.Thread(target=self._robot_connect_worker, daemon=True).start()

    def _robot_connect_worker(self):
        """机械臂拨号线程: 串口握手在后台执行, UI 保持刷新
        连接成功后自动串行执行 M84 + G92 设零流程 (省去用户手动按 [M84+G92])"""
        try:
            self.robot.connect()
            self.add_track_log(f"机械臂已连接: {self.robot.port} @ {self.robot.baudrate}")
            self.set_toast(f"机械臂已连接: {self.robot.port} | 自动设零中...")
            # 连接成功后自动 M84 + G92 设零 (无需用户再按按钮)
            self._m84_g92_worker()
        except Exception as e:
            self.add_track_log(f"连接失败: {e}", "err")
            self.set_toast(f"连接失败: {e}", True)
        finally:
            self.robot_connecting = False

    def refresh_port_options(self):
        """重新枚举系统可用串口 (打开串口下拉时调用, 保证列表最新)"""
        self.port_options = list_serial_ports()

    def select_port(self, port_key):
        """串口下拉选择: 更换机械臂目标串口 (连接中/拨号中禁止切换)"""
        if not port_key:
            self.set_toast("未枚举到可用串口, 请检查 USB 连接后重试", True)
            return
        if self.robot_connecting:
            self.set_toast("正在连接中, 请稍候再切换串口", True)
            return
        if self.robot.is_connected:
            self.set_toast("机械臂已连接, 请先断开再切换串口", True)
            return
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止切换串口", True)
            return
        self.robot.port = port_key
        self._save_viewer_state()
        self.set_toast(f"机械臂串口已选择: {port_key}, 点击 [连接机械臂] 拨号")

    def select_target_kind(self, kind):
        """目标类型下拉选择: "tag"=Tag 2号标靶 / "asparagus"=顶层芦笋;
        切换后旧目标的解算结果立即作废 (避免用 A 目标坐标驱动 B 目标跟踪)"""
        if kind == self.target_kind:
            return
        label = dict(self.target_options).get(kind, kind)
        self.target_kind = kind
        self.measured = None
        self.recog_tag2_on = False
        self.support_ids = []
        self._save_viewer_state()
        if kind == "asparagus":
            self.set_toast("目标已切换: 顶层芦笋 (识别功能需深度流接入, 即将上线)")
        else:
            self.set_toast(f"目标已切换: {label}")

    def _send_m84_g92(self):
        """[M84+G92] 合并按钮: 后台线程先发 M84 释放电机, 再发 G92 设零点"""
        if self.robot_connecting:
            self.set_toast("正在连接中, 请稍候", True)
            return
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止发送指令", True)
            return
        if self.robot_cmd_busy:
            return
        if not self.robot.is_connected:
            self.set_toast("机械臂未连接, 请先连接", True)
            return
        self.robot_cmd_busy = True
        threading.Thread(target=self._m84_g92_worker, daemon=True).start()

    def _m84_g92_worker(self):
        """M84+G92 顺序执行线程: M84 释放电机 → G92 对齐机械零位绝对坐标 → M114 回读
        (全程逐条写入机械臂消息面板)"""
        try:
            self.add_track_log("发送: M84", "cmd")
            if not self.robot.send_gcode("M84", timeout=3.0):
                self.add_track_log("M84 发送失败 (无 ok 应答)", "err")
                self.set_toast("M84 发送失败 (无 ok 应答), 详见终端日志", True)
                return
            # 机械零位的绝对坐标 (单一来源 LoaderConfig.home_pose = X0 Y600 Z80, R=90°→E 轴);
            # 声明"当前位置=机械零位绝对坐标", 绝非把当前位当 (0,0,0)
            hp = LoaderConfig().home_pose
            g92_cmd = f"G92 X{hp.x:.2f} Y{hp.y:.2f} Z{hp.z:.2f} E{hp.r:.2f}"
            self.add_track_log(f"发送: {g92_cmd}", "cmd")
            if not self.robot.send_gcode(g92_cmd, timeout=3.0):
                self.add_track_log("G92 发送失败 (无 ok 应答)", "err")
                self.set_toast("G92 发送失败 (无 ok 应答), 详见终端日志", True)
                return
            pos = self.robot.get_position()
            if pos is not None:
                self.robot_pos = pos
                self.add_track_log(f"M84+G92 完成 | 回读: {fmt_point(pos)}", "ok")
                self.set_toast(f"M84+G92 完成 | 末端 {pos[0]:.1f} {pos[1]:.1f} {pos[2]:.1f}")
            else:
                self.add_track_log("M84+G92 已发送 (M114 无回读)")
                self.set_toast("M84+G92 已发送")
        finally:
            self.robot_cmd_busy = False

    def _park_robot(self):
        """[Park] 按钮: 机械臂回到放料位 (X-250 Y350 Z80, R=90°→E 轴)"""
        if self.robot_connecting:
            self.set_toast("正在连接中, 请稍候", True)
            return
        if self.tracking:
            self.set_toast("跟踪任务执行中, 禁止发送指令", True)
            return
        if self.robot_cmd_busy:
            return
        if not self.robot.is_connected:
            self.set_toast("机械臂未连接, 请先连接", True)
            return
        self.robot_cmd_busy = True
        threading.Thread(target=self._park_worker, daemon=True).start()

    def _park_worker(self):
        """Park 执行线程: G1 直线插补到放料位 (X-250 Y350 Z80 E90, F5000 高速) → M114 回读确认
        (放料位 Z=80 为安全高度, 可直接平移, 无需三段式抬起→下探;
         用 G1 而非 G0 以严格遵循项目"水平对位 G0 + 精准插补 G1"双轨规范, 让 F5000 进给率真正生效)"""
        try:
            # 放料位绝对坐标 (基于机械零位 (0, 600, 80, 90) 的世界坐标系)
            cmd = f"G1 X-250.00 Y350.00 Z80.00 E90.00 F{self.PARK_FEEDRATE}"
            self.add_track_log(f"发送: {cmd}", "cmd")
            if not self.robot.send_gcode(cmd, timeout=30.0, wait_done=True):
                self.add_track_log("Park 移动失败 (无 ok 应答)", "err")
                self.set_toast("Park 移动失败, 详见终端日志", True)
                return
            pos = self.robot.get_position()
            if pos is not None:
                self.robot_pos = pos
                self.add_track_log(f"Park 完成 | 回读: {fmt_point(pos)}", "ok")
                self.set_toast(f"Park 完成 | 末端 {pos[0]:.1f} {pos[1]:.1f} {pos[2]:.1f}")
            else:
                self.add_track_log("Park 已发送 (M114 无回读)")
                self.set_toast("Park 已发送")
        finally:
            self.robot_cmd_busy = False

    def trigger_recog_target_once(self):
        """[识别目标·单次]: 实时流中解算一帧目标世界坐标, 结果经 Toast 显示"""
        if self.target_kind == "asparagus":
            self.set_toast("顶层芦笋识别需 RealSense 深度流接入, 功能开发中", True)
            return
        if self.recognizing or self.sampling:
            self.set_toast("世界坐标系流程执行中, 请稍候", True)
            return
        if not self.camera.pipeline_running:
            self.set_toast("请先开启相机 (标定世界坐标系请用 [一次性建立世界坐标系])", True)
            return
        self.recog_once_requested = True   # 下一帧解算后由主循环消费并 Toast 结果

    def trigger_tracking(self):
        """[跟踪目标·单次]: 触发一次"抬起→平移→下探"到位任务 (后台线程执行)"""
        if self.tracking:
            self.set_toast("跟踪任务执行中, 请稍候", True)
            return
        if not self.robot.is_connected:
            self.set_toast("机械臂未连接, 请先连接", True)
            return
        if self.measured is None:
            self.set_toast("尚无有效的目标解算结果, 请先识别目标", True)
            return
        target = self.measured.copy()
        target_r = getattr(self, "measured_r", None)
        self._last_track_target = target
        self._last_track_r = target_r
        self.tracking = True
        self.track_thread = threading.Thread(
            target=self._track_worker, args=(target, target_r), daemon=True)
        self.track_thread.start()

    def toggle_track_armed(self):
        """[√连续跟踪] 勾选框切换: 勾选=末端自动跟随目标最新位置 (实时流中调度)"""
        if self.track_armed:
            self.track_armed = False
            self.set_toast("连续跟踪已关闭 (当前移动到位后停止)")
            return
        if not self.robot.is_connected:
            self.set_toast("机械臂未连接, 请先连接后再勾选跟踪", True)
            return
        self.track_armed = True
        self.set_toast("连续跟踪已开启: 末端将自动跟随目标最新位置与角度 (位移>3mm 或 偏转>3° 触发)")

    def _maybe_continuous_track(self):
        """连续跟踪调度: 勾选状态下每帧检查, 空闲且目标位移或旋转偏转超阈值时发起移动"""
        if not self.track_armed or self.tracking or not self.robot.is_connected:
            return
        if self.measured is None:
            return
        target = self.measured.copy()
        target_r = getattr(self, "measured_r", None)
        dist_mm = (float(np.linalg.norm(target - self._last_track_target))
                   if self._last_track_target is not None else float("inf"))
        d_deg = (abs((target_r - self._last_track_r + 180.0) % 360.0 - 180.0)
                 if (target_r is not None and self._last_track_r is not None)
                 else (float("inf") if (target_r is not None) ^ (self._last_track_r is not None) else 0.0))
        if dist_mm < self.TRACK_RETRIGGER_MM and d_deg < self.TRACK_RETRIGGER_DEG:
            return
        if time.time() - self._last_track_done < self.TRACK_MIN_INTERVAL_S:
            return
        self._last_track_target = target
        self._last_track_r = target_r
        self.tracking = True
        self.track_thread = threading.Thread(
            target=self._track_worker, args=(target, target_r), daemon=True)
        self.track_thread.start()

    def _track_worker(self, target, target_r=None):
        """跟踪线程: 单条 G1 水平平移 (Z=80 固定, 联动 X/Y 与 R 轴/E 轴角度) -> M114 回读 -> 偏差计算"""
        try:
            r_val = float(target_r) if target_r is not None else getattr(self, "measured_r", None)
            e_str = f"E{r_val:.2f}" if r_val is not None else "E90.00"
            r_desc = f"R={r_val:.1f}°" if r_val is not None else "E=90°(默认)"
            target_pose = (target[0], target[1], 80.0, r_val if r_val is not None else 90.0)

            # 1. 确保夹爪预先打开，平移对准目标上方 (Z=80 安全高度)
            self.robot.set_gripper(close=False)
            self.add_track_log(
                f"目标 ← 视觉: X{target[0]:.1f} Y{target[1]:.1f} {r_desc} (Z=80 固定)")
            self.track_stage = f"移动中: 水平平移对位 (Z=80, {r_desc})"
            cmd = (f"G1 X{target[0]:.2f} Y{target[1]:.2f} "
                   f"Z80.00 {e_str} F{self.TRACK_FEEDRATE}")
            self.add_track_log(f"发送: {cmd}", "cmd")
            if not self.robot.send_gcode(cmd, timeout=30.0, wait_done=True):
                self.add_track_log("对位移动失败 (无 ok 应答)", "err")
                self.set_toast("机械臂移动失败, 详见终端日志", True)
                return

            # 对位完成，回读坐标计算视觉偏差
            pos = self.robot.get_position()
            self.robot_pos = pos
            if pos is not None:
                if len(pos) >= 4:
                    dev_xyz = np.array(pos[:3], dtype=np.float64) - np.array(target_pose[:3], dtype=np.float64)
                    dev_r = (float(pos[3]) - float(target_pose[3]) + 180.0) % 360.0 - 180.0
                    dev = np.array([dev_xyz[0], dev_xyz[1], dev_xyz[2], dev_r], dtype=np.float64)
                    self.last_dev = dev
                    self.add_track_log(f"对位回读: {fmt_point(pos)}", "ok")
                    self.add_track_log(
                        f"对位偏差: {dev[0]:+.1f} {dev[1]:+.1f} {dev[2]:+.1f} R:{dev[3]:+.1f}°"
                        f" (XYZ {np.linalg.norm(dev_xyz):.2f} mm)", "ok")
                else:
                    dev = np.array(pos, dtype=np.float64) - np.array(target_pose[:3], dtype=np.float64)
                    self.last_dev = dev
                    self.add_track_log(f"对位回读: {fmt_point(pos)}", "ok")
                    self.add_track_log(
                        f"对位偏差: {dev[0]:+.1f} {dev[1]:+.1f} {dev[2]:+.1f}"
                        f" (总 {np.linalg.norm(dev):.2f} mm)", "ok")

            # 2. 往下移 (下探至物料表面抓取位: 驱动 Servo 0 舵机升降 + 同步 G92 Z)
            down_z = float(np.clip(target[2], 0.0, 60.0)) if (len(target) >= 3 and target[2] is not None and 0.0 <= target[2] <= 60.0) else 20.0
            self.track_stage = f"移动中: 垂直下探至抓取高度 (Z={down_z:.1f})"
            self.add_track_log(f"执行: Z 轴下探 (Servo 0 -> Z={down_z:.1f}mm)", "cmd")
            if not self.robot.set_z_height(down_z):
                self.add_track_log("下探移动失败", "err")
                self.set_toast("下探移动失败", True)
                return
            time.sleep(0.3)  # 下探机械到位等待

            # 3. 夹爪夹住物料 (M4 + 舵机全闭)
            self.track_stage = "执行中: 夹爪夹紧物料"
            self.add_track_log("执行: 夹爪夹紧 (M4 / M280 P1/P2 S0)", "cmd")
            self.robot.set_gripper(close=True)
            time.sleep(0.3)  # 保压延时确保牢固抓持

            # 4. 上移 (提升回安全高度 Z=80: 驱动 Servo 0 舵机升降 + 同步 G92 Z)
            self.track_stage = "移动中: 提升至安全高度 (Z=80)"
            self.add_track_log("执行: Z 轴提升 (Servo 0 -> Z=80.0mm)", "cmd")
            if not self.robot.set_z_height(80.0):
                self.add_track_log("提升安全高度失败", "err")
                self.set_toast("提升安全高度失败", True)
                return
            time.sleep(0.3)

            # 5. 自动 Park (平移至放料位)
            self.track_stage = "移动中: 自动前往放料位 Park"
            cmd_park = f"G1 X-250.00 Y350.00 Z80.00 E90.00 F{self.PARK_FEEDRATE}"
            self.add_track_log(f"发送: {cmd_park}", "cmd")
            if not self.robot.send_gcode(cmd_park, timeout=30.0, wait_done=True):
                self.add_track_log("前往 Park 放料位失败", "err")
                self.set_toast("前往 Park 放料位失败", True)
                return

            # 6. 到了 Park 位置，夹爪打开释放物料 (M3 + 舵机全开)
            self.track_stage = "执行中: 放料位夹爪打开释放"
            self.add_track_log("执行: 放料位夹爪打开 (M3 / M280 P1/P2 S30)", "cmd")
            self.robot.set_gripper(close=False)
            time.sleep(0.2)

            # 7. 整个工作结束，回读放料位末端坐标
            pos_park = self.robot.get_position()
            if pos_park is not None:
                self.robot_pos = pos_park
                self.add_track_log(f"Park 到位回读: {fmt_point(pos_park)}", "ok")
            self.add_track_log("跟踪抓取与 Park 放料完成", "ok")
            self.set_toast("跟踪目标搬运完成: 下探 -> 夹紧 -> 提升 -> Park -> 释放")
        finally:
            self.track_stage = ""
            self.tracking = False
            self._last_track_done = time.time()

    def add_track_log(self, msg: str, kind: str = "info"):
        """跟踪消息面板追加一条消息 (时间戳 + 内容), 超出上限淘汰最旧条目"""
        self.track_log.append((time.strftime("%H:%M:%S"), msg, kind))
        if len(self.track_log) > self.TRACK_LOG_MAX:
            del self.track_log[:len(self.track_log) - self.TRACK_LOG_MAX]

    def set_toast(self, msg: str, is_err: bool = False):
        self.toast = msg
        self.toast_err = is_err
        self.toast_time = time.time()

    # ------------------------------ 界面渲染 (由 TrackerRenderer 负责) ------------------------------

    # ------------------------------ 鼠标交互 ------------------------------
    def _on_mouse(self, event, x, y, flags, param):
        # 画布按窗口尺寸真矢量重绘, imshow 1:1 呈现, 窗口坐标即画布坐标 (零偏移)
        if event == cv2.EVENT_MOUSEMOVE:
            self.renderer.on_mouse_move(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            hit = self.renderer.hit_test(x, y)
            if hit is not None:
                self._handle_action(*hit)
                return
            # 点击空白处收起下拉
            if self.active_dropdown is not None:
                self.active_dropdown = None

    def _handle_action(self, btn_id, payload):
        """工具栏按钮动作分发"""
        if btn_id == "TOGGLE_WS_DD":
            self.refresh_workspace_options()   # 打开时刷新枚举, 保证工位列表最新
            self.active_dropdown = None if self.active_dropdown == "WORKSPACE_DROPDOWN" \
                else "WORKSPACE_DROPDOWN"
        elif btn_id.startswith("DD_WS_"):
            self.active_dropdown = None
            self.select_workspace(payload)
        elif btn_id == "TOGGLE_CAM_DD":
            self.active_dropdown = None if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" \
                else "CAMERA_TYPE_DROPDOWN"
        elif btn_id == "TOGGLE_RES_DD":
            self.active_dropdown = None if self.active_dropdown == "RES_DROPDOWN" \
                else "RES_DROPDOWN"
        elif btn_id.startswith("DD_CAM_"):
            self.active_dropdown = None
            self._select_camera_type(payload)
        elif btn_id.startswith("DD_RES_"):
            self.active_dropdown = None
            self._change_resolution(payload)
        elif btn_id == "TOGGLE_CAMERA":
            self.active_dropdown = None
            self._toggle_camera()
        elif btn_id == "TOGGLE_ANCHORS":
            self.show_anchors_on = not self.show_anchors_on
            self.set_toast("已显示已知 Tag 位置 (绿=BA理论 / 蓝=实测)" if self.show_anchors_on
                           else "已知 Tag 位置显示已关闭")
        elif btn_id == "TOGGLE_RECOG":
            if not self.recog_tag2_on and self.target_kind == "asparagus":
                self.set_toast("顶层芦笋识别需 RealSense 深度流接入, 功能开发中", True)
                return
            self.recog_tag2_on = not self.recog_tag2_on
            if not self.recog_tag2_on:
                self.support_ids = []
            self.set_toast("连续识别已开启 (逐帧解算目标世界坐标)" if self.recog_tag2_on
                           else "连续识别已关闭 (纯预览)")
        elif btn_id == "TOGGLE_LOCK":
            self.toggle_world_lock()
        elif btn_id == "TOGGLE_PLANE_DD":
            self.active_dropdown = None if self.active_dropdown == "PLANE_DROPDOWN" else "PLANE_DROPDOWN"
        elif btn_id.startswith("DD_PLANE_"):
            self.active_dropdown = None
            if payload is None:
                self.show_xy_plane_on = False
                self.set_toast("XY 平面网格已关闭")
            else:
                self.show_xy_plane_on = True
                self.plane_z = int(payload)
                label = self.plane_z_labels.get(self.plane_z, "")
                self.set_toast(f"XY 平面已重绘至 Z={self.plane_z} mm{label}")
            self._save_viewer_state()
        elif btn_id == "TOGGLE_TARGET_DD":
            self.active_dropdown = None if self.active_dropdown == "TARGET_DROPDOWN" \
                else "TARGET_DROPDOWN"
        elif btn_id.startswith("DD_TARGET_"):
            self.active_dropdown = None
            self.select_target_kind(payload)
        elif btn_id == "TRIGGER_RECOG":
            self.trigger_recognize()
        elif btn_id == "TRIGGER_RECOG_TARGET":
            self.trigger_recog_target_once()
        elif btn_id == "TRIGGER_TRACK_ONCE":
            self.trigger_tracking()
        elif btn_id == "TOGGLE_ROBOT":
            self.toggle_robot()
        elif btn_id == "TOGGLE_PORT_DD":
            self.refresh_port_options()   # 打开时刷新枚举, 保证插拔 USB 后列表最新
            self.active_dropdown = None if self.active_dropdown == "PORT_DROPDOWN" \
                else "PORT_DROPDOWN"
        elif btn_id.startswith("DD_PORT_"):
            self.active_dropdown = None
            self.select_port(payload)
        elif btn_id == "ROBOT_M84_G92":
            self._send_m84_g92()
        elif btn_id == "ROBOT_PARK":
            self._park_robot()
        elif btn_id == "TOGGLE_TRACK":
            self.toggle_track_armed()
        elif btn_id == "QUIT":
            self._quit_requested = True

    # ------------------------------ 主循环 ------------------------------
    def run(self):
        """主事件循环 — GUI 先行启动, 相机等用户点击 [开启]

        窗口生命周期/拉伸自适应/矢量缩放统一由 GuiWindowManager 管理:
        - 拖拽窗口边框自由调整大小, 画布按窗口尺寸真矢量重绘 (1:1 像素对齐, 鼠标零偏移),
          停止拖拽 0.35s 后自动记忆;
        - Ctrl+鼠标滚轮 或 Ctrl+加减键 矢量缩放, Ctrl+0 复位;
        - 右上角红叉 [X] 优雅退出。
        """
        win_key = "robot_online_tracker"  # 窗口 key 纯 ASCII (namedWindow ANSI API)
        self.win_mgr.setup_window(win_key, mouse_callback=self._on_mouse)
        self.win_mgr.set_unicode_title("Robot 在线跟踪 | flux_vision_3d")
        log.info("Robot 在线跟踪系统已启动。")

        try:
            while not self._quit_requested:
                if self.camera.pipeline_running:
                    frame = self.camera.read_frame()
                    if frame is None:
                        canvas = self.renderer.make_canvas()
                        cw, ch = self.win_mgr.canvas_w, self.win_mgr.canvas_h
                        draw_text(canvas, "取流中...", (cw // 2 - 60, ch // 2), 22, COL_YELLOW, True)
                    else:
                        # 实时叠加直接画在原始帧上 (帧坐标), 再与工具栏拼合, 保证与画面内容对齐
                        # (连续识别/连续跟踪/单次识别请求时也需解算, 保证目标实测坐标最新)
                        if (self.recog_tag2_on or self.show_anchors_on
                                or self.show_xy_plane_on or self.track_armed
                                or self.recog_once_requested):
                            t_prev = self.measured_time
                            det = self.solve_frame(frame)
                            if det is not None and (self.recog_tag2_on or self.show_anchors_on
                                                    or self.show_xy_plane_on):
                                self.renderer.draw_overlay(frame, det)
                                self.renderer.draw_xy_plane_overlay(frame, det)
                                self.renderer.draw_anchor_overlay(frame, det)
                            if self.recog_once_requested:
                                self.recog_once_requested = False
                                if self.measured_time > t_prev:
                                    m = self.measured
                                    self.add_track_log(f"识别目标: {fmt_point(m)}")
                                    self.set_toast(f"单次识别目标: {m[0]:+.1f} {m[1]:+.1f} "
                                                   f"{m[2]:+.1f} mm")
                                else:
                                    self.add_track_log("识别目标失败: 未入镜或世界系不可用", "err")
                                    self.set_toast("单次识别失败: 目标未入镜或世界系不可用", True)
                        self._maybe_continuous_track()
                        canvas = self.renderer.compose_canvas(frame)
                        self.renderer.draw_info_panel(canvas, y_off=TOOLBAR_H)
                elif self.static_frame is not None:
                    # 单帧识别结果静态显示 (相机已关闭): 蓝=当帧实测棱柱 / 绿=地图理论棱柱
                    disp = self.static_frame.copy()
                    self.renderer.draw_xy_plane_overlay(disp, None)
                    self.renderer.draw_recognition_overlay(disp)
                    canvas = self.renderer.compose_canvas(disp)
                    self.renderer.draw_info_panel(canvas, y_off=TOOLBAR_H)
                    draw_text(canvas, "单帧识别结果 (相机已关闭): 蓝=当帧实测 / 绿=地图理论",
                              (14, TOOLBAR_H + 310), 15, COL_CYAN, True)
                else:
                    # 相机未开启: 窗口尺寸占位画布 (真矢量, 开启前后工具栏位置严格一致)
                    canvas = self.renderer.make_canvas()
                    cw, ch = self.win_mgr.canvas_w, self.win_mgr.canvas_h
                    draw_text(canvas, "相机未开启",
                              (cw // 2 - 120, ch // 2 - 50), 32, COLOR_ACCENT, True)
                    draw_text(canvas, "请先选择相机类型和分辨率，然后点击 [开启] 按钮",
                              (cw // 2 - 250, ch // 2 + 10), 18, COLOR_TEXT_SUB)

                self.renderer.draw_toolbar(canvas)
                self.renderer.draw_toast(canvas)
                cv2.imshow(win_key, canvas)

                key = cv2.waitKeyEx(30)
                poll = self.win_mgr.poll_events(key)
                if poll.should_quit:
                    break
                if poll.toast_msg:
                    self.set_toast(poll.toast_msg)
                if key == -1:
                    continue
                k = chr(key & 0xFF).lower() if (key & 0xFF) < 128 else ""
                if k == "x":
                    break
                elif k == "a":
                    self._handle_action("TOGGLE_ANCHORS", None)
                elif k == "s":
                    self.trigger_recognize()
                elif k == "r":
                    self._handle_action("TOGGLE_RECOG", None)
                elif k == "p":
                    self._handle_action("TOGGLE_PLANE_DD", None)
                elif k == "l":
                    self.toggle_world_lock()
                elif k == "c":
                    self.toggle_robot()
                elif k == "t":
                    self.toggle_track_armed()
        finally:
            self.robot.close()
            self._toggle_camera(force_off=True)
            cv2.destroyAllWindows()
            log.info("[OK] Robot 在线跟踪已退出")


def main():
    ap = argparse.ArgumentParser(
        description="Robot 在线跟踪 — Tag 世界坐标实时解算与机械臂联动校准 (FR-12)")
    ap.add_argument("--map", default=None, help="世界坐标地图 yaml 路径 (默认优先生产地图或当前工况场景地图)")
    ap.add_argument("--tag", type=int, default=2, help="跟踪目标 Tag ID (默认 2)")
    ap.add_argument("--port", default=None, help="机械臂串口 (默认读 config.yaml robot.port)")
    ap.add_argument("--baudrate", type=int, default=0, help="波特率 (默认读 config.yaml robot.baudrate)")
    args = ap.parse_args()

    try:
        app = RobotOnlineTracker(map_path=args.map, target_tag_id=args.tag,
                                 port=args.port, baudrate=args.baudrate)
    except Exception as e:
        log.warning(f"初始化失败: {e}")
        sys.exit(1)
    app.run()


if __name__ == "__main__":
    main()
