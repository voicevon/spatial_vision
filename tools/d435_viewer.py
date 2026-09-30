"""
D435 实时彩色/对齐深度流纯预览与深度探针工具
=====================================================
用途：
  1. 实时预览 RealSense D435 (或 USB 摄像头) 的 RGB 画面与对齐深度热力图
  2. 顶部按钮栏：相机类型、分辨率、开启/关闭、RGB/Depth 显隐、排列方式、暂停、缩放、退出
  3. 支持 [Space] 空格键一键定格/暂停画面
  4. 鼠标悬停/点击查看毫米级深度与 (X, Y, Z) 空间坐标
  5. 无任何识别/抓拍/G-code 等业务功能
"""

import os
import sys
import time
import json
import argparse
import yaml
import numpy as np
import cv2

# 解决 Windows 控制台中文编码
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass  # 编码重配置失败无伤大雅，终端仍可正常运行

try:
    import pyrealsense2 as rs
    HAVE_REALSENSE = True
except ImportError:
    HAVE_REALSENSE = False

# 导入通用 GUI 基础设施
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.ui.gui_window_manager import GuiWindowManager
from src.ui.gui_theme import GuiTheme
from src.ui.text_rendering import draw_text, get_cached_font, measure_text, put_text
from src.utils.logger import get_logger

log = get_logger(__name__)


class D435Viewer:
    # 调色板: 统一取自 GuiTheme 主题单源
    COLOR_BG = GuiTheme.BG
    COLOR_CARD_BG = GuiTheme.CARD_BG
    COLOR_CARD_HOVER = GuiTheme.CARD_HOVER
    COLOR_CARD_SEL = GuiTheme.CARD_SEL
    COLOR_BORDER = GuiTheme.BORDER
    COLOR_BORDER_HOVER = GuiTheme.BORDER_HOVER
    COLOR_BORDER_SEL = GuiTheme.BORDER_SEL
    COLOR_TEXT_TITLE = GuiTheme.TEXT
    COLOR_TEXT_SUB = GuiTheme.TEXT_SUB
    COLOR_TEXT_MUTED = GuiTheme.TEXT_MUTED
    COLOR_ACCENT = GuiTheme.ACCENT
    COLOR_GOLD = GuiTheme.GOLD

    TOOLBAR_H = 44  # 工具栏高度

    def __init__(self, config_path: str = "config/config.yaml", settings_file: str = None):
        self.config_path = config_path
        self.win_mgr = GuiWindowManager(app_id="d435_viewer", base_w=1024, base_h=640, settings_file=settings_file)
        self.load_config()

        self.pipeline = None
        self.rs_config = None
        self.align = None

        if HAVE_REALSENSE:
            self.pipeline = rs.pipeline()
            self.rs_config = rs.config()
            self.align = rs.align(rs.stream.color)

        # --- 显示控制开关 ---
        self.show_rgb = True       # RGB 画面开关
        self.show_depth = True     # 深度图开关
        self.split_vertical = True # True=上下排列, False=左右排列

        self.init_filters()

        # 鼠标交互状态
        self.hover_x = -1
        self.hover_y = -1
        self.selected_point = None

        self.color_intrinsics = None

        # 深度色彩映射范围 (米) - 默认 0.48m ~ 0.66m (适配当前 640mm 工作台)
        self.cmap_min = 0.48
        self.cmap_max = 0.66
        self.auto_range = False

        # 排版模式: split_vertical=True 上下排列, False 左右并排
        self.is_paused = False          # 空格键暂停/定格模式，消除花屏闪烁
        self.paused_color_frame = None
        self.paused_depth_frame = None

        # --- 下拉菜单状态 ---
        self.active_dropdown = None       # 当前展开的下拉名 (None = 收起)
        self.gui_buttons = []             # [(btn_id, (x1,y1,x2,y2), payload), ...] 每帧重建

        # --- 相机类型 ---
        self.camera_type = "realsense"    # "realsense" | "usb"
        self.camera_options = [
            ("realsense",  "RealSense D435"),
            ("usb",        "USB 普通摄像头"),
        ]
        self._camera_type_rect = None

        # --- 相机分辨率 ---
        self.resolution = "1280x720"      # 当前分辨率 key
        self.resolution_options = [       # 支持的分辨率列表 (key, label)
            ("1280x720",  "1280 × 720  (推荐)"),
            ("848x480",   "848 × 480"),
            ("640x480",   "640 × 480"),
            ("320x240",   "320 × 240"),
        ]
        self._resolution_rect = None      # 记录下拉按钮 rect (用于展开弹层位置)

        # --- 运行状态 ---
        self.pipeline_running = False     # 相机是否已开启
        self.usb_capture = None           # USB cv2.VideoCapture 引用

        # --- 持久化恢复工具栏状态 ---
        self._load_viewer_state()

    def load_config(self):
        """加载 config.yaml 中的相机与滤波配置"""
        default_config = {
            "camera": {
                "color": {"width": 1280, "height": 720, "fps": 30},
                "depth": {"width": 848, "height": 480, "fps": 30},
                "visual_preset": 3,
                "laser_power": 120,
                "filters": {
                    "spatial": {"enabled": True, "smooth_alpha": 0.5, "smooth_delta": 20, "magnitude": 2, "hole_fill": 1},
                    "temporal": {"enabled": True, "smooth_alpha": 0.4, "smooth_delta": 20, "persistence_control": 3},
                    "threshold": {"enabled": True, "min_distance": 0.40, "max_distance": 0.70},
                },
                "colormap": {
                    "min_distance": 0.48,
                    "max_distance": 0.66,
                    "auto_range": False
                }
            }
        }
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and "camera" in loaded:
                    default_config["camera"].update(loaded["camera"])

        self.cfg = default_config["camera"]
        self.cmap_min = self.cfg["colormap"].get("min_distance", 0.48)
        self.cmap_max = self.cfg["colormap"].get("max_distance", 0.66)
        self.auto_range = self.cfg["colormap"].get("auto_range", False)

    def init_filters(self):
        """初始化 RealSense 后处理滤波模块"""
        if not HAVE_REALSENSE:
            return

        f_cfg = self.cfg["filters"]
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, f_cfg["spatial"]["smooth_alpha"])
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, f_cfg["spatial"]["smooth_delta"])
        self.spatial_filter.set_option(rs.option.filter_magnitude, f_cfg["spatial"]["magnitude"])
        self.spatial_filter.set_option(rs.option.holes_fill, f_cfg["spatial"]["hole_fill"])

        self.temporal_filter = rs.temporal_filter()
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, f_cfg["temporal"]["smooth_alpha"])
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, f_cfg["temporal"]["smooth_delta"])
        self.temporal_filter.set_option(rs.option.holes_fill, f_cfg["temporal"]["persistence_control"])

        self.threshold_filter = rs.threshold_filter()
        self.threshold_filter.set_option(rs.option.min_distance, f_cfg["threshold"]["min_distance"])
        self.threshold_filter.set_option(rs.option.max_distance, f_cfg["threshold"]["max_distance"])

    def apply_filters(self, depth_frame):
        """对深度帧执行 SDK 级硬件滤波链 (始终启用，提升深度预览质量)"""
        if not HAVE_REALSENSE:
            return depth_frame
        try:
            filtered = self.threshold_filter.process(depth_frame)
            filtered = self.spatial_filter.process(filtered)
            filtered = self.temporal_filter.process(filtered)
            return filtered
        except Exception:
            return depth_frame

    def start(self):
        """启动设备数据流"""
        if not HAVE_REALSENSE:
            raise RuntimeError("未检测到 RealSense Python 绑定库 (pyrealsense2)，请先安装。")

        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError("未检测到任何物理连接的 RealSense 设备，请插入 USB 接口。")

        dev = devices[0]
        dev_name = dev.get_info(rs.camera_info.name)
        dev_sn = dev.get_info(rs.camera_info.serial_number)
        usb_desc = dev.get_info(rs.camera_info.usb_type_descriptor) if dev.supports(rs.camera_info.usb_type_descriptor) else "Unknown"

        log.info(f"成功连接设备: {dev_name} (S/N: {dev_sn}, USB 模式: {usb_desc})")

        c_w, c_h, fps = self.cfg["color"]["width"], self.cfg["color"]["height"], self.cfg["color"]["fps"]
        d_w, d_h = self.cfg["depth"]["width"], self.cfg["depth"]["height"]

        if "2." in usb_desc:
            log.warning("[!] 检测到当前相机工作在 USB 2.1 带宽下（建议连接电脑蓝色 USB 3.0 端口）。")
            log.info("[*] 正在自动启用 USB 2.1 自适应高清/流畅流配置...")
            self.rs_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
            self.rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
        else:
            self.rs_config.enable_stream(rs.stream.color, c_w, c_h, rs.format.bgr8, fps)
            self.rs_config.enable_stream(rs.stream.depth, d_w, d_h, rs.format.z16, fps)

        profile = self.pipeline.start(self.rs_config)
        self.apply_device_settings(profile)

        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.color_intrinsics = color_stream.get_intrinsics()
        self.actual_w = self.color_intrinsics.width
        self.actual_h = self.color_intrinsics.height

    def apply_device_settings(self, profile):
        """配置预设与红外发射功率"""
        try:
            adv_mode = rs.rs400_advanced_mode(profile.get_device())
            if adv_mode.is_enabled():
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_sensor.set_option(rs.option.visual_preset, float(self.cfg.get("visual_preset", 3)))
        except Exception as e:
            log.warning(f"设置 RealSense 视觉预设失败: {e}")
        # 激光
        try:
            depth_sensor = profile.get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, float(self.cfg.get("laser_power", 120)))
        except Exception as e:
            log.warning(f"设置 RealSense 激光功率失败: {e}")

    def _on_dropdown_select(self, dropdown_name, payload):
        """通用下拉选项选择回调"""
        if dropdown_name == "RES_DROPDOWN":
            self._change_resolution(payload)
        elif dropdown_name == "CAMERA_TYPE_DROPDOWN":
            self._select_camera_type(payload)

    def _select_camera_type(self, cam_key):
        """切换相机类型：如果 pipeline 已运行则先停再切换"""
        if cam_key == self.camera_type:
            return
        if self.pipeline_running:
            self._toggle_camera(force_off=True)
        self.camera_type = cam_key
        self._save_viewer_state()
        log.info(f"相机类型已切换为: {dict(self.camera_options).get(cam_key, cam_key)}")

    def _toggle_camera(self, force_off=False):
        """开启或关闭相机 pipeline"""
        if force_off or self.pipeline_running:
            # 关闭
            if self.camera_type == "realsense" and self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass  # 相机停止失败无伤大雅，后续会重置状态
            elif self.usb_capture:
                try:
                    self.usb_capture.release()
                except Exception:
                    pass  # 相机释放失败无伤大雅，后续会重置状态
                self.usb_capture = None
            self.pipeline_running = False
            self.is_paused = False
            self.paused_color_frame = None
            self.paused_depth_frame = None
            log.info("相机已关闭")
        else:
            # 开启
            try:
                if self.camera_type == "realsense":
                    self.start()
                elif self.camera_type == "usb":
                    self._start_usb()
                self.pipeline_running = True
                log.info(f"相机已开启 ({dict(self.camera_options).get(self.camera_type, self.camera_type)})")
            except Exception as e:
                log.warning(f"相机开启失败: {e}")
                self.pipeline_running = False
        self._save_viewer_state()

    def _start_usb(self):
        """启动普通 USB 摄像头 (cv2.VideoCapture)"""
        w_str, h_str = self.resolution.split("x")
        new_w, new_h = int(w_str), int(h_str)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("无法打开 USB 摄像头 (index=0)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_h)
        cap.set(cv2.CAP_PROP_FPS, 30)
        self.usb_capture = cap
        self.actual_w = new_w
        self.actual_h = new_h
        # USB 没有内参，用 dummy 让其他代码不崩
        class _DummyIntrinsics:
            fx = fy = 500.0
            ppx = new_w / 2.0
            ppy = new_h / 2.0
            width = new_w
            height = new_h
            coeffs = [0, 0, 0, 0, 0]
            model = 0
        self.color_intrinsics = _DummyIntrinsics()

    def _usb_read_frames(self):
        """USB 模式下模拟 RealSense 的 frames 返回 (color only, depth=None)"""
        if self.usb_capture is None:
            return None, None
        ret, frame = self.usb_capture.read()
        if not ret:
            return None, None
        return frame, None

    def _change_resolution(self, res_key):
        """切换相机分辨率：根据当前 camera_type 选择 RealSense restart 或 USB VideoCapture"""
        if res_key == self.resolution:
            return
        was_running = self.pipeline_running
        try:
            # 1. 解析
            w_str, h_str = res_key.split("x")
            new_w, new_h = int(w_str), int(h_str)
            self.resolution = res_key
            self.cfg["color"]["width"] = new_w
            self.cfg["color"]["height"] = new_h
            self.cfg["depth"]["width"] = new_w
            self.cfg["depth"]["height"] = new_h

            # 2. 如果正在运行，先停再按新分辨率重启
            if was_running:
                self._toggle_camera(force_off=True)
                self._toggle_camera()  # 按新 camera_type + resolution 启动
            else:
                # 不在运行，只更新实际宽高
                self.actual_w = new_w
                self.actual_h = new_h

            self._save_viewer_state()
            log.info(f"分辨率切换成功: {res_key}")
        except Exception as e:
            log.warning(f"分辨率切换失败: {e}")
            if was_running and not self.pipeline_running:
                try:
                    self._toggle_camera()
                except Exception:
                    pass  # 恢复相机失败的善后尝试，原始错误已在上方记录

    def on_mouse(self, event, x, y, flags, param):
        """鼠标移动/点击事件处理（含顶部工具栏 + 下拉菜单 hit-testing）"""
        # 0. 持续跟踪鼠标位置 (用于 hover 效果)
        self.mouse_pos = (x, y)

        # 1. 优先拦截 Ctrl + 滚轮缩放 (委托通用管理器)
        if event == 10:  # cv2.EVENT_MOUSEWHEEL
            if getattr(self, "win_mgr", None):
                self.win_mgr.handle_mouse_wheel(event, flags)
                return

        # 2. 检查下拉菜单展开状态：优先处理 popup 区域点击
        if self.active_dropdown and self.active_dropdown in ("RES_DROPDOWN", "CAMERA_TYPE_DROPDOWN"):
            prefix_map = {"RES_DROPDOWN": "DD_RES_", "CAMERA_TYPE_DROPDOWN": "DD_CAM_"}
            prefix = prefix_map.get(self.active_dropdown, "")
            for btn_id, (bx1, by1, bx2, by2), payload in list(self.gui_buttons):
                if btn_id.startswith(prefix) and bx1 <= x <= bx2 and by1 <= y <= by2:
                    self._on_dropdown_select(self.active_dropdown, payload)
                    self.active_dropdown = None
                    return
            # 没点中选项
            if event == cv2.EVENT_LBUTTONDOWN:
                if y < self.TOOLBAR_H:
                    self._hit_test_toolbar(x, y)
                    return
                self.active_dropdown = None
                return

        # 3. 顶部工具栏区域点击检测 (y < TOOLBAR_H)
        if event == cv2.EVENT_LBUTTONDOWN and y < self.TOOLBAR_H:
            self._hit_test_toolbar(x, y)
            return

        # param 可能为 None（OpenCV 回调未传时），做防御处理
        param = param or {}

        orig_w = getattr(self, "actual_w", 1280)
        orig_h = getattr(self, "actual_h", 720)

        # 获取当前内部逻辑排版尺寸
        cw = param.get("current_w", 960)
        ch = param.get("current_h", 1080)

        # 若窗口物理分辨率已缩放/拉伸，将当前物理坐标映射回内部逻辑坐标
        if getattr(self, "win_mgr", None) and (self.win_mgr.canvas_w != cw or self.win_mgr.canvas_h != ch):
            scale = min(self.win_mgr.canvas_w / float(cw), self.win_mgr.canvas_h / float(ch))
            pad_x = (self.win_mgr.canvas_w - int(cw * scale)) // 2
            pad_y = (self.win_mgr.canvas_h - int(ch * scale)) // 2
            x = int((x - pad_x) / max(1e-6, scale))
            y = int((y - pad_y) / max(1e-6, scale))
            x = max(0, min(cw - 1, x))
            y = max(0, min(ch - 1, y))

        u, v = -1, -1

        if self.split_vertical:
            # 上下排列模式：上半屏 (0 ~ ch/2) 为 RGB 画面，下半屏为深度图
            half_h = ch // 2
            if y < half_h:
                u = int(x * orig_w / cw)
                v = int(y * orig_h / half_h)
            else:
                u = int(x * orig_w / cw)
                v = int((y - half_h) * orig_h / half_h)
        else:
            # 左右排列模式：左半屏为 RGB，右半屏为深度图
            half_w = cw // 2
            local_x = x % half_w
            u = int(local_x * orig_w / half_w)
            v = int(y * orig_h / ch)

        if 0 <= u < orig_w and 0 <= v < orig_h:
            self.hover_x = u
            self.hover_y = v
            if event == cv2.EVENT_LBUTTONDOWN:
                self.selected_point = (u, v)

    # ================================================================
    # 工具栏状态持久化
    # ================================================================
    def _load_viewer_state(self):
        """从 gui_settings.json 恢复工具栏 toggle 状态"""
        try:
            settings_file = getattr(self.win_mgr, 'settings_file', None)
            if not settings_file or not os.path.exists(settings_file):
                return
            with open(settings_file, "r", encoding="utf-8") as f:
                root = json.load(f)
            node = root.get(self.win_mgr.app_id, {})
            state = node.get("viewer_state")
            if not state:
                return
            # 逐项恢复（仅恢复已知字段）
            bool_fields = ["show_rgb", "show_depth", "split_vertical", "auto_range"]
            for k in bool_fields:
                if k in state and isinstance(state[k], bool):
                    setattr(self, k, state[k])
            if "cmap_min" in state and isinstance(state["cmap_min"], (int, float)):
                self.cmap_min = float(state["cmap_min"])
            if "cmap_max" in state and isinstance(state["cmap_max"], (int, float)):
                self.cmap_max = float(state["cmap_max"])
            if "resolution" in state and any(k == state["resolution"] for k, _ in self.resolution_options):
                self.resolution = state["resolution"]
            if "camera_type" in state and any(k == state["camera_type"] for k, _ in self.camera_options):
                self.camera_type = state["camera_type"]
        except Exception as e:
            log.warning(f"恢复查看器工具栏状态失败，使用默认配置: {e}")

    def _save_viewer_state(self):
        """保存工具栏 toggle 状态到 gui_settings.json"""
        try:
            settings_file = getattr(self.win_mgr, 'settings_file', None)
            if not settings_file:
                return
            root = {}
            if os.path.exists(settings_file):
                try:
                    with open(settings_file, "r", encoding="utf-8") as f:
                        root = json.load(f)
                    if not isinstance(root, dict):
                        root = {}
                except Exception:
                    root = {}

            node = root.setdefault(self.win_mgr.app_id, {})
            node["viewer_state"] = {
                "camera_type": self.camera_type,
                "resolution": self.resolution,
                "show_rgb": self.show_rgb,
                "show_depth": self.show_depth,
                "split_vertical": self.split_vertical,
                "auto_range": self.auto_range,
                "cmap_min": self.cmap_min,
                "cmap_max": self.cmap_max,
            }
            node["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

            os.makedirs(os.path.dirname(settings_file), exist_ok=True)
            with open(settings_file, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存查看器工具栏状态失败: {e}")

    # ================================================================
    # 顶部工具栏
    # ================================================================
    def _build_toolbar_buttons(self):
        """构建工具栏按钮定义列表（仅预览相关，无业务开关）
        格式: (label, action_id, button_type, is_toggle_or_options, get_state_fn)
        button_type: "toggle" | "action" | "dropdown"
        """
        buttons = [
            # label    action_id        type       toggle/options  get_state
            ("RGB",    "toggle_rgb",    "toggle",  None,           lambda: self.show_rgb),
            ("Depth",  "toggle_depth",  "toggle",  None,           lambda: self.show_depth),
            ("排列",   "cycle_split",   "action",  None,           None),
            ("暂停",   "toggle_pause",  "toggle",  None,           lambda: self.is_paused),
            ("Zoom+",  "zoom_in",       "action",  None,           None),
            ("Zoom-",  "zoom_out",      "action",  None,           None),
        ]
        return buttons

    def _draw_dropdown_button(self, canvas, rect, label, is_open, mouse_pos=(0, 0)):
        """绘制扁平化下拉按钮（抄 studio_renderer.draw_dropdown_button）"""
        x1, y1, x2, y2 = rect
        mx, my = mouse_pos
        is_hover = (x1 <= mx <= x2 and y1 <= my <= y2)

        if is_open:
            bg_col = (48, 56, 72)
            border_col = (0, 220, 255)
            text_col = (255, 255, 255)
            arrow = "▲"
        elif is_hover:
            bg_col = (36, 40, 52)
            border_col = (0, 180, 220)
            text_col = (240, 240, 240)
            arrow = "▼"
        else:
            bg_col = self.COLOR_CARD_BG
            border_col = self.COLOR_BORDER
            text_col = self.COLOR_TEXT_SUB
            arrow = "▼"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_col, -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_col, 1)

        display = f"{label} {arrow}"
        (tw, th), _ = measure_text(display, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        tx = x1 + max(4, (x2 - x1 - tw) // 2)
        ty = y1 + (y2 - y1 + th) // 2
        put_text(canvas, display, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.40, text_col, 1, cv2.LINE_AA)

    def _render_dropdown_popup(self, canvas, rect, options, active_key, btn_prefix="DD_RES_"):
        """置顶悬浮下拉列表浮层（抄 studio_renderer.render_dropdown_popup）"""
        rx1, ry1, rx2, ry2 = rect
        item_h = 30
        pop_w = max(rx2 - rx1, 210)
        pop_h = len(options) * item_h + 6
        pop_x1 = rx1
        pop_y1 = ry2 + 2
        pop_x2 = pop_x1 + pop_w
        pop_y2 = pop_y1 + pop_h

        overlay = canvas.copy()
        cv2.rectangle(overlay, (pop_x1, pop_y1), (pop_x2, pop_y2), (18, 20, 26), -1)
        cv2.addWeighted(overlay, 0.95, canvas, 0.05, 0, canvas)
        cv2.rectangle(canvas, (pop_x1, pop_y1), (pop_x2, pop_y2), (0, 200, 255), 1)

        mx, my = self.mouse_pos if hasattr(self, 'mouse_pos') else (0, 0)
        for idx, (opt_key, opt_label) in enumerate(options):
            iy1 = pop_y1 + 3 + idx * item_h
            iy2 = iy1 + item_h - 1
            is_active = (opt_key == active_key)
            is_hover = (pop_x1 <= mx <= pop_x2 and iy1 <= my <= iy2)

            if is_active:
                row_bg = (52, 45, 20)
                txt_col = (0, 230, 255)
            elif is_hover:
                row_bg = (36, 42, 56)
                txt_col = (255, 255, 255)
            else:
                row_bg = (22, 25, 32)
                txt_col = (190, 190, 190)

            cv2.rectangle(canvas, (pop_x1 + 3, iy1), (pop_x2 - 3, iy2), row_bg, -1)
            prefix = "✔ " if is_active else "  "
            put_text(canvas, prefix + opt_label, (pop_x1 + 8, iy1 + (item_h + 10) // 2 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, txt_col, 1, cv2.LINE_AA)

            btn_id = f"{btn_prefix}{opt_key}"
            self.gui_buttons.append((btn_id, (pop_x1 + 3, iy1, pop_x2 - 3, iy2), opt_key))

        return (pop_x1, pop_y1, pop_x2, pop_y2)

    def _hit_test_toolbar(self, x, y):
        """检测工具栏区域的点击，执行对应 action"""
        toolbar_w = self.win_mgr.canvas_w
        buttons = self._build_toolbar_buttons()
        n = len(buttons)
        pad = 8
        gap = 6
        # 预留给: camera_type(120) + resolution(100) + 开关(70) + 退出(80) + 普通按钮
        reserved = 120 + 100 + 70 + 80
        btn_w = max(45, (toolbar_w - pad * 2 - gap * (n + 4) - reserved) // n)
        btn_h = self.TOOLBAR_H - 12
        y1 = 6
        y2 = y1 + btn_h

        # 1. camera_type 下拉 (最左 120px)
        cx = pad
        cam_x1 = cx; cam_x2 = cx + 120
        if cam_x1 <= x <= cam_x2 and y1 <= y <= y2:
            self.active_dropdown = None if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" else "CAMERA_TYPE_DROPDOWN"
            return
        self._camera_type_rect = (cam_x1, y1, cam_x2, y2)

        # 2. resolution 下拉 (100px)
        cx = cam_x2 + gap
        res_x1 = cx; res_x2 = cx + 100
        if res_x1 <= x <= res_x2 and y1 <= y <= y2:
            self.active_dropdown = None if self.active_dropdown == "RES_DROPDOWN" else "RES_DROPDOWN"
            return
        self._resolution_rect = (res_x1, y1, res_x2, y2)

        # 3. 开启/关闭 (70px)
        cx = res_x2 + gap
        sw_x1 = cx; sw_x2 = cx + 70
        if sw_x1 <= x <= sw_x2 and y1 <= y <= y2:
            self._toggle_camera()
            return

        # 4. 退出按钮 (最右 80px)
        exit_x1 = toolbar_w - 80
        exit_x2 = toolbar_w - 8
        if exit_x1 <= x <= exit_x2 and y1 <= y <= y2:
            self._quit_requested = True
            self.active_dropdown = None
            return

        # 如果下拉已展开，outside-click 收起
        if self.active_dropdown and y < self.TOOLBAR_H:
            self.active_dropdown = None
            return

        # 5. 普通按钮
        cx = sw_x2 + gap
        for label, action_id, btype, opt, get_state in buttons:
            x1 = cx
            x2 = cx + btn_w
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._execute_action(action_id)
                return
            cx = x2 + gap

    def _execute_action(self, action_id):
        """执行工具栏 action — 每次 toggle 后立刻保存状态"""
        needs_save = False
        if action_id == "toggle_rgb":
            self.show_rgb = not self.show_rgb; needs_save = True
        elif action_id == "toggle_depth":
            self.show_depth = not self.show_depth; needs_save = True
        elif action_id == "cycle_split":
            self.split_vertical = not self.split_vertical; needs_save = True
        elif action_id == "toggle_pause":
            self.is_paused = not self.is_paused
        elif action_id == "zoom_in":
            self.win_mgr.apply_zoom(+10)
        elif action_id == "zoom_out":
            self.win_mgr.apply_zoom(-10)
        if needs_save:
            self._save_viewer_state()

    def _draw_toolbar(self, canvas):
        """在画布顶部绘制工具栏"""
        tw = canvas.shape[1]
        th = self.TOOLBAR_H
        self.gui_buttons = []

        # 工具栏背景
        cv2.rectangle(canvas, (0, 0), (tw, th), self.COLOR_BG, -1)
        cv2.line(canvas, (0, th - 1), (tw, th - 1), self.COLOR_BORDER, 1)

        buttons = self._build_toolbar_buttons()
        n = len(buttons)
        pad = 8
        gap = 6
        reserved = 120 + 100 + 70 + 80
        btn_w = max(45, (tw - pad * 2 - gap * (n + 4) - reserved) // n)
        btn_h = th - 12
        y1 = 6
        y2 = y1 + btn_h

        # 1. camera_type 下拉 (120px 最左)
        cx = pad
        cam_x1 = cx; cam_x2 = cx + 120
        cam_label = dict(self.camera_options).get(self.camera_type, self.camera_type)
        self._draw_dropdown_button(canvas, (cam_x1, y1, cam_x2, y2), cam_label, is_open=(self.active_dropdown == "CAMERA_TYPE_DROPDOWN"))
        self.gui_buttons.append(("TOGGLE_CAM_DD", (cam_x1, y1, cam_x2, y2), "CAMERA_TYPE_DROPDOWN"))
        self._camera_type_rect = (cam_x1, y1, cam_x2, y2)

        # 2. resolution 下拉 (100px)
        cx = cam_x2 + gap
        res_x1 = cx; res_x2 = cx + 100
        self._draw_dropdown_button(canvas, (res_x1, y1, res_x2, y2), self.resolution, is_open=(self.active_dropdown == "RES_DROPDOWN"))
        self.gui_buttons.append(("TOGGLE_RES_DD", (res_x1, y1, res_x2, y2), "RES_DROPDOWN"))
        self._resolution_rect = (res_x1, y1, res_x2, y2)

        # 3. 开启/关闭按钮 (70px)
        cx = res_x2 + gap
        sw_x1 = cx; sw_x2 = cx + 70
        is_on = self.pipeline_running
        if is_on:
            sw_bg = (55, 45, 30); sw_border = (255, 160, 40); sw_txt = (255, 200, 80)
            sw_label = "关闭"
        else:
            sw_bg = (30, 50, 40); sw_border = (0, 200, 120); sw_txt = (80, 230, 160)
            sw_label = "开启"
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_bg, -1)
        cv2.rectangle(canvas, (sw_x1, y1), (sw_x2, y2), sw_border, 1)
        font = get_cached_font(12, bold=True)
        bbox = font.getbbox(sw_label)
        lw = bbox[2] - bbox[0]; lh = bbox[3] - bbox[1]
        tx = sw_x1 + (70 - lw) // 2 - bbox[0]
        ty = y1 + (btn_h - lh) // 2 - bbox[1] + 1
        draw_text(canvas, sw_label, (tx, ty), font_size=12, color=sw_txt, bold=True)
        self.gui_buttons.append(("TOGGLE_CAMERA", (sw_x1, y1, sw_x2, y2), None))

        # 4. 普通按钮 (开启/关闭右边)
        cx = sw_x2 + gap
        for label, action_id, btype, opt, get_state in buttons:
            x1 = cx
            x2 = cx + btn_w

            if btype == "toggle":
                active = get_state()
                bg = self.COLOR_CARD_SEL if active else self.COLOR_CARD_BG
                border = self.COLOR_BORDER_SEL if active else self.COLOR_BORDER
                text_col = self.COLOR_ACCENT if active else self.COLOR_TEXT_SUB
            else:
                bg = self.COLOR_CARD_BG
                border = self.COLOR_BORDER
                text_col = self.COLOR_TEXT_SUB

            cv2.rectangle(canvas, (x1, y1), (x2, y2), bg, -1)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), border, 1)

            is_active = (btype == "toggle" and get_state())
            font = get_cached_font(12, bold=is_active)
            bbox = font.getbbox(label)
            lw = bbox[2] - bbox[0]; lh = bbox[3] - bbox[1]
            tx = x1 + (btn_w - lw) // 2 - bbox[0]
            ty = y1 + (btn_h - lh) // 2 - bbox[1] + 1
            draw_text(canvas, label, (tx, ty), font_size=12, color=text_col, bold=is_active)

            self.gui_buttons.append((action_id, (x1, y1, x2, y2), action_id))
            cx = x2 + gap

        # 5. 退出按钮 (最右)
        exit_x1 = tw - 80
        exit_x2 = tw - 8
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (exit_x1, y1), (exit_x2, y2), (60, 60, 80), 1)
        draw_text(canvas, "退出 X", (exit_x1 + 10, y1 + (btn_h - 14) // 2), font_size=12, color=(190, 190, 200), bold=True)
        self.gui_buttons.append(("QUIT", (exit_x1, y1, exit_x2, y2), None))

        # 展开的下拉浮层
        if self.active_dropdown == "CAMERA_TYPE_DROPDOWN" and self._camera_type_rect:
            self._render_dropdown_popup(canvas, self._camera_type_rect, self.camera_options, self.camera_type, btn_prefix="DD_CAM_")
        elif self.active_dropdown == "RES_DROPDOWN" and self._resolution_rect:
            self._render_dropdown_popup(canvas, self._resolution_rect, self.resolution_options, self.resolution, btn_prefix="DD_RES_")

    def run(self):
        """主可视化与交互事件循环 — 不自动开相机，等用户点开启"""
        self._quit_requested = False

        window_name = "d435_viewer"  # 窗口 key 纯 ASCII (namedWindow ANSI API)
        self.win_mgr.setup_window(window_name, self.on_mouse)
        self.win_mgr.set_unicode_title("RealSense D435 深度相机诊断")

        content_w = 1024
        content_h = 576

        log.info("D435 深度相机诊断工具已启动。")

        fps_counter = 0
        fps_time = time.time()
        current_fps = 30.0

        try:
            while True:
                if self._quit_requested:
                    break

                # === 相机未开启：渲染占位画面 ===
                if not self.pipeline_running:
                    content = np.full((content_h, content_w, 3), self.COLOR_BG, dtype=np.uint8)
                    title = "相机未开启"
                    subtitle = "请先选择相机类型和分辨率，然后点击 [开启] 按钮"
                    put_text(content, title, (content_w // 2 - 140, content_h // 2 - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.COLOR_ACCENT, 2, cv2.LINE_AA)
                    put_text(content, subtitle, (content_w // 2 - 200, content_h // 2 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_TEXT_SUB, 1, cv2.LINE_AA)
                    # 拼合
                    tool_area = np.full((self.TOOLBAR_H, content_w, 3), self.COLOR_BG, dtype=np.uint8)
                    full_canvas = np.vstack((tool_area, content))
                    self._draw_toolbar(full_canvas)
                    cv2.imshow(window_name, full_canvas)
                    poll_res = self.win_mgr.poll_events()
                    if poll_res.should_quit or self._quit_requested:
                        break
                    raw_key = cv2.waitKeyEx(30)
                    if raw_key in [ord('q'), 27]:
                        break
                    continue

                # === 相机已开启：正常采集 ===
                if not self.is_paused:
                    if self.camera_type == "realsense":
                        frames = self.pipeline.wait_for_frames()
                        aligned_frames = self.align.process(frames)
                        color_frame = aligned_frames.get_color_frame()
                        depth_frame = aligned_frames.get_depth_frame()
                        if not color_frame or not depth_frame:
                            continue
                        filtered_depth = self.apply_filters(depth_frame)
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(filtered_depth.get_data())
                    else:  # USB
                        color_image, depth_image = self._usb_read_frames()
                        if color_image is None:
                            continue
                        depth_image = None  # USB 无深度

                    self.paused_color_frame = color_image.copy()
                    self.paused_depth_frame = depth_image.copy() if depth_image is not None else None
                else:
                    color_image = self.paused_color_frame.copy() if self.paused_color_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    depth_image = self.paused_depth_frame.copy() if self.paused_depth_frame is not None else None
                    time.sleep(0.03)

                # USB 无深度图 → 生成 dummy 让下游代码跑过
                has_depth = (depth_image is not None)
                if not has_depth and color_image is not None:
                    h, w = color_image.shape[:2]
                    depth_image = np.zeros((h, w), dtype=np.uint16)

                # 2. 深度热力图渲染 (相机原生绝对深度)
                depth_meters = depth_image.astype(float) * 0.001
                if self.auto_range:
                    valid_mask = (depth_image > 200) & (depth_image < 1500)
                    if np.count_nonzero(valid_mask) > 100:
                        act_min = float(np.percentile(depth_meters[valid_mask], 2))
                        act_max = float(np.percentile(depth_meters[valid_mask], 98))
                        act_max = max(act_max, act_min + 0.03)
                    else:
                        act_min, act_max = self.cmap_min, self.cmap_max
                else:
                    act_min, act_max = self.cmap_min, self.cmap_max

                norm = np.clip((act_max - depth_meters) / max(0.005, (act_max - act_min)) * 255.0, 0, 255).astype(np.uint8)
                depth_colormap = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                depth_colormap[depth_image == 0] = (25, 25, 25)

                display_color_image = color_image.copy()

                # 4. 鼠标探针测距
                target_pt = self.selected_point if self.selected_point else (self.hover_x, self.hover_y)
                probe_text = "Probe: N/A"
                if target_pt[0] >= 0 and target_pt[1] >= 0:
                    px, py = target_pt
                    if py < depth_image.shape[0] and px < depth_image.shape[1]:
                        depth_mm = depth_image[py, px]
                        if depth_mm > 0 and self.color_intrinsics:
                            pt_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [px, py], depth_mm * 0.001)
                            probe_text = f"({px},{py}) depth:{depth_mm}mm  3D:(X:{pt_3d[0]*1000:+.1f}, Y:{pt_3d[1]*1000:+.1f}, Z:{pt_3d[2]*1000:+.1f})mm"
                        elif depth_mm > 0:
                            probe_text = f"({px},{py}) depth:{depth_mm}mm"
                        cv2.drawMarker(display_color_image, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
                        cv2.drawMarker(depth_colormap, (px, py), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                fps_counter += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_time)
                    fps_counter = 0
                    fps_time = time.time()

                # 5. 合成内容区域画布（根据 show_rgb/show_depth/split_vertical）
                final_content = self._compose_content(
                    display_color_image if self.show_rgb else None,
                    depth_colormap if self.show_depth else None,
                    content_w, content_h
                )
                canvas_w = final_content.shape[1]
                canvas_h = final_content.shape[0]

                # 6. 状态条（画在内容区域顶部）
                status_info = self._build_status_line(current_fps)
                cv2.rectangle(final_content, (0, 0), (canvas_w, 26), (25, 30, 38), -1)
                put_text(final_content, status_info, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 220), 1)

                # 底部探针栏
                cv2.rectangle(final_content, (0, canvas_h - 24), (canvas_w, canvas_h), (20, 20, 20), -1)
                put_text(final_content, probe_text, (10, canvas_h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # 7. 拼合：工具栏 + 内容区域
                tool_area = np.full((self.TOOLBAR_H, canvas_w, 3), self.COLOR_BG, dtype=np.uint8)
                full_canvas = np.vstack((tool_area, final_content))

                # 8. 自适应窗口分辨率
                if self.win_mgr.canvas_w == full_canvas.shape[1] and self.win_mgr.canvas_h == full_canvas.shape[0]:
                    disp_canvas = full_canvas
                else:
                    disp_canvas = np.full((self.win_mgr.canvas_h, self.win_mgr.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)
                    sc_fit = min(self.win_mgr.canvas_w / float(full_canvas.shape[1]),
                                 self.win_mgr.canvas_h / float(full_canvas.shape[0]))
                    tw = int(round(full_canvas.shape[1] * sc_fit))
                    th = int(round(full_canvas.shape[0] * sc_fit))
                    interp = cv2.INTER_AREA if sc_fit < 1.0 else cv2.INTER_LANCZOS4
                    scaled = cv2.resize(full_canvas, (tw, th), interpolation=interp)
                    px = (self.win_mgr.canvas_w - tw) // 2
                    py = (self.win_mgr.canvas_h - th) // 2
                    disp_canvas[py:py + th, px:px + tw] = scaled

                # 在最终显示画布的工具栏区域绘制按钮
                self._draw_toolbar(disp_canvas)

                cv2.imshow(window_name, disp_canvas)

                # 9. 视窗事件与键盘响应
                poll_res = self.win_mgr.poll_events()
                if poll_res.should_quit or self._quit_requested:
                    break

                raw_key = cv2.waitKeyEx(1)
                if raw_key == -1:
                    continue

                fb_changed, _ = self.win_mgr.handle_keyboard_fallback(raw_key)
                if fb_changed:
                    continue

                key = raw_key & 0xFF
                if key in [ord('q'), 27]:
                    break

                # [Space] 暂停
                elif key == 32:
                    self.is_paused = not self.is_paused

                # [V] 排列切换
                elif key in [ord('v'), ord('V')]:
                    self.split_vertical = not self.split_vertical

                # [A] Auto-Range
                elif key == ord('a'):
                    self.auto_range = not self.auto_range

                # 色阶微调
                elif key == ord('['):
                    self.cmap_min = max(0.10, round(self.cmap_min - 0.01, 3))
                    self.auto_range = False
                elif key == ord(']'):
                    self.cmap_min = min(self.cmap_max - 0.02, round(self.cmap_min + 0.01, 3))
                    self.auto_range = False
                elif key == ord('-'):
                    self.cmap_max = max(self.cmap_min + 0.02, round(self.cmap_max - 0.01, 3))
                    self.auto_range = False
                elif key == ord('='):
                    self.cmap_max = min(2.0, round(self.cmap_max + 0.01, 3))
                    self.auto_range = False
                elif key == ord('r'):
                    self.cmap_min = 0.48
                    self.cmap_max = 0.66
                    self.auto_range = False

        finally:
            self._save_viewer_state()
            if hasattr(self, "win_mgr") and self.win_mgr:
                self.win_mgr.save_settings()
            # RealSense
            if self.pipeline:
                try:
                    self.pipeline.stop()
                except Exception:
                    pass  # 清理容错：退出时停止失败无伤大雅
            # USB
            if self.usb_capture:
                try:
                    self.usb_capture.release()
                except Exception:
                    pass  # 清理容错：退出时释放失败无伤大雅
                self.usb_capture = None
            self.pipeline_running = False
            cv2.destroyAllWindows()
            log.info("采集与可视化窗口已安全退出。")

    @staticmethod
    def _letterbox(src, dst_w, dst_h, bg_color=(15, 17, 21)):
        """保持原始宽高比缩放，黑边填充到 dst_w x dst_h"""
        src_h, src_w = src.shape[:2]
        if src_h == 0 or src_w == 0:
            return np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
        scale = min(dst_w / float(src_w), dst_h / float(src_h))
        nw, nh = int(round(src_w * scale)), int(round(src_h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(src, (nw, nh), interpolation=interp)
        canvas = np.full((dst_h, dst_w, 3), bg_color, dtype=np.uint8)
        x0 = (dst_w - nw) // 2
        y0 = (dst_h - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        return canvas

    def _compose_content(self, rgb_img, depth_img, target_w, target_h):
        """根据 show_rgb/show_depth/split_vertical 合成内容画布 (保持原始宽高比)"""
        bg = self.COLOR_BG

        if rgb_img is None and depth_img is None:
            return np.full((target_h, target_w, 3), bg, dtype=np.uint8)

        if rgb_img is not None and depth_img is None:
            return self._letterbox(rgb_img, target_w, target_h, bg)

        if rgb_img is None and depth_img is not None:
            return self._letterbox(depth_img, target_w, target_h, bg)

        # 两者都有 - 按排列方式组合
        if self.split_vertical:
            half_h = target_h // 2
            top = self._letterbox(rgb_img, target_w, half_h, bg)
            bottom = self._letterbox(depth_img, target_w, target_h - half_h, bg)
            cv2.line(bottom, (0, 0), (target_w, 0), (80, 80, 80), 1)
            return np.vstack((top, bottom))
        else:
            half_w = target_w // 2
            left = self._letterbox(rgb_img, half_w, target_h, bg)
            right = self._letterbox(depth_img, target_w - half_w, target_h, bg)
            cv2.line(right, (0, 0), (0, target_h), (80, 80, 80), 1)
            return np.hstack((left, right))

    def _build_status_line(self, fps):
        """构建状态行文字"""
        parts = []

        # 视图/排列
        view = f"{'上下' if self.split_vertical else '左右'}排列" if (self.show_rgb and self.show_depth) else \
               ("仅RGB" if self.show_rgb else "仅Depth")
        parts.append(view)

        # 暂停提示
        if self.is_paused:
            parts.append("[PAUSED]")

        # FPS
        parts.append(f"FPS:{fps:.0f}")

        # 深度色阶
        if self.auto_range:
            parts.append("色阶:自动")
        else:
            parts.append(f"色阶:{self.cmap_min:.2f}~{self.cmap_max:.2f}m")

        return " | ".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RealSense D435 实时深度探针与对齐查看工具")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="配置文件路径")
    args = parser.parse_args()

    try:
        viewer = D435Viewer(config_path=args.config)
        viewer.run()
    except Exception as e:
        log.warning(f"\n运行中断: {e}")
