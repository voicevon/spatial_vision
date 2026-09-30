#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 管理器 — 左右两栏布局的 cv2 原生 GUI
左侧 sidebar 为 Tab 卡片切换，右侧为内容区域。
Tab 1: 图纸生成 (import generate_tags() 函数)
Tab 2: 白名单管理 (30 个 Tag ID toggle + 世界锚点编辑 + 保存 config.yaml)
"""

import os
import sys
import json
import yaml
import time
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.ui.gui_window_manager import GuiWindowManager
from src.ui.gui_theme import GuiTheme
from src.ui.text_rendering import draw_text
from src.ui.gui_components import draw_app_header, TabBar, TabItem
from src.utils.config_guard import load_raw_config
from src.workspace.workspace_manager import (
    WorkspaceManager,
    load_workspace_anchor_tags,
    load_workspace_tag_whitelist,
    load_workspace_tag_config,
    save_workspace_tag_config,
)
from src.calibration.solvers.world_datum_aligner import WorldDatumAligner

# 复用旧代码的图纸生成函数 (不修改旧代码)
from tools.calibration.generate_apriltags import generate_tags

from src.utils.logger import get_logger

log = get_logger(__name__)


class TagManager:
    """AprilTag 管理器 GUI"""

    # —— 工业暗色主题 (与 d435_viewer / gui_launcher 一致) ——
    # 调色板: 统一取自 GuiTheme 主题单源 (原略有差异的局部色值已归一到全局色板)
    COLOR_BG = GuiTheme.BG
    COLOR_SIDEBAR = GuiTheme.CARD_BG
    COLOR_CARD_BG = GuiTheme.CARD_HOVER
    COLOR_CARD_SEL = GuiTheme.CARD_SEL
    COLOR_BORDER = GuiTheme.BORDER
    COLOR_BORDER_SEL = GuiTheme.BORDER_SEL
    COLOR_TEXT = GuiTheme.TEXT
    COLOR_TEXT_SUB = GuiTheme.TEXT_SUB
    COLOR_ACCENT = GuiTheme.ACCENT
    COLOR_ACCENT2 = (0, 170, 255)      # 次强调 (橙), 本地保留
    COLOR_OK = GuiTheme.OK
    COLOR_WARN = GuiTheme.WARN
    COLOR_ERR = GuiTheme.ERR
    COLOR_TAG_ON = (0, 160, 140)       # 白名单内 Tag (青绿色), 本地保留
    COLOR_TAG_OFF = (75, 80, 95)       # 白名单外 Tag, 本地保留
    COLOR_ANCHOR = (0, 215, 255)       # 已标定世界锚点 (金色描边/角标), 本地保留

    # 布局常量
    TOOLBAR_H = 46                      # 统一顶部工具栏高度
    MARGIN = 12

    # Tag 16h5 共 30 个 (ID 0~29)
    TAG_COUNT = 30
    TAG_PRESET_DEFAULT = [0, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")  # 合并: tag_manager 段统一存 config/gui_settings.json
    APP_ID = "tag_manager"

    def __init__(self):
        self.win_mgr = GuiWindowManager(
            self.APP_ID,
            base_w=1100, base_h=680,
            min_w=860, min_h=500,
            settings_file=self.SETTINGS_FILE,
        )
        self.app_id = self.APP_ID

        # —— 通用 TabBar 组件 (管理页签状态流转与单源排版) ——
        self.tab_bar = TabBar(
            tabs=[
                TabItem(key="generator", label="📐 图纸生成"),
                TabItem(key="whitelist", label="✅ 白名单管理"),
            ],
            active_key="generator",
            style=TabBar.STYLE_PILL,
            tab_height=30,
            spacing=8,
            fixed_width=140,
            font_size=12,
            on_change=lambda _: self._save_settings(),
        )
        self._quit_requested = False
        self.mouse_pos = (0, 0)
        self.gui_buttons = []             # 每帧重建, 用于 hit-test

        # —— Tab 1: 图纸生成参数 ——
        self.gen_tag_count = 30
        self.gen_pixel_size = 800
        self.gen_border_bits = 2
        self.gen_output_dir = os.path.join(PROJECT_ROOT, "data", "apriltags_16h5")
        self.gen_status = ""              # 生成进度/结果信息
        self.gen_preview = None           # 生成后的总览网格图
        self.gen_running = False

        # —— Tab 2: 白名单 ——
        self.valid_tag_ids = self._load_valid_tag_ids()
        # 世界锚点表 {tid: {"xyz_mm": [x,y,z], "known": [b,b,b]}} (约束积累式世界锚定数据源)
        # 工位沙盒感知: 当前活动工位自有 anchor_tags.yaml 优先, 缺失回退全局 config.yaml 旧源
        self.anchor_tags = self._load_anchor_tags_ws()
        # 锚点编辑器工作态: {"tid": int, "xyz": [f,f,f], "known": [b,b,b]} 或 None
        self.anchor_edit = None

        self._load_settings()

    @property
    def active_tab(self) -> str:
        return self.tab_bar.active_key or "generator"

    @active_tab.setter
    def active_tab(self, key: str):
        self.tab_bar.select(key)

    # ================================================================
    # 持久化
    # ================================================================
    def _load_settings(self):
        try:
            if not os.path.exists(self.SETTINGS_FILE):
                return
            with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                root = json.load(f)
            st = root.get(self.APP_ID, {})
            if "gen_tag_count" in st:
                self.gen_tag_count = int(st["gen_tag_count"])
            if "gen_pixel_size" in st:
                self.gen_pixel_size = int(st["gen_pixel_size"])
        except Exception as e:
            log.warning(f"恢复标靶生成器设置失败，使用默认值: {e}")

    def _save_settings(self):
        try:
            os.makedirs(os.path.dirname(self.SETTINGS_FILE), exist_ok=True)
            root = {}
            if os.path.exists(self.SETTINGS_FILE):
                try:
                    with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                        root = json.load(f)
                except Exception:
                    pass  # 已有设置读取失败时回退为空字典，避免阻塞保存
            st = root.setdefault(self.APP_ID, {})
            st["gen_tag_count"] = self.gen_tag_count
            st["gen_pixel_size"] = self.gen_pixel_size
            st["active_tab"] = self.active_tab
            st["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(root, f, indent=2, ensure_ascii=False)
        except Exception as e:
            log.warning(f"保存标靶生成器设置失败: {e}")

    def _load_valid_tag_ids(self):
        """Tag ID 白名单已 100% 下沉至工位 tag_whitelist.yaml (tags 单一真理源)"""
        try:
            ws = WorkspaceManager().get_current_workspace()
            return load_workspace_tag_whitelist(ws.workspace_dir)
        except Exception as e:
            log.warning(f"[TagMgr] 加载工位白名单失败: {e}")
            return []

    def _save_valid_tag_ids(self):
        """写穿白名单到当前活动工位的 tag_whitelist.yaml (tags 单一真理源)"""
        try:
            ws = WorkspaceManager().get_current_workspace()
            curr = load_workspace_tag_config(ws.workspace_dir)
            old_tags = curr.get("tags") or {}
            new_tags = {}
            for tid in sorted(self.valid_tag_ids):
                new_tags[tid] = old_tags.get(tid, {})
            ok = save_workspace_tag_config(ws, tags=new_tags)
            if ok:
                self.gen_status = f"✅ 白名单已保存到工位 {ws.workspace_id}: {sorted(self.valid_tag_ids)}"
            else:
                self.gen_status = "❌ 保存失败 (工位白名单写穿异常)"
        except Exception as e:
            self.gen_status = f"❌ 保存失败: {e}"

    def _load_anchor_tags_ws(self):
        """锚点已 100% 下沉至工位沙盒 tag_whitelist.yaml (tags.xyz_mm); 全局 config.yaml 不再兜底"""
        try:
            ws = WorkspaceManager().get_current_workspace()
            own = load_workspace_anchor_tags(ws.workspace_dir)
            if own is not None:
                return own
        except Exception as e:
            log.warning(f"[TagMgr] 加载工位锚点失败: {e}")
        return None  # 全局兜底已禁用 (FR-9.6 严禁以打印边长兜底)

    def _save_anchor_tags(self):
        """写穿世界锚点到当前活动工位的 tag_whitelist.yaml (tags 单一真理源)"""
        try:
            ws = WorkspaceManager().get_current_workspace()
            curr = load_workspace_tag_config(ws.workspace_dir)
            tags = curr.get("tags") or {}
            for tid, tcfg in tags.items():
                if tid not in (self.anchor_tags or {}):
                    tcfg.pop("xyz_mm", None)
            for tid, a in (self.anchor_tags or {}).items():
                if tid not in tags:
                    tags[tid] = {}
                if "xyz_mm" in a:
                    tags[tid]["xyz_mm"] = a["xyz_mm"]
            ok = save_workspace_tag_config(ws, tags=tags)
            ids = sorted(self.anchor_tags.keys()) if self.anchor_tags else []
            if ok:
                self.gen_status = f"✅ 世界锚点已保存到工位 {ws.workspace_id}: {ids if ids else '(空)'}"
            else:
                self.gen_status = "❌ 保存失败 (工位锚点写穿异常)"
        except Exception as e:
            self.gen_status = f"❌ 保存失败: {e}"

    # ================================================================
    # 事件处理
    # ================================================================
    def on_mouse(self, event, x, y, flags, param):
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._hit_test(x, y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._hit_test_right(x, y)

    def _map_mouse(self, x, y):
        """反算: mouse 回调返回的是窗口物理坐标, 如果 render 尺寸和窗口不一致需要映射"""
        cw = self.win_mgr.canvas_w
        ch = self.win_mgr.canvas_h
        full_w = getattr(self, "_last_full_w", cw)
        full_h = getattr(self, "_last_full_h", ch)
        if full_w > 0 and full_h > 0 and (cw != full_w or ch != full_h):
            scale = min(cw / float(full_w), ch / float(full_h))
            pad_x = (cw - int(full_w * scale)) // 2
            pad_y = (ch - int(full_h * scale)) // 2
            return int((x - pad_x) / max(1e-6, scale)), int((y - pad_y) / max(1e-6, scale))
        return x, y

    def _hit_test(self, x, y):
        x, y = self._map_mouse(x, y)
        for btn_id, (bx1, by1, bx2, by2), payload in self.gui_buttons:
            if bx1 <= x <= bx2 and by1 <= y <= by2:
                self._on_click(btn_id, payload)
                return

    def _hit_test_right(self, x, y):
        """右键: 白名单内 Tag 打开世界锚点编辑器 (编辑器打开时不响应)"""
        if self.anchor_edit is not None:
            return
        x, y = self._map_mouse(x, y)
        for btn_id, (bx1, by1, bx2, by2), payload in self.gui_buttons:
            if btn_id.startswith("TAG_") and bx1 <= x <= bx2 and by1 <= y <= by2:
                tid = int(payload)
                if tid in self.valid_tag_ids:
                    e = self.anchor_tags.get(tid)
                    self.anchor_edit = {
                        "tid": tid,
                        "xyz": [float(v) for v in e["xyz_mm"]] if e else [0.0, 0.0, 0.0],
                        "known": [bool(v) for v in e["known"]] if e else [True, True, True],
                    }
                return

    def _on_click(self, btn_id, payload):
        if btn_id == "QUIT":
            self._quit_requested = True
            return
        if btn_id.startswith("TAB_"):
            self.active_tab = payload
            self._save_settings()
        elif btn_id == "GENERATE":
            self._do_generate()
        elif btn_id == "WHITELIST_ALL":
            self.valid_tag_ids = list(range(self.TAG_COUNT))
            self._save_valid_tag_ids()
        elif btn_id == "WHITELIST_CLEAR":
            self.valid_tag_ids = []
            self._save_valid_tag_ids()
        elif btn_id == "WHITELIST_DEFAULT":
            self.valid_tag_ids = list(self.TAG_PRESET_DEFAULT)
            self._save_valid_tag_ids()
        elif btn_id == "WHITELIST_SAVE":
            self._save_valid_tag_ids()
        elif btn_id.startswith("TAG_"):
            tid = int(payload)
            if tid in self.valid_tag_ids:
                self.valid_tag_ids.remove(tid)
            else:
                self.valid_tag_ids.append(tid)
        elif btn_id == "ANCH_CANCEL":
            self.anchor_edit = None
        elif btn_id == "ANCH_SAVE":
            tid = self.anchor_edit["tid"]
            self.anchor_tags[tid] = {
                "xyz_mm": [float(v) for v in self.anchor_edit["xyz"]],
                "known": [bool(v) for v in self.anchor_edit["known"]],
            }
            self._save_anchor_tags()
            self.anchor_edit = None
        elif btn_id == "ANCH_DEL":
            self.anchor_tags.pop(self.anchor_edit["tid"], None)
            self._save_anchor_tags()
            self.anchor_edit = None
        elif btn_id.startswith("ANCH_AXY_"):
            axis = int(btn_id.rsplit("_", 1)[1])
            self.anchor_edit["xyz"][axis] = round(self.anchor_edit["xyz"][axis] + int(payload), 1)
        elif btn_id.startswith("ANCH_KN_"):
            axis = int(payload)
            self.anchor_edit["known"][axis] = not self.anchor_edit["known"][axis]

    def _is_hover(self, bx1, by1, bx2, by2):
        x, y = self.mouse_pos
        return bx1 <= x <= bx2 and by1 <= y <= by2

    def _do_generate(self):
        """调用 generate_tags() 生成图纸"""
        if self.gen_running:
            return
        self.gen_running = True
        self.gen_status = "⏳ 正在生成..."
        try:
            generate_tags(
                output_dir=self.gen_output_dir,
                tag_count=self.gen_tag_count,
                tag_pixel_size=self.gen_pixel_size,
                border_bits=self.gen_border_bits,
            )
            # 尝试加载生成的网格预览图
            grid_path = os.path.join(self.gen_output_dir, "apriltags_16h5_all_grid.png")
            if os.path.exists(grid_path):
                self.gen_preview = cv2.imread(grid_path)
            self.gen_status = f"✅ 完成: {self.gen_tag_count} 个 Tag → {self.gen_output_dir}"
        except Exception as e:
            self.gen_status = f"❌ 生成失败: {e}"
        finally:
            self.gen_running = False
            self._save_settings()

    # ================================================================
    # 渲染
    # ================================================================
    def _render(self, canvas_w, canvas_h):
        canvas = np.full((canvas_h, canvas_w, 3), self.COLOR_BG, dtype=np.uint8)
        self.gui_buttons = []

        # —— 1. 顶部工具栏 (高度 TOOLBAR_H) ——
        cv2.rectangle(canvas, (0, 0), (canvas_w, self.TOOLBAR_H), GuiTheme.CARD_BG, -1)
        cv2.line(canvas, (0, self.TOOLBAR_H - 1), (canvas_w, self.TOOLBAR_H - 1), GuiTheme.BORDER, 1)

        # 1.1 左侧品牌 LOGO 与模块名称
        tabs_start_x = draw_app_header(canvas, x=12, y=7, sub_title="AprilTag 管理器", icon_size=32)

        # 1.2 中间 Tab 选项卡按钮组 (由通用 TabBar 统一排版与渲染)
        bx = tabs_start_x + 10
        btn_h = 30
        by1 = (self.TOOLBAR_H - btn_h) // 2
        tab_layout = self.tab_bar.render(
            canvas,
            container_rect=(bx, by1, 320, btn_h),
            mouse_pos=self.mouse_pos,
        )
        for key, (x1, y1, w, h) in tab_layout:
            self.gui_buttons.append((f"TAB_{key}", (x1, y1, x1 + w, y1 + h), key))

        # 1.3 右上角统一退出按钮
        qx2 = canvas_w - 12
        qx1 = qx2 - 82
        qy1 = (self.TOOLBAR_H - btn_h) // 2
        qy2 = qy1 + btn_h
        hover_q = self._is_hover(qx1, qy1, qx2, qy2)
        q_bg = (60, 30, 30) if hover_q else (35, 28, 28)
        q_border = self.COLOR_ERR if hover_q else (90, 50, 50)
        cv2.rectangle(canvas, (qx1, qy1), (qx2, qy2), q_bg, -1)
        cv2.rectangle(canvas, (qx1, qy1), (qx2, qy2), q_border, 2 if hover_q else 1)
        draw_text(canvas, "✕ 退出 [Q]", (qx1 + 10, qy1 + 19), font_size=12, color=(255, 140, 140), bold=True)
        self.gui_buttons.append(("QUIT", (qx1, qy1, qx2, qy2), None))

        # —— 2. 下方工作内容区 ——
        rx1 = 0
        rx2 = canvas_w
        ry1 = self.TOOLBAR_H
        ry2 = canvas_h

        if self.active_tab == "generator":
            self._render_generator(canvas, rx1, ry1, rx2, ry2)
        elif self.active_tab == "whitelist":
            self._render_whitelist(canvas, rx1, ry1, rx2, ry2)

        return canvas

    def _render_generator(self, canvas, x1, y1, x2, y2):
        """图纸生成 Tab"""
        content_area = (x1 + self.MARGIN, y1 + self.MARGIN,
                        x2 - self.MARGIN, y2 - self.MARGIN)

        # —— 参数面板 (左半) ——
        param_w = 360
        param_x1, param_x2 = content_area[0], content_area[0] + param_w
        param_y1, param_y2 = content_area[1], content_area[3]
        cv2.rectangle(canvas, (param_x1, param_y1), (param_x2, param_y2), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (param_x1, param_y1), (param_x2, param_y2), self.COLOR_BORDER, 1)

        draw_text(canvas, "生成参数", (param_x1 + 14, param_y1 + 22), font_size=13, color=self.COLOR_TEXT, bold=True)
        cv2.line(canvas, (param_x1 + 14, param_y1 + 34), (param_x2 - 14, param_y1 + 34), self.COLOR_BORDER, 1)

        # 参数行
        ly = param_y1 + 54
        for label, value in [
            ("标靶数量",  f"{self.gen_tag_count} 个 (ID 00 ~ {self.gen_tag_count - 1:02d})"),
            ("像素尺寸",  f"{self.gen_pixel_size} × {self.gen_pixel_size}"),
            ("边框宽度",  f"{self.gen_border_bits} bits"),
            ("输出目录",  self.gen_output_dir),
        ]:
            draw_text(canvas, label, (param_x1 + 14, ly), font_size=11, color=self.COLOR_TEXT_SUB)
            draw_text(canvas, value, (param_x1 + 110, ly), font_size=11, color=self.COLOR_TEXT)
            ly += 24

        # 生成按钮
        btn_y = ly + 16
        btn_h = 36
        btn_x1 = param_x1 + 14
        btn_x2 = param_x2 - 14
        hover_g = self._is_hover(btn_x1, btn_y, btn_x2, btn_y + btn_h)
        if self.gen_running:
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), (40, 45, 55), -1)
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), self.COLOR_BORDER, 1)
            draw_text(canvas, "⏳ 生成中...", (btn_x1 + (btn_x2 - btn_x1) // 2 - 40, btn_y + 23),
                      font_size=12, color=self.COLOR_TEXT_SUB, bold=True)
        else:
            g_bg = (30, 95, 80) if hover_g else (20, 70, 60)
            g_border = self.COLOR_ACCENT if hover_g else (0, 170, 150)
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), g_bg, -1)
            cv2.rectangle(canvas, (btn_x1, btn_y), (btn_x2, btn_y + btn_h), g_border, 2)
            draw_text(canvas, "▶ 开始生成 PDF / PNG", (btn_x1 + 14, btn_y + 23),
                      font_size=12, color=self.COLOR_ACCENT, bold=True)
        self.gui_buttons.append(("GENERATE", (btn_x1, btn_y, btn_x2, btn_y + btn_h), None))

        # 状态行
        status_y = btn_y + btn_h + 14
        if self.gen_status:
            sc = self.COLOR_OK if "✅" in self.gen_status else (self.COLOR_ERR if "❌" in self.gen_status else self.COLOR_TEXT)
            draw_text(canvas, self.gen_status, (param_x1 + 14, status_y + 10), font_size=11, color=sc)

        # —— 预览面板 (右半) ——
        prev_x1 = param_x2 + self.MARGIN
        prev_x2 = content_area[2]
        prev_y1 = content_area[1]
        prev_y2 = content_area[3]
        cv2.rectangle(canvas, (prev_x1, prev_y1), (prev_x2, prev_y2), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (prev_x1, prev_y1), (prev_x2, prev_y2), self.COLOR_BORDER, 1)

        draw_text(canvas, "📋 预览 / 输出", (prev_x1 + 14, prev_y1 + 22), font_size=12, color=self.COLOR_TEXT, bold=True)
        cv2.line(canvas, (prev_x1 + 14, prev_y1 + 34), (prev_x2 - 14, prev_y1 + 34), self.COLOR_BORDER, 1)

        prev_inner_y = prev_y1 + 44
        prev_h = prev_y2 - prev_inner_y - 10
        prev_w = prev_x2 - prev_x1 - 20

        if self.gen_preview is not None:
            # letterbox 保持比例贴入
            h, w = self.gen_preview.shape[:2]
            scale = min(prev_w / float(w), prev_h / float(h))
            nw, nh = int(round(w * scale)), int(round(h * scale))
            resized = cv2.resize(self.gen_preview, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = prev_x1 + 10 + (prev_w - nw) // 2
            oy = prev_inner_y + (prev_h - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = resized
            cv2.rectangle(canvas, (prev_x1 + 10, prev_inner_y), (prev_x2 - 10, prev_y2 - 10), self.COLOR_BORDER, 1)
        else:
            # 占位
            cv2.rectangle(canvas, (prev_x1 + 10, prev_inner_y), (prev_x2 - 10, prev_y2 - 10), self.COLOR_BG, -1)
            cv2.rectangle(canvas, (prev_x1 + 10, prev_inner_y), (prev_x2 - 10, prev_y2 - 10), self.COLOR_BORDER, 1)
            draw_text(canvas, "生成后总览网格图将显示在这里",
                      (prev_x1 + prev_w // 2 - 140, prev_inner_y + prev_h // 2 - 6),
                      font_size=11, color=self.COLOR_TEXT_SUB)

    def _render_whitelist(self, canvas, x1, y1, x2, y2):
        """白名单管理 Tab — 30 个 Tag 切换方块 + 世界锚点 (金框) 管理"""
        content_area = (x1 + self.MARGIN, y1 + self.MARGIN,
                        x2 - self.MARGIN, y2 - self.MARGIN)

        # —— 顶部操作条 (两行: 白名单操作 + 锚点 DoF 状态) ——
        bar_y = content_area[1]
        bar_h = 58
        cv2.rectangle(canvas, (content_area[0], bar_y), (content_area[2], bar_y + bar_h), self.COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (content_area[0], bar_y), (content_area[2], bar_y + bar_h), self.COLOR_BORDER, 1)

        btn_specs = [
            ("WHITELIST_ALL",     "全选",      (80, 220, 120)),
            ("WHITELIST_CLEAR",   "清空",      (220, 80, 80)),
            ("WHITELIST_DEFAULT", "预设 0+18~29", (180, 160, 40)),
            ("WHITELIST_SAVE",    "💾 保存",   (0, 200, 180)),
        ]
        bx = content_area[0] + 14
        for bid, blabel, bcol in btn_specs:
            bw = 110 if bid != "WHITELIST_DEFAULT" else 140
            bh1, bh2 = bx, bx + bw
            by1, by2 = bar_y + 5, bar_y + 39
            hh = self._is_hover(bh1, by1, bh2, by2)
            if hh:
                bright = tuple(min(255, int(c * 1.2)) for c in bcol)
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), bright, -1)
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), (255, 255, 255), 2)
            else:
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), bcol, -1)
                cv2.rectangle(canvas, (bh1, by1), (bh2, by2), self.COLOR_BORDER, 1)
            draw_text(canvas, blabel, (bh1 + 14, by1 + 16), font_size=11, color=(15, 17, 21), bold=True)
            self.gui_buttons.append((bid, (bh1, by1, bh2, by2), None))
            bx += bw + 8

        # 第一行: 白名单状态
        status_text = f"当前白名单: {len(self.valid_tag_ids)} 个  →  {sorted(self.valid_tag_ids) if self.valid_tag_ids else '(空 = 全量探索)'}"
        draw_text(canvas, status_text, (bx + 6, bar_y + 23), font_size=10, color=self.COLOR_TEXT_SUB)

        # 第二行: 世界锚点 DoF 记账状态 (约束积累式锚定的配置级充足性)
        dof_y = bar_y + 50
        if self.anchor_tags:
            dof = WorldDatumAligner.evaluate_anchor_dof(self.anchor_tags)
            if dof["mode"] == "full":
                dof_text, dof_col = f"世界锚定就绪: {dof['dof_solved']}/5 DoF (可完全锚定)", self.COLOR_OK
            elif dof["mode"] == "partial":
                dof_text, dof_col = f"世界锚定受限: {dof['dof_solved']}/5 DoF (XY 锚定, Z 相对 — 建图后下游守门)", self.COLOR_WARN
            else:
                dof_text, dof_col = f"世界锚定不可用: {dof['dof_solved']}/5 DoF ({dof['reason']})", self.COLOR_ERR
        else:
            dof_text, dof_col = "未配置世界锚点 (右键白名单内 Tag 添加已知坐标)", self.COLOR_TEXT_SUB
        draw_text(canvas, dof_text, (content_area[0] + 14, dof_y), font_size=10, color=dof_col)

        # —— Tag 网格 (5 列 × 6 行) ——
        grid_y = bar_y + bar_h + self.MARGIN
        cols, rows = 5, 6
        gap = 10
        avail_w = content_area[2] - content_area[0] - 28
        avail_h = content_area[3] - grid_y - 28
        card_w = (avail_w - gap * (cols - 1)) // cols
        card_h = (avail_h - gap * (rows - 1)) // rows

        for i in range(self.TAG_COUNT):
            col = i % cols
            row = i // cols
            cx = content_area[0] + 14 + col * (card_w + gap)
            cy = grid_y + row * (card_h + gap)
            is_on = i in self.valid_tag_ids
            is_anchored = i in self.anchor_tags
            hover_t = self._is_hover(cx, cy, cx + card_w, cy + card_h)

            if is_anchored:
                bg = self.COLOR_TAG_ON if is_on else self.COLOR_TAG_OFF
                border = (255, 235, 150) if hover_t else self.COLOR_ANCHOR
                bw = 3
            elif is_on:
                bg = self.COLOR_TAG_ON
                border = (255, 255, 255) if hover_t else self.COLOR_ACCENT
                bw = 3 if hover_t else 2
            else:
                bg = (100, 110, 130) if hover_t else self.COLOR_TAG_OFF
                border = self.COLOR_TEXT if hover_t else self.COLOR_BORDER
                bw = 2 if hover_t else 1

            cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + card_w, cy + card_h), border, bw)

            # Tag ID 大号
            id_str = f"#{i:02d}"
            draw_text(canvas, id_str, (cx + card_w // 2 - 22, cy + card_h // 2 - 4),
                      font_size=16, color=(255, 255, 255), bold=True)

            # 底部状态小标签
            if is_anchored:
                status, sc = "√ 锚点", self.COLOR_ANCHOR
            elif is_on:
                status, sc = "√ IN", self.COLOR_OK
            else:
                status, sc = "× OUT", self.COLOR_TEXT_SUB
            draw_text(canvas, status, (cx + card_w // 2 - 26, cy + card_h - 10),
                      font_size=9, color=sc)

            # 编辑器打开时不注册网格按钮, 防止点击穿透到模态面板之下
            if self.anchor_edit is None:
                self.gui_buttons.append((f"TAG_{i}", (cx, cy, cx + card_w, cy + card_h), str(i)))

        # 底部提示
        tip = "💡 左键切换白名单 | 右键白名单内 Tag 编辑世界锚点 (金框 = 已标定) | 空名单 = 放行所有 30 个 Tag (探索模式)"
        draw_text(canvas, tip, (content_area[0] + 14, content_area[3] - 8),
                  font_size=10, color=self.COLOR_TEXT_SUB)

        if self.anchor_edit is not None:
            self._render_anchor_editor(canvas, content_area[0], content_area[1], content_area[2], content_area[3])

    def _render_anchor_editor(self, canvas, x1, y1, x2, y2):
        """世界锚点编辑器 (模态面板): 逐轴数值步进 + 逐轴已知标记"""
        tid = self.anchor_edit["tid"]
        pw, ph = 520, 236
        px1 = x1 + (x2 - x1 - pw) // 2
        py1 = y1 + (y2 - y1 - ph) // 2
        px2, py2 = px1 + pw, py1 + ph

        cv2.rectangle(canvas, (px1, py1), (px2, py2), (28, 34, 42), -1)
        cv2.rectangle(canvas, (px1, py1), (px2, py2), self.COLOR_ANCHOR, 2)
        draw_text(canvas, f"Tag #{tid:02d} 世界锚点编辑 (机械臂世界系, mm)",
                  (px1 + 16, py1 + 24), font_size=13, color=self.COLOR_TEXT, bold=True)
        cv2.line(canvas, (px1 + 16, py1 + 36), (px2 - 16, py1 + 36), self.COLOR_BORDER, 1)

        # 逐轴行: 轴名 | 数值 | [-10][-1][+1][+10] | 已知 toggle
        row_y = py1 + 48
        for axis, name in enumerate(("X", "Y", "Z")):
            known = self.anchor_edit["known"][axis]
            draw_text(canvas, name, (px1 + 18, row_y + 15), font_size=13,
                      color=self.COLOR_ANCHOR if known else self.COLOR_TEXT_SUB, bold=True)
            val_str = f"{self.anchor_edit['xyz'][axis]:+.1f}"
            draw_text(canvas, val_str, (px1 + 44, row_y + 15), font_size=12,
                      color=(255, 255, 255) if known else self.COLOR_TEXT_SUB, bold=True)

            bx = px1 + 150
            for delta, bw_ in ((-10, 48), (-1, 38), (1, 38), (10, 48)):
                hh = self._is_hover(bx, row_y, bx + bw_, row_y + 26)
                bg = (70, 80, 95) if hh else (52, 60, 72)
                cv2.rectangle(canvas, (bx, row_y), (bx + bw_, row_y + 26), bg, -1)
                cv2.rectangle(canvas, (bx, row_y), (bx + bw_, row_y + 26), self.COLOR_BORDER, 1)
                label = f"{'-' if delta < 0 else '+'}{abs(delta)}"
                draw_text(canvas, label, (bx + bw_ // 2 - 12, row_y + 15), font_size=11, color=(255, 255, 255))
                self.gui_buttons.append((f"ANCH_AXY_{axis}", (bx, row_y, bx + bw_, row_y + 26), str(delta)))
                bx += bw_ + 4

            kx = px1 + 340
            kw = 88
            kh = self._is_hover(kx, row_y, kx + kw, row_y + 26)
            k_bg = (30, 95, 80) if known else (60, 62, 70)
            k_border = self.COLOR_ACCENT if (known or kh) else self.COLOR_BORDER
            cv2.rectangle(canvas, (kx, row_y), (kx + kw, row_y + 26), k_bg, -1)
            cv2.rectangle(canvas, (kx, row_y), (kx + kw, row_y + 26), k_border, 2 if kh else 1)
            draw_text(canvas, "已知 √" if known else "未知 ×",
                      (kx + 14, row_y + 15), font_size=11,
                      color=self.COLOR_ACCENT if known else self.COLOR_TEXT_SUB, bold=True)
            self.gui_buttons.append((f"ANCH_KN_{axis}", (kx, row_y, kx + kw, row_y + 26), str(axis)))
            row_y += 34

        # DoF 即时提示
        preview = {tid: {"xyz_mm": self.anchor_edit["xyz"], "known": self.anchor_edit["known"]}}
        preview.update({k: v for k, v in self.anchor_tags.items() if k != tid})
        dof = WorldDatumAligner.evaluate_anchor_dof(preview) if any(preview[t]["known"][k] for t in preview for k in range(3)) else None
        if dof and dof["mode"] != "none":
            tip = f"保存后系统状态: {dof['dof_solved']}/5 DoF ({'可完全锚定' if dof['mode'] == 'full' else 'XY 锚定, Z 相对'})"
            tip_col = self.COLOR_OK if dof["mode"] == "full" else self.COLOR_WARN
        else:
            tip = "提示: 单枚锚点无尺度信息, 至少需要两枚存在共同已知轴的锚点"
            tip_col = self.COLOR_TEXT_SUB
        draw_text(canvas, tip, (px1 + 16, py1 + ph - 52), font_size=10, color=tip_col)

        # 底部按钮: 删除锚点 / 保存 / 取消 (从右往左排列)
        btn_y = py1 + ph - 38
        for bid, blabel, bcol, bw_ in (
            ("ANCH_DEL",    "删除锚点", (220, 80, 80),   100),
            ("ANCH_SAVE",   "💾 保存",  (0, 200, 180),   90),
            ("ANCH_CANCEL", "取消",     (90, 95, 105),   80),
        ):
            if bid == "ANCH_CANCEL":
                bx1_, bx2_ = px2 - 16 - bw_, px2 - 16
            elif bid == "ANCH_SAVE":
                bx1_, bx2_ = px2 - 16 - bw_ - 88, px2 - 16 - 88
            else:
                bx1_, bx2_ = px2 - 16 - bw_ - 176, px2 - 16 - 176
            hh = self._is_hover(bx1_, btn_y, bx2_, btn_y + 28)
            bg = tuple(min(255, int(c * 1.2)) for c in bcol) if hh else bcol
            cv2.rectangle(canvas, (bx1_, btn_y), (bx2_, btn_y + 28), bg, -1)
            cv2.rectangle(canvas, (bx1_, btn_y), (bx2_, btn_y + 28), (255, 255, 255) if hh else self.COLOR_BORDER, 1)
            draw_text(canvas, blabel, (bx1_ + 12, btn_y + 16), font_size=11, color=(15, 17, 21), bold=True)
            self.gui_buttons.append((bid, (bx1_, btn_y, bx2_, btn_y + 28), None))

    # ================================================================
    # 主循环
    # ================================================================
    def run(self):
        self._quit_requested = False
        window_name = "tag_manager"  # 窗口 key 纯 ASCII (namedWindow ANSI API)
        self.win_mgr.setup_window(window_name, self.on_mouse)
        self.win_mgr.set_unicode_title("AprilTag 管理器")

        # 恢复 active_tab
        try:
            if os.path.exists(self.SETTINGS_FILE):
                with open(self.SETTINGS_FILE, "r", encoding="utf-8") as f:
                    root = json.load(f)
                saved_tab = root.get(self.APP_ID, {}).get("active_tab")
                if saved_tab in ("generator", "whitelist"):
                    self.active_tab = saved_tab
        except Exception as e:
            log.warning(f"恢复上次活动 Tab 设置失败: {e}")

        log.info("AprilTag 管理器已启动。")

        try:
            while True:
                if self._quit_requested:
                    break

                canvas_w = self.win_mgr.canvas_w
                canvas_h = self.win_mgr.canvas_h
                full = self._render(canvas_w, canvas_h)
                self._last_full_w = full.shape[1]
                self._last_full_h = full.shape[0]

                cv2.imshow(window_name, full)

                poll_res = self.win_mgr.poll_events()
                if poll_res.should_quit or self._quit_requested:
                    break

                raw_key = cv2.waitKeyEx(1)
                if raw_key == -1:
                    continue
                if (raw_key & 0xFF) == 27:
                    if self.anchor_edit is not None:
                        self.anchor_edit = None  # ESC 优先关闭锚点编辑器
                        continue
                    break
                if (raw_key & 0xFF) == ord('q'):
                    break
                if (raw_key & 0xFF) == 13 and self.anchor_edit is not None:
                    self._on_click("ANCH_SAVE", None)  # Enter 保存锚点
                    continue
                if (raw_key & 0xFF) == ord('1'):
                    self.active_tab = "generator"
                elif (raw_key & 0xFF) == ord('2'):
                    self.active_tab = "whitelist"

        finally:
            self._save_settings()
            self.win_mgr.save_settings()
            cv2.destroyAllWindows()
            log.info("AprilTag 管理器已安全退出。")


def main():
    mgr = TagManager()
    mgr.run()


if __name__ == "__main__":
    main()
