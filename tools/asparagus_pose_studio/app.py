#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
芦笋位姿工作室 (Asparagus Pose Studio) - 主应用控制器
=====================================================
集成工位感知、流水线算法插拔、视口交互器与主事件循环。
"""

import os
import sys
import time
import argparse
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.gui_window_manager import GuiWindowManager
from src.utils.logger import get_logger
from src.utils.text_rendering import put_text
from src.calibration.workspace_manager import WorkspaceManager
from src.vision.asparagus_analyzer import AsparagusAnalyzer
from src.vision.pipelines import PipelineRegistry, BaseAsparagusPipeline, PipelineResult

from src.utils.base_cv_app import BaseCvApp
from tools.spatial_mapping_studio.mapping_viewport_interactor import MappingViewportInteractor
from tools.asparagus_pose_studio.data_io import (
    APP_ID, BASE_H, BASE_W, DEFAULT_DIR, REPORT_DIR, WINDOW_KEY,
    load_system_config, scan_samples,
    load_studio_settings, save_studio_settings
)
from tools.asparagus_pose_studio.renderer import AsparagusPoseStudioRenderer

log = get_logger(__name__)


class AsparagusPoseStudioApp(BaseCvApp):
    """芦笋位姿工作室 GUI 主应用 (基于 BaseCvApp 轻量基类)"""

    def __init__(self, sample_dir: Optional[str] = None, settings_file: Optional[str] = None):
        super().__init__(
            app_id=APP_ID,
            base_w=BASE_W,
            base_h=BASE_H,
            window_name=WINDOW_KEY,
            window_title="芦笋位姿工作室 - Asparagus Pose Studio",
            settings_file=settings_file,
            min_w=900,
            min_h=600,
            enable_keyboard_zoom=True,
            responsive=True,
        )
        self.settings_file = settings_file
        self.sys_cfg = load_system_config()

        # 工位管理器感知
        self.workspace_mgr = WorkspaceManager()
        cur_ws = self.workspace_mgr.get_current_workspace()
        self.current_workspace_id = cur_ws.workspace_id if cur_ws else ""

        # 样本目录：优先取当前工位 production 采图；若未指定且无工位，回退 DEFAULT_DIR
        if sample_dir:
            self.sample_dir = sample_dir
        else:
            self.sample_dir = cur_ws.prod_raw_images_dir if cur_ws else DEFAULT_DIR

        self.active_dropdown = None
        self._dd_items = []
        self._workspace_rect = None
        self._pipeline_rect = None

        # 标定链: AprilTag 建图定位器 + 手工外参回退
        self.tag_localizer = None
        self._init_localizer()

        # 状态持久化加载 (算法路线、上次选中的样本)
        self._persisted_state = load_studio_settings(self.settings_file)
        persisted_pipe = self._persisted_state.get("pipeline_key")
        valid_pipes = dict(PipelineRegistry.list_options())
        if persisted_pipe and persisted_pipe in valid_pipes:
            self.pipeline_key = persisted_pipe
        else:
            self.pipeline_key = "ridge_tracing"

        self._persisted_sample_name = self._persisted_state.get("selected_sample_name", "")

        # 样本与解算状态
        self.samples = []
        self.sel_idx = -1
        self.scroll_off = 0
        self.targets = []
        self.sel_target = 0
        self.vis_img = None
        self.mode = "3d"             # "3d" | "2d"
        self.error = ""

        # 多技术路线算法流水线
        self.pipeline: Optional[BaseAsparagusPipeline] = None
        self.pipeline_result: Optional[PipelineResult] = None
        self.active_step_key = "stage3_poses"
        self._init_pipeline()

        # 视口平移与无级缩放交互控制器 (自适应双排工具栏顶部高度 86)
        self.viewport = MappingViewportInteractor(top_bar_h=86, bottom_bar_h=46, win_w=BASE_W, win_h=BASE_H)

        # 交互与事件映射
        self.mouse_pos = (-1, -1)
        self._buttons = []
        self._slider_bars = []       # 当前帧步骤滑条命中区 [(rect, spec)]
        self._drag_slider = None     # 拖拽中的滑条 (rect, spec, handle_idx) 左滑块=下限, 右滑块=上限
        self._params_dirty = False  # 滑条参数是否被修改 (离开调参步骤时触发重解算)
        self._sample_rows = []
        self._result_rows = []

        self.rescan(auto_load=True)

    def _save_persisted_state(self):
        """持久化保存当前的算法路线、选中的样本名与各流水线的滑条参数"""
        sel_name = ""
        if 0 <= self.sel_idx < len(self.samples):
            sel_name = self.samples[self.sel_idx]["name"]
        elif self._persisted_sample_name:
            sel_name = self._persisted_sample_name

        # 采集当前流水线全部滑条声明属性的最新值 (HSV 阈值 / 腐蚀膨胀参数等)
        attrs = set()
        for specs in getattr(self.pipeline, "STEP_SLIDERS", {}).values():
            for sp in specs:
                for key in ("attr", "attr_low", "attr_high"):
                    if key in sp:
                        attrs.add(sp[key])
        params = {a: getattr(self.pipeline, a) for a in sorted(attrs)} if (self.pipeline and attrs) else {}

        all_params = dict(self._persisted_state.get("pipeline_params", {}))
        if params:
            all_params[self.pipeline_key] = params

        state = {
            "pipeline_key": self.pipeline_key,
            "selected_sample_name": sel_name,
            "pipeline_params": all_params,
        }
        self._persisted_state.update(state)
        save_studio_settings(state, settings_file=self.settings_file)

    # ------------------------------ 数据与标定 ------------------------------
    def _init_localizer(self):
        """装载当前工位的 AprilTag 空间标靶地图"""
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        tags_path = ws.map_path if ws else ""
        if tags_path and os.path.exists(tags_path) and os.path.getsize(tags_path) > 50:
            try:
                from src.vision.tag_localizer import TagLocalizer
                from src.calibration.workspace_manager import load_workspace_marker_size_mm
                marker_size_mm = load_workspace_marker_size_mm(ws.workspace_dir) if ws else None
                if marker_size_mm is None:
                    log.warning(
                        "工位 %s 的 tag_whitelist.yaml.tag_default_size_mm 缺失或非法, "
                        "TagLocalizer 暂不可用.", ws.workspace_id if ws else "<none>"
                    )
                    self.tag_localizer = None
                    return
                self.tag_localizer = TagLocalizer(
                    tags_map_path=tags_path,
                    marker_size_mm=marker_size_mm,
                )
            except Exception as exc:
                log.warning("AprilTag 定位器加载失败: %s", exc)
                self.tag_localizer = None
        else:
            self.tag_localizer = None

    @property
    def workspace_options(self) -> List[Tuple[str, str]]:
        """各工位立体地图下拉选项"""
        opts = []
        for s in self.workspace_mgr.list_workspaces():
            status = f"RMSE: {s.global_rmse_px:.2f}px" if s.ba_solved else "未平差"
            opts.append((s.workspace_id, f"{s.name} ({s.image_count}帧, {status})"))
        return opts

    @property
    def current_workspace_name(self) -> str:
        ws = self.workspace_mgr.get_workspace_by_id(self.current_workspace_id)
        return ws.name if ws else "默认工位"

    @property
    def pipeline_options(self) -> List[Tuple[str, str]]:
        """算法流水线选项"""
        return PipelineRegistry.list_options()

    @property
    def current_pipeline_name(self) -> str:
        return dict(self.pipeline_options).get(self.pipeline_key, self.pipeline_key)

    def switch_workspace(self, workspace_key: str):
        """动态切换工位并同步更新样本源与标靶地图"""
        self.current_workspace_id = workspace_key
        self.workspace_mgr.set_current_workspace(workspace_key)

        ws = self.workspace_mgr.get_workspace_by_id(workspace_key)
        self.sample_dir = ws.prod_raw_images_dir if ws else ""

        self._init_localizer()
        self.rescan(auto_load=True)

        if self.tag_localizer:
            tag_cnt = len(getattr(self.tag_localizer, "tag_poses", {}))
            self.set_toast(f"已装载【{self.current_workspace_name}】地图 ({tag_cnt}个标靶)")
        else:
            self.set_toast(f"【{self.current_workspace_name}】未平差 tags_map.yaml，降级估算！")

    def rescan(self, auto_load: bool = False):
        """重新扫描样本目录 (选中样本自动立即触发识别定位)"""
        self.samples = scan_samples(self.sample_dir)
        self.sel_idx = -1
        self.targets, self.vis_img = [], None
        self.sel_target, self.error = 0, ""
        if auto_load and self.samples:
            target_idx = 0
            if self._persisted_sample_name:
                for i, smp in enumerate(self.samples):
                    if smp["name"] == self._persisted_sample_name:
                        target_idx = i
                        break
            self._select_sample(target_idx, analyze_now=True)

    def _keep_selection_visible(self, idx: int):
        """键盘切换样本时保持选中项在列表中可见"""
        m = AsparagusPoseStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        _, _, _, bottom = AsparagusPoseStudioRenderer.compute_panels(
            self.win_mgr.canvas_w, self.win_mgr.canvas_h, m
        )[0]
        content_h = bottom - m["header_h"] - int(30 * m["s"])
        visible = max(1, content_h // (m["row_h"] + int(4 * m["s"])))
        if idx < self.scroll_off:
            self.scroll_off = idx
        elif idx >= self.scroll_off + visible:
            self.scroll_off = idx - visible + 1

    def _get_scaled_intrinsics(self, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
        """根据实际图像分辨率按比例调整相机内参"""
        intr = self.sys_cfg["intrinsics"]
        if intr:
            fx, fy, cx, cy, cfg_w, cfg_h = intr
            if cfg_w > 0 and cfg_h > 0 and (cfg_w != img_w or cfg_h != img_h):
                fx, cx = fx * img_w / cfg_w, cx * img_w / cfg_w
                fy, cy = fy * img_h / cfg_h, cy * img_h / cfg_h
        else:
            fx, fy, cx, cy = 909.12, 907.46, 647.46, 377.51
        return fx, fy, cx, cy

    def _init_pipeline(self):
        """初始化选中的算法流水线 (并回放上次持久化的滑条参数)"""
        fx, fy, cx, cy = self._get_scaled_intrinsics(1920, 1080)
        self.pipeline = PipelineRegistry.create(self.pipeline_key, fx=fx, fy=fy, cx=cx, cy=cy)
        if self.pipeline:
            saved = self._persisted_state.get("pipeline_params", {}).get(self.pipeline_key, {})
            for attr, val in saved.items():
                if hasattr(self.pipeline, attr):
                    try:
                        setattr(self.pipeline, attr, type(getattr(self.pipeline, attr))(val))
                    except (TypeError, ValueError):
                        pass
            steps = self.pipeline.get_steps()
            if steps:
                self.active_step_key = steps[-1].key

    def switch_pipeline(self, pipeline_key: str):
        """切换算法路线并重新触发分析"""
        if pipeline_key == self.pipeline_key and self.pipeline is not None:
            return
        self.pipeline_key = pipeline_key
        self._init_pipeline()
        self._save_persisted_state()
        self.set_toast(f"已切换算法路线: 【{self.current_pipeline_name}】")
        if 0 <= self.sel_idx < len(self.samples):
            self.run_analyze()

    def _build_analyzer(self, img_w: int, img_h: int) -> AsparagusAnalyzer:
        """构建 AsparagusAnalyzer 实例"""
        fx, fy, cx, cy = self._get_scaled_intrinsics(img_w, img_h)
        analyzer = AsparagusAnalyzer(fx=fx, fy=fy, cx=cx, cy=cy)
        analyzer.set_tag_localizer(self.tag_localizer)
        analyzer.set_hand_eye_matrix(self.sys_cfg["t_cam_to_scara"])
        return analyzer

    def _select_sample(self, idx: int, analyze_now: bool = False):
        """选中样本照片并重置视口"""
        if not (0 <= idx < len(self.samples)):
            return
        self.sel_idx = idx
        self._keep_selection_visible(idx)
        self.targets, self.vis_img = [], None
        self.sel_target, self.error = 0, ""
        self.viewport.reset()
        sample = self.samples[idx]

        self._persisted_sample_name = sample["name"]
        self._save_persisted_state()

        color = cv2.imread(sample["png"])
        if color is None:
            self.error, self.mode = "彩色图读取失败", "2d"
            return
        self.mode = "3d" if sample.get("depth") else "2d"
        self.vis_img = color.copy()

        if analyze_now:
            self.run_analyze()

    def run_analyze(self):
        """对当前样本执行完整的识别定位解算"""
        if not (0 <= self.sel_idx < len(self.samples)):
            self.set_toast("请先在左侧列表中选择一张样本照片")
            return

        sample = self.samples[self.sel_idx]
        color = cv2.imread(sample["png"])
        if color is None:
            self.error = "彩色图读取失败"
            self.set_toast(self.error, True)
            return

        depth = None
        if sample.get("depth"):
            try:
                depth = np.load(sample["depth"], allow_pickle=False)
                if depth.shape[:2] != color.shape[:2]:
                    depth = cv2.resize(depth, (color.shape[1], color.shape[0]),
                                       interpolation=cv2.INTER_NEAREST)
            except Exception as exc:
                log.warning("深度加载失败 (%s): %s", sample["depth"], exc)
                depth = None
        self.mode = "3d" if depth is not None else "2d"

        try:
            fx, fy, cx, cy = self._get_scaled_intrinsics(color.shape[1], color.shape[0])
            if self.pipeline:
                self.pipeline.update_intrinsics(fx, fy, cx, cy)
            else:
                self._init_pipeline()

            dummy_analyzer = self._build_analyzer(color.shape[1], color.shape[0])
            frame_transform, frame_calib_source = dummy_analyzer._resolve_calibration(color)
            plane_coeff = dummy_analyzer.fit_table_plane(depth) if depth is not None else None

            self.pipeline_result = self.pipeline.run(
                color_bgr=color,
                depth_mm=depth,
                plane_coeff=plane_coeff,
                frame_transform=frame_transform,
                frame_calib_source=frame_calib_source,
                nominal_z_mm=float(plane_coeff[2]) if (plane_coeff is not None and abs(plane_coeff[2]) > 300) else 640.0
            )
            self.targets = self.pipeline_result.targets
            self.sel_target = 0

            steps = self.pipeline.get_steps()
            step_keys = [s.key for s in steps]
            if (self.active_step_key not in step_keys
                    and self.active_step_key not in ("stage0_original", "stage1b_edge", "stage2b_morph")
                    and step_keys):
                self.active_step_key = step_keys[-1]

            self._params_dirty = False
            self._apply_active_step()
        except Exception as exc:
            log.exception("流水线执行异常")
            self.error = f"执行异常: {exc}"
            self.set_toast(self.error, True)
            return

        if self.targets:
            self.set_toast(f"解算完成: 检出 {len(self.targets)} 个目标 ({self.pipeline_result.elapsed_ms:.0f}ms)", duration=2.2)
        else:
            self.set_toast(f"未检出符合规格目标 ({self.pipeline_result.elapsed_ms:.0f}ms)", duration=2.2)

    # ------------------------------ 步骤滑条 (双滑块调参) ------------------------------
    def _hit_slider(self, x: int, y: int) -> bool:
        """检测滑条命中并开始拖拽 (双滑块选最近端: 左=下限 右=上限; 单滑块固定 handle=0)"""
        for rect, spec in reversed(self._slider_bars):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 - 6 <= y <= y2 + 6:
                if "attr" in spec:
                    handle = 0
                else:
                    lo = int(getattr(self.pipeline, spec["attr_low"]))
                    hi = int(getattr(self.pipeline, spec["attr_high"]))
                    span = max(1, x2 - x1)
                    val = spec["vmin"] + (x - x1) * (spec["vmax"] - spec["vmin"]) / span
                    handle = 0 if abs(val - lo) <= abs(val - hi) else 1
                self._drag_slider = (rect, spec, handle)
                self._move_slider(x)
                return True
        return False

    def _move_slider(self, x: int):
        """拖拽更新滑块值并实时写入流水线属性, 同时刷新当前调参步骤的预览"""
        if self._drag_slider is None:
            return
        (x1, _, x2, _), spec, handle = self._drag_slider
        val = int(round(spec["vmin"] + (x - x1) * (spec["vmax"] - spec["vmin"]) / max(1, x2 - x1)))
        val = max(spec["vmin"], min(spec["vmax"], val))
        if "attr" in spec:
            attr = spec["attr"]
        elif handle == 0:
            val = min(val, int(getattr(self.pipeline, spec["attr_high"])))
            attr = spec["attr_low"]
        else:
            val = max(val, int(getattr(self.pipeline, spec["attr_low"])))
            attr = spec["attr_high"]
        if int(getattr(self.pipeline, attr)) != val:
            setattr(self.pipeline, attr, val)
            self._params_dirty = True
            self._apply_active_step()   # 声明了实时预览器的调参步骤即时刷新预览

    def _select_step(self, step_key: str):
        """切换算法流水线的中间步骤视图"""
        spec_map = getattr(self.pipeline, "STEP_SLIDERS", {}) if self.pipeline else {}
        was_tuning = (self.active_step_key in spec_map and self._params_dirty)
        self.active_step_key = step_key
        if self.pipeline_result is None and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()
            return
        if was_tuning and (0 <= self.sel_idx < len(self.samples)):
            self.run_analyze()      # 滑条参数已变更, 离开调参步骤时重新解算生效
            return
        self._apply_active_step()

    def _apply_active_step(self):
        """根据当前激活步骤更新视口显示的特征图"""
        # 步骤 0 原始图: 虚拟步骤, 始终显示未经任何处理的原始输入图像
        if self.active_step_key == "stage0_original":
            if 0 <= self.sel_idx < len(self.samples):
                color = cv2.imread(self.samples[self.sel_idx]["png"])
                if color is not None:
                    self.vis_img = color.copy()
            return
        # 步骤 1B 边缘提取: 虚拟步骤, 对原始图像做 Canny 边缘检测预览 (与 1A HSV 分割并列分支)
        if self.active_step_key == "stage1b_edge":
            if 0 <= self.sel_idx < len(self.samples):
                color = cv2.imread(self.samples[self.sel_idx]["png"])
                if color is not None:
                    if self.pipeline is not None and hasattr(self.pipeline, "_stage1b_edges"):
                        edges = self.pipeline._stage1b_edges(color)
                    else:
                        edges = cv2.Canny(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 60, 160)
                    vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis, (12, 12), (680, 48), (20, 20, 20), -1)
                    cv2.rectangle(vis, (12, 12), (680, 48), (200, 200, 255), 2)
                    put_text(vis, "STAGE 1B: CANNY EDGE EXTRACTION", (22, 36),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 255), 2)
                    self.vis_img = vis
            return
        # 步骤 2B 腐蚀与膨胀 (B 系虚拟步骤): 对 1B Canny 边缘图做闭运算清理 (为步骤 3 融合供源)
        if self.active_step_key == "stage2b_morph":
            if 0 <= self.sel_idx < len(self.samples):
                color = cv2.imread(self.samples[self.sel_idx]["png"])
                if color is not None:
                    if self.pipeline is not None and hasattr(self.pipeline, "_stage1b_edges"):
                        bridged = self.pipeline._stage2b_edge_morph(self.pipeline._stage1b_edges(color))
                    else:
                        edges = cv2.Canny(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), 60, 160)
                        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        bridged = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3)
                    vis = cv2.cvtColor(bridged, cv2.COLOR_GRAY2BGR)
                    cv2.rectangle(vis, (12, 12), (680, 48), (20, 20, 20), -1)
                    cv2.rectangle(vis, (12, 12), (680, 48), (200, 200, 255), 2)
                    put_text(vis, "STAGE 2B: EDGE MORPH CLOSE | Gaps Bridged", (22, 36),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 255), 2)
                    self.vis_img = vis
            return
        if self.pipeline_result and self.active_step_key in self.pipeline_result.step_snapshots:
            # 流水线为该步骤声明了 preview_<stage_key> 实时预览器时, 以当前参数即时渲染
            preview = getattr(self.pipeline, f"preview_{self.active_step_key}", None) if self.pipeline else None
            if preview is not None and 0 <= self.sel_idx < len(self.samples):
                color = cv2.imread(self.samples[self.sel_idx]["png"])
                if color is not None:
                    self.vis_img = preview(color)
                    return
            img = self.pipeline_result.step_snapshots[self.active_step_key]
            if img is not None:
                self.vis_img = img
                step_obj = next((s for s in self.pipeline.get_steps() if s.key == self.active_step_key), None)
                if step_obj:
                    self.set_toast(f"视口显示: {step_obj.name} ({step_obj.description})", duration=1.8)
                return
        if 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                self.vis_img = color.copy()

    def _select_target(self, t_idx: int):
        """选择目标芦笋并联动 G-code"""
        if not (0 <= t_idx < len(self.targets)):
            return
        self.sel_target = t_idx
        if 0 <= self.sel_idx < len(self.samples):
            color = cv2.imread(self.samples[self.sel_idx]["png"])
            if color is not None:
                analyzer = self._build_analyzer(color.shape[1], color.shape[0])
                self.vis_img = analyzer.draw_detections(color, self.targets, sel_target_idx=t_idx)
                if self.pipeline_result:
                    self.pipeline_result.step_snapshots[self.active_step_key] = self.vis_img

    # ------------------------------ 交互与事件 ------------------------------
    def set_toast(self, msg: str, sticky: bool = False, duration: float = 2.2):
        super().set_toast(msg, duration=(3600.0 if sticky else duration))

    @property
    def _toast_msg(self) -> str:
        return self._toast

    @_toast_msg.setter
    def _toast_msg(self, msg: Optional[str]):
        self._toast = msg or ""

    def hit_test(self, x: int, y: int) -> Optional[Tuple[str, Any]]:
        for rect, action in reversed(self._buttons):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                return action
        return None

    def _on_button(self, label: str):
        if label.startswith("识别定位"):
            self.run_analyze()
        elif label.startswith("退出"):
            self.stop()

    def on_mouse_wheel(self, delta: int, flags: int):
        x, y = self.mouse_x, self.mouse_y
        m = AsparagusPoseStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        list_p, img_p, _ = AsparagusPoseStudioRenderer.compute_panels(
            self.win_mgr.canvas_w, self.win_mgr.canvas_h, m
        )
        wheel_up = (flags > 0)
        if list_p[0] <= x <= list_p[2]:
            if wheel_up:
                self.scroll_off = max(0, self.scroll_off - 2)
            else:
                self.scroll_off += 2
        elif img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
            vx, vy, vw, vh = img_p[0] + 2, img_p[1] + 2, img_p[2] - img_p[0] - 4, img_p[3] - img_p[1] - 4
            self.viewport.zoom_at(x, y, wheel_up, (vx, vy, vw, vh))

    def on_mouse_down(self, x: int, y: int, button: str):
        m = AsparagusPoseStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        _, img_p, _ = AsparagusPoseStudioRenderer.compute_panels(
            self.win_mgr.canvas_w, self.win_mgr.canvas_h, m
        )
        if button in ("right", "middle"):
            if img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
                self.viewport.start_pan(x, y)
                return
        elif button == "left":
            self.on_click(x, y)

    def on_mouse_up(self, x: int, y: int, button: str):
        if button in ("right", "middle"):
            if self.viewport.is_panning:
                self.viewport.end_pan()
        elif button == "left":
            if self._drag_slider is not None:
                self._drag_slider = None      # 结束拖拽 (参数已实时写入, 离开调参步骤时重解算)
                self._save_persisted_state()  # 滑条参数即时持久化, 重启后自动回放

    def on_mouse_move(self, x: int, y: int):
        self.mouse_pos = (x, y)
        # 步骤滑条拖拽中: 实时更新滑块值并刷新预览
        if self._drag_slider is not None:
            self._move_slider(x)
            return
        if self.viewport.update_pan(x, y):
            return

    def on_double_click(self, x: int, y: int):
        m = AsparagusPoseStudioRenderer.compute_metrics(self.win_mgr.canvas_w)
        _, img_p, _ = AsparagusPoseStudioRenderer.compute_panels(
            self.win_mgr.canvas_w, self.win_mgr.canvas_h, m
        )
        if img_p[0] <= x <= img_p[2] and img_p[1] <= y <= img_p[3]:
            self.viewport.reset()
            self.set_toast("视口已重置为适应窗口 (1.0x)")

    def on_click(self, x: int, y: int):
        # 步骤滑条: 命中即开始拖拽最近滑块
        if self._hit_slider(x, y):
            return
        if self.active_dropdown and self._dd_items:
            for rect, key in self._dd_items:
                if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                    dd_type = self.active_dropdown
                    self.active_dropdown = None
                    if dd_type == "WORKSPACE_DROPDOWN":
                        self.switch_workspace(key)
                    elif dd_type == "PIPELINE_DROPDOWN":
                        self.switch_pipeline(key)
                    return
            self.active_dropdown = None

        hit = self.hit_test(x, y)
        if hit:
            act_type, act_val = hit
            if act_type == "toggle_dd":
                self.active_dropdown = None if self.active_dropdown == act_val else act_val
            elif act_type == "btn":
                self._on_button(act_val)
            elif act_type == "set_step":
                self._select_step(act_val)
            return

        for rect, idx in self._sample_rows:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                self._select_sample(idx, analyze_now=True)
                return

        for rect, t_idx in self._result_rows:
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                self._select_target(t_idx)
                return

    def on_key(self, raw_key: int) -> bool:
        key = chr(raw_key & 0xFF).lower() if (raw_key & 0xFF) < 128 else ""
        if key in ("x", "\x1b"):
            self.stop()
            return True
        elif key == " " or raw_key == 32 or raw_key in (10, 13):
            self.run_analyze()
            return True
        elif raw_key in (2490368, 65362, 38):      # 上方向键
            if self.sel_idx > 0:
                self._select_sample(self.sel_idx - 1, analyze_now=True)
            return True
        elif raw_key in (2621440, 65364, 40):      # 下方向键
            if self.sel_idx < len(self.samples) - 1:
                self._select_sample(self.sel_idx + 1, analyze_now=True)
            return True
        return False

    def setup(self):
        log.info("芦笋位姿工作室 GUI 已启动: %s (%d 个样本)", self.sample_dir, len(self.samples))

    def cleanup(self):
        log.info("芦笋位姿工作室 GUI 已安全退出")

    # ------------------------------ 渲染接口 ------------------------------
    def render(self) -> np.ndarray:
        """重绘整个画布并更新交互命中区域"""
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        canvas = np.full((H, W, 3), (16, 18, 22), dtype=np.uint8)
        self._buttons, self._sample_rows, self._result_rows, self._dd_items, self._slider_bars = (
            AsparagusPoseStudioRenderer.render_scene(canvas, self)
        )
        return canvas


def main():
    parser = argparse.ArgumentParser(description="芦笋位姿工作室 GUI (Asparagus Pose Studio)")
    parser.add_argument("--dir", type=str, default=None,
                        help="样本目录 (默认当前工位 production/raw_images/, 彩色 png + 可选对齐深度 npy)")
    args = parser.parse_args()
    app = AsparagusPoseStudioApp(sample_dir=args.dir)
    app.run()


if __name__ == "__main__":
    main()
