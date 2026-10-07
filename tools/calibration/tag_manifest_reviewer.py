#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 观测样本轻量级 OpenCV 交互审核画板 (Tag Manifest Reviewer)
==================================================================
核心特性：
  1. 全 GUI 视觉交互操作栏 (Toolbar)：
     - 底部常驻美观的 GUI 按钮组，鼠标直接点击即可完成全部操作：
       [< 上一张 (A)] [下一张 (D) >] [重新识别本图] [全量重扫全部] [复位 (R)] [保存 (S)] [保存退出 (Q)]
  2. 智能增量重新扫描识别 (Rescan & Discover New Tags)：
     - 实时调用最新双路检测算法与 18~29 号 12 枚白名单；
     - 增量发掘新标靶，智能继承保留已有人工剔除/保留标记 (keep)，绝不误抹除工作成果。
  3. 鼠标直接单击标靶翻转状态：
     - 点击任意标靶多边形内部，毫秒级翻转【保留 (绿+3D棱柱)】与【剔除 (红+蒙版+大叉)】。
  4. 毫秒级拓扑连通性安全守门员：
     - 顶栏实时计算并显示共视连通健康度与红绿灯警报，防呆阻断孤立断网。
  5. 键鼠无缝协同：
     - 鼠标点击按钮与键盘快捷键 (A/D/R/S/Q) 双通道支持。
"""

import os
import sys
import time
import copy
import yaml
import argparse
import numpy as np
import cv2
from datetime import datetime
from typing import Optional, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
try:
    from tools.calibration.tag_map_builder import TagMapBuilder
except ImportError:
    TagMapBuilder = None

from src.calibration.solvers.covisibility_graph import CovisibilityGraphAnalyzer

try:
    from tools.window_helper import force_window_focus
except ImportError:
    force_window_focus = None

from src.ui.viewport_manager import (
    ViewportManager, get_safe_screen_size,
    draw_styled_button, draw_segmented_toggle
)

from src.utils.text_rendering import measure_text, put_text
from src.utils.logger import get_logger
from src.workspace.workspace_manager import load_workspace_marker_size_mm

log = get_logger(__name__)


class TagManifestReviewer:
    def __init__(self, 
                 manifest_path: Optional[str] = None,
                 builder: Optional[Any] = None,
                 focus_tag_id: Optional[int] = None,
                 initial_frame: Optional[str] = None):
        """
        初始化交互式审核画板
        :param manifest_path: tag_observations.yaml 文件路径
        :param builder: TagMapBuilder 实例（用于连通性校验与 3D 渲染）
        :param focus_tag_id: 指定靶向排查的标靶 ID (如盲测出现较大误差的 Tag 编号)
        :param initial_frame: 初始跳转定位的图像文件名 (如 'view_0016.png')
        """
        if manifest_path is None:
            try:
                from src.workspace.workspace_manager import WorkspaceManager
                manifest_path = WorkspaceManager().get_current_workspace().calib_manifest_path
            except Exception:
                manifest_path = ""
        self.manifest_path = manifest_path or ""
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"未找到观测清单文件: {self.manifest_path}，请先执行扫描导出！")

        self.builder = builder if builder is not None else TagMapBuilder(marker_size_mm=self._resolve_marker_size_mm())
        self.window_name = "AprilTag Observations Reviewer (GUI Button Toolbar & Interactive Review)"

        # 自适应屏幕工作区与分层视口管理器
        if get_safe_screen_size:
            self.win_w, self.win_h = get_safe_screen_size(preferred_w=1280, preferred_h=720)
        else:
            self.win_w, self.win_h = 1280, 720

        if ViewportManager:
            self.viewport = ViewportManager(win_w=self.win_w, win_h=self.win_h, top_bar_h=54, bottom_bar_h=52)
        else:
            self.viewport = None

        # 加载清单数据
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.raw_manifest = yaml.safe_load(f) or {}

        # 核心防呆自愈：启动时自动扫描磁盘图片目录，增量发掘并录入新采图 (如 view_0016~view_0022)
        self.sync_with_disk(auto_save=True)

        # 组织有序的图像列表
        self.image_keys = sorted(list(self.raw_manifest.get("images", {}).keys()))
        if not self.image_keys:
            raise ValueError("观测清单中未包含任何图像观测数据！")

        # 靶向排查与多帧定向跳跃状态
        self.focus_tag_id = int(focus_tag_id) if focus_tag_id is not None else None
        self.target_hit_frames = []
        self.focus_mode = False
        self.focus_idx = 0

        if self.focus_tag_id is not None:
            for k in self.image_keys:
                img_data = self.raw_manifest.get("images", {}).get(k, {})
                for obs in img_data.get("observations", []):
                    if int(obs.get("tag_id", -1)) == self.focus_tag_id:
                        self.target_hit_frames.append(k)
                        break
            if self.target_hit_frames:
                self.focus_mode = True
                self.focus_idx = 0

        # 备份初始状态以供单帧复位 (R 键)
        self.initial_states = {}
        # 初始化整帧使能状态 (支持整张图片一键临时剔除/恢复)
        self.frame_enabled_map = {}
        for k in self.image_keys:
            self.frame_enabled_map[k] = self.raw_manifest["images"][k].get("enabled", True)
            self.initial_states[k] = [
                copy.deepcopy(obs.get("keep", True)) 
                for obs in self.raw_manifest["images"][k].get("observations", [])
            ]

        if initial_frame is not None and initial_frame in self.image_keys:
            self.current_idx = self.image_keys.index(initial_frame)
            if self.focus_mode and initial_frame in self.target_hit_frames:
                self.focus_idx = self.target_hit_frames.index(initial_frame)
        elif self.focus_mode and self.target_hit_frames:
            self.current_idx = self.image_keys.index(self.target_hit_frames[0])
        else:
            self.current_idx = 0

        self.has_unsaved_changes = False
        self.has_modified_manifest = False
        self.display_img = None
        self.last_covis_report = {"is_valid": True, "message": "", "all_tags": []}

        # GUI 按钮热区列表: [(btn_action_id, (x1, y1, x2, y2), label, is_enabled)]
        self.gui_action_buttons = []
        self.mouse_hover_pos = (-1, -1)

        # 右键上下文悬浮菜单 (Context Menu)
        self.context_menu = {
            "visible": False,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
            "tag_id": None,
            "obs_idx": -1,
            "items": [],  # [(action_id, label, (x1, y1, x2, y2), color)]
            "bounds": (0, 0, 0, 0)
        }

        # Toast 通知
        self.toast_msg = ""
        self.toast_time = 0.0

        # 首次计算拓扑健康度
        self.update_topology()

    def _resolve_marker_size_mm(self) -> float:
        """标靶物理边长解析 (FR-9.x: 唯一权威来源 = 当前工位 tag_whitelist.yaml.tag_default_size_mm).
        缺失时抛出明确错误, 不允许任何硬编码兜底.
        """
        try:
            from src.workspace.workspace_manager import WorkspaceManager
            ws = WorkspaceManager().get_current_workspace()
            loaded = load_workspace_marker_size_mm(ws.workspace_dir) if ws else None
        except Exception as e:
            loaded = None
            log.warning(f"[REVIEWER] 解析工位标靶边长失败: {e}")
        if loaded is None:
            raise ValueError(
                "未在当前工位 tag_whitelist.yaml.tag_default_size_mm 找到标靶边长, "
                "TagManifestReviewer 拒绝启动. 请在 tag_whitelist.yaml 录入 tag_default_size_mm 后重试."
            )
        return loaded

    @property
    def current_image_key(self) -> str:
        """获取当前正在浏览的图像名"""
        if self.focus_mode and self.target_hit_frames:
            return self.target_hit_frames[self.focus_idx]
        return self.image_keys[self.current_idx]

    def prev_frame(self):
        """翻至上一张 (支持聚焦子集与全量模式)"""
        if self.focus_mode and self.target_hit_frames:
            if self.focus_idx > 0:
                self.focus_idx -= 1
                self.current_idx = self.image_keys.index(self.target_hit_frames[self.focus_idx])
                self.render_current_frame()
            else:
                self.set_toast(f"已是 Tag #{self.focus_tag_id} 命中帧的首张")
        else:
            if self.current_idx > 0:
                self.current_idx -= 1
                self.render_current_frame()
            else:
                self.set_toast("已经是第一张图片")

    def next_frame(self):
        """翻至下一张 (支持聚焦子集与全量模式)"""
        if self.focus_mode and self.target_hit_frames:
            if self.focus_idx < len(self.target_hit_frames) - 1:
                self.focus_idx += 1
                self.current_idx = self.image_keys.index(self.target_hit_frames[self.focus_idx])
                self.render_current_frame()
            else:
                self.set_toast(f"已是 Tag #{self.focus_tag_id} 命中帧的末张")
        else:
            if self.current_idx < len(self.image_keys) - 1:
                self.current_idx += 1
                self.render_current_frame()
            else:
                self.set_toast("已经是最后一张图片")

    def toggle_focus_mode(self):
        """在【靶向聚焦命中帧】与【全量浏览模式】之间无缝切换"""
        if not self.target_hit_frames:
            self.set_toast(f"当前无针对 Tag #{self.focus_tag_id} 的命中帧")
            return
        self.focus_mode = not self.focus_mode
        if self.focus_mode:
            curr_k = self.image_keys[self.current_idx]
            if curr_k in self.target_hit_frames:
                self.focus_idx = self.target_hit_frames.index(curr_k)
            else:
                self.focus_idx = 0
                self.current_idx = self.image_keys.index(self.target_hit_frames[0])
            self.set_toast(f"已进入【靶向排查模式: Tag #{self.focus_tag_id}】({len(self.target_hit_frames)} 帧)")
        else:
            self.set_toast(f"已切换为【全量浏览模式】(共 {len(self.image_keys)} 帧)")
        self.render_current_frame()

    def save_and_verify(self):
        """保存修改并退出审核画板 (BA 求解交由 Offline Studio 统一执行)"""
        self.save_changes()
        self.is_running = False
        log.info(f"[SAVED] 审核画板已保存修改并退出，可在 Offline Studio 中一键求解 BA 平差。")

    def sync_with_disk(self, auto_save: bool = True) -> int:
        """
        自动检查磁盘上的图片文件集合并与清单进行增量同步：
        1. 扫描所在目录下的所有原始图片（如 view_*.png，排除 visualized/ 标注图）；
        2. 若发现新照片未在清单中登记，自动运行双路检测提取标靶观测并生成标注图；
        3. 若清单中包含已在磁盘上删除的文件，自动予以清理；
        4. 严格保留现有已审核图片中用户的人工标记 (keep/note)，绝不抹除工作成果；
        5. 同步更新清单 summary 并写回 yaml 文件。
        :return: 新增的图片数量
        """
        import glob
        manifest_dir = os.path.dirname(os.path.abspath(self.manifest_path))
        vis_dir = os.path.join(manifest_dir, "visualized")
        os.makedirs(vis_dir, exist_ok=True)

        disk_files = sorted(glob.glob(os.path.join(manifest_dir, "view_*.png")))
        if not disk_files:
            disk_files = sorted([
                p for p in glob.glob(os.path.join(manifest_dir, "*.png"))
                if not p.endswith("_annotated.png") and not p.endswith("_quiver.png")
            ])

        images_dict = self.raw_manifest.setdefault("images", {})
        existing_keys = set(images_dict.keys())
        disk_map = {os.path.basename(p): p for p in disk_files}

        added_images = []
        for base_name, full_path in disk_map.items():
            if base_name not in existing_keys:
                added_images.append((base_name, full_path))

        if added_images:
            log.info(f"[AUTO-SYNC] 检测到磁盘新增 {len(added_images)} 张采图，正在自动执行增量识别与录入...")
            for base_name, full_path in added_images:
                raw_img = cv2.imread(full_path)
                if raw_img is None:
                    continue
                detected = self.builder.tag_detector.detect_tags(raw_img, refine=True)
                obs_list = []
                for tid in sorted(detected.keys()):
                    corners = detected[tid]
                    metrics = self.builder.compute_tag_metrics(corners)
                    obs_entry = {
                        "tag_id": int(tid),
                        "keep": True,
                        "cell_size_px": metrics["cell_size_px"],
                        "center_px": metrics["center_px"],
                        "area_px": metrics["area_px"],
                        "note": "采图新增自动同步检出",
                        "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                    }
                    obs_list.append(obs_entry)

                # 生成图示化标注
                annotated_name = os.path.splitext(base_name)[0] + "_annotated.png"
                annotated_path = os.path.join(vis_dir, annotated_name)
                ann_img = self.builder.render_annotated_frame(raw_img, detected)
                cv2.imwrite(annotated_path, ann_img)

                rel_img_path = os.path.relpath(full_path, PROJECT_ROOT).replace("\\", "/")
                rel_ann_path = os.path.relpath(annotated_path, PROJECT_ROOT).replace("\\", "/")

                images_dict[base_name] = {
                    "file_name": base_name,
                    "image_path": rel_img_path,
                    "annotated_path": rel_ann_path,
                    "detected_count": len(obs_list),
                    "observations": obs_list
                }
                has_t0 = 0 in detected
                log.info(f"  + [{base_name}] 检出 {len(obs_list)} 个标靶 (Tag 0: {'√ 已捕获' if has_t0 else '未出现'})")

        if added_images:
            self.image_keys = sorted(list(images_dict.keys()))
            if auto_save:
                self.save_changes()
                log.info(f"[AUTO-SYNC] 清单与磁盘已同步：当前共计 {len(self.image_keys)} 张图像，已保存至 {self.manifest_path}")

        return len(added_images)

    def set_toast(self, msg: str):
        self.toast_msg = msg
        self.toast_time = time.time()

    def update_topology(self):
        """实时重算当前全部保留观测下的共视连通性状态"""
        frame_detections = []
        for k in self.image_keys:
            # 若整帧被临时剔除，则直接旁路，不参与共视连通性图构建
            if not self.frame_enabled_map.get(k, True):
                continue
            tags_in_frame = {}
            for obs in self.raw_manifest["images"][k].get("observations", []):
                if obs.get("keep", True):
                    tid = int(obs["tag_id"])
                    corners = np.array(obs["corners"], dtype=np.float64)
                    tags_in_frame[tid] = corners
            if len(tags_in_frame) >= 2:
                frame_detections.append(tags_in_frame)

        self.last_covis_report = CovisibilityGraphAnalyzer.analyze(frame_detections)

    def toggle_current_frame_enabled(self):
        """一键临时剔除或启用当前整张图片（整帧旁路/使能切换）"""
        curr_key = self.image_keys[self.current_idx]
        current_state = self.frame_enabled_map.get(curr_key, True)
        new_state = not current_state
        self.frame_enabled_map[curr_key] = new_state
        self.raw_manifest["images"][curr_key]["enabled"] = new_state
        self.has_unsaved_changes = True
        self.save_changes(quiet=True)

        self.update_topology()
        self.render_current_frame()

        if not new_state:
            msg = f"已临时剔除本帧全部标靶 (按 X 恢复)"
        else:
            msg = f"已恢复本帧所有有效标靶"
        self.set_toast(msg)
        log.info(f"[TOGGLE FRAME] [{curr_key}] {'[已临时剔除]' if not new_state else '[已恢复启用]'}")

    def rescan_current_frame(self):
        """重新使用最新双路检测器与白名单扫描识别当前图像，增量补充新 Tag"""
        curr_key = self.image_keys[self.current_idx]
        img_info = self.raw_manifest["images"][curr_key]
        raw_path = img_info.get("image_path", "")
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(PROJECT_ROOT, raw_path)

        raw_img = cv2.imread(raw_path)
        if raw_img is None:
            self.set_toast(f"无法读取图片: {raw_path}")
            return

        # 运行最新检测器
        detected = self.builder.tag_detector.detect_tags(raw_img, refine=True)
        existing_obs = img_info.get("observations", [])
        existing_map = {int(obs["tag_id"]): obs for obs in existing_obs}

        added_count = 0
        updated_obs = []

        for tid in sorted(detected.keys()):
            corners = detected[tid]
            metrics = self.builder.compute_tag_metrics(corners)
            if tid in existing_map:
                # 继承已有人工标记，仅更新坐标
                old_entry = existing_map[tid]
                old_entry["corners"] = [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                old_entry["cell_size_px"] = metrics["cell_size_px"]
                old_entry["center_px"] = metrics["center_px"]
                old_entry["area_px"] = metrics["area_px"]
                updated_obs.append(old_entry)
            else:
                # 新发现的标靶！默认加入
                new_entry = {
                    "tag_id": int(tid),
                    "keep": True,
                    "cell_size_px": metrics["cell_size_px"],
                    "center_px": metrics["center_px"],
                    "area_px": metrics["area_px"],
                    "note": "全新重新扫描识别检出",
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                }
                updated_obs.append(new_entry)
                added_count += 1

        img_info["observations"] = updated_obs
        img_info["detected_count"] = len(updated_obs)
        self.has_unsaved_changes = True
        self.save_changes(quiet=True)
        self.update_topology()
        self.render_current_frame()

        if added_count > 0:
            msg = f"本图识别完成: 共 {len(updated_obs)} 个标靶 (+新增 {added_count} 个)"
        else:
            msg = f"本图识别完成: 检出 {len(updated_obs)} 个标靶 (已是最新)"
        self.set_toast(msg)
        log.info(f"[RESCAN] [{curr_key}] {msg}")

    def rescan_all_frames(self):
        """全量重新扫描数据集中所有图像并更新清单"""
        # 1. 先同步磁盘新采图
        self.sync_with_disk(auto_save=False)

        total_found = 0
        total_new = 0

        for k in self.image_keys:
            img_info = self.raw_manifest["images"][k]
            raw_path = img_info.get("image_path", "")
            if not os.path.isabs(raw_path):
                raw_path = os.path.join(PROJECT_ROOT, raw_path)
            raw_img = cv2.imread(raw_path)
            if raw_img is None:
                continue

            detected = self.builder.tag_detector.detect_tags(raw_img, refine=True)
            existing_obs = img_info.get("observations", [])
            existing_map = {int(obs["tag_id"]): obs for obs in existing_obs}

            updated_obs = []
            for tid in sorted(detected.keys()):
                corners = detected[tid]
                metrics = self.builder.compute_tag_metrics(corners)
                if tid in existing_map:
                    old_entry = existing_map[tid]
                    old_entry["corners"] = [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                    old_entry["cell_size_px"] = metrics["cell_size_px"]
                    old_entry["center_px"] = metrics["center_px"]
                    old_entry["area_px"] = metrics["area_px"]
                    updated_obs.append(old_entry)
                else:
                    new_entry = {
                        "tag_id": int(tid),
                        "keep": True,
                        "cell_size_px": metrics["cell_size_px"],
                        "center_px": metrics["center_px"],
                        "area_px": metrics["area_px"],
                        "note": "全量重扫检出",
                        "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                    }
                    updated_obs.append(new_entry)
                    total_new += 1
                total_found += 1

            img_info["observations"] = updated_obs
            img_info["detected_count"] = len(updated_obs)

        self.has_unsaved_changes = True
        self.save_changes(quiet=True)
        self.update_topology()
        self.render_current_frame()
        self.set_toast(f"全量重扫完成: 累计检出 {total_found} 次观测 (+新增 {total_new} 项)")
        log.info(f"[RESCAN ALL] 全量重扫完成: 累计 {total_found} 次观测，新增 {total_new} 项标靶！")

    def refine_single_tag(self, obs_idx: int):
        """对当前帧中的指定标靶进行局部 ROI 重新高精提取与角点重算"""
        curr_key = self.current_image_key
        img_info = self.raw_manifest["images"][curr_key]
        observations = img_info.get("observations", [])
        if obs_idx < 0 or obs_idx >= len(observations):
            return

        target_obs = observations[obs_idx]
        tid = int(target_obs["tag_id"])
        raw_path = img_info.get("image_path", "")
        if not os.path.exists(raw_path):
            raw_path = os.path.join(PROJECT_ROOT, raw_path)
        img = cv2.imread(raw_path)
        if img is None:
            self.set_toast(f"无法读取原图: {raw_path}")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        old_c = np.array(target_obs["corners"], dtype=np.float32)

        # 构造局部 ROI (外扩 35px)
        x_min = max(0, int(np.min(old_c[:, 0])) - 35)
        y_min = max(0, int(np.min(old_c[:, 1])) - 35)
        x_max = min(w, int(np.max(old_c[:, 0])) + 35)
        y_max = min(h, int(np.max(old_c[:, 1])) + 35)
        roi = gray[y_min:y_max, x_min:x_max]

        # 采用正统轮廓边界直线拟合求交点 (CORNER_REFINE_CONTOUR)，杜绝边缘滑移
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 33
        params.adaptiveThreshWinSizeStep = 2
        params.minMarkerPerimeterRate = 0.01
        det = cv2.aruco.ArucoDetector(self.builder.dictionary, params)

        new_corners = None
        # 1. 原生灰度 ROI
        c_list, ids, _ = det.detectMarkers(roi)
        if ids is not None and tid in ids.flatten():
            i = list(ids.flatten()).index(tid)
            new_corners = c_list[i].reshape(4, 2) + np.array([x_min, y_min])
        else:
            # 2. CLAHE 增强 ROI
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            roi_clahe = clahe.apply(roi)
            c_list, ids, _ = det.detectMarkers(roi_clahe)
            if ids is not None and tid in ids.flatten():
                i = list(ids.flatten()).index(tid)
                new_corners = c_list[i].reshape(4, 2) + np.array([x_min, y_min])

        if new_corners is not None:
            c_list_fmt = [[round(float(pt[0]), 2), round(float(pt[1]), 2)] for pt in new_corners]
            target_obs["corners"] = c_list_fmt
            metrics = self.builder.compute_tag_metrics(new_corners)
            target_obs["cell_size_px"] = metrics["cell_size_px"]
            target_obs["center_px"] = metrics["center_px"]
            target_obs["area_px"] = metrics["area_px"]
            target_obs["note"] = "右键重新计算刷新"
            self.has_unsaved_changes = True
            self.save_changes(quiet=True)
            self.set_toast(f"Tag #{tid} 局部高精重算完成！已更新角点坐标")
            log.info(f"[RE-EXTRACT] [{curr_key}] Tag #{tid} 角点已重新提取并刷新")
        else:
            self.set_toast(f"Tag #{tid} 局部重算未检出，保持现有角点")

        self.update_topology()
        self.render_current_frame()

    def open_context_menu(self, x: int, y: int, tid: int, obs_idx: int):
        """在 (x, y) 坐标处弹出 Tag #tid 的右键快捷菜单"""
        curr_key = self.current_image_key
        observations = self.raw_manifest["images"][curr_key].get("observations", [])
        if obs_idx < 0 or obs_idx >= len(observations):
            return

        obs = observations[obs_idx]
        is_kept = obs.get("keep", True)

        menu_items_def = [
            ("TOGGLE_KEEP", "[-] 剔除该标靶 (设为无效)" if is_kept else "[+] 恢复该标靶 (设为有效)", (0, 80, 255) if is_kept else (0, 220, 100)),
            ("FOCUS_TAG", f"[>] 靶向排查 Tag #{tid} (跳跃浏览命中帧)", (255, 120, 240)),
            ("REFINE_TAG", f"[*] 局部重新计算/精修角点 (Refine)", (0, 230, 255)),
            ("SAVE_AND_BA", "[>] 完成审核并保存退出", (50, 200, 255)),
            ("CLOSE", "[x] 取消 / 关闭菜单", (180, 180, 180))
        ]

        item_h = 36
        menu_w = 340
        header_h = 34
        menu_h = header_h + len(menu_items_def) * item_h + 8

        menu_x = max(10, min(x, self.win_w - menu_w - 15))
        menu_y = max(60, min(y, self.win_h - menu_h - 60))

        items_layout = []
        cur_y = menu_y + header_h + 4
        for act_id, label, col in menu_items_def:
            item_rect = (menu_x + 6, cur_y, menu_x + menu_w - 6, cur_y + item_h - 2)
            items_layout.append((act_id, label, item_rect, col))
            cur_y += item_h

        self.context_menu = {
            "visible": True,
            "x": menu_x,
            "y": menu_y,
            "w": menu_w,
            "h": menu_h,
            "tag_id": tid,
            "obs_idx": obs_idx,
            "items": items_layout,
            "bounds": (menu_x, menu_y, menu_x + menu_w, menu_y + menu_h)
        }
        self.render_current_frame()

    def on_mouse_event(self, event, x, y, flags, param):
        """鼠标交互处理：支持滚轮缩放、中键拖拽平移、左键点击按钮/标靶，右键弹出菜单并执行 (自动进行视口坐标逆变换)"""
        self.mouse_hover_pos = (x, y)

        # 0. 滚轮以鼠标为中心平滑缩放与中键平移拖拽 (通用视口引擎支持)
        if event == cv2.EVENT_MOUSEWHEEL:
            if self.viewport and self.viewport.handle_mouse_wheel(x, y, flags):
                self.render_current_frame()
            return

        if event == cv2.EVENT_MBUTTONDOWN:
            if self.viewport:
                self.viewport.start_pan(x, y)
            return
        elif event == cv2.EVENT_MBUTTONUP:
            if self.viewport:
                self.viewport.end_pan()
            return
        elif event == cv2.EVENT_MBUTTONDBLCLK:
            if self.viewport and self.viewport.reset_zoom():
                self.render_current_frame()
                self.set_toast("视口已复位 (1.0x)")
            return

        # 1. 右键按下：检查命中标靶并弹出上下文菜单
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.viewport:
                img_x, img_y = self.viewport.win_to_img_coords(x, y)
            else:
                img_x, img_y = x, y

            if img_x is not None:
                curr_key = self.current_image_key
                observations = self.raw_manifest["images"][curr_key].get("observations", [])
                hit_idx = -1
                for idx in reversed(range(len(observations))):
                    obs = observations[idx]
                    corners = np.array(obs["corners"], dtype=np.float32)
                    dist = cv2.pointPolygonTest(corners, (float(img_x), float(img_y)), False)
                    if dist >= 0:
                        hit_idx = idx
                        break

                if hit_idx != -1:
                    target_obs = observations[hit_idx]
                    tid = int(target_obs["tag_id"])
                    self.open_context_menu(x, y, tid, hit_idx)
                else:
                    if self.context_menu.get("visible", False):
                        self.context_menu["visible"] = False
                        self.render_current_frame()
            else:
                if self.context_menu.get("visible", False):
                    self.context_menu["visible"] = False
                    self.render_current_frame()
            return

        # 2. 鼠标移动：平移漫游拖拽或菜单高亮
        if event == cv2.EVENT_MOUSEMOVE:
            if self.viewport and self.viewport.is_panning:
                if self.viewport.update_pan(x, y):
                    self.render_current_frame()
                return
            if self.context_menu.get("visible", False):
                self.render_current_frame()
            return

        # 3. 左键按下
        if event == cv2.EVENT_LBUTTONDOWN:
            # 若当前右键菜单处于展开状态，优先响应菜单项点击
            if self.context_menu.get("visible", False):
                clicked_action = None
                for act_id, label, (ix1, iy1, ix2, iy2), col in self.context_menu["items"]:
                    if ix1 <= x <= ix2 and iy1 <= y <= iy2:
                        clicked_action = act_id
                        break

                target_tid = self.context_menu["tag_id"]
                target_obs_idx = self.context_menu["obs_idx"]
                self.context_menu["visible"] = False

                if clicked_action == "TOGGLE_KEEP":
                    curr_key = self.current_image_key
                    observations = self.raw_manifest["images"][curr_key].get("observations", [])
                    if 0 <= target_obs_idx < len(observations):
                        obs = observations[target_obs_idx]
                        obs["keep"] = not obs.get("keep", True)
                        self.has_unsaved_changes = True
                        self.save_changes(quiet=True)
                        status_str = "【保留】" if obs["keep"] else "【剔除】"
                        log.info(f"[MENU] [{curr_key}] Tag #{target_tid} 切换为: {status_str}")
                        self.set_toast(f"Tag #{target_tid} -> {status_str}")
                        self.update_topology()
                        self.render_current_frame()
                    return
                elif clicked_action == "FOCUS_TAG":
                    self.focus_tag_id = target_tid
                    self.target_hit_frames = []
                    for k in self.image_keys:
                        img_data = self.raw_manifest.get("images", {}).get(k, {})
                        for obs in img_data.get("observations", []):
                            if int(obs.get("tag_id", -1)) == self.focus_tag_id:
                                self.target_hit_frames.append(k)
                                break
                    self.focus_mode = True
                    curr_k = self.image_keys[self.current_idx]
                    if curr_k in self.target_hit_frames:
                        self.focus_idx = self.target_hit_frames.index(curr_k)
                    else:
                        self.focus_idx = 0
                        self.current_idx = self.image_keys.index(self.target_hit_frames[0])
                    self.set_toast(f"已锁定【靶向排查: Tag #{target_tid}】({len(self.target_hit_frames)} 帧)")
                    self.render_current_frame()
                    return
                elif clicked_action == "REFINE_TAG":
                    self.refine_single_tag(target_obs_idx)
                    return
                elif clicked_action == "SAVE_AND_BA":
                    self.save_and_verify()
                    return
                elif clicked_action == "CLOSE":
                    self.render_current_frame()
                    return
                else:
                    # 点击在菜单外部，直接关闭菜单
                    self.render_current_frame()
                    return

            # A. 检查是否点击了底部的 GUI 按钮 (在窗口坐标系中判断)
            for btn_id, (bx1, by1, bx2, by2), label, is_enabled in self.gui_action_buttons:
                if bx1 <= x <= bx2 and by1 <= y <= by2 and is_enabled:
                    if btn_id == "PREV":
                        self.prev_frame()
                    elif btn_id == "NEXT":
                        self.next_frame()
                    elif btn_id == "TOGGLE_FRAME":
                        self.toggle_current_frame_enabled()
                    elif btn_id == "TOGGLE_FOCUS":
                        self.toggle_focus_mode()
                    elif btn_id == "RESCAN_CURR":
                        self.rescan_current_frame()
                    elif btn_id == "RESCAN_ALL":
                        self.rescan_all_frames()
                    elif btn_id == "RESET":
                        curr_key = self.current_image_key
                        if curr_key in self.initial_states:
                            init_keeps = self.initial_states[curr_key]
                            observations = self.raw_manifest["images"][curr_key].get("observations", [])
                            for i, keep_val in enumerate(init_keeps):
                                if i < len(observations):
                                    observations[i]["keep"] = keep_val
                            self.has_unsaved_changes = True
                            self.save_changes(quiet=True)
                            self.update_topology()
                            self.render_current_frame()
                            self.set_toast("已恢复为初始加载状态 (已自动存盘)")
                    elif btn_id == "SAVE_AND_VERIFY":
                        self.save_and_verify()
                    elif btn_id == "EXIT":
                        self.is_running = False
                    return

            # B. 检查是否左键直接点击了画面中的标靶多边形内部 (通过视口逆变换映射至原图坐标)
            if self.viewport:
                img_x, img_y = self.viewport.win_to_img_coords(x, y)
            else:
                img_x, img_y = x, y

            if img_x is not None:
                curr_key = self.current_image_key
                observations = self.raw_manifest["images"][curr_key].get("observations", [])
                
                hit_idx = -1
                for idx in reversed(range(len(observations))):
                    obs = observations[idx]
                    corners = np.array(obs["corners"], dtype=np.float32)
                    dist = cv2.pointPolygonTest(corners, (float(img_x), float(img_y)), False)
                    if dist >= 0:
                        hit_idx = idx
                        break

                if hit_idx != -1:
                    target_obs = observations[hit_idx]
                    old_keep = target_obs.get("keep", True)
                    new_keep = not old_keep
                    target_obs["keep"] = new_keep
                    self.has_unsaved_changes = True
                    self.save_changes(quiet=True)

                    status_str = "【保留】" if new_keep else "【剔除】"
                    log.info(f"[CLICK] [{curr_key}] Tag #{target_obs['tag_id']} 状态切换为: {status_str}")
                    self.set_toast(f"Tag #{target_obs['tag_id']} -> {status_str} (已自动存盘)")

                    # 毫秒级重算拓扑
                    self.update_topology()
                    self.render_current_frame()
                self.render_current_frame()


    def render_current_frame(self):
        """绘制当前帧画面，融合标靶双态、顶部状态栏与底部 GUI 按钮栏"""
        curr_key = self.current_image_key
        img_info = self.raw_manifest["images"][curr_key]
        raw_path = img_info.get("image_path", "")

        raw_img = cv2.imread(raw_path)
        if raw_img is None:
            raw_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
            put_text(raw_img, f"Image not found: {raw_path}", (100, 540),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        disp = raw_img.copy()
        h, w = disp.shape[:2]
        observations = img_info.get("observations", [])

        num_kept = 0
        num_excl = 0
        dot_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]

        # 1. 渲染所有标靶
        for obs in observations:
            tid = int(obs["tag_id"])
            keep = obs.get("keep", True)
            corners = np.array(obs["corners"], dtype=np.float32)
            pts_int = corners.astype(np.int32)
            cx = int(np.mean(corners[:, 0]))
            min_y = int(np.min(corners[:, 1]))
            badge_x = max(10, cx - 60)
            badge_y = max(70, min_y - 14)

            cell_size = obs.get("cell_size_px", [0, 0])
            is_focus_target = (self.focus_tag_id is not None and tid == self.focus_tag_id)

            if keep:
                num_kept += 1
                # 绿色边框 (排查目标为高亮品红)
                border_color = (255, 50, 230) if is_focus_target else (0, 255, 0)
                border_thick = 4 if is_focus_target else 2
                cv2.polylines(disp, [pts_int], True, border_color, border_thick, cv2.LINE_AA)
                # 4 顶点彩色圆点
                for pt_idx, pt in enumerate(pts_int):
                    cv2.circle(disp, tuple(pt), 5, dot_colors[pt_idx], -1)

                # 标牌文字
                tag_text = f"Tag {tid}" + (" [ORIGIN]" if tid == 0 else "")
                if is_focus_target:
                    tag_text += " [排查目标]"
                cell_text = f"Cell: {cell_size[0]}x{cell_size[1]}px"
                b_w = 145 if is_focus_target else 116
                b_border = (255, 80, 240) if is_focus_target else (0, 255, 255)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + b_w, badge_y + 8), (20, 20, 20), -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 30), (badge_x + b_w, badge_y + 8), b_border, 1)
                put_text(disp, tag_text, (badge_x, badge_y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)
                put_text(disp, cell_text, (badge_x, badge_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)

                # 3D 正四棱柱 Z 轴 (实心方柱体)
                try:
                    self.builder.render_tag_3d_axes(disp, corners, tid)
                except Exception:
                    pass  # 热路径：每帧渲染 3D 轴失败静默跳过，避免日志刷屏

            else:
                num_excl += 1
                # 红色边框
                cv2.polylines(disp, [pts_int], True, (0, 0, 255), 3, cv2.LINE_AA)
                
                # 半透明红色蒙版
                overlay = disp.copy()
                cv2.fillPoly(overlay, [pts_int], (0, 0, 220))
                cv2.addWeighted(overlay, 0.38, disp, 0.62, 0, disp)

                # 红色对角大叉号
                cv2.line(disp, tuple(pts_int[0]), tuple(pts_int[2]), (0, 0, 255), 3, cv2.LINE_AA)
                cv2.line(disp, tuple(pts_int[1]), tuple(pts_int[3]), (0, 0, 255), 3, cv2.LINE_AA)

                # 红色醒目标牌
                tag_text = f"Tag {tid} [EXCLUDED]"
                if is_focus_target:
                    tag_text += " [排查目标]"
                b_w = 160 if is_focus_target else 140
                cv2.rectangle(disp, (badge_x - 6, badge_y - 25), (badge_x + b_w, badge_y + 5), (15, 15, 15), -1)
                cv2.rectangle(disp, (badge_x - 6, badge_y - 25), (badge_x + b_w, badge_y + 5), (0, 0, 255), 2)
                put_text(disp, tag_text, (badge_x, badge_y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 80, 255), 2, cv2.LINE_AA)

        is_frame_enabled = self.frame_enabled_map.get(curr_key, True)

        # 若整帧被临时剔除，叠加暗色蒙版并在右上角打上醒目大标牌
        if not is_frame_enabled:
            dark_overlay = disp.copy()
            cv2.rectangle(dark_overlay, (0, 0), (w, h), (15, 15, 15), -1)
            cv2.addWeighted(dark_overlay, 0.45, disp, 0.55, 0, disp)

            badge_text = "FRAME EXCLUDED / 本帧已临时剔除 (按 X 键恢复启用)"
            (bw_t, bh_t), _ = measure_text(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
            bx_r = w - bw_t - 25
            by_r = 72
            cv2.rectangle(disp, (bx_r - 12, by_r - 6), (bx_r + bw_t + 12, by_r + bh_t + 10), (0, 0, 160), -1)
            cv2.rectangle(disp, (bx_r - 12, by_r - 6), (bx_r + bw_t + 12, by_r + bh_t + 10), (0, 120, 255), 2)
            put_text(disp, badge_text, (bx_r, by_r + bh_t + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

        # 2. 将原始全分辨率绘制帧通过 ViewportManager 等比贴入视口画布
        canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
        if self.viewport:
            self.viewport.render_viewport(canvas, disp)
            top_bar_h = self.viewport.top_bar_h
            tb_h = self.viewport.bottom_bar_h
        else:
            top_bar_h = 56
            tb_h = 52
            canvas = cv2.resize(disp, (self.win_w, self.win_h))

        w, h = self.win_w, self.win_h

        # 3. 绘制顶部信息条 (在目标窗口物理像素上绘制，绝不缩水发糊)
        bar_overlay = canvas.copy()
        cv2.rectangle(bar_overlay, (0, 0), (w, top_bar_h), (12, 12, 12), -1)
        cv2.addWeighted(bar_overlay, 0.90, canvas, 0.10, 0, canvas)
        cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (70, 70, 70), 1)

        save_indicator = " [已即时存盘]"
        if self.focus_mode and self.target_hit_frames:
            left_info = f"[排查 {self.focus_idx + 1}/{len(self.target_hit_frames)}] {curr_key} | Tag #{self.focus_tag_id} ({self.current_idx + 1}/{len(self.image_keys)} 帧){save_indicator}"
        else:
            left_info = f"[{self.current_idx + 1}/{len(self.image_keys)}] {curr_key} | 观测:{len(observations)} (保留:{num_kept}, 剔除:{num_excl}){save_indicator}"
        put_text(canvas, left_info, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # 拓扑连通性安全指示灯
        if self.last_covis_report["is_valid"]:
            topo_str = f"[TOPOLOGY: PASS 连通网健康 ({len(self.last_covis_report['all_tags'])} Tags)]"
            topo_color = (0, 255, 0)
        else:
            unconn = self.last_covis_report.get("unconnected_tags", [])
            topo_str = f"[ALERT: Tag {unconn} 已失联!]"
            topo_color = (0, 0, 255)
        
        put_text(canvas, topo_str, (w - 380, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.46, topo_color, 2, cv2.LINE_AA)

        # 顶栏第二行：简易提示
        if self.focus_mode and self.target_hit_frames:
            tips_text = f"【靶向排查: Tag #{self.focus_tag_id}】A/D 定向跳跃 | 滚轮缩放/中键拖拽平移(0复位) | 右键标靶弹菜单 | [Tab] 切换全量 | [V] 立即平差验证"
            tips_color = (255, 120, 240)
        else:
            tips_text = "操作: 左键翻转剔除/保留; 滚轮无级缩放/中键平移漫游(0复位); 右键标靶弹菜单; [X] 剔除整帧; [Tab] 聚焦; [V] 平差验证"
            tips_color = (0, 220, 255)
        put_text(canvas, tips_text, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, tips_color, 1, cv2.LINE_AA)

        # 4. 底部全新 GUI 操作按钮栏 (1:1 独立物理像素绘制，永远常驻在窗口视线正下方)
        tb_y1 = h - tb_h
        tb_overlay = canvas.copy()
        cv2.rectangle(tb_overlay, (0, tb_y1), (w, h), (20, 22, 28), -1)
        cv2.addWeighted(tb_overlay, 0.94, canvas, 0.06, 0, canvas)
        cv2.line(canvas, (0, tb_y1), (w, tb_y1), (50, 54, 66), 1)

        self.gui_action_buttons.clear()
        bx = 10
        by1 = tb_y1 + 7
        by2 = tb_y1 + 43

        can_prev = self.focus_idx > 0 if (self.focus_mode and self.target_hit_frames) else self.current_idx > 0
        max_idx = len(self.target_hit_frames) if (self.focus_mode and self.target_hit_frames) else len(self.image_keys)
        cur_idx = self.focus_idx if (self.focus_mode and self.target_hit_frames) else self.current_idx
        can_next = cur_idx < max_idx - 1

        # A. 翻页按钮组
        for act_id, label, bw, enabled in [("PREV", "< 上张 (A)", 95, can_prev), ("NEXT", "下张 (D) >", 95, can_next)]:
            b_rect = (bx, by1, bx + bw, by2)
            draw_styled_button(canvas, b_rect, label, self.mouse_hover_pos, btn_type="normal", is_enabled=enabled)
            self.gui_action_buttons.append((act_id, b_rect, label, enabled))
            bx += bw + 6

        # B. 帧剔除/保留状态开关
        frame_lbl = "[* 本帧保留 (X)]" if is_frame_enabled else "[x 已剔除此帧 (X)]"
        frame_type = "normal" if is_frame_enabled else "danger"
        frame_bw = 135
        f_rect = (bx, by1, bx + frame_bw, by2)
        draw_styled_button(canvas, f_rect, frame_lbl, self.mouse_hover_pos, btn_type=frame_type)
        self.gui_action_buttons.append(("TOGGLE_FRAME", f_rect, frame_lbl, True))
        bx += frame_bw + 6

        # C. 靶向聚焦乒乓开关 (若当前指定了排查 Tag)
        if self.focus_tag_id is not None:
            foc_opts = [("ALL", "全量浏览"), ("FOCUS", f"Tag#{self.focus_tag_id}")]
            foc_idx = 1 if self.focus_mode else 0
            foc_w = 175
            foc_rect = (bx, by1, bx + foc_w, by2)
            sub_rects = draw_segmented_toggle(canvas, foc_rect, foc_opts, foc_idx, self.mouse_hover_pos, shortcut="Tab", active_color=(120, 45, 130))
            for key, srect in sub_rects:
                self.gui_action_buttons.append(("TOGGLE_FOCUS", srect, "TOGGLE_FOCUS", True))
            bx += foc_w + 6

        # D. 操作按钮 (全自动即时存盘，彻底移除冗余的手动保存按钮)
        action_defs = [
            ("RESET", "[复位本帧 (R)]", 105, "warning"),
            ("SAVE_AND_VERIFY", "[立即平差验证 (V)]", 150, "primary"),
        ]
        for act_id, label, bw, btype in action_defs:
            b_rect = (bx, by1, bx + bw, by2)
            draw_styled_button(canvas, b_rect, label, self.mouse_hover_pos, btn_type=btype)
            self.gui_action_buttons.append((act_id, b_rect, label, True))
            bx += bw + 6

        # E. 右侧退出按钮
        exit_bw = 85
        exit_bx = w - exit_bw - 10
        exit_rect = (exit_bx, by1, exit_bx + exit_bw, by2)
        draw_styled_button(canvas, exit_rect, "[退出 (Q)]", self.mouse_hover_pos, btn_type="danger")
        self.gui_action_buttons.append(("EXIT", exit_rect, "退出", True))

        # 5. Toast 临时浮层通知
        if time.time() - self.toast_time < 2.2 and self.toast_msg:
            (tw, th), _ = measure_text(self.toast_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            toast_x = (w - tw) // 2
            toast_y = h - tb_h - 45
            cv2.rectangle(canvas, (toast_x - 14, toast_y - 24), (toast_x + tw + 14, toast_y + 8), (15, 15, 15), -1)
            cv2.rectangle(canvas, (toast_x - 14, toast_y - 24), (toast_x + tw + 14, toast_y + 8), (0, 230, 255), 2)
            put_text(canvas, self.toast_msg, (toast_x, toast_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2, cv2.LINE_AA)

        # 6. 渲染右键上下文悬浮菜单 (若已激活)
        self.render_context_menu(canvas)

        self.display_img = canvas
        cv2.imshow(self.window_name, self.display_img)

    def render_context_menu(self, disp: np.ndarray):
        """在画面上叠加渲染右键上下文菜单"""
        if not self.context_menu.get("visible", False):
            return

        mx, my, mw, mh = self.context_menu["x"], self.context_menu["y"], self.context_menu["w"], self.context_menu["h"]
        tid = self.context_menu["tag_id"]
        mx2, my2 = mx + mw, my + mh

        # 半透明深色磨砂阴影底板
        overlay = disp.copy()
        cv2.rectangle(overlay, (mx, my), (mx2, my2), (16, 18, 24), -1)
        cv2.addWeighted(overlay, 0.94, disp, 0.06, 0, disp)

        # 发光外边框 (深灰色底 + 蓝色科技边框)
        cv2.rectangle(disp, (mx, my), (mx2, my2), (0, 200, 255), 2)

        # 标题栏 (高 34px)
        header_h = 34
        cv2.rectangle(disp, (mx, my), (mx2, my + header_h), (28, 32, 45), -1)
        cv2.line(disp, (mx, my + header_h), (mx2, my + header_h), (70, 85, 120), 1)
        header_txt = f"Tag #{tid} 操作快捷菜单"
        put_text(disp, header_txt, (mx + 14, my + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

        # 绘制各个选项
        hx, hy = self.mouse_hover_pos
        for act_id, label, (ix1, iy1, ix2, iy2), col in self.context_menu["items"]:
            is_hovered = (ix1 <= hx <= ix2 and iy1 <= hy <= iy2)
            
            # 悬停背景条与左侧发光指示条
            if is_hovered:
                cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), (42, 48, 65), -1)
                cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), (0, 220, 255), 1)
                cv2.rectangle(disp, (ix1, iy1), (ix1 + 5, iy2), (0, 230, 255), -1)
            else:
                cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), (22, 25, 35), -1)
                cv2.rectangle(disp, (ix1, iy1), (ix2, iy2), (40, 45, 60), 1)

            txt_color = (255, 255, 255) if is_hovered else (220, 220, 220)
            put_text(disp, label, (ix1 + 14, iy1 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.46, txt_color, 1, cv2.LINE_AA)

    def save_changes(self, quiet: bool = False):
        """将内存中的修改即时写回 tag_observations.yaml (自动持久化)"""
        total_obs = 0
        total_kept = 0
        total_excl = 0
        total_enabled_imgs = 0
        for k in self.image_keys:
            is_enabled = self.frame_enabled_map.get(k, True)
            self.raw_manifest["images"][k]["enabled"] = is_enabled
            if is_enabled:
                total_enabled_imgs += 1
            for obs in self.raw_manifest["images"][k].get("observations", []):
                total_obs += 1
                if is_enabled and obs.get("keep", True):
                    total_kept += 1
                else:
                    total_excl += 1

        if "summary" not in self.raw_manifest or not isinstance(self.raw_manifest["summary"], dict):
            self.raw_manifest["summary"] = {}

        self.raw_manifest["summary"]["total_images"] = len(self.image_keys)
        self.raw_manifest["summary"]["total_enabled_images"] = total_enabled_imgs
        self.raw_manifest["summary"]["total_observations"] = total_obs
        self.raw_manifest["summary"]["total_kept"] = total_kept
        self.raw_manifest["summary"]["total_excluded"] = total_excl
        self.raw_manifest["summary"]["generated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_comments = (
            "# ==============================================================================\n"
            "# AprilTag 离线建图与 BA 平差观测数据清单 (Observations Manifest)\n"
            "# (已由交互式审核画板 tag_manifest_reviewer.py 同步更新)\n"
            "# ==============================================================================\n\n"
        )
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            f.write(header_comments)
            yaml.dump(self.raw_manifest, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        self.has_unsaved_changes = False
        self.has_modified_manifest = True
        if not quiet:
            log.info(f"[SAVE] 审核修改已成功保存至: {self.manifest_path}")
            log.info(f"       累计统计: 有效帧 {total_enabled_imgs}/{len(self.image_keys)}，保留观测 {total_kept} 次，已剔除 {total_excl} 次")

    def run(self):
        """启动 OpenCV 交互事件循环"""
        self.is_running = True
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.win_w, self.win_h)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        print("\n" + "=" * 70)
        print("   【AprilTag 观测样本全 GUI 交互审核画板已启动】")
        print("   * 鼠标左键：单击标靶翻转剔除/保留；点击底部按钮直接执行操作")
        print("   * GUI 按钮：[◀ 上一张] [下一张 ▶] [剔除整帧] [重新识别] [全量重扫] [复位] [保存] [退出]")
        print("   * 快捷键  ：A/D 翻页 | X 一键剔除/启用整帧 | R 复位 | S 保存 | Q/ESC 退出")
        print("=" * 70 + "\n")

        self.render_current_frame()
        if force_window_focus:
            force_window_focus(self.window_name)

        focus_attempts = 0
        while self.is_running:
            # 实时动态感知用户拖拽 Resize 或点击最大化窗口后的实际物理尺寸并自适应重排
            if self.viewport and self.viewport.sync_window_size(self.window_name):
                self.win_w = self.viewport.win_w
                self.win_h = self.viewport.win_h
                self.render_current_frame()

            if focus_attempts < 3 and force_window_focus:
                force_window_focus(self.window_name)
                focus_attempts += 1

            key = cv2.waitKey(25) & 0xFF
            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            # 翻页: A / 左箭头
            if key in (ord('a'), ord('A'), 81):
                self.prev_frame()

            # 翻页: D / 右箭头
            elif key in (ord('d'), ord('D'), 83):
                self.next_frame()

            # 模式切换: Tab (切换靶向聚焦与全量浏览)
            elif key == 9:
                self.toggle_focus_mode()

            # 一键整帧临时剔除/恢复启用: X
            elif key in (ord('x'), ord('X')):
                self.toggle_current_frame_enabled()

            # 重新识别当前帧: F / 扫描
            elif key in (ord('f'), ord('F')):
                self.rescan_current_frame()

            # 复位本帧: R
            elif key in (ord('r'), ord('R')):
                curr_key = self.current_image_key
                if curr_key in self.initial_states:
                    init_keeps = self.initial_states[curr_key]
                    observations = self.raw_manifest["images"][curr_key].get("observations", [])
                    for i, keep_val in enumerate(init_keeps):
                        if i < len(observations):
                            observations[i]["keep"] = keep_val
                    self.has_unsaved_changes = True
                    self.save_changes(quiet=True)
                    self.update_topology()
                    self.render_current_frame()
                    self.set_toast(f"[{curr_key}] 已恢复为初始加载状态 (已自动存盘)")

            # 保存: S (兼容单键，提示自动存盘)
            elif key in (ord('s'), ord('S')):
                self.save_changes(quiet=False)
                self.set_toast("当前清单已持久化 (实时自动存盘生效中)")

            # 一键平差验证: V
            elif key in (ord('v'), ord('V')):
                self.save_and_verify()
                break

            # 视口缩放复位: 0 / Z
            elif key in (ord('0'), ord('z'), ord('Z')):
                if self.viewport and self.viewport.reset_zoom():
                    self.render_current_frame()
                    self.set_toast("视口已复位 (1.0x)")

            # 退出: Q / ESC
            elif key in (ord('q'), ord('Q'), 27):
                break

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass  # 清理容错：窗口可能已被红叉关闭
        log.info("[EXIT] 交互审核完成，窗口已安全关闭。")


def main():
    parser = argparse.ArgumentParser(description="AprilTag 观测样本交互式审核画板")
    parser.add_argument("--manifest", type=str, default=None,
                        help="观测清单 tag_observations.yaml 路径 (默认自动对接当前工位)")
    parser.add_argument("--frame", type=str, default=None, help="初始定位图像名 (如 view_0016.png)")
    parser.add_argument("--focus_tag", type=int, default=None, help="指定定向靶向排查的标靶 ID")
    args = parser.parse_args()

    reviewer = TagManifestReviewer(
        manifest_path=args.manifest,
        focus_tag_id=args.focus_tag,
        initial_frame=args.frame
    )
    reviewer.run()


if __name__ == "__main__":
    main()
