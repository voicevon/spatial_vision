#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCARA 抓取生产工作台 (SCARA Production Studio)
==============================================
专用于上料抓取工位的产线实时运行工作台：
1. 监听进料皮带 (Smart ROI: role='source') 芦笋物料;
2. 调度器 (SortingDispatcher) 动态决策目标落料槽位与防溢控制;
3. 轨迹规划器 (ScaraMotionPlanner) 实时输出防撞 G-code;
4. 驱动 SCARA 机械臂串口执行全自动搬运闭环。
"""

import os
import sys
import time
import argparse
from typing import Optional, List, Dict, Tuple, Any

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.base_cv_app import BaseCvApp
from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, put_text, measure_text
from src.utils.logger import get_logger

from src.calibration.workspace_manager import (
    WorkspaceManager, Workspace,
    load_workspace_coordinate_manager,
    load_workspace_roi_manager
)
from src.vision.asparagus_analyzer import AsparagusAnalyzer, AsparagusTarget
from src.control.sorting_dispatcher import SortingDispatcher, DestinationSlot
from src.control.scara_motion_planner import ScaraMotionPlanner, ScaraPickTask
from src.control.robot_serial import RobotSerial

log = get_logger(__name__)

APP_ID = "scara_production"
WINDOW_KEY = "flux_vision_scara_production"
BASE_W = 1280
BASE_H = 800


class ScaraProductionApp(BaseCvApp):
    """SCARA 抓取生产工作台 GUI 应用"""

    def __init__(self, settings_file: Optional[str] = None):
        super().__init__(
            app_id=APP_ID,
            base_w=BASE_W,
            base_h=BASE_H,
            window_name=WINDOW_KEY,
            window_title="SCARA 抓取生产工作台 - SCARA Production",
            settings_file=settings_file,
            min_w=1000,
            min_h=650,
            enable_keyboard_zoom=True,
            responsive=True,
        )

        # 1. 工位与沙盒管理
        self.workspace_mgr = WorkspaceManager()
        self.cur_ws: Optional[Workspace] = self.workspace_mgr.get_current_workspace()
        self.current_workspace_id = self.cur_ws.workspace_id if self.cur_ws else ""
        self.current_workspace_name = self.cur_ws.name if self.cur_ws else "未选择工位"

        # 3. 生产循环状态
        self.auto_running: bool = False
        self.total_picked_count: int = 0
        self.last_cycle_time_ms: float = 0.0
        self.last_action_log: List[str] = [
            f"[{time.strftime('%H:%M:%S')}] 系统已就绪，当前工位: {self.current_workspace_name}"
        ]

        # 2. 控制层核心引擎
        self.motion_planner = ScaraMotionPlanner(safe_z=80.0, feedrate_xy=4000)
        self.dispatcher = SortingDispatcher(default_drop_x=220.0, default_drop_y=0.0)
        self._init_workspace_rois()

        # 当前周期决策缓存
        self.current_target: Optional[AsparagusTarget] = None
        self.current_slot: Optional[DestinationSlot] = None
        self.current_task: Optional[ScaraPickTask] = None

        # 4. 模拟物料队列 (无真实相机时自愈运转)
        self.detected_targets: List[AsparagusTarget] = []
        self._sim_target_seq = 1

    def _init_workspace_rois(self):
        """装载工位 Smart ROI 并初始化调度器"""
        if not self.cur_ws:
            return
        roi_mgr = load_workspace_roi_manager(self.cur_ws)
        rois = roi_mgr.list_rois()
        self.dispatcher.load_from_rois(rois)
        self._append_log(f"已装载 {len(self.dispatcher.source_rois)} 个进料区, {len(self.dispatcher.destination_slots)} 个落料槽位")

    def _append_log(self, text: str):
        ts = time.strftime("%H:%M:%S")
        self.last_action_log.append(f"[{ts}] {text}")
        if len(self.last_action_log) > 10:
            self.last_action_log.pop(0)

    # ------------------------------ 生产业务核心动作 ------------------------------
    def trigger_production_cycle(self):
        """执行单次抓取生产节拍"""
        # 1. 若当前无检测物料，生成一个模拟进料测试样本
        if not self.detected_targets:
            t = AsparagusTarget(
                id=self._sim_target_seq,
                center_px=(320.0, 240.0),
                length_px=200.0,
                diam_px=15.0,
                yaw_deg=float(np.random.uniform(-45.0, 45.0)),
                axis_vector=(1.0, 0.0),
                box_corners=np.zeros((4, 2)),
                contour=np.zeros((10, 1, 2)),
                length_mm=190.0,
                diam_mm=13.5,
                grip_x=0.0, grip_y=0.0, grip_z=500.0, z_top=480.0,
                rel_height_mm=25.0,
                robot_x=float(np.random.uniform(80.0, 180.0)),
                robot_y=float(np.random.uniform(150.0, 260.0)),
                robot_z=15.0,
                robot_r=float(np.random.uniform(-30.0, 30.0)),
                is_topmost=True,
                calibration_source="tag_online",
                grade=np.random.choice(["A", "A", "B"]),
                confidence=0.92,
            )
            self._sim_target_seq += 1
            self.detected_targets = [t]

        t_start = time.perf_counter()
        target, slot, task = self.dispatcher.dispatch_cycle(self.detected_targets)
        t_end = time.perf_counter()
        self.last_cycle_time_ms = (t_end - t_start) * 1000.0

        self.current_target = target
        self.current_slot = slot
        self.current_task = task

        if target and slot and task:
            # 生成防撞 G-code
            gcode = self.motion_planner.plan(task)
            # 确认落料完成
            self.dispatcher.confirm_placed(slot.slot_index)
            self.total_picked_count += 1
            self._append_log(f"成功抓取 #{target.id}({target.grade}级) -> 落入槽位 #{slot.slot_index + 1}[{slot.name}]")
            self.set_toast(f"已执行抓取: 物料#{target.id} -> 槽位#{slot.slot_index + 1}", duration=2.0)
            self.detected_targets.clear()
        elif target and not slot:
            self._append_log(f"物料 #{target.id} 无可用槽位 (所有匹配槽已满溢！)")
            self.set_toast("⚠️ 槽位满溢报警：请清理满载料槽或按 [R] 复位！", sticky=True)
            self.auto_running = False
        else:
            self._append_log("当前无可用抓取物料")

    def toggle_auto(self):
        """开启或暂停全自动连续抓取"""
        self.auto_running = not self.auto_running
        state_str = "【自动抓取运行中】" if self.auto_running else "【已暂停挂起】"
        self._append_log(f"生产模式切换: {state_str}")
        self.set_toast(f"产线状态: {state_str}")

    def reset_all_slots(self):
        """人工换箱复位"""
        self.dispatcher.reset_all_slots()
        self._append_log("人工复位清空所有落料槽位计数")
        self.set_toast("各槽位计数已复位清零", duration=2.0)

    # ------------------------------ 事件与按键 ------------------------------
    def on_key(self, raw_key: int) -> bool:
        key = chr(raw_key & 0xFF).lower() if (raw_key & 0xFF) < 128 else ""
        if key in ("x", "\x1b"):
            self.stop()
            return True
        elif key == " ":
            self.toggle_auto()
            return True
        elif key == "s":
            self.trigger_production_cycle()
            return True
        elif key == "r":
            self.reset_all_slots()
            return True
        return False

    def on_mouse_click(self, x: int, y: int, button: int):
        for rect, action in reversed(self._buttons):
            x1, y1, x2, y2 = rect
            if x1 <= x <= x2 and y1 <= y <= y2:
                if action == "TOGGLE_AUTO":
                    self.toggle_auto()
                elif action == "SINGLE_CYCLE":
                    self.trigger_production_cycle()
                elif action == "RESET_SLOTS":
                    self.reset_all_slots()
                elif action == "EXIT":
                    self.stop()
                return

    # ------------------------------ 渲染逻辑 ------------------------------
    def render(self) -> np.ndarray:
        W, H = self.win_mgr.canvas_w, self.win_mgr.canvas_h
        s = min(W / BASE_W, H / BASE_H)
        canvas = np.full((H, W, 3), (16, 18, 22), dtype=np.uint8)

        self._buttons = []

        # 1. 顶部 Header (高 60px)
        th = int(60 * s)
        cv2.rectangle(canvas, (0, 0), (W, th), (22, 26, 32), -1)
        cv2.line(canvas, (0, th), (W, th), GuiTheme.BORDER, 1)

        # 标题 Logo
        draw_text(canvas, "SCARA 抓取生产工作台", (int(16 * s), int(15 * s)), int(18 * s), GuiTheme.TEXT, bold=True)
        ws_tag = f"工位: {self.current_workspace_name}"
        draw_text(canvas, ws_tag, (int(250 * s), int(20 * s)), int(13 * s), GuiTheme.ACCENT)

        # 运行状态胶囊
        stat_color = (0, 255, 120) if self.auto_running else (240, 180, 40)
        stat_text = "● 自动运行中" if self.auto_running else "○ 挂起暂停"
        draw_text(canvas, stat_text, (int(440 * s), int(19 * s)), int(14 * s), stat_color, bold=True)

        # 顶部按钮组
        bx = W - int(16 * s)
        # 退出按钮
        bw_exit = int(90 * s)
        bx -= bw_exit
        r_exit = (bx, int(12 * s), bx + bw_exit, th - int(12 * s))
        cv2.rectangle(canvas, (r_exit[0], r_exit[1]), (r_exit[2], r_exit[3]), (35, 40, 50), -1)
        draw_text(canvas, "退出 [X]", (r_exit[0] + int(16 * s), r_exit[1] + int(8 * s)), int(13 * s), GuiTheme.TEXT_SUB)
        self._buttons.append((r_exit, "EXIT"))

        # 换箱复位按钮
        bw_reset = int(110 * s)
        bx -= bw_reset + int(10 * s)
        r_reset = (bx, int(12 * s), bx + bw_reset, th - int(12 * s))
        cv2.rectangle(canvas, (r_reset[0], r_reset[1]), (r_reset[2], r_reset[3]), (35, 40, 50), -1)
        draw_text(canvas, "换箱复位 [R]", (r_reset[0] + int(12 * s), r_reset[1] + int(8 * s)), int(13 * s), GuiTheme.TEXT_SUB)
        self._buttons.append((r_reset, "RESET_SLOTS"))

        # 单拍抓取测试按钮
        bw_single = int(120 * s)
        bx -= bw_single + int(10 * s)
        r_single = (bx, int(12 * s), bx + bw_single, th - int(12 * s))
        cv2.rectangle(canvas, (r_single[0], r_single[1]), (r_single[2], r_single[3]), (35, 40, 50), -1)
        draw_text(canvas, "单拍测试 [S]", (r_single[0] + int(14 * s), r_single[1] + int(8 * s)), int(13 * s), GuiTheme.ACCENT)
        self._buttons.append((r_single, "SINGLE_CYCLE"))

        # 自动开关按钮
        bw_auto = int(130 * s)
        bx -= bw_auto + int(10 * s)
        r_auto = (bx, int(12 * s), bx + bw_auto, th - int(12 * s))
        btn_bg = (30, 80, 50) if self.auto_running else (30, 50, 80)
        cv2.rectangle(canvas, (r_auto[0], r_auto[1]), (r_auto[2], r_auto[3]), btn_bg, -1)
        btn_text = "暂停生产 [SPACE]" if self.auto_running else "启动生产 [SPACE]"
        draw_text(canvas, btn_text, (r_auto[0] + int(10 * s), r_auto[1] + int(8 * s)), int(13 * s), (0, 255, 180) if self.auto_running else GuiTheme.GOLD, bold=True)
        self._buttons.append((r_auto, "TOGGLE_AUTO"))

        # 2. 左右双栏布局
        pad = int(16 * s)
        left_w = int((W - pad * 3) * 0.58)
        right_w = W - pad * 3 - left_w
        left_h = H - th - pad * 2 - int(40 * s)

        lx1, ly1 = pad, th + pad
        lx2, ly2 = lx1 + left_w, ly1 + left_h

        rx1, ry1 = lx2 + pad, ly1
        rx2, ry2 = rx1 + right_w, ly2

        # 左栏：进料皮带视口监控
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), (20, 24, 30), -1)
        cv2.rectangle(canvas, (lx1, ly1), (lx2, ly2), GuiTheme.BORDER, 1)
        draw_text(canvas, "进料皮带感知监控 (Source ROI)", (lx1 + int(12 * s), ly1 + int(14 * s)), int(15 * s), GuiTheme.TEXT, bold=True)

        # 绘制仿真传送带背景网格
        grid_y_start = ly1 + int(45 * s)
        for gy in range(grid_y_start, ly2 - int(20 * s), int(30 * s)):
            cv2.line(canvas, (lx1 + 20, gy), (lx2 - 20, gy), (26, 32, 40), 1)

        # 绘制目标芦笋与拾取位姿
        if self.current_target:
            t = self.current_target
            # 绘制示意抓取物料
            cx = lx1 + int((lx2 - lx1) * 0.45)
            cy = ly1 + int((ly2 - ly1) * 0.5)
            cv2.ellipse(canvas, (cx, cy), (int(90 * s), int(12 * s)), t.robot_r, 0, 360, (0, 255, 120), -1)
            # 抓取中心十字
            cv2.drawMarker(canvas, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, int(20 * s), 2)
            draw_text(canvas, f"#{t.id} 拾取点 ({t.robot_x:.1f}, {t.robot_y:.1f}) R:{t.robot_r:.1f}°",
                      (cx - int(70 * s), cy - int(25 * s)), int(12 * s), GuiTheme.GOLD, bold=True)
        else:
            draw_text(canvas, "皮带待料中 / 等待下一批物料进入检测区...",
                      (lx1 + int(40 * s), ly1 + int(left_h * 0.45)), int(14 * s), GuiTheme.TEXT_MUTED)

        # 右栏：落料槽位分配与生产看板
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (20, 24, 30), -1)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), GuiTheme.BORDER, 1)

        # 产线 KPI 指标
        draw_text(canvas, "分选槽位与生产调度监控", (rx1 + int(12 * s), ry1 + int(14 * s)), int(15 * s), GuiTheme.TEXT, bold=True)
        kpi_y = ry1 + int(45 * s)
        draw_text(canvas, f"已搬运总量: {self.total_picked_count} 根", (rx1 + int(16 * s), kpi_y), int(13 * s), GuiTheme.ACCENT)
        draw_text(canvas, f"决策延时: {self.last_cycle_time_ms:.1f} ms", (rx1 + int(180 * s), kpi_y), int(13 * s), GuiTheme.TEXT_SUB)

        # 绘制 Destination 各槽位柱状进度
        slots = list(self.dispatcher.destination_slots.values())
        slot_start_y = kpi_y + int(30 * s)
        slot_h = int(36 * s)

        if slots:
            for idx, slot in enumerate(slots[:6]):
                sy1 = slot_start_y + idx * (slot_h + int(8 * s))
                sy2 = sy1 + slot_h
                cv2.rectangle(canvas, (rx1 + int(12 * s), sy1), (rx2 - int(12 * s), sy2), (26, 30, 38), -1)

                # 进度条底槽
                bar_x1 = rx1 + int(130 * s)
                bar_x2 = rx2 - int(70 * s)
                bar_y1 = sy1 + int(10 * s)
                bar_y2 = sy2 - int(10 * s)
                cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x2, bar_y2), (18, 20, 26), -1)

                # 填充进度
                pct = min(1.0, slot.current_count / max(1, slot.capacity_max))
                fill_w = int((bar_x2 - bar_x1) * pct)
                bar_col = (0, 100, 255) if slot.is_full else (0, 255, 120)
                if fill_w > 0:
                    cv2.rectangle(canvas, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), bar_col, -1)

                # 槽位标签与计数文本
                g_str = "/".join(slot.target_grades)
                draw_text(canvas, f"槽#{slot.slot_index + 1} [{g_str}]", (rx1 + int(18 * s), sy1 + int(10 * s)), int(12 * s), GuiTheme.TEXT, bold=True)
                stat_cnt_str = f"{slot.current_count}/{slot.capacity_max}"
                draw_text(canvas, stat_cnt_str, (bar_x2 + int(10 * s), sy1 + int(10 * s)), int(12 * s), bar_col, bold=True)
        else:
            draw_text(canvas, "工位未配置 destination 槽位 (使用默认落料点)",
                      (rx1 + int(20 * s), slot_start_y + int(20 * s)), int(13 * s), GuiTheme.TEXT_MUTED)

        # 底部日志面板
        log_y = ry2 - int(160 * s)
        cv2.rectangle(canvas, (rx1 + int(8 * s), log_y), (rx2 - int(8 * s), ry2 - int(8 * s)), (14, 16, 20), -1)
        draw_text(canvas, "实时调度与通信日志", (rx1 + int(14 * s), log_y + int(6 * s)), int(11 * s), GuiTheme.TEXT_MUTED, bold=True)
        curr_ly = log_y + int(22 * s)
        for line in self.last_action_log[-5:]:
            draw_text(canvas, line, (rx1 + int(14 * s), curr_ly), int(11 * s), GuiTheme.TEXT_SUB)
            curr_ly += int(16 * s)

        # 3. 底部状态提示条 (高 30px)
        bottom_y = H - int(30 * s)
        cv2.rectangle(canvas, (0, bottom_y), (W, H), (14, 16, 20), -1)
        toast = self._toast if self._toast and time.time() < self._toast_until else "快捷键: [SPACE] 启动/暂停自动生产 | [S] 单拍测试 | [R] 换箱清零 | [X] 退出"
        draw_text(canvas, toast, (int(16 * s), bottom_y + int(7 * s)), int(12 * s), GuiTheme.ACCENT if self._toast else GuiTheme.TEXT_MUTED)

        # 自动节拍轮询
        if self.auto_running:
            time.sleep(0.05)
            self.trigger_production_cycle()

        return canvas


def main():
    parser = argparse.ArgumentParser(description="SCARA 抓取生产工作台")
    parser.add_argument("--settings", default=None, help="GUI 配置文件路径")
    args = parser.parse_args()

    app = ScaraProductionApp(settings_file=args.settings)
    app.run()


if __name__ == "__main__":
    main()
