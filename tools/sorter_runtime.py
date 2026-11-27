#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
项目 1：单根绿芦笋视觉测量分拣极简生产主程序 (SorterRuntime)
==========================================================
面向客户现场的开机即跑、防呆单一生产运行时：
  - 复用 CameraService 采集 2D USB / 工业定焦相机图像；
  - 通过 PlanarRectifier 自动检测机架 4 标靶，消除透视误差并在相机被碰歪时零人工自动恢复；
  - 接收下位机硬触发 (`TRIG,<seq>`) -> 拉正 ROI -> SingleAsparagusMeasurer -> GradingEngine -> 串口回传 (`RES,...`)；
  - 支持 `--headless` 无界面纯服务模式与 `--sim` 无串口仿真联调模式（按空格键手动模拟触发）。

启动示例:
  python tools/sorter_runtime.py
  python tools/sorter_runtime.py --sim
  python tools/sorter_runtime.py --headless --config config/sorter_station.yaml
"""

import argparse
import collections
import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.control.grading_engine import GradingEngine
from src.devices.camera_service import CameraService
from src.devices.sorter_serial import SorterSerialLink
from src.utils.logger import get_logger
from src.vision.planar_rectifier import PlanarRectifier, PlanarRigConfig
from src.vision.single_asparagus_measurer import MeasureConfig, SingleAsparagusMeasurer

log = get_logger(__name__)

DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "sorter_station.yaml")


@dataclass
class TriggerOutcome:
    """单次硬触发处理闭环结果"""
    seq: int
    grade: int
    grade_name: str
    length_mm: float
    diameter_mm: float
    status: str
    elapsed_ms: float
    tx_line: str
    vis_frame: Optional[np.ndarray] = None


def load_station_config(config_path: str) -> Dict[str, Any]:
    """加载分拣工位单一配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"分拣工位配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_rig_config(cfg: Dict[str, Any]) -> PlanarRigConfig:
    rig = cfg.get("rig") or {}
    raw_tags = rig.get("fixed_tags_mm") or {}
    fixed_tags = {int(k): (float(v[0]), float(v[1])) for k, v in raw_tags.items()}
    roi = rig.get("roi_mm") or [20.0, 15.0, 300.0, 105.0]
    return PlanarRigConfig(
        tag_size_mm=float(rig.get("tag_size_mm", 30.0)),
        fixed_tags_mm=fixed_tags,
        roi_mm=(float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3])),
        px_per_mm=float(rig.get("px_per_mm", 4.0)),
        max_reproj_err_px=float(rig.get("max_reproj_err_px", 1.5)),
        shift_threshold_px=float(rig.get("shift_threshold_px", 1.0)),
        use_tag_corners=bool(rig.get("use_tag_corners", False)),
    )


def build_measure_config(cfg: Dict[str, Any], mm_per_px: float) -> MeasureConfig:
    m = cfg.get("measure") or {}
    return MeasureConfig(
        orientation=str(m.get("orientation", "left_to_right")),
        diameter_offset_mm=float(m.get("diameter_offset_mm", 20.0)),
        green_h_low=int(m.get("green_h_low", 25)),
        green_h_high=int(m.get("green_h_high", 95)),
        green_s_min=int(m.get("green_s_min", 35)),
        green_v_min=int(m.get("green_v_min", 30)),
        white_s_max=int(m.get("white_s_max", 65)),
        white_v_min=int(m.get("white_v_min", 135)),
        green_ratio_threshold=float(m.get("green_ratio_threshold", 0.55)),
        min_white_root_mm=float(m.get("min_white_root_mm", 3.0)),
        fallback_mm_per_px=float(mm_per_px),
        min_area_px=int(m.get("min_area_px", 200)),
    )


class SorterPipelineController:
    """
    项目 1 分拣闭环核心控制器（与 GUI 解耦，便于无界面服务与自动化单测）。
    """

    def __init__(
        self,
        rectifier: PlanarRectifier,
        measurer: SingleAsparagusMeasurer,
        grading: GradingEngine,
        serial_link: SorterSerialLink,
    ):
        self.rectifier = rectifier
        self.measurer = measurer
        self.grading = grading
        self.serial_link = serial_link

        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self.last_outcome: Optional[TriggerOutcome] = None
        self.total_count: int = 0
        self.grade_counts: Dict[int, int] = collections.defaultdict(int)
        self._trigger_timestamps: Deque[float] = collections.deque(maxlen=120)

        self.serial_link.on_trigger = self.handle_trigger
        self.serial_link.is_ready_fn = self.is_ready

    def is_ready(self) -> bool:
        return self.rectifier.is_locked and self._latest_frame is not None

    def update_latest_frame(self, frame_bgr: np.ndarray) -> None:
        with self._frame_lock:
            self._latest_frame = frame_bgr

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def current_ppm(self) -> float:
        """基于最近 60 秒触发记录估算实时节拍 (Pieces Per Minute)"""
        now = time.time()
        recent = [ts for ts in self._trigger_timestamps if now - ts <= 60.0]
        if len(recent) < 2:
            return float(len(recent))
        span_s = max(1.0, now - recent[0])
        return round(len(recent) * 60.0 / span_s, 1)

    def handle_trigger(self, seq: int) -> TriggerOutcome:
        """响应下位机单次硬触发: 鸟瞰拉正 -> 单根测量 -> 分级 -> 串口回传"""
        t0 = time.perf_counter()
        raw_frame = self.get_latest_frame()

        if raw_frame is None:
            tx = self.serial_link.send_result(seq, self.grading.default_reject_grade, 0.0, 0.0, "NO_FRAME")
            outcome = TriggerOutcome(
                seq=seq, grade=self.grading.default_reject_grade, grade_name="无图像",
                length_mm=0.0, diameter_mm=0.0, status="NO_FRAME",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0, tx_line=tx,
            )
            self._record_outcome(outcome)
            return outcome

        if not self.rectifier.is_locked:
            self.rectifier.update(raw_frame, force_recompute=True)

        if not self.rectifier.is_locked:
            tx = self.serial_link.send_result(seq, self.grading.default_reject_grade, 0.0, 0.0, "CALIB_LOST")
            outcome = TriggerOutcome(
                seq=seq, grade=self.grading.default_reject_grade, grade_name="标靶失锁",
                length_mm=0.0, diameter_mm=0.0, status="CALIB_LOST",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0, tx_line=tx,
            )
            self._record_outcome(outcome)
            return outcome

        warped = self.rectifier.warp_roi(raw_frame)
        if warped is None:
            tx = self.serial_link.send_result(seq, self.grading.default_reject_grade, 0.0, 0.0, "CALIB_LOST")
            outcome = TriggerOutcome(
                seq=seq, grade=self.grading.default_reject_grade, grade_name="拉正失败",
                length_mm=0.0, diameter_mm=0.0, status="CALIB_LOST",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0, tx_line=tx,
            )
            self._record_outcome(outcome)
            return outcome

        # 鸟瞰拉正图已严格等比 (1 / px_per_mm)，直接测量整幅 ROI
        self.measurer.config.fallback_mm_per_px = self.rectifier.mm_per_px
        m = self.measurer.measure(warped, roi_rect=None)

        if not m.valid:
            tx = self.serial_link.send_result(seq, self.grading.default_reject_grade, 0.0, 0.0, "NO_TARGET")
            outcome = TriggerOutcome(
                seq=seq, grade=self.grading.default_reject_grade, grade_name="无目标",
                length_mm=0.0, diameter_mm=0.0, status="NO_TARGET",
                elapsed_ms=(time.perf_counter() - t0) * 1000.0, tx_line=tx,
                vis_frame=m.step_snapshots.get("stage4_result", warped),
            )
            self._record_outcome(outcome)
            return outcome

        decision = self.grading.evaluate(
            valid=True,
            effective_length_mm=m.effective_length_mm,
            diameter_mm=m.diameter_mm,
            has_white_root=m.has_white_root,
        )
        tx = self.serial_link.send_result(
            seq, decision.grade, m.effective_length_mm, m.diameter_mm, "OK"
        )
        outcome = TriggerOutcome(
            seq=seq,
            grade=decision.grade,
            grade_name=decision.name,
            length_mm=m.effective_length_mm,
            diameter_mm=m.diameter_mm,
            status="OK",
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            tx_line=tx,
            vis_frame=m.step_snapshots.get("stage4_result", warped),
        )
        self._record_outcome(outcome)
        return outcome

    def _record_outcome(self, outcome: TriggerOutcome) -> None:
        self.last_outcome = outcome
        self.total_count += 1
        self.grade_counts[outcome.grade] += 1
        self._trigger_timestamps.append(time.time())
        log.info(
            f"[Sorter] #{outcome.seq} -> Grade={outcome.grade}({outcome.grade_name}) "
            f"L={outcome.length_mm:.1f}mm D={outcome.diameter_mm:.1f}mm "
            f"[{outcome.status}] {outcome.elapsed_ms:.1f}ms"
        )


def _render_hud(
    controller: SorterPipelineController,
    raw_frame: np.ndarray,
    sim_mode: bool,
) -> np.ndarray:
    """绘制极简现场监控画面（拉正视图 + 标靶健康灯 + 最近一次测量结果）"""
    if controller.last_outcome is not None and controller.last_outcome.vis_frame is not None:
        view = controller.last_outcome.vis_frame.copy()
    else:
        warped = controller.rectifier.warp_roi(raw_frame)
        view = warped if warped is not None else raw_frame.copy()

    # 统一显示宽度，避免过高或过窄
    target_w = 960
    h, w = view.shape[:2]
    if w > 0 and w != target_w:
        scale = target_w / float(w)
        view = cv2.resize(view, (target_w, max(120, int(round(h * scale)))))

    bar_h = 84
    canvas = np.zeros((view.shape[0] + bar_h, view.shape[1], 3), dtype=np.uint8)
    canvas[:bar_h] = (28, 30, 34)
    canvas[bar_h:] = view

    st = controller.rectifier.status
    lock_color = (60, 210, 90) if st.locked else (60, 60, 235)
    lock_text = f"RIG LOCKED ({len(st.visible_tag_ids)} tags, err={st.reproj_err_px:.2f}px)" if st.locked else f"RIG LOST: {st.reason}"
    cv2.circle(canvas, (22, 26), 8, lock_color, -1)
    cv2.putText(canvas, lock_text, (40, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 2, cv2.LINE_AA)

    mode_str = "SIM" if sim_mode else controller.serial_link.port
    stats_text = f"Total: {controller.total_count} | PPM: {controller.current_ppm():.1f} | Serial: {mode_str}"
    cv2.putText(canvas, stats_text, (40, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 215, 220), 1, cv2.LINE_AA)

    if controller.last_outcome is not None:
        o = controller.last_outcome
        res_text = f"Last #{o.seq}: G{o.grade} L={o.length_mm:.1f}mm D={o.diameter_mm:.1f}mm ({o.status}, {o.elapsed_ms:.1f}ms)"
        cv2.putText(canvas, res_text, (460, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 235, 255), 1, cv2.LINE_AA)

    hint = "[SPACE] Sim Trigger | [R] Re-Lock Rig | [Q/ESC] Quit" if sim_mode else "[R] Re-Lock Rig | [Q/ESC] Quit"
    cv2.putText(canvas, hint, (460, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 155, 165), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description="项目 1：单根绿芦笋视觉测量分拣极简生产主程序")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="工位 YAML 配置文件路径")
    parser.add_argument("--sim", action="store_true", help="串口仿真模式（无需连接下位机，按空格触发测量）")
    parser.add_argument("--headless", action="store_true", help="无界面纯后台服务模式")
    args = parser.parse_args()

    cfg = load_station_config(args.config)
    rig_cfg = build_rig_config(cfg)
    rectifier = PlanarRectifier(rig_cfg)
    measurer = SingleAsparagusMeasurer(build_measure_config(cfg, rectifier.mm_per_px))
    grading = GradingEngine.from_config(cfg.get("grading") or {})

    ser_cfg = cfg.get("serial") or {}
    serial_link = SorterSerialLink(
        port=str(ser_cfg.get("port", "COM3")),
        baudrate=int(ser_cfg.get("baudrate", 115200)),
        timeout_ms=int(ser_cfg.get("timeout_ms", 50)),
        simulate=args.sim,
    )
    controller = SorterPipelineController(rectifier, measurer, grading, serial_link)

    cam_cfg = cfg.get("camera") or {}
    camera = CameraService()
    recheck_interval = float((cfg.get("rig") or {}).get("recheck_interval_s", 1.0))

    try:
        camera.start_usb(
            width=int(cam_cfg.get("width", 1280)),
            height=int(cam_cfg.get("height", 720)),
            fps=int(cam_cfg.get("fps", 30)),
            device_index=int(cam_cfg.get("device_index", 0)),
        )
        serial_link.start()
        log.info("[SorterRuntime] 项目 1 生产运行时已启动")

        last_recheck = 0.0
        sim_seq = 1
        win_name = "Asparagus Sorter Runtime"

        while True:
            frame = camera.read_frame(timeout_ms=500)
            if frame is None:
                time.sleep(0.01)
                continue
            controller.update_latest_frame(frame)

            now = time.time()
            if (not rectifier.is_locked) or (now - last_recheck >= recheck_interval):
                rectifier.update(frame)
                last_recheck = now

            if args.headless:
                time.sleep(0.005)
                continue

            hud = _render_hud(controller, frame, args.sim)
            cv2.imshow(win_name, hud)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                rectifier.update(frame, force_recompute=True)
                last_recheck = time.time()
            if args.sim and key == 32:  # SPACE
                serial_link.inject_trigger(sim_seq)
                sim_seq += 1
    except KeyboardInterrupt:
        log.info("[SorterRuntime] 收到中断信号，正在优雅退出...")
    finally:
        serial_link.stop()
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
