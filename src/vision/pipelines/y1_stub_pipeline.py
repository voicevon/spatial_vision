"""
芦笋感知算法规划路线占位流水线 (Stub Pipeline: Y1)
=========================================================
为算法目录预注册后续演进路线：
  - 算法 Y1: YOLO深度学习法 (YOLO Deep Learning)
运行时提供无缝插拔降级与开发状态展示，避免应用异常。
"""

import time
from typing import Dict, List, Optional
import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry


class _BaseStubPipeline(BaseAsparagusPipeline):
    """规划中算法的占位抽象基类"""

    code: str = "C"
    pipeline_title: str = "算法规划"
    algorithm_summary: str = "正在开发接入中"
    theory_notes: str = ""

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep("stage_overview", "1.算法规划", f"{self.pipeline_title} 核心原理与技术演进方案"),
        ]

    def run(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated",
        nominal_z_mm: float = 640.0
    ) -> PipelineResult:
        t0 = time.perf_counter()
        h, w = color_bgr.shape[:2]

        # 构造高质感科技暗色底图
        vis = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)

        # 居中卡片区域
        card_w, card_h = min(int(w * 0.75), 780), 220
        cx, cy = w // 2, h // 2
        x1, y1 = cx - card_w // 2, cy - card_h // 2
        x2, y2 = x1 + card_w, y1 + card_h

        # 半透深色卡片背景
        overlay = vis.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (24, 28, 36), -1)
        cv2.addWeighted(overlay, 0.85, vis, 0.15, 0, vis)

        # 高亮边框与顶栏
        cv2.rectangle(vis, (x1, y1), (x2, y2), (60, 140, 240), 2)
        cv2.rectangle(vis, (x1, y1), (x2, y1 + 38), (35, 45, 60), -1)
        cv2.line(vis, (x1, y1 + 38), (x2, y1 + 38), (60, 140, 240), 1)

        # 标题与描述信息
        put_text(vis, f"[*] {self.pipeline_title}", (x1 + 16, y1 + 25),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        put_text(vis, f"[ 状态 ]: 规划路线预注册 (功能研发与算子接入中...)",
                 (x1 + 20, y1 + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (40, 220, 240), 1)
        put_text(vis, f"[ 核心思路 ]: {self.algorithm_summary}",
                 (x1 + 20, y1 + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 210, 220), 1)
        put_text(vis, f"[ 关键技术 ]: {self.theory_notes}",
                 (x1 + 20, y1 + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 175, 190), 1)
        put_text(vis, ">> 提示: 可在顶部算法下拉菜单中切换回 [算法 A] / [算法 B1] / [算法 B2] 进行即时解算",
                 (x1 + 20, y1 + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 220, 120), 1)

        step_snapshots = {"stage_overview": vis}
        elapsed = (time.perf_counter() - t0) * 1000.0

        return PipelineResult(
            targets=[],
            elapsed_ms=elapsed,
            step_snapshots=step_snapshots,
            extra_metrics={"status": "in_development", "code": self.code}
        )


@PipelineRegistry.register("yolo_deeplearning", "算法 Y1: YOLO深度学习法 (YOLO Deep Learning)")
class YoloDeepLearningPipeline(_BaseStubPipeline):
    """算法 Y1: YOLO深度学习法"""
    code = "Y1"
    name = "算法 Y1: YOLO深度学习法 (YOLO Deep Learning)"
    pipeline_title = "算法 Y1: YOLO深度学习法 (YOLO Deep Learning)"
    algorithm_summary = "YOLO-OBB / YOLO-Pose 轻量端到端关键点回归 (根部-抓取点-笋尖)"
    theory_notes = "超强抗泥土污损、严重交叉重叠与高光干扰，ONNXRuntime 工业级毫秒推理"

