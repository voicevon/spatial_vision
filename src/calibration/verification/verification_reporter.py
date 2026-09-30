# -*- coding: utf-8 -*-
"""
离线精度体检报告导出与统计分析器 (VerificationReporter)
======================================================
单一职责设计：
  1. 对 Leave-One-Out 盲测原始测量结果进行 Per-Tag / Per-Frame 多维稳健统计；
  2. 运用 MAD 稳健统计识别系统性偏差标靶与建议回审图像；
  3. 综合多维门限评定放行等级 (PASS / ACCEPTABLE / FAIL)；
  4. 渲染并持久化符合 GitHub 规范的 Markdown 详尽精度体验报表。
"""

import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def get_default_report_dir() -> str:
    """获取当前激活工位的标定报告输出目录"""
    try:
        from src.workspace.workspace_manager import WorkspaceManager
        return WorkspaceManager().get_current_workspace().calib_reports_dir
    except Exception:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "workspaces", "default", "calibration", "reports")
        )


class VerificationReporter:
    """标定验证报表生成与统计分析器 (纯领域计算与格式化，零 GUI 依赖)"""

    @classmethod
    def get_output_dir(cls) -> str:
        return get_default_report_dir()

    @staticmethod
    def aggregate_statistics(all_results: List[Dict]) -> Dict[str, Any]:
        """
        汇总 Per-Tag / Per-Frame 统计，识别系统性偏差与建议回审帧。
        :param all_results: LOO 盲测单次验证记录列表
        :return: 包含各项中位数、均值、嫌疑标靶、回审帧与评级的字典
        """
        if not all_results:
            return {
                "pass": False,
                "grade": "FAIL",
                "grade_label": "无有效盲测结果",
                "message": "无有效盲测结果",
                "total_tests": 0,
                "global_median_px": 0.0,
                "global_mean_px": 0.0,
                "global_std_px": 0.0,
                "global_median_mm": 0.0,
                "global_mean_mm": 0.0,
                "outlier_threshold_px": 0.0,
                "per_tag": {},
                "per_frame": {},
                "flagged_frames": [],
                "flagged_tags": [],
            }

        # 1. Per-Tag 统计
        tag_errors = {}
        for r in all_results:
            tid = r["blind_tag_id"]
            if tid not in tag_errors:
                tag_errors[tid] = {"err_px": [], "err_mm": []}
            tag_errors[tid]["err_px"].append(r["err_px"])
            tag_errors[tid]["err_mm"].append(r["err_mm"])

        per_tag = {}
        for tid, data in sorted(tag_errors.items()):
            arr_px = np.array(data["err_px"])
            arr_mm = np.array(data["err_mm"])
            per_tag[tid] = {
                "count": len(arr_px),
                "median_px": float(np.median(arr_px)),
                "mean_px": float(np.mean(arr_px)),
                "max_px": float(np.max(arr_px)),
                "std_px": float(np.std(arr_px)),
                "median_mm": float(np.median(arr_mm)),
                "mean_mm": float(np.mean(arr_mm)),
                "max_mm": float(np.max(arr_mm)),
            }

        # 2. Per-Frame 统计
        frame_errors = {}
        for r in all_results:
            fname = r["image"]
            if fname not in frame_errors:
                frame_errors[fname] = {"err_px": [], "tags": []}
            frame_errors[fname]["err_px"].append(r["err_px"])
            frame_errors[fname]["tags"].append(r["blind_tag_id"])

        per_frame = {}
        for fname, data in sorted(frame_errors.items()):
            arr = np.array(data["err_px"])
            per_frame[fname] = {
                "tag_count": len(arr),
                "mean_px": float(np.mean(arr)),
                "max_px": float(np.max(arr)),
                "tags_tested": sorted(data["tags"]),
            }

        # 3. 全局统计
        all_px = np.array([r["err_px"] for r in all_results])
        all_mm = np.array([r["err_mm"] for r in all_results])
        global_median_px = float(np.median(all_px))
        global_median_mm = float(np.median(all_mm))
        global_mean_px = float(np.mean(all_px))
        global_std_px = float(np.std(all_px))

        # 4. 系统性偏差检测: Tag 中位误差 > 全局中位数 × 2
        flagged_tags = []
        for tid, stats in per_tag.items():
            if stats["median_px"] > max(2.5, global_median_px * 2.0):
                flagged_tags.append(tid)

        # 5. 坏帧检测 (基于稳健统计门限)
        mad_px = float(np.median(np.abs(all_px - global_median_px)))
        outlier_thresh = max(5.0, min(15.0, global_median_px + 3.0 * (1.4826 * mad_px if mad_px > 0.1 else 1.0)))
        flagged_frames = []
        for fname, stats in per_frame.items():
            if stats["mean_px"] > outlier_thresh or stats["max_px"] > 25.0:
                flagged_frames.append(fname)

        # 6. 综合放行评级 (Pass / Acceptable / Fail)
        if global_median_px <= 1.5 and global_median_mm <= 1.0 and len(flagged_tags) == 0 and len(flagged_frames) == 0:
            grade = "PASS"
            grade_label = "合格 (PASS) — 精度极佳，允许直接上线"
        elif global_median_px <= 3.0 and global_median_mm <= 2.5 and len(flagged_tags) <= 1:
            grade = "ACCEPTABLE"
            grade_label = "基本合格 (ACCEPTABLE) — 可准入 AR 验证"
        else:
            grade = "FAIL"
            grade_label = "不合格 (FAIL) — 建议回审后重新 BA 求解"

        return {
            "pass": grade in ("PASS", "ACCEPTABLE"),
            "grade": grade,
            "grade_label": grade_label,
            "total_tests": len(all_results),
            "global_median_px": global_median_px,
            "global_mean_px": global_mean_px,
            "global_std_px": global_std_px,
            "global_median_mm": global_median_mm,
            "global_mean_mm": float(np.mean(all_mm)),
            "outlier_threshold_px": outlier_thresh,
            "per_tag": per_tag,
            "per_frame": per_frame,
            "flagged_tags": flagged_tags,
            "flagged_frames": flagged_frames,
        }

    @classmethod
    def export_markdown_report(
        cls,
        stats: Dict[str, Any],
        all_results: List[Dict],
        map_path: str,
        image_dir: str,
        actual_source: str,
        vis_dir: str,
        out_path: Optional[str] = None
    ) -> str:
        """
        输出完整 Markdown 精度体检报告至文件。
        """
        if out_path is None:
            out_dir = cls.get_output_dir()
            os.makedirs(out_dir, exist_ok=True)
            report_path = os.path.join(out_dir, "offline_verification_report.md")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
            report_path = out_path

        grade = stats.get("grade", "N/A")
        grade_label = stats.get("grade_label", "N/A")
        grade_emoji = "[PASS]" if grade == "PASS" else ("[WARN]" if grade == "ACCEPTABLE" else "[FAIL]")

        status_median_px = "[PASS]" if stats.get('global_median_px', 99) <= 1.5 else ("[WARN]" if stats.get('global_median_px', 99) <= 3.0 else "[FAIL]")
        status_median_mm = "[PASS]" if stats.get('global_median_mm', 99) <= 1.0 else ("[WARN]" if stats.get('global_median_mm', 99) <= 2.5 else "[FAIL]")
        status_flagged_tags = "[PASS]" if len(stats.get('flagged_tags', [])) == 0 else ("[WARN]" if len(stats.get('flagged_tags', [])) <= 2 else "[FAIL]")
        status_flagged_frames = "[PASS]" if len(stats.get('flagged_frames', [])) == 0 else "[WARN]"

        source_label = "已审核观测清单 (tag_observations.yaml)" if actual_source == "manifest" else "原始采图重新检测 (端到端独立复检)"

        lines = [
            "# 离线标定精度体检与 Leave-One-Out 盲测批量验证报告",
            "",
            f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"> 地图文件: `{map_path}`  ",
            f"> 采图目录: `{image_dir}`  ",
            f"> 验证数据源: **{source_label}**  ",
            f"> 评审结论: **{grade_emoji} {grade_label}**",
            "",
            "---",
            "",
            "## 1. 全局核心指标",
            "",
            "| 指标 | 测量值 | PASS 门限 | ACCEPTABLE 门限 | 状态 |",
            "| :--- | :--- | :--- | :--- | :---: |",
            f"| **LOO 盲测中位像元误差** | **{stats.get('global_median_px', 0):.3f} px** | ≤ 1.50 px | ≤ 3.00 px | {status_median_px} |",
            f"| **LOO 盲测中位空间偏差** | **{stats.get('global_median_mm', 0):.3f} mm** | ≤ 1.00 mm | ≤ 2.50 mm | {status_median_mm} |",
            f"| **LOO 盲测均值像元误差** | {stats.get('global_mean_px', 0):.3f} px | — | — | — |",
            f"| **系统性偏差嫌疑 Tag 数** | {len(stats.get('flagged_tags', []))} 个 | 0 个 | ≤ 2 个 | {status_flagged_tags} |",
            f"| **建议回审帧数** | {len(stats.get('flagged_frames', []))} 帧 | 0 帧 | — | {status_flagged_frames} |",
            f"| **盲测执行总数** | {stats.get('total_tests', 0)} 次 | — | — | — |",
            "",
            "## 2. Per-Tag 盲测精度统计",
            "",
            "| Tag ID | 盲测次数 | 中位误差 (px) | 均值 (px) | 最大 (px) | 标准差 (px) | 中位空间 (mm) | 状态 |",
            "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for tid, s in sorted(stats.get("per_tag", {}).items()):
            is_flagged = tid in stats.get("flagged_tags", [])
            status = "[FAIL] 偏差嫌疑" if is_flagged else ("[PASS]" if s["median_px"] <= 1.5 else "[WARN]")
            lines.append(
                f"| Tag #{tid:2d} | {s['count']:3d} | {s['median_px']:6.2f} | {s['mean_px']:6.2f} | "
                f"{s['max_px']:6.2f} | {s['std_px']:5.2f} | {s['median_mm']:6.2f} | {status} |"
            )

        # Per-Frame 统计
        lines.extend([
            "",
            "## 3. Per-Frame 盲测精度统计",
            "",
            "| 图像文件 | 盲测 Tag 数 | 帧均误差 (px) | 帧最大误差 (px) | 状态 |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for fname, s in sorted(stats.get("per_frame", {}).items()):
            is_flagged = fname in stats.get("flagged_frames", [])
            status = "[FAIL] 建议回审" if is_flagged else ("[PASS]" if s["mean_px"] <= 1.5 else "[WARN]")
            lines.append(
                f"| {fname} | {s['tag_count']:2d} | {s['mean_px']:6.2f} | {s['max_px']:6.2f} | {status} |"
            )

        # 系统性偏差 Tag 详情
        if stats.get("flagged_tags"):
            lines.extend([
                "",
                "## 4. 系统性偏差嫌疑标靶",
                "",
                "> [!WARNING]",
                f"> 以下 {len(stats['flagged_tags'])} 个标靶的盲测中位误差显著高于全局中位数的 2 倍 ({stats.get('global_median_px', 0):.2f} px × 2)，",
                "> 可能是 BA 地图中该 Tag 的空间坐标不够准确，建议检查相关观测样本并回审。",
                "",
            ])
            for tid in stats["flagged_tags"]:
                s = stats["per_tag"][tid]
                lines.append(f"- **Tag #{tid}**: 中位 {s['median_px']:.2f} px / {s['median_mm']:.2f} mm (测试 {s['count']} 次)")

        # 建议回审帧
        if stats.get("flagged_frames"):
            lines.extend([
                "",
                "## 5. 建议回审帧",
                "",
                "> [!WARNING]",
                f"> 以下帧的平均盲测误差超过异常阈值 ({stats.get('outlier_threshold_px', 0):.2f} px)，建议在审核画板中检查并考虑剔除。",
                "",
            ])
            for fname in stats["flagged_frames"]:
                s = stats["per_frame"][fname]
                lines.append(f"- **{fname}**: 帧均 {s['mean_px']:.2f} px (最大: {s['max_px']:.2f} px)")

        # 可视化指引
        lines.extend([
            "",
            "## 6. 逐帧 LOO 盲测可视化图",
            "",
            f"分析图像已输出至: `{vis_dir}`",
            "",
            "- 浅灰虚线轮廓 = 物理实测角点位置",
            "- 彩色实线轮廓 = LOO 盲测反推投影位置 (绿色=误差达标, 橙色=误差偏高)",
            "- 彩色箭头 = 逐角点残差矢量 (已放大 15 倍)",
            "",
            "---",
            f"*报告由 `VerificationReporter` 自动生成*",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        log.info(f"[OK] 离线精度体检报告已生成: {report_path}")
        return report_path
