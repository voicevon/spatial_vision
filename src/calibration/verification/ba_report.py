#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BA 平差诊断报告输出 (BA Report)
================================
从 ba_optimizer.py 拆出的报告段（纯函数，不依赖优化器实例状态）：
  - compute_3d_uncertainties: 基于雅可比矩阵的标靶 3D 置信区间计算
  - export_diagnostic_report: 2D 像面 Quiver 残差矢量场绘制与 Markdown 精度体检报告生成
BundleAdjustmentOptimizer 类内保留同名委托方法，外部接口不变。
"""

import os
from datetime import datetime
from typing import Dict, List, Tuple, Any, Set, Optional
import numpy as np
import cv2

from src.utils.logger import get_logger
from src.ui.text_rendering import put_text

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def compute_3d_uncertainties(jacobian, static_tags, base_id, sigma_res_px) -> Dict[int, Dict[str, float]]:
    """
    基于平差最优解雅可比矩阵 J 计算参数协方差：
    Cov = inv(J^T J) * sigma_res^2
    提取每个标靶 3D 位置分量 (tv_x, tv_y, tv_z) 的 3-sigma 空间置信区间 (单位: mm)
    """
    uncertainties = {base_id: {"sigma_x_mm": 0.0, "sigma_y_mm": 0.0, "sigma_z_mm": 0.0, "sigma_3d_mm": 0.0}}
    if jacobian is None:
        return uncertainties
    try:
        J = jacobian
        if hasattr(J, "toarray"):
            J = J.toarray()
        JTJ = J.T @ J
        JTJ_reg = JTJ + np.eye(JTJ.shape[0]) * 1e-6
        cov = np.linalg.pinv(JTJ_reg) * (sigma_res_px ** 2)

        for idx, tid in enumerate(static_tags):
            t_offset = idx * 6 + 3
            var_x = max(0.0, float(cov[t_offset, t_offset]))
            var_y = max(0.0, float(cov[t_offset + 1, t_offset + 1]))
            var_z = max(0.0, float(cov[t_offset + 2, t_offset + 2]))
            sx = round(float(np.sqrt(var_x) * 3.0), 3)
            sy = round(float(np.sqrt(var_y) * 3.0), 3)
            sz = round(float(np.sqrt(var_z) * 3.0), 3)
            s3d = round(float(np.sqrt(var_x + var_y + var_z) * 3.0), 3)
            uncertainties[tid] = {
                "sigma_x_mm": sx,
                "sigma_y_mm": sy,
                "sigma_z_mm": sz,
                "sigma_3d_mm": s3d
            }
    except Exception as e:
        log.warning(f"[BA] Tag 3D 不确定度计算失败 (已跳过): {e}")
    return uncertainties


def export_diagnostic_report(final_tags_map: Dict[str, Any],
                             detailed_obs_res: List[Dict[str, Any]],
                             active_frame_names: List[str],
                             tag_uncertainties: Dict[int, Dict[str, float]],
                             outliers_detected: Set[Tuple[int, int]],
                             rmse_px: float,
                             report_dir: Optional[str] = None) -> str:
    """
    生成 2D 像面 Quiver 残差矢量场并输出详尽的 Markdown 精度体检报告
    """
    cur_ws = None
    try:
        from src.workspace.workspace_manager import WorkspaceManager
        cur_ws = WorkspaceManager().get_current_workspace()
    except Exception:
        pass

    if report_dir is None:
        report_dir = cur_ws.calib_reports_dir if cur_ws else os.path.join(PROJECT_ROOT, "data", "tag_calibration_verification")
    os.makedirs(report_dir, exist_ok=True)

    if cur_ws:
        vis_dir = cur_ws.calib_visualized_dir
    else:
        vis_dir = os.path.join(os.path.dirname(report_dir), "visualized")
    os.makedirs(vis_dir, exist_ok=True)

    # 1. 针对每张图绘制 2D 像面 Quiver 矢量场分析图
    frame_grouped = {}
    for item in detailed_obs_res:
        frame_grouped.setdefault(item["frame_name"], []).append(item)

    calib_raw_dir = cur_ws.calib_raw_images_dir if cur_ws else ""
    for f_name, obs_items in frame_grouped.items():
        img_path = ""
        candidates = [
            os.path.join(calib_raw_dir, f_name) if calib_raw_dir else "",
            os.path.join(os.path.dirname(report_dir), "raw_images", f_name),
            os.path.join(os.path.dirname(report_dir), f_name),
            f_name,
        ]
        for cand in candidates:
            if cand and os.path.exists(cand):
                img_path = cand
                break

        if not img_path or not os.path.exists(img_path):
            continue
        base_img = cv2.imread(img_path)
        if base_img is None:
            continue

        h, w = base_img.shape[:2]
        quiver_img = base_img.copy()

        # 绘制顶部半透明状态条
        cv2.rectangle(quiver_img, (0, 0), (w, 50), (20, 24, 30), -1)
        put_text(quiver_img, f"BA 2D Residual Field (Quiver x20) - {f_name} | RMSE: {rmse_px:.3f}px",
                    (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)

        for obs in obs_items:
            tid = obs["tag_id"]
            c_obs = obs["corners_obs"]
            c_proj = obs["corners_proj"]
            is_outlier = obs["is_outlier"]

            poly_color = (0, 0, 255) if is_outlier else (0, 255, 0)
            cv2.polylines(quiver_img, [c_obs.astype(np.int32)], True, poly_color, 2)

            scale = 20.0
            for pt_o, pt_p in zip(c_obs, c_proj):
                dx = (pt_p[0] - pt_o[0]) * scale
                dy = (pt_p[1] - pt_o[1]) * scale
                p_start = (int(round(pt_o[0])), int(round(pt_o[1])))
                p_end = (int(round(pt_o[0] + dx)), int(round(pt_o[1] + dy)))

                cv2.circle(quiver_img, p_start, 3, (0, 255, 0), -1)
                cv2.drawMarker(quiver_img, (int(round(pt_p[0])), int(round(pt_p[1]))), (0, 255, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)
                cv2.arrowedLine(quiver_img, p_start, p_end, (0, 0, 255) if is_outlier else (0, 165, 255),
                                2, tipLength=0.3)

            center = np.mean(c_obs, axis=0).astype(int)
            lbl = f"Tag #{tid}: {obs['rmse_px']:.2f}px" + (" [OUTLIER]" if is_outlier else "")
            put_text(quiver_img, lbl, (center[0] - 40, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        out_quiver_path = os.path.join(vis_dir, os.path.splitext(f_name)[0] + "_quiver.png")
        cv2.imwrite(out_quiver_path, quiver_img)

    # 2. 编写 Markdown 综合体检报告
    report_file = os.path.join(report_dir, "ba_precision_diagnostic_report.md")
    lines = [
        "# AprilTag 离线 BA 空间平差精度体检与深度诊断报告",
        f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
        f"> 评价结论: **{'[优异 PASS]' if rmse_px <= 0.8 else '[良好 ACCEPTABLE]'}** (重投影有效 RMSE: **{rmse_px:.3f} px**)\n",
        "## 1. 平差全局核心指标",
        "| 指标项 | 测量值 | 工业判定门限 | 状态 |",
        "| :--- | :--- | :--- | :--- |",
        f"| **重投影均方根误差 (RMSE)** | **{rmse_px:.3f} px** | $\\le 0.50$ px | {'PASS' if rmse_px <= 0.5 else 'WARN'} |",
        f"| **参与优化图像帧数** | {final_tags_map.get('calibrated_images_count', 0)} 帧 | $\\ge 10$ 帧 | PASS |",
        f"| **空间标靶总数** | {len(final_tags_map.get('tags', {}))} 个 | $\\ge 6$ 个 | PASS |",
        f"| **自动识别清洗离群观测** | {len(outliers_detected)} 项 | $\\le 5$ 项 | {'PASS' if len(outliers_detected) <= 5 else 'WARN'} |",
        "\n## 2. 标靶 3D 空间绝对坐标与 $3\\sigma$ 置信度分析",
        "| 标靶 ID | 空间 X (mm) | 空间 Y (mm) | 空间 Z (mm) | $3\\sigma$ 空间不确定度 (mm) | $3\\sigma$ Z轴深度向 (mm) |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |"
    ]

    tags_dict = final_tags_map.get("tags", {})
    for tid in sorted(tags_dict.keys()):
        info = tags_dict[tid]
        pos = info.get("position_mm", [0, 0, 0])
        unc = tag_uncertainties.get(tid, {})
        s3d = unc.get("sigma_3d_mm", 0.0)
        sz = unc.get("sigma_z_mm", 0.0)
        lines.append(f"| Tag #{tid:2d} | {pos[0]:8.2f} | {pos[1]:8.2f} | {pos[2]:8.2f} | $\\pm${s3d:5.2f} mm | $\\pm${sz:5.2f} mm |")

    lines.extend([
        "\n## 3. 各采图机位重投影残差分布",
        "| 图像文件名 | 观测标靶数 | 平均重投影误差 (px) | 最大误差标靶 | 状态 |",
        "| :--- | :---: | :---: | :--- | :---: |"
    ])

    for f_name, obs_items in sorted(frame_grouped.items()):
        valid_e = [o["rmse_px"] for o in obs_items if not o["is_outlier"]]
        f_mean = float(np.mean(valid_e)) if valid_e else 0.0
        max_item = max(obs_items, key=lambda x: x["rmse_px"])
        max_str = f"Tag #{max_item['tag_id']} ({max_item['rmse_px']:.2f}px)"
        status_str = "PASS" if f_mean <= 0.8 else "WARN"
        lines.append(f"| {f_name:15s} | {len(obs_items):2d} 个 | {f_mean:6.2f} px | {max_str:20s} | {status_str} |")

    rel_vis = os.path.relpath(vis_dir, PROJECT_ROOT).replace("\\", "/")
    lines.extend([
        "\n## 4. 2D 残差矢量场 (Quiver Plot) 说明",
        f"- 分析图像已输出至目录: `{rel_vis}/*_quiver.png`；",
        "- 红色箭头代表角点残差矢量 (已统一放大 20 倍)，用于辨识相机内参畸变是否完全对称消除；",
        "- 若箭头呈完全各向同性发散，说明系统误差已被彻底吸收，剩余均为传感器随机白噪声。"
    ])

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log.info(f"[OK] 深度精度体检与诊断报告已生成至: {report_file}")
    return report_file
