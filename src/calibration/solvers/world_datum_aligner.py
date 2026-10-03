#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D 空间世界基准系对齐器 (World Datum Aligner)

专职负责将纯视觉 BA 平差产出的相对几何底图 (Relative Base Map) 对齐至物理/机械臂世界坐标系。

核心职责:
1. 锚点配置归一化与 DoF 自由度记账 (normalize_anchor_tags / evaluate_anchor_dof)
2. 锚点刚体几何形变与录入冲突校验 (conflict_pairs 守门)
3. Umeyama 3D 闭式解析相似变换解算 (>=3枚全知锚点) 与 Planar 2D 降级解算
4. 逐锚点物理残差质检报告构建 (alignment_report)
5. 生产地图装配 (tags_map.yaml)

【架构说明】:
本模块严格独立于 2D 像面像素重投影方程、相机内参及 Ceres/LM 迭代优化器。
作为两阶段平差中的【阶段二 (Phase 2)】独立执行，毫秒级响应。
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
from scipy.optimize import least_squares

from src.utils.logger import get_logger
from src.workspace.coordinate_manager import rot_mat_to_rpy_deg

log = get_logger(__name__)

# 锚点几何一致性校验与严重冲突告警门限
ANCHOR_CONFLICT_DIFF_WARN_MM = 20.0       # 绝对测距偏差告警门限 (mm)
ANCHOR_CONFLICT_REL_ERROR_WARN = 0.20     # 相对测距偏差告警门限 (20%)
ANCHOR_COLLINEAR_RATIO_THRESH = 0.05      # 锚点点云共线退化判定奇异值比门限 (s1/s0)


def format_conflict_pairs_report(conflict_pairs: List[Dict[str, Any]]) -> str:
    """
    将锚点几何冲突转化为易读的结构化排查诊断文本与线索分析 (向用户明示标称距离与实测距离的超差)
    """
    if not conflict_pairs:
        return ""
    lines = ["⚠️ 【锚点几何严重冲突 (标称距离 vs 视觉实测测距超差)】:"]
    tag_freq: Dict[int, int] = {}
    for c in conflict_pairs:
        pair = c.get("pair", (0, 0))
        t1, t2 = pair[0], pair[1]
        tag_freq[t1] = tag_freq.get(t1, 0) + 1
        tag_freq[t2] = tag_freq.get(t2, 0) + 1
        dw = c.get("world_dist_mm", 0.0)
        dm = c.get("measured_dist_mm", 0.0)
        diff = c.get("diff_mm", 0.0)
        rel_err = c.get("rel_error", 0.0)
        lines.append(
            f"  • Tag #{t1} ⇋ Tag #{t2}: "
            f"标称世界距离 {dw:.1f} mm, 视觉重构测距 {dm:.1f} mm "
            f"(偏差 {diff:.1f} mm, 相对误差 {rel_err*100.0:.1f}%)"
        )

    # 智能线索分析：找出冲突频次最高的标靶 ID
    sorted_tags = sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)
    if sorted_tags:
        max_cnt = sorted_tags[0][1]
        suspects = [f"Tag #{t} (涉及 {cnt} 组冲突)" for t, cnt in sorted_tags if cnt == max_cnt]
        lines.append("")
        lines.append("🔍 【智能纠错线索分析】:")
        lines.append(f"  • 高疑故障源: {', '.join(suspects)}")
        lines.append("  • 建议排查方向: 请优先核验上述高疑标靶在工位 tag_whitelist.yaml 中的世界坐标录入，或检查现场标靶物理张贴间距是否与图纸存在严重偏差。")

    return "\n".join(lines)


class WorldDatumAligner:
    """3D 空间世界坐标系基准对齐器"""

    def __init__(self, marker_size_mm: float = 50.0):
        self.marker_size_mm: float = float(marker_size_mm)

    @staticmethod
    def normalize_anchor_tags(anchor_input: Any) -> Optional[Dict[int, Dict[str, Any]]]:
        """
        锚点配置格式标准化 (严格遵循单一真理源 SSOT 规范):
        - 格式: {tag_id: {"xyz_mm": [x, y, z], "known": [b, b, b]}} (known 缺省视为三轴全知)
        :return: 归一化锚点字典; 无有效锚点返回 None
        """
        if not anchor_input or not isinstance(anchor_input, dict):
            return None
        out: Dict[int, Dict[str, Any]] = {}

        def _add(tid: Any, xyz: Any, known: Any, fid: Optional[str] = None) -> None:
            try:
                tid_i = int(tid)
            except (TypeError, ValueError):
                return
            if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
                return
            xyz_f = []
            known_from_xyz = []
            for v in xyz:
                if v is None or (isinstance(v, str) and v.strip().lower() in ("null", "none", "~", "nan", ".nan")):
                    xyz_f.append(0.0)
                    known_from_xyz.append(False)
                else:
                    try:
                        xyz_f.append(float(v))
                        known_from_xyz.append(True)
                    except (TypeError, ValueError):
                        return
            if isinstance(known, (list, tuple)) and len(known) == 3:
                known_b = [bool(known_from_xyz[i] and known[i]) for i in range(3)]
            else:
                known_b = known_from_xyz
            if not any(known_b):
                return
            entry = {"xyz_mm": xyz_f, "known": known_b}
            if fid:
                entry["frame_id"] = str(fid)
            out[tid_i] = entry

        for k, v in anchor_input.items():
            if isinstance(v, dict):
                xyz = v.get("xyz_mm")
                _add(k, xyz, v.get("known"), v.get("frame_id"))
        return out or None

    @staticmethod
    def evaluate_anchor_dof(anchor_tags: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        """
        配置级自由度记账 (UI 实时状态与求解器共用, 与当帧实际检出无关)。
        重力先验 (BA 系 Z 轴指向天) 固定 roll/pitch 后剩 5 DoF:
        - 尺度 s: 存在共同已知轴且距离非零的锚点对 (中位数聚合)
        - 偏航 yaw: 存在共同已知 XY 的锚点对 (连线方向)
        - t_x / t_y / t_z: 任一锚点已知对应轴
        :return: {"mode": full|partial|none, "dof_solved": 0~5, "dof_total": 5,
                  "scale_pairs": [(ia, ib)...], "yaw_pairs": [...],
                  "t_axes": [has_x, has_y, has_z], "reason": str}
        """
        tids = sorted(anchor_tags.keys())
        scale_pairs: List[Tuple[int, int]] = []
        yaw_pairs: List[Tuple[int, int]] = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                a, b = anchor_tags[tids[i]], anchor_tags[tids[j]]
                common = [k for k in range(3) if a["known"][k] and b["known"][k]]
                if not common:
                    continue
                d_w = math.sqrt(sum((a["xyz_mm"][k] - b["xyz_mm"][k]) ** 2 for k in common))
                if d_w < 1e-6:
                    continue  # 共同已知轴上重合, 该对无尺度信息
                scale_pairs.append((tids[i], tids[j]))
                if all(a["known"][k] and b["known"][k] for k in (0, 1)):
                    d_xy = math.hypot(a["xyz_mm"][0] - b["xyz_mm"][0], a["xyz_mm"][1] - b["xyz_mm"][1])
                    if d_xy >= 1e-6:
                        yaw_pairs.append((tids[i], tids[j]))
        has_x = any(a["known"][0] for a in anchor_tags.values())
        has_y = any(a["known"][1] for a in anchor_tags.values())
        has_z = any(a["known"][2] for a in anchor_tags.values())
        dof = (1 if scale_pairs else 0) + (1 if yaw_pairs else 0) + int(has_x) + int(has_y) + int(has_z)

        if scale_pairs and yaw_pairs and has_x and has_y:
            mode = "full" if has_z else "partial"
        else:
            mode = "none"
        reason = ""
        if mode == "none":
            if not anchor_tags:
                reason = "未配置任何世界锚点"
            elif not scale_pairs:
                reason = "无共同已知轴且距离非零的锚点对, 尺度不可解 (不允许打印边长兜底)"
            elif not yaw_pairs:
                reason = "无共同已知 XY 的锚点对, 偏航不可解"
            else:
                reason = "X/Y 平移约束不足 (需至少一枚已知 X 与一枚已知 Y 的锚点)"
        return {"mode": mode, "dof_solved": dof, "dof_total": 5,
                "scale_pairs": scale_pairs, "yaw_pairs": yaw_pairs,
                "t_axes": [has_x, has_y, has_z], "reason": reason}

    @staticmethod
    def umeyama_alignment(src_pts: np.ndarray, dst_pts: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Umeyama 算法求解最小二乘 3D 相似变换: dst = s * (src @ R.T) + t
        :param src_pts: Nx3 (BA 坐标点云)
        :param dst_pts: Nx3 (世界目标坐标真值)
        :return: (s, R, t) 其中 R in SO(3), det(R) = +1
        """
        n, m = src_pts.shape
        mu_src = np.mean(src_pts, axis=0)
        mu_dst = np.mean(dst_pts, axis=0)

        src_centered = src_pts - mu_src
        dst_centered = dst_pts - mu_dst

        var_src = np.mean(np.sum(src_centered ** 2, axis=1))
        if var_src < 1e-9:
            return 1.0, np.eye(3), mu_dst - mu_src

        H = (dst_centered.T @ src_centered) / n
        U, D, Vt = np.linalg.svd(H)
        S = np.eye(m)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[m - 1, m - 1] = -1

        R = U @ S @ Vt
        s = float((1.0 / var_src) * np.trace(np.diag(D) @ S))
        t = mu_dst - s * (R @ mu_src)
        return s, R, t

    @classmethod
    def solve_similarity_from_anchors(cls,
                                      tag_poses: Dict[int, np.ndarray],
                                      anchor_tags: Dict[int, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
        """
        自适应求解 BA 系 -> 世界系相似变换 (世界坐标系绝对优先):
        - 优先分支 (Umeyama 3D): 当存在 >=3 枚三轴全知且非共线锚点时, 采用闭式解析 Umeyama 算法
          求解全局最优 3D 刚体旋转 R in SO(3) 与尺度/平移, 彻底解除世界系法向对单个基准 Tag
          自身平贴倾角的绑架, 使得世界坐标系严格以用户标定的 3D 地面真值为绝对基准!
        - 调平分支 (Planar Leveled): 当三轴全知锚点不足但存在 >=3 枚已知 Z 且共面的锚点时，
          通过 SVD 拟合工作台法向量并进行空间调平旋转 (消除单目平差基准标靶贴纸法向微小倾角被大跨度
          杠杆放大的 Z 轴系统误差)，再在调平水平面求解偏航角 yaw 并逐轴平移.
        - 纯 2D 分支 (Planar 2D): 约束进一步不足时，以共同轴测距求解尺度, 偏航角 yaw 并独立平移各轴.
        - 锚点一致性守门: 对所有锚点对的世界几何距离与相机重构距离进行相对形变校验, 发现严重录入
          冲突时记录警告, 杜绝错误几何污染全图.
        :return: (mode, info); mode in full|partial|none
        """
        usable = {tid: a for tid, a in anchor_tags.items() if tid in tag_poses}
        missing = sorted(tid for tid in anchor_tags if tid not in tag_poses)
        if missing:
            log.warning(f"[ANCHOR] 配置的世界锚点标靶未参与本次平差解算 (已忽略其约束): Tag {missing}")
        if not usable:
            return "none", {"reason": "配置的世界锚点标靶均未参与本次平差解算"}

        p_ba = {tid: tag_poses[tid][:3, 3].astype(np.float64) for tid in usable}
        tids = sorted(usable.keys())

        # ① 收集全部可用锚点对的真实几何尺度观测
        # 旋转不变性原则: 只有三轴全知 (3D 空间直线欧氏距离) 或水平面内 XY 两轴全知才能作为旋转不变量!
        # 单轴已知 (如仅已知 Y 轴距离) 在未知旋转与未知偏航角下，BA 系投影与世界系投影不具有等价性，严禁直接相除或作为 3D 测距冲突
        scale_obs: List[Tuple[int, int, float, float, float]] = []
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                ia, ib = tids[i], tids[j]
                a, b = usable[ia], usable[ib]
                common = [k for k in range(3) if a["known"][k] and b["known"][k]]

                if len(common) == 3:
                    # 严格 3D 空间直线几何距离 (任意 3D 旋转下严格保持不变)
                    d_w = math.sqrt(sum((a["xyz_mm"][k] - b["xyz_mm"][k]) ** 2 for k in range(3)))
                    d_b = math.sqrt(sum((p_ba[ia][k] - p_ba[ib][k]) ** 2 for k in range(3)))
                elif 0 in common and 1 in common:
                    # 水平面 2D 几何距离 (在重力水平面先验约束下保持不变)
                    d_w = math.hypot(a["xyz_mm"][0] - b["xyz_mm"][0], a["xyz_mm"][1] - b["xyz_mm"][1])
                    d_b = math.hypot(p_ba[ia][0] - p_ba[ib][0], p_ba[ia][1] - p_ba[ib][1])
                else:
                    # 单轴已知点对 (如仅已知 Y 轴分量): 无法作为旋转不变量计算 3D 空间欧氏测距，跳过
                    continue

                if d_w < 1e-6 or d_b < 1e-6:
                    continue  # 重合点无尺度信息
                scale_obs.append((ia, ib, d_w / d_b, d_w, d_b))

        if not scale_obs:
            return "none", {"reason": "无任何锚点对具备三维空间 (XYZ) 或水平面 (XY) 确定几何距离 (单轴已知标靶在姿态未定时不可测距)"}

        ratios = [r[2] for r in scale_obs]
        scale_median = float(np.median(ratios))

        # 锚点几何形变与录入冲突校验
        conflict_pairs = []
        for ia, ib, ratio, d_w, d_b in scale_obs:
            expected_dw = scale_median * d_b
            abs_diff = abs(d_w - expected_dw)
            rel_diff = abs_diff / max(1e-3, expected_dw)
            if abs_diff > 15.0 and rel_diff > 0.15:
                conflict_pairs.append({
                    "pair": (int(ia), int(ib)),
                    "world_dist_mm": round(d_w, 2),
                    "measured_dist_mm": round(expected_dw, 2),
                    "diff_mm": round(abs_diff, 2),
                    "rel_error": round(rel_diff, 3)
                })
                log.warning(
                    f"[ANCHOR CONFLICT] 锚点几何严重冲突! Tag #{ia} 与 Tag #{ib} 间输入的世界距离为 {d_w:.1f} mm, "
                    f"但相机视觉重构等效距离约为 {expected_dw:.1f} mm (偏差 {abs_diff:.1f} mm, 相对误差 {rel_diff*100:.1f}%)! "
                    f"请务必核对工位锚点/白名单中这两个标靶的已知世界坐标输入!"
                )

        # ② 检查是否有 >=3 枚三轴全知锚点且在 3D 空间有效非共线 -> 优先使用 Umeyama 3D 相似变换
        full_3d_tids = [tid for tid in tids if all(usable[tid]["known"])]
        use_umeyama = False
        if len(full_3d_tids) >= 3:
            src_test = np.array([p_ba[tid] for tid in full_3d_tids])
            src_c = src_test - np.mean(src_test, axis=0)
            _, sv_src, _ = np.linalg.svd(src_c)
            # 有效空间秩 >= 2 (非单一退化直线，采用相对比值与绝对容差双重约束)
            if len(sv_src) >= 2 and (sv_src[1] > 20.0 or (sv_src[0] > 1e-4 and sv_src[1] / sv_src[0] > ANCHOR_COLLINEAR_RATIO_THRESH)):
                use_umeyama = True

        if use_umeyama:
            src = np.array([p_ba[tid] for tid in full_3d_tids])
            dst = np.array([usable[tid]["xyz_mm"] for tid in full_3d_tids])
            scale_init, R_init, t_init = cls.umeyama_alignment(src, dst)
            mode = "full"
            yaw_pairs_count = len(full_3d_tids) * (len(full_3d_tids) - 1) // 2
            solver_type = "umeyama_3d"
        else:
            # 降级分支: 优先利用多锚点空间法向调平 (planar_leveled)，若约束不足退化为纯 2D 偏航 (planar_2d)
            z_tids = [tid for tid in tids if usable[tid]["known"][2]]
            use_leveling = False
            R_level = np.eye(3, dtype=np.float64)

            if len(z_tids) >= 3:
                # 检查这些已知 Z 标靶在空间上的散布 (必须满足二维非共线，以唯一确定法向)
                z_pts = np.array([p_ba[tid] for tid in z_tids])
                z_c = z_pts - np.mean(z_pts, axis=0)
                _, sv_z, vh_z = np.linalg.svd(z_c)
                if len(sv_z) >= 2 and (sv_z[1] > 20.0 or (sv_z[0] > 1e-4 and sv_z[1] / sv_z[0] > ANCHOR_COLLINEAR_RATIO_THRESH)):
                    z_targets = np.array([usable[tid]["xyz_mm"][2] for tid in z_tids], dtype=np.float64)
                    # 通用调平法向拟合：支持等高平面以及不同设计高度的台面
                    if (np.max(z_targets) - np.min(z_targets)) <= 2.0:
                        normal = vh_z[2].copy()
                    else:
                        # 不同高度锚点：求解中心化平面高程投影
                        target_c = (z_targets - np.mean(z_targets)) / max(1e-4, scale_median)
                        n_est, _, _, _ = np.linalg.lstsq(z_c, target_c, rcond=None)
                        norm_val = np.linalg.norm(n_est)
                        normal = n_est / norm_val if norm_val > 1e-6 else np.array([0.0, 0.0, 1.0])

                    if normal[2] < 0:
                        normal = -normal
                    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    v_rot = np.cross(normal, z_axis)
                    s_rot = float(np.linalg.norm(v_rot))
                    c_rot = float(np.dot(normal, z_axis))
                    if s_rot > 1e-6:
                        vx = np.array([
                            [0.0, -v_rot[2], v_rot[1]],
                            [v_rot[2], 0.0, -v_rot[0]],
                            [-v_rot[1], v_rot[0], 0.0]
                        ], dtype=np.float64)
                        R_level = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c_rot) / (s_rot ** 2))
                    use_leveling = True
                    tilt_deg = math.degrees(math.acos(np.clip(normal[2], -1.0, 1.0)))
                    log.info(
                        f"[ANCHOR] 激活水平面法向调平先验 (Planar Leveling, {len(z_tids)} 枚已知 Z 锚点): "
                        f"校正法向倾角 {tilt_deg:.2f}°"
                    )

            # 偏航角计算: 在调平系下基于共同已知 XY 的锚点对进行加权闭式解算
            p_for_yaw = {tid: R_level @ p_ba[tid] for tid in tids}
            sum_cross = 0.0
            sum_dot = 0.0
            yaw_obs_count = 0
            for i in range(len(tids)):
                for j in range(i + 1, len(tids)):
                    ia, ib = tids[i], tids[j]
                    a, b = usable[ia], usable[ib]
                    if all(a["known"][k] and b["known"][k] for k in (0, 1)):
                        wdx, wdy = a["xyz_mm"][0] - b["xyz_mm"][0], a["xyz_mm"][1] - b["xyz_mm"][1]
                        bdx, bdy = p_for_yaw[ia][0] - p_for_yaw[ib][0], p_for_yaw[ia][1] - p_for_yaw[ib][1]
                        if math.hypot(wdx, wdy) >= 1e-6 and math.hypot(bdx, bdy) >= 1e-6:
                            sum_cross += (bdx * wdy - bdy * wdx)
                            sum_dot += (bdx * wdx + bdy * wdy)
                            yaw_obs_count += 1

            if yaw_obs_count == 0:
                return "none", {"reason": "无任何共同已知 XY 的锚点对, 偏航不可解", "conflict_pairs": conflict_pairs}

            scale_init = scale_median
            yaw = math.atan2(sum_cross, sum_dot)
            cos_y, sin_y = math.cos(yaw), math.sin(yaw)
            R_yaw = np.array([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
            R_init = R_yaw @ R_level

            # 平移初值: 逐轴对已知锚点残差取均值
            t_axes_init: List[Optional[float]] = []
            for axis in range(3):
                vals = [a["xyz_mm"][axis] - scale_init * float((R_init @ p_ba[tid])[axis])
                        for tid, a in usable.items() if a["known"][axis]]
                t_axes_init.append(float(np.mean(vals)) if vals else None)
            if t_axes_init[0] is None or t_axes_init[1] is None:
                return "none", {"reason": "X/Y 平移约束不足 (需至少一枚已知 X 与一枚已知 Y 的锚点)"}
            t_init = np.array([v if v is not None else 0.0 for v in t_axes_init], dtype=np.float64)
            mode = "full" if t_axes_init[2] is not None else "partial"
            yaw_pairs_count = yaw_obs_count
            solver_type = "planar_leveled" if use_leveling else "planar_2d"

        # 联合 7-DoF 非线性最小二乘精修 (Unified 7-DoF Masked LSQ)
        # 将尺度 s, 旋转 R, 平移 t 在已知轴掩码下联合优化，彻底消除 Z 轴斜坡误差与各步割裂误差
        try:
            rvec_init, _ = cv2.Rodrigues(R_init)
            x0 = np.hstack([rvec_init.flatten(), t_init.flatten(), np.log(max(1e-4, scale_init))])

            def _unified_residuals(params: np.ndarray) -> np.ndarray:
                rv = params[:3]
                tv = params[3:6]
                s_val = math.exp(params[6])
                R_mat, _ = cv2.Rodrigues(rv)
                res = []
                for tid, a in usable.items():
                    p_w_est = s_val * (R_mat @ p_ba[tid]) + tv
                    for k in range(3):
                        if a["known"][k]:
                            res.append(p_w_est[k] - a["xyz_mm"][k])
                # 若无充分 Z 约束，加入极弱水平先验正则项防止 roll/pitch 漂移
                res.append(0.005 * rv[0])
                res.append(0.005 * rv[1])
                return np.array(res, dtype=np.float64)

            opt_res = least_squares(_unified_residuals, x0, method="trf", ftol=1e-5, xtol=1e-5, max_nfev=50)
            if opt_res.success:
                rvec_opt = opt_res.x[:3]
                t = opt_res.x[3:6]
                scale = float(math.exp(opt_res.x[6]))
                R, _ = cv2.Rodrigues(rvec_opt)
                yaw = math.atan2(R[1, 0], R[0, 0])
            else:
                scale, R, t = scale_init, R_init, t_init
        except Exception as e:
            log.warning(f"[ANCHOR] 7-DoF 联合最小二乘精修未收敛，保留闭式初值: {e}")
            scale, R, t = scale_init, R_init, t_init

        # ④ 残差: 各锚点已知轴的变换后偏差 (锚定质量指标与质检单)
        per_tag: Dict[str, List[float]] = {}
        all_res: List[float] = []
        report_rows: List[Dict[str, Any]] = []
        WARN_DIST_THRESHOLD_MM = 3.0  # 超过 3.0mm 即视为质检黄色告警 (标靶坐标疑似录入偏差)

        for tid, a in usable.items():
            p_w = scale * (R @ p_ba[tid]) + t
            target_xyz = [float(a["xyz_mm"][k]) if a["known"][k] else None for k in range(3)]
            fitted_xyz = [round(float(p_w[k]), 2) for k in range(3)]

            delta_xyz = []
            for k in range(3):
                if a["known"][k]:
                    delta_xyz.append(round(float(p_w[k] - a["xyz_mm"][k]), 2))
                else:
                    delta_xyz.append(None)

            known_diffs = [p_w[k] - a["xyz_mm"][k] for k in range(3) if a["known"][k]]
            dist_3d = math.sqrt(sum(d ** 2 for d in known_diffs)) if known_diffs else 0.0
            dist_3d = round(dist_3d, 2)

            res = [float(p_w[k] - a["xyz_mm"][k]) for k in range(3) if a["known"][k]]
            if res:
                per_tag[str(tid)] = [round(v, 3) for v in res]
                all_res.extend(res)

            is_warn = (dist_3d > WARN_DIST_THRESHOLD_MM)
            report_rows.append({
                "tag_id": int(tid),
                "target_xyz": target_xyz,
                "fitted_xyz": fitted_xyz,
                "delta_xyz": delta_xyz,
                "dist_3d_mm": dist_3d,
                "is_warn": is_warn,
                "status_str": "⚠️ 异常过大" if is_warn else "🟢 吻合"
            })

        mean_mm = round(float(np.mean(np.abs(all_res))), 3) if all_res else 0.0
        max_mm = round(float(np.max(np.abs(all_res))), 3) if all_res else 0.0
        has_warn = any(r["is_warn"] for r in report_rows) or bool(conflict_pairs)

        alignment_report = {
            "solver_type": solver_type,
            "mean_mm": mean_mm,
            "max_mm": max_mm,
            "has_warn": has_warn,
            "warn_threshold_mm": WARN_DIST_THRESHOLD_MM,
            "rows": report_rows,
            "conflict_pairs": conflict_pairs
        }

        residuals = {
            "mean_mm": mean_mm,
            "max_mm": max_mm,
            "per_tag": per_tag,
            "has_warn": has_warn,
            "report_rows": report_rows
        }
        info = {
            "scale_factor": scale, "yaw_rad": yaw, "R": R, "t": t,
            "scale_pair_count": len(scale_obs), "yaw_pair_count": yaw_pairs_count,
            "residuals": residuals,
            "solver_type": solver_type,
            "conflict_pairs": conflict_pairs,
            "alignment_report": alignment_report
        }
        return mode, info

    def _anchor_fallback_relative(self,
                                  tag_poses: Dict[int, np.ndarray],
                                  origin_tag_id: int,
                                  x_align_tag_id: int,
                                  reason: str) -> Dict[str, Any]:
        """锚定不可行时的统一退化出口: 相对对齐 + none 模式标记"""
        log.warning(f"[ANCHOR] 世界锚定不可行: {reason} — 退化为相对对齐！")
        rel = self.align_to_scara_world(tag_poses, origin_tag_id, x_align_tag_id)
        rel["anchor_mode"] = "none"
        rel["anchor_skip_reason"] = reason
        return rel

    def anchor_to_absolute_world(self,
                                 tag_poses: Dict[int, np.ndarray],
                                 anchor_input: Any,
                                 origin_tag_id: int = 0,
                                 x_align_tag_id: int = 1,
                                 strict: bool = False) -> Dict[str, Any]:
        """
        FR-9.6 世界坐标系绝对锚定 (约束积累式):
        支持任意数量全知/部分已知锚点 (逐轴 known 标记), 重力先验 (BA 系 Z 轴指向天) 固定 roll/pitch,
        分阶段闭式求解相似变换 (尺度 s + 偏航 yaw + 平移 t) 将整张 BA 平差地图变换到机械臂世界坐标系:
        - full:    5/5 DoF 全部解算, 绝对世界系地图
        - partial: 仅 XY 链可解 (t_z 悬空), XY 绝对锚定 + Z 保持 BA 尺度相对坐标, 下游须按 anchor_mode 守门
        - none:    约束不足或退化, strict=True 抛错终止, strict=False 退化为 align_to_scara_world 相对对齐
        :param anchor_input: 新格式 {tid: {"xyz_mm","known"}} 或旧双锚点格式 (自动归一化)
        :param strict: 是否严格模式 (平差主干默认 True, 禁止任何静默兜底)
        """
        anchor_tags = self.normalize_anchor_tags(anchor_input)
        if not anchor_tags:
            if strict:
                raise ValueError("anchor_to_absolute_world: 锚点配置为空或字段不合法 (FR-9.6 禁止兜底)")
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, "锚点配置为空或字段不合法")

        # 坐标系隔离过滤: 世界基准系 (world) 对齐仅能使用属于 world 的锚点，严禁将未解算的子坐标系局部标靶混入世界系
        # 核心规约: 0~9 恒定归属 world；>=10 归属各子坐标系
        world_anchors = {}
        for tid, a in anchor_tags.items():
            fid = a.get("frame_id")
            if fid == "world":
                world_anchors[tid] = a
            elif fid is not None:
                log.info(f"[WORLD_ALIGN] 标靶 Tag #{tid} 显式指定归属于子坐标系 [{fid}]，已隔离不参与阶段二世界基准系对齐")
            else:
                # 纯净格式: 0 <= tid <= 9 为 world 锚点
                if 0 <= int(tid) <= 9:
                    world_anchors[tid] = a
                else:
                    log.info(f"[WORLD_ALIGN] 标靶 Tag #{tid} 按 ID 区间规约归属于子坐标系 (Tag >= 10)，已隔离不参与阶段二世界基准系对齐")

        # P0-4 修正: 严禁在 world_anchors 为空时静默 fallback 回退到 anchor_tags，必须显式抛错拦截
        if not world_anchors:
            if strict:
                raise ValueError(
                    "anchor_to_absolute_world: 未找到任何归属于世界基准系 (world) 的有效锚点 (Tag 0~9 或 frame_id='world')，"
                    "严禁将子坐标系局部标靶作为世界真值 (FR-9.6 禁止兜底)"
                )
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, "无有效世界锚点")

        active_anchors = world_anchors

        dof = self.evaluate_anchor_dof(active_anchors)
        if dof["mode"] == "none":
            if strict:
                raise ValueError(
                    f"锚点 DoF 约束不足 ({dof['dof_solved']}/5): {dof['reason']}"
                    " — 请增配已知世界坐标的 tag 或放宽当前部分已知标记 (FR-9.6 禁止兜底)"
                )
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, dof["reason"])

        mode, solve = self.solve_similarity_from_anchors(tag_poses, active_anchors)
        if mode == "none":
            if strict:
                conflict_pairs = solve.get("conflict_pairs", [])
                conflict_diag = format_conflict_pairs_report(conflict_pairs)
                conflict_sec = f"\n\n{conflict_diag}" if conflict_diag else ""
                raise ValueError(
                    f"锚点求解退化: {solve['reason']}{conflict_sec}"
                    "\n\n— 请检查锚点 tag 之间的已知轴距离与 XY 共线方向 (FR-9.6 禁止兜底)"
                )
            return self._anchor_fallback_relative(tag_poses, origin_tag_id, x_align_tag_id, solve["reason"])

        if solve.get("conflict_pairs"):
            fatal_conflicts = [
                c for c in solve["conflict_pairs"]
                if c.get("diff_mm", 0.0) > ANCHOR_CONFLICT_DIFF_WARN_MM and c.get("rel_error", 0.0) > ANCHOR_CONFLICT_REL_ERROR_WARN
            ]
            if fatal_conflicts:
                conflict_diag = format_conflict_pairs_report(fatal_conflicts)
                log.error(f"[ANCHOR CONFLICT FATAL]\n{conflict_diag}")

        scale = solve["scale_factor"]
        R = solve["R"]
        t_vec = solve["t"]

        # 真实边长更新 = 名义 × 锚定尺度
        self.marker_size_mm = self.marker_size_mm * scale

        res = solve["residuals"]
        solver_type = solve.get("solver_type", "planar_2d")
        conflict_pairs = solve.get("conflict_pairs", [])
        aligned_map = {
            "origin_tag_id": origin_tag_id,
            "x_axis_align_tag_id": x_align_tag_id,
            "anchor_mode": mode,
            "world_anchor": {
                "anchor_mode": mode,
                "solver_type": solver_type,
                "scale_factor": round(float(scale), 6),
                "yaw_deg": round(float(math.degrees(solve["yaw_rad"])), 3),
                "t_xyz_mm": [round(float(v), 3) for v in t_vec],
                "R_matrix": [[round(float(val), 6) for val in row] for row in R],
                "scale_pair_count": int(solve["scale_pair_count"]),
                "yaw_pair_count": int(solve["yaw_pair_count"]),
                "anchor_residual_mm": res,
                "anchor_tags": {str(tid): {"xyz_mm": [round(float(v), 3) for v in a["xyz_mm"]],
                                           "known": [bool(v) for v in a["known"]]}
                                 for tid, a in sorted(anchor_tags.items())},
                "conflict_pairs": conflict_pairs,
                "real_marker_size_mm": round(float(self.marker_size_mm), 3),
                "alignment_report": solve.get("alignment_report", {})
            },
            "tags": {}
        }
        log.info(f"[+] [ANCHOR] FR-9.6 世界系锚定完成 (mode={mode}, solver={solver_type}): "
              f"尺度因子: {scale:.6f} ({solve['scale_pair_count']} 对), 偏航: {math.degrees(solve['yaw_rad']):.2f}°, "
              f"锚点残差 mean/max: {res['mean_mm']:.2f}/{res['max_mm']:.2f} mm, "
              f"反算真实边长: {self.marker_size_mm:.2f} mm")
        if conflict_pairs:
            log.warning(f"[ANCHOR] ⚠️ 注意：检测到 {len(conflict_pairs)} 组锚点输入存在严重几何形变冲突，可能影响世界对齐精度！请检查锚点录入。")
        if mode == "partial":
            log.warning("[ANCHOR] partial 模式: XY 已绝对锚定, Z 轴保持 BA 尺度相对坐标 (t_z 悬空) — 下游须按 anchor_mode 守门！")

        for t_id, T_w_t in tag_poses.items():
            pos_aligned = scale * (R @ T_w_t[:3, 3]) + t_vec
            R_aligned = R @ T_w_t[:3, :3]
            rpy = rot_mat_to_rpy_deg(R_aligned)

            aligned_map["tags"][t_id] = {
                "position_mm": [round(float(v), 2) for v in pos_aligned],
                "rpy_deg": [round(float(v), 2) for v in rpy],
                "transform_matrix": [[round(float(val), 5) for val in row] for row in np.vstack([np.hstack([R_aligned, pos_aligned.reshape(3, 1)]), [0, 0, 0, 1]])],
                "is_origin": bool(t_id == origin_tag_id),
                "is_dynamic_yaw": bool(t_id == origin_tag_id)
            }

        return aligned_map

    def align_relative_map_to_world(self,
                                    relative_map: Dict[str, Any],
                                    anchor_tags: Dict[int, Dict[str, Any]],
                                    origin_tag_id: int = 0,
                                    x_align_tag_id: int = 1,
                                    strict: bool = True) -> Dict[str, Any]:
        """
        【阶段二独立解算核心入口】将自由平差产出的相对几何底图对齐到世界坐标系
        :param relative_map: 阶段一产出的相对底图 (包含 raw_relative_poses 或 tags 的位姿矩阵)
        :param anchor_tags: 用户录入的世界锚点真值表
        :param origin_tag_id: 原点标靶 ID
        :param x_align_tag_id: X 轴对齐标靶 ID
        :param strict: 是否严格模式 (发现冲突或锚点不足时抛错，否则降级)
        :return: 具有绝对世界坐标的完整 tags_map 生产字典
        """
        # 提取标靶 4x4 位姿矩阵 (优先使用 raw_relative_poses)
        tag_poses: Dict[int, np.ndarray] = {}
        if "raw_relative_poses" in relative_map:
            for tid, mat in relative_map["raw_relative_poses"].items():
                tag_poses[int(tid)] = np.array(mat, dtype=np.float64)
        elif "tags" in relative_map:
            for tid_raw, t_info in relative_map["tags"].items():
                try:
                    tid = int(tid_raw)
                except (ValueError, TypeError):
                    continue
                if isinstance(t_info, dict):
                    if "transform_matrix" in t_info:
                        tag_poses[tid] = np.array(t_info["transform_matrix"], dtype=np.float64)
                    elif "position_mm" in t_info or "center" in t_info:
                        pos = t_info.get("position_mm", t_info.get("center"))
                        if pos is not None and len(pos) >= 3:
                            T = np.eye(4, dtype=np.float64)
                            T[:3, 3] = np.array(pos[:3], dtype=np.float64)
                            tag_poses[tid] = T
                elif isinstance(t_info, np.ndarray) and t_info.shape == (4, 4):
                    tag_poses[tid] = t_info.astype(np.float64)

        if not tag_poses:
            raise ValueError("align_relative_map_to_world: 相对底图中未找到任何有效的标靶位姿矩阵")

        # 同步名义边长
        if "marker_size_mm" in relative_map:
            self.marker_size_mm = float(relative_map["marker_size_mm"])

        aligned = self.anchor_to_absolute_world(
            tag_poses, anchor_tags,
            origin_tag_id=origin_tag_id,
            x_align_tag_id=x_align_tag_id,
            strict=strict
        )

        # 完整继承阶段一的所有平差质检与元数据属性
        result_map = dict(relative_map)
        result_map["tags"] = aligned["tags"]
        result_map["world_anchor"] = aligned.get("world_anchor", {})
        result_map["anchor_mode"] = aligned.get("anchor_mode", "full")
        result_map["marker_size_mm"] = round(float(self.marker_size_mm), 3)
        if "raw_relative_poses" not in result_map:
            result_map["raw_relative_poses"] = {
                int(tid): [[round(float(val), 5) for val in row] for row in T]
                for tid, T in tag_poses.items()
            }
        return result_map

    def align_to_scara_world(self,
                             tag_poses: Dict[int, np.ndarray],
                             origin_tag_id: int,
                             x_align_tag_id: int) -> Dict[str, Any]:
        """
        通过刚体变换将地图整体平移旋转，使得：
        1. Tag 0 的中心处于 (0.0, 0.0)
        2. Tag 0 -> Tag 1 的水平向量严格处于 +X 轴 (Y=0, X>0)
        """
        aligned_map = {
            "origin_tag_id": origin_tag_id,
            "x_axis_align_tag_id": x_align_tag_id,
            "tags": {}
        }

        if origin_tag_id in tag_poses:
            p_origin = tag_poses[origin_tag_id][:3, 3].copy()
        else:
            p_origin = np.zeros(3)
            log.warning(f"[WARN] 未在有效图像中检出 Tag {origin_tag_id}，将以参考标靶相对对齐！")

        yaw_rad = 0.0
        aligned_x_target_id = x_align_tag_id

        if origin_tag_id in tag_poses and x_align_tag_id in tag_poses:
            vec_x = tag_poses[x_align_tag_id][:3, 3] - p_origin
            yaw_rad = math.atan2(vec_x[1], vec_x[0])
            log.info(f"[+] [ALIGN] 成功锚定基准 Tag {origin_tag_id} -> Tag {x_align_tag_id}，坐标系 X 轴对齐旋转角: {-math.degrees(yaw_rad):.2f}°")
        else:
            # 指定对齐标靶缺失，打印显式告警
            available_tags = [tid for tid in tag_poses.keys() if tid != origin_tag_id]
            log.warning(f"[WARN] [ALIGN] 指定的 X 轴对齐标靶 Tag {x_align_tag_id} 不在解算标靶中 (可用静态标靶: {sorted(available_tags)})！")

            # 自适应寻找候选远端标靶（水平距离最大且在有效范围内的标靶）
            if available_tags and origin_tag_id in tag_poses:
                candidate_dists = []
                for tid in available_tags:
                    dist_xy = np.linalg.norm(tag_poses[tid][:2, 3] - p_origin[:2])
                    candidate_dists.append((dist_xy, tid))
                candidate_dists.sort(reverse=True)
                fallback_id = candidate_dists[0][1]
                aligned_x_target_id = fallback_id
                vec_x = tag_poses[fallback_id][:3, 3] - p_origin
                yaw_rad = math.atan2(vec_x[1], vec_x[0])
                log.warning(f"[!] [ALIGN] 自动降级使用最远端刚体标靶 Tag {fallback_id} (距离 {candidate_dists[0][0]:.1f}mm) 进行 X 轴定向校正: {-math.degrees(yaw_rad):.2f}°")
            else:
                log.error(f"[ERROR] [ALIGN] 无法进行世界 X 轴对齐，世界系方向将退化保持为基准标靶印刷朝向！")

        aligned_map["x_axis_align_tag_id"] = aligned_x_target_id

        cos_y = math.cos(-yaw_rad)
        sin_y = math.sin(-yaw_rad)
        R_align = np.array([
            [cos_y, -sin_y, 0.0],
            [sin_y,  cos_y, 0.0],
            [0.0,    0.0,   1.0]
        ], dtype=np.float64)

        for t_id, T_w_t in tag_poses.items():
            pos_rel = T_w_t[:3, 3] - p_origin
            pos_aligned = R_align @ pos_rel
            R_aligned = R_align @ T_w_t[:3, :3]
            rpy = rot_mat_to_rpy_deg(R_aligned)

            aligned_map["tags"][t_id] = {
                "position_mm": [round(float(v), 2) for v in pos_aligned],
                "rpy_deg": [round(float(v), 2) for v in rpy],
                "transform_matrix": [[round(float(val), 5) for val in row] for row in np.vstack([np.hstack([R_aligned, pos_aligned.reshape(3, 1)]), [0, 0, 0, 1]])],
                "is_origin": bool(t_id == origin_tag_id),
                "is_dynamic_yaw": bool(t_id == origin_tag_id)
            }

        return aligned_map

    @staticmethod
    def format_alignment_report_markdown(report: Dict[str, Any]) -> str:
        """格式化质检单报告为 Markdown 文本"""
        if not report:
            return ""
        lines = [
            "# 世界坐标系对齐质检单 (World Datum Alignment Report)",
            f"- **解算算法**: `{report.get('solver_type', 'Umeyama 3D')}`",
            f"- **均值物理残差**: `{report.get('mean_mm', 0.0):.2f} mm`",
            f"- **最大物理残差**: `{report.get('max_mm', 0.0):.2f} mm`",
            f"- **整体质检结论**: {'⚠️ 存在超标标靶 (请核对输入坐标)' if report.get('has_warn') else '🟢 全部标靶优良吻合'}",
            "",
            "| 标靶 ID | 设定世界坐标 (X, Y, Z) mm | 实测对齐坐标 (X, Y, Z) mm | 分轴偏差 (ΔX, ΔY, ΔZ) mm | 3D 绝对残差 | 质检状态 |",
            "| :---: | :--- | :--- | :--- | :---: | :---: |"
        ]
        rows = list(report.get("rows", []))
        for row in rows:
            tag_str = f"Tag #{row['tag_id']}"
            t_vals = [f"{v:.1f}" if v is not None else "--" for v in row.get("target_xyz", [])]
            tgt = f"({', '.join(t_vals)})"
            f_vals = [f"{v:.1f}" if v is not None else "--" for v in row.get("fitted_xyz", [])]
            fit = f"({', '.join(f_vals)})"
            delta = f"({', '.join(str(v) for v in row.get('delta_xyz', []))})"
            dist = f"{row.get('dist_3d_mm', 0.0):.2f} mm"
            stat = "⚠️ 偏差过大" if row.get("is_warn") else "🟢 吻合"
            lines.append(f"| {tag_str} | {tgt} | {fit} | {delta} | {dist} | {stat} |")
        return "\n".join(lines)
