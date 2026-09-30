#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多坐标系分层里程碑解算器 (Multi-Frame Milestone Solver)
===================================================
核心设计哲学:
1. 主干基准层 (Global Backbone):
   - M1 (自由平差底图 Free BA): 纯视觉约束下的刚体空间结构，提供无偏的高精度标靶相对几何。
   - M2 (绝对世界基准 World Datum): 仅由工位静态底座世界锚点 (frame_id == 'world') 决定世界原点与偏航角。
2. 动态子部件层 (Dynamic Sub-Frame Pipelines, 支持 1..N 个并行机构):
   - 阶段 A (局部刚体一致性质检 Local Rigidity Check):
     在自由底图下直接核验机构内部标靶名义间距 vs 视觉实测间距。
     完全解耦世界坐标系，具备独立自检与故障隔离能力。
   - 阶段 B (空间外参反推挂接 Extrinsics Attachment):
     当阶段 A 质检通过且主干世界基准锚定后，闭式求解 T_{world <- F_k} 并固化外参。
     若阶段 A 质检超差，则实施熔断隔离，杜绝错误外参污染坐标系拓扑树。
"""

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

from src.workspace.coordinate_manager import (
    CoordinateTreeManager,
    FrameDefinition,
    rot_mat_to_rpy_deg,
)
from src.calibration.solvers.frame_extrinsic_solver import (
    FrameExtrinsicSolver,
    rigid_transform_3d,
)
from src.calibration.solvers.world_datum_aligner import (
    WorldDatumAligner,
    format_conflict_pairs_report,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class SubFrameMilestoneReport:
    """单个子坐标系的分层质检报告"""
    frame_id: str
    frame_name: str
    parent_frame_id: str = "world"

    # 阶段 A: 局部刚体一致性质检 (Local Rigidity Check)
    local_rigidity_passed: bool = False
    local_matched_tags: List[int] = field(default_factory=list)
    local_tag_pairs: List[Dict[str, Any]] = field(default_factory=list)
    local_conflict_pairs: List[Dict[str, Any]] = field(default_factory=list)
    local_max_err_mm: float = 0.0
    local_mean_err_mm: float = 0.0
    local_rmse_mm: Optional[float] = None
    local_status_msg: str = ""

    # 阶段 B: 空间外参反推挂接 (Extrinsics Attachment)
    extrinsic_solved: bool = False
    is_isolated: bool = False  # 是否触发故障熔断隔离
    translation_xyz_mm: Optional[List[float]] = None
    rotation_rpy_deg: Optional[List[float]] = None
    extrinsic_rmse_mm: Optional[float] = None
    extrinsic_method: str = ""
    extrinsic_status_msg: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "frame_name": self.frame_name,
            "parent_frame_id": self.parent_frame_id,
            "local_rigidity": {
                "passed": self.local_rigidity_passed,
                "matched_tags": self.local_matched_tags,
                "max_err_mm": round(self.local_max_err_mm, 3),
                "mean_err_mm": round(self.local_mean_err_mm, 3),
                "rmse_mm": round(self.local_rmse_mm, 3) if self.local_rmse_mm is not None else None,
                "conflict_count": len(self.local_conflict_pairs),
                "status_msg": self.local_status_msg,
                "pairs": self.local_tag_pairs,
            },
            "extrinsic": {
                "solved": self.extrinsic_solved,
                "isolated": self.is_isolated,
                "translation_xyz_mm": [round(x, 2) for x in self.translation_xyz_mm] if self.translation_xyz_mm else None,
                "rotation_rpy_deg": [round(r, 2) for r in self.rotation_rpy_deg] if self.rotation_rpy_deg else None,
                "rmse_mm": round(self.extrinsic_rmse_mm, 3) if self.extrinsic_rmse_mm is not None else None,
                "method": self.extrinsic_method,
                "status_msg": self.extrinsic_status_msg,
            },
        }


@dataclass
class MultiFrameMilestoneReport:
    """全工位多坐标系分层里程碑总报告"""
    # M1: 全局自由平差底图 (Free BA)
    m1_free_ba_passed: bool = False
    m1_reprojection_rmse_px: float = 0.0
    m1_tag_count: int = 0
    m1_msg: str = ""

    # M2: 绝对世界基准 (World Datum)
    m2_world_datum_passed: bool = False
    m2_solver_type: str = "3D"
    m2_mean_residual_mm: float = 0.0
    m2_max_residual_mm: float = 0.0
    m2_conflict_pairs: List[Dict[str, Any]] = field(default_factory=list)
    m2_msg: str = ""

    # 动态子坐标系列表 (支持 1..N 个并行机构)
    sub_frames: Dict[str, SubFrameMilestoneReport] = field(default_factory=dict)

    # 总体决策与状态
    all_milestones_passed: bool = False
    summary_toast: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "m1_free_ba": {
                "passed": self.m1_free_ba_passed,
                "reprojection_rmse_px": round(self.m1_reprojection_rmse_px, 3),
                "tag_count": self.m1_tag_count,
                "msg": self.m1_msg,
            },
            "m2_world_datum": {
                "passed": self.m2_world_datum_passed,
                "solver_type": self.m2_solver_type,
                "mean_residual_mm": round(self.m2_mean_residual_mm, 3),
                "max_residual_mm": round(self.m2_max_residual_mm, 3),
                "conflict_count": len(self.m2_conflict_pairs),
                "msg": self.m2_msg,
            },
            "sub_frames": {fid: r.to_dict() for fid, r in self.sub_frames.items()},
            "all_passed": self.all_milestones_passed,
            "summary_toast": self.summary_toast,
        }


def extract_tag_positions_from_map(tags_map: Dict[str, Any]) -> Dict[int, np.ndarray]:
    """从 tags_map 提取所有标靶的 3D 中心位置字典 {tag_id: array([x, y, z])}"""
    res: Dict[int, np.ndarray] = {}
    raw_tags = tags_map.get("tags", {}) if isinstance(tags_map, dict) else {}
    for k, v in raw_tags.items():
        try:
            tid = int(k)
        except (ValueError, TypeError):
            continue
        if isinstance(v, dict):
            pos = v.get("position_mm", v.get("center"))
            if pos is not None and len(pos) >= 3:
                res[tid] = np.array(pos[:3], dtype=np.float64)
            elif "transform_matrix" in v:
                T = np.array(v["transform_matrix"], dtype=np.float64)
                if T.shape == (4, 4):
                    res[tid] = T[:3, 3].copy()
        elif isinstance(v, np.ndarray) and v.shape == (4, 4):
            res[tid] = v[:3, 3].copy()
    return res


def check_sub_frame_local_rigidity(
    frame: FrameDefinition,
    relative_tags: Dict[int, np.ndarray],
    dist_tol_mm: float = 3.0,
    dist_tol_ratio: float = 0.015,
    fallback_anchor_tags: Optional[Dict[int, Any]] = None,
    coord_mgr: Optional[CoordinateTreeManager] = None,
) -> SubFrameMilestoneReport:
    """
    【阶段 A 核心】子坐标系局部刚体一致性质检:
    在纯视觉自由底图 (相对系) 下核对该机构内部标靶的图纸标称间距 vs 视觉重构间距。
    完全脱离世界坐标系，验证构件自身是否发生几何形变、贴错标或名义配置错误。
    """
    rep = SubFrameMilestoneReport(
        frame_id=frame.frame_id,
        frame_name=frame.name or frame.frame_id,
        parent_frame_id=frame.parent_frame_id or "world",
    )

    # 1. 收集子坐标系内部的标靶名义坐标表 {tag_id: np.array([x, y, z])}
    nominal_tags: Dict[int, np.ndarray] = {}
    spec = frame.calibration_spec or {}

    # 优先从 calibration_spec["reference_tags"] 提取
    ref_tags = spec.get("reference_tags", {})
    if isinstance(ref_tags, dict):
        for tid_raw, pos in ref_tags.items():
            try:
                tid = int(tid_raw)
                if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 3:
                    nominal_tags[tid] = np.array(pos[:3], dtype=np.float64)
            except (ValueError, TypeError):
                continue

    # 兼容定轴模式 (axis_align)
    if not nominal_tags and spec.get("origin_tag_id") is not None and spec.get("x_axis_tag_id") is not None:
        try:
            o_tid = int(spec["origin_tag_id"])
            x_tid = int(spec["x_axis_tag_id"])
            o_pos = spec.get("origin_local_xyz_mm", [0.0, 0.0, 0.0])
            x_pos = spec.get("x_axis_local_xyz_mm", [100.0, 0.0, 0.0])
            nominal_tags[o_tid] = np.array(o_pos[:3], dtype=np.float64)
            nominal_tags[x_tid] = np.array(x_pos[:3], dtype=np.float64)
        except (ValueError, TypeError):
            pass

    # 备选: 从 fallback_anchor_tags 中提取归属此 frame 的标靶
    # 原则: 优先依据 coord_mgr / 全局 Tag ID 分段映射规约 (如 frame_sub_1 对应 10~19)
    if not nominal_tags and fallback_anchor_tags:
        target_range = None
        if coord_mgr:
            try:
                target_range = set(coord_mgr.get_frame_tag_range(frame.frame_id))
            except Exception:
                pass
        if target_range is None:
            # 根据 frame_id 解析序号，如 frame_sub_1 -> 1 -> range(10, 20)
            parts = frame.frame_id.split("_")
            idx = 1
            for p in parts:
                if p.isdigit():
                    idx = int(p)
                    break
            start = idx * 10
            target_range = set(range(start, start + 10))

        for tid_raw, a_entry in fallback_anchor_tags.items():
            try:
                tid = int(tid_raw)
                fid = a_entry.get("frame_id") if isinstance(a_entry, dict) else None
                # 若显式匹配或落在区间内
                if fid == frame.frame_id or (fid is None and tid in target_range):
                    coords = a_entry.get("xyz_mm") or a_entry.get("coords") or a_entry.get("position_mm") or a_entry.get("center")
                    if coords and len(coords) >= 3 and None not in coords[:3]:
                        nominal_tags[tid] = np.array(coords[:3], dtype=np.float64)
            except (ValueError, TypeError):
                continue

    # 2. 匹配当前相对底图中存在的标靶
    matched_tids = sorted([tid for tid in nominal_tags if tid in relative_tags])
    rep.local_matched_tags = matched_tids

    # 针对单标靶或无标靶配置的边界判定
    if len(nominal_tags) <= 1:
        if len(matched_tids) == 1:
            rep.local_rigidity_passed = True
            rep.local_status_msg = f"单标靶模式 (Tag #{matched_tids[0]} 已检出，无内部间距质检项)"
            return rep
        else:
            rep.local_rigidity_passed = False
            rep.local_status_msg = f"标靶未检出 (需检测标靶: {list(nominal_tags.keys())})"
            return rep

    if len(matched_tids) < 2:
        rep.local_rigidity_passed = False
        rep.local_status_msg = (
            f"有效检出标靶数不足 (已检出 {len(matched_tids)} 个: Tag {matched_tids}，"
            f"名义要求 {len(nominal_tags)} 个: Tag {sorted(nominal_tags.keys())})"
        )
        return rep

    # 3. 计算所有匹配点对的 名义内部距离 vs 视觉重构实测距离
    pairs_info = []
    conflict_pairs = []
    errors = []

    for i in range(len(matched_tids)):
        for j in range(i + 1, len(matched_tids)):
            t1, t2 = matched_tids[i], matched_tids[j]
            p1_nom, p2_nom = nominal_tags[t1], nominal_tags[t2]
            p1_rel, p2_rel = relative_tags[t1], relative_tags[t2]

            d_nominal = float(np.linalg.norm(p1_nom - p2_nom))
            d_measured = float(np.linalg.norm(p1_rel - p2_rel))
            diff = d_measured - d_nominal
            abs_diff = abs(diff)
            rel_err = abs_diff / d_nominal if d_nominal > 1e-6 else 0.0

            # 动态容差判定: 结合绝对误差与相对误差
            # 允许容差为 max(dist_tol_mm, d_nominal * dist_tol_ratio)
            allowed_tol = max(dist_tol_mm, d_nominal * dist_tol_ratio)
            is_conflict = abs_diff > allowed_tol

            pair_record = {
                "pair": [t1, t2],
                "nominal_dist_mm": round(d_nominal, 2),
                "measured_dist_mm": round(d_measured, 2),
                "diff_mm": round(diff, 2),
                "abs_diff_mm": round(abs_diff, 2),
                "rel_error_pct": round(rel_err * 100.0, 2),
                "allowed_tol_mm": round(allowed_tol, 2),
                "is_conflict": is_conflict,
            }
            pairs_info.append(pair_record)
            errors.append(abs_diff)

            if is_conflict:
                conflict_pairs.append({
                    "pair": [t1, t2],
                    "world_dist_mm": round(d_nominal, 2),
                    "measured_dist_mm": round(d_measured, 2),
                    "diff_mm": round(diff, 2),
                    "rel_error": round(rel_err, 4),
                })

    rep.local_tag_pairs = pairs_info
    rep.local_conflict_pairs = conflict_pairs
    rep.local_max_err_mm = float(max(errors)) if errors else 0.0
    rep.local_mean_err_mm = float(np.mean(errors)) if errors else 0.0

    # 4. 若标靶数 >= 3，执行纯相对刚体配准计算自身局部 RMSE
    if len(matched_tids) >= 3:
        try:
            pts_child = np.array([nominal_tags[t] for t in matched_tids], dtype=np.float64)
            pts_rel = np.array([relative_tags[t] for t in matched_tids], dtype=np.float64)
            _, _, local_rmse = rigid_transform_3d(pts_child, pts_rel)
            rep.local_rmse_mm = float(local_rmse)
        except Exception as e:
            log.debug(f"[MultiFrame] 子系 {frame.frame_id} 局部刚体配准残差求解跳过: {e}")

    # 5. 综合判定局部刚体是否合格
    if conflict_pairs:
        rep.local_rigidity_passed = False
        rep.local_status_msg = (
            f"局部尺寸一致性超差！检出 {len(conflict_pairs)} 组测距冲突 "
            f"(最大偏差: {rep.local_max_err_mm:.2f} mm)"
        )
    else:
        rep.local_rigidity_passed = True
        rmse_str = f", 局部 RMSE: {rep.local_rmse_mm:.2f} mm" if rep.local_rmse_mm is not None else ""
        rep.local_status_msg = (
            f"局部尺寸高度吻合 (匹配 {len(matched_tids)} 枚标靶, "
            f"最大偏差: {rep.local_max_err_mm:.2f} mm{rmse_str})"
        )

    return rep


class MultiFrameMilestoneSolver:
    """全工位多坐标系分层里程碑求解器"""

    def __init__(
        self,
        marker_size_mm: float = 80.0,
        dist_tol_mm: float = 3.0,
        dist_tol_ratio: float = 0.015,
    ):
        self.marker_size_mm = marker_size_mm
        self.dist_tol_mm = dist_tol_mm
        self.dist_tol_ratio = dist_tol_ratio

    def solve(
        self,
        relative_map: Dict[str, Any],
        anchor_tags: Dict[int, Any],
        coord_mgr: Optional[CoordinateTreeManager] = None,
        origin_tag_id: Optional[int] = None,
        x_align_tag_id: Optional[int] = None,
        strict_world_datum: bool = True,
    ) -> Tuple[bool, MultiFrameMilestoneReport, Optional[Dict[str, Any]]]:
        """
        执行全流程分层里程碑解算与质检:
        1. 验证 M1 (自由平差底图)
        2. 针对全部 1..N 个子坐标系执行 阶段 A (局部刚体一致性质检)
        3. 执行 M2 (绝对世界基准锚定)
        4. 针对合格子坐标系执行 阶段 B (外参反推与挂接)，超差子系熔断隔离
        """
        report = MultiFrameMilestoneReport()

        # ---------------------------------------------------------
        # 里程碑 M1: 全局纯视觉自由平差底图 (Free BA Base Map)
        # ---------------------------------------------------------
        if not relative_map or not relative_map.get("tags"):
            report.m1_free_ba_passed = False
            report.m1_msg = "工位尚未生成相对平差几何底图 (请先运行全局自由平差)"
            report.summary_toast = "❌ 自由平差底图缺失"
            return False, report, None

        rel_tags_3d = extract_tag_positions_from_map(relative_map)
        reproj_rmse = float(relative_map.get("rmse_reprojection_px", relative_map.get("final_rmse", 0.0)))
        tag_count = len(rel_tags_3d)

        report.m1_free_ba_passed = True
        report.m1_reprojection_rmse_px = reproj_rmse
        report.m1_tag_count = tag_count
        report.m1_msg = f"自由平差底图合格 (检出标靶: {tag_count} 个, 像面 RMSE: {reproj_rmse:.3f} px)"

        # ---------------------------------------------------------
        # 识别工位内配置的全部子坐标系 (1..N 个)
        # ---------------------------------------------------------
        frames_to_check: List[FrameDefinition] = []
        if coord_mgr:
            for f in coord_mgr.list_frames():
                if f.type != "world" and f.frame_id != "world":
                    frames_to_check.append(f)

        # ---------------------------------------------------------
        # 动态子坐标系 阶段 A: 局部刚体一致性质检 (Local Rigidity Check)
        # ---------------------------------------------------------
        for frame in frames_to_check:
            sub_rep = check_sub_frame_local_rigidity(
                frame=frame,
                relative_tags=rel_tags_3d,
                dist_tol_mm=self.dist_tol_mm,
                dist_tol_ratio=self.dist_tol_ratio,
                fallback_anchor_tags=anchor_tags,
                coord_mgr=coord_mgr,
            )
            report.sub_frames[frame.frame_id] = sub_rep

        # ---------------------------------------------------------
        # 里程碑 M2: 绝对世界基准系锚定 (World Datum Base Alignment)
        # ---------------------------------------------------------
        aligner = WorldDatumAligner(marker_size_mm=self.marker_size_mm)
        world_map = None
        world_align_ok = False

        if not anchor_tags:
            world_align_ok = False
            report.m2_world_datum_passed = False
            report.m2_msg = "工位未配置已知世界锚点 (请在 anchor_tags.yaml 录入 >=3 枚标靶物理坐标以完成世界基准校准)"
        else:
            try:
                world_map = aligner.align_relative_map_to_world(
                    relative_map=relative_map,
                    anchor_tags=anchor_tags,
                    origin_tag_id=origin_tag_id,
                    x_align_tag_id=x_align_tag_id,
                    strict=strict_world_datum,
                )
                world_align_ok = True
            except Exception as e:
                world_align_ok = False
                report.m2_world_datum_passed = False
                report.m2_msg = f"世界基准对齐受阻: {e}"
                log.warning(f"[MultiFrame] 世界基准对齐拦截: {e}")

        if world_align_ok and world_map:
            w_info = world_map.get("world_anchor", {})
            res_info = w_info.get("anchor_residual_mm", {})
            conflicts = w_info.get("conflict_pairs", [])

            if conflicts:
                report.m2_world_datum_passed = False
                report.m2_msg = (
                    f"世界基准拦截报错 (FATAL ERROR): 检测到 {len(conflicts)} 组锚点严重几何冲突！"
                    f"标称与实测测距超差 (残差均值: {report.m2_mean_residual_mm:.2f} mm, 最大: {report.m2_max_residual_mm:.2f} mm)"
                )
            else:
                report.m2_world_datum_passed = True
                report.m2_msg = (
                    f"绝对世界基准锚定成功 [{report.m2_solver_type}] "
                    f"(锚点均值残差: {report.m2_mean_residual_mm:.2f} mm, 最大: {report.m2_max_residual_mm:.2f} mm)"
                )

        # ---------------------------------------------------------
        # 动态子坐标系 阶段 B: 空间外参反推与挂接 (Extrinsics Attachment)
        # ---------------------------------------------------------
        # ---------------------------------------------------------
        # 动态子坐标系 阶段 B: 空间外参反推与挂接 (Extrinsics Attachment)
        # ---------------------------------------------------------
        if world_map and coord_mgr and report.m2_world_datum_passed:
            ext_solver = FrameExtrinsicSolver(tags_map=world_map)

            for frame in frames_to_check:
                sub_rep = report.sub_frames[frame.frame_id]

                # 熔断策略: 阶段 A 未通过，严禁写入外参！
                if not sub_rep.local_rigidity_passed:
                    sub_rep.extrinsic_solved = False
                    sub_rep.is_isolated = True
                    sub_rep.extrinsic_status_msg = (
                        "⛔ 阶段 A 局部刚体质检未通过，已触发外参熔断隔离！"
                        "未向系统写入错误外参，不影响主干世界基准。"
                    )
                    continue

                # 若未配置 calibration_spec 但 anchor_tags 中已有本子坐标系标靶，自动合成规范
                if not frame.calibration_spec and anchor_tags:
                    sub_anchors = {}
                    for tid_raw, a in anchor_tags.items():
                        if isinstance(a, dict) and a.get("frame_id") == frame.frame_id:
                            pos = a.get("xyz_mm") or a.get("coords") or a.get("position_mm")
                            if pos:
                                sub_anchors[str(tid_raw)] = [float(x) for x in pos[:3]]
                    if sub_anchors:
                        frame.calibration_spec = {
                            "method": "anchor_tags_registration",
                            "reference_tags": sub_anchors
                        }

                # 求解外参
                succ, t_xyz, r_rpy, rmse, msg = ext_solver.solve_frame_extrinsic(frame, coord_mgr)
                if succ and t_xyz is not None and r_rpy is not None:
                    sub_rep.extrinsic_solved = True
                    sub_rep.is_isolated = False
                    sub_rep.translation_xyz_mm = t_xyz
                    sub_rep.rotation_rpy_deg = r_rpy
                    sub_rep.extrinsic_rmse_mm = rmse
                    sub_rep.extrinsic_method = frame.calibration_spec.get("method", "registration_3d") if frame.calibration_spec else "registration_3d"
                    sub_rep.extrinsic_status_msg = f"外参成功挂接至 {sub_rep.parent_frame_id} (RMSE: {rmse:.3f} mm)"

                    # 原子更新坐标系树
                    coord_mgr.update_frame_solved_extrinsic(
                        frame_id=frame.frame_id,
                        translation_xyz_mm=t_xyz,
                        rotation_rpy_deg=r_rpy,
                        rmse_mm=rmse,
                        method=sub_rep.extrinsic_method,
                    )
                else:
                    sub_rep.extrinsic_solved = False
                    sub_rep.extrinsic_status_msg = f"外参反推失败: {msg}"
        elif not report.m2_world_datum_passed:
            # 世界基准未通过时，更新子坐标系阶段 B 的挂起提示
            for frame in frames_to_check:
                sub_rep = report.sub_frames[frame.frame_id]
                if sub_rep.local_rigidity_passed:
                    sub_rep.extrinsic_solved = False
                    sub_rep.extrinsic_status_msg = (
                        "⏸️ 局部尺寸质检合格，等待世界基准就绪后挂接外参"
                    )

        # ---------------------------------------------------------
        # 全流程综合状态裁定
        # ---------------------------------------------------------
        all_sub_ok = all(
            (s.local_rigidity_passed and s.extrinsic_solved)
            for s in report.sub_frames.values()
        )
        report.all_milestones_passed = bool(
            report.m1_free_ba_passed and report.m2_world_datum_passed and all_sub_ok
        )

        # 构造用户通知摘要
        sub_count = len(report.sub_frames)
        sub_succ = sum(1 for s in report.sub_frames.values() if s.extrinsic_solved)
        if report.all_milestones_passed:
            sub_info = f" | 子坐标系挂接: {sub_succ}/{sub_count}" if sub_count > 0 else ""
            report.summary_toast = f"✅ 世界系与多坐标系校准全流程绿灯！{sub_info}"
        elif not report.m2_world_datum_passed:
            report.summary_toast = "❌ 世界基准拦截报错 (存在严重几何冲突/底座标靶不足)"
        else:
            report.summary_toast = f"⚠️ 校准完成但存在隔离项 (子坐标系外参: {sub_succ}/{sub_count})"

        return report.all_milestones_passed, report, world_map

    @staticmethod
    def format_diagnostic_dialog_text(report: MultiFrameMilestoneReport) -> str:
        """
        生成工业级分层里程碑弹窗诊断文本，便于现场工程师秒级定位故障源
        """
        lines = []
        lines.append("【建图里程碑多坐标系质检诊断】")
        lines.append("=" * 60)

        # M1 状态
        m1_icon = "✅" if report.m1_free_ba_passed else "❌"
        lines.append(f"[M1 自由平差底图]: {m1_icon} {report.m1_msg}")

        # M2 状态
        m2_icon = "✅" if (report.m2_world_datum_passed and not report.m2_conflict_pairs) else ("⚠️" if report.m2_world_datum_passed else "❌")
        lines.append(f"[M2 绝对世界基准]: {m2_icon} {report.m2_msg}")
        if report.m2_conflict_pairs:
            lines.append("  " + "\n  ".join(format_conflict_pairs_report(report.m2_conflict_pairs).splitlines()))

        lines.append("-" * 60)

        # 动态子坐标系列表
        sub_cnt = len(report.sub_frames)
        lines.append(f"【动态子坐标系独立质检与外参挂接 (共 {sub_cnt} 个)】")

        if sub_cnt == 0:
            lines.append("  (未定义独立子坐标系，仅使用绝对世界坐标系)")
        else:
            for fid, s in report.sub_frames.items():
                lines.append(f"• [{fid} ({s.frame_name})]:")

                # 阶段 A
                a_icon = "✅" if s.local_rigidity_passed else "❌"
                lines.append(f"  - 阶段 A (局部刚体一致性): {a_icon} {s.local_status_msg}")
                if s.local_tag_pairs:
                    lines.append(f"    匹配标靶: Tag {s.local_matched_tags}")
                    lines.append("    内部测距对:")
                    for p in s.local_tag_pairs:
                        p_icon = "⚠️" if p.get("is_conflict") else " "
                        t1, t2 = p["pair"]
                        lines.append(
                            f"      {p_icon} Tag #{t1} ⇋ #{t2}: 标称 {p['nominal_dist_mm']} mm | "
                            f"实测 {p['measured_dist_mm']} mm | 误差 {p['diff_mm']:+.1f} mm ({p['rel_error_pct']}%)"
                        )

                # 阶段 B
                b_icon = "✅" if s.extrinsic_solved else ("⛔" if s.is_isolated else "⏸️")
                lines.append(f"  - 阶段 B (空间外参反推挂接): {b_icon} {s.extrinsic_status_msg}")
                if s.extrinsic_solved and s.translation_xyz_mm and s.rotation_rpy_deg:
                    t_str = f"[{s.translation_xyz_mm[0]:.1f}, {s.translation_xyz_mm[1]:.1f}, {s.translation_xyz_mm[2]:.1f}]"
                    r_str = f"[{s.rotation_rpy_deg[0]:.1f}, {s.rotation_rpy_deg[1]:.1f}, {s.rotation_rpy_deg[2]:.1f}]"
                    lines.append(f"    平移 T: {t_str} mm | 旋转 RPY: {r_str}°")

                lines.append("")

        lines.append("=" * 60)

        # 智能诊断指导建议
        if not report.all_milestones_passed:
            lines.append("💡 【智能排查建议】:")
            if not report.m1_free_ba_passed:
                lines.append("  1. 请先执行【自由平差 (B)】生成高质量像面几何底图；")
            if not report.m2_world_datum_passed:
                lines.append("  2. 底座世界标靶不足或不共线：请在工作台底座补充 >=3 枚已知世界坐标标靶；")
            isolated_subs = [fid for fid, s in report.sub_frames.items() if s.is_isolated]
            if isolated_subs:
                lines.append(f"  3. 子坐标系 {isolated_subs} 内部标靶测距超差：请检查该构件上的标靶贴附位置或名义尺寸参数，世界系不受影响。")

        return "\n".join(lines)

    @staticmethod
    def format_markdown_report(report: MultiFrameMilestoneReport) -> str:
        """
        生成 Markdown 格式的完整质检报告 (支持一键复制到剪贴板)
        """
        lines = []
        lines.append("# 工位多坐标系分层里程碑质检报告")
        lines.append(f"- **解算时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"- **综合判定**: {'✅ 全部通过 (ALL PASSED)' if report.all_milestones_passed else '⚠️ 存在警告/隔离项'}")
        lines.append("")

        lines.append("## 一、主干基准层 (Global Backbone)")
        lines.append(f"### M1. 全局自由平差底图 (Free BA Base Map)")
        lines.append(f"- **状态**: {'✅ 合格' if report.m1_free_ba_passed else '❌ 不合格'}")
        lines.append(f"- **像面重投影残差**: {report.m1_reprojection_rmse_px:.3f} px")
        lines.append(f"- **检出标靶总数**: {report.m1_tag_count} 个")
        lines.append(f"- **详细说明**: {report.m1_msg}")
        lines.append("")

        lines.append(f"### M2. 绝对世界基准锚定 (World Datum Base Alignment)")
        lines.append(f"- **状态**: {'✅ 合格' if report.m2_world_datum_passed else '❌ 不合格'}")
        lines.append(f"- **解算器类型**: {report.m2_solver_type}")
        lines.append(f"- **锚点残差均值**: {report.m2_mean_residual_mm:.2f} mm (最大: {report.m2_max_residual_mm:.2f} mm)")
        lines.append(f"- **几何测距冲突**: {len(report.m2_conflict_pairs)} 组")
        lines.append(f"- **详细说明**: {report.m2_msg}")
        lines.append("")

        lines.append("## 二、动态子坐标系独立质检与外参挂接")
        if not report.sub_frames:
            lines.append("*(当前工位未定义附加子坐标系)*\n")
        else:
            for fid, s in report.sub_frames.items():
                lines.append(f"### 子坐标系: `{fid}` ({s.frame_name})")
                lines.append(f"- **父坐标系**: `{s.parent_frame_id}`")
                lines.append(f"- **阶段 A (局部刚体一致性)**: {'✅ 合格' if s.local_rigidity_passed else '❌ 超差'}")
                lines.append(f"  - 说明: {s.local_status_msg}")
                lines.append(f"  - 内部匹配标靶: `Tag {s.local_matched_tags}`")
                if s.local_rmse_mm is not None:
                    lines.append(f"  - 局部配准 RMSE: {s.local_rmse_mm:.3f} mm")

                if s.local_tag_pairs:
                    lines.append("\n| 标靶点对 | 图纸标称 (mm) | 视觉实测 (mm) | 偏差 (mm) | 相对误差 | 状态 |")
                    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- |")
                    for p in s.local_tag_pairs:
                        st = "⚠️ 超差" if p.get("is_conflict") else "✅ 正常"
                        lines.append(
                            f"| Tag #{p['pair'][0]} ⇋ #{p['pair'][1]} | {p['nominal_dist_mm']} | "
                            f"{p['measured_dist_mm']} | {p['diff_mm']:+.1f} | {p['rel_error_pct']}% | {st} |"
                        )
                    lines.append("")

                lines.append(f"- **阶段 B (空间外参挂接)**: {'✅ 已反推并固化' if s.extrinsic_solved else ('⛔ 熔断隔离' if s.is_isolated else '⏸️ 挂起')}")
                lines.append(f"  - 说明: {s.extrinsic_status_msg}")
                if s.extrinsic_solved and s.translation_xyz_mm and s.rotation_rpy_deg:
                    lines.append(f"  - 平移向量 T: `{[round(x, 2) for x in s.translation_xyz_mm]}` mm")
                    lines.append(f"  - 旋转姿态 RPY: `{[round(r, 2) for r in s.rotation_rpy_deg]}` deg")
                    if s.extrinsic_rmse_mm is not None:
                        lines.append(f"  - 配准拟合残差 RMSE: `{s.extrinsic_rmse_mm:.3f}` mm")
                lines.append("")

        return "\n".join(lines)
