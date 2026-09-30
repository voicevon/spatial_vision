# -*- coding: utf-8 -*-
"""
工位空间场景与元数据全量健康体检与自愈服务 (Workspace Health Auditor & Self-Healer)
================================================================================
核心职责：
1. 物理磁盘与工位元数据 (workspace_meta.yaml) 一致性核验与自动自愈；
2. 标靶白名单 (tag_whitelist.yaml) 与已建图空间标靶 (tags_map.yaml) 权威一致性核验：
   - 彻底发现并自动清洗“未建图幽灵标靶 (Ghost Tags)”与“悬空/全 null 无效锚点”；
   - 对尚未建图标定中的工位执行保护性保留（不误伤用户建图前的预设白名单）；
3. 空间几何场景 (spatial_scene.yaml) 坐标系树与 3D ROI 完整性检查；
4. 工业级精度指标 (RMSE) 达标评级；
5. 自动渲染并保存标准 Markdown 格式体检报告至工程根目录 temp/ 目录 (严格遵循 AGENTS.md 准则)。
"""

import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import yaml

from src.utils.logger import get_logger
from src.workspace.workspace_manager import (
    Workspace,
    WorkspaceManager,
    load_workspace_tag_config,
    save_workspace_tag_config,
)

log = get_logger(__name__)


def audit_workspace(ws: Workspace, auto_fix: bool = True) -> Dict[str, Any]:
    """
    对单个工位执行深度健康体检与自愈修剪。

    :param ws: Workspace 工位实例
    :param auto_fix: 是否自动修复发现的问题（如修剪幽灵标靶、同步图片数）
    :return: 包含该工位体检指标与自愈动作的字典
    """
    ws_id = ws.workspace_id
    ws_name = ws.name or ws_id
    actions_taken: List[str] = []
    warnings: List[str] = []

    # ---------------- 1. 物理照片与元数据一致性核验 ----------------
    disk_calib = ws.get_image_count("calibration")
    disk_prod = ws.get_image_count("production")
    recorded_calib = ws.image_count
    recorded_prod = ws.prod_image_count

    img_synced = (disk_calib == recorded_calib and disk_prod == recorded_prod)
    if not img_synced:
        msg = f"物理照片与元数据偏差: 标定记录 {recorded_calib} (实际 {disk_calib}), 生产记录 {recorded_prod} (实际 {disk_prod})"
        warnings.append(msg)
        if auto_fix:
            ws.refresh_stats()
            ws.save_meta()
            actions_taken.append(f"物理照片数量已自愈同步: 标定 {disk_calib} 帧, 生产 {disk_prod} 帧")

    # ---------------- 2. 标靶白名单与锚点一致性核验 (幽灵标靶防腐) ----------------
    wl_cfg = load_workspace_tag_config(ws.workspace_dir)
    allowed_ids = list(wl_cfg.get("allowed_ids") or [])
    anchor_tags = dict(wl_cfg.get("anchor_tags") or {})
    tag_size = wl_cfg.get("tag_default_size_mm", 40.0)

    # 检查 tags_map.yaml
    mapped_tags: List[int] = []
    has_map = os.path.isfile(ws.map_path)
    if has_map:
        try:
            with open(ws.map_path, "r", encoding="utf-8") as f:
                map_data = yaml.safe_load(f) or {}
            raw_tags = map_data.get("tags") or {}
            for k in raw_tags.keys():
                try:
                    mapped_tags.append(int(k))
                except Exception:
                    continue
            mapped_tags = sorted(set(mapped_tags))
        except Exception as e:
            warnings.append(f"读取 tags_map.yaml 异常: {e}")

    ghost_allowed: List[int] = []
    ghost_anchors: List[int] = []
    empty_anchors: List[int] = []
    illegal_map_tags: List[int] = []

    if has_map and mapped_tags:
        mapped_set = set(mapped_tags)
        allowed_set = set(allowed_ids)

        # ---------------- (A) 输出端核验: 平差地图 tags_map.yaml 是否收录了未放行的非法标靶 ----------------
        if allowed_set:
            illegal_map_tags = [t for t in mapped_tags if t not in allowed_set]
            if illegal_map_tags:
                warnings.append(f"平差输出包含未放行非法标靶: {illegal_map_tags}")
                if auto_fix:
                    raw_tags = map_data.get("tags") or {}
                    raw_poses = map_data.get("raw_relative_poses") or {}
                    for bad_id in illegal_map_tags:
                        raw_tags.pop(bad_id, None)
                        raw_tags.pop(str(bad_id), None)
                        raw_poses.pop(bad_id, None)
                        raw_poses.pop(str(bad_id), None)
                    map_data["tags"] = raw_tags
                    if raw_poses:
                        map_data["raw_relative_poses"] = raw_poses
                    try:
                        with open(ws.map_path, "w", encoding="utf-8") as f:
                            yaml.dump(map_data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                        actions_taken.append(f"平差输出自愈: 从 tags_map.yaml 彻底清除未放行非法标靶 {illegal_map_tags}")
                        mapped_tags = sorted([t for t in mapped_tags if t in allowed_set])
                        mapped_set = set(mapped_tags)
                    except Exception as e:
                        warnings.append(f"自愈更新 tags_map.yaml 失败: {e}")

        # ---------------- (B) 输入端核验: 白名单与锚点是否残留未建图的幽灵标靶 ----------------
        ghost_allowed = [t for t in allowed_ids if t not in mapped_set]
        ghost_anchors = [t for t in anchor_tags.keys() if t not in mapped_set]
        for tid, a in anchor_tags.items():
            xyz = a.get("xyz_mm") or []
            if not any(c is not None for c in xyz):
                empty_anchors.append(tid)

        if ghost_allowed or ghost_anchors or empty_anchors:
            issue_desc = []
            if ghost_allowed:
                issue_desc.append(f"幽灵白名单标靶 {ghost_allowed}")
            if ghost_anchors:
                issue_desc.append(f"悬空锚点 {ghost_anchors}")
            if empty_anchors:
                issue_desc.append(f"全空无效锚点 {empty_anchors}")
            warnings.append("；".join(issue_desc))

            if auto_fix:
                clean_allowed = sorted([t for t in allowed_ids if t in mapped_set])
                clean_anchors = {
                    tid: a for tid, a in anchor_tags.items()
                    if tid in mapped_set and any(c is not None for c in (a.get("xyz_mm") or []))
                }
                save_workspace_tag_config(
                    ws,
                    allowed_ids=clean_allowed,
                    anchor_tags=clean_anchors,
                    tag_default_size_mm=tag_size
                )
                actions_taken.append(
                    f"标靶输入自愈: 剔除幽灵白名单 {ghost_allowed}, 剔除无效锚点 {sorted(set(ghost_anchors + empty_anchors))}"
                )
                allowed_ids = clean_allowed
                anchor_tags = clean_anchors
    else:
        # 工位尚未完成建图，保护性保留用户预设
        pass

    # ---------------- 3. 空间几何场景文件 (spatial_scene.yaml) 核验 ----------------
    scene_ok = os.path.isfile(ws.spatial_scene_path)
    frame_count = 0
    roi_count = 0
    has_world_frame = False

    if scene_ok:
        try:
            with open(ws.spatial_scene_path, "r", encoding="utf-8") as f:
                scene_data = yaml.safe_load(f) or {}
            frames = scene_data.get("frames") or []
            rois = scene_data.get("rois") or []
            frame_count = len(frames)
            roi_count = len(rois)
            for fr in frames:
                if fr.get("frame_id") == "world" or fr.get("type") == "world" or fr.get("parent_frame_id") is None:
                    has_world_frame = True
                    break
            if not has_world_frame:
                warnings.append("spatial_scene.yaml 缺失 world 基准坐标系")
        except Exception as e:
            warnings.append(f"spatial_scene.yaml 解析失败: {e}")
            scene_ok = False
    else:
        warnings.append("spatial_scene.yaml 场景文件缺失")

    # ---------------- 4. 精度与质检放行评级 ----------------
    rmse = ws.global_rmse_px
    if ws.ba_solved and rmse > 1e-6:
        if rmse < 0.8:
            quality_status = "🟢 优良放行 (RMSE < 0.8px)"
        else:
            quality_status = f"🟡 需优化 (RMSE {rmse:.3f}px > 0.8px)"
    else:
        quality_status = "⚪ 尚未平差 (无几何真值)"

    return {
        "workspace_id": ws_id,
        "workspace_name": ws_name,
        "ba_solved": ws.ba_solved,
        "rmse_px": rmse,
        "quality_status": quality_status,
        "images": {
            "calib_disk": disk_calib,
            "calib_recorded": recorded_calib,
            "prod_disk": disk_prod,
            "prod_recorded": recorded_prod,
            "synced": img_synced,
        },
        "tags": {
            "has_map": has_map,
            "mapped_tags": mapped_tags,
            "allowed_ids": allowed_ids,
            "anchor_tag_ids": sorted(list(anchor_tags.keys())),
            "illegal_map_tags": illegal_map_tags,
            "ghost_allowed": ghost_allowed,
            "ghost_anchors": ghost_anchors,
            "empty_anchors": empty_anchors,
        },
        "scene": {
            "exists": scene_ok,
            "frame_count": frame_count,
            "roi_count": roi_count,
            "has_world_frame": has_world_frame,
        },
        "warnings": warnings,
        "actions_taken": actions_taken,
        "is_healthy": len(warnings) == 0,
    }


def audit_all_workspaces(workspace_mgr: WorkspaceManager, auto_fix: bool = True) -> Dict[str, Any]:
    """
    全工位深度健康体检与自愈，并生成标准 Markdown 报告至 temp/ 目录。

    :param workspace_mgr: WorkspaceManager 实例
    :param auto_fix: 是否自动修复
    :return: 包含总览统计、逐工位体检结果及报告路径的字典
    """
    workspaces = workspace_mgr.list_workspaces()
    results: List[Dict[str, Any]] = []

    total_ws = len(workspaces)
    healed_ws_count = 0
    total_ghost_pruned = 0
    total_illegal_map_tags_pruned = 0
    total_img_synced_ws = 0

    log.info(f"[HEALTH_AUDITOR] 开始全工位健康体检 (共 {total_ws} 个工位, auto_fix={auto_fix})...")

    for ws in workspaces:
        res = audit_workspace(ws, auto_fix=auto_fix)
        results.append(res)
        if res["actions_taken"]:
            healed_ws_count += 1
        ghost_count = len(res["tags"]["ghost_allowed"]) + len(res["tags"]["ghost_anchors"])
        total_ghost_pruned += ghost_count
        illegal_count = len(res["tags"].get("illegal_map_tags") or [])
        total_illegal_map_tags_pruned += illegal_count
        if not res["images"]["synced"]:
            total_img_synced_ws += 1

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"workspace_health_report_{timestamp_str}.md"

    # 严格保证输出到项目根目录 temp/ 下 (遵循 AGENTS.md 准则)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(curr_dir, "..", ".."))
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    report_path = os.path.join(temp_dir, report_filename)

    # 渲染 Markdown 报告
    report_md = _render_health_report_markdown(
        results=results,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        total_ws=total_ws,
        healed_ws_count=healed_ws_count,
        total_ghost_pruned=total_ghost_pruned,
        auto_fix=auto_fix,
    )

    try:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        log.info(f"[HEALTH_AUDITOR] 全工位体检报告已生成: {report_path}")
    except Exception as e:
        log.error(f"[HEALTH_AUDITOR] 写入体检报告失败: {e}")

    # 生成弹窗及控制台摘要文本
    summary_lines = [
        f"全工位健康体检与自愈完成！",
        f"--------------------------------------------------",
        f"• 扫描工位总数 : {total_ws} 个",
        f"• 触发自愈工位 : {healed_ws_count} 个",
        f"• 清洗幽灵标靶 : {total_ghost_pruned} 枚",
        f"• 照片同步工位 : {total_img_synced_ws} 个",
        f"• 详细体检报告 : temp/{report_filename}",
        f"--------------------------------------------------",
    ]
    for r in results:
        status_symbol = "🟢" if r["is_healthy"] or (auto_fix and r["actions_taken"]) else "⚠️"
        act_info = f" [已自愈 {len(r['actions_taken'])} 项]" if r["actions_taken"] else ""
        summary_lines.append(f"{status_symbol} 【{r['workspace_name']}】({r['workspace_id']}){act_info}")
        for act in r["actions_taken"]:
            summary_lines.append(f"    ✔ {act}")
        for warn in r["warnings"]:
            if not r["actions_taken"]:
                summary_lines.append(f"    ⚠ {warn}")

    summary_text = "\n".join(summary_lines)

    return {
        "timestamp": timestamp_str,
        "total_workspaces": total_ws,
        "healed_workspaces": healed_ws_count,
        "total_ghost_pruned": total_ghost_pruned,
        "report_path": report_path,
        "report_filename": report_filename,
        "summary_text": summary_text,
        "results": results,
    }


def _render_health_report_markdown(
    results: List[Dict[str, Any]],
    timestamp: str,
    total_ws: int,
    healed_ws_count: int,
    total_ghost_pruned: int,
    auto_fix: bool
) -> str:
    """渲染高雅规范的 Markdown 格式体检与自愈报告"""
    lines = [
        f"# 工位空间场景与标靶全量健康体检报告",
        f"",
        f"> **生成时间**：{timestamp}  ",
        f"> **自愈模式**：{'已启用 (Auto-Healed)' if auto_fix else '仅检查 (Read-Only)'}  ",
        f"> **报告存储**：`temp/` (零版本依赖易失性报告)  ",
        f"",
        f"---",
        f"",
        f"## 一、 总体体检与自愈概览",
        f"",
        f"| 检查项 | 统计结果 | 状态评价 |",
        f"| :--- | :--- | :--- |",
        f"| **扫描工位总数** | **{total_ws}** 个 | - |",
        f"| **健康完好工位** | **{sum(1 for r in results if r['is_healthy'])}** 个 | 资产与物理现实 100% 吻合 |",
        f"| **完成自愈工位** | **{healed_ws_count}** 个 | 历史残留脏数据已完成自动修剪 |",
        f"| **清洗幽灵标靶总数** | **{total_ghost_pruned}** 枚 | 消除平差发散与误检隐患 |",
        f"",
        f"---",
        f"",
        f"## 二、 工位明细核验与自愈清单",
        f"",
    ]

    for idx, r in enumerate(results, 1):
        ws_id = r["workspace_id"]
        ws_name = r["workspace_name"]
        lines.append(f"### {idx}. 工位：{ws_name} (`{ws_id}`)")
        lines.append(f"")
        lines.append(f"- **质检放行评级**：{r['quality_status']}")
        lines.append(f"- **平差 RMSE 指标**：`{r['rmse_px']:.4f} px` (已解算: `{r['ba_solved']}`)")
        lines.append(f"- **物理照片状态**：标定帧 `{r['images']['calib_disk']}` 帧, 生产帧 `{r['images']['prod_disk']}` 帧 "
                     f"({'● 100% 一致' if r['images']['synced'] else '⚠ 已自愈修正元数据'})")
        
        # 标靶核验
        t_info = r["tags"]
        lines.append(f"- **实测空间建图标靶**：共 `{len(t_info['mapped_tags'])}` 枚 `({t_info['mapped_tags']})`")
        lines.append(f"- **物理准入白名单 (allowed_ids)**：共 `{len(t_info['allowed_ids'])}` 枚 `({t_info['allowed_ids']})`")
        lines.append(f"- **世界参考锚点 (anchor_tags)**：共 `{len(t_info['anchor_tag_ids'])}` 枚 `({t_info['anchor_tag_ids']})`")

        has_tag_issue = bool(t_info.get("illegal_map_tags") or t_info["ghost_allowed"] or t_info["ghost_anchors"] or t_info["empty_anchors"])
        if has_tag_issue:
            lines.append(f"- **⚠ 标靶自洽性问题与非法数据**：")
            if t_info.get("illegal_map_tags"):
                lines.append(f"  - 🚫 平差输出非法标靶 (未放行却记入地图)：`{t_info['illegal_map_tags']}`")
            if t_info["ghost_allowed"]:
                lines.append(f"  - 👻 幽灵白名单标靶 (未建图却放行)：`{t_info['ghost_allowed']}`")
            if t_info["ghost_anchors"]:
                lines.append(f"  - ⚓ 悬空锚点 (未建图却设锚点)：`{t_info['ghost_anchors']}`")
            if t_info["empty_anchors"]:
                lines.append(f"  - ❓ 无效全空锚点：`{t_info['empty_anchors']}`")
        else:
            lines.append(f"- **标靶防腐健康度**：🟢 物理白名单与空间平差地图 100% 权威闭环，无非法/幽灵标靶")

        # 空间场景核验
        s_info = r["scene"]
        lines.append(f"- **空间场景 (spatial_scene.yaml)**：坐标系 `{s_info['frame_count']}` 个, ROI `{s_info['roi_count']}` 个, "
                     f"World基准: `{'正常' if s_info['has_world_frame'] else '缺失'}`")

        # 自愈动作
        if r["actions_taken"]:
            lines.append(f"- **🛠️ 自动自愈执行记录**：")
            for act in r["actions_taken"]:
                lines.append(f"  - `{act}`")
        lines.append(f"")
        lines.append(f"---")

    lines.append(f"")
    lines.append(f"## 三、 架构防腐准则建议")
    lines.append(f"")
    lines.append(f"1. **单一真理源原则 (SSOT)**：已完成 BA 平差建图的工位，空间场景中实测存在的标靶以 `tags_map.yaml` 为唯一物理事实；")
    lines.append(f"2. **严防误检污染**：白名单 `allowed_ids` 严禁包含现场未张贴或未建图的孤立 Tag ID，避免图像噪点触发误识别后拉扯平差；")
    lines.append(f"3. **定期自愈**：建议在批量替换标靶、采图或工位归档前后，通过 Workspace Hub 左下角的 `[全工位体检与自愈]` 按钮执行全量体检。")
    lines.append(f"")

    return "\n".join(lines)
