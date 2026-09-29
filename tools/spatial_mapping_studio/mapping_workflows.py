#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间建图工作站 - 异步工作流 Mixin (mapping_workflows.py)
=============================================
从 app.py 拆分出的长耗时工作流调度职责模块，由 SpatialMappingStudioApp 以 Mixin 方式继承：
  - 全量超精重提取的异步调度与结果轮询 (start_async_super_extract_all / poll_super_extract_result)
  - 智能剪枝平差的启动 / 采纳 / 撤销 (start_auto_prune_ba / accept_prune_results / undo_prune_results)
  - 异步全局 BA 平差启动 (start_async_bundle_adjustment)
  - 空间立体地图保存与全景质检报告导出 (save_current_workspace_map / export_verification_report)
无 __init__、无新增实例属性，全部通过宿主 self 与主控制器协作。
"""

import os
import time
import threading
from typing import Optional, Tuple

from src.calibration.manifest_repository import ManifestRepository
from tools.spatial_mapping_studio.mapping_app_meta import PROJECT_ROOT
from src.utils.logger import get_logger

log = get_logger(__name__)


class MappingWorkflowMixin:

    """异步工作流调度职责 (无状态，依赖宿主 self 属性)"""

    def start_async_super_extract_all(self) -> bool:
        """启动后台异步线程执行全量采图工序 3 工业级超精重提取并从头重建"""
        if self.is_extracting_all:
            self.set_toast("全量超精提取已在后台运行中，请稍候...")
            return False
        if self.is_ba_running:
            self.set_toast("全局平差计算中，请待平差完成后再执行提取")
            return False
        if not self.image_files:
            self.set_toast("未扫描到采图文件，无法执行超精重提取")
            return False

        self.is_extracting_all = True
        self.extract_progress = 0.01
        self.extract_stage_text = f"正在启动全局全量超精提取 (共 {len(self.image_files)} 帧)..."
        self.set_toast(self.extract_stage_text)

        def _worker():
            try:
                def on_progress(cur, total, bname, count):
                    self.extract_progress = cur / max(1, total)
                    self.extract_stage_text = f"全量超精提取 ({cur}/{total}): {bname} (检出 {count} 个标靶)"

                total_frames, total_tags = self.data_mgr.super_extract_all_frames(progress_callback=on_progress)
                self.extract_progress = 1.0
                msg = f"全局超精提取完成！处理 {total_frames} 帧，累计提取 {total_tags} 个高精标靶"
                self.extract_result_queue = (True, msg)
            except Exception as e:
                self.extract_result_queue = (False, f"全量超精重提取失败: {e}")

        self.extract_thread = threading.Thread(target=_worker, daemon=True)
        self.extract_thread.start()
        return True

    def poll_super_extract_result(self) -> Optional[Tuple[bool, str]]:
        """检查异步全量超精提取任务是否完成"""
        if self.extract_result_queue is not None:
            res = self.extract_result_queue
            self.extract_result_queue = None
            self.is_extracting_all = False
            return res
        return None

    def start_auto_prune_ba(self) -> bool:
        """启动全自动基于边际收益与共视拓扑守门的残差剪枝平差"""
        if self.is_ba_running or self.is_extracting_all:
            self.set_toast("后台任务正在计算中，请稍候...")
            return False
        # 联动质检视角: 自动将左侧图像序列切换为【残差降序 (最差优先 ↓)】并展开多轮残差矩阵视图
        self.sort_mode = "err_desc"
        self.matrix_view_mode = True
        self.left_bar_w = self.dynamic_left_bar_w
        res = self.ba_runner.start_auto_prune(max_rounds=10, min_improvement_px=0.01)
        if res:
            # 自动将主视口聚焦至残差最大、最亟待排查的首张图像
            f_indices = self._get_filtered_indices()
            if f_indices:
                self.current_img_idx = f_indices[0]
        return res

    def accept_prune_results(self):
        """采纳智能剪枝平差结果并清空结算单"""
        self.ba_runner.prune_settlement_data = None
        self.set_toast("已采纳智能剪枝平差结果！可按 [M] 保存为最新地图")
        log.info("[*] [SPATIAL_MAPPING] 操作员确认采纳智能剪枝平差结果。")

    def undo_prune_results(self):
        """一键无损撤销智能剪枝，回滚至快照状态"""
        succ = self.data_mgr.restore_manifest_snapshot()
        self.ba_runner.prune_settlement_data = None
        if succ:
            self.set_toast("已撤销智能剪枝！观测清单与地图已完全恢复至剪枝前状态")
            log.warning("[*] [SPATIAL_MAPPING] 操作员已撤销智能剪枝，状态已无损回滚。")
        else:
            self.set_toast("未找到有效快照，撤销未执行")

    def start_async_bundle_adjustment(self):
        """启动后台线程执行两阶段全局 BA 平差优化，前台持续平滑响应"""
        return self.ba_runner.start()

    def save_current_workspace_map(self):
        """将当前优化好的高精度几何地图原子保存至当前工位沙盒 (tags_map.yaml)"""
        if not self.tags_map_data:
            self.set_toast("当前尚无有效地图，请先按 [B] 进行 BA 平差！")
            return

        ManifestRepository.save_map(self.tags_map_data, self.map_path)
        ws_name = self.current_workspace.name if self.current_workspace else "当前工位"
        self.set_toast(f"地图已成功保存至【{ws_name}】工位沙盒 (tags_map.yaml)！")
        log.info(f"[SPATIAL_MAPPING] 地图已持久化至工位: {self.map_path}")

    def export_verification_report(self):
        """导出 Markdown 全景精度质检单"""
        if self.current_workspace:
            report_dir = self.current_workspace.calib_reports_dir
        else:
            try:
                from src.calibration.workspace_manager import WorkspaceManager
                report_dir = WorkspaceManager().get_current_workspace().calib_reports_dir
            except Exception:
                report_dir = os.path.join(PROJECT_ROOT, "data", "workspaces", "default", "calibration", "reports")
        os.makedirs(report_dir, exist_ok=True)
        ts = int(time.time())
        report_path = os.path.join(report_dir, f"spatial_mapping_qa_report_{ts}.md")

        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"# AprilTag 空间建图工作站全景精度质检单 (Spatial Mapping Studio)\n\n")
                f.write(f"- **质检时间**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`\n")
                f.write(f"- **总采图集**: `{len(self.image_files)} 帧`\n")
                f.write(f"- **全景 RMSE**: `{self.global_rmse:.3f} px`\n")
                f.write(f"- **已知标靶数**: `{len(self.tags_map_data.get('tags', {}))} 个`\n")
                f.write(f"- **空间地图**: `{self.map_path}`\n\n")
                f.write(f"## 图像帧逐项质检明细\n\n")
                f.write(f"| 图像帧 | 观测标靶数 | 平均残差 | 最大残差 | 状态 |\n")
                f.write(f"| :--- | :---: | :---: | :---: | :---: |\n")

                for p in self.image_files:
                    bname = os.path.basename(p)
                    meta = self.frame_metrics_cache.get(bname, {})
                    status_str = "❌ 已剔除" if meta.get("is_excluded", False) else "✅ 参与解算"
                    f.write(f"| `{bname}` | {meta.get('tag_count', 0)} | {meta.get('mean_err', 0.0):.2f} px | {meta.get('max_err', 0.0):.2f} px | {status_str} |\n")

                # 2. 智能剪枝平差逐帧多轮残差收敛矩阵 (若存在多轮历史)
                headers = getattr(self.data_mgr, "convergence_headers", [])
                matrix = getattr(self.data_mgr, "frame_convergence_matrix", {})
                if headers and matrix and len(headers) >= 1:
                    f.write(f"\n## 2. 智能剪枝平差逐帧多轮残差收敛矩阵 (Per-Frame Convergence Matrix)\n\n")
                    f.write(f"> 记录各图像帧在每一轮平差求解后的残差演进变化情况：\n\n")
                    header_cols = ["图像帧", "标靶数"] + headers + ["累计降幅"]
                    f.write("| " + " | ".join(header_cols) + " |\n")
                    f.write("| " + " | ".join([":---"] + [":---:"] * (len(header_cols) - 1)) + " |\n")

                    for p in self.image_files:
                        bname = os.path.basename(p)
                        meta = self.frame_metrics_cache.get(bname, {})
                        tag_cnt = meta.get("tag_count", 0)
                        row_vals = matrix.get(bname, [])
                        r_strs = []
                        for val in row_vals:
                            r_strs.append(f"{val:.2f} px" if val is not None else "--")
                        while len(r_strs) < len(headers):
                            r_strs.append("--")

                        first_val = row_vals[0] if (row_vals and row_vals[0] is not None) else None
                        last_val = None
                        for v in reversed(row_vals):
                            if v is not None:
                                last_val = v
                                break
                        if first_val is not None and last_val is not None and first_val > 0.001:
                            drop_px = first_val - last_val
                            drop_pct = (drop_px / first_val) * 100.0
                            drop_str = f"↓{drop_pct:.1f}% ({drop_px:+.2f}px)"
                        else:
                            drop_str = "--"

                        f.write(f"| `{bname}` | {tag_cnt} | " + " | ".join(r_strs) + f" | {drop_str} |\n")

            rel_dir = os.path.relpath(report_dir, PROJECT_ROOT).replace("\\", "/")
            self.set_toast(f"全景质检报告已成功导出至 {rel_dir}/！")
            log.info(f"[OK] 质检报告导出成功: {report_path}")
        except Exception as e:
            self.set_toast(f"导出质检报告失败: {e}")
            log.warning(f"导出质检报告异常: {e}")

    def align_current_workspace_world_datum(self):
        """【阶段二/三交互入口】执行世界坐标系校准与子坐标系外参逆解，毫秒级生效"""
        succ, msg, world_map = self.ba_runner.execute_world_alignment()
        if succ and world_map:
            rep = world_map.get("world_anchor", {}).get("alignment_report")
            if rep:
                self.alignment_report = rep
            if hasattr(self, "_load_workspace_geometry"):
                self._load_workspace_geometry()

            # 校验是否存在锚点几何严重冲突 (即使强制对齐成功，也必须在 GUI 弹出 Warning 提醒)
            conflict_pairs = world_map.get("world_anchor", {}).get("conflict_pairs", [])
            if conflict_pairs:
                from src.calibration.world_datum_aligner import format_conflict_pairs_report
                from src.utils.dialog_utils import show_error_dialog
                conflict_text = format_conflict_pairs_report(conflict_pairs)
                cur_ws_name = getattr(self, "current_workspace_name", "当前工位")
                warn_dialog_msg = (
                    f"【当前工位】{cur_ws_name}\n\n"
                    f"世界坐标系校准已解算完成，但在锚点间检测到严重几何形变/测距冲突：\n\n"
                    f"{conflict_text}\n\n"
                    f"⚠️ 提示：上述标靶的世界坐标标称值与视觉实测距离存在显著超差，可能严重影响下游机械臂或作业定位精度，请仔细核对已知锚点坐标！"
                )
                show_error_dialog("世界坐标系校准警告 (几何形变冲突)", warn_dialog_msg)

            self.set_toast(msg)
        else:
            from src.utils.dialog_utils import show_error_dialog
            self.alignment_report = None
            cur_ws_name = getattr(self, "current_workspace_name", "未知工位")
            cur_ws_id = getattr(self, "current_workspace_id", "")
            configured_anchors = sorted(self.ba_runner.anchor_tags.keys()) if (hasattr(self, "ba_runner") and getattr(self.ba_runner, "anchor_tags", None)) else []
            observed_tags = sorted(list(self.data_mgr.tags_map_data.get("tags", {}).keys())) if (hasattr(self, "data_mgr") and getattr(self.data_mgr, "tags_map_data", None) and "tags" in self.data_mgr.tags_map_data) else []
            intersection_anchors = sorted(list(set(configured_anchors) & set(observed_tags)))

            detail_msg = (
                f"【当前工位】{cur_ws_name} ({cur_ws_id})\n"
                f"【工位配置已知锚点】Tag {configured_anchors}\n"
                f"【当前平差有效检出锚点】Tag {intersection_anchors}\n\n"
                f"世界坐标系校准拦截失败：\n"
                f"{msg}\n\n"
                f"【排查指引】\n"
                f"1. 确认当前工位是否已先执行阶段一【自由平差 (B)】生成相对底图；\n"
                f"2. 检查当前工位世界锚点配置（anchor_tags.yaml 或 tag_whitelist.yaml）；\n"
                f"3. 确认锚点标靶是否在观测数据中至少检出 >= 3 个且空间分布不共线（XY 坐标不可重合）。"
            )
            brief_err = msg.splitlines()[0] if msg else "未知异常"
            self.set_toast(f"❌ 校准失败: {brief_err}")
            show_error_dialog("世界坐标系校准失败 (World Datum Error)", detail_msg)

