"""
空间建图工作站 - 异步 BA 平差调度器 (MappingBARunner)
================================================================================
负责工作站中高耗时计算任务的生命周期调度与状态通知：
1. 后台异步线程执行两阶段 Cauchy 稳健核平差优化 (Bundle Adjustment)
2. 细粒度双进度条推进 (大阶段全局进度 ba_progress + 求解器迭代子进度 ba_sub_progress)
3. 实时迭代收敛指标监听与文本反馈 (轮次、迭代 RMSE)
4. 平差后空间立体地图就地热更新与持久化保存
"""

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.calibration.solvers.ba_optimizer import BundleAdjustmentOptimizer
from src.calibration.manifest_repository import ManifestRepository
from src.workspace.workspace_manager import (
    load_workspace_anchor_tags,
)
from src.calibration.solvers.world_datum_aligner import WorldDatumAligner
from src.calibration.solvers.multiframe_milestone_solver import (
    MultiFrameMilestoneSolver,
    MultiFrameMilestoneReport,
)
from tools.spatial_mapping_studio.mapping_state import MappingDataManager
from src.utils.logger import get_logger

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

log = get_logger(__name__)


class MappingBARunner:
    """异步 BA 平差执行器与进度调度器"""

    def __init__(
        self,
        data_mgr: MappingDataManager,
        optimizer: BundleAdjustmentOptimizer,
        manifest_repo: ManifestRepository,
        map_path: str,
        manifest_path: str,
        marker_size_mm: float,
        on_status_change: Optional[Callable[[str], None]] = None,
        workspace: Optional[Any] = None
    ):
        self.data_mgr = data_mgr
        self.optimizer = optimizer
        self.manifest_repo = manifest_repo
        self.map_path = map_path
        self.workspace = workspace  # 当前工位对象引用 (用于锚点装载)
        if self.workspace and hasattr(self.workspace, "raw_map_path"):
            self.raw_map_path = self.workspace.raw_map_path
        elif self.map_path:
            self.raw_map_path = os.path.join(os.path.dirname(self.map_path), "calibration", "tags_map_raw.yaml")
        else:
            self.raw_map_path = ""
        self.manifest_path = manifest_path
        self.marker_size_mm = marker_size_mm
        self.on_status_change = on_status_change
        self.latest_milestone_report: Optional[MultiFrameMilestoneReport] = None

        # 运行状态与指标
        self.is_ba_running: bool = False
        self.ba_thread: Optional[threading.Thread] = None
        self.ba_result_queue: Optional[Tuple[bool, str]] = None
        self.ba_progress: float = 0.0
        self.ba_stage_text: str = ""
        self.ba_sub_progress: float = 0.0
        self.ba_sub_text: str = ""

        # 智能剪枝平差专属状态
        self.is_auto_pruning: bool = False
        self.prune_round: int = 0
        self.max_prune_rounds: int = 10
        self.min_improvement_px: float = 0.01
        self.should_stop_pruning: bool = False
        self.prune_history: List[Dict[str, Any]] = []
        self.prune_settlement_data: Optional[Dict[str, Any]] = None
        self.current_pruning_target: str = ""

        # 世界系对齐锚定配置 (从 config.yaml 动态加载, 杜绝幽灵 Tag 1)
        self.origin_tag_id: int = 0
        self.x_align_tag_id: int = 28
        # 每-Tag 世界坐标锚点表 (约束积累式锚定, 支持全知/部分已知; 兼容迁移旧 world_anchor)
        self.anchor_tags: Optional[Dict[int, Dict[str, Any]]] = None
        self._load_alignment_config()

    def _load_alignment_config(self):
        """
        锚点装载优先级链 (FR-9.6 严禁以打印边长兜底):
        ① 工位 anchor_tags.yaml (独立世界坐标数据源)
        ② 工位 tag_whitelist.yaml 的 tag_anchors (用户在白名单页签录入的已知世界坐标)
        Tag 数据已 100% 下沉至工位沙盒, 全局 config.yaml 不再持有任何 Tag ID/世界坐标.
        工位两源全空 → BA 后续将抛错终止.
        """
        if self.workspace and hasattr(self.workspace, "raw_map_path"):
            self.raw_map_path = self.workspace.raw_map_path
        elif self.map_path:
            self.raw_map_path = os.path.join(os.path.dirname(self.map_path), "calibration", "tags_map_raw.yaml")
        try:
            import yaml
            cfg_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
            calib = {}
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                calib = cfg.get("calibration", {})
                self.origin_tag_id = int(calib.get("origin_tag_id", 0))
                self.x_align_tag_id = int(calib.get("x_axis_tag_id", 28))

            ws = self.workspace
            ws_dir = getattr(ws, "workspace_dir", None) if ws else None

            # 方案 B 统一真理源: 从 tag_whitelist.yaml 加载标靶配置与锚点
            if ws_dir:
                cfg_anchors = load_workspace_anchor_tags(ws_dir)
                if cfg_anchors:
                    self.anchor_tags = cfg_anchors
                    log.info(f"[SPATIAL_MAPPING] 标靶配置真理源命中 (tag_whitelist.yaml): {sorted(self.anchor_tags.keys())}")
                    return

            # 工位未配置已知锚点
            self.anchor_tags = None
            log.warning("[SPATIAL_MAPPING] 锚点配置缺失: BA 平差将拒绝以打印边长兜底 (需在工位 tag_whitelist.yaml 录入已知世界坐标)")
        except Exception as e:
            log.warning(f"[SPATIAL_MAPPING] 读取对齐标靶配置异常: {e}")
            self.anchor_tags = None


    def _notify(self, msg: str):
        if self.on_status_change is not None:
            try:
                self.on_status_change(msg)
            except Exception:
                pass  # 状态回调失败不应中断 BA 主流程

    def _execute_ba_solve(self, callback: Optional[Any] = None) -> Tuple[bool, Optional[Dict[str, Any]], str]:
        """
        【阶段一核心】执行纯视觉自由平差求解 (Free BA):
        仅基于相机重投影误差优化，100% 独立，不依赖任何世界锚点真值，输出相对底图并持久化。
        """
        frame_detections, valid_frame_names, _ = self.manifest_repo.load_manifest(self.manifest_path)
        if len(frame_detections) < 2:
            return False, None, "有效图像帧不足 2 帧，无法执行 BA 平差"

        # 阶段一: anchor_tags=None 纯自由平差
        opt_res = self.optimizer.optimize(
            frame_detections=frame_detections,
            active_frame_names=valid_frame_names,
            origin_tag_id=self.origin_tag_id,
            x_align_tag_id=self.x_align_tag_id,
            anchor_tags=None,
            callback=callback
        )
        if opt_res and "tags" in opt_res:
            raw_tags = opt_res.get("tags", {})
            tags_dict = {}
            for tid, t_info in raw_tags.items():
                tags_dict[int(tid)] = {
                    "transform_matrix": t_info.get("transform_matrix"),
                    "position_mm": t_info.get("position_mm"),
                    "rpy_deg": t_info.get("rpy_deg", [0.0, 0.0, 0.0]),
                    "is_origin": bool(t_info.get("is_origin", False)),
                    "is_dynamic_yaw": bool(t_info.get("is_dynamic_yaw", False))
                }
            raw_map = {
                "origin_tag_id": opt_res.get("origin_tag_id", self.origin_tag_id),
                "x_axis_align_tag_id": opt_res.get("x_axis_align_tag_id", self.x_align_tag_id),
                "marker_size_mm": opt_res.get("marker_size_mm", self.marker_size_mm),
                "rmse_px": opt_res.get("final_rmse", 0.0),
                "rmse_reprojection_px": opt_res.get("rmse_reprojection_px", opt_res.get("final_rmse", 0.0)),
                "final_rmse": opt_res.get("final_rmse", 0.0),
                "calibrated_images_count": opt_res.get("calibrated_images_count", len(valid_frame_names)),
                "anchor_mode": "unaligned",
                "tags": tags_dict,
                "raw_relative_poses": opt_res.get("raw_relative_poses", {})
            }
            # 1. 固化持久化相对底图 tags_map_raw.yaml (存放于 calibration/ 专区)
            if self.raw_map_path:
                os.makedirs(os.path.dirname(self.raw_map_path), exist_ok=True)
                ManifestRepository.save_map(raw_map, self.raw_map_path)
                log.info(f"[SPATIAL_MAPPING] 相对底图已持久化至: {self.raw_map_path}")

            # 2. 同步更新视口预览地图 (若尚未做世界对齐, 视口可查看相对三维构型)
            ManifestRepository.save_map(raw_map, self.map_path)
            self.data_mgr.tags_map_data = raw_map
            if getattr(self.data_mgr, "pnp_solver", None):
                self.data_mgr.pnp_solver.tags_map = raw_map
            if raw_map.get("marker_size_mm"):
                self.data_mgr.set_marker_size_mm(raw_map["marker_size_mm"])
            self.data_mgr.refresh_all_frame_metrics()
            return True, opt_res, f"自由平差完成！像面 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (相对底图已就绪，请点击【校准世界系】)"
        return False, None, "BA 优化未能收敛，请检查有效观测标靶数"

    def execute_world_alignment(self) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        【阶段二核心】独立世界坐标系校准 (World Datum Calibration):
        基于阶段一已有的 tags_map_raw.yaml，读取最新工位世界锚点，执行 Umeyama 3D 相似变换。
        毫秒级完成，完全不触碰阶段一的图像平差，并具备几何冲突拦截！
        """
        # 1. 加载相对底图
        rel_map = None
        if os.path.exists(self.raw_map_path):
            rel_map = ManifestRepository.load_map(self.raw_map_path)
        elif self.data_mgr.tags_map_data and "tags" in self.data_mgr.tags_map_data:
            rel_map = self.data_mgr.tags_map_data

        if not rel_map or not rel_map.get("tags"):
            return False, "当前工位尚未生成相对几何地图，请先点击【全局平差】！", None

        # 2. 重新加载工位最新世界锚点配置
        self._load_alignment_config()

        # 3. 接入多坐标系分层里程碑解算器 (主干 M1/M2 + 动态 1..N 个子坐标系局部质检与熔断反推)
        coord_mgr = None
        if self.workspace:
            try:
                from src.workspace.workspace_manager import load_workspace_coordinate_manager
                coord_mgr = load_workspace_coordinate_manager(self.workspace)
            except Exception as e:
                log.debug(f"[SPATIAL_MAPPING] load_workspace_coordinate_manager: {e}")

        milestone_solver = MultiFrameMilestoneSolver(marker_size_mm=self.marker_size_mm)
        milestone_ok, milestone_rep, world_map = milestone_solver.solve(
            relative_map=rel_map,
            anchor_tags=self.anchor_tags,
            coord_mgr=coord_mgr,
            origin_tag_id=self.origin_tag_id,
            x_align_tag_id=self.x_align_tag_id,
            strict_world_datum=True,
        )
        self.latest_milestone_report = milestone_rep

        if not world_map:
            # 世界基准锚定未通过
            err_msg = milestone_rep.m2_msg or "世界基准解算未收敛"
            return False, f"世界坐标系校准拦截: {err_msg}", None

        # 4. 持久化生产世界地图 tags_map.yaml 并更新运行时引擎
        world_map["milestone_report"] = milestone_rep.to_dict()
        world_map["milestone_markdown"] = milestone_solver.format_markdown_report(milestone_rep)
        world_map["milestone_dialog_text"] = milestone_solver.format_diagnostic_dialog_text(milestone_rep)

        ManifestRepository.save_map(world_map, self.map_path)
        self.data_mgr.tags_map_data = world_map
        if getattr(self.data_mgr, "pnp_solver", None):
            self.data_mgr.pnp_solver.tags_map = world_map
        if world_map.get("marker_size_mm"):
            self.data_mgr.set_marker_size_mm(world_map["marker_size_mm"])
        self.data_mgr.refresh_all_frame_metrics()

        # 5. 【阶段三: 保存空间几何场景 (若有子坐标系外参更新)】
        if coord_mgr:
            try:
                succ_frames = sum(1 for s in milestone_rep.sub_frames.values() if s.extrinsic_solved)
                if succ_frames > 0:
                    coord_mgr.save()
                    log.info(f"[SPATIAL_MAPPING] [PHASE 3] 子坐标系外参反推完成 ({succ_frames} 个成功) 并已写穿 spatial_scene.yaml")
            except Exception as e:
                log.warning(f"[SPATIAL_MAPPING] [PHASE 3] 保存空间场景配置失败: {e}")

        w_info = world_map.get("world_anchor", {})
        res = w_info.get("anchor_residual_mm", {})
        mean_res = res.get("mean_mm", 0.0)
        max_res = res.get("max_mm", 0.0)
        has_warn = res.get("has_warn", False)
        solver = w_info.get("solver_type", "3D")
        sub_count = len(milestone_rep.sub_frames)
        sub_succ = sum(1 for s in milestone_rep.sub_frames.values() if s.extrinsic_solved)
        sub_isolated = sum(1 for s in milestone_rep.sub_frames.values() if s.is_isolated)
        sub_tip = f" | 子坐标系外参: {sub_succ}/{sub_count}" if sub_count > 0 else ""
        if sub_isolated > 0:
            sub_tip += f" (⚠️ {sub_isolated}个局部超差已熔断)"

        if has_warn or sub_isolated > 0:
            warn_desc = "⚠️注意：存在偏差过大标靶/隔离项" if sub_isolated > 0 else "⚠️注意：存在偏差过大标靶"
            msg = f"世界系校准完成({warn_desc})：[{solver}] 均值残差: {mean_res:.2f}mm, 最大偏差: {max_res:.2f}mm{sub_tip}"
        else:
            msg = f"世界坐标系校准成功！[{solver}] 锚点残差均值: {mean_res:.2f} mm，生产地图已更新。{sub_tip}"
        self._notify(msg)
        log.info(f"[SPATIAL_MAPPING] {msg}")
        return True, msg, world_map

    def _calibrate_sub_frames(self, world_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        【阶段三核心】反推工位内所有待定 (unknown) 子坐标系的外参，并原子写穿 frames.yaml。
        """
        ws = self.workspace
        if not ws:
            return {}
        try:
            from src.workspace.workspace_manager import (
                load_workspace_coordinate_manager,
                load_workspace_anchor_tags,
            )
            from src.calibration.solvers.frame_extrinsic_solver import FrameExtrinsicSolver

            coord_mgr = load_workspace_coordinate_manager(ws)
            if not coord_mgr:
                return {}

            whitelist_anchors = load_workspace_anchor_tags(getattr(ws, "workspace_dir", "")) or {}
            solver = FrameExtrinsicSolver(tags_map=world_map, whitelist_anchors=whitelist_anchors)
            solved_summary = solver.solve_all_unknown_frames(coord_mgr)

            succ_count = sum(1 for item in solved_summary.values() if item.get("success"))
            if succ_count > 0:
                coord_mgr.save()
                log.info(f"[SPATIAL_MAPPING] [PHASE 3] 子坐标系外参反推完成 ({succ_count} 个成功) 并已写穿 spatial_scene.yaml: {list(solved_summary.keys())}")
            return solved_summary
        except Exception as e:
            log.warning(f"[SPATIAL_MAPPING] [PHASE 3] 子坐标系外参反推异常 (跳过): {e}")
            return {}

    def start(self) -> bool:
        """
        启动后台线程执行两阶段全局 BA 平差优化
        :return: True 如果成功启动，False 如果已有任务在运行
        """
        if self.is_ba_running:
            self._notify("平差优化已在运行中，请稍候...")
            return False

        self.is_ba_running = True
        self.is_auto_pruning = False
        self.ba_progress = 0.05
        self.ba_sub_progress = 0.0
        self.ba_stage_text = "正在启动两阶段全局 BA 平差优化计算..."
        self.ba_sub_text = "初始化优化工作空间..."
        self._notify("正在启动两阶段全局 BA 平差优化计算...")
        log.info("\n[*] [SPATIAL_MAPPING] 正在启动异步 BA 全局平差优化计算...")

        def _worker():
            try:
                self.ba_progress = 0.10
                self.ba_sub_progress = 0.30
                self.ba_stage_text = "阶段 1/4: 准备观测清单与共视拓扑分析..."
                self.ba_sub_text = "校验数据清单并分析共视连通性图..."
                self.data_mgr._save_manifest()
                time.sleep(0.05)

                self.ba_progress = 0.25
                self.ba_sub_progress = 0.60
                self.ba_stage_text = "阶段 2/4: 过滤已剔除样本并构建全局初值..."
                self.ba_sub_text = "构建超定 PnP 初始机位与标靶三维姿态..."

                def on_ba_callback(info: Dict[str, Any]):
                    stg = info.get("stage", 1)
                    stg_name = info.get("stage_name", "")
                    cur_it = info.get("iter", 0)
                    max_it = info.get("max_iter", 1)
                    cur_rmse = info.get("rmse", 0.0)
                    sub_pct = max(0.0, min(1.0, info.get("sub_progress", 0.0)))

                    if stg == 1:
                        self.ba_progress = 0.40 + 0.22 * sub_pct
                    else:
                        self.ba_progress = 0.62 + 0.22 * sub_pct

                    self.ba_sub_progress = sub_pct
                    display_max = max(max_it, cur_it)
                    self.ba_sub_text = f"[{stg_name}] 轮次 #{cur_it}/{display_max} | 实时 RMSE: {cur_rmse:.3f} px"

                self.ba_progress = 0.40
                self.ba_stage_text = "阶段 3/4: 两阶段 Cauchy 稳健核平差全局收敛求解..."
                succ, opt_res, msg = self._execute_ba_solve(callback=on_ba_callback)

                if succ:
                    self.ba_progress = 1.0
                    self.ba_sub_progress = 1.0
                    self.ba_stage_text = "全局平差完成！正在同步就地热更新..."
                    self.ba_sub_text = f"最终全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px (已就地生效)"
                    self.ba_result_queue = (True, f"BA 优化成功！新全局 RMSE: {opt_res.get('final_rmse', 0.0):.3f} px")
                else:
                    self.ba_result_queue = (False, msg)
            except Exception as e:
                self.ba_result_queue = (False, f"BA 平差优化异常: {e}")

        self.ba_thread = threading.Thread(target=_worker, daemon=True)
        self.ba_thread.start()
        return True

    def start_auto_prune(self, max_rounds: int = 10, min_improvement_px: float = 0.01) -> bool:
        """
        启动后台线程执行基于边际收益收敛与共视拓扑守门的全自动迭代剪枝平差
        """
        if self.is_ba_running:
            self._notify("平差优化计算中，请稍候...")
            return False

        # 启动前自动制作状态快照
        self.data_mgr.create_manifest_snapshot()
        self.data_mgr.sort_mode = "err_desc"

        self.is_ba_running = True
        self.is_auto_pruning = True
        self.should_stop_pruning = False
        self.prune_round = 0
        self.max_prune_rounds = max_rounds
        self.min_improvement_px = min_improvement_px
        self.prune_history = []
        self.prune_settlement_data = None
        self.current_pruning_target = ""
        self.initial_rmse = self.data_mgr.global_rmse

        self.ba_progress = 0.05
        self.ba_sub_progress = 0.0
        self.ba_stage_text = "正在启动工序 5-Auto: 迭代残差剪枝平差..."
        self.ba_sub_text = "创建状态快照并准备首轮基准平差..."
        self._notify("智能剪枝平差启动: 已制作状态快照")
        log.info("\n[*] [SPATIAL_MAPPING AUTO-PRUNE] 启动自动迭代残差剪枝平差...")

        def _auto_prune_worker():
            try:
                # 1. 确保有初始有效基准 RMSE
                init_rmse = self.data_mgr.global_rmse
                init_mm = self.data_mgr.global_median_mm
                if init_rmse <= 0.0:
                    self.ba_stage_text = "初始化: 运行基准平差获取初始误差网格..."
                    succ, _, msg = self._execute_ba_solve()
                    if not succ:
                        self.ba_result_queue = (False, f"初始基准平差失败: {msg}")
                        self.is_auto_pruning = False
                        return
                    init_rmse = self.data_mgr.global_rmse
                    init_mm = self.data_mgr.global_median_mm

                self.initial_rmse = init_rmse
                prev_rmse = init_rmse
                stop_reason = "达到最大预设轮数"

                # 初始化逐帧多轮残差收敛矩阵: 首列为 R0(基准)
                self.data_mgr.convergence_headers = ["R0"]
                self.data_mgr.frame_convergence_matrix = {}
                for p in self.data_mgr.image_files:
                    bn = os.path.basename(p)
                    meta = self.data_mgr.frame_metrics_cache.get(bn, {})
                    err = meta.get("mean_err", 0.0)
                    is_excl = meta.get("is_excluded", False)
                    val = round(float(err), 2) if (not is_excl and err is not None) else None
                    self.data_mgr.frame_convergence_matrix[bn] = [val]

                # 2. 迭代剪枝主循环
                for r in range(1, self.max_prune_rounds + 1):
                    if self.should_stop_pruning:
                        stop_reason = "用户手动急停"
                        break

                    self.prune_round = r
                    self.ba_progress = min(0.95, 0.10 + 0.85 * (r / self.max_prune_rounds))
                    self.ba_stage_text = f"智能剪枝平差 (第 {r}/{self.max_prune_rounds} 轮): 寻找最大离差安全样本..."

                    # 寻找本轮最大离差的 1~2 个安全标靶
                    prunable = self.data_mgr.find_worst_prunable_observations(top_k=2)
                    if not prunable:
                        stop_reason = "拓扑安全守门触发: 已无安全可剔除标靶"
                        log.info(f"[*] [AUTO-PRUNE] 轮次 #{r}: 已无安全可剔除项，安全停机。")
                        break

                    # 格式化剔除描述
                    p_desc = ", ".join([f"{bname} 的 #{tid} ({err:.2f}px)" for bname, tid, err, _ in prunable])
                    self.current_pruning_target = p_desc
                    self.ba_stage_text = f"智能剪枝平差 (第 {r}/{self.max_prune_rounds} 轮): 正在重平差求解全场景优化..."
                    self.ba_sub_text = f"本轮淘汰: {p_desc} -> 全局 BA 优化求解中..."
                    log.info(f"\n[======== [AUTO-PRUNE] 剪枝平差 第 {r}/{self.max_prune_rounds} 轮 ========]")
                    log.info(f"[*] 拟剔除坏样本: {p_desc}")
                    log.info(f"[*] 剔除前全局基准 RMSE: {prev_rmse:.3f} px (中位数重投影误差: {self.data_mgr.global_median_mm:.3f} mm)")

                    # 轮次前制作即时快照 (用于防反弹单调保护)
                    round_snapshot = self.data_mgr.create_manifest_snapshot()

                    # 执行剔除
                    self.data_mgr.prune_observations(prunable)

                    # 重新运行平差求解
                    log.info(f"[*] 正在重平差求解全场景优化...")
                    succ, _, msg = self._execute_ba_solve()
                    self.current_pruning_target = ""
                    if not succ:
                        stop_reason = f"平差求解发散: {msg} (已自动恢复本轮前状态)"
                        log.warning(f"[!] [AUTO-PRUNE] 求解发散: {msg}，自动回滚本轮剔除并停机。")
                        self.data_mgr.restore_manifest_snapshot(round_snapshot)
                        break

                    new_rmse = self.data_mgr.global_rmse
                    new_mm = self.data_mgr.global_median_mm
                    delta_rmse = prev_rmse - new_rmse

                    # 防反弹单调刚性保护: 若剔除导致误差反弹上升 (delta_rmse < 0)，说明该标靶是关键拓扑支撑点，自动回滚并锁定最优停机
                    if delta_rmse < -0.01:
                        stop_reason = f"触发刚性拓扑保护: 拟淘汰标靶属于关键支撑骨架，剔除后残差反弹 (+{abs(delta_rmse):.2f}px)，已自动回滚并锁定最优收敛状态"
                        log.warning(f"\n[!] [AUTO-PRUNE] 警告: 本轮剔除导致平差残差反弹 (从 {prev_rmse:.3f} px 恶化至 {new_rmse:.3f} px)!")
                        log.warning(f"[*] 正在自动回滚撤销本轮剔除，并精准恢复至最优地图状态...")
                        self.data_mgr.restore_manifest_snapshot(round_snapshot)
                        log.info(f"[✓] 已安全恢复至最优 RMSE: {prev_rmse:.3f} px，自动触发最优收敛停机！\n")
                        break

                    # 本轮求解成功且有效，横向自动增加一列记录各图像在求解后的最新残差
                    self.data_mgr.convergence_headers.append(f"R{r}")
                    for p in self.data_mgr.image_files:
                        bn = os.path.basename(p)
                        meta = self.data_mgr.frame_metrics_cache.get(bn, {})
                        err = meta.get("mean_err", 0.0)
                        is_excl = meta.get("is_excluded", False)
                        val = round(float(err), 2) if (not is_excl and err is not None) else None
                        if bn not in self.data_mgr.frame_convergence_matrix:
                            self.data_mgr.frame_convergence_matrix[bn] = []
                        self.data_mgr.frame_convergence_matrix[bn].append(val)

                    self.prune_history.append({
                        "round": r,
                        "pruned": prunable,
                        "rmse_before": prev_rmse,
                        "rmse_after": new_rmse,
                        "delta_rmse": delta_rmse,
                        "median_mm": new_mm
                    })

                    log.info(f"[✓] 轮次 #{r} 完成: RMSE 从 {prev_rmse:.3f} px -> {new_rmse:.3f} px (改善幅度: {delta_rmse:+.3f} px)")

                    # 核心终止判定: 边际收益见顶
                    if delta_rmse < self.min_improvement_px:
                        stop_reason = f"边际收益见顶 (本轮改善 {delta_rmse:.4f} px < 门限 {self.min_improvement_px:.3f} px)"
                        log.info(f"[*] [AUTO-PRUNE] 改善幅度低于边际门限，已达最优收敛状态，自动停机！")
                        break

                    prev_rmse = new_rmse

                # 3. 生成结算对比卡片数据包
                import copy
                final_rmse = self.data_mgr.global_rmse
                final_mm = self.data_mgr.global_median_mm
                total_pruned = sum(len(item["pruned"]) for item in self.prune_history)

                self.prune_settlement_data = {
                    "initial_rmse": init_rmse,
                    "initial_mm": init_mm,
                    "final_rmse": final_rmse,
                    "final_mm": final_mm,
                    "rounds_executed": len(self.prune_history),
                    "total_pruned_count": total_pruned,
                    "stop_reason": stop_reason,
                    "history": self.prune_history,
                    "convergence_headers": list(self.data_mgr.convergence_headers),
                    "frame_convergence_matrix": copy.deepcopy(self.data_mgr.frame_convergence_matrix)
                }

                self.ba_progress = 1.0
                self.ba_sub_progress = 1.0
                self.ba_stage_text = "智能剪枝平差完成！请在结算卡片中确认采纳或撤销"
                self.ba_sub_text = f"优化成效: RMSE {init_rmse:.2f}px -> {final_rmse:.2f}px (剔除 {total_pruned} 个外点)"
                self.ba_result_queue = (True, f"智能剪枝平差结束: {stop_reason}，累计剔除 {total_pruned} 个外点")
                print("\n==================================================", flush=True)
                print(f"[★] [AUTO-PRUNE] 智能剪枝平差全流程完成！", flush=True)
                print(f"    - 执行轮次: {len(self.prune_history)} 轮", flush=True)
                print(f"    - 累计剔除外点: {total_pruned} 个", flush=True)
                print(f"    - 全局 RMSE: {init_rmse:.3f} px  ==>  {final_rmse:.3f} px", flush=True)
                print(f"    - 停止原因: {stop_reason}", flush=True)
                print("==================================================\n", flush=True)
            except Exception as e:
                import traceback
                log.warning(f"\n[!] [AUTO-PRUNE ERROR] 智能剪枝平差发生未捕获异常: {e}")
                traceback.print_exc()
                self.ba_result_queue = (False, f"智能剪枝平差异常: {e}")
            finally:
                self.is_auto_pruning = False

        self.ba_thread = threading.Thread(target=_auto_prune_worker, daemon=True)
        self.ba_thread.start()
        return True

    def request_stop_pruning(self):
        """请求中途安全急停智能剪枝平差"""
        if self.is_auto_pruning:
            self.should_stop_pruning = True
            self._notify("已发送急停请求，将在当前平差轮次完成后安全停止...")

    def poll_result(self) -> Optional[Tuple[bool, str]]:
        """
        检查异步 BA 任务是否完成
        :return: (is_success, message) 或 None
        """
        if self.ba_result_queue is not None:
            result = self.ba_result_queue
            self.ba_result_queue = None
            self.is_ba_running = False
            return result
        return None
