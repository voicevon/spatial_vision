#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
空间建图数据变更动作 Mixin (Mapping Data Actions)
==============================================
从 mapping_state.py 拆出的变更型动作层 (无状态 Mixin, 不新增实例属性):
  - 帧/观测保留剔除翻转: toggle_image_exclusion / toggle_observation_keep /
    toggle_current_frame_exclusion / toggle_tag_exclusion_in_current_frame
  - 超精重提取: super_extract_current_frame / super_extract_all_frames
  - 智能剪枝: find_worst_prunable_observations / prune_observations
  - 复位与快照: reset_map / reset_all_keep_status / create_manifest_snapshot / restore_manifest_snapshot
所有方法通过 self 访问 MappingDataManager 的数据状态与持久化能力。
"""

import os
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import yaml
from src.utils.logger import get_logger

log = get_logger(__name__)


class MappingDataActionsMixin:
    """空间建图工作站数据变更动作 (供 MappingDataManager 继承)"""

    def toggle_image_exclusion(self, base_name: str) -> bool:
        """翻转单张图像的保留/剔除状态并持久化 (双向同步 excluded 与 enabled)"""
        self.manifest_data.setdefault("images", {}).setdefault(base_name, {})
        curr = self.is_image_excluded(base_name)
        new_excluded = not curr
        self.manifest_data["images"][base_name]["excluded"] = new_excluded
        self.manifest_data["images"][base_name]["enabled"] = not new_excluded
        self._sync_manifest_summary()
        self._save_manifest()
        return new_excluded

    def super_extract_current_frame(self, img_idx: Optional[int] = None) -> Tuple[str, int]:
        """
        对指定帧或当前选定帧执行工序 3 工业级超精重提取并原子持久化
        动用 5 路增强底图 + 16 级致密网格 + 微靶 2x 超分 + 0.01px 轮廓正交亚像素精修
        """
        if img_idx is None:
            img_idx = self.current_img_idx
        if not self.image_files or img_idx >= len(self.image_files):
            return "", 0

        cur_file = self.image_files[img_idx]
        bname = os.path.basename(cur_file)

        # 动用工序 3 离线超精提取引擎
        super_res = self.super_extractor.extract_from_image(cur_file)

        # 智能继承已有的人工保留/剔除决策与备注
        existing_img_entry = self.manifest_data.get("images", {}).get(bname, {})
        existing_prefs = {}
        for obs in existing_img_entry.get("observations", []):
            tid = obs.get("tag_id")
            existing_prefs[tid] = (obs.get("keep", True), obs.get("note", ""))

        new_obs = []
        for tid in sorted(super_res.keys()):
            item = super_res[tid]
            keep_val, note_val = existing_prefs.get(tid, (True, f"超精提取 [{item.get('channel', 'SUPER')}]"))
            m = item.get("metrics", {})
            c_arr = item["corners"]
            new_obs.append({
                "tag_id": int(tid),
                "keep": bool(keep_val),
                "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in c_arr.reshape(4, 2)],
                "cell_size_px": m.get("cell_size_px", [0, 0]),
                "center_px": m.get("center_px", [0.0, 0.0]),
                "area_px": m.get("area_px", 0.0),
                "channel": item.get("channel", "SUPER"),
                "note": str(note_val)
            })

        # 组装符合 Manifest 标准格式的完整结构
        self.manifest_data.setdefault("images", {})[bname] = {
            "file_name": bname,
            "image_path": cur_file.replace("\\", "/"),
            "enabled": not self.is_image_excluded(bname),
            "excluded": self.is_image_excluded(bname),
            "detected_count": len(new_obs),
            "observations": new_obs
        }

        # 更新 summary 统计
        self._sync_manifest_summary()

        # 立即原子持久化落盘
        self._save_manifest()

        # 刷新体检指标缓存
        self.refresh_all_frame_metrics()
        log.info(f"[OK] [STUDIO] 帧 {bname} 超精重提取完成并已原子持久化: 检出 {len(new_obs)} 个标靶")
        return bname, len(new_obs)

    def super_extract_all_frames(
        self,
        progress_callback: Optional[Any] = None
    ) -> Tuple[int, int]:
        """
        全局全量超精重提取：
        对所有采图帧原有提取的角点与空间观测完全清空删除，
        从头开始重新调用工序 3 工业级超精重提取引擎 (5路增强 + 16级致密网格 + 2x超分 + 0.01px轮廓正交亚像素拟合)
        提取所有标靶角点，默认全部从头启用为有效保留状态，并原子持久化至 manifest_path。
        :param progress_callback: 可选进度回调 callback(cur_idx, total_count, bname, tag_count)
        :return: (处理的总帧数, 累计检出的标靶总数)
        """
        if not self.image_files:
            return 0, 0

        total_frames = len(self.image_files)
        total_tags = 0

        # 清空原有的所有帧 observations，从头彻底重建 images 字典
        new_images_dict: Dict[str, Any] = {}

        for i, cur_file in enumerate(self.image_files):
            bname = os.path.basename(cur_file)

            # 调用工序 3 离线超精提取引擎
            try:
                super_res = self.super_extractor.extract_from_image(cur_file)
            except Exception as e:
                log.warning(f"[STUDIO] 超精提取帧 {bname} 异常: {e}")
                super_res = {}

            new_obs = []
            for tid in sorted(super_res.keys()):
                item = super_res[tid]
                m = item.get("metrics", {})
                c_arr = item["corners"]
                new_obs.append({
                    "tag_id": int(tid),
                    "keep": True,  # 全部从头开始重新启用
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in c_arr.reshape(4, 2)],
                    "cell_size_px": m.get("cell_size_px", [0, 0]),
                    "center_px": m.get("center_px", [0.0, 0.0]),
                    "area_px": m.get("area_px", 0.0),
                    "channel": item.get("channel", "SUPER"),
                    "note": f"全局超精提取 [{item.get('channel', 'SUPER')}]"
                })

            new_images_dict[bname] = {
                "file_name": bname,
                "image_path": cur_file.replace("\\", "/"),
                "enabled": True,
                "excluded": False,
                "detected_count": len(new_obs),
                "observations": new_obs
            }
            total_tags += len(new_obs)

            if progress_callback is not None:
                try:
                    progress_callback(i + 1, total_frames, bname, len(new_obs))
                except Exception:
                    pass  # 进度回调失败不影响批量提取主流程

        self.manifest_data["images"] = new_images_dict

        # 更新 summary 统计
        self._sync_manifest_summary()

        # 立即原子持久化落盘
        self._save_manifest()

        # 刷新所有体检指标
        self.refresh_all_frame_metrics()
        log.info(f"[OK] [STUDIO] 全局全量超精提取完成并持久化: 共处理 {total_frames} 帧，累计检出 {total_tags} 个标靶")
        return total_frames, total_tags

    def reset_map(self) -> bool:
        """
        清空当前解算的空间立体地图，使工作站完全复位为未建图的初始状态
        同时将已存在的 tags_map.yaml 自动备份为 tags_map.yaml.bak 并写空
        """
        if os.path.exists(self.map_path):
            try:
                bak_path = self.map_path + ".bak"
                import shutil
                shutil.copyfile(self.map_path, bak_path)
                log.info(f"[*] [STUDIO] 旧地图已安全备份至: {bak_path}")
            except Exception as e:
                log.warning(f"备份地图失败: {e}")

        # 清空内存与地图
        self.tags_map_data = {"version": "2.0_reset", "tags": {}}
        if getattr(self, "pnp_solver", None):
            self.pnp_solver.tags_map = self.tags_map_data

        # 写回空地图文件
        try:
            with open(self.map_path, "w", encoding="utf-8") as f:
                yaml.dump(self.tags_map_data, f, allow_unicode=True, sort_keys=False)
            log.info(f"[OK] [STUDIO] 地图文件已成功复位清空: {self.map_path}")
        except Exception as e:
            log.warning(f"写入空地图文件失败: {e}")

        self.refresh_all_frame_metrics()
        return True

    def reset_all_keep_status(self) -> int:
        """
        一键复位全量观测状态：将所有帧设为保留 (enabled: true, excluded: false)，所有标靶观测设为有效 (keep: true)
        :return: 恢复的总观测数量
        """
        restored_count = 0
        images_dict = self.manifest_data.get("images", {})
        for img_info in images_dict.values():
            img_info["enabled"] = True
            img_info["excluded"] = False
            for obs in img_info.get("observations", []):
                obs["keep"] = True
                restored_count += 1

        self._sync_manifest_summary()
        self._save_manifest()
        self.refresh_all_frame_metrics()
        log.info(f"[OK] [STUDIO] 已一键复位所有标靶保留状态: 共恢复 {restored_count} 次标靶观测为有效")
        return restored_count

    def toggle_observation_keep(self, base_name: str, tag_id: int) -> bool:
        """翻转某帧中特定标靶的保留/剔除状态并持久化"""
        obs_list = self.get_observations_for_image(base_name)
        new_keep = False
        for obs in obs_list:
            if obs.get("tag_id") == tag_id:
                curr_keep = obs.get("keep", True)
                new_keep = not curr_keep
                obs["keep"] = new_keep
                break
        self._save_manifest()
        return new_keep

    def toggle_current_frame_exclusion(self) -> Tuple[str, bool]:
        """翻转当前选中帧的保留/剔除状态"""
        if not self.image_files or self.current_img_idx >= len(self.image_files):
            return "", False
        cur_file = self.image_files[self.current_img_idx]
        base_name = os.path.basename(cur_file)

        is_now_excluded = self.toggle_image_exclusion(base_name)
        if base_name in self.frame_metrics_cache:
            self.frame_metrics_cache[base_name]["is_excluded"] = is_now_excluded
        return base_name, is_now_excluded

    def toggle_tag_exclusion_in_current_frame(self, target_tag_id: int) -> Tuple[str, bool]:
        """翻转当前帧中特定标靶的保留/剔除状态并重算当前帧指标"""
        if not self.image_files or self.current_img_idx >= len(self.image_files):
            return "", False
        cur_file = self.image_files[self.current_img_idx]
        base_name = os.path.basename(cur_file)

        is_kept = self.toggle_observation_keep(base_name, target_tag_id)
        meta = self.frame_metrics_cache.get(base_name, {})
        obs_list = self.get_observations_for_image(base_name)
        mean_err, max_err, errors_dict, rvec, tvec, mean_mm, errs_mm = self._evaluate_frame_reprojection(obs_list)
        meta["observations"] = obs_list
        meta["mean_err"] = mean_err
        meta["max_err"] = max_err
        meta["mean_err_mm"] = mean_mm
        meta["tag_errors"] = errors_dict
        meta["tag_errors_mm"] = errs_mm
        meta["rvec"] = rvec
        meta["tvec"] = tvec
        return base_name, is_kept

    def create_manifest_snapshot(self) -> Dict[str, Any]:
        """创建当前审核清单与地图的状态快照，支持无损撤销"""
        import copy
        self._manifest_snapshot = {
            "manifest_data": copy.deepcopy(self.manifest_data),
            "tags_map_data": copy.deepcopy(self.tags_map_data),
            "global_rmse": self.global_rmse,
            "global_median_mm": self.global_median_mm,
            "global_mean_mm": self.global_mean_mm,
            "gate_status": self.gate_status,
            "topology_status": copy.deepcopy(self.topology_status),
            "frame_convergence_matrix": copy.deepcopy(self.frame_convergence_matrix),
            "convergence_headers": copy.deepcopy(self.convergence_headers)
        }
        return self._manifest_snapshot

    def restore_manifest_snapshot(self, snapshot: Optional[Dict[str, Any]] = None) -> bool:
        """从快照恢复审核清单与地图状态"""
        import copy
        snap = snapshot or getattr(self, "_manifest_snapshot", None)
        if not snap:
            return False
        self.manifest_data = copy.deepcopy(snap["manifest_data"])
        self.tags_map_data = copy.deepcopy(snap["tags_map_data"])
        self.global_rmse = snap["global_rmse"]
        self.global_median_mm = snap["global_median_mm"]
        self.global_mean_mm = snap["global_mean_mm"]
        self.gate_status = snap["gate_status"]
        self.topology_status = copy.deepcopy(snap["topology_status"])
        if "frame_convergence_matrix" in snap:
            self.frame_convergence_matrix = copy.deepcopy(snap["frame_convergence_matrix"])
        if "convergence_headers" in snap:
            self.convergence_headers = copy.deepcopy(snap["convergence_headers"])

        self._save_manifest()
        if self.map_path and os.path.exists(os.path.dirname(self.map_path)):
            try:
                with open(self.map_path, "w", encoding="utf-8") as f:
                    yaml.dump(self.tags_map_data, f, allow_unicode=True, sort_keys=False)
            except Exception as e:
                log.warning(f"恢复地图文件异常: {e}")
        self.refresh_all_frame_metrics()
        return True

    def find_worst_prunable_observations(
        self,
        top_k: int = 2
    ) -> List[Tuple[str, int, float, float]]:
        """
        寻找当前全局残差最大的 Top-K 个有效标靶观测，严格受共视拓扑与最小观测度（>=2次）保护
        返回: [(image_basename, tag_id, err_px, err_mm)]
        """
        from src.calibration.covisibility_graph import CovisibilityGraphAnalyzer
        import copy

        # 1. 收集全局所有有效观测及其残差
        candidates = []
        tag_active_counts: Dict[int, int] = {}
        for bname, meta in self.frame_metrics_cache.items():
            if meta.get("is_excluded", False):
                continue
            tag_errors = meta.get("tag_errors", {})
            tag_errors_mm = meta.get("tag_errors_mm", {})
            for obs in meta.get("observations", []):
                if not obs.get("keep", True):
                    continue
                tid = obs["tag_id"]
                tag_active_counts[tid] = tag_active_counts.get(tid, 0) + 1
                err_px = tag_errors.get(tid, 0.0)
                err_mm = tag_errors_mm.get(tid, 0.0)
                candidates.append((err_px, err_mm, bname, tid))

        # 按像素残差降序排列
        candidates.sort(key=lambda item: item[0], reverse=True)

        selected = []
        sim_manifest = copy.deepcopy(self.manifest_data)
        sim_counts = dict(tag_active_counts)

        for err_px, err_mm, bname, tid in candidates:
            if len(selected) >= top_k:
                break
            if err_px <= 0.0:
                continue

            # 守门规则 1: 剔除后该 Tag 的全局观测次数不得 < 2
            if sim_counts.get(tid, 0) - 1 < 2:
                continue

            # 守门规则 2: 模拟剔除后用共视拓扑图审查，必须保持单一连通且无孤岛
            obs_entry = None
            for o in sim_manifest.get("images", {}).get(bname, {}).get("observations", []):
                if o.get("tag_id") == tid:
                    obs_entry = o
                    break
            if obs_entry is None:
                continue

            obs_entry["keep"] = False

            # 正确构建 detections 结构传递给 CovisibilityGraphAnalyzer
            sim_detections = []
            sim_names = []
            for sim_bname, sim_img in sim_manifest.get("images", {}).items():
                if sim_img.get("excluded", False) or not sim_img.get("enabled", True):
                    continue
                d_map = {}
                for o in sim_img.get("observations", []):
                    if o.get("keep", True):
                        d_map[o["tag_id"]] = np.array(o["corners"], dtype=np.float32)
                if d_map:
                    sim_detections.append(d_map)
                    sim_names.append(sim_bname)

            topo_res = CovisibilityGraphAnalyzer.analyze(sim_detections, sim_names)
            if not topo_res.get("is_valid", False) or topo_res.get("unconnected_tags"):
                obs_entry["keep"] = True
                continue

            selected.append((bname, tid, err_px, err_mm))
            sim_counts[tid] -= 1

        return selected

    def prune_observations(self, prune_list: List[Tuple[str, int, float, float]]) -> int:
        """批量将指定的观测标靶标记为剔除状态并持久化与更新指标"""
        count = 0
        for bname, tid, _, _ in prune_list:
            img_entry = self.manifest_data.get("images", {}).get(bname, {})
            for obs in img_entry.get("observations", []):
                if obs.get("tag_id") == tid and obs.get("keep", True):
                    obs["keep"] = False
                    count += 1
                    break
        if count > 0:
            self._save_manifest()
            self.refresh_all_frame_metrics()
        return count
