# -*- coding: utf-8 -*-
"""
Workspace Hub 几何与空间物件状态机 (GeometryState)
================================================
管理工位坐标系树、3D ROI 空间物件及模态交互表单：
1. 机构相对坐标系结构 (FrameDefinition) 与 6DoF 外参状态 (known/unknown/planar)
2. 3D ROI 空间物件 (RoiDefinition)、Smart ROI 工艺角色 (source/destination/keepout/general) 与动作意图
3. 机构坐标系模态弹窗 (FrameModal) 表单缓存与下拉选择框生命周期
4. 6DoF 外参位姿与自由度约束独立模态弹窗 (Pose6DModal) 软键盘交互与快捷预设
5. 3D ROI 空间物件模态弹窗 (RoiModal) 表单缓存与槽位绑定
6. 坐标系分配 Tag 命名空间区间 (10-Slot) 与区间放行状态同步
"""

import os
from typing import Any, Optional
import yaml
from src.utils.logger import get_logger

log = get_logger(__name__)


class GeometryState:
    """机构相对坐标系与 3D ROI 空间物件状态机"""

    # 3D ROI 空间物件列表单屏可见行数 (高 552px / 80px = 6.9 -> 7行完整铺满)
    ROI_VISIBLE_COUNT = 7

    def __init__(self, parent_hub_state: Any):
        self.hub = parent_hub_state

        # 底层核心管理器实例
        self.coord_mgr = None
        self.roi_mgr = None
        self._coord_mgr_cache: dict[str, Any] = {}
        self._roi_mgr_cache: dict[str, Any] = {}

        # 3D ROI 物件列表滚动偏移
        self.roi_scroll_offset = 0

        # 下拉选择框激活状态 ("frame_type" / "frame_parent" / "roi_category" / "roi_frame" / None)
        self.active_dropdown: str | None = None

        # 机构坐标系弹窗状态
        self.frame_modal_open = False
        self.frame_modal_is_new = False
        self.frame_modal_orig_id: str | None = None
        self.frame_modal_data: dict = {}

        # 6DoF 外参位姿独立模态窗
        self.pose6d_modal_open = False
        self.pose6d_modal_vals = [0.0] * 6
        self.pose6d_modal_known = [False] * 6
        self.pose6d_modal_axis_sel = 0
        self.pose6d_modal_axis_buf = ""

        # 3D ROI 空间物件弹窗状态
        self.roi_modal_open = False
        self.roi_modal_is_new = False
        self.roi_modal_orig_id: str | None = None
        self.roi_modal_data: dict = {}

    def load_geometry_managers(self):
        """加载当前选中工位的多坐标系与 ROI 管理器 (利用内存缓存杜绝重复磁盘 I/O)"""
        ws = self.hub.get_selected_workspace()
        if ws:
            ws_id = ws.workspace_id
            if ws_id in self._coord_mgr_cache and ws_id in self._roi_mgr_cache:
                self.coord_mgr = self._coord_mgr_cache[ws_id]
                self.roi_mgr = self._roi_mgr_cache[ws_id]
            else:
                from src.calibration.workspace_manager import (
                    load_workspace_coordinate_manager,
                    load_workspace_roi_manager
                )
                self.coord_mgr = load_workspace_coordinate_manager(ws)
                self.roi_mgr = load_workspace_roi_manager(ws)
                self._coord_mgr_cache[ws_id] = self.coord_mgr
                self._roi_mgr_cache[ws_id] = self.roi_mgr
        else:
            self.coord_mgr = None
            self.roi_mgr = None

    def refresh_geometry_cache(self, ws_id: Optional[str] = None):
        """失效几何管理器缓存并按需重新载入"""
        if ws_id:
            self._coord_mgr_cache.pop(ws_id, None)
            self._roi_mgr_cache.pop(ws_id, None)
        else:
            self._coord_mgr_cache.clear()
            self._roi_mgr_cache.clear()
        self.load_geometry_managers()

    def get_workspace_coord_mgr(self, ws) -> Any:
        """获取指定工位的坐标系管理器（带内存缓存，避免渲染树形结构每帧重复加载磁盘 YAML）"""
        if not ws:
            return None
        ws_id = ws.workspace_id
        if ws_id in self._coord_mgr_cache:
            return self._coord_mgr_cache[ws_id]
        from src.calibration.workspace_manager import load_workspace_coordinate_manager
        mgr = load_workspace_coordinate_manager(ws)
        self._coord_mgr_cache[ws_id] = mgr
        return mgr

    def get_coordinate_frames(self):
        """获取当前工位的所有坐标系定义列表"""
        if self.coord_mgr:
            return self.coord_mgr.list_frames()
        return []

    def get_roi_spaces(self):
        """获取当前工位的所有 ROI 空间物件列表"""
        if self.roi_mgr:
            return self.roi_mgr.list_rois()
        return []


    def get_frame_tag_range(self, frame_id: str) -> list[int]:
        """获取坐标系分配的专属 Tag ID 命名空间区间 (0~9, 10~19, 20~29...)"""
        if self.coord_mgr:
            return self.coord_mgr.get_frame_tag_range(frame_id)
        if frame_id == "world":
            return list(range(0, 10))
        return list(range(10, 20))

    def toggle_frame_tag_allowed(self, frame_id: str, tag_id: int) -> bool:
        """在当前坐标系专属区间内切换某 Tag 的放行状态，并原子写穿工位 tag_whitelist.yaml"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return False
        import os
        import yaml
        wl_path = self.hub.workspace_mgr.ensure_tag_whitelist(ws.workspace_id)
        curr_cfg = {}
        if os.path.isfile(wl_path):
            try:
                with open(wl_path, "r", encoding="utf-8") as f:
                    curr_cfg = yaml.safe_load(f) or {}
            except Exception:
                curr_cfg = {}
        raw_allowed = curr_cfg.get("allowed_ids") or []
        allowed = {int(x) for x in raw_allowed if str(x).isdigit()}
        if tag_id in allowed:
            allowed.remove(tag_id)
            now_allowed = False
        else:
            allowed.add(tag_id)
            now_allowed = True
        curr_cfg["allowed_ids"] = sorted(list(allowed))
        with open(wl_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(curr_cfg, f, allow_unicode=True)
        self.hub.whitelist.refresh_whitelist_cache()
        action_desc = "放行" if now_allowed else "禁行"
        self.hub.set_toast(f"坐标系 [{frame_id}] 标靶 Tag #{tag_id} 已{action_desc} (已同步工位白名单)")
        return now_allowed

    def get_frame_tags_status(self, frame_id: str) -> list[int]:
        """获取当前坐标系在工位白名单中属于其分配区间的已放行 Tag ID 列表"""
        tag_range = set(self.get_frame_tag_range(frame_id))
        wl = self.hub.whitelist.get_tag_whitelist() or {}
        allowed_set = set(wl.get("allowed_ids") or [])
        return sorted(list(tag_range.intersection(allowed_set)))

    def get_frame_rois(self, frame_id: str):
        """获取专属归属于当前坐标系的 3D ROI 空间物件"""
        return [r for r in self.get_roi_spaces() if r.frame_id == frame_id]

    def scroll_roi_list(self, delta_items: int):
        """3D ROI 物件列表滚动 (滚轮/翻页按钮驱动)"""
        cur_frame = self.get_selected_frame()
        if not cur_frame:
            self.roi_scroll_offset = 0
            return
        rois = self.get_frame_rois(cur_frame.frame_id)
        max_offset = max(0, len(rois) - self.ROI_VISIBLE_COUNT)
        self.roi_scroll_offset = max(0, min(self.roi_scroll_offset + delta_items, max_offset))

    def jump_roi_scroll_by_y(self, click_y: int, track_y: int = 100, track_h: int = 552):
        """点击滚动条轨道快速跳转 ROI 视口"""
        cur_frame = self.get_selected_frame()
        if not cur_frame:
            return
        rois = self.get_frame_rois(cur_frame.frame_id)
        max_offset = max(0, len(rois) - self.ROI_VISIBLE_COUNT)
        if max_offset <= 0:
            self.roi_scroll_offset = 0
            return
        ratio = max(0.0, min(1.0, (click_y - track_y) / float(track_h)))
        self.roi_scroll_offset = int(round(ratio * max_offset))

    def get_selected_frame(self):
        """获取当前树导航选中的机构坐标系对象"""
        item_type, ws_idx, frame_id = self.hub.selected_tree_item
        if item_type == "frame" and frame_id and self.coord_mgr:
            return self.coord_mgr.get_frame(frame_id)
        return None

    # ---------------- 结构化坐标系表单弹窗方法 ----------------
    def open_frame_modal(self, frame_id: str | None = None):
        if not self.coord_mgr:
            self.load_geometry_managers()
        if not self.coord_mgr:
            self.hub.set_toast("未选择任何工位，无法配置坐标系！")
            return

        if frame_id and frame_id in self.coord_mgr._frames:
            f = self.coord_mgr.get_frame(frame_id)
            self.frame_modal_is_new = False
            self.frame_modal_orig_id = f.frame_id
            b_tags = f.get_tag_ids() if hasattr(f, "get_tag_ids") else ([f.tag_id] if f.tag_id is not None else [])
            tag_str = ",".join(str(x) for x in b_tags) if b_tags else str(f.tag_id or 0)
            
            st = getattr(f, "status", "unknown")
            k_dof = getattr(f, "known_dof", None)
            if k_dof is None:
                if st in ("manual", "calibrated"):
                    k_dof = [True] * 6
                else:
                    k_dof = [False] * 6

            t_val = [float(x) for x in f.translation_xyz_mm] if f.translation_xyz_mm is not None else [0.0, 0.0, 0.0]
            r_val = [float(x) for x in f.rotation_rpy_deg] if f.rotation_rpy_deg is not None else [0.0, 0.0, 0.0]

            self.frame_modal_data = {
                "frame_id": f.frame_id,
                "name": f.name,
                "parent_frame_id": f.parent_frame_id or "world",
                "type": f.type,
                "status": st,
                "known_dof": list(k_dof),
                "translation_xyz_mm": t_val,
                "rotation_rpy_deg": r_val,
                "tag_id": tag_str,
                "offset_xyz_mm": [float(x) for x in f.offset_xyz_mm],
                "offset_rpy_deg": [float(x) for x in f.offset_rpy_deg],
            }
        else:
            self.frame_modal_is_new = True
            self.frame_modal_orig_id = None
            idx = len(self.coord_mgr.list_frames())
            self.frame_modal_data = {
                "frame_id": f"frame_sub_{idx}",
                "name": f"{idx}号机构坐标系",
                "parent_frame_id": "world",
                "type": "fixed_transform",
                "status": "unknown",
                "known_dof": [False] * 6,
                "translation_xyz_mm": [0.0, 0.0, 0.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
                "tag_id": 0,
                "offset_xyz_mm": [0.0, 0.0, 0.0],
                "offset_rpy_deg": [0.0, 0.0, 0.0],
            }
        self.frame_modal_open = True
        self.roi_modal_open = False
        self.active_dropdown = None

    def close_frame_modal(self):
        self.frame_modal_open = False
        self.frame_modal_data = {}
        self.frame_modal_orig_id = None
        self.active_dropdown = None
        self.close_pose6d_modal()

    def save_frame_modal(self) -> tuple[bool, str]:
        if not self.coord_mgr or not self.frame_modal_data:
            return False, "无有效坐标系数据"
        from src.calibration.coordinate_manager import FrameDefinition
        d = self.frame_modal_data
        fid = str(d.get("frame_id", "")).strip()
        if not fid:
            return False, "坐标系 ID 不能为空"

        orig_fid = self.frame_modal_orig_id
        if not self.frame_modal_is_new and orig_fid and orig_fid != fid:
            if orig_fid == "world":
                return False, "绝对世界坐标系禁止修改 ID"
            ok_rename = self.coord_mgr.rename_frame(orig_fid, fid)
            if not ok_rename:
                return False, f"重命名坐标系 ID 失败 (ID '{fid}' 可能已被占用)"
            if self.roi_mgr:
                roi_modified = False
                for r in self.roi_mgr.list_rois():
                    if r.frame_id == orig_fid:
                        r.frame_id = fid
                        roi_modified = True
                if roi_modified:
                    self.roi_mgr.save()
        
        ftype = d.get("type", "fixed_transform")
        parent = None if fid == "world" else d.get("parent_frame_id", "world")
        raw_tid = str(d.get("tag_id", "0")).replace("，", ",").strip()
        parsed_tag_ids = []
        for part in raw_tid.split(","):
            part_s = part.strip()
            if part_s.isdigit():
                parsed_tag_ids.append(int(part_s))
        primary_tid = parsed_tag_ids[0] if parsed_tag_ids else 0

        existing_frame = self.coord_mgr.get_frame(orig_fid) if (not self.frame_modal_is_new and orig_fid) else None
        calib_spec = existing_frame.calibration_spec if existing_frame else None
        calib_metrics = existing_frame.calibration_metrics if existing_frame else None

        st = d.get("status", "unknown")
        k_dof = d.get("known_dof", [False]*6)
        n_known = sum(1 for b in k_dof if b)

        if n_known == 0 and st != "calibrated":
            final_status = "unknown"
            t_xyz = None
            r_rpy = None
        elif n_known == 6:
            final_status = "manual" if st != "calibrated" else "calibrated"
            t_xyz = [float(x) for x in d.get("translation_xyz_mm", [0, 0, 0])]
            r_rpy = [float(x) for x in d.get("rotation_rpy_deg", [0, 0, 0])]
        else:
            final_status = "partial" if st != "calibrated" else "calibrated"
            t_xyz = [float(x) for x in d.get("translation_xyz_mm", [0, 0, 0])]
            r_rpy = [float(x) for x in d.get("rotation_rpy_deg", [0, 0, 0])]

        frame = FrameDefinition(
            frame_id=fid,
            name=str(d.get("name", fid)).strip(),
            parent_frame_id=parent,
            type=ftype,
            status=final_status,
            known_dof=k_dof,
            translation_xyz_mm=t_xyz,
            rotation_rpy_deg=r_rpy,
            calibration_spec=calib_spec,
            calibration_metrics=calib_metrics,
            tag_id=primary_tid,
            tag_ids=parsed_tag_ids,
            offset_xyz_mm=[float(x) for x in d.get("offset_xyz_mm", [0, 0, 0])],
            offset_rpy_deg=[float(x) for x in d.get("offset_rpy_deg", [0, 0, 0])],
        )
        ok = self.coord_mgr.add_frame(frame)
        if not ok:
            return False, "保存坐标系失败 (可能导致拓扑环路或父级不存在)"
        self.coord_mgr.save()
        ws = self.hub.get_selected_workspace()
        if ws:
            self.hub.expanded_workspaces.add(ws.workspace_id)
        self.close_frame_modal()
        self.hub.set_toast(f"已成功保存坐标系: 【{frame.name}】")
        return True, "保存成功"

    def delete_frame_cascade(self, frame_id: str) -> tuple[bool, str]:
        """
        级联深度清理并删除指定子坐标系：
        1. 绝对世界基准 (world) 严禁删除；
        2. 级联清理依附于该坐标系的所有 3D ROI 空间物件 (写穿 rois.yaml)；
        3. 回收工位 tag_whitelist.yaml 中该坐标系专属分段内所有已放行的 Tag ID；
        4. 从坐标系树中移除该坐标系，将以其为父的子坐标系重定向到 world，写穿 frames.yaml；
        5. 安全重置左侧工位树选中项至当前工位的绝对世界坐标系 (world)；
        6. 强制刷新几何缓存与视图。
        """
        if not self.coord_mgr:
            return False, "坐标系管理器未就绪"
        if frame_id == "world":
            return False, "绝对世界基准坐标系 (world) 严禁删除"

        target_frame = self.coord_mgr.get_frame(frame_id)
        if not target_frame:
            return False, f"坐标系 [{frame_id}] 不存在"

        ws = self.hub.get_selected_workspace()
        if not ws:
            return False, "未选择有效工位"

        log.info(f"[CascadeDelete] 正在执行子坐标系级联删除: {frame_id} (工位: {ws.name})")

        # 1. 提前记录该坐标系专属分段与当前放行的 Tag (必须在 remove_frame 前提取)
        tag_range = {int(x) for x in self.get_frame_tag_range(frame_id)}

        # 2. 级联清除依附于该坐标系的 3D ROI 空间物件
        deleted_rois_count = 0
        if self.roi_mgr:
            to_remove_rois = [r.roi_id for r in self.roi_mgr.list_rois() if r.frame_id == frame_id]
            for rid in to_remove_rois:
                self.roi_mgr.remove_roi(rid)
                deleted_rois_count += 1
            if deleted_rois_count > 0:
                self.roi_mgr.save()
                log.info(f"[CascadeDelete] 已级联清除 {deleted_rois_count} 个归属于 [{frame_id}] 的 3D ROI")

        # 3. 回收该坐标系专属分段内所有已放行的 Tag 标靶 (同步更新 tag_whitelist.yaml)
        cleaned_tags_count = 0
        try:
            wl_path = self.hub.workspace_mgr.ensure_tag_whitelist(ws.workspace_id)
            if os.path.exists(wl_path):
                with open(wl_path, "r", encoding="utf-8") as f:
                    curr_cfg = yaml.safe_load(f) or {}
                raw_allowed = curr_cfg.get("allowed_ids") or []
                allowed = {int(x) for x in raw_allowed if str(x).isdigit()}
                to_remove_tags = allowed.intersection(tag_range)
                if to_remove_tags:
                    allowed.difference_update(to_remove_tags)
                    curr_cfg["allowed_ids"] = sorted(list(allowed))
                    with open(wl_path, "w", encoding="utf-8") as f:
                        yaml.safe_dump(curr_cfg, f, allow_unicode=True)
                    cleaned_tags_count = len(to_remove_tags)
                    log.info(f"[CascadeDelete] 已从工位白名单中回收专属 Tag: {sorted(list(to_remove_tags))}")
            self.hub.whitelist.refresh_whitelist_cache()
        except Exception as e:
            log.warning(f"[CascadeDelete] 回收专属 Tag 发生异常 ({e})")

        # 4. 从坐标系树中移除该坐标系 (下级子坐标系自动上挂至 world)，并写穿 frames.yaml
        ok = self.coord_mgr.remove_frame(frame_id)
        if not ok:
            return False, f"从坐标系树中删除 [{frame_id}] 失败"
        self.coord_mgr.save()

        # 5. 安全重置左侧树导航选择项 -> 当前工位的 world 坐标系
        item_type, ws_idx, cur_fid = self.hub.selected_tree_item
        if cur_fid == frame_id:
            self.hub.selected_tree_item = ("frame", ws_idx, "world")

        # 5. 刷新几何状态与缓存
        self.refresh_geometry_cache()
        msg = f"已成功删除坐标系 【{target_frame.name}】 ({frame_id})，连带清理 {cleaned_tags_count} 个专属Tag与 {deleted_rois_count} 个ROI物件"
        self.hub.set_toast(msg)
        log.info(f"[CascadeDelete] {msg}")
        return True, msg

    def delete_frame(self, frame_id: str) -> tuple[bool, str]:
        """兼容别名: 执行级联删除"""
        return self.delete_frame_cascade(frame_id)

    # ---------------- 6DoF 外参位姿与约束独立模态弹窗方法 ----------------
    def open_pose6d_modal(self, axis_idx: int = 0):
        """打开 6DoF 外参位姿与约束编辑模态窗"""
        d = self.frame_modal_data
        t = d.get("translation_xyz_mm", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        r = d.get("rotation_rpy_deg", [0.0, 0.0, 0.0]) or [0.0, 0.0, 0.0]
        st = d.get("status", "unknown")
        default_known = [True]*6 if st in ("manual", "calibrated") else [False]*6
        k_dof = list(d.get("known_dof", default_known))
        if len(k_dof) < 6:
            k_dof = k_dof + [False]*(6 - len(k_dof))

        self.pose6d_modal_vals = [float(t[0]), float(t[1]), float(t[2]), float(r[0]), float(r[1]), float(r[2])]
        self.pose6d_modal_known = [bool(b) for b in k_dof]
        self.pose6d_modal_axis_sel = max(0, min(5, axis_idx))
        self.pose6d_modal_axis_buf = ""
        self.pose6d_modal_open = True

    def close_pose6d_modal(self):
        """关闭 6DoF 模态窗"""
        self.pose6d_modal_open = False
        self.pose6d_modal_axis_buf = ""

    def pose6d_select_axis(self, axis_idx: int):
        """选中指定轴并提交之前轴的输入缓冲区"""
        if self.pose6d_modal_axis_buf:
            try:
                v = float(self.pose6d_modal_axis_buf)
                self.pose6d_modal_vals[self.pose6d_modal_axis_sel] = v
                self.pose6d_modal_known[self.pose6d_modal_axis_sel] = True
            except ValueError:
                pass
            self.pose6d_modal_axis_buf = ""
        self.pose6d_modal_axis_sel = max(0, min(5, axis_idx))

    def pose6d_clear_axis(self, axis_idx: int):
        """将指定轴标记为未知"""
        if self.pose6d_modal_axis_sel == axis_idx:
            self.pose6d_modal_axis_buf = ""
        self.pose6d_modal_known[axis_idx] = False

    def pose6d_toggle_axis(self, axis_idx: int):
        """切换指定轴的已知/未知状态"""
        self.pose6d_modal_known[axis_idx] = not self.pose6d_modal_known[axis_idx]

    def pose6d_set_all_unknown(self):
        """一键设为全未知 (待BA平差反向求解)"""
        self.pose6d_modal_known = [False] * 6
        self.pose6d_modal_axis_buf = ""

    def pose6d_set_all_known(self):
        """一键设为全已知"""
        self.pose6d_modal_known = [True] * 6
        self.pose6d_modal_axis_buf = ""

    def pose6d_set_planar_preset(self):
        """快捷应用水平面运动约束: Roll=0.0°, Pitch=0.0° 已知，其余轴保持"""
        self.pose6d_modal_vals[3] = 0.0
        self.pose6d_modal_known[3] = True
        self.pose6d_modal_vals[4] = 0.0
        self.pose6d_modal_known[4] = True
        self.pose6d_modal_axis_buf = ""

    def pose6d_pad_key(self, label: str):
        """处理 15 键软键盘按键输入"""
        axis = self.pose6d_modal_axis_sel
        buf = self.pose6d_modal_axis_buf

        if label == "确认":
            if buf:
                try:
                    v = float(buf)
                    self.pose6d_modal_vals[axis] = v
                    self.pose6d_modal_known[axis] = True
                except ValueError:
                    pass
                self.pose6d_modal_axis_buf = ""
            self.pose6d_modal_axis_sel = (axis + 1) % 6
            return

        if label == "清空":
            self.pose6d_modal_axis_buf = ""
            return

        if label == "退格":
            if buf:
                self.pose6d_modal_axis_buf = buf[:-1]
            return

        if label == "-/+":
            if buf.startswith("-"):
                self.pose6d_modal_axis_buf = buf[1:]
            else:
                self.pose6d_modal_axis_buf = "-" + buf
            return

        if label == ".":
            if "." not in buf:
                self.pose6d_modal_axis_buf = (buf if buf else "0") + "."
            return

        if label.isdigit():
            if buf == "0":
                self.pose6d_modal_axis_buf = label
            else:
                self.pose6d_modal_axis_buf = buf + label
            try:
                self.pose6d_modal_vals[axis] = float(self.pose6d_modal_axis_buf)
                self.pose6d_modal_known[axis] = True
            except ValueError:
                pass

    def save_pose6d_modal(self):
        """保存 6DoF 位姿与约束回 frame_modal_data"""
        if self.pose6d_modal_axis_buf:
            try:
                v = float(self.pose6d_modal_axis_buf)
                self.pose6d_modal_vals[self.pose6d_modal_axis_sel] = v
                self.pose6d_modal_known[self.pose6d_modal_axis_sel] = True
            except ValueError:
                pass
            self.pose6d_modal_axis_buf = ""

        d = self.frame_modal_data
        d["translation_xyz_mm"] = [round(float(x), 2) for x in self.pose6d_modal_vals[:3]]
        d["rotation_rpy_deg"] = [round(float(x), 2) for x in self.pose6d_modal_vals[3:]]
        d["known_dof"] = list(self.pose6d_modal_known)

        n_known = sum(1 for b in self.pose6d_modal_known if b)
        if n_known == 0:
            d["status"] = "unknown"
            self.hub.set_toast("已应用外参配置: 【全未知】(待 BA 平差反向求解)")
        elif n_known == 6:
            d["status"] = "manual"
            self.hub.set_toast("已应用外参配置: 【全已知人工指定】")
        else:
            d["status"] = "partial"
            self.hub.set_toast(f"已应用外参配置: 【部分已知先验约束】(已知 {n_known}/6 轴)")

        self.close_pose6d_modal()

    # ---------------- 结构化 ROI 表单弹窗方法 ----------------
    def open_roi_modal(self, roi_id: str | None = None):
        if not self.roi_mgr:
            self.load_geometry_managers()
        if not self.roi_mgr:
            self.hub.set_toast("未选择任何工位，无法配置 ROI！")
            return

        frames = self.get_coordinate_frames()
        default_frame = frames[1].frame_id if len(frames) > 1 else "world"

        if roi_id and roi_id in self.roi_mgr._rois:
            r = self.roi_mgr.get_roi(roi_id)
            self.roi_modal_is_new = False
            self.roi_modal_orig_id = r.roi_id
            self.roi_modal_data = {
                "roi_id": r.roi_id,
                "name": r.name,
                "frame_id": r.frame_id,
                "category": r.category,
                "center_xyz_mm": [float(x) for x in r.center_xyz_mm],
                "size_xyz_mm": [float(x) for x in r.size_xyz_mm],
                "rotation_rpy_deg": [float(x) for x in r.rotation_rpy_deg],
                "visual_color_rgb": [int(c) for c in (r.visual_color_rgb or [0, 255, 128])],
                # Smart ROI 生产意图与工艺角色扩展
                "role": getattr(r, "role", "source") or "source",
                "target_intent": getattr(r, "target_intent", "pose_pick") or "pose_pick",
                "binding": dict(getattr(r, "binding", {}) or {}),
                "pipeline_override": getattr(r, "pipeline_override", None),
                "min_confidence": float(getattr(r, "min_confidence", 0.3) or 0.3),
            }
        else:
            self.roi_modal_is_new = True
            self.roi_modal_orig_id = None
            idx = len(self.roi_mgr.list_rois()) + 1
            self.roi_modal_data = {
                "roi_id": f"roi_part_{idx}",
                "name": f"{idx}号机构部件空间",
                "frame_id": default_frame,
                "category": "belt",
                "center_xyz_mm": [0.0, 100.0, 20.0],
                "size_xyz_mm": [80.0, 200.0, 30.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
                "visual_color_rgb": [0, 255, 128],
                # Smart ROI 生产意图与工艺角色默认值
                "role": "source",
                "target_intent": "pose_pick",
                "binding": {"slot_index": 0, "capacity_max": 20},
                "pipeline_override": None,
                "min_confidence": 0.3,
            }
        self.roi_modal_open = True
        self.frame_modal_open = False
        self.active_dropdown = None

    def close_roi_modal(self):
        self.roi_modal_open = False
        self.roi_modal_data = {}
        self.roi_modal_orig_id = None
        self.active_dropdown = None

    def set_roi_modal_role(self, role: str):
        """切换 ROI 工艺角色 (source / destination / keepout / general)"""
        if not self.roi_modal_data:
            return
        self.roi_modal_data["role"] = role
        # 联动优化默认意图与视觉颜色
        if role == "source":
            if self.roi_modal_data.get("target_intent") == "piece_count":
                self.roi_modal_data["target_intent"] = "pose_pick"
            self.roi_modal_data["visual_color_rgb"] = [0, 255, 128]  # 翠绿色
        elif role == "destination":
            self.roi_modal_data["visual_color_rgb"] = [0, 200, 255]  # 天蓝色
            binding = self.roi_modal_data.setdefault("binding", {})
            if "slot_index" not in binding:
                binding["slot_index"] = 0
            if "capacity_max" not in binding:
                binding["capacity_max"] = 20
        elif role == "keepout":
            self.roi_modal_data["target_intent"] = "occupancy"
            self.roi_modal_data["visual_color_rgb"] = [255, 60, 60]   # 警示红
        elif role == "general":
            self.roi_modal_data["visual_color_rgb"] = [200, 200, 200] # 工业灰

    def set_roi_modal_intent(self, intent: str):
        """切换 ROI 动作意图 (pose_pick / piece_count / occupancy / general)"""
        if not self.roi_modal_data:
            return
        self.roi_modal_data["target_intent"] = intent

    def cycle_roi_modal_slot(self):
        """循环切换落料槽位编号 (0~7)"""
        if not self.roi_modal_data:
            return 0
        binding = self.roi_modal_data.setdefault("binding", {})
        cur_slot = int(binding.get("slot_index", 0))
        next_slot = (cur_slot + 1) % 8
        binding["slot_index"] = next_slot
        return next_slot

    def save_roi_modal(self) -> tuple[bool, str]:
        if not self.roi_mgr or not self.roi_modal_data:
            return False, "无有效 ROI 数据"
        from src.calibration.roi_manager import RoiDefinition
        d = self.roi_modal_data
        rid = str(d.get("roi_id", "")).strip()
        if not rid:
            return False, "ROI ID 不能为空"

        orig_rid = self.roi_modal_orig_id
        if not self.roi_modal_is_new and orig_rid and orig_rid != rid:
            ok_rename = self.roi_mgr.rename_roi(orig_rid, rid)
            if not ok_rename:
                return False, f"重命名 ROI ID 失败 (ID '{rid}' 可能已被占用)"
        
        # 强 Schema 尺寸校验: dx, dy, dz 必须 > 0
        sizes = [float(x) for x in d.get("size_xyz_mm", [10, 10, 10])]
        if any(s <= 0 for s in sizes):
            return False, "空间尺寸 (长宽高) 必须严格大于 0"

        roi = RoiDefinition(
            roi_id=rid,
            name=str(d.get("name", rid)).strip(),
            frame_id=str(d.get("frame_id", "world")),
            category=str(d.get("category", "general")),
            enabled=True,
            center_xyz_mm=[float(x) for x in d.get("center_xyz_mm", [0, 0, 0])],
            size_xyz_mm=sizes,
            rotation_rpy_deg=[float(x) for x in d.get("rotation_rpy_deg", [0, 0, 0])],
            visual_color_rgb=[int(c) for c in d.get("visual_color_rgb", [0, 255, 128])],
            # Smart ROI 关键属性写穿持久化
            role=str(d.get("role", "source")),
            target_intent=str(d.get("target_intent", "pose_pick")),
            binding=dict(d.get("binding", {}) or {}),
            pipeline_override=str(d.get("pipeline_override", "")).strip() or None,
            min_confidence=float(d.get("min_confidence", 0.3)),
        )
        self.roi_mgr.add_roi(roi)
        self.roi_mgr.save()
        self.close_roi_modal()
        self.hub.set_toast(f"已成功保存 3D ROI 物件: 【{roi.name}】 ({roi.role}/{roi.target_intent})")
        return True, "保存成功"

    def delete_roi(self, roi_id: str) -> tuple[bool, str]:
        if not self.roi_mgr:
            return False, "ROI 管理器未就绪"
        ok = self.roi_mgr.remove_roi(roi_id)
        if ok:
            self.roi_mgr.save()
            self.scroll_roi_list(0)
            self.hub.set_toast(f"已成功删除 ROI 物件: {roi_id}")
            return True, "删除成功"
        return False, "删除失败"
