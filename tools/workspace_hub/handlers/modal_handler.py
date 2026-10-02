# -*- coding: utf-8 -*-
"""
Workspace Hub 模态弹窗事件处理器 (ModalHandler)
==============================================
统一处理 Workspace Hub 各类模态弹窗中的点击交互：
1. Pose6DModal: 6DoF 外参位姿与自由度约束独立模态弹窗
2. MarkerSizeModal: 标靶物理名义边长录入与小键盘交互
3. AnchorModal: AprilTag 锚点世界坐标三轴录入与小键盘交互
4. FrameModal: 机构相对坐标系结构化配置与下拉框交互
5. RoiModal: 3D ROI 空间物件配置、Smart ROI 工艺角色与意图交互
6. HelpModal: 业务说明窗关闭拦截
"""

from typing import Any
from src.ui.dialog_utils import prompt_confirm, prompt_input_text


class ModalHandler:
    """模态弹窗点击事件处理器"""

    def __init__(self, app: Any):
        self.app = app
        self.state = app.state
        self.renderer = app.renderer

    def prompt_vector_axis(
        self,
        title: str,
        prompt: str,
        target_dict: dict,
        key: str,
        axis_idx: int,
        default_vec: list[float],
        unit: str = "mm",
        must_positive: bool = False,
    ):
        """通用三维向量分轴安全输入提示框"""
        cur_vec = list(target_dict.get(key, default_vec) or default_vec)
        cur_val = cur_vec[axis_idx] if axis_idx < len(cur_vec) else 0.0
        val_str = prompt_input_text(title, prompt, initial=f"{cur_val:.1f}")
        if val_str is not None and val_str.strip():
            try:
                v = float(val_str.strip())
                if must_positive and v <= 0:
                    self.state.set_toast(f"输入无效: 该参数必须为正数 (>0 {unit})！")
                    return
                while len(cur_vec) < 3:
                    cur_vec.append(0.0)
                cur_vec[axis_idx] = round(v, 2)
                target_dict[key] = cur_vec
                self.state.set_toast(f"已更新分量: {cur_vec[axis_idx]} {unit}")
            except ValueError:
                self.state.set_toast("输入无效，请输入有效数字！")

    def handle_anchor_modal_click(self, x: int, y: int):
        """锚点坐标编辑弹窗点击分发: 轴行选择/清除 / 15 键键盘 / 底部保存取消清除"""
        from tools.workspace_hub.hub_renderer import (
            WL_ANCHOR_SAVE, WL_ANCHOR_CANCEL, WL_ANCHOR_DELETE,
            anchor_row_rect, anchor_clear_rect, anchor_padkey_rect, point_in_rect
        )

        ws = self.state.whitelist

        # 三轴行: [清除] 按钮优先于行选择 (避免点清除误触发行切换)
        for axis in range(3):
            if point_in_rect(x, y, anchor_clear_rect(axis)):
                ws.anchor_axis_clear(axis)
                self.state.set_toast(f"{'XYZ'[axis]} 轴已标记为未知。")
                return
        for axis in range(3):
            if point_in_rect(x, y, anchor_row_rect(axis)):
                ws.anchor_axis_select(axis)
                return

        # 15 键键盘: 1~9 / . / 0 / -+/ 清空 / 退格 / 确认
        for idx, label in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "0", "-/+", "清空", "退格", "确认"]):
            if point_in_rect(x, y, anchor_padkey_rect(idx)):
                ws.anchor_pad_key(label)
                return

        # 底部按钮
        if point_in_rect(x, y, WL_ANCHOR_SAVE):
            ok, msg = ws.save_anchor_modal()
            self.state.set_toast(msg)
            return
        if point_in_rect(x, y, WL_ANCHOR_CANCEL):
            ws.close_anchor_modal()
            self.state.set_toast("已取消锚点编辑 (未保存)。")
            return
        if point_in_rect(x, y, WL_ANCHOR_DELETE):
            ok, msg = ws.clear_anchor_modal()
            self.state.set_toast(msg)
            return

    def handle_marker_size_modal_click(self, x: int, y: int):
        """标靶物理边长专属模态弹窗点击处理"""
        from tools.workspace_hub.hub_renderer import (
            MS_MODAL_X, MS_MODAL_Y, MS_MODAL_W, MS_MODAL_H,
            MS_BTN_SAVE, MS_BTN_CANCEL, MS_PAD_LABELS, ms_padkey_rect, point_in_rect
        )

        ws = self.state.whitelist

        # 1. 16 键数字与快捷尺寸小键盘
        for idx, lbl in enumerate(MS_PAD_LABELS):
            if point_in_rect(x, y, ms_padkey_rect(idx)):
                if lbl in ("35.5", "50.0", "40.0"):
                    ws.marker_size_buf = lbl
                else:
                    ws.marker_size_pad_key(lbl)
                return

        # 2. 底部操作按钮
        if point_in_rect(x, y, MS_BTN_SAVE):
            ok, msg = ws.save_marker_size_modal()
            self.state.set_toast(msg)
            return
        if point_in_rect(x, y, MS_BTN_CANCEL):
            ws.cancel_marker_size_modal()
            self.state.set_toast("已取消标靶边长编辑。")
            return

        # 3. 点击外部半透明遮罩关闭
        if not (MS_MODAL_X <= x <= MS_MODAL_X + MS_MODAL_W and MS_MODAL_Y <= y <= MS_MODAL_Y + MS_MODAL_H):
            ws.cancel_marker_size_modal()
            self.state.set_toast("已取消标靶边长编辑。")
            return

    def handle_frame_modal_click(self, x: int, y: int):
        """处理机构相对坐标系表单弹窗交互"""
        hit = self.renderer.hit_tester.hit_test(x, y, self.state)
        if not hit:
            return

        geom = self.state.geometry

        # 0. 下拉选择框事件拦截
        if hit == "dropdown_dismiss":
            geom.active_dropdown = None
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_toggle":
            dd_type = hit[1]
            geom.active_dropdown = None if geom.active_dropdown == dd_type else dd_type
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_select":
            dd_type, val = hit[1], hit[2]
            geom.active_dropdown = None
            d = geom.frame_modal_data
            if dd_type == "frame_type":
                d["type"] = val
                desc = "固定刚体外参" if val == "fixed_transform" else "AprilTag动标绑定"
                self.state.set_toast(f"已切换坐标系类型为: {desc}")
            elif dd_type == "frame_parent":
                d["parent_frame_id"] = val
                self.state.set_toast(f"已变更父坐标系为: [{val}]")
            return

        geom.active_dropdown = None
        d = geom.frame_modal_data
        if hit in ("frame_modal_close", "frame_modal_cancel", "frame_modal_mask"):
            geom.close_frame_modal()
            self.state.set_toast("已取消编辑坐标系。")
            return
        if hit == "frame_modal_save":
            ok, msg = geom.save_frame_modal()
            if not ok:
                self.state.set_toast(f"保存失败: {msg}")
            return
        if hit == "frame_field_name":
            old_name = d.get("name", "")
            new_name = prompt_input_text("编辑坐标系名称", "请输入坐标系人类可读名称:", initial=old_name)
            if new_name and new_name.strip():
                d["name"] = new_name.strip()
                self.state.set_toast(f"已修改坐标系名称为: 【{new_name.strip()}】")
            return
        if hit == "frame_field_id":
            old_id = d.get("frame_id", "")
            if old_id == "world":
                self.state.set_toast("绝对世界基准坐标系 [world] 禁止修改 ID！")
                return
            new_id = prompt_input_text("编辑坐标系唯一ID", "请输入唯一标识符 (英文字母/数字/下划线):", initial=old_id)
            if new_id and new_id.strip():
                new_id_clean = new_id.strip()
                frames = geom.get_coordinate_frames()
                conflict = any(f.frame_id == new_id_clean and f.frame_id != geom.frame_modal_orig_id for f in frames)
                if conflict:
                    self.state.set_toast(f"修改失败: 坐标系 ID [{new_id_clean}] 已被占用！")
                    return
                d["frame_id"] = new_id_clean
                self.state.set_toast(f"已设置坐标系唯一 ID 为: {new_id_clean} (点击[保存]后正式生效并级联更新)")
            return
        if hit == "frame_set_unknown":
            d["status"] = "unknown"
            d["prior_translation_xyz_mm"] = [None, None, None]
            d["prior_rotation_rpy_deg"] = [None, None, None]
            d["translation_xyz_mm"] = [0.0, 0.0, 0.0]
            d["rotation_rpy_deg"] = [0.0, 0.0, 0.0]
            self.state.set_toast("已将坐标系外参标记为【全未知】，BA平差时将自动通过绑定的标靶反向求解！")
            return
        if hit == "frame_open_pose6d":
            geom.open_pose6d_modal(0)
            return
        if isinstance(hit, tuple) and hit[0] == "frame_field_num":
            field_category, axis_idx = hit[1], hit[2]
            if field_category == "translation":
                geom.open_pose6d_modal(axis_idx)
                return
            elif field_category == "rotation":
                geom.open_pose6d_modal(axis_idx + 3)
                return
            elif field_category == "tag_id":
                curr_val = d.get("tag_id", 0)
                val_str = prompt_input_text(
                    "动标绑定 AprilTag ID 列表",
                    "请输入绑定的动标标靶编号 (支持单个如 10，或逗号分隔多动标冗余组如 10,11):",
                    initial=str(curr_val)
                )
                if val_str is not None and val_str.strip():
                    raw = val_str.replace("，", ",").strip()
                    parts = [p.strip() for p in raw.split(",") if p.strip().isdigit()]
                    if parts:
                        d["tag_id"] = ",".join(parts)
                        self.state.set_toast(f"已设置绑定动标列表: 【{d['tag_id']}】")
                    else:
                        self.state.set_toast("输入无效，Tag ID 必须包含有效数字编号！")
            elif field_category == "offset":
                axis_name = ["dx (前向)", "dy (横向)", "dz (垂向)"][axis_idx]
                self.prompt_vector_axis(f"动标局部偏移 {axis_name}", "请输入局部偏移数值 (mm):", d, "offset_xyz_mm", axis_idx, [0.0, 0.0, 0.0], unit="mm")

    def handle_pose6d_modal_click(self, x: int, y: int):
        """处理 6DoF 外参位姿与约束独立模态窗交互 (纯画布交互，无 OS 弹窗)"""
        geom = self.state.geometry
        from tools.workspace_hub.hub_renderer import (
            P6_BTN_UNKNOWN_ALL, P6_BTN_KNOWN_ALL, P6_BTN_PLANAR,
            P6_PAD_LABELS, pose6d_row_rect, pose6d_clear_rect, pose6d_padkey_rect,
            P6_BTN_SAVE, P6_BTN_CANCEL, point_in_rect
        )

        # 1. 顶部预设快捷按钮
        if point_in_rect(x, y, P6_BTN_UNKNOWN_ALL):
            geom.pose6d_set_all_unknown()
            self.state.set_toast("已切换为: 【全未知模式】(等待 BA 平差自动反推)")
            return
        if point_in_rect(x, y, P6_BTN_KNOWN_ALL):
            geom.pose6d_set_all_known()
            self.state.set_toast("已切换为: 【全已知人工指定】")
            return
        if point_in_rect(x, y, P6_BTN_PLANAR):
            geom.pose6d_set_planar_preset()
            self.state.set_toast("已应用快捷约束: Roll=0°, Pitch=0° (水平面运动)")
            return

        # 2. 6 轴位姿行: [设未知] 按钮优先
        for axis in range(6):
            if point_in_rect(x, y, pose6d_clear_rect(axis)):
                geom.pose6d_toggle_axis(axis)
                name = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"][axis]
                st = "已知" if geom.pose6d_modal_known[axis] else "未知 (待BA解算)"
                self.state.set_toast(f"{name} 轴已切换为: {st}")
                return

        for axis in range(6):
            if point_in_rect(x, y, pose6d_row_rect(axis)):
                geom.pose6d_select_axis(axis)
                return

        # 3. 15 键软键盘
        for idx, label in enumerate(P6_PAD_LABELS):
            if point_in_rect(x, y, pose6d_padkey_rect(idx)):
                geom.pose6d_pad_key(label)
                return

        # 4. 底部操作按钮
        if point_in_rect(x, y, P6_BTN_SAVE):
            geom.save_pose6d_modal()
            return
        if point_in_rect(x, y, P6_BTN_CANCEL):
            geom.close_pose6d_modal()
            self.state.set_toast("已取消 6DoF 外参编辑。")
            return

    def handle_roi_modal_click(self, x: int, y: int):
        """处理 3D ROI 空间物件表单弹窗交互"""
        hit = self.renderer.hit_tester.hit_test(x, y, self.state)
        if not hit:
            return

        geom = self.state.geometry

        # 0. 下拉选择框事件拦截
        if hit == "dropdown_dismiss":
            geom.active_dropdown = None
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_toggle":
            dd_type = hit[1]
            geom.active_dropdown = None if geom.active_dropdown == dd_type else dd_type
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_select":
            dd_type, val = hit[1], hit[2]
            geom.active_dropdown = None
            d = geom.roi_modal_data
            if dd_type == "roi_category":
                d["category"] = val
                cat_map = {"belt": "同步带工作面", "wheel": "驱动轮干涉区", "tray": "料盘工装区", "general": "通用机构部件"}
                self.state.set_toast(f"已切换部件类别为: [{cat_map.get(val, val)}]")
            elif dd_type == "roi_frame":
                d["frame_id"] = val
                self.state.set_toast(f"已变更所属坐标系为: [{val}]")
            return

        geom.active_dropdown = None
        d = geom.roi_modal_data
        if hit in ("roi_modal_close", "roi_modal_cancel", "roi_modal_mask"):
            geom.close_roi_modal()
            self.state.set_toast("已取消编辑 3D ROI。")
            return
        if hit == "roi_modal_save":
            ok, msg = geom.save_roi_modal()
            if not ok:
                self.state.set_toast(f"保存失败: {msg}")
            return
        if hit == "roi_field_name":
            old_name = d.get("name", "")
            new_name = prompt_input_text("编辑 ROI 物件名称", "请输入 3D ROI 物件名称:", initial=old_name)
            if new_name and new_name.strip():
                d["name"] = new_name.strip()
                self.state.set_toast(f"已修改 3D ROI 名称为: 【{new_name.strip()}】")
            return
        if hit == "roi_field_id":
            old_id = d.get("roi_id", "")
            new_id = prompt_input_text("编辑 ROI 唯一ID", "请输入唯一标识符 (英文字母/数字/下划线):", initial=old_id)
            if new_id and new_id.strip():
                new_id_clean = new_id.strip()
                rois = geom.get_roi_spaces()
                conflict = any(r.roi_id == new_id_clean and r.roi_id != geom.roi_modal_orig_id for r in rois)
                if conflict:
                    self.state.set_toast(f"修改失败: ROI ID [{new_id_clean}] 已被占用！")
                    return
                d["roi_id"] = new_id_clean
                self.state.set_toast(f"已设置 3D ROI 唯一 ID 为: {new_id_clean} (点击[保存]后正式生效)")
            return

        # Smart ROI 生产意图与工艺角色交互
        if isinstance(hit, tuple) and hit[0] == "roi_set_role":
            role_val = hit[1]
            geom.set_roi_modal_role(role_val)
            role_names = {"source": "★ 进料源", "destination": "▼ 落料槽", "keepout": "⛔ 禁行区", "general": "◌ 通用区"}
            self.state.set_toast(f"已设置工艺角色为: [{role_names.get(role_val, role_val)}]")
            return

        if isinstance(hit, tuple) and hit[0] == "roi_set_intent":
            intent_val = hit[1]
            geom.set_roi_modal_intent(intent_val)
            intent_names = {"pose_pick": "🎯 位姿抓取", "piece_count": "🔢 根数统计", "occupancy": "📦 在席检测", "general": "⚙ 通用意图"}
            self.state.set_toast(f"已设置动作意图为: [{intent_names.get(intent_val, intent_val)}]")
            return

        if hit == "roi_field_slot":
            next_slot = geom.cycle_roi_modal_slot()
            self.state.set_toast(f"已切换绑定落料槽位为: #{next_slot}")
            return

        if hit == "roi_field_capacity":
            binding = d.setdefault("binding", {})
            cur_cap = str(binding.get("capacity_max", 20))
            new_cap = prompt_input_text("修改落料槽最大容量", "请输入单槽最大容纳物料根数:", initial=cur_cap)
            if new_cap and new_cap.strip().isdigit():
                binding["capacity_max"] = max(1, int(new_cap.strip()))
                self.state.set_toast(f"已设置落料槽最大容量为: {binding['capacity_max']} 根")
            return

        if hit == "roi_field_conf":
            cur_conf = f"{float(d.get('min_confidence', 0.3) or 0.3):.2f}"
            new_conf = prompt_input_text("修改识别最低置信度", "请输入识别置信度门限 (0.05 ~ 0.95):", initial=cur_conf)
            if new_conf and new_conf.strip():
                try:
                    c_val = float(new_conf.strip())
                    if 0.01 <= c_val <= 1.0:
                        d["min_confidence"] = round(c_val, 2)
                        self.state.set_toast(f"已设置最低置信度门限为: {d['min_confidence']:.2f}")
                except Exception:
                    pass
            return

        if isinstance(hit, tuple) and hit[0] == "roi_field_num":
            field_category, axis_idx = hit[1], hit[2]
            if field_category == "center":
                axis_name = ["X", "Y", "Z"][axis_idx]
                self.prompt_vector_axis(f"局部中心 {axis_name}", "请输入中心坐标 (mm):", d, "center_xyz_mm", axis_idx, [0.0, 0.0, 0.0], unit="mm")
            elif field_category == "size":
                axis_name = ["长 dx", "宽 dy", "高 dz"][axis_idx]
                self.prompt_vector_axis(f"空间尺寸 {axis_name}", "请输入长方体尺寸 (mm, 必须 > 0):", d, "size_xyz_mm", axis_idx, [50.0, 50.0, 50.0], unit="mm", must_positive=True)
            elif field_category == "rotation":
                axis_name = ["Roll 翻滚", "Pitch 俯仰", "Yaw 偏航"][axis_idx]
                self.prompt_vector_axis(f"局部旋转 {axis_name}", "请输入旋转角 (°):", d, "rotation_rpy_deg", axis_idx, [0.0, 0.0, 0.0], unit="°")
