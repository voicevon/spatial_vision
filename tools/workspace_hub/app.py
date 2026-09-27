"""
工作空间综合管理中枢 (Workspace Hub)
==============================================
提供现代深色科技风格 GUI 界面：
- Workspace 画廊管理 (选择、切换、新建、重命名、克隆、删除)
- 历史采样照片卡片网格墙与全宽大图自适应视口 (双击卡片放大)
- 几何健康度与两阶段 BA 平差残差看板
- 数据工作空间与生命周期管理
- 一键直通标定离线 Studio 深度平差与原子发布至生产环境
- 完美支持中英文 Workspace 别名输入与显示
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.workspace_manager import WorkspaceManager
from src.utils.base_cv_app import BaseCvApp
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import (
    HubRenderer, grid_hit_test, HELP_MODAL_W, HELP_MODAL_H,
    HEADER_TAB_X0, HEADER_TAB_Y0, HEADER_TAB_H, HEADER_TAB_STEP,
    BTN_EXIT_X0, BTN_EXIT_Y0, BTN_EXIT_W, BTN_EXIT_H,
    WL_BTN_ALL, WL_BTN_CLEAR, WL_BTN_ANCHOR,
    WL_ANCHOR_SAVE, WL_ANCHOR_CANCEL, WL_ANCHOR_DELETE,
    whitelist_cell_rect, anchor_row_rect, anchor_clear_rect, anchor_padkey_rect, point_in_rect,
    MS_MODAL_X, MS_MODAL_Y, MS_MODAL_W, MS_MODAL_H,
    MS_BTN_SAVE, MS_BTN_CANCEL, MS_PAD_LABELS, ms_padkey_rect
)
from src.utils.logger import get_logger

log = get_logger(__name__)


from src.utils.dialog_utils import prompt_confirm, prompt_input_text


class WorkspaceHubApp(BaseCvApp):
    """Workspace Hub 主应用 (基于 BaseCvApp 轻量基类)"""

    def __init__(self, force_mock: bool = False, settings_file: str = None, workspace_mgr: WorkspaceManager = None):
        super().__init__(
            app_id="workspace_hub",
            base_w=960,
            base_h=720,
            window_name="flux_vision_3d | workspace",
            window_title="flux_vision_3d | Workspace",
            settings_file=settings_file,
            enable_keyboard_zoom=False,
        )
        self.force_mock = force_mock
        self.workspace_mgr = workspace_mgr or WorkspaceManager()
        self.state = HubState(self.workspace_mgr, force_mock=force_mock)
        self.renderer = HubRenderer()

        if self.win_mgr.scale_pct != 100 or self.win_mgr.canvas_w != 960 or self.win_mgr.canvas_h != 720:
            self.state.set_toast(f"已恢复偏好设置：放大镜 {self.win_mgr.scale_pct}%，视窗 {self.win_mgr.canvas_w}×{self.win_mgr.canvas_h}")

    # ==================== BaseCvApp 钩子实现 ====================
    def set_toast(self, msg: str, duration: float = 4.0):
        super().set_toast(msg, duration)
        self.state.set_toast(msg)

    def render(self) -> np.ndarray:
        """核心渲染: 委托 HubRenderer 进行渲染"""
        return self.renderer.render(self.state)

    def _launch_capture_wizard(self):
        """启动多视角交互式采图向导 (tools/capture/capture_wizard.py)"""
        ws = self.state.get_selected_workspace()
        cmd = [sys.executable, os.path.join(PROJECT_ROOT, "tools", "capture", "capture_wizard.py")]
        if ws:
            cmd.extend(["--workspace", ws.workspace_id])
        self._run_subtool(cmd, "多视角交互采图向导")

    def on_mouse_move(self, x: int, y: int):
        """实时跟踪鼠标坐标，支持全部按钮平滑 Hover 高亮"""
        self.state.mouse_x = x
        self.state.mouse_y = y

    def on_mouse_wheel(self, delta: int, flags: int):
        """普通滚轮极速翻页/切换 Workspace"""
        wheel_dir = -1 if delta > 0 else 1
        x, y = self.mouse_x, self.mouse_y
        if x <= 340 and 58 <= y <= 520:
            self.state.select_workspace_by_offset(wheel_dir)
        elif self.state.view_mode == HubState.VIEW_EXPANDED:
            # 全宽大图沉浸模式下滚轮切换大图
            self.state.select_image_by_offset(wheel_dir)
        else:
            # 卡片网格墙模式下滚轮翻页
            self._scroll_grid_for_active_tab(wheel_dir)

    def on_click(self, x: int, y: int):
        """处理鼠标左键单击与双击交互"""
        # =================== 1.8 6DoF 外参位姿与约束独立模态窗 ===================
        if getattr(self.state, "pose6d_modal_open", False):
            self._handle_pose6d_modal_click(x, y)
            return

        # =================== 1.9 标靶物理边长模态窗优先交互 ===================
        if self.state.marker_size_modal_open:
            self._handle_marker_size_modal_click(x, y)
            return

        # =================== 2.0 专用坐标编辑模态窗优先交互 (无 Windows 输入框) ===================
        if self.state.anchor_modal_open:
            self._handle_anchor_modal_click(x, y)
            return


        # =================== 2.5 坐标系与 3D ROI 结构化弹窗交互 ===================
        if self.state.frame_modal_open:
            self._handle_frame_modal_click(x, y)
            return

        if self.state.roi_modal_open:
            self._handle_roi_modal_click(x, y)
            return

        # =================== 3. 生产机制 Help 说明窗下的点击 ===================
        if self.state.is_help_modal_open:
            mx = (self.renderer.canvas_w - HELP_MODAL_W) // 2
            my = (self.renderer.canvas_h - HELP_MODAL_H) // 2
            bx1 = mx + HELP_MODAL_W - 116
            by1 = my + 11
            bx2 = bx1 + 100
            by2 = by1 + 32

            # 点击右上角 [X] 关闭按钮 (带 6px 宽容防抖热区)
            if (bx1 - 6) <= x <= (bx2 + 6) and (by1 - 6) <= y <= (by2 + 6):
                self.state.is_help_modal_open = False
                self.state.set_toast("已关闭说明窗。")
                return

            # 点击弹窗外部半透明遮罩：关闭
            if x < mx or x > mx + HELP_MODAL_W or y < my or y > my + HELP_MODAL_H:
                self.state.is_help_modal_open = False
                self.state.set_toast("已关闭说明窗。")
            return

        # =================== 4. 正常看板与采图模式下的鼠标点击 ===================
        # 统一使用 renderer.hit_test 进行像素级高精度命中测试
        hit = self.renderer.hit_test(x, y, self.state)

        # 4.0 顶部标题栏交互 (退出按钮与自适应 Tab 胶囊)
        if hit == "btn_exit":
            self.stop()
            return
        if isinstance(hit, tuple) and hit[0] == "hdr_tab_key":
            self.state.set_tab(hit[1])
            return

        # 4.1 左侧面板两层树结构交互
        if hit == "btn_new_workspace":
            self._handle_create_workspace()
            return
        if isinstance(hit, tuple) and hit[0] == "tree_ws_toggle":
            ws_id = hit[2]
            self.state.toggle_workspace_expanded(ws_id)
            return
        if isinstance(hit, tuple) and hit[0] == "tree_ws_select":
            ws_idx = hit[1]
            self.state.select_tree_workspace(ws_idx)
            return
        if isinstance(hit, tuple) and hit[0] == "tree_frame_select":
            ws_idx, frame_id = hit[1], hit[2]
            self.state.select_tree_frame(ws_idx, frame_id)
            return

        # 4.2 坐标系专属视图交互
        if hit == "btn_edit_frame_pose":
            cur_frame = self.state.get_selected_frame()
            if cur_frame:
                self.state.open_frame_modal(cur_frame.frame_id)
            return
        if hit == "btn_edit_marker_size":
            self.state.open_marker_size_editor()
            return
        if isinstance(hit, tuple) and hit[0] == "frame_tag_toggle":
            tag_id = hit[1]
            cur_frame = self.state.get_selected_frame()
            if cur_frame:
                is_now_allowed = self.state.toggle_frame_tag_allowed(cur_frame.frame_id, tag_id)
                status_txt = "已放行 (已加入工位白名单)" if is_now_allowed else "已取消放行 (已移出工位白名单)"
                self.state.set_toast(f"标靶 Tag #{tag_id:02d} {status_txt}")
            return
        if isinstance(hit, tuple) and hit[0] == "frame_tag_edit_xyz":
            tag_id = hit[1]
            self._handle_frame_tag_edit_xyz(tag_id)
            return
        if hit == "btn_add_frame_roi":
            cur_frame = self.state.get_selected_frame()
            self.state.open_roi_modal()
            if cur_frame:
                self.state.roi_modal_data["frame_id"] = cur_frame.frame_id
            return
        if hit == "btn_roi_prev":
            self.state.scroll_roi_list(-1)
            return
        if hit == "btn_roi_next":
            self.state.scroll_roi_list(1)
            return
        if isinstance(hit, tuple) and hit[0] == "roi_scrollbar_click":
            self.state.jump_roi_scroll_by_y(hit[1])
            return
        if isinstance(hit, tuple) and hit[0] == "frame_roi_edit":
            roi_id = hit[1]
            self.state.open_roi_modal(roi_id)
            return
        if isinstance(hit, tuple) and hit[0] == "frame_roi_delete":
            roi_id = hit[1]
            if prompt_confirm("确认删除 3D ROI", f"确定要删除 3D ROI 空间物件 [{roi_id}] 吗？"):
                self.state.delete_roi(roi_id)
                self.state.set_toast(f"已删除 ROI: {roi_id}")
            return

        # 4.3 全宽大图预览模式下的按钮交互
        if hit == "exp_prev":
            self.state.select_image_by_offset(-1)
            return
        if hit == "exp_next":
            self.state.select_image_by_offset(1)
            return
        if hit == "album_delete":
            self.state.delete_selected_image()
            return
        if hit == "exp_restore":
            self.state.toggle_expanded_preview()
            return

        # 4.4 工位大盘看板 (TAB_REPORT) 卡片内嵌按钮交互
        if hit == "ws_rename":
            self._handle_rename_workspace()
            return
        if hit == "ws_open_dir":
            self._handle_open_directory()
            return
        if hit == "ws_edit_desc":
            self._handle_edit_description()
            return
        if hit == "ws_toggle_prod_mode":
            self.state.cycle_workspace_production_mode()
            return
        if hit == "ws_sync_data":
            self._handle_sync_data_consistency()
            return
        if hit == "ws_clone":
            self._handle_clone_workspace()
            return
        if hit == "ws_delete":
            self._handle_delete_workspace()
            return
        if hit in ("btn_add_frame", "ws_new_frame"):
            self.state.open_frame_modal()
            return

        # 4.5 Tag 白名单页签顶部操作与编辑态
        if hit == "wl_refresh":
            self.state.refresh_whitelist_cache()
            self.state.set_toast("已刷新 Tag 白名单状态。")
            return
        if hit == "wl_edit":
            if self.state.whitelist_edit_mode:
                self.state.exit_whitelist_edit()
                self.state.set_toast("已完成白名单编辑。")
            else:
                self._handle_tag_whitelist()
            return

        if self.state.whitelist_edit_mode:
            if self.state.anchor_modal_open:
                self._handle_anchor_modal_click(x, y)
            else:
                self._handle_whitelist_edit_click(x, y)
            return


            # 5.5.2 图片卡片网格墙点击: 单击选中卡片
            cell_idx = grid_hit_test(x, y)
            if cell_idx is not None:
                if tab == HubState.TAB_CALIB_IMAGES:
                    target = self.state.image_grid_offset + cell_idx
                    if 0 <= target < len(self.state.current_images):
                        self.state.select_image_at_index(target)
                elif tab == HubState.TAB_PROD_IMAGES:
                    target = self.state.prod_grid_offset + cell_idx
                    if 0 <= target < len(self.state.prod_images):
                        self.state.select_prod_image_at_index(target)
                return

    def on_double_click(self, x: int, y: int):
        """鼠标左键双击: 支持相册卡片放大进入全宽大图沉浸预览，或双击大图返回卡片网格墙"""
        if self.state.expanded_preview_mode:
            if 340 <= x <= 960 and 96 <= y <= 670:
                self.state.toggle_expanded_preview()
                return
        else:
            cell_idx = grid_hit_test(x, y)
            if cell_idx is not None and self.state.active_tab in (HubState.TAB_CALIB_IMAGES, HubState.TAB_PROD_IMAGES):
                self.state.toggle_expanded_preview()
                return
        self.on_click(x, y)


    def _select_image_for_active_tab(self, delta: int):
        """按当前激活页签切换对应的相册照片 (标定相册/生产相册; 其余页签无相册则忽略)"""
        if self.state.view_mode == HubState.VIEW_EXPANDED or self.state.active_tab == HubState.TAB_CALIB_IMAGES:
            self.state.select_image_by_offset(delta)
        elif self.state.active_tab == HubState.TAB_PROD_IMAGES:
            self.state.select_prod_image_by_offset(delta)

    def _scroll_grid_for_active_tab(self, delta_rows: int):
        """按当前激活页签滚动卡片网格 / ROI 列表 (滚轮驱动)"""
        if self.state.active_tab == HubState.TAB_FRAME_ROIS:
            self.state.scroll_roi_list(delta_rows)
        elif self.state.active_tab == HubState.TAB_PROD_IMAGES:
            self.state.scroll_prod_grid(delta_rows)
        else:
            self.state.scroll_image_grid(delta_rows)

    def _handle_open_directory(self):
        """在系统资源管理器中打开工位目录"""
        ws = self.state.get_selected_workspace()
        if ws and os.path.exists(ws.workspace_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(ws.workspace_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", ws.workspace_dir])
                else:
                    subprocess.run(["xdg-open", ws.workspace_dir])
                self.state.set_toast(f"已在资源管理器中打开: {ws.name}")
            except Exception as e:
                self.state.set_toast(f"打开目录异常: {e}")

    def _handle_delete_workspace(self):
        """删除当前 Workspace"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        ok, msg = self.workspace_mgr.delete_workspace(ws.workspace_id)
        self.state.refresh_workspaces()
        self.state.set_toast(msg)

    def _run_subtool(self, cmd: list, desc: str):
        """统一子工具拉起执行器：销毁主窗、运行子工具、恢复环境与刷新状态"""
        self.state.set_toast(f"正在唤起 {desc}...")
        cv2.destroyAllWindows()

        try:
            subprocess.run(cmd)
        except Exception as e:
            log.warning(f"执行工具异常: {e}")

        # 重新创建主窗体并重新绑定鼠标事件
        self.create_window()

        ws = self.state.get_selected_workspace()
        if ws:
            ws.refresh_stats()
            ws.save_meta()
        self.state.refresh_workspaces()
        self.state.load_current_workspace_images()
        self.state.set_toast(f"已完成 {desc} 并返回 Workspace 驾驶舱，数据已同步！")

    def _launch_spatial_mapping_studio(self):
        """启动 AprilTag 空间建图工作站 (Spatial Mapping Studio)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return
        cmd = [sys.executable, "-m", "tools.spatial_mapping_studio",
               "--workspace", ws.workspace_id,
               "--images", ws.calib_raw_images_dir,
               "--map", ws.map_path]
        self._run_subtool(cmd, "空间建图工作站 (Spatial Mapping Studio)")

    def _launch_image_diagnostics(self):
        """启动标靶单帧漏检病因深度切片与梯度诊断"""
        cmd = [sys.executable, "tools/calibration/diagnose_tag_frame.py"]
        self._run_subtool(cmd, "图像深度病因诊断切片系统")

    def _launch_tag_generator(self):
        """启动 AprilTag 标靶图纸生成与 1:1 A4 排版"""
        cmd = [sys.executable, "tools/calibration/generate_apriltags.py"]
        self._run_subtool(cmd, "标靶高清生成与排版工具")

    def _handle_tag_whitelist(self):
        """进入白名单页内芯片矩阵编辑模式: 缺失时自动创建模板, 点击芯片写穿保存 yaml"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选择任何 Workspace")
            return

        self.state.ensure_tag_whitelist_file()

        # 页内芯片矩阵编辑 (写穿保存, 不再委托外部文本编辑器)
        self.state.enter_whitelist_edit()
        self.state.set_toast("已进入白名单编辑: 单击芯片切换放行/拦截, 即时写回 yaml")

    def _handle_whitelist_edit_click(self, x: int, y: int):
        """白名单编辑态点击分发: 锚点模式切换 / 批量按钮 / 芯片矩阵 (几何与渲染器单源共用)"""
        state = self.state

        # 批量操作按钮
        if point_in_rect(x, y, WL_BTN_ALL):
            n = state.whitelist_batch("all")
            state.set_toast(f"已全量放行 0~29 (白名单 {n} 个)。")
            return
        if point_in_rect(x, y, WL_BTN_CLEAR):
            state.whitelist_batch("clear")
            state.set_toast("已清空白名单 → 探索模式 (全量放行检测标靶)。")
            return
        if point_in_rect(x, y, WL_BTN_ANCHOR):
            if state.anchor_mode:
                state.exit_anchor_mode()
                state.set_toast("已退出锚点模式, 返回白名单编辑。")
            else:
                state.enter_anchor_mode()
                n = state.anchor_known_count()
                state.set_toast(f"已进入锚点模式: 单击 Tag 芯片编辑世界坐标 (已知 {n} 枚)。")
            return

        # 芯片矩阵: 0~29 基础网格 + 超范围追加芯片 (同一矩形公式)
        extra_ids = sorted(t for t in state.whitelist_edit_ids if t >= 30)
        for t_id in list(range(30)) + extra_ids:
            if point_in_rect(x, y, whitelist_cell_rect(t_id)):
                if state.anchor_mode:
                    # 锚点模式: 单击芯片打开该 Tag 的世界坐标锚点编辑弹窗
                    state.open_anchor_editor(t_id)
                    return
                n = state.toggle_whitelist_id(t_id)
                on = t_id in state.whitelist_edit_ids
                state.set_toast(f"Tag #{t_id:02d} {'已放行' if on else '已拦截'} (白名单 {n} 个)。")
                return

    def _handle_anchor_modal_click(self, x: int, y: int):
        """锚点坐标编辑弹窗点击分发: 轴行选择/清除 / 15 键键盘 / 底部保存取消清除"""
        state = self.state

        # 三轴行: [清除] 按钮优先于行选择 (避免点清除误触发行切换)
        for axis in range(3):
            if point_in_rect(x, y, anchor_clear_rect(axis)):
                state.anchor_axis_clear(axis)
                state.set_toast(f"{'XYZ'[axis]} 轴已标记为未知。")
                return
        for axis in range(3):
            if point_in_rect(x, y, anchor_row_rect(axis)):
                state.anchor_axis_select(axis)
                return

        # 15 键键盘: 1~9 / . / 0 / -+/ 清空 / 退格 / 确认
        for idx, label in enumerate(["1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "0", "-/+", "清空", "退格", "确认"]):
            if point_in_rect(x, y, anchor_padkey_rect(idx)):
                state.anchor_pad_key(label)
                return

        # 底部按钮
        if point_in_rect(x, y, WL_ANCHOR_SAVE):
            ok, msg = state.save_anchor_modal()
            state.set_toast(msg)
            return
        if point_in_rect(x, y, WL_ANCHOR_CANCEL):
            state.cancel_anchor_modal()
            state.set_toast("已取消锚点编辑 (未保存)。")
            return
        if point_in_rect(x, y, WL_ANCHOR_DELETE):
            ok, msg = state.clear_anchor_modal()
            state.set_toast(msg)
            return

    def _handle_marker_size_modal_click(self, x: int, y: int):
        """标靶物理边长专属模态弹窗点击处理"""
        state = self.state

        # 1. 16 键数字与快捷尺寸小键盘
        for idx, lbl in enumerate(MS_PAD_LABELS):
            if point_in_rect(x, y, ms_padkey_rect(idx)):
                if lbl in ("35.5", "50.0", "40.0"):
                    state.marker_size_buf = lbl
                else:
                    state.marker_size_pad_key(lbl)
                return

        # 2. 底部操作按钮
        if point_in_rect(x, y, MS_BTN_SAVE):
            ok, msg = state.save_marker_size_modal()
            state.set_toast(msg)
            return
        if point_in_rect(x, y, MS_BTN_CANCEL):
            state.cancel_marker_size_modal()
            state.set_toast("已取消标靶边长编辑。")
            return

        # 3. 点击外部半透明遮罩关闭
        if not (MS_MODAL_X <= x <= MS_MODAL_X + MS_MODAL_W and MS_MODAL_Y <= y <= MS_MODAL_Y + MS_MODAL_H):
            state.cancel_marker_size_modal()
            state.set_toast("已取消标靶边长编辑。")
            return

    def _handle_rename_workspace(self):
        """修改 Workspace 显示名称 (支持中文)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        new_name = prompt_input_text(
            "修改 Workspace 名称",
            f"请输入 Workspace【{ws.name}】的新显示名称\n(支持中文、英文、数字，如: 1号机台主标定):",
            initial=ws.name
        )
        if new_name and new_name != ws.name:
            self.state.rename_current_workspace(new_name)

    def _handle_edit_description(self):
        """修改 Workspace 备注说明 (支持中文单行文本)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        cur_desc = getattr(ws, "description", "") or ""
        new_desc = prompt_input_text(
            "修改工位备注",
            f"请输入工位【{ws.name}】的备注信息 (单行文本):",
            initial=cur_desc
        )
        if new_desc is not None:
            self.state.update_current_workspace_description(new_desc.strip())

    def _handle_create_workspace(self):
        """新建 Workspace (支持中文名称弹窗)"""
        idx = len(self.state.workspaces) + 1
        default_alias = f"Workspace_{idx}"

        chosen_name = prompt_input_text(
            "新建 Workspace",
            "请输入新 Workspace 名称/别名 (支持中文、英文、数字，如: 2号机架高位):",
            initial=default_alias
        )
        if not chosen_name:
            self.state.set_toast("已取消新建 Workspace。")
            return

        new_ws = self.workspace_mgr.create_workspace(alias=chosen_name, description=f"Workspace {chosen_name}")
        self.state.refresh_workspaces()
        target_idx = 0
        for i, s in enumerate(self.state.workspaces):
            if s.workspace_id == new_ws.workspace_id:
                target_idx = i
                break
        self.state.select_workspace_at_index(target_idx)
        self.state.set_toast(f"已成功新建 Workspace: 【{new_ws.name}】({new_ws.workspace_id})，按 [C] 开始采图！")

    def _handle_clone_workspace(self):
        """克隆 Workspace (支持中文名称弹窗)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何 Workspace，无法克隆！")
            return

        default_clone_name = f"{ws.name}_对照组"
        chosen_name = prompt_input_text(
            "克隆 Workspace",
            f"请输入克隆后的新 Workspace 名称 (基于原 Workspace【{ws.name}】):",
            initial=default_clone_name
        )
        if not chosen_name:
            return

        cloned = self.workspace_mgr.clone_workspace(ws.workspace_id, new_alias=chosen_name)
        if cloned:
            # 立即刷新 Workspace 列表
            self.state.refresh_workspaces()
            target_idx = 0
            for i, s in enumerate(self.state.workspaces):
                if s.workspace_id == cloned.workspace_id:
                    target_idx = i
                    break
            self.state.select_workspace_at_index(target_idx)
            self.state.set_toast(f"已成功克隆 Workspace: 【{cloned.name}】并定位至新 Workspace！")
        else:
            self.state.set_toast("克隆 Workspace 失败，请检查源目录！")

    def _handle_delete_workspace(self):
        """删除当前选中的 Workspace"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何 Workspace，无法删除！")
            return

        if len(self.state.workspaces) <= 1:
            self.state.set_toast("至少需保留一个工位，禁止删除唯一工位！")
            return

        confirmed = prompt_confirm(
            "确认删除 Workspace",
            f"确定要永久删除工位【{ws.name}】吗？\n\n物理ID: {ws.workspace_id}\n此操作将删除该工位的所有图片和标定数据，不可恢复！"
        )
        if not confirmed:
            self.state.set_toast("已取消删除操作。")
            return

        ok, msg = self.workspace_mgr.delete_workspace(ws.workspace_id)
        if ok:
            self.state.refresh_workspaces()
            self.state.set_toast(f"已成功删除工位: 【{ws.name}】")
        else:
            self.state.set_toast(f"删除工位失败: {msg}")

    def _handle_sync_data_consistency(self):
        """核验物理磁盘与元数据一致性，重新扫描并自动自愈同步"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何工位，无法核验！")
            return

        old_calib = ws.image_count
        old_prod = ws.prod_image_count

        # 1. 强制重新扫描物理磁盘并更新元数据
        ws.refresh_stats()
        ws.save_meta()

        # 2. 刷新相册与列表数据
        self.state.load_current_workspace_images()
        self.state.load_prod_images()
        self.state.refresh_workspaces()

        diff_calib = ws.image_count - old_calib
        diff_prod = ws.prod_image_count - old_prod
        if diff_calib == 0 and diff_prod == 0:
            self.state.set_toast(f"一致性核验完成: 物理与元数据已是最新 (标定 {ws.image_count} 帧, 生产 {ws.prod_image_count} 帧)")
        else:
            self.state.set_toast(f"已同步数据一致性: 标定 {old_calib}→{ws.image_count} 帧, 生产 {old_prod}→{ws.prod_image_count} 帧")

    def _handle_frame_tag_edit_xyz(self, tag_id: int):
        """打开专用局部坐标编辑模态窗 (完全替代 Windows 文本输入框)"""
        cur_frame = self.state.get_selected_frame()
        if not cur_frame:
            return
        self.state.open_anchor_editor(tag_id)

    def _handle_frames_rois_click(self, x: int, y: int):
        """处理【坐标系&ROI】列表页签的按钮交互"""
        hit = self.renderer.hit_test(x, y, self.state)
        if not hit:
            return
        if hit == "geom_refresh":
            self.state.load_geometry_managers()
            self.state.set_toast("已重新加载当前工位的坐标系与 ROI 配置。")
        elif hit == "geom_open_dir":
            self._handle_open_directory()
        elif hit == "btn_add_frame":
            self.state.open_frame_modal()
        elif hit == "btn_add_roi":
            self.state.open_roi_modal()
        elif isinstance(hit, tuple) and hit[0] == "frame_edit":
            frames = self.state.get_coordinate_frames()
            idx = hit[1]
            if 0 <= idx < len(frames):
                self.state.open_frame_modal(frames[idx].frame_id)
        elif isinstance(hit, tuple) and hit[0] == "frame_del":
            frames = self.state.get_coordinate_frames()
            idx = hit[1]
            if 0 <= idx < len(frames) and frames[idx].frame_id != "world":
                f = frames[idx]
                if prompt_confirm("确认删除坐标系", f"确定要删除机构相对坐标系 【{f.name}】 ({f.frame_id}) 吗？\n关联的子坐标系或 ROI 将自动回退至父级。"):
                    self.state.delete_frame(f.frame_id)
        elif isinstance(hit, tuple) and hit[0] == "roi_edit":
            rois = self.state.get_roi_spaces()
            idx = hit[1]
            if 0 <= idx < len(rois):
                self.state.open_roi_modal(rois[idx].roi_id)
        elif isinstance(hit, tuple) and hit[0] == "roi_del":
            rois = self.state.get_roi_spaces()
            idx = hit[1]
            if 0 <= idx < len(rois):
                r = rois[idx]
                if prompt_confirm("确认删除 3D ROI", f"确定要删除 3D ROI 空间物件 【{r.name}】 ({r.roi_id}) 吗？"):
                    self.state.delete_roi(r.roi_id)

    def _prompt_vector_axis(
        self,
        title: str,
        prompt: str,
        data_dict: dict,
        field_key: str,
        axis_idx: int,
        default_vec: list[float],
        unit: str = "mm",
        must_positive: bool = False,
    ):
        """交互式弹窗编辑 3D 向量中的单轴数值，并写回字典字段"""
        curr_val = data_dict.get(field_key, default_vec)[axis_idx]
        val_str = prompt_input_text(title, prompt, initial=f"{curr_val:.1f}")
        if val_str is not None and val_str.strip():
            try:
                v = float(val_str.strip())
                if must_positive and v <= 0:
                    self.state.set_toast(f"{title}必须严格大于 0！")
                    return
                vec = data_dict.setdefault(field_key, list(default_vec))
                vec[axis_idx] = v
                self.state.set_toast(f"已更新{title}: {v:.1f} {unit}".strip())
            except ValueError:
                self.state.set_toast("输入无效，请输入有效数字！")

    def _handle_frame_modal_click(self, x: int, y: int):
        """处理机构相对坐标系表单弹窗交互"""
        hit = self.renderer.hit_test(x, y, self.state)
        if not hit:
            return

        # 0. 下拉选择框事件拦截
        if hit == "dropdown_dismiss":
            self.state.active_dropdown = None
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_toggle":
            dd_type = hit[1]
            self.state.active_dropdown = None if self.state.active_dropdown == dd_type else dd_type
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_select":
            dd_type, val = hit[1], hit[2]
            self.state.active_dropdown = None
            d = self.state.frame_modal_data
            if dd_type == "frame_type":
                d["type"] = val
                desc = "固定刚体外参" if val == "fixed_transform" else "AprilTag动标绑定"
                self.state.set_toast(f"已切换坐标系类型为: {desc}")
            elif dd_type == "frame_parent":
                d["parent_frame_id"] = val
                self.state.set_toast(f"已变更父坐标系为: [{val}]")
            return

        self.state.active_dropdown = None
        d = self.state.frame_modal_data
        if hit in ("frame_modal_close", "frame_modal_cancel", "frame_modal_mask"):
            self.state.close_frame_modal()
            self.state.set_toast("已取消编辑坐标系。")
            return
        if hit == "frame_modal_save":
            ok, msg = self.state.save_frame_modal()
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
                frames = self.state.get_coordinate_frames()
                conflict = any(f.frame_id == new_id_clean and f.frame_id != self.state.frame_modal_orig_id for f in frames)
                if conflict:
                    self.state.set_toast(f"修改失败: 坐标系 ID [{new_id_clean}] 已被占用！")
                    return
                d["frame_id"] = new_id_clean
                self.state.set_toast(f"已设置坐标系唯一 ID 为: {new_id_clean} (点击[保存]后正式生效并级联更新)")
            return
        if hit == "frame_set_unknown":
            d["status"] = "unknown"
            d["known_dof"] = [False] * 6
            d["translation_xyz_mm"] = [0.0, 0.0, 0.0]
            d["rotation_rpy_deg"] = [0.0, 0.0, 0.0]
            self.state.set_toast("已将坐标系外参标记为【全未知】，BA平差时将自动通过绑定的标靶反向求解！")
            return
        if hit == "frame_open_pose6d":
            self.state.open_pose6d_modal(0)
            return
        if isinstance(hit, tuple) and hit[0] == "frame_field_num":
            field_category, axis_idx = hit[1], hit[2]
            if field_category == "translation":
                self.state.open_pose6d_modal(axis_idx)
                return
            elif field_category == "rotation":
                self.state.open_pose6d_modal(axis_idx + 3)
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
                self._prompt_vector_axis(f"动标局部偏移 {axis_name}", "请输入局部偏移数值 (mm):", d, "offset_xyz_mm", axis_idx, [0.0, 0.0, 0.0], unit="mm")

    def _handle_pose6d_modal_click(self, x: int, y: int):
        """处理 6DoF 外参位姿与约束独立模态窗交互 (纯画布交互，无 OS 弹窗)"""
        state = self.state
        from tools.workspace_hub.hub_renderer import (
            P6_BTN_UNKNOWN_ALL, P6_BTN_KNOWN_ALL, P6_BTN_PLANAR,
            P6_PAD_LABELS, pose6d_row_rect, pose6d_clear_rect, pose6d_padkey_rect,
            P6_BTN_SAVE, P6_BTN_CANCEL, point_in_rect
        )

        # 1. 顶部预设快捷按钮
        if point_in_rect(x, y, P6_BTN_UNKNOWN_ALL):
            state.pose6d_set_all_unknown()
            state.set_toast("已切换为: 【全未知模式】(等待 BA 平差自动反推)")
            return
        if point_in_rect(x, y, P6_BTN_KNOWN_ALL):
            state.pose6d_set_all_known()
            state.set_toast("已切换为: 【全已知人工指定】")
            return
        if point_in_rect(x, y, P6_BTN_PLANAR):
            state.pose6d_set_planar_preset()
            state.set_toast("已应用快捷约束: Roll=0°, Pitch=0° (水平面运动)")
            return

        # 2. 6 轴位姿行: [设未知] 按钮优先
        for axis in range(6):
            if point_in_rect(x, y, pose6d_clear_rect(axis)):
                state.pose6d_toggle_axis(axis)
                name = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"][axis]
                st = "已知" if state.pose6d_modal_known[axis] else "未知 (待BA解算)"
                state.set_toast(f"{name} 轴已切换为: {st}")
                return

        for axis in range(6):
            if point_in_rect(x, y, pose6d_row_rect(axis)):
                state.pose6d_select_axis(axis)
                return

        # 3. 15 键软键盘
        for idx, label in enumerate(P6_PAD_LABELS):
            if point_in_rect(x, y, pose6d_padkey_rect(idx)):
                state.pose6d_pad_key(label)
                return

        # 4. 底部操作按钮
        if point_in_rect(x, y, P6_BTN_SAVE):
            state.save_pose6d_modal()
            return
        if point_in_rect(x, y, P6_BTN_CANCEL):
            state.close_pose6d_modal()
            state.set_toast("已取消 6DoF 外参编辑。")
            return


    def _handle_roi_modal_click(self, x: int, y: int):
        """处理 3D ROI 空间物件表单弹窗交互"""
        hit = self.renderer.hit_test(x, y, self.state)
        if not hit:
            return

        # 0. 下拉选择框事件拦截
        if hit == "dropdown_dismiss":
            self.state.active_dropdown = None
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_toggle":
            dd_type = hit[1]
            self.state.active_dropdown = None if self.state.active_dropdown == dd_type else dd_type
            return

        if isinstance(hit, tuple) and hit[0] == "dropdown_select":
            dd_type, val = hit[1], hit[2]
            self.state.active_dropdown = None
            d = self.state.roi_modal_data
            if dd_type == "roi_category":
                d["category"] = val
                cat_map = {"belt": "同步带工作面", "wheel": "驱动轮干涉区", "tray": "料盘工装区", "general": "通用机构部件"}
                self.state.set_toast(f"已切换部件类别为: [{cat_map.get(val, val)}]")
            elif dd_type == "roi_frame":
                d["frame_id"] = val
                self.state.set_toast(f"已变更所属坐标系为: [{val}]")
            return

        self.state.active_dropdown = None
        d = self.state.roi_modal_data
        if hit in ("roi_modal_close", "roi_modal_cancel", "roi_modal_mask"):
            self.state.close_roi_modal()
            self.state.set_toast("已取消编辑 3D ROI。")
            return
        if hit == "roi_modal_save":
            ok, msg = self.state.save_roi_modal()
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
                rois = self.state.get_rois()
                conflict = any(r.roi_id == new_id_clean and r.roi_id != self.state.roi_modal_orig_id for r in rois)
                if conflict:
                    self.state.set_toast(f"修改失败: ROI ID [{new_id_clean}] 已被占用！")
                    return
                d["roi_id"] = new_id_clean
                self.state.set_toast(f"已设置 3D ROI 唯一 ID 为: {new_id_clean} (点击[保存]后正式生效)")
            return

        # Smart ROI 生产意图与工艺角色交互
        if isinstance(hit, tuple) and hit[0] == "roi_set_role":
            role_val = hit[1]
            self.state.set_roi_modal_role(role_val)
            role_names = {"source": "★ 进料源", "destination": "▼ 落料槽", "keepout": "⛔ 禁行区", "general": "◌ 通用区"}
            self.state.set_toast(f"已设置工艺角色为: [{role_names.get(role_val, role_val)}]")
            return

        if isinstance(hit, tuple) and hit[0] == "roi_set_intent":
            intent_val = hit[1]
            self.state.set_roi_modal_intent(intent_val)
            intent_names = {"pose_pick": "🎯 位姿抓取", "piece_count": "🔢 根数统计", "occupancy": "📦 在席检测", "general": "⚙ 通用意图"}
            self.state.set_toast(f"已设置动作意图为: [{intent_names.get(intent_val, intent_val)}]")
            return

        if hit == "roi_field_slot":
            next_slot = self.state.cycle_roi_modal_slot()
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
                self._prompt_vector_axis(f"局部中心 {axis_name}", "请输入中心坐标 (mm):", d, "center_xyz_mm", axis_idx, [0.0, 0.0, 0.0], unit="mm")
            elif field_category == "size":
                axis_name = ["长 dx", "宽 dy", "高 dz"][axis_idx]
                self._prompt_vector_axis(f"空间尺寸 {axis_name}", "请输入长方体尺寸 (mm, 必须 > 0):", d, "size_xyz_mm", axis_idx, [50.0, 50.0, 50.0], unit="mm", must_positive=True)
            elif field_category == "rotation":
                axis_name = ["Roll 翻滚", "Pitch 俯仰", "Yaw 偏航"][axis_idx]
                self._prompt_vector_axis(f"局部旋转 {axis_name}", "请输入旋转角 (°):", d, "rotation_rpy_deg", axis_idx, [0.0, 0.0, 0.0], unit="°")

    def on_key(self, key: int) -> bool:
        """
        键盘快捷键分发:
        - 专用坐标编辑弹窗优先消费所有键盘输入 (Tab 换轴, 数字/. 退格, ? 设未知, Enter 保存, ESC 取消)
        - ESC (27): 逐层退出当前展开视图或模态弹窗 (大图 -> 弹窗 -> 编辑态 -> 应用)
        - Enter (13, 10): 模态表单快捷保存
        """
        state = self.state

        # 0.0 6DoF 外参位姿与约束编辑模态窗独占键盘输入
        if getattr(state, "pose6d_modal_open", False):
            if key == 27:  # ESC 取消
                state.close_pose6d_modal()
                state.set_toast("已取消 6DoF 外参编辑。")
                return True
            if key in (13, 10):  # Enter 保存并应用
                state.save_pose6d_modal()
                return True
            if key == 9:  # Tab: 切换到下一个输入轴 (0~5 循环)
                state.pose6d_select_axis((state.pose6d_modal_axis_sel + 1) % 6)
                return True
            if key in (8, 127):  # Backspace 退格
                state.pose6d_pad_key("退格")
                return True
            if key in (ord('c'), ord('C')):  # 清空当前轴缓冲
                state.pose6d_pad_key("清空")
                return True
            if key in (ord('?'), ord('u'), ord('U')):  # 设为未知
                cur_axis = state.pose6d_modal_axis_sel
                state.pose6d_clear_axis(cur_axis)
                name = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"][cur_axis]
                state.set_toast(f"{name} 轴已标记为未知。")
                return True
            if key == ord('.'):  # 小数点
                state.pose6d_pad_key(".")
                return True
            if key in (ord('-'), ord('+')):  # 正负翻转
                state.pose6d_pad_key("-/+")
                return True
            if ord('0') <= key <= ord('9'):  # 数字 0~9
                state.pose6d_pad_key(chr(key))
                return True
            return True

        # 0.0 标靶物理边长专属模态窗 (marker_size_modal) 独占键盘输入
        if state.marker_size_modal_open:

            if key == 27:  # ESC 取消
                state.cancel_marker_size_modal()
                state.set_toast("已取消标靶边长编辑。")
                return True
            if key in (13, 10):  # Enter 保存
                ok, msg = state.save_marker_size_modal()
                state.set_toast(msg)
                return True
            if key in (8, 127):  # Backspace 退格
                state.marker_size_pad_key("退格")
                return True
            if key in (ord('c'), ord('C')):  # 清空
                state.marker_size_pad_key("清空")
                return True
            if key == ord('.'):  # 小数点
                state.marker_size_pad_key(".")
                return True
            if ord('0') <= key <= ord('9'):  # 数字 0~9
                state.marker_size_pad_key(chr(key))
                return True
            return True

        # 0.1 专用坐标编辑弹窗 (anchor_modal) 独占键盘输入 (无 Windows 输入框)
        if state.anchor_modal_open:
            if key == 27:  # ESC 取消
                state.cancel_anchor_modal()
                state.set_toast("已取消坐标编辑 (未保存)。")
                return True
            if key in (13, 10):  # Enter 保存
                ok, msg = state.save_anchor_modal()
                state.set_toast(msg)
                return True
            if key == 9:  # Tab: 切换到下一个输入轴 (X -> Y -> Z -> X)
                cur_axis = state.anchor_axis_sel
                next_axis = 0 if cur_axis < 0 else (cur_axis + 1) % 3
                state.anchor_axis_select(next_axis)
                return True
            if key in (8, 127):  # Backspace 退格
                state.anchor_pad_key("退格")
                return True
            if key in (ord('c'), ord('C')):  # 清空当前轴缓冲
                state.anchor_pad_key("清空")
                return True
            if key in (ord('?'), ord('x'), ord('X')):  # 设为未知
                cur_axis = state.anchor_axis_sel
                if cur_axis >= 0:
                    state.anchor_axis_clear(cur_axis)
                    state.set_toast(f"{'XYZ'[cur_axis]} 轴已标记为未知。")
                return True
            if key == ord('-'):  # 切换正负号
                state.anchor_pad_key("-/+")
                return True
            if key == ord('.'):  # 小数点
                state.anchor_pad_key(".")
                return True
            if ord('0') <= key <= ord('9'):  # 数字 0~9
                state.anchor_pad_key(chr(key))
                return True
            return True  # 弹窗打开时拦截其余所有按键

        # 1. ESC 键层次化退出拦截
        if key == 27:
            if state.active_dropdown:
                state.active_dropdown = None
                return True
            if state.view_mode == HubState.VIEW_EXPANDED:
                state.set_view_mode(HubState.VIEW_STANDARD)
                state.set_toast("已退出全宽看图")
                return True
            if state.frame_modal_open:
                state.close_frame_modal()
                state.set_toast("已取消编辑坐标系。")
                return True
            if state.roi_modal_open:
                state.close_roi_modal()
                state.set_toast("已取消编辑 3D ROI。")
                return True
            if state.anchor_modal_open:
                state.exit_anchor_mode()
                state.set_toast("已退出锚点编辑。")
                return True
            if state.whitelist_edit_mode:
                state.exit_whitelist_edit()
                state.set_toast("已退出白名单编辑态。")
                return True
            if state.is_help_modal_open:
                state.is_help_modal_open = False
                return True
            return False

        # 2. Enter 键提交保存当前激活表单
        if key in (13, 10):
            if state.frame_modal_open:
                ok, msg = state.save_frame_modal()
                if not ok:
                    state.set_toast(f"保存失败: {msg}")
                return True
            if state.roi_modal_open:
                ok, msg = state.save_roi_modal()
                if not ok:
                    state.set_toast(f"保存失败: {msg}")
                return True

        return False


def main():
    parser = argparse.ArgumentParser(description="工作空间综合管理中枢 (Workspace Hub)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟仿真相机模式运行")
    args = parser.parse_args()

    app = WorkspaceHubApp(force_mock=args.mock)
    app.run()


if __name__ == "__main__":
    main()
