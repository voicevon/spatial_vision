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
from tools.workspace_hub.handlers import ModalHandler, WorkspaceHandler
from tools.workspace_hub.hub_renderer import (
    HubRenderer, grid_hit_test, HELP_MODAL_W, HELP_MODAL_H,
    HEADER_TAB_X0, HEADER_TAB_Y0, HEADER_TAB_H, HEADER_TAB_STEP,
    BTN_EXIT_X0, BTN_EXIT_Y0, BTN_EXIT_W, BTN_EXIT_H,
    point_in_rect
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
        self.modal_handler = ModalHandler(self)
        self.workspace_handler = WorkspaceHandler(self)

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
        elif self.state.gallery.view_mode == HubState.VIEW_EXPANDED:
            # 全宽大图沉浸模式下滚轮切换大图
            self.state.gallery.select_image_by_offset(wheel_dir)
        else:
            # 卡片网格墙模式下滚轮翻页
            self._scroll_grid_for_active_tab(wheel_dir)

    # 静态动作映射表 (将命令字符串映射至处理函数)
    _STATIC_DISPATCH = {
        "btn_exit": lambda app: app.stop(),
        "btn_new_workspace": lambda app: app.workspace_handler.handle_create_workspace(),
        "btn_delete_frame": lambda app: app._action_delete_frame(),
        "btn_edit_frame_pose": lambda app: app._action_edit_frame_pose(),
        "btn_edit_marker_size": lambda app: app.state.whitelist.open_marker_size_editor(),
        "btn_add_frame_roi": lambda app: app._action_add_frame_roi(),
        "btn_roi_prev": lambda app: app.state.geometry.scroll_roi_list(-1),
        "btn_roi_next": lambda app: app.state.geometry.scroll_roi_list(1),
        "exp_prev": lambda app: app.state.gallery.select_image_by_offset(-1),
        "exp_next": lambda app: app.state.gallery.select_image_by_offset(1),
        "album_delete": lambda app: app.state.gallery.delete_selected_image(),
        "exp_restore": lambda app: app.state.gallery.toggle_expanded_preview(),
        "ws_rename": lambda app: app.workspace_handler.handle_rename_workspace(),
        "ws_open_dir": lambda app: app.workspace_handler.handle_open_directory(),
        "ws_edit_desc": lambda app: app.workspace_handler.handle_edit_description(),
        "ws_toggle_prod_mode": lambda app: app.state.cycle_workspace_production_mode(),
        "ws_sync_data": lambda app: app.workspace_handler.handle_sync_data_consistency(),
        "ws_clone": lambda app: app.workspace_handler.handle_clone_workspace(),
        "ws_delete": lambda app: app.workspace_handler.handle_delete_workspace(),
        "btn_add_frame": lambda app: app.state.geometry.open_frame_modal(),
        "ws_new_frame": lambda app: app.state.geometry.open_frame_modal(),
    }

    # 参数化动作映射表
    _PARAMETRIC_DISPATCH = {
        "hdr_tab_key": lambda app, h: app.state.set_tab(h[1]),
        "tree_ws_toggle": lambda app, h: app.state.toggle_workspace_expanded(h[2]),
        "tree_ws_select": lambda app, h: app.state.select_tree_workspace(h[1]),
        "tree_frame_select": lambda app, h: app.state.select_tree_frame(h[1], h[2]),
        "frame_tag_toggle": lambda app, h: app._action_frame_tag_toggle(h[1]),
        "frame_tag_edit_xyz": lambda app, h: app._handle_frame_tag_edit_xyz(h[1]),
        "roi_scrollbar_click": lambda app, h: app.state.geometry.jump_roi_scroll_by_y(h[1]),
        "frame_roi_edit": lambda app, h: app.state.geometry.open_roi_modal(h[1]),
        "frame_roi_delete": lambda app, h: app._action_delete_roi(h[1]),
    }

    def _action_delete_frame(self):
        """响应删除当前选中子坐标系动作 (含详细级联影响清单与安全确认)"""
        cur_frame = self.state.geometry.get_selected_frame()
        if not cur_frame:
            return
        if cur_frame.type == "world" or cur_frame.frame_id == "world" or not cur_frame.parent_frame_id:
            self.state.set_toast("绝对世界基准坐标系 (world) 禁止删除！")
            return

        fid = cur_frame.frame_id
        fname = cur_frame.name

        # 统计将要级联删除的专属资源
        allowed_tags = self.state.geometry.get_frame_tags_status(fid)
        tag_desc = f"{len(allowed_tags)} 个专属 Tag (#{', #'.join(str(x) for x in allowed_tags)})" if allowed_tags else "无专属放行 Tag"

        frame_rois = self.state.geometry.get_frame_rois(fid)
        roi_desc = f"{len(frame_rois)} 个 3D ROI 物件" if frame_rois else "无依附 ROI 物件"

        child_frames = [f.frame_id for f in self.state.geometry.coord_mgr.list_frames() if f.parent_frame_id == fid]
        child_desc = f"\n3. 存在下级坐标系: {child_frames}，将自动重定向挂载至 world 基准" if child_frames else ""

        confirm_msg = (
            f"确定要删除子坐标系 【{fname}】 ({fid}) 吗？\n\n"
            f"【连带级联清理影响】：\n"
            f"1. 专属放行白名单: 将清理并回收 {tag_desc}\n"
            f"2. 3D 空间物件: 将永久删除 {roi_desc}{child_desc}\n\n"
            f"⚠️ 此操作不可撤销，请确认是否立即执行删除？"
        )

        if prompt_confirm("确认删除子机构坐标系", confirm_msg):
            self.state.geometry.delete_frame_cascade(fid)

    def _action_edit_frame_pose(self):
        cur_frame = self.state.geometry.get_selected_frame()
        if cur_frame:
            self.state.geometry.open_frame_modal(cur_frame.frame_id)

    def _action_add_frame_roi(self):
        cur_frame = self.state.geometry.get_selected_frame()
        self.state.geometry.open_roi_modal()
        if cur_frame:
            self.state.geometry.roi_modal_data["frame_id"] = cur_frame.frame_id

    def _action_delete_roi(self, roi_id: str):
        if prompt_confirm("确认删除 3D ROI", f"确定要删除 3D ROI 空间物件 [{roi_id}] 吗？"):
            self.state.geometry.delete_roi(roi_id)
            self.state.set_toast(f"已删除 ROI: {roi_id}")

    def _action_frame_tag_toggle(self, tag_id: int):
        cur_frame = self.state.geometry.get_selected_frame()
        if cur_frame:
            is_now_allowed = self.state.geometry.toggle_frame_tag_allowed(cur_frame.frame_id, tag_id)
            status_txt = "已放行 (已加入工位白名单)" if is_now_allowed else "已取消放行 (已移出工位白名单)"
            self.state.set_toast(f"标靶 Tag #{tag_id:02d} {status_txt}")

    def _handle_help_modal_click(self, x: int, y: int):
        mx = (self.renderer.canvas_w - HELP_MODAL_W) // 2
        my = (self.renderer.canvas_h - HELP_MODAL_H) // 2
        bx1 = mx + HELP_MODAL_W - 116
        by1 = my + 11
        bx2 = bx1 + 100
        by2 = by1 + 32
        if (bx1 - 6) <= x <= (bx2 + 6) and (by1 - 6) <= y <= (by2 + 6):
            self.state.is_help_modal_open = False
            self.state.set_toast("已关闭说明窗。")
            return
        if x < mx or x > mx + HELP_MODAL_W or y < my or y > my + HELP_MODAL_H:
            self.state.is_help_modal_open = False
            self.state.set_toast("已关闭说明窗。")

    def on_click(self, x: int, y: int):
        """处理鼠标左键单击与双击交互 (表驱动极速动作分发)"""
        # 1. 模态弹窗优先拦截处理
        if self.state.geometry.pose6d_modal_open:
            self.modal_handler.handle_pose6d_modal_click(x, y)
            return
        if self.state.whitelist.marker_size_modal_open:
            self.modal_handler.handle_marker_size_modal_click(x, y)
            return
        if self.state.whitelist.anchor_modal_open:
            self.modal_handler.handle_anchor_modal_click(x, y)
            return
        if self.state.geometry.frame_modal_open:
            self.modal_handler.handle_frame_modal_click(x, y)
            return
        if self.state.geometry.roi_modal_open:
            self.modal_handler.handle_roi_modal_click(x, y)
            return
        if self.state.is_help_modal_open:
            self._handle_help_modal_click(x, y)
            return

        # 2. 交互热区碰撞测试
        hit = self.renderer.hit_test(x, y, self.state)
        if hit is None:
            return

        # 3. 静态无参动作映射表分发
        if isinstance(hit, str) and hit in self._STATIC_DISPATCH:
            self._STATIC_DISPATCH[hit](self)
            return

        # 4. 参数化动作分发
        if isinstance(hit, tuple) and hit and hit[0] in self._PARAMETRIC_DISPATCH:
            self._PARAMETRIC_DISPATCH[hit[0]](self, hit)
            return

        # 5. 相册卡片网格墙点击: 单击选中卡片
        cell_idx = grid_hit_test(x, y)
        if cell_idx is not None:
            tab = self.state.active_tab
            if tab == HubState.TAB_CALIB_IMAGES:
                target = self.state.gallery.image_grid_offset + cell_idx
                if 0 <= target < len(self.state.gallery.current_images):
                    self.state.gallery.select_image_at_index(target)
            elif tab == HubState.TAB_PROD_IMAGES:
                target = self.state.gallery.prod_grid_offset + cell_idx
                if 0 <= target < len(self.state.gallery.prod_images):
                    self.state.gallery.select_prod_image_at_index(target)

    def on_double_click(self, x: int, y: int):
        """鼠标左键双击: 支持相册卡片放大进入全宽大图沉浸预览，或双击大图返回卡片网格墙"""
        if self.state.gallery.expanded_preview_mode:
            if 340 <= x <= 960 and 96 <= y <= 670:
                self.state.gallery.toggle_expanded_preview()
                return
        else:
            cell_idx = grid_hit_test(x, y)
            if cell_idx is not None and self.state.active_tab in (HubState.TAB_CALIB_IMAGES, HubState.TAB_PROD_IMAGES):
                self.state.gallery.toggle_expanded_preview()
                return
        self.on_click(x, y)

    def _select_image_for_active_tab(self, delta: int):
        """按当前激活页签切换对应的相册照片 (标定相册/生产相册; 其余页签无相册则忽略)"""
        if self.state.active_tab == HubState.TAB_PROD_IMAGES:
            self.state.gallery.select_prod_image_by_offset(delta)
        elif self.state.gallery.view_mode == HubState.VIEW_EXPANDED or self.state.active_tab == HubState.TAB_CALIB_IMAGES:
            self.state.gallery.select_image_by_offset(delta)

    def _scroll_grid_for_active_tab(self, delta_rows: int):
        """按当前激活页签滚动卡片网格 / ROI 列表 (滚轮驱动)"""
        if self.state.active_tab == HubState.TAB_FRAME_ROIS:
            self.state.geometry.scroll_roi_list(delta_rows)
        elif self.state.active_tab == HubState.TAB_PROD_IMAGES:
            self.state.gallery.scroll_prod_grid(delta_rows)
        else:
            self.state.gallery.scroll_image_grid(delta_rows)

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
        self.state.gallery.load_current_workspace_images()
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

    def _handle_frame_tag_edit_xyz(self, tag_id: int):
        """打开专用局部坐标编辑模态窗 (完全替代 Windows 文本输入框)"""
        cur_frame = self.state.geometry.get_selected_frame()
        if not cur_frame:
            return
        self.state.whitelist.open_anchor_editor(tag_id)

    def on_key(self, key: int) -> bool:
        """
        键盘快捷键分发:
        - 专用坐标编辑弹窗优先消费所有键盘输入 (Tab 换轴, 数字/. 退格, ? 设未知, Enter 保存, ESC 取消)
        - ESC (27): 逐层退出当前展开视图或模态弹窗 (大图 -> 弹窗 -> 编辑态 -> 应用)
        - Enter (13, 10): 模态表单快捷保存
        """
        state = self.state

        # 0.0 6DoF 外参位姿与约束编辑模态窗独占键盘输入
        if state.geometry.pose6d_modal_open:
            if key == 27:  # ESC 取消
                state.geometry.close_pose6d_modal()
                state.set_toast("已取消 6DoF 外参编辑。")
                return True
            if key in (13, 10):  # Enter 保存并应用
                state.geometry.save_pose6d_modal()
                return True
            if key == 9:  # Tab: 切换到下一个输入轴 (0~5 循环)
                state.geometry.pose6d_select_axis((state.geometry.pose6d_modal_axis_sel + 1) % 6)
                return True
            if key in (8, 127):  # Backspace 退格
                state.geometry.pose6d_pad_key("退格")
                return True
            if key in (ord('c'), ord('C')):  # 清空当前轴缓冲
                state.geometry.pose6d_pad_key("清空")
                return True
            if key in (ord('?'), ord('u'), ord('U')):  # 设为未知
                cur_axis = state.geometry.pose6d_modal_axis_sel
                state.geometry.pose6d_clear_axis(cur_axis)
                name = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"][cur_axis]
                state.set_toast(f"{name} 轴已标记为未知。")
                return True
            if key == ord('.'):  # 小数点
                state.geometry.pose6d_pad_key(".")
                return True
            if key in (ord('-'), ord('+')):  # 正负翻转
                state.geometry.pose6d_pad_key("-/+")
                return True
            if ord('0') <= key <= ord('9'):  # 数字 0~9
                state.geometry.pose6d_pad_key(chr(key))
                return True
            return True

        # 0.1 标靶物理边长专属模态窗 (marker_size_modal) 独占键盘输入
        if state.whitelist.marker_size_modal_open:
            if key == 27:  # ESC 取消
                state.whitelist.cancel_marker_size_modal()
                state.set_toast("已取消标靶边长编辑。")
                return True
            if key in (13, 10):  # Enter 保存
                ok, msg = state.whitelist.save_marker_size_modal()
                state.set_toast(msg)
                return True
            if key in (8, 127):  # Backspace 退格
                state.whitelist.marker_size_pad_key("退格")
                return True
            if key in (ord('c'), ord('C')):  # 清空
                state.whitelist.marker_size_pad_key("清空")
                return True
            if key == ord('.'):  # 小数点
                state.whitelist.marker_size_pad_key(".")
                return True
            if ord('0') <= key <= ord('9'):  # 数字 0~9
                state.whitelist.marker_size_pad_key(chr(key))
                return True
            return True

        # 0.2 专用坐标编辑弹窗 (anchor_modal) 独占键盘输入 (无 Windows 输入框)
        if state.whitelist.anchor_modal_open:
            if key == 27:  # ESC 取消
                state.whitelist.cancel_anchor_modal()
                state.set_toast("已取消坐标编辑 (未保存)。")
                return True
            if key in (13, 10):  # Enter 保存
                ok, msg = state.whitelist.save_anchor_modal()
                state.set_toast(msg)
                return True
            if key == 9:  # Tab: 切换到下一个输入轴 (X -> Y -> Z -> X)
                cur_axis = state.whitelist.anchor_axis_sel
                next_axis = 0 if cur_axis < 0 else (cur_axis + 1) % 3
                state.whitelist.anchor_axis_select(next_axis)
                return True
            if key in (8, 127):  # Backspace 退格
                state.whitelist.anchor_pad_key("退格")
                return True
            if key in (ord('c'), ord('C')):  # 清空当前轴缓冲
                state.whitelist.anchor_pad_key("清空")
                return True
            if key in (ord('?'), ord('x'), ord('X')):  # 设为未知
                cur_axis = state.whitelist.anchor_axis_sel
                if cur_axis >= 0:
                    state.whitelist.anchor_axis_clear(cur_axis)
                    state.set_toast(f"{'XYZ'[cur_axis]} 轴已标记为未知。")
                return True
            if key == ord('-'):  # 切换正负号
                state.whitelist.anchor_pad_key("-/+")
                return True
            if key == ord('.'):  # 小数点
                state.whitelist.anchor_pad_key(".")
                return True
            if ord('0') <= key <= ord('9'):  # 数字 0~9
                state.whitelist.anchor_pad_key(chr(key))
                return True
            return True  # 弹窗打开时拦截其余所有按键

        # 1. ESC 键层次化退出拦截
        if key == 27:
            if state.geometry.active_dropdown:
                state.geometry.active_dropdown = None
                return True
            if state.gallery.view_mode == HubState.VIEW_EXPANDED:
                state.gallery.set_view_mode(HubState.VIEW_STANDARD)
                state.set_toast("已退出全宽看图")
                return True
            if state.geometry.frame_modal_open:
                state.geometry.close_frame_modal()
                state.set_toast("已取消编辑坐标系。")
                return True
            if state.geometry.roi_modal_open:
                state.geometry.close_roi_modal()
                state.set_toast("已取消编辑 3D ROI。")
                return True
            if state.whitelist.anchor_modal_open:
                state.whitelist.exit_anchor_mode()
                state.set_toast("已退出锚点编辑。")
                return True
            if state.is_help_modal_open:
                state.is_help_modal_open = False
                return True
            return False

        # 2. Enter 键提交保存当前激活表单
        if key in (13, 10):
            if state.geometry.frame_modal_open:
                ok, msg = state.geometry.save_frame_modal()
                if not ok:
                    state.set_toast(f"保存失败: {msg}")
                return True
            if state.geometry.roi_modal_open:
                ok, msg = state.geometry.save_roi_modal()
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
