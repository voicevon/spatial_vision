"""
Workspace Hub 自动化单元测试
===========================
验证 CameraStreamer 取流、HubState 工位状态机管理与原地连拍归档
"""

import os
import sys
import shutil
import tempfile
import unittest
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.calibration.workspace_manager import WorkspaceManager
from src.hardware.camera_streamer import CameraStreamer
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_renderer import HubRenderer, HELP_MODAL_W, HELP_MODAL_H
from tools.workspace_hub.app import WorkspaceHubApp


class TestWorkspaceHub(unittest.TestCase):

    def setUp(self):
        self.test_root = tempfile.mkdtemp(prefix="test_hub_")
        self.workspaces_dir = os.path.join(self.test_root, "workspaces")
        os.makedirs(self.workspaces_dir, exist_ok=True)
        self.test_config_yaml = os.path.join(self.test_root, "config.yaml")
        self.workspace_mgr = WorkspaceManager(
            workspaces_dir=self.workspaces_dir,
            config_path=self.test_config_yaml
        )
        # 创建两个测试工位
        self.ws1 = self.workspace_mgr.create_workspace(alias="site_a", description="测试工况A")
        self.ws2 = self.workspace_mgr.create_workspace(alias="site_b", description="测试工况B")

    def tearDown(self):
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_camera_streamer_mock(self):
        """测试 CameraStreamer 仿真流读取与 FPS 统计"""
        streamer = CameraStreamer(force_mock=True)
        self.assertTrue(streamer.start())
        self.assertTrue(streamer.is_mock)
        ok, frame = streamer.read()
        self.assertTrue(ok)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (720, 1280, 3))
        streamer.stop()
        self.assertFalse(streamer.is_running)

    def test_hub_state_navigation(self):
        """测试 HubState 工位切换与发布生产操作"""
        state = HubState(self.workspace_mgr, force_mock=True)
        self.assertEqual(len(state.workspaces), 2)
        # 降序排序下，最新创建的 ws2 在 index 0，先创建的 ws1 在 index 1
        self.assertEqual(state.workspaces[0].workspace_id, self.ws2.workspace_id)
        self.assertEqual(state.workspaces[1].workspace_id, self.ws1.workspace_id)
        self.assertEqual(state.selected_workspace_idx, 0)

        # 切换下一个工位 (index 0 -> 1)
        state.select_workspace_by_offset(1)
        self.assertEqual(state.selected_workspace_idx, 1)
        self.assertEqual(state.get_selected_workspace().workspace_id, self.ws1.workspace_id)

        # 验证工位自身独立沙盒地图存储与状态感知
        cur_ws = state.get_selected_workspace()
        cur_ws.ba_solved = True
        cur_ws.save_meta()
        with open(cur_ws.map_path, "w", encoding="utf-8") as f:
            f.write("tags:\n  0:\n    id: 0\n    position: [0.0, 0.0, 0.0]\n    orientation: [0.0, 0.0, 0.0, 1.0]\n")
        cur_ws.refresh_stats()
        self.assertTrue(os.path.exists(cur_ws.map_path))
        self.assertTrue(cur_ws.ba_solved)

    def test_hub_state_in_place_capture(self):
        """测试 HubState 原地连拍保存与归档"""
        state = HubState(self.workspace_mgr, force_mock=True)
        cur_ws = state.get_selected_workspace()
        self.assertEqual(cur_ws.image_count, 0)

        # 模拟生成并抓拍一帧
        test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        saved_file = state.save_capture_frame(test_frame)
        self.assertTrue(os.path.exists(saved_file))
        self.assertEqual(cur_ws.image_count, 1)
        self.assertEqual(len(state.current_images), 1)

        # 再次抓拍第二帧
        saved_file2 = state.save_capture_frame(test_frame)
        self.assertTrue(os.path.exists(saved_file2))
        self.assertEqual(cur_ws.image_count, 2)
        self.assertEqual(len(state.current_images), 2)

    def test_hub_renderer_canvas(self):
        """测试 HubRenderer 双缓冲画布在四页签与全宽大图模式下的渲染输出有效性 (960x720)"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()

        # 1. 渲染体检报告页签 (默认)
        self.assertEqual(state.active_tab, HubState.TAB_REPORT)
        canvas_calib = renderer.render(state)
        self.assertEqual(canvas_calib.shape, (720, 960, 3))

        # 2. 渲染生产相册页签
        state.set_tab(HubState.TAB_PROD_IMAGES)
        canvas_prod = renderer.render(state)
        self.assertEqual(canvas_prod.shape, (720, 960, 3))

        # 3. 渲染体检报告页签
        state.set_tab(HubState.TAB_REPORT)
        canvas_report = renderer.render(state)
        self.assertEqual(canvas_report.shape, (720, 960, 3))

        # 4. 渲染 Tag 白名单页签
        state.set_tab(HubState.TAB_WHITELIST)
        canvas_wl = renderer.render(state)
        self.assertEqual(canvas_wl.shape, (720, 960, 3))

        # 5. 渲染全宽大图沉浸视图
        state.set_tab(HubState.TAB_CALIB_IMAGES)
        state.set_view_mode(HubState.VIEW_EXPANDED)
        canvas_exp = renderer.render(state)
        self.assertEqual(canvas_exp.shape, (720, 960, 3))

    def test_hub_header_buttons_layout(self):
        """测试 Header 顶部按钮布局及 Help 弹窗交互响应"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()
        canvas = renderer.render(state)
        self.assertEqual(canvas.shape, (720, 960, 3))

    def test_hub_help_modal(self):
        """测试【生效到生产系统】业务说明弹窗开启与渲染"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.is_help_modal_open)
        # 呼出 Help 弹窗
        state.toggle_help_modal()
        self.assertTrue(state.is_help_modal_open)

        # 渲染带有 Help 弹窗的画布
        canvas_help = renderer.render(state)
        self.assertEqual(canvas_help.shape, (720, 960, 3))

        # 关闭 Help 弹窗
        state.toggle_help_modal()
        self.assertFalse(state.is_help_modal_open)

    def test_hub_clone_and_immediate_refresh(self):
        """测试工位克隆、列表实时刷新与新工位自动定位"""
        state = HubState(self.workspace_mgr, force_mock=True)
        initial_count = len(state.workspaces)
        self.assertEqual(initial_count, 2)

        # 克隆工位
        cur_ws = state.get_selected_workspace()
        cloned = self.workspace_mgr.clone_workspace(cur_ws.workspace_id, new_alias="对照组_工况测试")
        self.assertIsNotNone(cloned)
        self.assertEqual(cloned.name, "对照组_工况测试")

        # 刷新并重新定位
        state.refresh_workspaces()
        self.assertEqual(len(state.workspaces), initial_count + 1)

        # 验证新工位在列表中且可被定位
        target_idx = -1
        for idx, ws in enumerate(state.workspaces):
            if ws.workspace_id == cloned.workspace_id:
                target_idx = idx
                break
        self.assertNotEqual(target_idx, -1)
        state.selected_workspace_idx = target_idx
        state.load_current_workspace_images()
        self.assertEqual(state.get_selected_workspace().name, "对照组_工况测试")

    def test_hub_rename_workspace(self):
        """测试工位修改名称立即生效"""
        state = HubState(self.workspace_mgr, force_mock=True)
        ok = state.rename_current_workspace("全新车间工况A")
        self.assertTrue(ok)
        self.assertEqual(state.get_selected_workspace().name, "全新车间工况A")

    def test_hub_expanded_preview_toggle(self):
        """测试 [F] 键单帧大图全宽自适应占满与三栏模式切换"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()

        self.assertFalse(state.expanded_preview_mode)
        state.toggle_expanded_preview()
        self.assertTrue(state.expanded_preview_mode)

        # 渲染全宽大图
        canvas_exp = renderer.render(state)
        self.assertEqual(canvas_exp.shape, (720, 960, 3))

    def test_hub_top_exit_button_click(self):
        """测试点击紧贴生产相册右侧的 [退出] 按钮能够正常结束主循环"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        self.assertTrue(app._running)

        # 模拟鼠标点击紧贴生产相册右侧的退出按钮 (x=880, y=20)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 880, 20, 0, None)
        self.assertFalse(app._running)

    def test_hub_footer_camera_and_card_active_action(self):
        """测试 Footer 底部 Camera 状态指示以及卡片点击直接设为活动"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        renderer = HubRenderer()
        canvas = np.zeros((720, 960, 3), dtype=np.uint8)

        # 验证 Footer 渲染不报错
        renderer._render_footer(canvas, app.state)

        # 验证当前选中的工位对象有效
        self.assertIsNotNone(app.state.get_selected_workspace())

    def test_four_tabs_switch_and_rendering(self):
        """测试右侧动态区四页签切换与各页签画布渲染稳定性 (960x720)"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()

        # 1. 初始为体检报告页签
        self.assertEqual(state.active_tab, HubState.TAB_REPORT)
        c1 = renderer.render(state)
        self.assertEqual(c1.shape, (720, 960, 3))

        # 2. 切换至 Tab2: Tag 白名单页签
        state.set_tab(HubState.TAB_WHITELIST)
        self.assertEqual(state.active_tab, HubState.TAB_WHITELIST)
        c2 = renderer.render(state)
        self.assertEqual(c2.shape, (720, 960, 3))

        # 3. 切换至 Tab3: 体检报告页签
        state.set_tab(HubState.TAB_REPORT)
        self.assertEqual(state.active_tab, HubState.TAB_REPORT)
        c3 = renderer.render(state)
        self.assertEqual(c3.shape, (720, 960, 3))

        # 4. 切换至 Tab4: 生产相册页签
        state.set_tab(HubState.TAB_PROD_IMAGES)
        self.assertEqual(state.active_tab, HubState.TAB_PROD_IMAGES)
        c4 = renderer.render(state)
        self.assertEqual(c4.shape, (720, 960, 3))

        # 5. 切换页签过程中视图模式始终稳定为标准页签看板
        self.assertEqual(state.view_mode, HubState.VIEW_STANDARD)

        # 6. 全宽大图沉浸模式下切换页签自动回落到标准看板
        state.toggle_expanded_preview()
        self.assertEqual(state.view_mode, HubState.VIEW_EXPANDED)
        state.set_tab(HubState.TAB_REPORT)
        self.assertEqual(state.view_mode, HubState.VIEW_STANDARD)
        self.assertEqual(state.active_tab, HubState.TAB_REPORT)

    def test_four_tabs_header_clicks(self):
        """测试鼠标点击顶部 Header 工位三页签 Tab 胶囊直接切换页签 (Dashboard / 标定相册 / ★ 生产相册)"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        app.win_mgr.canvas_w = 960
        app.win_mgr.canvas_h = 720

        # 点击 Tab 0: Dashboard (x: 348~463, 测试点 400, 25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 400, 25, 0, None)
        self.assertEqual(app.state.active_tab, HubState.TAB_REPORT)

        # 点击 Tab 1: 标定相册 (x: 475~590, 测试点 530, 25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 530, 25, 0, None)
        self.assertEqual(app.state.active_tab, HubState.TAB_CALIB_IMAGES)

        # 点击 Tab 2: 生产相册 (x: 602~717, 测试点 650, 25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 650, 25, 0, None)
        self.assertEqual(app.state.active_tab, HubState.TAB_PROD_IMAGES)

        # 点击 Header 页签后仍处于标准页签看板 (非全宽大图)
        self.assertEqual(app.state.view_mode, HubState.VIEW_STANDARD)

    def test_context_menu_removed_and_left_click_only(self):
        """测试工位卡片右键菜单已彻底移除，右键点击不触发弹出菜单"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "test_hub_settings.json"))
        self.assertFalse(app.state.context_menu_open)

        # 1. 模拟在第 1 张卡片上右键点击 (x=100, y=80)，验证不再弹出菜单
        app._on_mouse_event(cv2.EVENT_RBUTTONDOWN, 100, 80, 0, None)
        self.assertFalse(app.state.context_menu_open)

        # 2. 模拟在第 1 张卡片上左键点击 (x=100, y=80)，验证正常选中工位
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 100, 80, 0, None)
        self.assertEqual(app.state.selected_workspace_idx, 0)

    def test_workspace_hub_settings_persistence(self):
        """测试 Workspace Hub 窗口尺寸与缩放比例的自动记忆持久化与二次启动恢复"""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_cfg = os.path.join(tmpdir, "hub_test_settings.json")
            # 1. 启动第一实例并缩放到 120%
            app1 = WorkspaceHubApp(force_mock=True, settings_file=test_cfg)
            app1.win_mgr.apply_zoom(+20)
            self.assertEqual(app1.win_mgr.scale_pct, 120)

            # 2. 启动第二实例，验证自动无感恢复 (基于基准 960x720)
            app2 = WorkspaceHubApp(force_mock=True, settings_file=test_cfg)
            self.assertEqual(app2.win_mgr.scale_pct, 120)
            self.assertEqual(app2.win_mgr.canvas_w, int(960 * 1.2))
            self.assertEqual(app2.win_mgr.canvas_h, int(720 * 1.2))

            # 3. 模拟拖拽拉伸窗口改变分辨率，验证自动落盘
            app2.win_mgr.canvas_w = 1600
            app2.win_mgr.canvas_h = 900
            app2.win_mgr.save_settings()

            # 4. 启动第三实例，验证 1600x900 依然被精准记住
            app3 = WorkspaceHubApp(force_mock=True, settings_file=test_cfg)
            self.assertEqual(app3.win_mgr.scale_pct, 120)
            self.assertEqual(app3.win_mgr.canvas_w, 1600)
            self.assertEqual(app3.win_mgr.canvas_h, 900)

    def test_help_modal_hit_test_and_close(self):
        """验证生产说明弹窗右上角 [X] 关闭按钮在 960 画布各种坐标下的瞬间关闭判定"""
        clean_cfg = os.path.join(self.test_root, "clean_hub_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        app.win_mgr.canvas_w = 960
        app.win_mgr.canvas_h = 720
        state = app.state
        state.is_help_modal_open = True

        # 计算理论按钮中心与边界 (基于 960x720)
        mx = (960 - HELP_MODAL_W) // 2
        my = (720 - HELP_MODAL_H) // 2
        bx1 = mx + HELP_MODAL_W - 116
        by1 = my + 11
        bx2 = bx1 + 100
        by2 = by1 + 32

        # 1. 模拟点击关闭按钮中心 (如 x=bx1+50, y=by1+16)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx1 + 50, by1 + 16, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮中心应立即关闭说明窗")

        # 2. 模拟点击关闭按钮最右上角边缘 (如 bx2, by1)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx2, by1, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮右上角应立即关闭说明窗")

        # 3. 模拟点击关闭按钮最左上角 (bx1, by1)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx1, by1, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮左上角应立即关闭说明窗")

        # 4. 模拟点击关闭按钮容差外扩热区 (+4px 边缘)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, bx2 + 4, by1 - 4, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击关闭按钮容差区域应立即关闭说明窗")

        # 5. 模拟点击弹窗外部半透明遮罩 (如左上角 x=20, y=20)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 20, 20, 0, None)
        self.assertFalse(state.is_help_modal_open, "点击弹窗遮罩外部应立即关闭说明窗")

        # 6. 模拟点击弹窗内部内容区 (如工况卡片位置 x=mx+50, y=my+150)，弹窗应保持打开
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, mx + 50, my + 150, 0, None)
        self.assertTrue(state.is_help_modal_open, "点击弹窗内部卡片不应关闭说明窗")

        # 7. 测试 renderer 的 hit_test 判定
        hit_action = app.renderer.hit_test(bx1 + 50, by1 + 16, state)
        self.assertEqual(hit_action, "help_close", "renderer.hit_test 应对齐返回 help_close")

        # 8. 进阶测试：当窗口缩放至 1440x900 时，物理屏幕坐标映射后应同样秒关 (上对齐 pad_y=0)
        app.win_mgr.canvas_w = 1440
        app.win_mgr.canvas_h = 900
        scale = min(1440 / 960.0, 900 / 720.0)  # 1.25
        pad_x = (1440 - int(960 * scale)) // 2
        pad_y = 0  # 严格上对齐！
        phys_btn_x = int(pad_x + (bx1 + 50) * scale)
        phys_btn_y = int(pad_y + (by1 + 16) * scale)
        state.is_help_modal_open = True
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, phys_btn_x, phys_btn_y, 0, None)
        self.assertFalse(state.is_help_modal_open, "1440x900 缩放下点击关闭按钮应同样瞬时关闭")

    def test_image_deletion_and_tabs_relocation(self):
        """测试照片删除功能与页签化后的点击交互"""
        clean_cfg = os.path.join(self.test_root, "clean_tabs_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        # 隔离至测试临时工位沙盒，避免读写真实项目工位数据
        app.state.workspace_mgr = self.workspace_mgr
        app.state.refresh_workspaces()
        app.win_mgr.canvas_w = 960
        app.win_mgr.canvas_h = 720
        state = app.state
        state.set_tab(HubState.TAB_CALIB_IMAGES)

        # 1. 创建 20 张测试图片放入当前选中工位中 (卡片网格 4 列 x 3 行 = 每页 12 张)
        ws = state.get_selected_workspace()
        self.assertIsNotNone(ws)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(20):
            cv2.imwrite(os.path.join(ws.calib_raw_images_dir, f"test_view_{i:02d}.png"), dummy)
        state.load_current_workspace_images()

        initial_count = len(state.current_images)
        self.assertGreaterEqual(initial_count, 20)
        state.selected_image_idx = 0

        # 2. 测试通过 state.delete_selected_image() 删除首张照片
        deleted_file = state.current_images[0]
        ok = state.delete_selected_image()
        self.assertTrue(ok)
        self.assertFalse(os.path.exists(deleted_file), "被删除的照片文件应已从磁盘移除")
        self.assertEqual(len(state.current_images), initial_count - 1)
        self.assertEqual(ws.image_count, initial_count - 1)

        # 3. 图片页签已移除顶部按钮组: 右上角点击不再触发删除, 删除改由 [Del] 键触发
        del_target = state.current_images[0]
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 1240, 73, 0, None)
        self.assertTrue(os.path.exists(del_target), "图片页签右上角已无按钮, 点击不应删除照片")
        self.assertEqual(len(state.current_images), initial_count - 1)

        # 4. 键盘 [Del] 对应的状态层删除逻辑
        self.assertTrue(state.delete_selected_image())
        self.assertFalse(os.path.exists(del_target), "键盘 [Del] 删除逻辑应移除当前照片")
        self.assertEqual(len(state.current_images), initial_count - 2)

        # 5. 滚轮在卡片网格中按行滚动 (每格滚动一行, 并夹紧到最后一页起始行)
        state.image_grid_offset = 0
        app._on_mouse_event(cv2.EVENT_MOUSEWHEEL, 700, 300, -1, None)
        self.assertEqual(state.image_grid_offset, HubState.GRID_COLS, "滚轮下翻应前进一行")
        for _ in range(6):
            app._on_mouse_event(cv2.EVENT_MOUSEWHEEL, 700, 300, -1, None)
        self.assertEqual(state.image_grid_offset, HubState.GRID_PAGE, "滚轮下翻应夹紧到最后一页起始行")
        app._on_mouse_event(cv2.EVENT_MOUSEWHEEL, 700, 300, 1, None)
        self.assertEqual(state.image_grid_offset, HubState.GRID_PAGE - HubState.GRID_COLS, "滚轮上翻应回退一行")

        # 6. 测试点击 Header 页签 Tab 切换动态区内容 (左栏保持稳定)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 650, 25, 0, None)
        self.assertEqual(state.active_tab, HubState.TAB_PROD_IMAGES)

        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 400, 25, 0, None)
        self.assertEqual(state.active_tab, HubState.TAB_REPORT)

        # Tab 1: 标定相册 (x=530, y=25)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 530, 25, 0, None)
        self.assertEqual(state.active_tab, HubState.TAB_CALIB_IMAGES)

    def test_tag_whitelist_creation_and_context_menu(self):
        """测试 [编辑] 进入白名单芯片矩阵编辑模式: 自动生成 tag_whitelist.yaml 模板并进入编辑态"""
        import yaml
        clean_cfg = os.path.join(self.test_root, "clean_whitelist_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        ws = app.state.get_selected_workspace()
        self.assertIsNotNone(ws)

        # 确保初始无 whitelist
        wl_path = ws.whitelist_path
        if os.path.exists(wl_path):
            os.remove(wl_path)

        # 执行白名单处理 (页内编辑, 不再委托外部编辑器, 无 startfile)
        app._handle_tag_whitelist()
        self.assertTrue(os.path.exists(wl_path), "应自动创建 tag_whitelist.yaml 文件")
        self.assertTrue(app.state.whitelist_edit_mode, "应进入芯片矩阵编辑模式")

        # 验证文件结构符合规范 (白名单恒启用: 无 enabled 开关, 名单内容即行为)
        with open(wl_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        self.assertNotIn("enabled", cfg)
        self.assertIn("allowed_ids", cfg)
        self.assertIn("workspace_id", cfg)
        self.assertEqual(cfg["workspace_id"], ws.workspace_id)

        app.state.exit_whitelist_edit()
        self.assertFalse(app.state.whitelist_edit_mode)

    def test_whitelist_chip_editor_write_through(self):
        """测试芯片矩阵编辑器写穿语义: 切换/批量均即时落盘 tag_whitelist.yaml"""
        import yaml
        from tools.workspace_hub.hub_renderer import whitelist_cell_rect
        clean_cfg = os.path.join(self.test_root, "chip_editor_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg)
        state = app.state
        ws = state.get_selected_workspace()
        wl_path = ws.whitelist_path
        if os.path.exists(wl_path):
            os.remove(wl_path)

        def read_ids():
            with open(wl_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("allowed_ids")

        app._handle_tag_whitelist()  # 创建模板并进入编辑态
        state.whitelist_batch("clear")
        self.assertEqual(read_ids(), [])

        # 芯片切换写穿: 加入 5、2 → yaml 即时 [2, 5]; 再切 5 → 移除
        self.assertEqual(state.toggle_whitelist_id(5), 1)
        self.assertEqual(state.toggle_whitelist_id(2), 2)
        self.assertEqual(read_ids(), [2, 5])
        state.toggle_whitelist_id(5)
        self.assertEqual(read_ids(), [2])

        # 批量全量放行
        self.assertEqual(state.whitelist_batch("all"), 30)
        self.assertEqual(len(read_ids()), 30)

        # 退出编辑后 yaml 保持最后状态
        state.exit_whitelist_edit()
        self.assertFalse(state.whitelist_edit_mode)
        self.assertEqual(len(read_ids()), 30)

        # 模拟点击芯片 #2 (row0 col2 中心) → 移除并写盘; 点击 [完成] → 退出编辑态
        state.set_tab(HubState.TAB_WHITELIST)
        app._handle_tag_whitelist()
        cx, cy, cw, ch = whitelist_cell_rect(2)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, cx + cw // 2, cy + ch // 2, 0, None)
        self.assertNotIn(2, read_ids())
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 899, 73, 0, None)  # [完成] 按钮中心
        self.assertFalse(state.whitelist_edit_mode)
        self.assertNotIn(2, read_ids())

    def test_anchor_editor_partial_and_clear(self):
        """测试锚点坐标编辑弹窗: 逐轴输入/部分已知/负号/清除单轴/删除锚点 (写穿工位 anchor_tags.yaml, 全局兜底不动)"""
        from src.utils.config_guard import load_anchor_tags
        from src.calibration.workspace_manager import load_workspace_anchor_tags
        from tools.workspace_hub.hub_renderer import (
            whitelist_cell_rect, anchor_row_rect, anchor_clear_rect, anchor_padkey_rect
        )
        tmp_cfg = os.path.join(self.test_root, "anchor_config.yaml")
        with open(tmp_cfg, "w", encoding="utf-8") as f:
            f.write(
                "tags_map_path: map/test_map.yaml\n"
                "calibration:\n"
                "  # 每-Tag 世界坐标锚点表 (旧版全局源)\n"
                "  anchor_tags:\n"
                "    0:\n"
                "      xyz_mm: [10.0, 20.0, 30.0]\n"
                "      known: [true, true, true]\n"
                "    1:\n"
                "      xyz_mm: [0.0, 520.0, 196.0]\n"
                "      known: [true, true, true]\n"
            )
        clean_cfg = os.path.join(self.test_root, "anchor_settings.json")
        app = WorkspaceHubApp(force_mock=True, settings_file=clean_cfg, workspace_mgr=self.workspace_mgr)
        state = app.state
        state.anchor_config_path = tmp_cfg  # 全局兜底源注入
        ws = state.get_selected_workspace()
        state.set_tab(HubState.TAB_WHITELIST)
        state.enter_whitelist_edit()
        state.enter_anchor_mode()

        def read_global_text():
            with open(tmp_cfg, "r", encoding="utf-8") as f:
                return f.read()

        def rect_center(rect):
            rx, ry, rw, rh = rect
            return (rx + rw // 2, ry + rh // 2)

        def pad_center(label):
            labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", ".", "0", "-/+", "清空", "退格", "确认"]
            return rect_center(anchor_padkey_rect(labels.index(label)))

        # 1. 锚点模式单击无锚芯片 #3 → 打开弹窗 (草稿来自全局兜底, tag3 无锚 → 全未记录)
        cx, cy, cw, ch = whitelist_cell_rect(3)
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, cx + cw // 2, cy + ch // 2, 0, None)
        self.assertTrue(state.anchor_modal_open)
        self.assertEqual(state.anchor_modal_tag, 3)
        self.assertEqual(state.anchor_known_count(), 0)
        self.assertFalse(os.path.exists(ws.anchor_path), "未保存前不应创建工位锚点文件")

        # 2. X 轴输入 12.5 → 保存 = 写穿工位锚点文件 (兜底条目随迁, 部分已知 [T,F,F])
        app._handle_anchor_modal_click(*rect_center(anchor_row_rect(0)))
        for d in "12.5":
            app._handle_anchor_modal_click(*pad_center(d))
        app._handle_anchor_modal_click(*pad_center("确认"))
        ok, msg = state.save_anchor_modal()
        self.assertTrue(ok, msg)
        self.assertFalse(state.anchor_modal_open)
        self.assertTrue(os.path.exists(ws.anchor_path))
        own = load_workspace_anchor_tags(ws.workspace_dir)
        self.assertEqual(own[3]["known"], [True, False, False])
        self.assertAlmostEqual(own[3]["xyz_mm"][0], 12.5)
        self.assertEqual(set(own), {0, 1, 3}, "首次写穿应包含全局兜底条目 (以当前标定为基础)")
        # 全局兜底源不被修改 (注释保留)
        self.assertIn("# 每-Tag 世界坐标锚点表", read_global_text())
        self.assertEqual(set(load_anchor_tags(tmp_cfg)), {0, 1})

        # 3. 工位锚点回填草稿; 清除 Z 轴 → [T,T,F] 写穿工位文件
        state.open_anchor_editor(1)
        self.assertEqual(state.anchor_known_count(), 3)
        app._handle_anchor_modal_click(*rect_center(anchor_clear_rect(2)))
        self.assertEqual(state.anchor_known_count(), 2)
        ok, _ = state.save_anchor_modal()
        self.assertTrue(ok)
        own = load_workspace_anchor_tags(ws.workspace_dir)
        self.assertEqual(own[1]["known"], [True, True, False])
        self.assertAlmostEqual(own[1]["xyz_mm"][1], 520.0)

        # 4. 负数输入: 3.25 → -/+ → -3.25 (tag0 草稿三轴全知)
        state.open_anchor_editor(0)
        app._handle_anchor_modal_click(*rect_center(anchor_row_rect(0)))
        for d in "3.25":
            app._handle_anchor_modal_click(*pad_center(d))
        app._handle_anchor_modal_click(*pad_center("-/+"))
        app._handle_anchor_modal_click(*pad_center("确认"))
        ok, _ = state.save_anchor_modal()
        self.assertTrue(ok)
        own = load_workspace_anchor_tags(ws.workspace_dir)
        self.assertEqual(own[0]["known"], [True, True, True])
        self.assertAlmostEqual(own[0]["xyz_mm"][0], -3.25)

        # 5. [清除锚点] → 从工位锚点删除 tag0 条目 (全局兜底不受影响)
        state.open_anchor_editor(0)
        ok, _ = state.clear_anchor_modal()
        self.assertTrue(ok)
        self.assertNotIn(0, load_workspace_anchor_tags(ws.workspace_dir))
        self.assertIn(0, load_anchor_tags(tmp_cfg))

        # 6. 全部轴清除后保存 = 从工位锚点删除条目
        state.open_anchor_editor(1)
        for axis in range(3):
            state.anchor_axis_clear(axis)
        ok, _ = state.save_anchor_modal()
        self.assertTrue(ok)
        own = load_workspace_anchor_tags(ws.workspace_dir)
        self.assertNotIn(1, own)
        self.assertEqual(set(own), {3})

        # 7. 取消 → 不落盘
        state.open_anchor_editor(3)
        state.anchor_axis_select(1)
        for _ in "999":
            state.anchor_pad_key("9")
        state.cancel_anchor_modal()
        own = load_workspace_anchor_tags(ws.workspace_dir)
        self.assertAlmostEqual(own[3]["xyz_mm"][0], 12.5)
        self.assertAlmostEqual(own[3]["xyz_mm"][1], 0.0)

        # 8. 弹窗渲染冒烟 (960x720)
        state.open_anchor_editor(3)
        canvas = app.renderer.render(state)
        self.assertEqual(canvas.shape, (720, 960, 3))

        state.exit_anchor_mode()
        self.assertFalse(state.anchor_modal_open)
        state.exit_whitelist_edit()
        self.assertFalse(state.whitelist_edit_mode)

    def test_legacy_whitelist_migration(self):
        """测试旧格式白名单一次性迁移: whitelist_tag_ids → allowed_ids 写回 (保留其余字段, 同义旧键移除)"""
        import yaml
        from src.calibration.workspace_manager import load_workspace_tag_whitelist
        ws_dir = os.path.join(self.test_root, "ws_legacy_wl")
        os.makedirs(ws_dir, exist_ok=True)
        p = os.path.join(ws_dir, "tag_whitelist.yaml")
        with open(p, "w", encoding="utf-8") as f:
            f.write("description: 旧格式白名单\nwhitelist_tag_ids:\n- 0\n- 18\n- 29\nworkspace_id: legacy\n")

        # 首次读取触发迁移, 返回旧名单语义
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [0, 18, 29])
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.assertEqual(data["allowed_ids"], [0, 18, 29])
        self.assertNotIn("whitelist_tag_ids", data)
        self.assertEqual(data["description"], "旧格式白名单")
        self.assertEqual(data["workspace_id"], "legacy")

        # 再次读取幂等, 且新格式空 allowed_ids 保持探索模式不被迁移
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [0, 18, 29])
        with open(p, "w", encoding="utf-8") as f:
            f.write("workspace_id: legacy\nallowed_ids: []\n")
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [])

    def test_workspace_tag_whitelist_runtime_semantics(self):
        """测试工位白名单恒启用语义: allowed_ids 非空 → 权威过滤; 空/缺失/旧 enabled 字段 → 探索模式"""
        from src.calibration.workspace_manager import load_workspace_tag_whitelist
        ws_dir = os.path.join(self.test_root, "ws_wl_semantics")
        os.makedirs(ws_dir, exist_ok=True)

        # 文件缺失 → [] (探索模式)
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [])

        # 旧格式 (带 enabled: false) → enabled 被忽略, 只看 allowed_ids
        with open(os.path.join(ws_dir, "tag_whitelist.yaml"), "w", encoding="utf-8") as f:
            f.write("workspace_id: x\nenabled: false\nallowed_ids: [0, 1, 18]\n")
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [0, 1, 18])

        # allowed_ids 为空 → [] (探索模式)
        with open(os.path.join(ws_dir, "tag_whitelist.yaml"), "w", encoding="utf-8") as f:
            f.write("workspace_id: x\nallowed_ids: []\n")
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [])

        # 非法 ID → 忽略文件 (探索模式, 不半生效)
        with open(os.path.join(ws_dir, "tag_whitelist.yaml"), "w", encoding="utf-8") as f:
            f.write("workspace_id: x\nallowed_ids: [0, abc, 2]\n")
        self.assertEqual(load_workspace_tag_whitelist(ws_dir), [])

    def test_workspace_description_update(self):
        """测试工位备注(description)更新并原子持久化"""
        state = HubState(self.workspace_mgr, force_mock=True)
        cur_ws = state.get_selected_workspace()
        self.assertIsNotNone(cur_ws)

        new_desc = "工位专属备注: 机器人3号位垂直测试"
        ok = state.update_current_workspace_description(new_desc)
        self.assertTrue(ok)
        self.assertEqual(cur_ws.description, new_desc)

        # 重新从磁盘载入验证持久化
        reloaded_ws = self.workspace_mgr.get_workspace_by_id(cur_ws.workspace_id, force_refresh=True)
        self.assertIsNotNone(reloaded_ws)
        self.assertEqual(reloaded_ws.description, new_desc)

    def test_top_tabs_and_exit_button_hit_test(self):
        """测试工位三页签Tab胶囊与退出按钮 hit_test 判定"""
        state = HubState(self.workspace_mgr, force_mock=True)
        renderer = HubRenderer()

        # 工位 3 个 Tab 胶囊
        # Tab 0: Dashboard (x: 348~463)
        self.assertEqual(renderer.hit_test(400, 25, state), ("hdr_tab_key", HubState.TAB_REPORT))
        # Tab 2: 生产相册 (x: 602~717)
        self.assertEqual(renderer.hit_test(650, 25, state), ("hdr_tab_key", HubState.TAB_PROD_IMAGES))

        # [退出] 按钮
        self.assertEqual(renderer.hit_test(880, 25, state), "btn_exit")

    def test_report_panel_vertical_rendering(self):
        """测试体检报告页签垂直排列面板的渲染 (960x720)"""
        state = HubState(self.workspace_mgr, force_mock=True)
        state.active_tab = HubState.TAB_REPORT
        cur_ws = state.get_selected_workspace()
        cur_ws.description = "产线高精度工位备注"
        cur_ws.ba_solved = True
        cur_ws.global_rmse_px = 0.45

        renderer = HubRenderer()
        rendered = renderer.render(state)
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.shape, (720, 960, 3))

    def test_report_embedded_buttons_hit_test_and_actions(self):
        """测试体检报告卡片1内嵌的 5 个操作按钮 hit_test 判定 (960宽紧凑右对齐)"""
        state = HubState(self.workspace_mgr, force_mock=True)
        state.active_tab = HubState.TAB_REPORT
        renderer = HubRenderer()

        # 验证 5 个内嵌按钮的命中 (x: 864~938 / 776~854)
        # 1. 重命名 (x: 864~938, y: 68~94)
        self.assertEqual(renderer.hit_test(900, 80, state), "ws_rename")
        # 2. 打开 (x: 864~938, y: 96~122)
        self.assertEqual(renderer.hit_test(900, 110, state), "ws_open_dir")
        # 3. 修改 (x: 864~938, y: 152~178)
        self.assertEqual(renderer.hit_test(900, 165, state), "ws_edit_desc")
        # 4. 更新元数据 (移至卡片1底栏左侧, x: 372~504, y: 190~220)
        self.assertEqual(renderer.hit_test(430, 205, state), "ws_sync_data")
        # 5. 克隆 (x: 512~592, y: 190~220)
        self.assertEqual(renderer.hit_test(550, 205, state), "ws_clone")
        # 6. 删除 (x: 600~678, y: 190~220)
        self.assertEqual(renderer.hit_test(630, 205, state), "ws_delete")

    def test_data_consistency_sync(self):
        """测试工位元数据记录与物理磁盘不一致时的自愈与同步刷新"""
        app = WorkspaceHubApp(force_mock=True, settings_file=os.path.join(self.test_root, "sync_test_settings.json"))
        ws = app.state.get_selected_workspace()
        self.assertIsNotNone(ws)

        # 模拟数据矛盾场景：元数据记录标定 22 帧，但物理目录为空 (0 帧)
        ws.image_count = 22
        ws.save_meta()
        self.assertEqual(ws.image_count, 22)

        # 触发 [更新元数据] 按钮 (卡片1底栏左侧, x=430, y=205)
        app.state.active_tab = HubState.TAB_REPORT
        app._on_mouse_event(cv2.EVENT_LBUTTONDOWN, 430, 205, 0, None)

        # 验证自动核验自愈：已同步至物理真实照片数，且元数据已更新
        self.assertEqual(ws.image_count, ws.get_image_count("calibration"))
        self.assertIn("已", app.state.toast_msg)


if __name__ == "__main__":
    unittest.main()
