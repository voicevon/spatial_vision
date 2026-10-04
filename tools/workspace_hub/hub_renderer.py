"""
Workspace Hub 视觉渲染引擎 (HubRenderer)
=======================================
专业工业级暗黑系 GUI 渲染管线，960x720 紧凑布局：
- 左栏 (x: 0~340): Workspace 列表导航 (固定稳定)
- 右栏 (x: 340~960): 动态页签区 (1 Dashboard / 2 Tag白名单 / 3 标定相册 / 4 ★ 生产相册)
- 标定相册页签内支持双击卡片进入全宽大图沉浸预览
"""

import os
import time
from typing import Any
import cv2
import numpy as np

from src.ui.gui_components import (
    render_floating_tooltip,
    draw_dropdown_button,
    render_dropdown_popup,
    draw_rounded_rectangle,
)
from src.ui.gui_theme import GuiTheme
from src.ui.text_rendering import draw_text, put_text
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_modals_renderer import HubModalsRenderer
from tools.workspace_hub.hub_hit_tester import HubHitTester
from tools.workspace_hub.pages import (
    DashboardPageRenderer,
    GalleryPageRenderer,
    FramesRoisPageRenderer,
)


# 顶部 Header 与页签胶囊几何常量 (单源标准)
HEADER_TAB_X0 = 348
HEADER_TAB_Y0 = 8
HEADER_TAB_W = 98
HEADER_TAB_H = 34
HEADER_TAB_STEP = 104
BTN_EXIT_X0 = 874
BTN_EXIT_Y0 = 8
BTN_EXIT_W = 74
BTN_EXIT_H = 34

# 图片卡片网格墙几何常量 (3 列 x 3 行, 大卡片 186x186, 网格铺满 340~960 区域)
GRID_X0 = 352
GRID_Y0 = 64
GRID_CELL_W = 186
GRID_CELL_H = 186
GRID_GAP_X = 16
GRID_GAP_Y = 12
GRID_THUMB_H = 156
GRID_COLS = 3
GRID_ROWS = 3

# 生产机制说明窗几何常量
HELP_MODAL_W = 860
HELP_MODAL_H = 490

# 工位大盘看板卡片内嵌按钮与下拉框几何常量 (单源标准)
WS_BTN_RENAME = (864, 68, 74, 26)
WS_BTN_OPEN_DIR = (864, 96, 74, 26)
WS_DROPDOWN_PROD_MODE = (426, 124, 512, 26)
WS_BTN_TOGGLE_PROD_MODE = WS_DROPDOWN_PROD_MODE
WS_BTN_EDIT_DESC = (864, 152, 74, 26)
WS_BTN_SYNC_DATA = (372, 192, 132, 28)
WS_BTN_CLONE = (512, 192, 80, 28)
WS_BTN_DELETE = (600, 192, 74, 28)
WS_BTN_NEW_FRAME = (682, 192, 130, 28)

# 【相机&内参】TAB 页签原位交互控件几何常量 (固定双行卡片, 界面零晃动)
INTR_BTN_REFRESH_DEV = (862, 66, 80, 24)
INTR_BTN_EDIT_CONFIG = (862, 100, 80, 24)
INTR_BTN_CAPTURE = (734, 154, 98, 28)
INTR_BTN_CALIB = (838, 154, 98, 28)


# ==================== 白名单芯片矩阵编辑器几何常量 (渲染与命中测试单源共用) ====================
WL_BOX_X, WL_BOX_Y, WL_BOX_W = 340, 50, 620          # 白名单页签容器
WL_GRID_X0, WL_GRID_Y0 = 360, 296                    # 芯片网格左上 (与只读矩阵视图一致)
WL_CELL_W, WL_CELL_H = 88, 38
WL_GAP_X, WL_GAP_Y = 10, 6
WL_COLS = 6
WL_BTN_DONE = (854, 58, 90, 30)                      # 标题行 [编辑]/[完成]
WL_BTN_ALL = (360, 624, 120, 32)                     # 全部放行
WL_BTN_CLEAR = (488, 624, 90, 32)                    # 清空 (探索模式)
WL_BTN_ANCHOR = (586, 624, 110, 32)                  # 锚点坐标 / 退出锚点
# 锚点弹窗 (Tag 世界坐标逐轴编辑, 支持部分已知)
WL_ANCHOR_X, WL_ANCHOR_Y, WL_ANCHOR_W, WL_ANCHOR_H = 420, 120, 460, 500
WL_ANCHOR_ROW_X0, WL_ANCHOR_ROW_Y0, WL_ANCHOR_ROW_W = 438, 182, 424
WL_ANCHOR_ROW_H, WL_ANCHOR_ROW_STEP = 38, 44
WL_ANCHOR_CLR_W = 74                                 # 行内 [清除] 按钮宽
WL_ANCHOR_KEY_X0, WL_ANCHOR_KEY_Y0, WL_ANCHOR_KEY_W, WL_ANCHOR_KEY_H = 438, 330, 96, 40
WL_ANCHOR_KEY_STEP_X, WL_ANCHOR_KEY_STEP_Y = 104, 48
WL_ANCHOR_SAVE = (438, 572, 130, 30)
WL_ANCHOR_CANCEL = (578, 572, 90, 30)
WL_ANCHOR_DELETE = (678, 572, 120, 30)

# ==================== 坐标系与 ROI 结构化编辑表单弹窗几何常量 ====================
GEOM_MODAL_W = 680
GEOM_MODAL_H = 560
GEOM_MODAL_X = (960 - GEOM_MODAL_W) // 2   # 140
GEOM_MODAL_Y = (720 - GEOM_MODAL_H) // 2   # 80
GEOM_MODAL_SAVE = (GEOM_MODAL_X + 180, GEOM_MODAL_Y + GEOM_MODAL_H - 48, 140, 36)
GEOM_MODAL_CANCEL = (GEOM_MODAL_X + 360, GEOM_MODAL_Y + GEOM_MODAL_H - 48, 140, 36)
GEOM_MODAL_CLOSE = (GEOM_MODAL_X + GEOM_MODAL_W - 46, GEOM_MODAL_Y + 12, 34, 30)

# 坐标系外参操作按钮
FRAME_PARAM_BTN_UNKNOWN = (GEOM_MODAL_X + GEOM_MODAL_W - 270, GEOM_MODAL_Y + 56 + 128 + 8, 115, 26)
FRAME_PARAM_BTN_EDIT6D = (GEOM_MODAL_X + GEOM_MODAL_W - 145, GEOM_MODAL_Y + 56 + 128 + 8, 125, 26)

# Smart ROI 专属交互按钮 (工艺角色、动作意图与槽位绑定)
ROI_ROLE_BTN_SOURCE = (GEOM_MODAL_X + 115, GEOM_MODAL_Y + 154, 120, 26)
ROI_ROLE_BTN_DEST = (GEOM_MODAL_X + 245, GEOM_MODAL_Y + 154, 130, 26)
ROI_ROLE_BTN_KEEPOUT = (GEOM_MODAL_X + 385, GEOM_MODAL_Y + 154, 120, 26)
ROI_ROLE_BTN_GENERAL = (GEOM_MODAL_X + 515, GEOM_MODAL_Y + 154, 125, 26)

ROI_INTENT_BTN_PICK = (GEOM_MODAL_X + 115, GEOM_MODAL_Y + 188, 120, 26)
ROI_INTENT_BTN_COUNT = (GEOM_MODAL_X + 245, GEOM_MODAL_Y + 188, 130, 26)
ROI_INTENT_BTN_OCC = (GEOM_MODAL_X + 385, GEOM_MODAL_Y + 188, 120, 26)
ROI_INTENT_BTN_GEN = (GEOM_MODAL_X + 515, GEOM_MODAL_Y + 188, 125, 26)

ROI_BIND_SLOT_BTN = (GEOM_MODAL_X + 115, GEOM_MODAL_Y + 222, 125, 26)
ROI_BIND_CAP_BTN = (GEOM_MODAL_X + 250, GEOM_MODAL_Y + 222, 125, 26)
ROI_BIND_CONF_BTN = (GEOM_MODAL_X + 385, GEOM_MODAL_Y + 222, 120, 26)

# ==================== 6DoF 位姿与约束独立模态弹窗几何常量 ====================
P6_MODAL_W = 620
P6_MODAL_H = 550
P6_MODAL_X = (960 - P6_MODAL_W) // 2  # 170
P6_MODAL_Y = (720 - P6_MODAL_H) // 2  # 85

# 顶部预设按钮
P6_BTN_UNKNOWN_ALL = (P6_MODAL_X + 24, P6_MODAL_Y + 62, 170, 28)
P6_BTN_KNOWN_ALL = (P6_MODAL_X + 204, P6_MODAL_Y + 62, 110, 28)
P6_BTN_PLANAR = (P6_MODAL_X + 324, P6_MODAL_Y + 62, 272, 28)

def pose6d_row_rect(axis_idx: int) -> tuple[int, int, int, int]:
    """6 轴位姿行矩形 (0~2: 平移 X, Y, Z; 3~5: 旋转 Roll, Pitch, Yaw)"""
    col = 0 if axis_idx < 3 else 1
    row = axis_idx % 3
    x = P6_MODAL_X + 24 if col == 0 else P6_MODAL_X + 316
    y = P6_MODAL_Y + 118 + row * 44
    return (x, y, 280, 38)

def pose6d_clear_rect(axis_idx: int) -> tuple[int, int, int, int]:
    """6 轴行内 [设为未知] 切换按钮矩形"""
    rx, ry, rw, rh = pose6d_row_rect(axis_idx)
    return (rx + rw - 72, ry + 6, 64, rh - 12)

# 15 键软键盘 (5 列 x 3 行: 1~5 / 6~0 / . -/+ 清空 退格 确认)
P6_PAD_LABELS = [
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9", "0",
    ".", "-/+", "清空", "退格", "确认"
]

def pose6d_padkey_rect(idx: int) -> tuple[int, int, int, int]:
    row, col = divmod(idx, 5)
    kx = P6_MODAL_X + 24 + col * 116
    ky = P6_MODAL_Y + 262 + row * 46
    return (kx, ky, 108, 38)

P6_BTN_SAVE = (P6_MODAL_X + 150, P6_MODAL_Y + P6_MODAL_H - 50, 150, 36)
P6_BTN_CANCEL = (P6_MODAL_X + 320, P6_MODAL_Y + P6_MODAL_H - 50, 150, 36)


def frame_btn_add_rect() -> tuple[int, int, int, int]:
    return (824, 98, 108, 24)

def roi_btn_add_rect() -> tuple[int, int, int, int]:
    return (836, 368, 96, 24)

def frame_row_edit_rect(idx: int) -> tuple[int, int, int, int]:
    fy = 136 + idx * 72
    return (818, fy + 6, 50, 24)

def frame_row_del_rect(idx: int) -> tuple[int, int, int, int]:
    fy = 136 + idx * 72
    return (874, fy + 6, 50, 24)

def roi_row_edit_rect(idx: int) -> tuple[int, int, int, int]:
    ry = 406 + idx * 76
    return (818, ry + 6, 50, 24)

def roi_row_del_rect(idx: int) -> tuple[int, int, int, int]:
    ry = 406 + idx * 76
    return (874, ry + 6, 50, 24)

# ==================== 坐标系专属视图几何常量 ====================
# 内嵌于坐标系卡片 (c1_y = 60) 顶行右侧
FRAME_EDIT_POSE_BTN = (862, 70, 70, 26)
FRAME_DELETE_BTN = (788, 70, 66, 26)
FRAME_ADD_ROI_BTN = (780, 58, 160, 30)

# 内嵌于 Tags 标靶卡片 (c2_y = 208) 首行右侧
FRAME_TAG_EDIT_SIZE_BTN = (796, 222, 136, 28)

# ==================== 标靶物理边长专属模态弹窗几何 ====================
MS_MODAL_W = 460
MS_MODAL_H = 350
MS_MODAL_X = 250
MS_MODAL_Y = 185

MS_BTN_SAVE = (MS_MODAL_X + 224, MS_MODAL_Y + MS_MODAL_H - 46, 210, 36)
MS_BTN_CANCEL = (MS_MODAL_X + 24, MS_MODAL_Y + MS_MODAL_H - 46, 110, 36)

MS_PAD_LABELS = [
    "1", "2", "3", "退格",
    "4", "5", "6", "清空",
    "7", "8", "9", "0",
    ".", "35.5", "50.0", "40.0"
]

def ms_padkey_rect(idx: int) -> tuple[int, int, int, int]:
    r, c = divmod(idx, 4)
    kx = MS_MODAL_X + 24 + c * 105
    ky = MS_MODAL_Y + 136 + r * 40
    return (kx, ky, 96, 34)

# 内嵌于 Tags 标靶卡片下半部分的 10-Slot 放行矩阵
FT_GRID_X0 = 359
FT_GRID_Y0 = 326
FT_CHIP_W = 110
FT_CHIP_H = 68
FT_GAP_X = 8
FT_GAP_Y = 14

def frame_tag_chip_rect(idx: int) -> tuple[int, int, int, int]:
    """计算 10-Slot Tag 芯片矩阵中第 idx (0~9) 个芯片的矩形 (2行x5列)"""
    row, col = divmod(idx, 5)
    return (FT_GRID_X0 + col * (FT_CHIP_W + FT_GAP_X),
            FT_GRID_Y0 + row * (FT_CHIP_H + FT_GAP_Y),
            FT_CHIP_W, FT_CHIP_H)

def frame_roi_row_rect(idx: int) -> tuple[int, int, int, int]:
    return (356, 100 + idx * 80, 574, 72)

def frame_roi_edit_btn(idx: int) -> tuple[int, int, int, int]:
    return (804, 100 + idx * 80 + 22, 54, 26)

def frame_roi_del_btn(idx: int) -> tuple[int, int, int, int]:
    return (864, 100 + idx * 80 + 22, 54, 26)

FRAME_ROI_PREV_BTN = (660, 58, 54, 28)
FRAME_ROI_NEXT_BTN = (718, 58, 54, 28)
FRAME_ROI_SCROLL_TRACK = (942, 100, 8, 552)


def whitelist_cell_rect(t_id: int) -> tuple[int, int, int, int]:
    """白名单芯片格位矩形 (0~29 基础网格与超出范围的追加芯片共用同一公式)"""
    row, col = divmod(t_id, WL_COLS)
    return (WL_GRID_X0 + col * (WL_CELL_W + WL_GAP_X),
            WL_GRID_Y0 + row * (WL_CELL_H + WL_GAP_Y),
            WL_CELL_W, WL_CELL_H)


def anchor_row_rect(axis: int) -> tuple[int, int, int, int]:
    """锚点弹窗轴行矩形 (axis 0/1/2 → X/Y/Z)"""
    return (WL_ANCHOR_ROW_X0, WL_ANCHOR_ROW_Y0 + axis * WL_ANCHOR_ROW_STEP,
            WL_ANCHOR_ROW_W, WL_ANCHOR_ROW_H)


def anchor_clear_rect(axis: int) -> tuple[int, int, int, int]:
    """锚点弹窗轴行内 [清除] 按钮矩形"""
    rx, ry, rw, rh = anchor_row_rect(axis)
    return (rx + rw - 10 - WL_ANCHOR_CLR_W, ry + 6, WL_ANCHOR_CLR_W, rh - 12)


def anchor_padkey_rect(idx: int) -> tuple[int, int, int, int]:
    """锚点键盘按键矩形 (idx 0~14: 1~9/./0/-+/清空/退格/确认)"""
    row, col = divmod(idx, 3)
    return (WL_ANCHOR_KEY_X0 + col * WL_ANCHOR_KEY_STEP_X,
            WL_ANCHOR_KEY_Y0 + row * WL_ANCHOR_KEY_STEP_Y,
            WL_ANCHOR_KEY_W, WL_ANCHOR_KEY_H)


def point_in_rect(x: int, y: int, rect: tuple[int, int, int, int]) -> bool:
    """点是否落在矩形内 (Hover 高亮与命中测试共用)"""
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def grid_hit_test(mx: int, my: int, y_offset: int = 0) -> int | None:
    """根据逻辑坐标返回命中的卡片格位索引 (0~8)；落在卡片间隙或网格外返回 None"""
    y0 = GRID_Y0 + y_offset
    if mx < GRID_X0 or my < y0:
        return None
    col = (mx - GRID_X0) // (GRID_CELL_W + GRID_GAP_X)
    row = (my - y0) // (GRID_CELL_H + GRID_GAP_Y)
    if col >= GRID_COLS or row >= GRID_ROWS:
        return None
    local_x = (mx - GRID_X0) % (GRID_CELL_W + GRID_GAP_X)
    local_y = (my - y0) % (GRID_CELL_H + GRID_GAP_Y)
    if local_x >= GRID_CELL_W or local_y >= GRID_CELL_H:
        return None
    return row * GRID_COLS + col


class HubRenderer:
    """Workspace Hub 统一界面渲染器"""

    # 图片卡片网格墙几何 (引用模块级常量，便于渲染与命中测试共用)
    GRID_X0 = GRID_X0
    GRID_Y0 = GRID_Y0
    GRID_CELL_W = GRID_CELL_W
    GRID_CELL_H = GRID_CELL_H
    GRID_GAP_X = GRID_GAP_X
    GRID_GAP_Y = GRID_GAP_Y
    GRID_THUMB_H = GRID_THUMB_H

    # 页签显示文案单源位于 HubState.get_current_tabs()

    def __init__(self):
        self.canvas_w = 960
        self.canvas_h = 720

        # 调色盘: 结构色统一取自 GuiTheme 主题单源, 品牌色(青/金/暗灰) 本地保留
        self.COLOR_BG = GuiTheme.BG             # 全局底色
        self.COLOR_PANEL = GuiTheme.CARD_BG     # 侧边栏/卡片底色
        self.COLOR_CARD_ACTIVE = GuiTheme.CARD_SEL  # 选中卡片底色
        self.COLOR_BORDER = GuiTheme.BORDER     # 普通线条
        self.COLOR_ACTIVE_BORDER = GuiTheme.BORDER_SEL  # 选中项高亮描边
        self.COLOR_CYAN = (230, 200, 0)        # 科技青 (BGR: 0, 200, 230)
        self.COLOR_GOLD = (50, 190, 255)       # 金黄色 (BGR)
        self.COLOR_WHITE = GuiTheme.WHITE
        self.COLOR_GRAY = GuiTheme.GRAY
        self.COLOR_DARK_GRAY = (70, 75, 85)

        # 帧级极速缓存 (毫秒级响应 Hover 交互)
        self._cached_canvas: np.ndarray | None = None
        self._last_cache_key: Any = None

        # 专职模态弹窗与浮层渲染器
        self.modals = HubModalsRenderer(self)
        # 专职交互热区与碰撞探测引擎
        self.hit_tester = HubHitTester(self)
        # 专职页面渲染组件
        self.dashboard_page = DashboardPageRenderer(self)
        self.gallery_page = GalleryPageRenderer(self)
        self.frames_rois_page = FramesRoisPageRenderer(self)

    def render(self, state: HubState) -> np.ndarray:
        """根据当前状态机渲染 1280x720 最终画布 (双缓冲极速渲染)"""
        # 计算当前交互状态的哈希指纹，命中缓存则零拷贝直接返回！
        selected_ws = state.get_selected_workspace()
        cache_key = (
            state.selected_workspace_idx,
            selected_ws.workspace_id if selected_ws else None,
            len(state.workspaces),
            state.gallery.selected_image_idx,
            len(state.gallery.current_images),
            state.gallery.image_grid_offset,
            state.gallery.selected_prod_image_idx,
            len(state.gallery.prod_images),
            state.gallery.prod_grid_offset,
            state.gallery.view_mode,
            state.active_tab,
            state.selected_tree_item,
            tuple(sorted(state.expanded_workspaces)),
            state.whitelist._whitelist_cache_ws,
            state.whitelist._whitelist_cache_mtime,
            state.toast_msg,
            state.is_help_modal_open,
            state.geometry.frame_modal_open,
            state.geometry.roi_modal_open,
            state.geometry.active_dropdown,
            state.active_dropdown,
            str(state.geometry.frame_modal_data),
            str(state.geometry.roi_modal_data),
            state.mouse_x,
            state.mouse_y,
            int(time.time() * 2)  # 每 500ms 刷新时间敏感的 Toast 与动画
        )
        if self._cached_canvas is not None and self._last_cache_key == cache_key:
            return self._cached_canvas

        canvas = np.full((self.canvas_h, self.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        # 1. 顶部状态栏 (y: 0~50): 标题 + 右侧动态区四页签 Tab + 紧邻的退出按钮
        self._render_header(canvas, state)

        # 3. 左侧综合导航栏 (x: 0~340, y: 50~670) - 切换页签过程中保持稳定
        self._render_left_panel(canvas, state)
        cv2.line(canvas, (340, 50), (340, 670), self.COLOR_BORDER, 1)

        # 4. 右侧动态区 (x: 340~960, y: 50~670): 动态页签内容 + 全宽大图沉浸
        ws = state.get_selected_workspace()
        if state.gallery.view_mode == HubState.VIEW_EXPANDED:
            self.gallery_page.render_expanded_photo_preview(canvas, state, ws)
        elif state.active_tab == HubState.TAB_REPORT:
            self.dashboard_page.render(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAME_POSE_TAGS:
            self.frames_rois_page.render_frame_pose_tags(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAME_ROIS:
            self.frames_rois_page.render_frame_rois(canvas, state, ws)
        elif state.active_tab == HubState.TAB_INTRINSICS_IMAGES:
            self.gallery_page.render_intrinsics_images(canvas, state, ws)
        elif state.active_tab == HubState.TAB_PROD_IMAGES:
            self.gallery_page.render_prod_images(canvas, state, ws)
        else:
            self.gallery_page.render_calib_images(canvas, state, ws)

        # 5. 底部系统反馈提示栏 (y: 670~720)
        self._render_footer(canvas, state)

        # 6. 如果打开了结构化弹窗，最高优先级置顶展示
        if state.geometry.pose6d_modal_open:
            self.modals.render_pose6d_modal(canvas, state)
        elif state.whitelist.marker_size_modal_open:
            self.modals.render_marker_size_modal(canvas, state)
        elif state.geometry.frame_modal_open:
            self.modals.render_frame_modal(canvas, state)
        elif state.geometry.roi_modal_open:
            self.modals.render_roi_modal(canvas, state)
        elif state.whitelist.anchor_modal_open:
            self.modals.render_anchor_modal(canvas, state)
        elif state.is_help_modal_open:
            self.modals.render_help_modal(canvas, state)

        # 7. 悬浮 Tooltip 气泡提示 (置于最顶层，无遮挡呈现)
        if not (state.geometry.pose6d_modal_open or state.whitelist.marker_size_modal_open or state.geometry.frame_modal_open or state.geometry.roi_modal_open or state.whitelist.anchor_modal_open or state.is_help_modal_open):
            if state.active_tab == HubState.TAB_FRAME_POSE_TAGS and state.gallery.view_mode == HubState.VIEW_STANDARD:
                mpos = (state.mouse_x, state.mouse_y)
                if self._should_show_tag_bound_tooltip(state, mpos):
                    self._draw_tag_bound_tooltip(canvas, mpos)


        self._cached_canvas = canvas
        self._last_cache_key = cache_key
        return canvas

    def _render_header(self, canvas: np.ndarray, state: HubState):
        """渲染顶部标题栏 (0~50px) - 包含右侧动态区四页签Tab及紧贴生产相册的退出按钮"""
        cv2.rectangle(canvas, (0, 0), (self.canvas_w, 50), (14, 16, 20), -1)
        cv2.line(canvas, (0, 50), (self.canvas_w, 50), self.COLOR_BORDER, 1)
        mpos = (state.mouse_x, state.mouse_y)

        # 1. 系统标题与状态点 (x: 16~260)
        cv2.circle(canvas, (22, 25), 6, (0, 255, 180), -1)
        put_text(canvas, "flux_vision_3d", (36, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COLOR_CYAN, 2, cv2.LINE_AA)
        draw_text(canvas, "Workspace", (165, 16), font_size=17, color=self.COLOR_WHITE, bold=True)

        # 2. 右侧动态区四页签 Tab 胶囊
        self._render_header_tabs(canvas, state)

        # 3. [退出] 按钮紧贴生产相册右侧
        self._draw_button(canvas, (BTN_EXIT_X0, BTN_EXIT_Y0, BTN_EXIT_W, BTN_EXIT_H), "退出", mpos, theme_color=(180, 60, 60))

    def _render_header_tabs(self, canvas: np.ndarray, state: HubState):
        """渲染顶部自适应 Tab 胶囊 (委托给统一通用 TabBar 控件)"""
        mpos = (state.mouse_x, state.mouse_y)
        state.tab_bar.render(canvas, (HEADER_TAB_X0, HEADER_TAB_Y0, 510, HEADER_TAB_H), mouse_pos=mpos)

    def _draw_button(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str,
                     mouse_pos: tuple[int, int], is_active: bool = False,
                     theme_color: tuple[int, int, int] = (0, 200, 140),
                     enabled: bool = True) -> bool:
        """统一绘制现代科技风交互按钮，带精准 Hover 高亮检测，返回是否处于 hover 状态"""
        bx, by, bw, bh = rect
        mx, my = mouse_pos
        is_hover = enabled and (bx <= mx <= bx + bw and by <= my <= by + bh)

        if not enabled:
            bg_col = (18, 22, 28)
            border_col = (34, 42, 52)
            text_col = (85, 98, 115)
            thickness = 1
        elif is_hover:
            bg_col = (34, 46, 56)
            border_col = (0, 255, 180)  # 荧光亮绿高亮
            text_col = (0, 255, 200)
            thickness = 2
        elif is_active:
            bg_col = (28, 40, 48)
            border_col = theme_color
            text_col = self.COLOR_WHITE
            thickness = 2
        else:
            bg_col = (22, 28, 36)
            border_col = (45, 68, 62)   # 统一沉稳科技暗绿框
            text_col = (205, 225, 220)
            thickness = 1

        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), bg_col, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), border_col, thickness)

        # 文字自适应居中
        approx_w = sum(14 if ord(c) > 127 else 8 for c in text)
        tx = bx + max(4, (bw - approx_w) // 2)
        draw_text(canvas, text, (tx, by + (bh - 18) // 2), font_size=13,
                  color=text_col, bold=is_hover)
        return is_hover

    def _render_left_panel(self, canvas: np.ndarray, state: HubState):
        """渲染左侧两层树结构导航 (x: 0~340, y: 50~670)
        - 一级节点: Workspace (工位)，带 ▼ / ▶ 展开折叠指示器与统计
        - 二级节点: Coordinate Frame (坐标系)，带缩进
        - 底部保留: + 新建 Workspace
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        tree_items = self.hit_tester.get_tree_layout(state)
        for item in tree_items:
            if item["type"] == "workspace":
                ws_idx = item["ws_idx"]
                ws = item["workspace"]
                is_sel = item["is_selected"]
                is_exp = item["is_expanded"]
                wx, wy, ww, wh = item["ws_rect"]

                card_bg = self.COLOR_CARD_ACTIVE if is_sel else (26, 30, 40)
                card_border = self.COLOR_ACTIVE_BORDER if is_sel else self.COLOR_BORDER
                cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), card_bg, -1)
                cv2.rectangle(canvas, (wx, wy), (wx + ww, wy + wh), card_border, 2 if is_sel else 1)
                if is_sel:
                    cv2.rectangle(canvas, (wx, wy), (wx + 4, wy + wh), (0, 255, 160), -1)

                arrow_char = "▼" if is_exp else "▶"
                arrow_col = (0, 255, 200) if is_sel else (140, 160, 180)
                draw_text(canvas, arrow_char, (wx + 8, wy + 14), font_size=13, color=arrow_col, bold=True)

                disp_title = f"{ws_idx + 1:02d}. {ws.name}"
                title_col = (0, 255, 200) if is_sel else self.COLOR_WHITE
                draw_text(canvas, disp_title, (wx + 28, wy + 6), font_size=13, color=title_col, bold=is_sel)

                if ws.ba_solved and ws.global_rmse_px > 1e-6:
                    ba_badge = f"RMSE: {ws.global_rmse_px:.2f}px"
                    ba_col = (0, 220, 100) if ws.global_rmse_px < 0.8 else self.COLOR_GOLD
                else:
                    ba_badge = "未平差"
                    ba_col = self.COLOR_DARK_GRAY
                put_text(canvas, ba_badge, (wx + 28, wy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, ba_col, 1, cv2.LINE_AA)

                cnt_text = f"{ws.image_count}帧/{ws.prod_image_count}帧"
                put_text(canvas, cnt_text, (wx + 200, wy + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 160, 180), 1, cv2.LINE_AA)

            elif item["type"] == "frame":
                fx, fy, fw, fh = item["rect"]
                f = item["frame"]
                is_sel = item["is_selected"]

                if is_sel:
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (22, 42, 46), -1)
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (0, 255, 200), 1)
                    cv2.rectangle(canvas, (fx, fy), (fx + 3, fy + fh), (0, 255, 200), -1)
                else:
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (20, 24, 32), -1)
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), (35, 42, 54), 1)

                tree_branch = "└" if f.frame_id != "world" else "•"
                draw_text(canvas, tree_branch, (fx + 6, fy + 7), font_size=12, color=(0, 200, 240))

                f_label = f"{f.frame_id}"
                if f.name and f.name != f.frame_id:
                    f_label += f" ({f.name})"
                if len(f_label) > 22:
                    f_label = f_label[:20] + ".."
                f_col = (0, 255, 220) if is_sel else (200, 215, 230)
                draw_text(canvas, f_label, (fx + 20, fy + 7), font_size=12, color=f_col, bold=is_sel)

        # ==== 2. Workspace 通用全局操作区 (经典上下结构，预留56px底边距防截断) ====
        div_y1 = 574
        cv2.line(canvas, (10, div_y1), (330, div_y1), self.COLOR_BORDER, 1)

        btn_w = 320
        btn_h = 36
        btn1_y = 584
        self._draw_button(canvas, (10, btn1_y, btn_w, btn_h), "+ 新建工位", mpos)

        btn2_y = 628
        self._draw_button(canvas, (10, btn2_y, btn_w, btn_h), "全工位体检", mpos, theme_color=(0, 200, 160))

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部暗色底栏 (670~720px) - 保持纯净留白，无冗余干扰文本"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)


    def _should_show_tag_bound_tooltip(self, state: HubState, mpos: tuple[int, int]) -> bool:
        """检测鼠标是否悬停在动标帮助提示胶囊或动标参数信息区域"""
        mx, my = mpos
        card_x = 352
        c1_y = 60
        # 1. 动标定义说明徽章
        help_btn_x, help_btn_y, help_btn_w, help_btn_h = card_x + 280, c1_y + 34, 126, 22
        if help_btn_x <= mx <= help_btn_x + help_btn_w and help_btn_y <= my <= help_btn_y + help_btn_h:
            return True
        # 2. 如果是动标类型坐标系，悬停在动标参数卡片上亦弹出完整解释
        cur_frame = state.geometry.get_selected_frame()
        if cur_frame and cur_frame.type == "tag_bound":
            tag_box_y = c1_y + 64
            if card_x + 16 <= mx <= card_x + 596 - 16 and tag_box_y <= my <= tag_box_y + 44:
                return True
        return False

    def _draw_tag_bound_tooltip(self, canvas: np.ndarray, anchor_pos: tuple[int, int]):
        title = "AprilTag 动标机制与多 Tag 绑定说明"
        lines = [
            "【动标 (Dynamic Tag) 的定义】",
            "• 安装在【运动机构部件】(如活动滑块、推手、法兰夹爪) 上的视觉标靶；",
            "• 随机构运动实时改变空间坐标，相机每一帧识别并动态解算该部件位姿。",
            "",
            "【为什么此处是一个列表 (可配置多个 Tag)？】",
            "• 工业现场中单动标极易发生受光反光、物料遮挡或大倾角失锁；",
            "• 部件绑定多动标列表 (如 #10, #11) 时，系统启用多标冗余追踪机制；",
            "• 只要视野中能稳定观测到列表中的任意一个动标，即可持续求解位姿！",
            "",
            "【标称安装偏移 (offset)】",
            "• 动标贴片几何中心到机构部件实际旋转轴或受控特征原点的物理装配偏差。",
            "",
            "★ 注意: world 世界坐标系是固定空间基准原点，依靠下方静态标靶阵列定位。"
        ]
        render_floating_tooltip(canvas, title, lines, anchor_pos)



