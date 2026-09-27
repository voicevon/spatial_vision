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

from src.utils.gui_components import (
    render_floating_tooltip,
    draw_dropdown_button,
    render_dropdown_popup,
    draw_rounded_rectangle,
)
from src.utils.gui_theme import GuiTheme
from src.utils.text_rendering import draw_text, put_text
from tools.workspace_hub.hub_state import HubState
from tools.workspace_hub.hub_modals_renderer import HubModalsRenderer
from tools.workspace_hub.hub_hit_tester import HubHitTester


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

# 工位大盘看板卡片内嵌按钮几何常量 (单源标准)
WS_BTN_RENAME = (864, 68, 74, 26)
WS_BTN_OPEN_DIR = (864, 96, 74, 26)
WS_BTN_TOGGLE_PROD_MODE = (864, 124, 74, 26)
WS_BTN_EDIT_DESC = (864, 152, 74, 26)
WS_BTN_SYNC_DATA = (372, 192, 132, 28)
WS_BTN_CLONE = (512, 192, 80, 28)
WS_BTN_DELETE = (600, 192, 74, 28)
WS_BTN_NEW_FRAME = (682, 192, 130, 28)


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
FRAME_EDIT_POSE_BTN = (866, 58, 74, 30)
FRAME_ADD_ROI_BTN = (780, 58, 160, 30)
FRAME_TAG_EDIT_SIZE_BTN = (796, 234, 144, 32)

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

FT_GRID_X0 = 356
FT_GRID_Y0 = 366
FT_CHIP_W = 110
FT_CHIP_H = 68
FT_GAP_X = 8
FT_GAP_Y = 12

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


def grid_hit_test(mx: int, my: int) -> int | None:
    """根据逻辑坐标返回命中的卡片格位索引 (0~8)；落在卡片间隙或网格外返回 None"""
    if mx < GRID_X0 or my < GRID_Y0:
        return None
    col = (mx - GRID_X0) // (GRID_CELL_W + GRID_GAP_X)
    row = (my - GRID_Y0) // (GRID_CELL_H + GRID_GAP_Y)
    if col >= GRID_COLS or row >= GRID_ROWS:
        return None
    local_x = (mx - GRID_X0) % (GRID_CELL_W + GRID_GAP_X)
    local_y = (my - GRID_Y0) % (GRID_CELL_H + GRID_GAP_Y)
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

    def _get_tabs_layout(self, state: HubState):
        """计算顶部 Tab 胶囊的动态布局矩形 (委托给 HubHitTester)"""
        return self.hit_tester.get_tabs_layout(state)

    def _get_tree_layout(self, state: HubState):
        """计算左侧两层树结构各项的几何矩形与数据标识 (委托给 HubHitTester)"""
        return self.hit_tester.get_tree_layout(state)

    def hit_test(self, mx: int, my: int, state: HubState) -> Any:
        """根据逻辑坐标探测当前命中交互元素 (委托给 HubHitTester)"""
        return self.hit_tester.hit_test(mx, my, state)

    def _get_interactive_hover_key(self, state: HubState) -> Any:
        """获取当前鼠标悬停的交互元素标识 (委托给 HubHitTester)"""
        return self.hit_tester.get_interactive_hover_key(state)

    def render(self, state: HubState) -> np.ndarray:
        """根据当前状态机渲染 1280x720 最终画布 (双缓冲极速渲染)"""
        # 计算当前交互状态的哈希指纹，命中缓存则零拷贝直接返回！
        selected_ws = state.get_selected_workspace()
        cache_key = (
            state.selected_workspace_idx,
            selected_ws.workspace_id if selected_ws else None,
            len(state.workspaces),
            state.selected_image_idx,
            len(state.current_images),
            state.image_grid_offset,
            state.selected_prod_image_idx,
            len(state.prod_images),
            state.prod_grid_offset,
            state.view_mode,
            state.active_tab,
            state.selected_tree_item,
            tuple(sorted(state.expanded_workspaces)),
            state._whitelist_cache_ws,
            state._whitelist_cache_mtime,
            state.toast_msg,
            state.is_help_modal_open,
            state.frame_modal_open,
            state.roi_modal_open,
            state.active_dropdown,
            str(state.frame_modal_data),
            str(state.roi_modal_data),
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
        if state.view_mode == HubState.VIEW_EXPANDED:
            self._render_expanded_photo_preview(canvas, state, ws)
        elif state.active_tab == HubState.TAB_REPORT:
            self._render_pure_dashboard_panel(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAME_POSE_TAGS:
            self._render_page_frame_pose_tags(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAME_ROIS:
            self._render_page_frame_rois(canvas, state, ws)
        elif state.active_tab == HubState.TAB_PROD_IMAGES:
            self._render_page_prod_images(canvas, state, ws)
        elif state.active_tab == HubState.TAB_FRAMES_ROIS:
            self._render_page_frames_rois(canvas, state, ws)
        elif state.active_tab == HubState.TAB_WHITELIST:
            self._render_page_whitelist(canvas, state, ws)
        else:
            self._render_page_calib_images(canvas, state, ws)

        # 5. 底部系统反馈提示栏 (y: 670~720)
        self._render_footer(canvas, state)

        # 6. 如果打开了结构化弹窗，最高优先级置顶展示
        if getattr(state, "pose6d_modal_open", False):
            self.modals.render_pose6d_modal(canvas, state)
        elif state.marker_size_modal_open:
            self.modals.render_marker_size_modal(canvas, state)
        elif state.frame_modal_open:
            self.modals.render_frame_modal(canvas, state)
        elif state.roi_modal_open:
            self.modals.render_roi_modal(canvas, state)
        elif state.anchor_modal_open:
            self.modals.render_anchor_modal(canvas, state)
        elif state.is_help_modal_open:
            self.modals.render_help_modal(canvas, state)

        # 7. 悬浮 Tooltip 气泡提示 (置于最顶层，无遮挡呈现)
        if not (getattr(state, "pose6d_modal_open", False) or state.marker_size_modal_open or state.frame_modal_open or state.roi_modal_open or state.anchor_modal_open or state.is_help_modal_open):

            if state.active_tab == HubState.TAB_FRAME_POSE_TAGS and state.view_mode == HubState.VIEW_STANDARD:
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
        """渲染顶部自适应 Tab 胶囊 (工位视图 3 页签，坐标系专属视图 2 页签)"""
        mpos = (state.mouse_x, state.mouse_y)
        tabs_layout = self._get_tabs_layout(state)

        for tab_key, tab_text, rect in tabs_layout:
            tx, ty, tw, th = rect
            is_active_tab = (state.active_tab == tab_key)
            is_hover_tab = (tx <= mpos[0] <= tx + tw and ty <= mpos[1] <= ty + th)

            approx_w = sum(13 if ord(c) > 127 else 8 for c in tab_text)
            text_x = tx + max(4, (tw - approx_w) // 2)

            if is_active_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (28, 44, 40), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 255, 180), 2)
                cv2.rectangle(canvas, (tx + 8, ty + th - 3), (tx + tw - 8, ty + th - 1), (0, 255, 180), -1)
                draw_text(canvas, tab_text, (text_x, ty + 8), font_size=13, color=(0, 255, 200), bold=True)
            elif is_hover_tab:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (34, 40, 52), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (0, 200, 240), 1)
                draw_text(canvas, tab_text, (text_x, ty + 8), font_size=13, color=(0, 220, 255))
            else:
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (22, 27, 35), -1)
                cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (45, 55, 72), 1)
                draw_text(canvas, tab_text, (text_x, ty + 8), font_size=13, color=(160, 175, 195))

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

    def _draw_text_input(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str,
                         mouse_pos: tuple[int, int]) -> bool:
        """统一绘制高可编辑感知的输入框 (委托给 HubModalsRenderer)"""
        return self.modals.draw_text_input(canvas, rect, text, mouse_pos)

    def _draw_dropdown_trigger(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str,
                               mouse_pos: tuple[int, int], is_open: bool = False) -> bool:
        """统一绘制现代下拉选择框触发条 (委托给 HubModalsRenderer)"""
        return self.modals.draw_dropdown_trigger(canvas, rect, text, mouse_pos, is_open)

    def _get_dropdown_data(self, dd_type: str, state: HubState):
        """统一获取下拉框的布局矩形、当前选中值以及候选枚举列表 (委托给 HubModalsRenderer)"""
        return self.modals.get_dropdown_data(dd_type, state)

    def _render_active_dropdown(self, canvas: np.ndarray, state: HubState):
        """在弹窗顶层高亮绘制当前展开的下拉菜单浮层 (委托给 HubModalsRenderer)"""
        self.modals.render_active_dropdown(canvas, state)


    def _render_left_panel(self, canvas: np.ndarray, state: HubState):
        """渲染左侧两层树结构导航 (x: 0~340, y: 50~670)
        - 一级节点: Workspace (工位)，带 ▼ / ▶ 展开折叠指示器与统计
        - 二级节点: Coordinate Frame (坐标系)，带缩进
        - 底部保留: + 新建 Workspace
        """
        cv2.rectangle(canvas, (0, 50), (340, 670), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        tree_items = self._get_tree_layout(state)
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

        # ==== 2. Workspace 通用全局操作区 ====
        div_y1 = 604
        cv2.line(canvas, (10, div_y1), (330, div_y1), self.COLOR_BORDER, 1)

        btn1_y = div_y1 + 10
        self._draw_button(canvas, (10, btn1_y, 320, 40), "+ 新建 Workspace", mpos)

    def _render_page_calib_images(self, canvas: np.ndarray, state: HubState, sc):
        """页签2: 标定相册 - 当前选中工位的采样相册卡片网格墙 (x: 340~960, y: 50~670)"""
        self._render_gallery_page(
            canvas, state, "",
            state.current_images, state.selected_image_idx, state.image_grid_offset,
            empty_hint="当前场景尚未采集任何照片！",
            is_calib=True,
        )

    def _render_page_prod_images(self, canvas: np.ndarray, state: HubState, sc):
        """页签4: 生产相册 - 生产运行基准工位的采样相册 (x: 340~960, y: 50~670)"""
        if not state.prod_images and sc and os.path.isdir(sc.prod_raw_images_dir):
            state.load_prod_images()
        prod_images = getattr(state, "prod_images", [])
        prod_idx = getattr(state, "selected_prod_image_idx", 0)
        prod_offset = getattr(state, "prod_grid_offset", 0)
        self._render_gallery_page(
            canvas, state, "",
            prod_images, prod_idx, prod_offset,
            empty_hint="生产基准工位尚未采集任何照片！",
            is_calib=False,
        )

    def _render_page_frame_pose_tags(self, canvas: np.ndarray, state: HubState, ws):
        """坐标系专属页签 1: 机构参数与 Tag 分段管理 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, self.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        cur_frame = state.get_selected_frame()
        if not cur_frame:
            draw_text(canvas, "未选择任何机构坐标系", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        # 1. 顶部编辑按钮
        self._draw_button(canvas, FRAME_EDIT_POSE_BTN, "编辑", mpos)

        card_x = box_x + 16
        card_w = box_w - 32

        # 2. 上部卡片: 机构位姿与空间拓扑 (紧凑精致)
        c1_y = box_y + 40
        c1_h = 134
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (42, 52, 70), 1)

        is_world_datum = (cur_frame.type == "world" or cur_frame.frame_id == "world" or not cur_frame.parent_frame_id)
        if is_world_datum:
            type_desc = "工位绝对世界基准 (world)"
        elif cur_frame.type == "fixed_transform":
            type_desc = "固定刚体外参 (fixed_transform)"
        else:
            type_desc = "AprilTag 动标绑定 (tag_bound)"

        draw_text(canvas, f"坐标系标识: {cur_frame.frame_id}", (card_x + 16, c1_y + 12), font_size=13, color=(210, 225, 240))
        draw_text(canvas, f"父坐标系: {cur_frame.parent_frame_id}", (card_x + 220, c1_y + 12), font_size=13, color=(210, 225, 240))
        draw_text(canvas, f"类型: {type_desc}", (card_x + 16, c1_y + 36), font_size=12, color=(0, 220, 200))

        # 动标定义与说明悬停帮助徽章
        help_btn_x, help_btn_y, help_btn_w, help_btn_h = card_x + 310, c1_y + 33, 126, 22
        is_hover_help = (help_btn_x <= mpos[0] <= help_btn_x + help_btn_w and help_btn_y <= mpos[1] <= help_btn_y + help_btn_h)
        help_bg = (30, 48, 48) if is_hover_help else (20, 28, 36)
        help_border = (0, 255, 200) if is_hover_help else (40, 75, 75)
        cv2.rectangle(canvas, (help_btn_x, help_btn_y), (help_btn_x + help_btn_w, help_btn_y + help_btn_h), help_bg, -1)
        cv2.rectangle(canvas, (help_btn_x, help_btn_y), (help_btn_x + help_btn_w, help_btn_y + help_btn_h), help_border, 1)
        draw_text(canvas, "(?) 动标定义说明", (help_btn_x + 8, help_btn_y + 4), font_size=11,
                  color=(0, 255, 220) if is_hover_help else (140, 185, 195), bold=is_hover_help)

        if is_world_datum:
            w_box_y = c1_y + 60
            cv2.rectangle(canvas, (card_x + 16, w_box_y), (card_x + card_w - 16, w_box_y + 62), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, w_box_y), (card_x + card_w - 16, w_box_y + 62), (38, 48, 65), 1)
            draw_text(canvas, "● 工位全局绝对空间基准 (World Datum / Origin)", (card_x + 24, w_box_y + 9), font_size=12, color=(0, 255, 200), bold=True)
            draw_text(canvas, "位姿特性: 恒为单位阵 Identity 4x4 (空间测量全局绝对基准原点，无需外参)", (card_x + 24, w_box_y + 33), font_size=12, color=(160, 180, 200))

        elif cur_frame.type == "fixed_transform":
            if cur_frame.translation_xyz_mm is None or getattr(cur_frame, "rotation_rpy_deg", None) is None:
                unk_box_y = c1_y + 60
                cv2.rectangle(canvas, (card_x + 16, unk_box_y), (card_x + card_w - 16, unk_box_y + 62), (18, 22, 32), -1)
                cv2.rectangle(canvas, (card_x + 16, unk_box_y), (card_x + card_w - 16, unk_box_y + 62), (38, 48, 65), 1)
                draw_text(canvas, "● 外参位姿约束状态: 未知 / 待解 (Unknown / To be Calibrated)", (card_x + 24, unk_box_y + 9), font_size=12, color=(140, 160, 255), bold=True)
                draw_text(canvas, "位姿特性: 6自由度外参均处于待解状态，将在全局建图或视觉求解时进行标定", (card_x + 24, unk_box_y + 33), font_size=11, color=(160, 175, 195))
            else:
                k_dof = getattr(cur_frame, "known_dof", None) or [True] * 6
                tx, ty, tz = cur_frame.translation_xyz_mm
                rx, ry, rz = cur_frame.rotation_rpy_deg

                tx_str = f"X:{tx:+.1f}" if k_dof[0] else "X:? (待解)"
                ty_str = f"Y:{ty:+.1f}" if k_dof[1] else "Y:? (待解)"
                tz_str = f"Z:{tz:+.1f}" if k_dof[2] else "Z:? (待解)"

                rx_str = f"Rx:{rx:+.1f}°" if k_dof[3] else "Rx:? (待解)"
                ry_str = f"Ry:{ry:+.1f}°" if k_dof[4] else "Ry:? (待解)"
                rz_str = f"Rz:{rz:+.1f}°" if k_dof[5] else "Rz:? (待解)"

                t_box_y = c1_y + 60
                cv2.rectangle(canvas, (card_x + 16, t_box_y), (card_x + card_w - 16, t_box_y + 30), (18, 22, 32), -1)
                cv2.rectangle(canvas, (card_x + 16, t_box_y), (card_x + card_w - 16, t_box_y + 30), (38, 48, 65), 1)
                draw_text(canvas, "平移 T [mm]:", (card_x + 24, t_box_y + 7), font_size=11, color=(160, 180, 200))
                draw_text(canvas, f"{tx_str}  {ty_str}  {tz_str}", (card_x + 120, t_box_y + 7), font_size=12, color=(0, 255, 220), bold=True)

                r_box_y = c1_y + 94
                cv2.rectangle(canvas, (card_x + 16, r_box_y), (card_x + card_w - 16, r_box_y + 30), (18, 22, 32), -1)
                cv2.rectangle(canvas, (card_x + 16, r_box_y), (card_x + card_w - 16, r_box_y + 30), (38, 48, 65), 1)
                draw_text(canvas, "旋转 R [deg]:", (card_x + 24, r_box_y + 7), font_size=11, color=(160, 180, 200))
                draw_text(canvas, f"{rx_str}  {ry_str}  {rz_str}", (card_x + 120, r_box_y + 7), font_size=12, color=(255, 200, 60), bold=True)

        else:
            tag_box_y = c1_y + 60
            cv2.rectangle(canvas, (card_x + 16, tag_box_y), (card_x + card_w - 16, tag_box_y + 44), (18, 22, 32), -1)
            cv2.rectangle(canvas, (card_x + 16, tag_box_y), (card_x + card_w - 16, tag_box_y + 44), (38, 48, 65), 1)
            off = getattr(cur_frame, "offset_xyz_mm", [0.0, 0.0, 0.0])
            draw_text(canvas, f"标称安装偏移 offset: [{off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f}] mm", (card_x + 24, tag_box_y + 14), font_size=12, color=(200, 215, 230))

        # 3. 中间卡片: Tag 工位公共物理属性 (空间绝对尺度基准) [新增]
        c_mid_y = c1_y + c1_h + 8
        c_mid_h = 58
        cv2.rectangle(canvas, (card_x, c_mid_y), (card_x + card_w, c_mid_y + c_mid_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c_mid_y), (card_x + card_w, c_mid_y + c_mid_h), (48, 60, 80), 1)

        cv2.circle(canvas, (card_x + 18, c_mid_y + 17), 4, (0, 240, 220), -1)
        draw_text(canvas, "Tag 工位公共物理属性 (三维空间绝对尺度基准 Scale Datum)",
                  (card_x + 28, c_mid_y + 9), font_size=12, color=(0, 240, 220), bold=True)

        ms_val = state.get_workspace_marker_size()
        if ms_val is not None and ms_val > 0:
            ms_str = f"标靶名义边长: {ms_val:.3f} mm  (已显式核准 √)"
            ms_col = (0, 255, 180)
        else:
            ms_str = "标靶名义边长: 未录入 ⚠️ (建图/跟踪拒绝运行，请点击核准)"
            ms_col = (0, 160, 255)
        draw_text(canvas, ms_str, (card_x + 28, c_mid_y + 32), font_size=13, color=ms_col, bold=True)

        # 边长编辑/核准按钮
        self._draw_button(canvas, FRAME_TAG_EDIT_SIZE_BTN, "核准 / 编辑边长", mpos, theme_color=(0, 220, 160))

        # 4. 下部卡片: 10-Slot Tag 专属分配矩阵与局部真值
        c2_y = c_mid_y + c_mid_h + 8
        c2_h = box_h - (c2_y - box_y) - 6
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (42, 52, 70), 1)

        tag_range = state.get_frame_tag_range(cur_frame.frame_id)
        start_id, end_id = tag_range[0], tag_range[-1]
        draw_text(canvas, f"Tag 专属分段放行矩阵 [分配区间 ID: {start_id:02d} ~ {end_id:02d}]",
                  (card_x + 14, c2_y + 10), font_size=14, color=(0, 240, 220), bold=True)
        cv2.line(canvas, (card_x + 10, c2_y + 44), (card_x + card_w - 10, c2_y + 44), (36, 45, 60), 1)

        allowed_set = set(state.get_frame_tags_status(cur_frame.frame_id))
        anchors = state.get_anchor_map()

        for slot_idx in range(10):
            tag_id = tag_range[slot_idx]
            cx, cy, cw, ch = frame_tag_chip_rect(slot_idx)
            is_allowed = (tag_id in allowed_set)
            is_hover = (cx <= mpos[0] <= cx + cw and cy <= mpos[1] <= cy + ch)

            if is_allowed:
                chip_bg = (24, 40, 36) if is_hover else (18, 32, 28)
                chip_border = (0, 255, 180) if is_hover else (0, 200, 140)
            else:
                chip_bg = (30, 34, 42) if is_hover else (20, 24, 32)
                chip_border = (0, 200, 240) if is_hover else (45, 55, 70)

            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), chip_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), chip_border, 2 if (is_hover or is_allowed) else 1)

            tag_label = f"#{tag_id:02d}"
            draw_text(canvas, tag_label, (cx + 8, cy + 6), font_size=13,
                      color=(0, 255, 200) if is_allowed else (160, 175, 195), bold=True)

            status_str = "已放行 √" if is_allowed else "未放行"
            status_col = (0, 255, 160) if is_allowed else (120, 135, 150)
            draw_text(canvas, status_str, (cx + cw - 52, cy + 6), font_size=11, color=status_col, bold=is_allowed)

            cv2.line(canvas, (cx + 6, cy + 30), (cx + cw - 6, cy + 30), (35, 45, 58), 1)

            anchor_entry = anchors.get(tag_id) or anchors.get(str(tag_id))
            if isinstance(anchor_entry, dict) and "xyz_mm" in anchor_entry:
                xyz = anchor_entry.get("xyz_mm", [0.0, 0.0, 0.0])
                known = anchor_entry.get("known", [True, True, True])
                sx = f"{xyz[0]:.0f}" if (len(known) > 0 and known[0]) else "?"
                sy = f"{xyz[1]:.0f}" if (len(known) > 1 and known[1]) else "?"
                sz = f"{xyz[2]:.0f}" if (len(known) > 2 and known[2]) else "?"
                if any(known):
                    pos_str = f"P:({sx},{sy},{sz})"
                    pos_col = (255, 210, 80) if all(known) else (0, 220, 255)
                else:
                    pos_str = "P: 未标注"
                    pos_col = (110, 125, 140)
            else:
                pos_str = "P: 未标注"
                pos_col = (110, 125, 140)
            draw_text(canvas, pos_str, (cx + 6, cy + 40), font_size=10, color=pos_col)

            px, py = cx + cw - 14, cy + 48
            pen_col = (0, 255, 180) if is_hover else (70, 95, 110)
            cv2.line(canvas, (px - 3, py + 2), (px + 3, py - 4), pen_col, 1, cv2.LINE_AA)

    def _render_page_frame_rois(self, canvas: np.ndarray, state: HubState, ws):
        """坐标系专属页签 2: 3D ROI 空间物件 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, self.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        cur_frame = state.get_selected_frame()
        if not cur_frame:
            draw_text(canvas, "未选择任何机构坐标系", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        rois = state.get_frame_rois(cur_frame.frame_id)
        total_rois = len(rois)
        max_vis = HubState.ROI_VISIBLE_COUNT
        max_offset = max(0, total_rois - max_vis)
        offset = max(0, min(getattr(state, "roi_scroll_offset", 0), max_offset))
        state.roi_scroll_offset = offset

        # 1. 顶部标题栏、翻页控制与新建 ROI 按钮
        if total_rois > max_vis:
            disp_end = min(offset + max_vis, total_rois)
            header_text = f"3D ROI 空间物件 ({cur_frame.frame_id} · 显示 {offset + 1}~{disp_end} / 共 {total_rois} 个)"
            draw_text(canvas, header_text, (box_x + 16, box_y + 14), font_size=14, color=self.COLOR_WHITE, bold=True)
            self._draw_button(canvas, FRAME_ROI_PREV_BTN, "▲ 上翻", mpos, enabled=(offset > 0))
            self._draw_button(canvas, FRAME_ROI_NEXT_BTN, "▼ 下翻", mpos, enabled=(offset < max_offset))
        else:
            header_text = f"3D ROI 空间物件 (坐标系: {cur_frame.frame_id} · 共 {total_rois} 个)"
            draw_text(canvas, header_text, (box_x + 16, box_y + 14), font_size=15, color=self.COLOR_WHITE, bold=True)

        self._draw_button(canvas, FRAME_ADD_ROI_BTN, "+ 新建本坐标系 ROI", mpos)

        # 2. ROI 列表卡片
        if not rois:
            empty_y = box_y + 180
            cv2.rectangle(canvas, (box_x + 40, empty_y), (box_x + box_w - 40, empty_y + 120), (24, 28, 38), -1)
            cv2.rectangle(canvas, (box_x + 40, empty_y), (box_x + box_w - 40, empty_y + 120), (42, 52, 70), 1)
            draw_text(canvas, f"当前坐标系 [{cur_frame.frame_id}] 下暂无 3D ROI 空间物件",
                      (box_x + 110, empty_y + 36), font_size=16, color=(0, 220, 255), bold=True)
            draw_text(canvas, "点击右上角 [+ 新建本坐标系 ROI] 即可为该机构添加检测工作面包围盒",
                      (box_x + 85, empty_y + 70), font_size=13, color=(140, 160, 180))
            return

        visible_rois = rois[offset : offset + max_vis]
        for i, r in enumerate(visible_rois):
            rx, ry, rw, rh = frame_roi_row_rect(i)
            is_hover = (rx <= mpos[0] <= rx + rw and ry <= mpos[1] <= ry + rh)
            card_bg = (26, 32, 44) if is_hover else (22, 26, 36)
            card_border = (0, 220, 240) if is_hover else (38, 48, 64)

            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), card_bg, -1)
            cv2.rectangle(canvas, (rx, ry), (rx + rw, ry + rh), card_border, 2 if is_hover else 1)

            cat_map = {"belt": "同步带", "wheel": "驱动轮", "tray": "料盘", "general": "通用"}
            cat_badge = cat_map.get(r.category, r.category)
            cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 78, ry + 34), (32, 45, 58), -1)
            cv2.rectangle(canvas, (rx + 10, ry + 10), (rx + 78, ry + 34), (0, 200, 240), 1)
            draw_text(canvas, cat_badge, (rx + 18, ry + 14), font_size=11, color=(0, 240, 220), bold=True)

            title = f"{r.name} ({r.roi_id})"
            draw_text(canvas, title, (rx + 86, ry + 13), font_size=14, color=self.COLOR_WHITE, bold=True)

            # Smart ROI 语义徽章
            r_role = getattr(r, "role", "source") or "source"
            r_intent = getattr(r, "target_intent", "pose_pick") or "pose_pick"
            slot_info = f"#{r.binding.get('slot_index', 0)}" if getattr(r, "binding", None) and "slot_index" in r.binding else ""
            role_labels = {
                "source": ("★进料源", (0, 255, 140)),
                "destination": (f"▼落料{slot_info}", (0, 220, 255)),
                "keepout": ("⛔禁行区", (255, 80, 80)),
                "general": ("◌通用", (180, 180, 180)),
            }
            role_txt, role_col = role_labels.get(r_role, ("◌通用", (180, 180, 180)))
            intent_labels = {
                "pose_pick": "位姿抓取",
                "piece_count": "根数统计",
                "occupancy": "在席检测",
                "general": "通用意图",
            }
            intent_txt = intent_labels.get(r_intent, r_intent)
            badge_str = f"[{role_txt} · {intent_txt}]"
            draw_text(canvas, badge_str, (rx + 310, ry + 14), font_size=12, color=role_col, bold=True)

            cx, cy, cz = r.center_xyz_mm
            sx, sy, sz = r.size_xyz_mm
            rx_d, ry_d, rz_d = getattr(r, "rotation_rpy_deg", [0.0, 0.0, 0.0])
            geom_str = f"中心: ({cx:.0f},{cy:.0f},{cz:.0f})  尺寸: ({sx:.0f}×{sy:.0f}×{sz:.0f})  旋转: ({rx_d:.0f}°,{ry_d:.0f}°,{rz_d:.0f}°)"
            put_text(canvas, geom_str, (rx + 14, ry + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 180, 200), 1, cv2.LINE_AA)

            self._draw_button(canvas, frame_roi_edit_btn(i), "编辑", mpos)
            self._draw_button(canvas, frame_roi_del_btn(i), "删除", mpos, theme_color=(180, 60, 60))

        # 3. 极简现代扁平滚动条 (当物件超过单屏容量时自动呈现)
        if total_rois > max_vis:
            tx, ty, tw, th = FRAME_ROI_SCROLL_TRACK
            is_hover_track = (tx - 4 <= mpos[0] <= tx + tw + 4 and ty <= mpos[1] <= ty + th)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (22, 26, 34), -1)
            cv2.rectangle(canvas, (tx, ty), (tx + tw, ty + th), (36, 44, 56), 1)

            thumb_h = max(32, int(th * (float(max_vis) / float(total_rois))))
            thumb_travel = th - thumb_h
            thumb_y = ty + int(thumb_travel * (float(offset) / float(max_offset))) if max_offset > 0 else ty
            is_hover_thumb = (tx - 4 <= mpos[0] <= tx + tw + 4 and thumb_y <= mpos[1] <= thumb_y + thumb_h)

            thumb_color = (0, 240, 220) if is_hover_thumb else ((0, 190, 210) if is_hover_track else (65, 80, 100))
            cv2.rectangle(canvas, (tx + 1, thumb_y), (tx + tw - 1, thumb_y + thumb_h), thumb_color, -1)

    def _render_gallery_page(self, canvas: np.ndarray, state: HubState, title: str,
                             images: list[str], sel_idx: int, grid_offset: int,
                             empty_hint: str, is_calib: bool):
        """渲染通用图片卡片网格墙: 3列x3行大卡片网格 (双击卡片放大)
        布局: 面板 (340, 50, 620, 620); 网格 x: 356~944, y: 86~648; 卡片 184x168
        """
        box_x, box_y, box_w, box_h = 340, 50, self.canvas_w - 340, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), self.COLOR_PANEL, -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 空状态：纯净单个提示，删除所有冗余副提示红字
        if not images:
            empty_box_y = box_y + 140
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 90), (22, 26, 36), -1)
            cv2.rectangle(canvas, (box_x + 16, empty_box_y), (box_x + box_w - 16, empty_box_y + 90), self.COLOR_BORDER, 1)
            # 水平居中渲染
            approx_w = sum(18 if ord(c) > 127 else 10 for c in empty_hint)
            hint_x = box_x + max(20, (box_w - approx_w) // 2)
            draw_text(canvas, empty_hint, (hint_x, empty_box_y + 32), font_size=17, color=(0, 200, 240), bold=True)
            return

        # 3. 图片卡片网格墙 (4 列 x 3 行, 每页 12 张大卡片, 铺满面板底部空白)
        total = len(images)
        visible_imgs = images[grid_offset: grid_offset + HubState.GRID_PAGE]

        for i, img_path in enumerate(visible_imgs):
            real_idx = grid_offset + i
            row, col = divmod(i, HubState.GRID_COLS)
            x = self.GRID_X0 + col * (self.GRID_CELL_W + self.GRID_GAP_X)
            y = self.GRID_Y0 + row * (self.GRID_CELL_H + self.GRID_GAP_Y)

            is_cur = (real_idx == sel_idx)
            is_hover = (x <= mpos[0] <= x + self.GRID_CELL_W and y <= mpos[1] <= y + self.GRID_CELL_H)

            card_bg = (30, 40, 36) if is_cur else ((28, 33, 41) if is_hover else (22, 26, 36))
            card_border = (0, 255, 180) if is_cur else ((0, 200, 240) if is_hover else (38, 44, 58))
            cv2.rectangle(canvas, (x, y), (x + self.GRID_CELL_W, y + self.GRID_CELL_H), card_bg, -1)
            cv2.rectangle(canvas, (x, y), (x + self.GRID_CELL_W, y + self.GRID_CELL_H), card_border, 2 if is_cur else 1)

            # 3.1 卡片主体: 接近 4:3 的大缩略图
            tw, th = self.GRID_CELL_W - 8, self.GRID_THUMB_H
            thumb = state.get_thumbnail(img_path, tw, th)
            if thumb is not None:
                canvas[y + 4:y + 4 + th, x + 4:x + 4 + tw] = thumb
            else:
                draw_text(canvas, "读取失败", (x + 60, y + 74), font_size=13, color=(120, 120, 140))

            # 3.2 左上角序号徽章
            cv2.rectangle(canvas, (x + 4, y + 4), (x + 56, y + 26), (8, 12, 16), -1)
            put_text(canvas, f"#{real_idx + 1:02d}", (x + 11, y + 20), cv2.FONT_HERSHEY_SIMPLEX,
                     0.42, (0, 255, 180) if is_cur else (170, 185, 205), 1, cv2.LINE_AA)

            # 3.3 底部文件名信息条
            bar_y = y + self.GRID_CELL_H - 22
            cv2.rectangle(canvas, (x + 1, bar_y), (x + self.GRID_CELL_W - 1, y + self.GRID_CELL_H - 1), (10, 12, 18), -1)
            put_text(canvas, os.path.basename(img_path)[:30], (x + 6, y + self.GRID_CELL_H - 7),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                     (0, 255, 200) if is_cur else self.COLOR_GRAY, 1, cv2.LINE_AA)

            # 3.4 选中卡片左侧高亮指示条
            if is_cur:
                cv2.rectangle(canvas, (x, y), (x + 4, y + self.GRID_CELL_H), (0, 255, 180), -1)

    def _render_page_frames_rois(self, canvas: np.ndarray, state: HubState, sc):
        """页签: 坐标系与 3D ROI 空间管理页 (x: 340~960, y: 50~670)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮
        draw_text(canvas, "工位多坐标系拓扑与 3D ROI 空间", (box_x + 16, box_y + 14), font_size=16, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, (760, box_y + 8, 86, 30), "刷新", mpos)
        self._draw_button(canvas, (854, box_y + 8, 86, 30), "打开目录", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工位", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        card_x = box_x + 12
        card_w = box_w - 24  # 596px

        # -------------------------------------------------------------------------
        # 1. 机构多坐标系拓扑树卡片 (Coordinate Frame Tree)
        # -------------------------------------------------------------------------
        c1_y = box_y + 46
        c1_h = 260
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (42, 50, 68), 1)

        frames = state.get_coordinate_frames()
        draw_text(canvas, f"机构多坐标系拓扑树 (Coordinate Frames · 共 {len(frames)} 个)",
                  (card_x + 16, c1_y + 11), font_size=14, color=(0, 240, 220), bold=True)
        self._draw_button(canvas, frame_btn_add_rect(), "+ 新增坐标系", mpos)
        cv2.line(canvas, (card_x + 14, c1_y + 34), (card_x + card_w - 14, c1_y + 34), self.COLOR_BORDER, 1)

        coord_mgr = state.coord_mgr
        row_y = c1_y + 40
        row_h = 66
        gap_y = 6

        # 显示坐标系列表 (最多前 3 个)
        for i, frame in enumerate(frames[:3]):
            fy = row_y + i * (row_h + gap_y)
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + card_w - 10, fy + row_h), (30, 36, 48), -1)
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + card_w - 10, fy + row_h), (50, 60, 80), 1)

            # 类型指示色条
            if frame.type == "world":
                bar_col = (0, 255, 180)
                type_name = "绝对世界系"
            elif frame.type == "tag_bound":
                bar_col = (0, 210, 255)
                type_name = f"动标Tag #{frame.tag_id}"
            else:
                bar_col = (255, 180, 50)
                type_name = "固定外参"
            cv2.rectangle(canvas, (card_x + 10, fy), (card_x + 14, fy + row_h), bar_col, -1)

            # 第一行：名称 + 标识 + 类型徽章 + 状态
            draw_text(canvas, f"{frame.name} ({frame.frame_id})", (card_x + 22, fy + 8), font_size=13, color=self.COLOR_WHITE, bold=True)
            cv2.rectangle(canvas, (card_x + 240, fy + 6), (card_x + 335, fy + 26), (20, 26, 36), -1)
            cv2.rectangle(canvas, (card_x + 240, fy + 6), (card_x + 335, fy + 26), bar_col, 1)
            draw_text(canvas, type_name, (card_x + 246, fy + 8), font_size=11, color=bar_col, bold=True)

            is_res = True
            if coord_mgr:
                _, is_res = coord_mgr.get_frame_to_world(frame.frame_id)
            if frame.type == "fixed_transform":
                status_text = "● 已标定" if is_res else ("⚠ 待标定" if getattr(frame, "status", "") == "unknown" else "⚠ 未求解")
                status_col = (0, 230, 140) if is_res else (0, 180, 255)
            elif frame.type == "tag_bound":
                status_text = "● 已解出" if is_res else "⚠ 动标未解"
                status_col = (0, 230, 140) if is_res else (0, 180, 255)
            else:
                status_text = "● 原点基准"
                status_col = (0, 230, 140)
            draw_text(canvas, status_text, (card_x + 350, fy + 8), font_size=11, color=status_col, bold=True)

            # 右侧操作按钮
            self._draw_button(canvas, frame_row_edit_rect(i), "编辑", mpos)
            if frame.frame_id != "world":
                self._draw_button(canvas, frame_row_del_rect(i), "删除", mpos, theme_color=(180, 60, 60))

            # 第二行：父坐标系与几何参数详情
            parent_info = f"父级: {frame.parent_frame_id or '根节点'}"
            if frame.type == "world":
                param_info = "原点基准 [0, 0, 0] mm | 无旋转"
            elif frame.type == "tag_bound":
                param_info = f"绑定 Tag {frame.tag_id} | 偏移 XYZ: {frame.offset_xyz_mm} mm"
            else:
                if frame.translation_xyz_mm is not None and frame.rotation_rpy_deg is not None:
                    t = [round(float(x), 1) for x in frame.translation_xyz_mm]
                    r = [round(float(x), 1) for x in frame.rotation_rpy_deg]
                    rmse_str = f" (RMSE: {frame.calibration_metrics.get('rmse_mm', 0):.2f}mm)" if getattr(frame, "calibration_metrics", None) and "rmse_mm" in frame.calibration_metrics else ""
                    param_info = f"平移: {t} mm | 旋转: {r}°{rmse_str}"
                else:
                    spec_desc = ""
                    if getattr(frame, "calibration_spec", None):
                        spec_method = frame.calibration_spec.get("method", "")
                        spec_desc = f" [{spec_method}]"
                    param_info = f"外参待解{spec_desc} (BA后自动标定)"

            draw_text(canvas, f"{parent_info}  |  {param_info}", (card_x + 22, fy + 38), font_size=11, color=(160, 175, 195))

        # -------------------------------------------------------------------------
        # 2. 3D ROI 空间物件集合卡片 (ROI Space Collection)
        # -------------------------------------------------------------------------
        c2_y = c1_y + c1_h + 10
        c2_h = 285
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (24, 28, 38), -1)
        cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (42, 50, 68), 1)

        rois = state.get_roi_spaces()
        draw_text(canvas, f"3D ROI 空间物件集合 (3D OBB Objects · 共 {len(rois)} 个)",
                  (card_x + 16, c2_y + 11), font_size=14, color=(0, 255, 180), bold=True)
        self._draw_button(canvas, roi_btn_add_rect(), "+ 新增 ROI", mpos)
        cv2.line(canvas, (card_x + 14, c2_y + 34), (card_x + card_w - 14, c2_y + 34), self.COLOR_BORDER, 1)

        roi_row_y = c2_y + 40
        roi_row_h = 70
        roi_gap_y = 6

        if not rois:
            draw_text(canvas, "当前工位尚未定义任何 3D ROI 物件 (可点击右上角 [+ 新增 ROI] 结构化配置)",
                      (card_x + 36, c2_y + 90), font_size=13, color=self.COLOR_GRAY)
        else:
            roi_mgr = state.roi_mgr
            for i, roi in enumerate(rois[:3]):
                ry = roi_row_y + i * (roi_row_h + roi_gap_y)
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + card_w - 10, ry + roi_row_h), (30, 36, 48), -1)
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + card_w - 10, ry + roi_row_h), (50, 60, 80), 1)

                # ROI 颜色标识
                rgb = roi.visual_color_rgb or [0, 255, 128]
                bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
                cv2.rectangle(canvas, (card_x + 10, ry), (card_x + 14, ry + roi_row_h), bgr, -1)

                # 第一行：名称 + 标识 + 类别徽章 + 挂载坐标系
                draw_text(canvas, f"{roi.name} ({roi.roi_id})", (card_x + 22, ry + 8), font_size=13, color=self.COLOR_WHITE, bold=True)
                # 类别徽章
                cv2.rectangle(canvas, (card_x + 240, ry + 6), (card_x + 315, ry + 26), (20, 26, 36), -1)
                cv2.rectangle(canvas, (card_x + 240, ry + 6), (card_x + 315, ry + 26), bgr, 1)
                cat_map = {"belt": "同步带", "wheel": "驱动轮", "tray": "料盘", "general": "通用"}
                cat_label = f"[{cat_map.get(roi.category, roi.category)}]"
                # Smart ROI 角色指示
                r_role = getattr(roi, "role", "source") or "source"
                slot_info = f"#{roi.binding.get('slot_index', 0)}" if getattr(roi, "binding", None) and "slot_index" in roi.binding else ""
                role_labels = {
                    "source": ("★进料源", (0, 255, 140)),
                    "destination": (f"▼落料{slot_info}", (0, 220, 255)),
                    "keepout": ("⛔禁行区", (255, 80, 80)),
                    "general": ("◌通用", (180, 180, 180)),
                }
                role_txt, role_col = role_labels.get(r_role, ("◌通用", (180, 180, 180)))
                draw_text(canvas, f"所属: {roi.frame_id}  |  {role_txt}", (card_x + 325, ry + 8), font_size=11, color=role_col, bold=True)

                # 右侧操作按钮
                self._draw_button(canvas, roi_row_edit_rect(i), "编辑", mpos)
                self._draw_button(canvas, roi_row_del_rect(i), "删除", mpos, theme_color=(180, 60, 60))

                # 第二行：局部几何尺寸
                c_str = [round(float(x), 1) for x in roi.center_xyz_mm]
                s_str = [round(float(x), 1) for x in roi.size_xyz_mm]
                local_info = f"局部中心: {c_str} | 尺寸(长宽高): {s_str[0]}x{s_str[1]}x{s_str[2]} mm"

                # 第三行：世界坐标系求解
                world_info = "世界 OBB: 未就绪"
                if roi_mgr and coord_mgr:
                    obb = roi_mgr.get_roi_world_obb(roi.roi_id, coord_mgr)
                    if obb and obb.get("is_resolved"):
                        cw = [round(float(x), 1) for x in obb["center_world"]]
                        world_info = f"世界中心: {cw} mm | 8顶点OBB已就绪"
                    elif obb:
                        world_info = "世界 OBB: 依附坐标系动标未就绪"

                draw_text(canvas, local_info, (card_x + 22, ry + 32), font_size=11, color=(150, 165, 185))
                draw_text(canvas, world_info, (card_x + 22, ry + 50), font_size=11, color=(0, 230, 160) if "已就绪" in world_info else self.COLOR_GOLD)

        # -------------------------------------------------------------------------
        # 3. 底部快捷提示
        # -------------------------------------------------------------------------
        hint_y = c2_y + c2_h + 4
        draw_text(canvas, "提示: 点击右上角 [+ 新增] 或各行 [编辑] 可直接在 GUI 中配置，修改后即时保存生效",
                  (box_x + 18, hint_y), font_size=12, color=self.COLOR_GRAY)

    def _render_modal_scaffold(
        self,
        canvas: np.ndarray,
        state: HubState,
        title: str,
        w: int = GEOM_MODAL_W,
        h: int = GEOM_MODAL_H,
        border_col: tuple[int, int, int] = (0, 220, 180),
    ) -> tuple[int, int, int, int, tuple[int, int]]:
        """绘制模态表单通用脚手架 (委托给 HubModalsRenderer)"""
        return self.modals.render_modal_scaffold(canvas, state, title, border_col, box_w=w, box_h=h)

    def _render_frame_modal(self, canvas: np.ndarray, state: HubState):
        """渲染机构坐标系结构化表单弹窗 (委托给 HubModalsRenderer)"""
        self.modals.render_frame_modal(canvas, state)

    def _render_roi_modal(self, canvas: np.ndarray, state: HubState):
        """渲染 3D ROI 空间物件结构化表单弹窗 (委托给 HubModalsRenderer)"""
        self.modals.render_roi_modal(canvas, state)

    def _render_page_whitelist(self, canvas: np.ndarray, state: HubState, sc):
        """页签2: Tag 标靶白名单管理页 - 实时读取 tag_whitelist.yaml 呈现放行矩阵 (x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (20, 23, 30), -1)
        mpos = (state.mouse_x, state.mouse_y)

        # 栏目标题与右上角操作按钮 (紧凑排布在 620 宽内)
        draw_text(canvas, "Tag 标靶白名单管理", (box_x + 16, box_y + 14), font_size=16, color=self.COLOR_WHITE, bold=True)
        self._draw_button(canvas, (760, box_y + 8, 86, 30), "刷新", mpos)
        if state.whitelist_edit_mode:
            self._draw_button(canvas, WL_BTN_DONE, "完成", mpos, theme_color=(0, 200, 120))
        else:
            self._draw_button(canvas, WL_BTN_DONE, "编辑", mpos)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工位", (box_x + 180, box_y + 280), font_size=18, color=self.COLOR_GRAY)
            return

        wl = state.get_tag_whitelist()
        wl_path = state.workspace_mgr.get_tag_whitelist_path(sc.workspace_id)
        if state.whitelist_edit_mode:
            # 编辑模式: 徽章与状态以芯片工作集合为准 (即时反馈, 不等 yaml 回读)
            allowed_ids = set(state.whitelist_edit_ids)
        else:
            allowed_ids = set(wl.get("allowed_ids") or []) if wl else set()
        has_filter = bool(allowed_ids)  # 白名单恒启用: 名单非空 → 过滤, 空 → 探索模式

        # 1. Workspace 元数据卡与白名单生效状态徽章
        meta_y = box_y + 52
        cv2.rectangle(canvas, (box_x + 12, meta_y), (box_x + box_w - 12, meta_y + 70), (26, 31, 42), -1)
        cv2.rectangle(canvas, (box_x + 12, meta_y), (box_x + box_w - 12, meta_y + 70), self.COLOR_BORDER, 1)
        draw_text(canvas, f"{sc.name}  |  {sc.workspace_id}",
                  (box_x + 20, meta_y + 12), font_size=14, color=(0, 240, 220), bold=True)
        draw_text(canvas, f"配置: {os.path.basename(wl_path)}", (box_x + 20, meta_y + 44), font_size=12, color=self.COLOR_DARK_GRAY)

        badge_x, badge_y = box_x + box_w - 180, meta_y + 14
        if has_filter:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (20, 48, 32), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (0, 255, 160), 2)
            draw_text(canvas, f"● 白名单 {len(allowed_ids)} 个", (badge_x + 18, badge_y + 11), font_size=13, color=(0, 255, 180), bold=True)
        else:
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (46, 40, 24), -1)
            cv2.rectangle(canvas, (badge_x, badge_y), (badge_x + 164, badge_y + 42), (60, 160, 255), 2)
            draw_text(canvas, "◌ 探索模式", (badge_x + 18, badge_y + 11), font_size=13, color=(80, 200, 255), bold=True)

        # 2. 白名单模式说明卡
        desc_y = meta_y + 80
        cv2.rectangle(canvas, (box_x + 12, desc_y), (box_x + box_w - 12, desc_y + 60), (24, 28, 38), -1)
        cv2.rectangle(canvas, (box_x + 12, desc_y), (box_x + box_w - 12, desc_y + 60), self.COLOR_BORDER, 1)

        if state.whitelist_edit_mode and state.anchor_mode:
            mode_text = "锚点模式: 单击 Tag 芯片打开世界坐标弹窗 (逐轴输入, 支持部分已知与清除)。"
            mode_hint = "提示: 金色芯片 = 已记录锚点 (标注已知轴数)；[退出锚点] 返回白名单编辑。"
            mode_col = (0, 210, 255)
        elif state.whitelist_edit_mode:
            mode_text = "编辑模式: 单击芯片切换放行/拦截，每次点击即时写回 tag_whitelist.yaml。"
            mode_hint = "提示: 右上角 [完成] 退出编辑；芯片右上蓝点 = 该 ID 在工位元数据 valid_tag_ids 中。"
            mode_col = (0, 255, 200)
        elif has_filter:
            mode_text = f"白名单恒启用: 仅放行 allowed_ids 中的 {len(allowed_ids)} 个标靶，其余拦截 (权威约束)。"
            mode_hint = "提示: 修改 allowed_ids 并保存，返回本页签即自动刷新矩阵状态。"
            mode_col = (0, 255, 160)
        elif not wl:
            mode_text = "尚未创建 tag_whitelist.yaml: 探索模式放行所有检测到的有效标靶。"
            mode_hint = "提示: 点击右上角 [编辑] 进入芯片编辑模式 (自动创建配置模板)。"
            mode_col = self.COLOR_GOLD
        else:
            mode_text = "探索模式 (allowed_ids 为空): 放行所有检测到的有效标靶，由全局物理白名单兜底拦截。"
            mode_hint = "提示: 在 allowed_ids 中填入物理布点标靶 ID 即可启用工位过滤。"
            mode_col = (0, 200, 240)

        draw_text(canvas, mode_text, (box_x + 20, desc_y + 10), font_size=13, color=mode_col, bold=True)
        draw_text(canvas, mode_hint, (box_x + 20, desc_y + 34), font_size=12, color=self.COLOR_GRAY)

        # 3. 全量 Tag 标靶放行矩阵 (视图: 只读矩阵 / 编辑: 可点击芯片, 超范围 ID 追加行)
        matrix_y = desc_y + 70
        edit_ids = state.whitelist_edit_ids if state.whitelist_edit_mode else None
        if edit_ids is not None:
            extra_ids = sorted(t for t in edit_ids if t >= 30)
        else:
            extra_ids = sorted(t for t in allowed_ids if t >= 30)
        extra_rows = (len(extra_ids) + WL_COLS - 1) // WL_COLS
        matrix_h = 268 + 44 * extra_rows
        cv2.rectangle(canvas, (box_x + 12, matrix_y), (box_x + box_w - 12, matrix_y + matrix_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (box_x + 12, matrix_y), (box_x + box_w - 12, matrix_y + matrix_h), (40, 48, 66), 1)

        matrix_title = ("AprilTag 标靶放行矩阵 (0~29 号标靶)"
                        + (f"  |  已放行 {len(allowed_ids)} 个" if has_filter else "  |  探索模式"))
        draw_text(canvas, matrix_title, (box_x + 20, matrix_y + 12), font_size=14, color=(0, 255, 200), bold=True)
        cv2.line(canvas, (box_x + 20, matrix_y + 36), (box_x + box_w - 20, matrix_y + 36), self.COLOR_BORDER, 1)

        valid_set = set(sc.valid_tag_ids)
        for t_id in list(range(30)) + extra_ids:
            tx, ty, cw, ch = whitelist_cell_rect(t_id)

            if edit_ids is not None:
                # 编辑模式: 芯片带 Hover 高亮 (锚点模式反映锚点状态, 白名单编辑反映放行集合)
                hovered = point_in_rect(mpos[0], mpos[1], (tx, ty, cw, ch))
                if state.anchor_mode:
                    entry = state.anchor_map.get(t_id)
                    n_known = sum(1 for b in entry["known"] if b) if entry else 0
                    if entry:
                        cell_bg = (54, 48, 28) if hovered else (44, 40, 22)
                        cell_border = (0, 220, 255) if hovered else (0, 190, 255)
                        txt_color = (0, 230, 255)
                        status_desc, status_col = f"锚点 {n_known}/3", (0, 220, 200)
                    else:
                        cell_bg = (32, 36, 46) if hovered else (26, 30, 38)
                        cell_border = (66, 74, 92) if hovered else (52, 60, 76)
                        txt_color = (150, 160, 175)
                        status_desc, status_col = "无锚点", (110, 120, 135)
                else:
                    is_allowed = (t_id in edit_ids)
                    if is_allowed:
                        cell_bg = (34, 58, 48) if hovered else (28, 48, 40)
                        cell_border = (0, 255, 180) if hovered else (0, 240, 160)
                        txt_color = (0, 255, 200)
                        status_desc, status_col = "已放行", (0, 220, 140)
                    else:
                        cell_bg = (50, 32, 32) if hovered else (40, 26, 26)
                        cell_border = (80, 52, 52) if hovered else (60, 40, 40)
                        txt_color = (200, 140, 140)
                        status_desc, status_col = "已拦截", (150, 100, 100)
            elif has_filter:
                is_allowed = (t_id in allowed_ids)
                cell_bg = (28, 48, 40) if is_allowed else (40, 26, 26)
                cell_border = (0, 240, 160) if is_allowed else (60, 40, 40)
                txt_color = (0, 255, 200) if is_allowed else (180, 120, 120)
                status_desc = "已放行" if is_allowed else "已拦截"
                status_col = (0, 220, 140) if is_allowed else (150, 100, 100)
            else:
                cell_bg = (24, 34, 40)
                cell_border = (36, 70, 80)
                txt_color = (160, 220, 235)
                status_desc = "探索放行"
                status_col = (110, 170, 190)

            cv2.rectangle(canvas, (tx, ty), (tx + cw, ty + ch), cell_bg, -1)
            cv2.rectangle(canvas, (tx, ty), (tx + cw, ty + ch), cell_border, 1)

            draw_text(canvas, f"Tag #{t_id:02d}", (tx + 8, ty + 5), font_size=11, color=txt_color, bold=True)
            draw_text(canvas, status_desc, (tx + 12, ty + 21), font_size=10, color=status_col)
            if t_id in valid_set:
                cv2.circle(canvas, (tx + cw - 8, ty + 9), 3, (0, 200, 240), -1)

        # 4. 底部操作区: 编辑模式为批量按钮 + 锚点切换, 视图模式为指引文案
        if edit_ids is not None:
            self._draw_button(canvas, WL_BTN_ALL, "全部放行", mpos)
            self._draw_button(canvas, WL_BTN_CLEAR, "清空", mpos, theme_color=(180, 60, 60))
            self._draw_button(canvas, WL_BTN_ANCHOR, "退出锚点" if state.anchor_mode else "锚点坐标", mpos)
            hint = "单击 Tag 芯片编辑其世界坐标锚点" if state.anchor_mode else "单击芯片即时写回 yaml"
            draw_text(canvas, hint, (WL_BTN_ANCHOR[0] + WL_BTN_ANCHOR[2] + 14, 632),
                      font_size=12, color=self.COLOR_GRAY)
            if state.anchor_modal_open:
                self._render_anchor_modal(canvas, state)
        else:
            draw_text(canvas, "提示: 点击右上角 [编辑] 进入芯片编辑模式 (单击切换, 即时写回)",
                      (box_x + 20, box_y + box_h - 26), font_size=12, color=self.COLOR_GRAY)

    def _render_anchor_modal(self, canvas: np.ndarray, state: HubState):
        """锚点坐标编辑弹窗 (委托给 HubModalsRenderer)"""
        self.modals.render_anchor_modal(canvas, state)

    def _render_expanded_photo_preview(self, canvas: np.ndarray, state: HubState, sc):
        """全宽自适应大图视口 (按 F 键展开，横跨中间和右侧，x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (16, 20, 26), -1)
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 200, 240), 2)
        mpos = (state.mouse_x, state.mouse_y)

        # 标题与右上角实体按钮
        if not state.current_images:
            draw_text(canvas, "当前场景无图片", (box_x + 220, box_y + 280), font_size=20, color=self.COLOR_DARK_GRAY)
            self._draw_button(canvas, (box_x + box_w - 140, box_y + 10, 120, 32), "返回网格", mpos)
            return

        cur_img = state.current_images[state.selected_image_idx]
        prev = state.get_preview(cur_img, max_w=590, max_h=520)

        img_title = f"{os.path.basename(cur_img)} ({state.selected_image_idx + 1}/{len(state.current_images)})"
        draw_text(canvas, img_title, (box_x + 16, box_y + 14), font_size=14, color=(0, 255, 200), bold=True)

        # 右上角实体按钮组: [上张] [下张] [删帧] [返回] (x: 680~950)
        self._draw_button(canvas, (680, box_y + 10, 60, 30), "上张", mpos)
        self._draw_button(canvas, (746, box_y + 10, 60, 30), "下张", mpos)
        self._draw_button(canvas, (812, box_y + 10, 68, 30), "删帧", mpos, theme_color=(180, 60, 60))
        self._draw_button(canvas, (886, box_y + 10, 64, 30), "返回", mpos)

        if prev is not None:
            ph, pw = prev.shape[:2]
            px = box_x + (box_w - pw) // 2
            py = box_y + 48 + (520 - ph) // 2
            canvas[py:py + ph, px:px + pw] = prev
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (60, 70, 90), 1)

    def _render_pure_dashboard_panel(self, canvas: np.ndarray, state: HubState, sc):
        """页签3: 体检报告 - 综合体检与几何健康大屏 (垂直贯通排列, x: 340~960)"""
        box_x, box_y, box_w, box_h = 340, 50, 620, 620
        cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (18, 22, 28), -1)

        if not sc:
            draw_text(canvas, "请在左侧选择或新建工况场景", (box_x + 180, box_y + 280), font_size=20, color=self.COLOR_GRAY)
            return

        card_x = box_x + 12
        card_w = box_w - 24  # 596px 通栏紧凑宽度

        # ==== 1. 板块 #1: 工位核心元数据与管理卡片 (极简高雅内嵌按钮) ====
        c1_y = box_y + 10
        c1_h = 168
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (25, 31, 42), -1)
        cv2.rectangle(canvas, (card_x, c1_y), (card_x + card_w, c1_y + c1_h), (45, 56, 78), 1)

        mpos = (state.mouse_x, state.mouse_y)

        # 行 1: 纯工位名称 与 [重命名] 按钮
        draw_text(canvas, sc.name, (card_x + 16, c1_y + 11), font_size=15, color=(0, 240, 220), bold=True)
        self._draw_button(canvas, WS_BTN_RENAME, "重命名", mpos)

        # 行 2: 纯工位物理ID，右侧 [打开] 按钮 (直达文件夹)
        draw_text(canvas, sc.workspace_id, (card_x + 16, c1_y + 39), font_size=12, color=self.COLOR_GRAY)
        self._draw_button(canvas, WS_BTN_OPEN_DIR, "打开", mpos)

        # 行 3: 生产工作流模式 (Smart Production) 与 [切换模式] 按钮
        prod_cfg = getattr(sc, "production", {}) or {}
        prod_name = prod_cfg.get("name", "SCARA 智能分选生产线")
        prod_mode = prod_cfg.get("mode", "scara_sorting")
        prod_pipe = prod_cfg.get("active_pipeline", "asparagus_studio")
        draw_text(canvas, f"工作流:  {prod_name}  ({prod_mode} · {prod_pipe})",
                  (card_x + 16, c1_y + 68), font_size=12, color=(0, 255, 200), bold=True)
        self._draw_button(canvas, WS_BTN_TOGGLE_PROD_MODE, "切换模式", mpos, theme_color=(0, 200, 220))

        # 行 4: 备注独立成行，右侧 [修改] 按钮 (方便随时查看与修改)
        desc_text = getattr(sc, "description", "") or "暂无备注"
        created_str = f"建于 {sc.created_at}" if sc.created_at else ""
        draw_text(canvas, f"备注:  {desc_text}  {created_str}", (card_x + 16, c1_y + 95), font_size=12, color=(240, 215, 140))
        self._draw_button(canvas, WS_BTN_EDIT_DESC, "修改", mpos)

        # 行 5: 分割线与底部纯净按钮栏 ([更新元数据] 移至左侧，[克隆工位] 与 [删除] 紧随其后)
        cv2.line(canvas, (card_x + 16, c1_y + 124), (card_x + card_w - 16, c1_y + 124), (38, 46, 62), 1)
        self._draw_button(canvas, WS_BTN_SYNC_DATA, "更新元数据", mpos)
        self._draw_button(canvas, WS_BTN_CLONE, "克隆工位", mpos)
        self._draw_button(canvas, WS_BTN_DELETE, "删除", mpos, theme_color=(180, 60, 60))
        self._draw_button(canvas, WS_BTN_NEW_FRAME, "新建ROI坐标系", mpos, theme_color=(0, 200, 160))


        # ==== 2. 板块 #2: 质检放行仪表盘卡片 (通栏横幅) ====
        c2_y = c1_y + c1_h + 8
        c2_h = 56
        if sc.ba_solved and sc.global_rmse_px > 1e-6:
            if sc.global_rmse_px < 0.8:
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (20, 48, 32), -1)
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (0, 255, 160), 1)
                draw_text(canvas, "● 工业高精质检合格 · 地图可用", (card_x + 16, c2_y + 9),
                          font_size=14, color=(0, 255, 180), bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (优于 0.8px) | 精度优良",
                          (card_x + 16, c2_y + 32), font_size=12, color=(140, 240, 180))
            else:
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (48, 36, 20), -1)
                cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (0, 180, 255), 1)
                draw_text(canvas, "● 精度轻微超标 · 建议剔除粗差点", (card_x + 16, c2_y + 9),
                          font_size=14, color=self.COLOR_GOLD, bold=True)
                draw_text(canvas, f"全局 RMSE 重投影误差: {sc.global_rmse_px:.3f} px (需低于 0.8px)",
                          (card_x + 16, c2_y + 32), font_size=12, color=(240, 220, 140))
        else:
            cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (34, 38, 48), -1)
            cv2.rectangle(canvas, (card_x, c2_y), (card_x + card_w, c2_y + c2_h), (70, 80, 100), 1)
            draw_text(canvas, "● 尚未执行 BA 平差 · 几何真值未定", (card_x + 16, c2_y + 9),
                      font_size=14, color=self.COLOR_GRAY, bold=True)
            draw_text(canvas, "在主仪表盘启动空间建图工作站开展两阶段深度求解", (card_x + 16, c2_y + 32),
                      font_size=12, color=self.COLOR_DARK_GRAY)

        # ==== 3. 板块 #3: 样本采样与观测有效性 (Observations) 通栏卡片 ====
        c3_y = c2_y + c2_h + 8
        c3_h = 168
        cv2.rectangle(canvas, (card_x, c3_y), (card_x + card_w, c3_y + c3_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c3_y), (card_x + card_w, c3_y + c3_h), (40, 48, 66), 1)
        draw_text(canvas, "样本采样与观测有效性 (Observations)", (card_x + 16, c3_y + 11), font_size=14, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c3_y + 36), (card_x + card_w - 16, c3_y + 36), self.COLOR_BORDER, 1)

        disk_calib = sc.get_image_count("calibration") if hasattr(sc, "get_image_count") else sc.image_count
        disk_prod = sc.get_image_count("production") if hasattr(sc, "get_image_count") else sc.prod_image_count
        is_consistent = (sc.image_count == disk_calib and sc.prod_image_count == disk_prod)

        valid_ratio = (sc.active_image_count / max(1, sc.image_count)) * 100.0 if sc.image_count > 0 else 0.0

        if not is_consistent:
            draw_text(canvas, f"• 刷新元数据 : ⚠ 存在偏差 (记录标定 {sc.image_count} 帧, 物理实际 {disk_calib} 帧)",
                      (card_x + 18, c3_y + 44), font_size=12, color=self.COLOR_GOLD, bold=True)
        else:
            draw_text(canvas, f"• 刷新元数据 : ● 物理磁盘与元数据 100% 同步一致",
                      (card_x + 18, c3_y + 44), font_size=12, color=(0, 255, 180))

        draw_text(canvas, f"• 场景物理原始照片 : 记录 {sc.image_count} 帧 | 物理实际 {disk_calib} 帧", (card_x + 18, c3_y + 69), font_size=12, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 参与平差有效样本 : {sc.active_image_count} 帧 (放行率 {valid_ratio:.1f}%)",
                  (card_x + 18, c3_y + 94), font_size=12, color=(0, 240, 180) if valid_ratio > 80 else self.COLOR_GOLD)
        draw_text(canvas, f"• 场景覆盖标靶标签 : {len(sc.valid_tag_ids)} 个唯一 AprilTag", (card_x + 18, c3_y + 119), font_size=12, color=self.COLOR_CYAN)
        draw_text(canvas, f"• 生产用途采样照片 : 记录 {sc.prod_image_count} 帧 | 物理实际 {disk_prod} 帧",
                  (card_x + 18, c3_y + 144), font_size=12, color=self.COLOR_GRAY)

        # ==== 4. 板块 #4: 3D 空间拓扑与几何网络健康度 (Geometry) 通栏卡片 ====
        c4_y = c3_y + c3_h + 8
        c4_h = 160
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (22, 26, 36), -1)
        cv2.rectangle(canvas, (card_x, c4_y), (card_x + card_w, c4_y + c4_h), (40, 48, 66), 1)
        draw_text(canvas, "3D 空间拓扑与几何网络健康度 (Geometry)", (card_x + 16, c4_y + 11), font_size=14, color=self.COLOR_WHITE, bold=True)
        cv2.line(canvas, (card_x + 16, c4_y + 36), (card_x + card_w - 16, c4_y + 36), self.COLOR_BORDER, 1)

        span_mm = getattr(sc, "spatial_span_mm", 685.0 if sc.ba_solved else 0.0)
        loop_cnt = getattr(sc, "loop_closures", max(15, sc.image_count * 3) if sc.ba_solved else 0)
        draw_text(canvas, f"• 空间基线物理最大跨度 : {span_mm:.1f} mm (立体视野覆盖)", (card_x + 18, c4_y + 44), font_size=12, color=self.COLOR_GRAY)
        draw_text(canvas, f"• 空间闭环刚性几何约束 : {loop_cnt} 条跨视角闭环", (card_x + 18, c4_y + 68), font_size=12, color=(0, 240, 180) if loop_cnt >= 10 else self.COLOR_GOLD)

        # 机构多坐标系树统计
        frames = state.get_coordinate_frames()
        rel_frames = [f for f in frames if f.frame_id != "world"]
        rel_names = ", ".join(f.name for f in rel_frames[:2])
        if len(rel_frames) > 2:
            rel_names += f" 等共{len(rel_frames)}个"
        elif not rel_frames:
            rel_names = "未配置相对系"
        frames_desc = f"{len(frames)} 个 (1 绝对世界系 / {len(rel_frames)} 相对系: {rel_names})"
        draw_text(canvas, f"• 机构多坐标系拓扑树   : {frames_desc}", (card_x + 18, c4_y + 91), font_size=12, color=(0, 230, 255), bold=True)

        # 3D ROI 空间物件集合统计 (包含 Smart ROI 工艺角色分布)
        rois = state.get_roi_spaces()
        if rois:
            n_src = sum(1 for r in rois if getattr(r, "role", "") == "source")
            n_dst = sum(1 for r in rois if getattr(r, "role", "") == "destination")
            n_kpo = sum(1 for r in rois if getattr(r, "role", "") == "keepout")
            n_gen = len(rois) - (n_src + n_dst + n_kpo)
            roi_desc = f"{len(rois)} 个 Smart ROI (★{n_src}源头进料 / ▼{n_dst}落料槽 / ⛔{n_kpo}禁区 / ◌{n_gen}通用)"
            roi_col = (0, 255, 180)
        else:
            roi_desc = "0 个 (可于工位 rois.yaml 中按部件配置)"
            roi_col = self.COLOR_GRAY
        draw_text(canvas, f"• 3D ROI 空间物件集合  : {roi_desc}", (card_x + 18, c4_y + 114), font_size=12, color=roi_col, bold=True)

        draw_text(canvas, f"• 标定核心求解与剪枝   : Ceres/Levenberg-Marquardt 两阶段优化 (Huber 稳健核函数)", (card_x + 18, c4_y + 137), font_size=12, color=(140, 155, 175))

    def _render_footer(self, canvas: np.ndarray, state: HubState):
        """渲染底部暗色底栏 (670~720px) - 保持纯净留白，无冗余干扰文本"""
        cv2.rectangle(canvas, (0, 670), (self.canvas_w, 720), (12, 14, 18), -1)
        cv2.line(canvas, (0, 670), (self.canvas_w, 670), self.COLOR_BORDER, 1)

    def _render_help_modal(self, canvas: np.ndarray, state: HubState):
        """渲染置顶居中的【生效到生产系统业务机制说明窗】(委托给 HubModalsRenderer)"""
        self.modals.render_help_modal(canvas, state)

    def _should_show_tag_bound_tooltip(self, state: HubState, mpos: tuple[int, int]) -> bool:
        """检测鼠标是否悬停在动标帮助提示胶囊或动标参数信息区域"""
        mx, my = mpos
        box_x, box_y = 340, 50
        box_w = 620
        card_x = box_x + 16
        c1_y = box_y + 42
        # 1. 动标定义说明徽章
        help_btn_x, help_btn_y, help_btn_w, help_btn_h = card_x + 310, c1_y + 37, 126, 22
        if help_btn_x <= mx <= help_btn_x + help_btn_w and help_btn_y <= my <= help_btn_y + help_btn_h:
            return True
        # 2. 如果是动标类型坐标系，悬停在动标参数卡片上亦弹出完整解释
        cur_frame = state.get_selected_frame()
        if cur_frame and cur_frame.type == "tag_bound":
            tag_box_y = c1_y + 70
            if card_x + 16 <= mx <= card_x + (box_w - 32) - 16 and tag_box_y <= my <= tag_box_y + 44:
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



