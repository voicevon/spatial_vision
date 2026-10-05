#!/usr/bin/env python3
"""
芦笋上料自动化 (Dashboard)
=================================================
基于 1280x830 工业科技大屏，统一调度 flux_vision_3d 视觉系统的所有核心应用：
- 原生 Windows Unicode 窗口标题，杜绝任何乱码
- 窗口拖拽缩放自适应动态分辨率 (Dynamic Native Canvas)，文字 1:1 矢量超采样，无任何拉伸锯齿
- 高端工业冷峻暗色系配色（低饱和度科技冷蓝、工业深青与钛银灰），视觉沉稳专业不刺眼
- 左侧模块化卡片：【核心生产与工况】、【视觉标定与建图流水线】、【自动化测试与系统运维】
- 右侧动态即时说明大屏 (Live Inspector)：自适应扩展宽度，提供详尽的现场工程指南
"""

import os
import sys
import time
import argparse
import subprocess
from typing import List, Optional, Tuple

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

GUI_SETTINGS_FILE = os.path.join(PROJECT_ROOT, "config", "gui_settings.json")

from src.workspace.workspace_manager import WorkspaceManager
from src.ui.gui_theme import GuiTheme
from src.ui.gui_window_manager import GuiWindowManager
from src.ui.terminal_panel import TerminalPanel
from src.ui.text_rendering import draw_text, get_cached_font, put_text
from src.utils.logger import get_logger
from tools.env_utils import check_env_status

log = get_logger(__name__)


def wrap_text_by_width(text: str, font_size: int, max_width: int, bold: bool = False) -> List[str]:
    """根据像素最大可用宽度对文本进行精确折行（支持中英混合与标点符号）"""
    if not text or max_width <= 30:
        return [text] if text else []

    font = get_cached_font(font_size, bold)
    lines: List[str] = []
    curr_line = ""

    for char in text:
        test_line = curr_line + char
        bbox = font.getbbox(test_line)
        tw = bbox[2] - bbox[0]
        if tw > max_width and curr_line:
            lines.append(curr_line)
            curr_line = char
        else:
            curr_line = test_line

    if curr_line:
        lines.append(curr_line)

    return lines


def draw_multiline_text(img: np.ndarray, text: str, pos: Tuple[int, int], max_width: int,
                        font_size: int = 16, color: Tuple[int, int, int] = (240, 240, 240),
                        line_spacing: int = 4, bold: bool = False, max_lines: int = 99) -> int:
    """按最大像素宽度折行绘制多行文本，返回下一段可用的起始 y 坐标"""
    lines = wrap_text_by_width(text, font_size, max_width, bold=bold)
    if not lines:
        return pos[1]

    x, y = pos
    line_h = font_size + line_spacing
    for idx, line in enumerate(lines[:max_lines]):
        draw_text(img, line, (x, y + idx * line_h), font_size=font_size, color=color, bold=bold)

    return y + min(len(lines), max_lines) * line_h


class ToolCardMeta:
    """工具卡片定义与其详细说明元数据"""

    def __init__(self, key_id: str, shortcut: str, title: str, subtitle: str,
                 category: str, is_gui: bool, command: List[str],
                 tag_color: Tuple[int, int, int],
                 summary: str, details: List[str], inputs: List[str],
                 outputs: List[str], quick_tips: str, mode: str = "auto"):
        self.key_id = key_id
        self.shortcut = shortcut
        self.title = title
        self.subtitle = subtitle
        self.category = category
        self.is_gui = is_gui
        self.command = command
        self.tag_color = tag_color
        self.summary = summary
        self.details = details
        self.inputs = inputs
        self.outputs = outputs
        self.quick_tips = quick_tips
        # 运行模式三态: GUI (独立视窗) / CMD (外部控制台) / TERM (dashboard 内嵌终端)
        self.mode = ("GUI" if is_gui else "CMD") if mode == "auto" else mode


def build_tools_catalog() -> List[ToolCardMeta]:
    """构建全系统核心工具目录：12 张卡片，三大功能分组 (A环境场景→B Tag标定→D生产调试)"""

    COLOR_A = (195, 155, 45)   # A 环境场景  : 琥珀金 (Amber)
    COLOR_B = (65,  175, 160)  # B Tag标定   : 精密工业深青 (Teal)
    COLOR_D = (90,  140, 195)  # D 生产调试  : 钢蓝 (Steel Blue)

    catalog = [
        # ===== A — Workspace =====
        ToolCardMeta(
            key_id="workspace_hub",
            shortcut="1",
            title="Workspace",
            subtitle="列表，白名单，评估",
            category="A — Workspace",
            is_gui=True,
            command=[sys.executable, "-m", "tools.workspace_hub"],
            tag_color=COLOR_A,
            summary="工业视觉 Workspace 统一数据中枢，支持多工位沙盒隔离、Tag 白名单管理与精度体检评估。",
            details=[
                "列表管理：多 Workspace 沙盒创建、重命名、克隆与物理隔离",
                "白名单管理：各 Workspace 独立维护 tag_whitelist.yaml 白名单配置与同步",
                "精度与几何评估：全局立体地图 (tags_map.yaml) 与残差质量体检",
                "生产原子发布：平差解算成果一键原子发布至生产运行基准"
            ],
            inputs=["data/workspaces/ 工位目录"],
            outputs=["workspace_meta.yaml、tag_whitelist.yaml、工位顶层 tags_map.yaml"],
            quick_tips="快捷键: [1] 启动 | 中枢内 [⏎] 激活 | [P] 发布生产 | [W] 编辑白名单 | [S] 进Studio",
        ),

        # ===== B — 标定建图与生产验证流水线 =====
        ToolCardMeta(
            key_id="tag_wizard",
            shortcut="2",
            title="图像采集",
            subtitle="双用途 (标定/生产) 快速采集",
            category="B — 标定建图与生产验证",
            is_gui=True,
            command=[sys.executable, "tools/capture/capture_wizard.py"],
            tag_color=COLOR_B,
            summary="交互式工业图像采集工作台，提供实时画面预览与标定/生产双用途采集路由。",
            details=[
                "工作空间与双用途：顶部自由切换归档工作空间与用途 (标定 calibration / 生产 production)",
                "快捷拍照与连拍：点击 [拍照] 或按空格键一键拍照并自动归档至对应用途的 raw_images/ 目录",
                "曝光实时调节：支持曝光微调与白闪快门视觉反馈",
                "中枢自动同步：采集完成后数据自动同步至 Workspace 管理中枢"
            ],
            inputs=["RealSense 深度相机或 USB 摄像头"],
            outputs=["当前工位 raw_images/ 原始图集 (标定/生产隔离存储)"],
            quick_tips="快捷键: [2] 启动 | 预览中点击 [拍照] 或按 [空格] | [ESC]/[Q] 退出"
        ),

        ToolCardMeta(
            key_id="spatial_mapping_studio",
            shortcut="3",
            title="空间建图工作站",
            subtitle="深度平差与空间立体建图",
            category="B — 标定建图与生产验证",
            is_gui=True,
            command=[sys.executable, "-m", "tools.spatial_mapping_studio"],
            tag_color=COLOR_B,
            summary="空间立体建图核心工作站，提供多视角图像样本交互审核、两阶段稳健 BA 全局平差与智能残差剪枝。",
            details=[
                "样本交互审核：多视角图像列表与演进矩阵浏览，支持残差过滤与剔除",
                "两阶段 BA 平差：Cauchy 粗平差结合 LM 精平差高精度空间解算",
                "智能残差剪枝：自动剔除高残差反光外点，守门拓扑连通结构",
                "空间地图落盘：解算成果直接落盘工位顶层 tags_map.yaml 与全景质检单"
            ],
            inputs=["当前工位 calibration/raw_images/、相机内参文件"],
            outputs=["当前工位顶层 tags_map.yaml、全景精度质检单"],
            quick_tips="快捷键: [3] 启动 | 工作站内 [B] BA平差 | [A] 剪枝 | [R] 导出报告 | [M] 保存地图"
        ),

        ToolCardMeta(
            key_id="asparagus_offline",
            shortcut="4",
            title="芦笋位姿工作室",
            subtitle="生产样本与位姿解算",
            category="B — 标定建图与生产验证",
            is_gui=True,
            command=[sys.executable, "-m", "tools.asparagus_pose_studio"],
            tag_color=COLOR_B,
            summary="离线解算芦笋空间位姿，对比多技术路线算法流水线并精调特征参数。",
            details=[
                "工位样本载入：优先读取当前工位 production/raw_images/ 样本",
                "3D 空间解算：台面点云拟合、实例切分与顶层芦笋空间位姿判决",
                "3D 抓取位姿验证：实时解算抓取中轴线、法向矢量与品质评级 (A/B/C)",
                "多阶段步骤视图：深度对比 10+ 种脊线骨架算法并一键调参"
            ],
            inputs=["工位生产照片目录 (production/raw_images/ 或 snapshots)"],
            outputs=["芦笋位姿检测结果、品质指标分析与批量评估报表"],
            quick_tips="快捷键: [4] 启动 | [↑↓] 切换样本 | [B] 批量解算 | [ESC] 退出"
        ),

        ToolCardMeta(
            key_id="scara_production",
            shortcut="5",
            title="SCARA 抓取生产",
            subtitle="皮带感知与 SCARA 抓取闭环",
            category="B — 标定建图与生产验证",
            is_gui=True,
            command=[sys.executable, "-m", "tools.scara_production"],
            tag_color=COLOR_B,
            summary="实时解算进料皮带物料位姿，智能调度落料槽位并驱动 SCARA 机械臂自动抓取。",
            details=[
                "实时工位感知：基于工位顶层 tags_map.yaml 与 Smart ROI 进料区感知物料",
                "业务调度决策：根据品质分级 (A/B/C) 自动路由至空闲落料槽位",
                "动态轨迹规划：调用 ScaraMotionPlanner 生成三段式防撞安全 G-code",
                "实机串口闭环：通过 MKS Base 串口驱动 SCARA 机械臂执行闭环搬运"
            ],
            inputs=["当前工位顶层 tags_map.yaml、rois.yaml、机械臂串口"],
            outputs=["实时分选调度看板、各槽位根数统计与 SCARA 搬运执行"],
            quick_tips="快捷键: [5] 启动 | [SPACE] 启动/暂停自动生产 | [S] 单拍测试 | [R] 换箱复位"
        ),

        ToolCardMeta(
            key_id="isolate_wheels_production",
            shortcut="0",
            title="分离轮生产工作台",
            subtitle="[0] 8 通道自动分选与连续生产",
            category="B — 标定建图与生产验证",
            is_gui=True,
            command=[sys.executable, "-m", "tools.isolate_wheels_production"],
            tag_color=COLOR_B,
            summary="分离轮 8 通道自动化生产工作台，支持配方出料调度、速度倍率缩放、PPM 效能监视与连续自动循环生产。",
            details=[
                "实时物流拓扑：8 托架槽位横向流水线（1->8 流向），出口指示与实时出料统计",
                "自动化调度：单拍步进出料与连续无人化循环生产，根据 idle/done 自动节拍闭环",
                "速度与节拍调优：集成 0.1x~2.0x 动态速度缩放及 0.5s~3.0s 循环间隔设置",
                "生产效能统计：实时统计总节拍数、累计出料件数、平均节拍耗时 (CT) 与 PPM 速率"
            ],
            inputs=["MQTT Broker (voicevon.vicp.io:1883, 账号 von)"],
            outputs=["节拍命令下发、自动连续循环生产、实时生产效能监控看板"],
            quick_tips="快捷键: [0] 启动 | [SPACE] 启动/暂停自动循环 | [S] 单拍出料 | [C] 重置统计"
        ),

        # ===== D — 硬件调试与系统运维 =====
        ToolCardMeta(
            key_id="d435_live",
            shortcut="6",
            title="RealSense 诊断",
            subtitle="硬件检测/深度探针/双流画面",
            category="D — 硬件调试与系统运维",
            is_gui=True,
            command=[sys.executable, "tools/d435_viewer.py"],
            tag_color=COLOR_D,
            summary="RealSense 深度相机纯预览诊断，支持 RGB 与对齐深度双流探针。",
            details=[
                "实时双流显示：独立开关 RGB 与对齐深度流画面",
                "多视窗排布：支持上下/左右排列切换与画面缩放",
                "空间深度探针：鼠标悬停读取毫米级深度与 3D 坐标",
                "深度色阶微调：支持自动量程与热力图色彩映射"
            ],
            inputs=["Intel RealSense D435 深度相机 或 USB 摄像头"],
            outputs=["屏幕实时预览画面与深度探针读数 (不落盘)"],
            quick_tips="快捷键: [6] 启动 | [Space] 暂停 | [V] 排列 | [S] 抓拍 | [Q] 退出"
        ),

        # ===== D — 硬件调试与系统运维 (续，SCARA 机械臂调试) =====
        ToolCardMeta(
            key_id="scara_debug",
            shortcut="7",
            title="SCARA 机械臂调试",
            subtitle="点动/回零/设零/舵机/G-code 透传",
            category="D — 硬件调试与系统运维",
            is_gui=True,
            command=[sys.executable, "tools/scara_debug/app.py"],
            tag_color=COLOR_D,
            summary="SCARA 机械臂图形化调试终端，支持轴点动、回零设零与 G-code 透传。",
            details=[
                "串口连接管理：自动枚举端口连接并支持仿真模式",
                "核心运动控制：限位诊断、三轴回零、坐标设零与电机释放",
                "点动与舵机控制：笛卡尔与关节角点动，支持末端夹爪开闭",
                "工位跳转与透传：预设工位跳转、搬运节拍宏与原生 G-code 透传"
            ],
            inputs=["MKS Base V1.6 串口 (如 COM11)", "几何参数 loader_core/config.py"],
            outputs=["串口 G-code 指令下发、机械臂动作执行与通信日志"],
            quick_tips="快捷键: [7] 启动 | 界面内 W/S/A/D 点动 | [G28] 回零 | [ESC] 退出"
        ),

        # ===== D — 硬件调试与系统运维 (续，分离轮 MQTT 调试) =====
        ToolCardMeta(
            key_id="isolate_wheels_debug",
            shortcut="8",
            title="Isolator WHEELS 调试",
            subtitle="[8] 分离轮 MQTT 调试 (8 托架节拍)",
            category="D — 硬件调试与系统运维",
            is_gui=True,
            command=[sys.executable, "tools/isolate_wheels_debug/app.py"],
            tag_color=COLOR_D,
            summary="flux_isolate_wheels 分离轮 ESP32 的 MQTT 调试视窗，支持设备发现、状态监视与 8 托架节拍命令下发。",
            details=[
                "设备发现：订阅 flux/loader/+/* 通配符自动列出在线分离轮设备",
                "状态监视：state 保留消息初始化、done 节拍应答、log 固件日志实时滚动",
                "节拍下发：8 托架数量步进器 (0~9)、全部清零与稳妥模式发送 (idle 受理)",
                "协议对齐：voicevon.vicp.io:1883, 协议 v1.0, 仅 idle 受理命令"
            ],
            inputs=["MQTT Broker (voicevon.vicp.io:1883, 账号 von)"],
            outputs=["节拍命令 JSON {\"cmd\":\"load\",\"counts\":[...]} 与运行日志面板"],
            quick_tips="快捷键: [8] 启动 | 启动即自动连接 Broker，点设备芯片切换目标设备",
        ),

        # ===== D — 硬件调试与系统运维 (续，NetCamera 调试) =====
        ToolCardMeta(
            key_id="net_camera_debug",
            shortcut="9",
            title="NetCamera 调试",
            subtitle="[9] 手机网络摄像机 MQTT+RTSP 调试",
            category="D — 硬件调试与系统运维",
            is_gui=True,
            command=[sys.executable, "tools/net_camera_debug/app.py"],
            tag_color=COLOR_D,
            summary="android_as_camera 手机网络摄像机的调试视窗，支持 MQTT 设备发现、状态监视、远程命令下发与 RTSP 实时预览。",
            details=[
                "设备发现：订阅 camera/+/state 通配符自动列出在线摄像机及其推流状态",
                "远程控制：开始/停止推流、前后摄切换、变焦、拍照、录像、移动侦测开关与灵敏度",
                "RTSP 预览：按 state 上报 IP 拉取 rtsp://{ip}:8554/live 实时画面 (TCP 交错)",
                "事件告警：motion/photo/record 事件实时滚动显示与 toast 提示"
            ],
            inputs=["MQTT Broker (voicevon.vicp.io:1883, 账号 von)", "手机 App RTSP 流 rtsp://{ip}:8554/live"],
            outputs=["命令 JSON {\"action\":...} 下发至 camera/{id}/cmd 与事件日志面板"],
            quick_tips="快捷键: [9] 启动 | 启动即自动连接 Broker，点设备芯片切换目标，点拉流预览看画面",
        ),

        # ===== D — 硬件调试与系统运维 (续，系统诊断) =====
        ToolCardMeta(
            key_id="sys_diagnose_tests",
            shortcut="T",
            title="系统环境诊断与测试",
            subtitle="[T] 驱动与依赖诊断 / 自动化测试套件",
            category="D — 硬件调试与系统运维",
            is_gui=False,
            command=[sys.executable, "tools/diagnose_env.py"],
            tag_color=COLOR_D,
            summary="全面检查系统运行环境依赖，并提供全量自动化测试套件守门。",
            details=[
                "核心依赖检查：排查 Python、OpenCV、NumPy 及相机驱动状态",
                "科学计算库诊断：检查 yaml、matplotlib、scipy 等工业计算包",
                "全量自动化测试：一键运行 unittest 测试矩阵验证代码稳定性",
                "质量安全守门：覆盖平差算法、拓扑连通性与 UI 状态机"
            ],
            inputs=["系统底层环境注册表与 tests/ 全量测试框架"],
            outputs=["嵌入式终端实时输出逐项绿勾诊断报告与全工程测试矩阵"],
            quick_tips="快捷键: [T] 启动环境深度诊断 (内嵌终端) | 遇到红叉时依提示执行修复",
            mode="TERM"
        ),

        ToolCardMeta(
            key_id="pip_install",
            shortcut="P",
            title="安装/更新项目依赖",
            subtitle="[P] pip install -r requirements.txt",
            category="D — 硬件调试与系统运维",
            is_gui=False,
            command=[sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            tag_color=COLOR_D,
            summary="一键安装或更新项目 Python 依赖包，确保与 requirements.txt 严格对齐。",
            details=[
                "依赖自动安装：安装 requirements.txt 中声明的全部依赖",
                "核心库版本锁定：自动处理 numpy、opencv、pyrealsense2 等核心包",
                "离线与在线兼容：支持本地轮子包与源配置提示",
                "环境快速就绪：克隆项目或更新依赖后的一键初始化入口"
            ],
            inputs=["requirements.txt 文件"],
            outputs=["pip 安装进度与版本锁定结果"],
            quick_tips="快捷键: [P] 启动 (内嵌终端) | 首次克隆项目后必执行",
            mode="TERM"
        ),

        # ===== D — 硬件调试与系统运维 (续，AprilTag 图纸生成) =====
        ToolCardMeta(
            key_id="tag_paper_gen",
            shortcut="W",
            title="AprilTag 图纸生成",
            subtitle="[W] 一键生成 ID 0~29 标靶与 A4 打印 PDF",
            category="D — 硬件调试与系统运维",
            is_gui=False,
            command=[sys.executable, "tools/calibration/generate_apriltags.py"],
            tag_color=COLOR_D,
            summary="一键批量生成 AprilTag 16h5 标靶（ID 0~29）矢量高清图及标准 A4 排版打印 PDF。",
            details=[
                "图纸批量生成：一键生成 ID 0~29 标靶高清图像与矢量排版文件",
                "标准 A4 布局：自动按工业规范以 50mm 物理尺寸排版于 A4 页面",
                "矢量 PDF 导出：调用 ReportLab 高保真输出至 data/apriltags_16h5/",
                "打印标准警示：按 100% 实际尺寸打印，禁止缩放或适应页面"
            ],
            inputs=["ReportLab 依赖库 (用于矢量 PDF 排版导出)"],
            outputs=["data/apriltags_16h5/apriltags_16h5_A4_print.pdf"],
            quick_tips="快捷键: [W] 启动 (控制台生成) | 自动生成图纸并写盘，请以 100% 比例打印",
            mode="CMD"
        ),
    ]
    return catalog


class GuiLauncherApp:
    """3D Vision 统一 GUI 控制中心主应用"""

    # 主题调色板统一取自 GuiTheme 单源 (暗色工业风; 明暗切换改 src/utils/gui_theme.py)
    COLOR_BG = GuiTheme.BG              # 钛黑背景
    COLOR_CARD_BG = GuiTheme.CARD_BG    # 碳灰底色
    COLOR_CARD_HOVER = GuiTheme.CARD_HOVER   # 悬停轻提亮
    COLOR_CARD_SEL = GuiTheme.CARD_SEL  # 选中微蓝底色
    COLOR_BORDER = GuiTheme.BORDER      # 沉稳边框线
    COLOR_BORDER_HOVER = GuiTheme.BORDER_HOVER  # 悬停微光冷青
    COLOR_BORDER_SEL = GuiTheme.BORDER_SEL      # 选中发光青冷线
    COLOR_TEXT_TITLE = GuiTheme.TEXT    # 纯白冷色
    COLOR_TEXT_SUB = GuiTheme.TEXT_SUB  # 冷银灰副标题
    COLOR_TEXT_MUTED = GuiTheme.TEXT_MUTED  # 辅助提示暗灰
    COLOR_ACCENT = GuiTheme.ACCENT      # 科技主强调色 (冰魄冷青)
    COLOR_GOLD = GuiTheme.GOLD          # 关键生产资产点缀金

    def __init__(self, settings_file: Optional[str] = None):
        self._settings_file = settings_file or GUI_SETTINGS_FILE
        self._is_active = True
        # 窗口内部 key 标识使用纯英文，通过 Windows API 设定中文标题杜绝乱码
        self.window_name = "flux_vision_3d_suite_dashboard"
        self._running = True

        # 视口/缩放/窗口偏好统一委托 GuiWindowManager 单源管理 (基准 1280x1000)
        self._base_w = 1280
        self._base_h = 1000
        self.win_mgr = GuiWindowManager(
            app_id="gui_launcher", base_w=self._base_w, base_h=self._base_h,
            settings_file=self._settings_file)

        self.workspace_mgr = WorkspaceManager()
        self.tools = build_tools_catalog()
        self.selected_tool_idx = -1    # 初始无选中，键盘/点击才激活焦点
        self.hover_tool_idx = -1

        # 子工具前台运行与暗化挂起态
        self.is_subtool_running: bool = False
        self.running_tool_meta: Optional[ToolCardMeta] = None

        # 嵌入式终端 (纯输出型工具: 系统环境诊断 / pip 依赖安装)
        # key_id → TerminalPanel; 启动后右侧大屏切换为终端视图, 进程后台持续运行
        self.terminals = {}
        self._term_btn_rects = []      # 每帧由终端视图重建: [(btn_id, rect), ...]

        self.mouse_x = -1
        self.mouse_y = -1
        if self.win_mgr.scale_pct != 100 or self.win_mgr.canvas_w != self._base_w or self.win_mgr.canvas_h != self._base_h:
            self.toast_msg = f"已自动恢复偏好设置：放大镜 {self.win_mgr.scale_pct}%，视窗 {self.win_mgr.canvas_w}×{self.win_mgr.canvas_h} (按 Ctrl+0 可随时复位)"
        else:
            self.toast_msg = "欢迎使用芦笋上料自动化系统！按数字键或点击卡片进入工况中枢。"
        self.toast_time = time.time() + 4.5

        # 系统状态缓存
        self.system_status = {}
        self.refresh_system_status()

    def _save_settings(self):
        """持久化保存当前缩放比例与窗口尺寸 (委托 GuiWindowManager, 支持多应用隔离)"""
        if not getattr(self, "_is_active", False):
            return
        self.win_mgr.save_settings()

    def set_toast(self, msg: str, duration: float = 3.5):
        """设置底部提示消息"""
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def refresh_system_status(self):
        """刷新底层系统与场景状态"""
        try:
            self.workspace_mgr.invalidate_cache()
            self.system_status = check_env_status(force_refresh=True)
        except Exception:
            self.system_status = {}

    def _apply_zoom(self, delta_pct: int, reset: bool = False):
        """执行全局真矢量缩放 (委托 GuiWindowManager 单源)：卡片尺寸、字号、间距等比矢量缩放并持久化"""
        _, hint = self.win_mgr.apply_zoom(delta_pct, reset=reset)
        if hint:
            self.set_toast(hint, duration=2.2)

    def _present_canvas(self):
        """在当前物理窗口分辨率下原生呈现矢量画布 (零位图拉伸，零锯齿)"""
        canvas = self._render_canvas()
        cv2.imshow(self.window_name, canvas)

    def run(self):
        """主事件循环 (GuiWindowManager 单源窗口管理 + 全屏真矢量动态排版重绘)"""
        self.win_mgr.setup_window(self.window_name, mouse_callback=self._on_mouse)
        self.win_mgr.set_unicode_title("芦笋上料自动化 | Dashboard")

        # 首次呈现
        self._present_canvas()

        while self._running:
            # 0. 窗口关闭检测：若用户直接点击右上角红叉 [X]，安全退出
            if not self.win_mgr.is_window_alive():
                break

            # 1. 硬件级按键轮询 (绕过中文输入法对加减号的拦截)
            hw_changed, hw_toast = self.win_mgr.poll_hardware_zoom()
            if hw_changed and hw_toast:
                self.set_toast(hw_toast, duration=2.2)

            # 2. 动态检测窗口拖拽拉伸尺寸，防抖 0.35s 后自动落盘持久化
            self.win_mgr.sync_window_size()

            # 3. 呈现真矢量画布 (无任何 cv2.resize 插值，字形完美)
            self._present_canvas()

            # 4. OpenCV 事件捕获通道
            raw_key = cv2.waitKeyEx(20)
            if raw_key == -1:
                continue

            # 处理退出
            if raw_key in (27, ord('q'), ord('Q')):
                break

            # 键盘选择与启动分发
            self._handle_keyboard(raw_key)

        # 退出前终止内嵌终端子进程并持久化保存最终视口偏好
        for panel in self.terminals.values():
            panel.stop_quiet()
        self._save_settings()
        cv2.destroyAllWindows()

    def _on_mouse(self, event, x, y, flags, param):
        """鼠标移动、点击与滚轮缩放事件 (与物理坐标 1:1 原生对齐)"""
        if self.is_subtool_running:
            return  # 子应用运行期间，主视窗处于安全挂起待命态，屏蔽一切鼠标操作

        self.mouse_x = x
        self.mouse_y = y

        # ── 1. Ctrl + 鼠标滚轮缩放 (委托 GuiWindowManager 单源处理) ─────────────
        handled, toast = self.win_mgr.handle_mouse_wheel(event, flags)
        if toast:
            self.set_toast(toast, duration=2.2)
        if handled:
            return

        # ── 1.5 嵌入式终端滚轮回看 (普通滚轮 + 终端视图 + 鼠标在右侧大屏内) ──────
        if event == cv2.EVENT_MOUSEWHEEL and not handled and self._is_terminal_view():
            s = self.win_mgr.scale_pct / 100.0
            CW = max(130, int(222 * s))
            SX = max(6, int(12 * s))
            X0 = max(8, int(15 * s))
            split_x = X0 + CW * 2 + SX + max(8, int(15 * s))
            if x >= split_x:
                panel = self.terminals.get(self.tools[self.selected_tool_idx].key_id)
                body = self._terminal_body_rect(split_x)
                visible = max(1, (body[3] - body[1] - 8) // panel.row_h)
                panel.scroll(3 if flags > 0 else -3, visible)   # 上滚回看, 触底恢复跟随
                return

        # 检测鼠标悬停在哪个卡片上（根据当前 scale 坐标直接命中检测）
        card_idx = self._hit_test_cards(x, y)
        self.hover_tool_idx = card_idx

        s = self.win_mgr.scale_pct / 100.0
        # 鼠标左键点击
        if event == cv2.EVENT_LBUTTONDOWN:
            # 顶部右上角退出按钮 (自适应右对齐，纯“退出”)
            bw = max(60, int(80 * s))
            bh = max(24, int(34 * s))
            bx = self.win_mgr.canvas_w - bw - max(8, int(15 * s))
            by = max(6, int(10 * s))
            if bx <= x <= bx + bw and by <= y <= by + bh:
                self._running = False
                return

            # 终端视图标题栏按钮 ([停止]/[返回])
            if self._is_terminal_view():
                for bid, (bx1, by1, bx2, by2) in self._term_btn_rects:
                    if bx1 <= x <= bx2 and by1 <= y <= by2:
                        self._on_terminal_button(bid)
                        return

            # 点击左侧卡片
            if card_idx != -1:
                self.selected_tool_idx = card_idx
                self._launch_tool(self.tools[card_idx])
                return
            else:
                # 点击非卡片区域复位选中状态，返回系统大屏总览
                self.selected_tool_idx = -1

        # 鼠标双击直接启动
        elif event == cv2.EVENT_LBUTTONDBLCLK:
            if card_idx != -1:
                self.selected_tool_idx = card_idx
                self._launch_tool(self.tools[card_idx])

    def _handle_keyboard(self, raw_key: int):
        """键盘快捷键响应 (3分组: row0 A 1张全宽, rows1-2 B 4张(2+2), rows3-5 D 6张(2+2+2))"""
        if self.is_subtool_running:
            return  # 子应用运行期间，主视窗处于安全挂起待命态，屏蔽一切按键操作

        if raw_key == 8:  # Backspace 键复位选中状态
            self.selected_tool_idx = -1
            self.hover_tool_idx = -1
            return

        if raw_key in (13, 10):  # 回车
            if 0 <= self.selected_tool_idx < len(self.tools):
                self._launch_tool(self.tools[self.selected_tool_idx])
            return

        # 方向键：将卡片索引映射到 (row, col) 坐标后导航
        # row 0: idx 0 (A Workspace，全宽)
        # rows 1-3: idx 1-5 (B 区 5张: 2+2+1)
        # rows 4-7: idx 6-12 (D 区 7张: 2+2+2+1)
        def idx_to_rc(i: int) -> Tuple[int, int]:
            if i <= 0:
                return (0, 0)
            if 1 <= i <= 5:   # B 区 5张 (row1: idx 1, 2; row2: idx 3, 4; row3: idx 5)
                b = i - 1
                return (b // 2 + 1, b % 2)
            d = i - 6         # D 区 7张 (row4: idx 6, 7; row5: idx 8, 9; row6: idx 10, 11; row7: idx 12)
            return (d // 2 + 4, d % 2)

        def rc_to_idx(r: int, c: int) -> int:
            if r == 0:
                return 0
            if 1 <= r <= 3:   # B 区 (row1: idx 1, 2; row2: idx 3, 4; row3: idx 5)
                base_b = (r - 1) * 2
                return min(1 + base_b + c, 5)
            if 4 <= r <= 7:   # D 区 (row4: idx 6, 7; row5: idx 8, 9; row6: idx 10, 11; row7: idx 12)
                base_d = (r - 4) * 2
                return min(6 + base_d + c, 12)
            return 12

        row, col = idx_to_rc(self.selected_tool_idx)

        if raw_key in (2490368, 65362, 38):    # 上
            if row > 0:
                row -= 1
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2621440, 65364, 40):    # 下
            if row < 7:
                row += 1
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2424832, 65361, 37):    # 左
            col = 0
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        if raw_key in (2555904, 65363, 39):    # 右
            col = 1
            self.selected_tool_idx = rc_to_idx(row, col)
            self.hover_tool_idx = self.selected_tool_idx
            return

        # ── 检测 Ctrl 键状态 (全面兼容 Windows 各种键盘布局与笔记本) ────────────
        ctrl_held = False
        if getattr(self.win_mgr, "_force_ctrl_pressed", False):
            ctrl_held = True
        elif sys.platform == "win32":
            try:
                import ctypes
                u32 = ctypes.windll.user32
                # VK_CONTROL = 0x11, 同步检测当前线程消息队列与物理硬件实时状态
                ctrl_held = bool((u32.GetKeyState(0x11) & 0x8000) or (u32.GetAsyncKeyState(0x11) & 0x8000))
            except Exception:
                pass  # GUI 可选功能：Ctrl 键状态探测失败按未按下处理

        # ── 缩放键委托 GuiWindowManager 单源处理 (笔记本 =/+ -/_ 0 全面兼容) ──────
        handled, toast = self.win_mgr.handle_keyboard_fallback(raw_key)
        if toast:
            self.set_toast(toast, duration=2.2)
        if handled:
            return

        # 若按住 Ctrl 且未命中缩放，拦截避免误触普通单键快捷键
        if ctrl_held:
            return

        # 数字 / 字母快捷键（与新卡片顺序严格对齐）
        key_char = chr(raw_key & 0xFF).lower() if (raw_key & 0xFF) < 128 else ""

        shortcut_map = {
            '1': "workspace_hub",         # A Workspace 工作空间中枢
            '2': "tag_wizard",            # B 图像采集 (双用途)
            '3': "spatial_mapping_studio", # B 空间建图工作站
            '4': "asparagus_offline",     # B 芦笋抓取位姿离线解算
            '5': "scara_production",      # B SCARA 抓取生产工作台
            '0': "isolate_wheels_production", # B 分离轮生产工作台
            '6': "d435_live",             # D RealSense 诊断
            '7': "scara_debug",           # D SCARA 机械臂调试
            '8': "isolate_wheels_debug",  # D Isolator WHEELS 调试
            '9': "net_camera_debug",      # D NetCamera 调试
            't': "sys_diagnose_tests",
            'p': "pip_install",
            'w': "tag_paper_gen",         # D AprilTag 图纸生成 (存量补缺)
        }

        if key_char in shortcut_map:
            target_id = shortcut_map[key_char]
            for idx, tool in enumerate(self.tools):
                if tool.key_id == target_id:
                    self.selected_tool_idx = idx
                    self.hover_tool_idx = idx
                    self._launch_tool(tool)
                    return

    def _get_card_rect(self, idx: int) -> Tuple[int, int, int, int]:
        """返回第 idx 张卡片的 (x, y, w, h)，与渲染布局严格保持一致

        布局 (8行，3分组，共 13 张卡片):
          row 0     A Workspace (1张全宽: idx 0)
          rows 1-3  B 标定建图与生产验证流水线 (5张: 2+2+1, idx 1~5)
          rows 4-7  D 硬件调试与系统运维 (7张: 2+2+2+1, idx 6~12)
        """
        s = self.win_mgr.scale_pct / 100.0
        LH = max(14, int(20 * s))
        CH = max(40, int(70 * s))
        CW = max(130, int(222 * s))
        SX = max(6, int(12 * s))
        SY = max(4, int(8 * s))
        GY = max(16, int(40 * s))
        X0 = max(8, int(15 * s))
        Y0 = max(40, int(66 * s))
        FW = CW * 2 + SX

        if idx == 0:          # A: 顶部全宽卡片 (Workspace)
            return X0, Y0 + LH, FW, CH

        if 1 <= idx <= 5:     # B: 5张 (2+2+1, 3行: idx 1~5)
            b = idx - 1
            base_y = Y0 + LH + CH + GY + LH
            return X0 + (b % 2) * (CW + SX), base_y + (b // 2) * (CH + SY), CW, CH

        # D: 7张 (2+2+2+1, 4行: idx 6~12)
        d = idx - 6
        base_y = Y0 + LH + CH + GY + LH + 3 * (CH + SY) + GY + LH
        return X0 + (d % 2) * (CW + SX), base_y + (d // 2) * (CH + SY), CW, CH

    def _hit_test_cards(self, x: int, y: int) -> int:
        """鼠标命中测试：委托给 _get_card_rect，与渲染位置严格一致"""
        for idx in range(len(self.tools)):
            cx, cy, cw, ch = self._get_card_rect(idx)
            if cx <= x <= cx + cw and cy <= y <= cy + ch:
                return idx
        return -1

    def _launch_tool(self, tool: ToolCardMeta):
        """执行启动子工具或测试 (支持全屏暗化蒙版与控制权移交挂起浮岛)"""
        # 纯输出型工具 (系统环境诊断/pip 依赖安装) 走内嵌终端: 右侧大屏切换, GUI 保持可交互
        if tool.key_id in ("sys_diagnose_tests", "pip_install"):
            self._launch_in_terminal(tool)
            return
        self._save_settings()  # 立即落盘记忆当前大小与比例
        self.is_subtool_running = True
        self.running_tool_meta = tool
        self.set_toast(f"已移交控制权，正在拉起: 【{tool.title}】...")

        # 立即在主窗口渲染暗化挂起画布并强制上屏
        self._present_canvas()
        cv2.waitKey(40)

        cmd = tool.command
        try:
            if tool.is_gui:
                subprocess.run(cmd)
                self.set_toast(f"【{tool.title}】已安全返回，控制中心已重新就绪。")
            else:
                if sys.platform == "win32":
                    full_cmd_str = " ".join([f'"{c}"' if " " in c else c for c in cmd])
                    wrapper_cmd = f'cmd.exe /c "{full_cmd_str} & echo. & echo [完成] 请按任意键返回控制中心... & pause > nul"'
                    subprocess.run(wrapper_cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
                    self.set_toast(f"【{tool.title}】执行完毕，控制中心已重新就绪。")
                else:
                    subprocess.run(cmd)
                    self.set_toast(f"【{tool.title}】执行完毕，控制中心已重新就绪。")
        except Exception as e:
            self.set_toast(f"启动失败: {e}", duration=5.0)
        finally:
            self.is_subtool_running = False
            self.running_tool_meta = None
            self.refresh_system_status()
            self._present_canvas()

    # ========================== 嵌入式终端 (纯输出型工具) ==========================

    def _launch_in_terminal(self, tool: ToolCardMeta):
        """内嵌终端启动: 右侧大屏切换为终端视图, 子进程后台流式运行 (GUI 保持可交互)"""
        self._save_settings()
        idx = self.tools.index(tool)
        self.selected_tool_idx = idx
        self.hover_tool_idx = idx
        panel = self.terminals.get(tool.key_id)
        if panel is None:
            panel = TerminalPanel()
            self.terminals[tool.key_id] = panel
        if panel.is_running():
            self.set_toast(f"【{tool.title}】正在运行, 已切换到终端视图")
            return
        self.set_toast(f"已在右侧终端启动: 【{tool.title}】")
        panel.start(tool.command, cwd=PROJECT_ROOT)

    def _is_terminal_view(self) -> bool:
        """当前选中卡片是否处于内嵌终端视图"""
        idx = self.selected_tool_idx
        return 0 <= idx < len(self.tools) and self.tools[idx].key_id in self.terminals

    def _terminal_button_rects(self, split_x: int):
        """终端标题栏 [停止][返回] 按钮矩形 (渲染与鼠标命中共用, 返回最右为 [返回])"""
        s = self.scale_pct / 100.0
        top_h = max(36, int(54 * s))
        header_h = max(36, int(46 * s))
        bw, bh = max(56, int(78 * s)), max(22, int(28 * s))
        y1 = top_h + (header_h - bh) // 2
        rects, bx2 = [], self.canvas_w - max(8, int(14 * s))
        for bid in ("TERM_BACK", "TERM_STOP"):   # 从右往左排布
            bx1 = bx2 - bw
            rects.append((bid, (bx1, y1, bx2, y1 + bh)))
            bx2 = bx1 - max(6, int(10 * s))
        return rects

    def _terminal_body_rect(self, split_x: int):
        """终端本体区域 (标题栏之下, 状态行之上)"""
        s = self.scale_pct / 100.0
        top_h = max(36, int(54 * s))
        footer_h = max(34, int(50 * s))
        header_h = max(36, int(46 * s))
        status_h = max(20, int(24 * s))
        pad = max(2, int(4 * s))
        return (split_x + pad, top_h + header_h + pad,
                self.canvas_w - pad, self.canvas_h - footer_h - status_h)

    def _on_terminal_button(self, btn_id: str):
        """终端标题栏按钮: [停止] 终止子进程 / [返回] 回系统总览"""
        idx = self.selected_tool_idx
        panel = self.terminals.get(self.tools[idx].key_id) if 0 <= idx < len(self.tools) else None
        if panel is None:
            return
        if btn_id == "TERM_STOP":
            if panel.is_running():
                panel.stop()
                self.set_toast("已发送停止信号, 正在终止子进程...")
            else:
                self.set_toast("终端当前没有正在运行的子进程")
        elif btn_id == "TERM_BACK":
            self.selected_tool_idx = -1
            self.hover_tool_idx = -1

    def _render_terminal_panel(self, canvas: np.ndarray, tool: ToolCardMeta, split_x: int):
        """右侧大屏终端视图: 标题栏 (状态灯 + 标题 + 停止/返回按钮) + 终端本体 + 状态行"""
        s = self.win_mgr.scale_pct / 100.0
        top_h = max(36, int(54 * s))
        header_h = max(36, int(46 * s))
        panel = self.terminals.get(tool.key_id)
        if panel is None:
            return

        # 标题栏底色与状态灯 (绿=运行中/已成功, 红=失败/已停止)
        cv2.rectangle(canvas, (split_x, top_h), (self.win_mgr.canvas_w, top_h + header_h), (17, 20, 26), -1)
        dot_col = (90, 210, 120) if panel.status_ok() else (80, 80, 240)
        dot_x = split_x + max(8, int(16 * s))
        dot_cy = top_h + header_h // 2
        cv2.circle(canvas, (dot_x, dot_cy), max(3, int(5 * s)), dot_col, -1)
        draw_text(canvas, f"嵌入式终端 | {tool.title}",
                  (dot_x + max(8, int(14 * s)), dot_cy - max(7, int(9 * s))),
                  max(10, int(14 * s)), self.COLOR_TEXT_TITLE, bold=True)

        # [停止] / [返回] 按钮 (悬停高亮, 与卡片按钮风格一致)
        self._term_btn_rects = []
        for bid, (bx1, by1, bx2, by2) in self._terminal_button_rects(split_x):
            is_hover = bx1 <= self.mouse_x <= bx2 and by1 <= self.mouse_y <= by2
            if bid == "TERM_STOP":
                enabled = panel.is_running()
                bg = (48, 22, 24) if (is_hover and enabled) else ((32, 20, 22) if enabled else (24, 26, 30))
                border = (210, 60, 60) if (is_hover and enabled) else ((95, 36, 40) if enabled else (48, 52, 60))
                txt = (220, 170, 170) if enabled else (100, 108, 120)
                label = "停止"
            else:
                bg = (48, 56, 72) if is_hover else self.COLOR_CARD_BG
                border = (0, 180, 220) if is_hover else self.COLOR_BORDER
                txt = (240, 244, 250) if is_hover else (190, 190, 200)
                label = "返回"
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), bg, -1)
            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), border, 2 if is_hover else 1)
            draw_text(canvas, label, (bx1 + (bx2 - bx1) // 2 - 14, by1 + (by2 - by1 - 14) // 2),
                      max(10, int(13 * s)), txt, bold=True)
            self._term_btn_rects.append((bid, (bx1, by1, bx2, by2)))

        # 终端本体 + 底部状态行
        body = self._terminal_body_rect(split_x)
        panel.draw(canvas, body)
        cv2.rectangle(canvas, (body[0], body[1]), (body[2], body[3]), self.COLOR_BORDER, 1)
        draw_text(canvas, panel.status_text,
                  (body[0] + max(6, int(10 * s)), body[3] + max(3, int(5 * s))),
                  max(9, int(12 * s)), (150, 170, 185))

    # ========================== 核心渲染逻辑 ==========================

    def _render_canvas(self) -> np.ndarray:
        """渲染完整画布 (在当前窗口实际物理分辨率下进行真·矢量绘制，无任何位图拉伸与锯齿)"""
        canvas = np.full((self.win_mgr.canvas_h, self.win_mgr.canvas_w, 3), self.COLOR_BG, dtype=np.uint8)

        s = self.win_mgr.scale_pct / 100.0
        CW = max(130, int(222 * s))
        SX = max(6, int(12 * s))
        X0 = max(8, int(15 * s))
        FW = CW * 2 + SX
        split_x = X0 + FW + max(8, int(15 * s))

        # 1. 顶部监控与状态栏
        self._render_top_bar(canvas)

        # 2. 中间主分割线
        top_h = max(36, int(54 * s))
        footer_h = max(34, int(50 * s))
        cv2.line(canvas, (split_x, top_h), (split_x, self.win_mgr.canvas_h - footer_h), self.COLOR_BORDER, 1)

        # 3. 左侧工具网格区
        self._render_tools_grid(canvas)

        # 4. 右侧实时说明大屏 (终端视图优先: 选中纯输出型工具时切换为嵌入式终端;
        #    其余有卡片高亮/选中时渲染 Inspector，否则渲染默认环境与硬件总览大屏)
        cur_idx = self.hover_tool_idx if self.hover_tool_idx != -1 else self.selected_tool_idx
        if self._is_terminal_view():
            self._render_terminal_panel(canvas, self.tools[self.selected_tool_idx], split_x)
        elif 0 <= cur_idx < len(self.tools):
            self._render_inspector_panel(canvas, self.tools[cur_idx], split_x)
        else:
            self._render_default_overview_panel(canvas, split_x)

        # 5. 底部状态与快捷键指引栏
        self._render_footer(canvas)

        # 6. 子工具运行挂起态：整体深度暗化并叠加控制权移交模态浮岛
        if self.is_subtool_running and self.running_tool_meta:
            self._render_suspended_modal(canvas, self.running_tool_meta)

        return canvas

    def _render_top_bar(self, canvas: np.ndarray):
        """渲染顶部全局标题 (纯净极简，无当前工况胶囊)"""
        s = self.win_mgr.scale_pct / 100.0
        top_h = max(36, int(54 * s))
        cv2.rectangle(canvas, (0, 0), (self.win_mgr.canvas_w, top_h), (17, 20, 26), -1)
        cv2.line(canvas, (0, top_h), (self.win_mgr.canvas_w, top_h), self.COLOR_BORDER, 1)

        # 标题与微光状态灯
        c_x, c_y = max(12, int(24 * s)), top_h // 2
        cv2.circle(canvas, (c_x, c_y), max(2, int(5 * s)), self.COLOR_ACCENT, -1)
        cv2.circle(canvas, (c_x, c_y), max(4, int(8 * s)), self.COLOR_ACCENT, 1)
        draw_text(canvas, "FLUX VISION 3D", (max(20, int(40 * s)), max(5, int(10 * s))),
                  font_size=max(11, int(15 * s)), color=self.COLOR_ACCENT, bold=True)
        draw_text(canvas, "芦笋上料自动化", (max(20, int(40 * s)), max(18, int(28 * s))),
                  font_size=max(9, int(12 * s)), color=(150, 170, 185))

        # 右上角退出按钮 (自适应靠右，纯“退出”两字)
        bw = max(60, int(80 * s))
        bh = max(24, int(34 * s))
        bx = self.win_mgr.canvas_w - bw - max(8, int(15 * s))
        by = max(6, int(10 * s))

        is_hover_exit = (bx <= self.mouse_x <= bx + bw and by <= self.mouse_y <= by + bh)
        exit_bg = (48, 22, 24) if is_hover_exit else (32, 20, 22)
        exit_border = (210, 60, 60) if is_hover_exit else (95, 36, 40)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_bg, -1)
        cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), exit_border, 2 if is_hover_exit else 1)

        exit_font_sz = max(11, int(14 * s))
        tx = bx + (bw - exit_font_sz * 2) // 2
        ty = by + (bh - exit_font_sz) // 2 - max(1, int(2 * s))
        draw_text(canvas, "退出", (tx, ty),
                  font_size=exit_font_sz, color=(220, 170, 170), bold=True)

    def _render_tools_grid(self, canvas: np.ndarray):
        """渲染左侧工具卡片网格 (真矢量自适应缩放，无杂乱左侧竖线)"""
        s = self.win_mgr.scale_pct / 100.0
        LH = max(14, int(20 * s))
        CW = max(130, int(222 * s))
        SX = max(6, int(12 * s))
        X0 = max(8, int(15 * s))
        FW = CW * 2 + SX

        HEADER_TEXT = (192, 206, 222)
        group_headers = [
            (self._get_card_rect(0)[1] - LH,  FW, "A  Workspace",                                 (195, 155,  45)),
            (self._get_card_rect(1)[1] - LH,  FW, "B  标定建图与生产验证 (Pipelines)",           ( 65, 175, 160)),
            (self._get_card_rect(6)[1] - LH,  FW, "D  硬件调试与系统运维 (Hardware & System)",   ( 90, 140, 195)),
        ]
        for hy, hw, label, accent in group_headers:
            cv2.rectangle(canvas, (X0, hy), (X0 + hw, hy + LH - max(1, int(2 * s))), (18, 22, 30), -1)
            cv2.rectangle(canvas, (X0, hy), (X0 + max(2, int(4 * s)), hy + LH - max(1, int(2 * s))), accent, -1)
            draw_text(canvas, label, (X0 + max(6, int(10 * s)), hy + max(2, int(3 * s))),
                      font_size=max(9, int(12 * s)), color=HEADER_TEXT, bold=False)

        # 卡片渲染
        for idx, tool in enumerate(self.tools):
            cx, cy, cw, ch = self._get_card_rect(idx)

            is_selected = (self.selected_tool_idx >= 0 and idx == self.selected_tool_idx)
            is_hover    = (idx == self.hover_tool_idx)

            if is_selected:
                card_bg, card_border, border_th = self.COLOR_CARD_SEL, self.COLOR_BORDER_SEL, 2
            elif is_hover:
                card_bg, card_border, border_th = self.COLOR_CARD_HOVER, self.COLOR_BORDER_HOVER, 1
            else:
                card_bg, card_border, border_th = self.COLOR_CARD_BG, self.COLOR_BORDER, 1

            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), card_bg, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), card_border, border_th)

            # 快捷键徽章与标题布局 (自适应有无副标题)
            badge_w = max(18, int(28 * s))
            badge_h = max(14, int(20 * s))
            title_col = self.COLOR_TEXT_TITLE if (is_selected or is_hover) else (205, 215, 225)
            has_sub = bool(tool.subtitle and tool.subtitle.strip())

            if has_sub:
                badge_y = cy + max(6, int(10 * s))
                title_y = cy + max(6, int(10 * s))
                cv2.rectangle(canvas, (cx + max(6, int(12 * s)), badge_y),
                              (cx + max(6, int(12 * s)) + badge_w, badge_y + badge_h), (16, 20, 26), -1)
                cv2.rectangle(canvas, (cx + max(6, int(12 * s)), badge_y),
                              (cx + max(6, int(12 * s)) + badge_w, badge_y + badge_h), (45, 58, 72), 1)
                draw_text(canvas, tool.shortcut, (cx + max(8, int(16 * s)), badge_y + max(1, int(2 * s))),
                          font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_TITLE, bold=True)

                draw_text(canvas, tool.title, (cx + max(10, int(18 * s)) + badge_w, title_y),
                          font_size=max(10, int(14 * s)), color=title_col, bold=True)

                sub_col = (170, 185, 195) if is_hover else self.COLOR_TEXT_MUTED
                max_sub = 52 if cw > int(500 * s) else 24
                draw_text(canvas, tool.subtitle[:max_sub], (cx + max(8, int(14 * s)), cy + max(20, int(36 * s))),
                          font_size=max(9, int(12 * s)), color=sub_col)
            else:
                # 纯标题极简模式：徽章与标题垂直居中对齐
                badge_y = cy + (ch - badge_h) // 2
                title_font_sz = max(10, int(14 * s))
                title_y = cy + (ch - title_font_sz) // 2 - max(1, int(2 * s))

                cv2.rectangle(canvas, (cx + max(6, int(12 * s)), badge_y),
                              (cx + max(6, int(12 * s)) + badge_w, badge_y + badge_h), (16, 20, 26), -1)
                cv2.rectangle(canvas, (cx + max(6, int(12 * s)), badge_y),
                              (cx + max(6, int(12 * s)) + badge_w, badge_y + badge_h), (45, 58, 72), 1)
                draw_text(canvas, tool.shortcut, (cx + max(8, int(16 * s)), badge_y + max(1, int(2 * s))),
                          font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_TITLE, bold=True)

                draw_text(canvas, tool.title, (cx + max(10, int(18 * s)) + badge_w, title_y),
                          font_size=title_font_sz, color=title_col, bold=True)

            # 运行模式微标 (三态: GUI / CMD / TERM)
            mode_color = {"GUI": (0, 190, 150), "TERM": (120, 210, 130)}.get(
                tool.mode, (135, 150, 170))     # CMD 灰
            mode_y = cy + max(14, int(24 * s)) if has_sub else cy + (ch // 2) + max(3, int(4 * s))
            put_text(canvas, tool.mode, (cx + cw - max(28, int(42 * s)), mode_y),
                     cv2.FONT_HERSHEY_SIMPLEX, max(0.24, 0.32 * s), mode_color, 1, cv2.LINE_AA)

    def _render_inspector_panel(self, canvas: np.ndarray, tool: ToolCardMeta, split_x: int):
        """渲染右侧动态即时说明大屏 (自适应全宽与全高，1:1 矢量清晰无模糊，全文本自适应折行)"""
        s = self.win_mgr.scale_pct / 100.0
        px = split_x + max(8, int(15 * s))
        py = max(40, int(66 * s))
        pw = max(int(360 * s), self.win_mgr.canvas_w - px - max(10, int(20 * s)))
        ph = max(int(450 * s), self.win_mgr.canvas_h - max(30, int(50 * s)) - py - max(8, int(15 * s)))

        # 大屏底板
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (34, 44, 58), 1)

        # 快捷键徽章与主标题（无装饰条，直接贴顶）
        b_w, b_h = max(24, int(36 * s)), max(20, int(32 * s))
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), tool.tag_color, 1)
        draw_text(canvas, f"[{tool.shortcut}]", (px + max(10, int(20 * s)), py + max(8, int(18 * s))),
                  font_size=max(10, int(14 * s)), color=tool.tag_color, bold=True)

        draw_text(canvas, tool.title, (px + max(36, int(62 * s)), py + max(4, int(10 * s))),
                  font_size=max(12, int(18 * s)), color=self.COLOR_TEXT_TITLE, bold=True)
        mode_str = {"TERM": "内嵌终端"}.get(tool.mode,
                                            "原生 GUI 视窗" if tool.is_gui else "控制台")
        draw_text(canvas, f"{tool.category}  |  {mode_str}",
                  (px + max(36, int(62 * s)), py + max(20, int(34 * s))), font_size=max(9, int(12 * s)), color=self.COLOR_TEXT_SUB)

        avail_w = pw - max(24, int(42 * s))
        text_x = px + max(10, int(20 * s))
        curr_y = py + max(44, int(64 * s))
        avail_bottom_y = py + ph - max(8, int(12 * s))

        # 1. 核心概述 (Summary) - 支持自适应折行
        draw_text(canvas, "【核心功能】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(23 * s))
        curr_y = draw_multiline_text(canvas, tool.summary, (text_x, curr_y),
                                     max_width=avail_w, font_size=max(10, int(13 * s)),
                                     color=(215, 225, 235), line_spacing=max(3, int(5 * s)))
        curr_y += max(12, int(16 * s))

        # 2. 详细特性清单 (Details) - 支持每项条目自适应折行
        draw_text(canvas, "【工程要点】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(23 * s))
        bullet_icon_x = px + max(12, int(22 * s))
        bullet_text_x = px + max(20, int(34 * s))
        bullet_w = pw - (bullet_text_x - px) - max(12, int(20 * s))

        for d in tool.details:
            if curr_y > avail_bottom_y - max(30, int(50 * s)):
                break
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 190, 160), -1)
            curr_y = draw_multiline_text(canvas, d, (bullet_text_x, curr_y),
                                         max_width=bullet_w, font_size=max(9, int(12 * s)),
                                         color=(195, 208, 220), line_spacing=max(2, int(4 * s)))
            curr_y += max(3, int(5 * s))
        curr_y += max(4, int(6 * s))

        # 3. 前置依赖与输入 (Inputs)
        if curr_y < avail_bottom_y - max(60, int(90 * s)):
            draw_text(canvas, "【输入依赖】", (px + max(8, int(16 * s)), curr_y),
                      font_size=max(10, int(13 * s)), color=(140, 180, 220), bold=True)
            curr_y += max(15, int(21 * s))
            for inp in tool.inputs:
                if curr_y > avail_bottom_y - max(40, int(60 * s)):
                    break
                cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
                curr_y = draw_multiline_text(canvas, inp, (bullet_text_x, curr_y),
                                             max_width=bullet_w, font_size=max(9, int(12 * s)),
                                             color=(185, 200, 215), line_spacing=max(2, int(4 * s)))
                curr_y += max(3, int(5 * s))
            curr_y += max(4, int(6 * s))

        # 4. 输出产物 (Outputs)
        if curr_y < avail_bottom_y - max(50, int(70 * s)):
            draw_text(canvas, "【输出产物】", (px + max(8, int(16 * s)), curr_y),
                      font_size=max(10, int(13 * s)), color=(120, 200, 180), bold=True)
            curr_y += max(15, int(21 * s))
            for out in tool.outputs:
                if curr_y > avail_bottom_y - max(30, int(45 * s)):
                    break
                cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (100, 180, 160), -1)
                curr_y = draw_multiline_text(canvas, out, (bullet_text_x, curr_y),
                                             max_width=bullet_w, font_size=max(9, int(12 * s)),
                                             color=(180, 215, 205), line_spacing=max(2, int(4 * s)))
                curr_y += max(3, int(5 * s))
            curr_y += max(4, int(6 * s))

        # 5. 操作提示 (Quick Tips) - 动态自适应卡片框
        if curr_y < avail_bottom_y - max(20, int(30 * s)):
            tip_font_size = max(9, int(12 * s))
            tip_w = pw - max(16, int(32 * s))
            tip_lines = wrap_text_by_width(tool.quick_tips, tip_font_size, tip_w - max(16, int(24 * s)))
            line_h = tip_font_size + max(2, int(4 * s))
            tip_h = len(tip_lines) * line_h + max(8, int(12 * s))

            if curr_y + tip_h <= avail_bottom_y:
                tip_box_x = px + max(8, int(16 * s))
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + tip_h), (22, 28, 36), -1)
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + tip_h), (36, 48, 62), 1)
                draw_multiline_text(canvas, tool.quick_tips, (tip_box_x + max(8, int(12 * s)), curr_y + max(4, int(6 * s))),
                                    max_width=tip_w - max(16, int(24 * s)), font_size=tip_font_size,
                                    color=self.COLOR_TEXT_SUB, line_spacing=max(2, int(4 * s)))

    def _render_default_overview_panel(self, canvas: np.ndarray, split_x: int):
        """当焦点未在任何工具卡片上时，在右侧渲染系统环境、库依赖与硬件健康状态大屏总览"""
        s = self.win_mgr.scale_pct / 100.0
        px = split_x + max(8, int(15 * s))
        py = max(40, int(66 * s))
        pw = max(int(360 * s), self.win_mgr.canvas_w - px - max(10, int(20 * s)))
        ph = max(int(450 * s), self.win_mgr.canvas_h - max(30, int(50 * s)) - py - max(8, int(15 * s)))

        # 大屏底板
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
        cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (34, 44, 58), 1)

        # 标题徽标与文字（无装饰条，直接贴顶）
        b_w, b_h = max(24, int(36 * s)), max(20, int(32 * s))
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (px + max(8, int(16 * s)), py + max(8, int(12 * s))),
                      (px + max(8, int(16 * s)) + b_w, py + max(8, int(12 * s)) + b_h), self.COLOR_ACCENT, 1)
        draw_text(canvas, "[i]", (px + max(12, int(24 * s)), py + max(8, int(18 * s))),
                  font_size=max(10, int(14 * s)), color=self.COLOR_ACCENT, bold=True)

        draw_text(canvas, "系统运行环境与硬件健康总览", (px + max(36, int(62 * s)), py + max(4, int(10 * s))),
                  font_size=max(12, int(18 * s)), color=self.COLOR_TEXT_TITLE, bold=True)

        # 数据提取
        st = self.system_status
        py_ver = sys.version.split()[0]
        cv_ver = st.get("cv_version", cv2.__version__)
        rs_tuple = st.get("realsense", (False, "未检测", False))
        rs_ok = rs_tuple[2] if len(rs_tuple) > 2 else False
        rs_msg = rs_tuple[1] if len(rs_tuple) > 1 else str(rs_tuple)
        snaps_c = st.get("snapshot_count", 0)
        act_ws = self.workspace_mgr.get_current_workspace()

        # 正文排版
        curr_y = py + max(44, int(64 * s))
        avail_bottom_y = py + ph - max(8, int(12 * s))
        bullet_icon_x = px + max(12, int(22 * s))
        bullet_text_x = px + max(22, int(36 * s))
        bullet_w = pw - (bullet_text_x - px) - max(12, int(20 * s))

        # ── 模块 1：底层环境与科学计算库依赖 ──────────────────────────────
        draw_text(canvas, "【底层运行环境与科学计算依赖】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_ACCENT, bold=True)
        curr_y += max(16, int(24 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 210, 160), -1)
        curr_y = draw_multiline_text(canvas, f"Python 解释器: v{py_ver}  ({sys.executable})",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(210, 225, 238), line_spacing=max(2, int(4 * s)))
        curr_y += max(3, int(5 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 210, 160), -1)
        curr_y = draw_multiline_text(canvas, f"核心视觉库: OpenCV v{cv_ver}  |  矩阵计算: NumPy v{np.__version__}",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(200, 215, 228), line_spacing=max(2, int(4 * s)))
        curr_y += max(3, int(5 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (0, 210, 160), -1)
        curr_y = draw_multiline_text(canvas, f"工程工作空间: {PROJECT_ROOT}",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(170, 185, 200), line_spacing=max(2, int(4 * s)))
        curr_y += max(12, int(16 * s))

        # ── 模块 2：感知硬件与数据资产 ────────────────────────────────────
        draw_text(canvas, "【感知层硬件与数据资产】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=(140, 180, 220), bold=True)
        curr_y += max(16, int(24 * s))

        rs_dot_col = (0, 210, 160) if rs_ok else (135, 145, 160)
        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), rs_dot_col, -1)
        curr_y = draw_multiline_text(canvas, f"RealSense 深度相机: {rs_msg}",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(210, 225, 240) if rs_ok else (160, 175, 185),
                                     line_spacing=max(2, int(4 * s)))
        curr_y += max(3, int(5 * s))

        cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
        curr_y = draw_multiline_text(canvas, f"现场工业快照库: 已归档 {snaps_c} 帧 (RGB+Depth+PLY，位于 data/snapshots/)",
                                     (bullet_text_x, curr_y), max_width=bullet_w,
                                     font_size=max(9, int(12 * s)), color=(185, 200, 215), line_spacing=max(2, int(4 * s)))
        curr_y += max(12, int(16 * s))

        # ── 模块 3：当前生产工位与三维标靶地图 ──────────────────────────────
        draw_text(canvas, "【当前活动工位与标靶地图资产】", (px + max(8, int(16 * s)), curr_y),
                  font_size=max(10, int(13 * s)), color=self.COLOR_GOLD, bold=True)
        curr_y += max(16, int(24 * s))

        if act_ws:
            status_tag = f"已求解平差 (RMSE: {act_ws.global_rmse_px:.2f}px)" if act_ws.ba_solved else "未平差 (草稿)"
            ws_color = (0, 210, 160) if act_ws.ba_solved else (160, 180, 200)
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), ws_color, -1)
            curr_y = draw_multiline_text(canvas, f"当前工位: 【{act_ws.name}】 ({status_tag})",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=ws_color, bold=True,
                                         line_spacing=max(2, int(4 * s)))
            curr_y += max(3, int(5 * s))

            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (180, 160, 100), -1)
            curr_y = draw_multiline_text(canvas, f"工位标定样本: 已采集 {act_ws.extrinsic_calib_image_count} 帧原始图集",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=(190, 205, 220), line_spacing=max(2, int(4 * s)))
            curr_y += max(3, int(5 * s))

            map_status = "已生成 tags_map.yaml (坐标系对齐已锁定)" if act_ws.ba_solved else "未求解 (需执行两阶段平差)"
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (180, 160, 100), -1)
            curr_y = draw_multiline_text(canvas, f"标靶地图状态: {map_status}",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=(190, 205, 220), line_spacing=max(2, int(4 * s)))
        else:
            cv2.circle(canvas, (bullet_icon_x, curr_y + max(5, int(7 * s))), max(2, int(3 * s)), (120, 130, 140), -1)
            curr_y = draw_multiline_text(canvas, "活动工况工位: 未选定工位 (请点击 [1] Workspace 新建或切换)",
                                         (bullet_text_x, curr_y), max_width=bullet_w,
                                         font_size=max(9, int(12 * s)), color=(150, 160, 170), line_spacing=max(2, int(4 * s)))
        curr_y += max(12, int(16 * s))

        # ── 模块 4：控制中心快捷操作指南 ──────────────────────────────────
        if curr_y < avail_bottom_y - max(20, int(30 * s)):
            tip_w = pw - max(16, int(32 * s))
            guide_text = "操作小贴士: 鼠标悬停左侧任意卡片即可即时查阅该模块的工程定位与执行逻辑；按键盘数字键或双击卡片直接拉起对应工具。"
            tip_font_size = max(9, int(12 * s))
            tip_lines = wrap_text_by_width(guide_text, tip_font_size, tip_w - max(16, int(24 * s)))
            line_h = tip_font_size + max(2, int(4 * s))
            box_h = len(tip_lines) * line_h + max(8, int(12 * s))
            if curr_y + box_h <= avail_bottom_y:
                tip_box_x = px + max(8, int(16 * s))
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + box_h), (22, 28, 36), -1)
                cv2.rectangle(canvas, (tip_box_x, curr_y), (tip_box_x + tip_w, curr_y + box_h), (36, 48, 62), 1)
                draw_multiline_text(canvas, guide_text, (tip_box_x + max(8, int(12 * s)), curr_y + max(4, int(6 * s))),
                                    max_width=tip_w - max(16, int(24 * s)), font_size=tip_font_size,
                                    color=self.COLOR_TEXT_SUB, line_spacing=max(2, int(4 * s)))

    def _render_footer(self, canvas: np.ndarray):
        """渲染底部状态反馈与快捷键指引栏 (自适应贴底)"""
        s = self.win_mgr.scale_pct / 100.0
        footer_h = max(34, int(50 * s))
        fy = self.win_mgr.canvas_h - footer_h
        cv2.rectangle(canvas, (0, fy), (self.win_mgr.canvas_w, self.win_mgr.canvas_h), (13, 15, 19), -1)
        cv2.line(canvas, (0, fy), (self.win_mgr.canvas_w, fy), self.COLOR_BORDER, 1)

        # 右侧公司/版权标识 (右对齐)
        f_size = max(9, int(13 * s))
        corp_text = "山东卷积分公司 · 2026 年 9 月"
        text_w = max(200, int(360 * s))
        text_x = max(int(500 * s), self.win_mgr.canvas_w - text_w)
        draw_text(canvas, corp_text, (text_x, fy + max(10, int(16 * s))),
                  font_size=f_size, color=self.COLOR_TEXT_MUTED)

    def _render_suspended_modal(self, canvas: np.ndarray, tool: ToolCardMeta):
        """当子工具/控制台在前台运行时，将主界面整体冷黑深度暗化并呈现挂起模态提示框"""
        s = self.win_mgr.scale_pct / 100.0

        # 1. 全局画面深度暗化 (降至 ~20% 亮度，呈现沉静只读休眠态)
        canvas[:] = (canvas.astype(np.float32) * 0.20).astype(np.uint8)

        # 2. 居中模态卡片几何尺寸
        cx, cy = self.win_mgr.canvas_w // 2, self.win_mgr.canvas_h // 2
        mw = max(int(460 * s), min(int(650 * s), self.win_mgr.canvas_w - 40))
        mh = max(int(170 * s), min(int(230 * s), self.win_mgr.canvas_h - 40))
        x1 = cx - mw // 2
        y1 = cy - mh // 2
        x2 = x1 + mw
        y2 = y1 + mh

        # 模态浮岛背景与外阴影/双层边框
        cv2.rectangle(canvas, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (10, 14, 20), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (20, 26, 36), -1)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 170), 2)

        # 顶栏装饰条 (冷青指示带)
        head_h = max(28, int(42 * s))
        cv2.rectangle(canvas, (x1, y1), (x2, y1 + head_h), (26, 34, 48), -1)
        cv2.line(canvas, (x1, y1 + head_h), (x2, y1 + head_h), (45, 65, 90), 1)

        # 顶栏标题
        cv2.circle(canvas, (x1 + max(12, int(20 * s)), y1 + head_h // 2), max(3, int(5 * s)), (0, 220, 180), -1)
        draw_text(canvas, "控制权已移交 · 控制中心安全待命挂起", (x1 + max(22, int(34 * s)), y1 + max(6, int(10 * s))),
                  font_size=max(11, int(15 * s)), color=(0, 230, 190), bold=True)

        # 正文内容排版
        content_y = y1 + head_h + max(12, int(16 * s))
        text_x = x1 + max(16, int(24 * s))

        # 当前运行子工具
        mode_label = "独立 GUI 视窗" if tool.is_gui else "交互式控制台"
        running_title = f"当前前台运行: 【{tool.title}】 ({mode_label})"
        draw_text(canvas, running_title, (text_x, content_y),
                  font_size=max(11, int(15 * s)), color=self.COLOR_TEXT_TITLE, bold=True)
        content_y += max(20, int(28 * s))

        # 说明项 1：安全待命
        cv2.circle(canvas, (text_x + max(4, int(6 * s)), content_y + max(6, int(8 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
        draw_text(canvas, "主视窗已进入后台只读待命模式，已自动屏蔽鼠标与键盘交互。",
                  (text_x + max(12, int(18 * s)), content_y), font_size=max(9, int(12 * s)), color=(185, 200, 215))
        content_y += max(16, int(22 * s))

        # 说明项 2：唤醒指引
        cv2.circle(canvas, (text_x + max(4, int(6 * s)), content_y + max(6, int(8 * s))), max(2, int(3 * s)), (120, 160, 200), -1)
        draw_text(canvas, "请在前台子应用中完成操作；关闭子视窗后，控制中心将自动唤醒并刷新状态。",
                  (text_x + max(12, int(18 * s)), content_y), font_size=max(9, int(12 * s)), color=(185, 200, 215))

        # 底部提示小胶囊
        bot_bar_h = max(24, int(32 * s))
        bot_y1 = y2 - bot_bar_h - max(6, int(10 * s))
        cv2.rectangle(canvas, (text_x, bot_y1), (x2 - max(16, int(24 * s)), bot_y1 + bot_bar_h), (14, 18, 24), -1)
        cv2.rectangle(canvas, (text_x, bot_y1), (x2 - max(16, int(24 * s)), bot_y1 + bot_bar_h), (35, 48, 65), 1)
        draw_text(canvas, "状态: 独占通道就绪 · 等待前台子应用退出信号...",
                  (text_x + max(10, int(16 * s)), bot_y1 + max(5, int(8 * s))),
                  font_size=max(8, int(11 * s)), color=(140, 160, 180))


def main():
    parser = argparse.ArgumentParser(description="芦笋上料自动化 (Dashboard)")
    parser.parse_args()

    app = GuiLauncherApp()
    app.run()


if __name__ == "__main__":
    main()
