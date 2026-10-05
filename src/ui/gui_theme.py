#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局 GUI 主题单源 (GuiTheme)
============================
所有 cv2 GUI 的颜色统一从此处取值，实现"一处修改，全部 GUI 生效"。

用法：
    from src.ui.gui_theme import GuiTheme
    GuiTheme.apply("dark")            # 启动时选择主题 (默认暗色工业风)
    canvas = np.full(shape, GuiTheme.BG, dtype=np.uint8)

约定：
  - 颜色一律 BGR 元组 (OpenCV 原生)；
  - 各应用 renderer 内的本地常量 (COLOR_BG / COL_PANEL 等) 应改写为本类的别名，
    保持本地命名不变、只换数据来源；
  - 应用专属功能色 (如 tracker 数据可视化 5 色、gui_launcher 分组色) 留在本地，
    但需保证在明暗两套主题下均有足够对比度。
"""

# 当前激活主题名 (模块导入时即注入默认暗色，应用可在启动前改调 apply 切换)
_ACTIVE = "dark"

# 暗色工业风 (钛黑背景 + 冰魄冷青强调) — 与 gui_launcher/d435_viewer 原始调色板一致
_DARK = {
    "BG":           (15, 17, 21),      # 全局底色
    "CARD_BG":      (22, 26, 33),      # 卡片/面板/按钮常态底色
    "CARD_HOVER":   (30, 38, 50),      # 悬停轻提亮
    "CARD_SEL":     (28, 44, 58),      # 选中/激活底色
    "BORDER":       (38, 46, 58),      # 常态描边
    "BORDER_HOVER": (0, 220, 180),     # 悬停描边
    "BORDER_SEL":   (0, 240, 200),     # 选中发光描边
    "TEXT":         (242, 245, 248),   # 主文字 (纯白冷色)
    "TEXT_SUB":     (155, 170, 185),   # 副文字 (冷银灰)
    "TEXT_MUTED":   (115, 130, 145),   # 辅助提示 (暗灰)
    "ACCENT":       (0, 210, 180),     # 主题强调色 (冰魄冷青)
    "GOLD":         (210, 175, 60),    # 关键资产点缀
    "OK":           (80, 190, 115),    # 成功/连接正常
    "WARN":         (220, 145, 60),    # 警告/执行中
    "ERR":          (210, 80, 80),     # 错误
    "BTN":          (28, 34, 44),      # 按钮底色
    "BTN_BORDER":   (52, 64, 80),      # 按钮描边
    "BTN_HOVER":    (40, 52, 66),      # 按钮悬停
    "BTN_DISABLED_BG":     (24, 29, 37),    # 按钮禁用底色 (仍是按钮外观, 不是黑洞)
    "BTN_DISABLED_BORDER": (36, 44, 56),    # 按钮禁用描边
    "BTN_TEXT":        (205, 215, 225),     # 按钮常态文字
    "BTN_TEXT_HOVER":  (240, 240, 240),     # 按钮悬停文字 (提亮白)
    "TEXT_DISABLED":   (115, 130, 145),     # 禁用文字
    "WHITE":        (240, 240, 240),   # 高亮白
    "GRAY":         (165, 165, 170),   # 中性灰
    # 3D 棱柱与标靶位姿视觉语言规范
    "PRISM_THEORY": (0, 220, 120),     # 翡翠绿 (BA 理论真值棱柱)
    "PRISM_OBS":    (240, 180, 0),     # 科技天蓝 (单帧实测感知棱柱, BGR)
    "RESIDUAL_OK":  (80, 190, 115),    # 微小残差
    "RESIDUAL_WARN":(0, 165, 255),     # 中度残差警告
    "RESIDUAL_ERR": (60, 60, 220),     # 严重残差超限
}

# 按钮 hover 行为参数 (与主题颜色无关, 全局统一; 各 GUI renderer 必须引用, 不得自行硬编码)
BTN_BEHAVIOR = {
    "HOVER_BOLD": True,    # hover 时文字加粗
    "HOVER_SCALE": 1.0,    # hover 时字号倍数 (1.0 = 不放大; 放大会导致文本溢出按钮, 慎用)
}

# 亮色主题 (白底深字，强调色加深保证对比度；值为初始标定，可在本文件统一微调)
_LIGHT = {
    "BG":           (246, 244, 240),
    "CARD_BG":      (255, 255, 255),
    "CARD_HOVER":   (240, 244, 248),
    "CARD_SEL":     (218, 233, 240),
    "BORDER":       (200, 208, 218),
    "BORDER_HOVER": (0, 150, 190),
    "BORDER_SEL":   (0, 165, 150),
    "TEXT":         (40, 44, 52),
    "TEXT_SUB":     (100, 110, 124),
    "TEXT_MUTED":   (150, 158, 170),
    "ACCENT":       (0, 150, 130),
    "GOLD":         (40, 130, 190),
    "OK":           (40, 140, 90),
    "WARN":         (30, 120, 220),
    "ERR":          (60, 60, 200),
    "BTN":          (248, 250, 252),
    "BTN_BORDER":   (185, 194, 206),
    "BTN_HOVER":    (232, 238, 244),
    "BTN_DISABLED_BG":     (238, 241, 245),   # 按钮禁用底色
    "BTN_DISABLED_BORDER": (210, 216, 224),   # 按钮禁用描边
    "BTN_TEXT":        (60, 68, 80),          # 按钮常态文字
    "BTN_TEXT_HOVER":  (20, 24, 30),          # 按钮悬停文字 (加深)
    "TEXT_DISABLED":   (170, 178, 190),       # 禁用文字
    "WHITE":        (255, 255, 255),
    "GRAY":         (120, 126, 136),
    # 3D 棱柱与标靶位姿视觉语言规范 (亮色适配)
    "PRISM_THEORY": (0, 160, 80),
    "PRISM_OBS":    (200, 140, 0),
    "RESIDUAL_OK":  (40, 140, 90),
    "RESIDUAL_WARN":(0, 120, 220),
    "RESIDUAL_ERR": (40, 40, 180),
}

_THEMES = {"dark": _DARK, "light": _LIGHT}


class GuiTheme:
    """主题访问入口：apply() 后以类属性直接取色 (GuiTheme.BG / GuiTheme.ACCENT ...)"""

    ACTIVE = _ACTIVE
    BTN_BEHAVIOR = BTN_BEHAVIOR   # 按钮 hover 行为参数 (与颜色无关, 同一单源)

    # 3D 标靶棱柱与双轨比对视图模式选项 (全局跨应用标准)
    VIEW_MODE_OPTIONS = [
        ("3d", "3D 双四棱柱对比"),
        ("2d", "2D 识别框与残差矢量"),
        ("mix", "混合透视模式"),
    ]

    BA_VIEW_OPTIONS = [
        ("3d_prism", "3D 翡翠绿四棱柱"),
        ("3d_pyramid", "3D 翡翠绿金字塔"),
        ("2d", "2D 理论投影框"),
        ("off", "隐藏 (关闭显示)"),
    ]

    OBS_VIEW_OPTIONS = [
        ("3d_prism", "3D 科技天蓝四棱柱"),
        ("3d_pyramid", "3D 科技天蓝金字塔"),
        ("2d", "2D 实测识别框"),
        ("off", "隐藏 (关闭显示)"),
    ]

    @classmethod
    def apply(cls, name: str = "dark"):
        """切换主题并注入类属性 (启动时调用一次即可)"""
        if name not in _THEMES:
            raise ValueError(f"未知主题: {name} (可选: {', '.join(_THEMES)})")
        cls.ACTIVE = name
        for key, value in _THEMES[name].items():
            setattr(cls, key, value)

    @classmethod
    def available(cls):
        """返回可选主题名列表"""
        return tuple(_THEMES)


# 模块导入即注入默认主题，保证各应用 import 后可直接取色
GuiTheme.apply(_ACTIVE)
