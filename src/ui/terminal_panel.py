#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""嵌入式终端面板 (TerminalPanel)
=================================
在 OpenCV GUI 画布内嵌一块流式终端视图, 供纯输出型命令行工具使用
(如系统环境深度诊断 / pip 依赖安装); 交互式控制台 (需要 stdin) 不适用。

核心机制:
  - start(command) 以 PIPE 启动子进程 (stderr 合并), 后台读线程流式接收输出;
  - 解析 ANSI SGR 前景色码 (30-37 / 90-97 / 0 复位), 每行拆为 (文本, BGR颜色) 片段,
    其余转义序列 (光标移动等) 直接剔除;
  - "\\r" 原地刷新语义: 丢弃未提交内容, 后续文本覆盖同一行 (兼容 pip/tqdm 进度条);
  - 环形缓冲 (默认 2000 行), 提交时按面板宽度预折行, draw() 只做可见行 blit
    (事件驱动, 无新输出时零开销);
  - 默认自动跟随底部, scroll() 上翻回看, 回到底部恢复跟随;
  - 文本渲染统一走 src/utils/text_rendering.py (PIL 掩膜缓存, 颜色仅是参数, 无逐像素额外成本)。
"""

import re
import os
import time
import codecs
import threading
import subprocess
from collections import deque

import cv2
import numpy as np

from src.utils.text_rendering import draw_text, get_cached_font
from src.utils.logger import get_logger

log = get_logger(__name__)

# SGR 前景色 → BGR (深底配色, 含明亮档 90-97)
_SGR_FG = {
    30: (110, 110, 110), 90: (150, 150, 150),          # 黑/亮黑
    31: (70, 70, 235), 91: (90, 90, 255),              # 红
    32: (90, 215, 90), 92: (130, 255, 130),            # 绿
    33: (70, 195, 250), 93: (110, 225, 255),           # 黄
    34: (250, 170, 80), 94: (255, 195, 120),           # 蓝
    35: (215, 110, 215), 95: (240, 150, 240),          # 品红
    36: (215, 215, 90), 96: (240, 240, 130),           # 青
    37: (215, 215, 215), 97: (245, 245, 245),          # 白
}
DEFAULT_FG = (205, 210, 218)
SYS_FG = (140, 150, 160)          # 系统提示行 (命令回显/退出状态)
ERR_FG = (90, 110, 255)           # 错误 (BGR 红)
OK_FG = (120, 210, 130)           # 成功 (BGR 绿)

# 项目字形规约: msyh 缺失字形替换 (✔✗⚠ → √×!), 常见 pip emoji 剔除
_GLYPH_TRANS = str.maketrans({
    "✔": "√", "✓": "√", "✅": "√", "❗": "!",
    "✗": "×", "✘": "×", "❌": "×",
    "⚠": "!",
    "⏳": "", "⏸": "", "▶": "", "⏺": "", "🎉": "", "✨": "", "🚀": "", "🔥": "", "💡": "",
})

# ANSI 转义序列 (CSI); SGR 颜色码以 'm' 结尾, 其余 (光标/清屏) 剔除
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")


class TerminalPanel:
    """嵌入式终端面板: 子进程流式输出 → ANSI 解析 → 环形缓冲 → 画布区域渲染"""

    def __init__(self, max_lines: int = 2000, font_size: int = 13):
        self.max_lines = max_lines
        self.font_size = font_size
        self.row_h = font_size + 7          # 行高 (含行距)

        self._proc = None
        self._thread = None
        self._running = False
        self._exit_code = None
        self._started_at = 0.0
        self._finished_at = 0.0
        self.command_str = ""

        self._lines = deque(maxlen=max_lines)   # 逻辑行: [(text, bgr), ...] 片段列表
        self._rows = deque(maxlen=max_lines * 3)  # 折行后的可视行 (pip 长行会拆多行)
        self._body_w = 800                      # 上次折行用的面板宽度 (draw 时更新)
        self.total_lines = 0
        self._scroll_pos = None                 # None=跟随底部; int=窗口起始可视行号
        self._pending = ""                      # 尚未以换行结束的残余输出
        self._cr_seen = False                   # CR 延迟判定标志 (区分刷新与 CRLF 行尾)

    # ------------------------------ 生命周期 ------------------------------
    def start(self, command, cwd=None):
        """启动子进程 (PIPE 合并 stderr, stdin 关闭), 后台线程流式读取"""
        self.stop_quiet()
        self._lines.clear()
        self._rows.clear()
        self._scroll_pos = None
        self._exit_code = None
        self._finished_at = 0.0
        self.total_lines = 0
        self.command_str = " ".join(str(c) for c in command)
        self._started_at = time.time()
        self._pending, self._cr_seen = "", False
        try:
            # 强制子进程 (python/pip) 以 UTF-8 无缓冲输出, 避免中文在 cp936 管道下乱码
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUNBUFFERED", "1")
            self._proc = subprocess.Popen(
                list(command), cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, shell=False)
        except Exception as e:
            self._running = False
            self._commit_line(f"[终端] 启动失败: {e}", ERR_FG)
            log.error(f"[TerminalPanel] 启动失败: {e}")
            return
        self._running = True
        self._commit_line(f"[终端] $ {self.command_str}", SYS_FG)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """终止子进程 (terminate → 超时 kill), 读线程随管道关闭自然退出"""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
            self._commit_line("[终端] 已由用户手动停止", ERR_FG)

    def stop_quiet(self):
        """无提示终止 (切换/复用面板前调用)"""
        p = self._proc
        if p is not None and p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=3)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def status_text(self) -> str:
        """面板底部状态行 (运行中 / 已完成 / 已停止 / 失败)"""
        if self.is_running():
            elapsed = time.time() - self._started_at
            return f"运行中...  {elapsed:.0f}s  |  {self.total_lines} 行"
        if self._exit_code is None:
            return "就绪  |  0 行"
        tag = "已完成" if self._exit_code == 0 else "失败" if self._exit_code > 0 else "已停止"
        dur = (self._finished_at - self._started_at) if self._finished_at else 0.0
        return f"{tag}  exit={self._exit_code}  {dur:.0f}s  |  {self.total_lines} 行"

    def status_ok(self) -> bool:
        """终态是否成功 (exit=0); 运行中返回 True 用于状态灯绿色"""
        return self._exit_code is None or self._exit_code == 0

    # ------------------------------ 滚动 ------------------------------
    def scroll(self, rows_delta: int, visible_rows: int):
        """滚轮滚动: 向上为正 (回看), 到底/向下越界恢复自动跟随"""
        if not self._rows:
            return
        base = len(self._rows) - visible_rows if self._scroll_pos is None else self._scroll_pos
        pos = max(0, base - rows_delta)
        if pos + visible_rows >= len(self._rows):
            self._scroll_pos = None       # 触底 → 恢复跟随
        else:
            self._scroll_pos = pos

    # ------------------------------ 读线程 ------------------------------
    def _reader_loop(self):
        proc = self._proc
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                self._feed(dec.decode(chunk))
            self._feed(dec.decode(b"", final=True))
        except Exception as e:
            self._commit_line(f"[终端] 读取异常: {e}", ERR_FG)
            log.warning(f"[TerminalPanel] 读取异常: {e}")
        finally:
            self._flush_pending()
            rc = proc.wait()
            self._exit_code = rc
            self._finished_at = time.time()
            self._commit_line(f"[终端] 进程退出 exit={rc} | 共 {self.total_lines} 行",
                              OK_FG if rc == 0 else ERR_FG)
            self._running = False

    def _feed(self, text: str):
        """输出流喂入 (逐 token): \\n=行结束提交; \\r 延迟判定 —
        CR 后跟内容 = 原地刷新 (丢弃 pending, 进度条覆盖语义);
        CR 后跟 LF = Windows 行结束符, 正常提交, 不产生额外空行"""
        for tok in re.split(r"([\r\n])", text):
            if tok == "\n":
                self._commit_line(self._pending)
                self._pending, self._cr_seen = "", False
            elif tok == "\r":
                self._cr_seen = True
            elif tok:
                if self._cr_seen:
                    self._pending = ""       # CR 后有新内容 → 原地刷新覆盖
                    self._cr_seen = False
                self._pending += tok

    def _flush_pending(self):
        """EOF 兜底: 提交尚未以换行结束的残余行"""
        if self._pending or self._cr_seen:
            self._commit_line(self._pending)
            self._pending, self._cr_seen = "", False

    # ------------------------------ 行处理 ------------------------------
    def _commit_line(self, raw: str, force_color=None):
        """提交一行: ANSI 解析 → 字形规约 → 按当前面板宽度折行 → 入环形缓冲"""
        self.total_lines += 1
        raw = raw.translate(_GLYPH_TRANS)
        if raw.strip() == "\x1b" or not raw:
            frags = []
        elif force_color is not None:
            frags = [(raw, force_color)] if raw.strip() else []
        else:
            frags = self._parse_ansi(raw)
        self._lines.append(frags)
        for row in self._wrap_frags(frags, self._body_w):
            self._rows.append(row)

    @staticmethod
    def _parse_ansi(raw: str):
        """解析一行原始文本为 [(text, bgr), ...]: 仅 SGR 前景色生效, 其余转义剔除。
        逐段构建 — 每段文本携带其位置处的当前颜色 (而非循环结束后的最终颜色)"""
        frags, color, pos = [], DEFAULT_FG, 0
        for m in _ANSI_RE.finditer(raw):
            if m.start() > pos:
                frags.append((raw[pos:m.start()], color))
            seq = m.group(0)
            if seq.endswith("m"):
                for c in seq[2:-1].split(";"):
                    if not c.isdigit():
                        continue
                    code = int(c)
                    if code == 0:
                        color = DEFAULT_FG
                    elif code in _SGR_FG:
                        color = _SGR_FG[code]
                    # 加粗(1)/背景色(40-47)等忽略
            pos = m.end()
        if pos < len(raw):
            frags.append((raw[pos:], color))
        return frags or ([("", color)] if raw else [])

    def _wrap_frags(self, frags, max_w: int):
        """按像素宽度对片段做字符级折行 (中文无空格, 逐字测量), 同色字符合并为文本段,
        返回可视行列表 [[(text, bgr), ...], ...]"""
        font = get_cached_font(self.font_size)
        width_cache = {}
        rows, cur, cur_w = [], [], 0

        def flush():
            if cur:
                merged, last_color, run = [], None, []
                for ch, color in cur:
                    if color != last_color and run:
                        merged.append(("".join(run), last_color))
                        run = []
                    run.append(ch)
                    last_color = color
                if run:
                    merged.append(("".join(run), last_color))
                rows.append(merged)
                cur.clear()

        for text, color in frags:
            for ch in text:
                w = width_cache.get(ch)
                if w is None:
                    w = max(1, int(round(font.getlength(ch))))
                    width_cache[ch] = w
                if cur_w + w > max_w and cur:
                    flush()
                    cur_w = 0
                cur.append((ch, color))
                cur_w += w
        flush()
        return rows

    # ------------------------------ 渲染 ------------------------------
    def draw(self, canvas: np.ndarray, rect):
        """在 rect=(x1, y1, x2, y2) 区域渲染终端 (深底 + 可视行 blit + 滚动条)"""
        x1, y1, x2, y2 = rect
        body_w = x2 - x1 - 2 * 10
        if body_w <= 0 or y2 - y1 < self.row_h:
            return
        if abs(body_w - self._body_w) > 2:      # 面板宽度变化 → 全量重折行 (低频)
            self._body_w = body_w
            self._rows.clear()
            for frags in self._lines:
                for row in self._wrap_frags(frags, body_w):
                    self._rows.append(row)

        canvas[y1:y2, x1:x2] = (18, 20, 24)

        visible = max(1, (y2 - y1 - 8) // self.row_h)
        follow = self._scroll_pos is None
        start = (max(0, len(self._rows) - visible) if follow else self._scroll_pos)
        rows = list(self._rows)
        font = get_cached_font(self.font_size)
        width_cache = {}
        font_y_base = y1 + 4
        for i in range(visible):
            ri = start + i
            if ri >= len(rows):
                break
            x = x1 + 10
            ty = font_y_base + i * self.row_h
            for txt, color in rows[ri]:
                if x > x2 - 10:
                    break
                w = width_cache.get(txt)
                if w is None:
                    w = max(1, int(round(font.getlength(txt))))
                    width_cache[txt] = w
                draw_text(canvas, txt, (x, ty), self.font_size, color)
                x += w

        # 右侧细滚动条 (可视窗口位置指示)
        if rows:
            track_y1, track_y2 = y1 + 2, y2 - 2
            frac = min(1.0, visible / len(rows))
            bar_h = max(20, int((track_y2 - track_y1) * frac))
            top = track_y1 if follow else track_y1 + int(
                (track_y2 - track_y1 - bar_h) * (start / max(1, len(self._rows) - visible)))
            cv2.rectangle(canvas, (x2 - 5, track_y1), (x2 - 3, track_y2), (40, 44, 52), -1)
            cv2.rectangle(canvas, (x2 - 5, top), (x2 - 3, top + bar_h), (90, 100, 115), -1)
