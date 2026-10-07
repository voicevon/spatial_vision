"""
技术路线 F1：冯氏寻找法 —— HSV 绿色分割 + 边缘融合 + 区域外框 + 主干轴线感知流水线
==========================================================================
(Feng's Green Axis Pipeline: HSV Green Segmentation -> Edge Fusion -> Frame Boxes -> Stem Axis)
核心原理 (A/B 分支命名: A=HSV 色彩路线, B=边缘形态路线, 双分支在步骤 3 汇合融合):
  1A. HSV 色彩空间绿色分割：Hue 色相带锁定芦笋绿色，S/V 双下限抑制灰白反光与暗影
  1B. 边缘提取：灰度 Canny 梯度阈值化提取轮廓边缘 (与 1A 并行的 B 分支)
  2A. 腐蚀与膨胀：开运算湮灭散点碎屑噪点 (数据源 1A 绿色掩膜)，小于面积阈值的连通碎块直接丢弃
  2B. 腐蚀与膨胀：闭运算弥合边缘断裂 (数据源 1B 边缘图)
  2S. 个体分离：距离变换 + 峰值种子 + minimax 泛洪 + 浅鞍合并，切开黏连的长条芦笋 (数据源 2A)
  3A. 轴线提取：逐个体骨架拓扑提取 (Zhang-Suen 细化 + 毛刺修剪 + 交叉臂配对)，一根一轴 (A 分支)
  3.  融合：A 系 2A 掩膜 ∪ B 系 2B 清理边缘，按位或并集合成
  4.  区域成形：闭运算弥合断裂与孔洞，凝聚为完整绿色连通区域
  5.  绿色区域外框提取：最小外接矩形框定每根芦笋候选，按面积与长宽比过滤
  6.  区域像素轴线拟合：对框内绿色像素云做鲁棒直线拟合，得到主干轴线
  7.  主干确定 (单层假设)：每根芦笋仅一层、不考虑叠压，一个外框唯一确定一条主干
  8.  主干中点定位：取主干轴向中点作为输出位置，X/Y 反投影解算，Z 轴恒为 0
"""

import heapq
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry


@PipelineRegistry.register("feng_green_axis", "算法 F1: 冯氏寻找法 (Feng's Green Axis)")
class FengGreenAxisPipeline(BaseAsparagusPipeline):
    """算法 F1：冯氏寻找法 —— 纯 HSV 色彩分割的单层芦笋主干定位流水线"""

    name = "算法 F1: 冯氏寻找法 (Feng's Green Axis)"
    description = "HSV 绿点分割 / Canny 边缘双分支 → 融合 → 腐蚀膨胀去散点 → 连通区域成形 → 外框/轴线/主干 → 中点定位 (Z 恒为 0)"

    # 步骤滑条声明 (Studio 通用机制): 仅激活对应步骤时在视口渲染调参滑条
    # spec 含 attr_low/attr_high 为双滑块区间 (下限/上限联动), 含 attr 为单滑块参数
    # 约定: 步骤声明 preview_<stage_key> 同名方法即可在该步骤实时预览调参效果
    STEP_SLIDERS = {
        "stage1_hsv_mask": [
            {"label": "H 色相", "attr_low": "h_low", "attr_high": "h_high", "vmin": 0, "vmax": 179},
            {"label": "S 饱和", "attr_low": "s_min", "attr_high": "s_high", "vmin": 0, "vmax": 256},
            {"label": "V 明度", "attr_low": "v_min", "attr_high": "v_high", "vmin": 0, "vmax": 256},
        ],
        "stage2a_morph": [
            {"label": "闭核", "attr": "morph_close_k", "vmin": 1, "vmax": 15},
            {"label": "开核", "attr": "morph_ksize", "vmin": 1, "vmax": 15},
            {"label": "最小面积", "attr": "min_blob_area", "vmin": 0, "vmax": 3000},
        ],
        "stage2s_split": [
            {"label": "峰值窗口", "attr": "split_peak_k", "vmin": 3, "vmax": 21},
            {"label": "最小脊半径", "attr": "split_min_ridge", "vmin": 1, "vmax": 30},
            {"label": "合并鞍深", "attr": "split_merge_saddle", "vmin": 0, "vmax": 8},
        ],
        "stage3a_axis": [
            {"label": "最小轴长", "attr": "min_axis_len", "vmin": 0, "vmax": 300},
        ],
        "stage3_fuse": [
            {"label": "Canny 下限", "attr": "canny_lo", "vmin": 0, "vmax": 500},
            {"label": "Canny 上限", "attr": "canny_hi", "vmin": 0, "vmax": 500},
        ],
    }

    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        super().__init__(fx, fy, cx, cy)
        # HSV 绿色分割阈值 (H: 绿色色相带; S/V 下限: 抑制灰白反光与暗影噪点)
        self.h_low = 35
        self.h_high = 85
        self.s_min = 40
        self.s_high = 255
        self.v_min = 40
        self.v_high = 255
        # 外框与主干过滤参数
        self.min_area_px = 400        # 最小区域像素面积 (滤除碎屑噪块)
        self.min_aspect = 1.5         # 最小长宽比 (保证细长杆状特征)
        self.min_length_mm = 60.0     # 主干最小物理长度
        self.max_length_mm = 600.0    # 主干最大物理长度
        # 步骤 2A 腐蚀与膨胀参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.morph_close_k = 7        # 闭核尺寸: 先弥合 HSV 茎干断裂保住完整长条 (奇数化)
        self.morph_ksize = 5          # 开核尺寸: 后去散点碎屑 (使用时归一化为奇数)
        self.min_blob_area = 120      # 小面积连通碎块丢弃阈值 px (0=不丢弃)
        # 步骤 2S 个体分离参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.split_peak_k = 7         # 距离变换峰值检测窗口 (使用时奇数化)
        self.split_min_ridge = 3      # 脊峰最小半径 px (低于此的噪声峰忽略)
        self.split_merge_saddle = 2.0  # 浅鞍合并阈值 px (同脊噪声鼓包鞍深~0.5, 真黏连颈~4+, 0=不合并)
        # 步骤 3A 轴线提取参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.min_axis_len = 80.0      # 轴线准入门槛 px (碎渣短轴不输出不显示, 0=不过滤)
        # 步骤 3 融合 B 分支 Canny 参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.canny_lo = 60            # Canny 低阈值 (滞后下限)
        self.canny_hi = 160           # Canny 高阈值 (滞后上限)

    def _stage2a_morph(self, green_mask: np.ndarray):
        """步骤 2A 处理核: 闭运算弥合茎干断裂 -> 开运算去散点 -> 小面积连通碎块丢弃
        返回 (掩膜, 被丢弃碎块数)。芦笋茎干在 HSV 掩膜上因高光/阴影天然断裂,
        必须先 CLOSE 接缝保住完整长条, 再 OPEN 清散点; 纯 OPEN 会把窄条腐蚀碎裂。"""
        k = max(1, int(self.morph_ksize)) | 1          # 开核 (奇数)
        kc = max(1, int(self.morph_close_k)) | 1       # 闭核 (奇数)
        mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        removed = 0
        if self.min_blob_area > 0:
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            mask = np.zeros_like(mask)
            for lbl in range(1, n_labels):
                if stats[lbl, cv2.CC_STAT_AREA] >= self.min_blob_area:
                    mask[labels == lbl] = 255
                else:
                    removed += 1
        return mask, removed

    def preview_stage2a_morph(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 2A 腐蚀与膨胀调参实时预览: 按滑条当前核尺寸与面积阈值叠加掩膜高亮"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        mask, _ = self._stage2a_morph(green_mask)
        vis = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis[mask > 0] = (80, 240, 120)
        put_text(vis, f"2A TUNING  k={max(1, int(self.morph_ksize)) | 1}  min_area={int(self.min_blob_area)}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2)
        return vis

    def _stage1b_edges(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 1B 处理核: 灰度 Canny 边缘提取 (B 分支数据源)"""
        return cv2.Canny(cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY),
                         int(self.canny_lo), int(self.canny_hi))

    def _stage2b_edge_morph(self, edges: np.ndarray) -> np.ndarray:
        """步骤 2B 处理核: 边缘图闭运算弥合断裂 (先膨胀再腐蚀细化, 开运算会擦除 1px 细线)"""
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k3)

    def preview_stage3_fuse(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 3 融合调参实时预览: A 系掩膜 (绿) 与 B 系边缘 (黄) 并集合成的双分支汇合效果"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        mask_a, _ = self._stage2a_morph(green_mask)
        edges_b = self._stage2b_edge_morph(self._stage1b_edges(color_bgr))
        mask_fused = cv2.bitwise_or(mask_a, edges_b)
        vis = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis[mask_a > 0] = (80, 240, 120)
        vis[(edges_b > 0) & (mask_a == 0)] = (250, 230, 90)
        put_text(vis, f"3 FUSE TUNING  canny {int(self.canny_lo)}/{int(self.canny_hi)}  "
                      f"Fused px: {int(cv2.countNonZero(mask_fused))}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 255), 2)
        return vis

    def preview_stage3a_axis(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 3A 轴线提取调参实时预览: 重算 2A->2S->3A 链, 按最小轴长门槛绘制准入轴线"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        mask_a, _ = self._stage2a_morph(green_mask)
        instances, _ = self._stage2s_split(mask_a)
        axes = self._stage3a_axis(instances)
        vis = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis[mask_a > 0] = (vis[mask_a > 0] * 0.5
                           + np.array((40, 120, 60), dtype=np.float32)).astype(np.uint8)
        for i, ax in enumerate(axes):
            p1 = (int(ax["p1"][0]), int(ax["p1"][1]))
            p2 = (int(ax["p2"][0]), int(ax["p2"][1]))
            cv2.line(vis, p1, p2, (255, 255, 255), 2)
            cx, cy = int(ax["centroid"][0]), int(ax["centroid"][1])
            cv2.circle(vis, (cx, cy), 4, (255, 200, 40), -1)
        put_text(vis, f"3A TUNING  min_len={self.min_axis_len:.0f}px  Axes: {len(axes)}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 40), 2)
        return vis

    def _stage2s_split(self, mask_open: np.ndarray):
        """步骤 2S 处理核: 距离变换峰值种子 -> minimax 泛洪竞争 -> 浅鞍并查集合并, 切开黏连长条
        返回 (实例标签图 instances: 0=背景, 1..n=个体, 个体数)"""
        fg = mask_open > 0
        if not fg.any():
            return np.zeros(mask_open.shape, np.int32), 0
        dist = cv2.distanceTransform(mask_open, cv2.DIST_L2, 5)
        # 高斯平滑: 消除宽度波动与边缘锯齿导致的脊线断裂 (否则一根会碎成多个峰簇 → 过分割)
        dist = cv2.GaussianBlur(dist, (15, 15), 0)
        k = max(3, int(self.split_peak_k)) | 1
        peak = (cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) == dist) \
            & (dist >= float(self.split_min_ridge))
        # 峰簇面积过滤: 原始峰点聚簇, 面积过小的杂簇丢弃 (锯齿/噪点伪峰), 再膨胀聚合成种子
        n_raw, raw_lbl = cv2.connectedComponents(peak.astype(np.uint8), connectivity=8)
        peak_keep = np.zeros(peak.shape, np.uint8)
        for lbl in range(1, n_raw):
            if int((raw_lbl == lbl).sum()) >= 20:
                peak_keep[raw_lbl == lbl] = 255
        if not peak_keep.any():   # 无有效峰: 退化为普通连通域
            n, labels = cv2.connectedComponents(mask_open, connectivity=8)
            return labels, n - 1
        seeds = cv2.dilate(peak_keep,
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        n_seeds, seed_lbl = cv2.connectedComponents(seeds, connectivity=8)
        if n_seeds <= 2:   # 背景 + 至多 1 簇种子: 无黏连, 退化为普通连通域
            n, labels = cv2.connectedComponents(mask_open, connectivity=8)
            return labels, n - 1
        dmax = float(dist.max())
        feed = np.full(dist.shape, 255.0, np.float32)   # 背景为最高海拔墙 (不参与泛洪)
        if dmax > 0:
            feed[fg] = (1.0 - dist[fg] / dmax) * 255.0  # 分水岭地形: 脊(高dist)低海拔先泛洪
        zones = self._flood_minimax(feed, seed_lbl, n_seeds - 1, fg)
        # 浅鞍合并: 同一条脊线被噪声鼓包分隔出的多个种子, 泛洪碰撞鞍深极浅 (约 0.5px);
        # 真黏连颈缩鞍深显著更深 (约 4px+), 据此并查集合并回同一根, 同时保留真分离
        zones = self._merge_shallow_saddles(zones, feed, dmax, float(self.split_merge_saddle))
        return zones, int(zones.max())

    @staticmethod
    def _merge_shallow_saddles(zones: np.ndarray, feed: np.ndarray, dmax: float,
                               saddle_px: float) -> np.ndarray:
        """泛洪区域浅鞍合并: 相邻区域接触边最高 dist 即鞍部海拔, 其低于两侧区域脊顶的
        深度 = min(脊顶dist) - 鞍部dist, 浅于 saddle_px (同一根脊线被噪声鼓包分隔的碎片)
        则并查集合并回同一根; 真黏连颈缩深度显著更深, 保留分离。saddle_px <= 0 时不合并。"""
        if saddle_px <= 0 or zones.max() < 2:
            return zones
        parent: Dict[int, int] = {}

        def find(i: int) -> int:
            parent.setdefault(i, i)
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # 每区域脊顶海拔 = 区域内最高 dist = (1 - 区域内最低 feed / 255) * dmax
        flat_z, flat_f = zones.ravel(), feed.ravel()
        m = flat_z > 0
        order = np.lexsort((flat_f[m], flat_z[m]))
        z_s, f_s = flat_z[m][order], flat_f[m][order]
        first = np.ones(len(z_s), bool)
        first[1:] = z_s[1:] != z_s[:-1]
        zone_min_feed = np.full(int(zones.max()) + 1, 255.0, np.float32)
        zone_min_feed[z_s[first]] = f_s[first]
        zone_peak = (1.0 - zone_min_feed / 255.0) * dmax

        # 扫描 4 邻域接缝, 记录每对相邻区域的接触边最高 dist (鞍部海拔)
        saddle_feed: Dict[tuple, float] = {}
        for a, b, f in ((zones[:-1, :], zones[1:, :], feed[:-1, :]),
                        (zones[:, :-1], zones[:, 1:], feed[:, :-1])):
            m2 = (a > 0) & (b > 0) & (a != b)
            if not m2.any():
                continue
            lo = np.minimum(a, b)[m2].astype(np.int64)
            hi = np.maximum(a, b)[m2].astype(np.int64)
            fv = f[m2]
            order = np.lexsort((fv, hi, lo))
            lo_s, hi_s, fv_s = lo[order], hi[order], fv[order]
            first = np.ones(len(lo_s), bool)
            first[1:] = (lo_s[1:] != lo_s[:-1]) | (hi_s[1:] != hi_s[:-1])
            for L, H, F in zip(lo_s[first].tolist(), hi_s[first].tolist(), fv_s[first].tolist()):
                saddle_feed[(L, H)] = F
        for (za, zb), fmin in saddle_feed.items():
            saddle_dist = (1.0 - fmin / 255.0) * dmax
            depth = min(zone_peak[za], zone_peak[zb]) - saddle_dist
            if depth < saddle_px:
                ra, rb = find(za), find(zb)
                if ra != rb:
                    parent[ra] = rb
        # 重编号: 并查集根 -> 1..n 连续标签
        lut = np.zeros(int(zones.max()) + 1, np.int32)
        roots: Dict[int, int] = {}
        nxt = 0
        for lbl in range(1, int(zones.max()) + 1):
            r = find(lbl)
            if r not in roots:
                nxt += 1
                roots[r] = nxt
            lut[lbl] = roots[r]
        return lut[zones]

    @staticmethod
    def _flood_minimax(feed: np.ndarray, seed_lbl: np.ndarray, n_seeds: int,
                       fg: np.ndarray) -> np.ndarray:
        """最小最大路径 (minimax) 多源优先队列泛洪 —— 教科书式分水岭定义:
        每个前景像素归属"路径最高海拔最低"的种子, 泛洪边界即黏连鞍部。
        不用 cv2.watershed: 其泛洪次序在细颈+薄桥条拓扑下不确定
        (低地泛洪可提前穿越高海拔桥条, 把邻个体两翼抢走导致边界错位)。"""
        h, w = feed.shape
        out = np.zeros((h, w), np.int32)
        heap: List[tuple] = []
        for lbl in range(1, n_seeds + 1):
            ys, xs = np.nonzero((seed_lbl == lbl) & fg)
            for y, x in zip(ys.tolist(), xs.tolist()):
                out[y, x] = lbl
                heap.append((feed[y, x], lbl, y, x))
        heapq.heapify(heap)
        while heap:
            cost, lbl, y, x = heapq.heappop(heap)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and fg[ny, nx] and out[ny, nx] == 0:
                    out[ny, nx] = lbl
                    ncost = cost if cost >= feed[ny, nx] else feed[ny, nx]
                    heapq.heappush(heap, (ncost, lbl, ny, nx))
        return out

    # ---------------- 3A: 骨架拓扑轴线提取 (细化 -> 毛刺修剪 -> 交叉臂配对) ----------------
    @staticmethod
    def _skeletonize_zs(mask01: np.ndarray) -> np.ndarray:
        """Zhang-Suen 矢量化细化: 0/1 uint8 掩膜 -> 单像素宽 8 连通骨架"""
        img = np.pad((mask01 > 0).astype(np.uint8), 1)
        while True:
            changed = False
            for step in (0, 1):
                c = img[1:-1, 1:-1]
                p2, p3, p4 = img[:-2, 1:-1], img[:-2, 2:], img[1:-1, 2:]
                p5, p6, p7 = img[2:, 2:], img[2:, 1:-1], img[2:, :-2]
                p8, p9 = img[1:-1, :-2], img[:-2, :-2]
                seq = (p2, p3, p4, p5, p6, p7, p8, p9)
                a = np.zeros(c.shape, np.uint8)
                for i in range(8):
                    a += ((seq[i] == 0) & (seq[(i + 1) % 8] == 1)).astype(np.uint8)
                b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
                if step == 0:
                    cond = (p2 * p4 * p6 == 0) & (p4 * p6 * p8 == 0)
                else:
                    cond = (p2 * p4 * p8 == 0) & (p2 * p6 * p8 == 0)
                rm = (c == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
                if rm.any():
                    img[1:-1, 1:-1][rm] = 0
                    changed = True
            if not changed:
                break
        return img[1:-1, 1:-1]

    @staticmethod
    def _topo_junctions(skel: np.ndarray) -> np.ndarray:
        """骨架交叉点: 环邻域 0->1 跳变数 A >= 3 (比对邻域计数更抗对角锯齿误判)"""
        p = np.pad(skel, 1)
        seq = (p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:],
               p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2])
        a = np.zeros(skel.shape, np.uint8)
        for i in range(8):
            a += ((seq[i] == 0) & (seq[(i + 1) % 8] == 1)).astype(np.uint8)
        return (skel == 1) & (a >= 3)

    @staticmethod
    def _axis_from_pts(pts: np.ndarray) -> Dict:
        """像素云 PCA 主方向拟合, 端点取轴向投影极值 (弯曲轴的 p1/p2 即两端笋尖)"""
        mean_pt = pts.mean(axis=0)
        _, _, vt = np.linalg.svd((pts - mean_pt).astype(np.float32), full_matrices=False)
        v = vt[0] / (float(np.linalg.norm(vt[0])) + 1e-9)
        if v[0] < 0:
            v = -v
        proj = (pts - mean_pt) @ v
        return {
            "axis": (float(v[0]), float(v[1])),
            "p1": mean_pt + float(proj.min()) * v,
            "p2": mean_pt + float(proj.max()) * v,
            "len_px": float(proj.max() - proj.min()),
            "centroid": (float(mean_pt[0]), float(mean_pt[1])),
        }

    def _mk_axis(self, pts: np.ndarray, ox: int, oy: int, area_px: int) -> Dict:
        """组内骨架点 PCA 拟合并偏移回全图坐标"""
        ax = self._axis_from_pts(pts)
        ax["p1"] = ax["p1"] + np.array([ox, oy], dtype=np.float32)
        ax["p2"] = ax["p2"] + np.array([ox, oy], dtype=np.float32)
        ax["centroid"] = (ax["centroid"][0] + ox, ax["centroid"][1] + oy)
        ax["area_px"] = int(area_px)
        return ax

    def _skeleton_groups(self, crop: np.ndarray) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """骨架拓扑分组核心 (F1 逐组 PCA 与 F2 中心线追踪共用): Zhang-Suen 细化 -> 小支路毛刺修剪
        -> 交叉臂共线连续性配对 (并查集), 返回 (每组骨架点掩膜列表, 交叉点掩膜);
        组掩膜不含交叉点像素 (F2 按组洪泛归还相邻交叉点以桥接配对臂, F1 逐组 PCA 无需断缝连通);
        骨架退化 (极小碎域) 返回 None, 由调用方决定退化行为。
        弯曲芦笋骨架为单条路径不再拆段, 天然一根一轴; 交叉黏连簇在交叉点按方向连续性两两配对"""
        skel = self._skeletonize_zs(crop)
        ys, xs = np.nonzero(skel)
        if len(xs) < 10:   # 骨架退化 (极小碎域): 交由调用方回退
            return None

        # 毛刺修剪: 交叉点处的小支路 (锯齿/叶柄伪分支), 最多 3 轮
        # 交叉点必须 3x3 扩张后再切割: 仅删交叉中心像素时 8 连通对角捷径仍贯通, 支路断不开
        skel_img = skel.copy()
        k3 = np.ones((3, 3), np.uint8)
        for _ in range(3):
            junctions = self._topo_junctions(skel_img)
            if not junctions.any():
                break
            cut = cv2.dilate(junctions.astype(np.uint8), k3) > 0
            br_img = skel_img.copy()
            br_img[cut] = 0
            n_br, br_lbl = cv2.connectedComponents(br_img, connectivity=8)
            if n_br <= 1:
                break
            near_j = cv2.dilate(cut.astype(np.uint8), k3) > 0
            sizes = [int((br_lbl == i).sum()) for i in range(1, n_br)]
            spur_max = max(12, int(max(sizes) * 0.12))
            pruned = False
            for i in range(1, n_br):
                if sizes[i - 1] <= spur_max and ((br_lbl == i) & near_j).any():
                    skel_img[br_lbl == i] = 0
                    pruned = True
            if not pruned:
                break

        # 终态拓扑分解: 交叉点 3x3 扩张切割后每段支路是简单路径 (返回切割区掩膜供 F2 洪泛桥接)
        junctions = self._topo_junctions(skel_img)
        cut = cv2.dilate(junctions.astype(np.uint8), k3) > 0
        br_img = skel_img.copy()
        br_img[cut] = 0
        n_br, br_lbl = cv2.connectedComponents(br_img, connectivity=8)
        if n_br <= 1:   # 全为交叉点 (极小骨架): 整条骨架为单组
            return [skel_img > 0], cut

        parent = list(range(n_br))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        # 交叉臂配对: 交叉簇处入射臂指向点积最负 (转向最小) 优先结合, 夹角 > ~110 度才接
        n_j, j_lbl = cv2.connectedComponents(junctions.astype(np.uint8), connectivity=8)
        for j in range(1, n_j):
            jm = j_lbl == j
            near = cv2.dilate(jm.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            jc = np.argwhere(jm).mean(axis=0)
            ends = []
            for i in range(1, n_br):
                bm = br_lbl == i
                ep = np.argwhere(bm & near)
                if len(ep) == 0:
                    continue
                e = ep[int(((ep - jc) ** 2).sum(axis=1).argmin())]
                fp = np.argwhere(bm & ~near)
                local = fp[((fp - e) ** 2).sum(axis=1) < 36] if len(fp) else fp
                if len(local) == 0:
                    local = ep
                u = local.mean(axis=0) - e
                nu = float(np.linalg.norm(u))
                ends.append((i, u / nu if nu > 1e-6 else np.array([1.0, 0.0])))
            pairs = sorted((ends[a_i][1] @ ends[b_i][1], ends[a_i][0], ends[b_i][0])
                           for a_i in range(len(ends)) for b_i in range(a_i + 1, len(ends)))
            used = set()
            for dotv, ia, ib in pairs:
                if dotv > -0.35:
                    break
                if ia in used or ib in used:
                    continue
                parent[find(ia)] = find(ib)
                used.update((ia, ib))

        # 配对分支并成组 (每组为一条轴/中心线的骨架点集, 不含交叉点像素)
        groups: Dict[int, np.ndarray] = {}
        for i in range(1, n_br):
            r = find(i)
            groups[r] = groups.get(r, np.zeros_like(skel, bool)) | (br_lbl == i)
        return list(groups.values()), cut

    def _instance_axes(self, crop: np.ndarray, ox: int, oy: int, area_px: int) -> List[Dict]:
        """单实例骨架轴线提取: 骨架拓扑分组 (_skeleton_groups) 后逐组 PCA 拟合一条轴 (面积均摊)"""
        result = self._skeleton_groups(crop)
        if result is None:   # 骨架退化 (极小碎域): 回退整体 PCA
            my, mx = np.nonzero(crop)
            if len(mx) < 30:
                return []
            return [self._mk_axis(np.column_stack((mx, my)).astype(np.float32), ox, oy, area_px)]
        group_masks, _junctions = result
        per_area = area_px // max(1, len(group_masks))
        axes: List[Dict] = []
        for gm in group_masks:
            gy, gx = np.nonzero(gm)
            axes.append(self._mk_axis(np.column_stack((gx, gy)).astype(np.float32), ox, oy, per_area))
        return axes

    def _stage3a_axis(self, instances: np.ndarray) -> List[Dict]:
        """步骤 3A 处理核: 逐实例骨架拓扑提取轴线 (一根一轴, 弯曲不拆段, 交叉按共线配对)
        短于 min_axis_len 的碎渣轴不输出不显示 (碎域像素仍保留供步骤 3 融合使用)"""
        axes: List[Dict] = []
        for lbl in range(1, int(instances.max()) + 1):
            inst_mask = (instances == lbl).astype(np.uint8)
            area = int(inst_mask.sum())
            if area < 30:          # 像素过少的碎域不参与拟合
                continue
            ys, xs = np.nonzero(inst_mask)
            y0, x0 = int(ys.min()), int(xs.min())
            crop = inst_mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            axes.extend(self._instance_axes(crop, x0, y0, area))
        axes = [a for a in axes if a["len_px"] >= float(self.min_axis_len)]   # 碎渣轴准入门槛
        axes.sort(key=lambda a: a["len_px"], reverse=True)   # 按轴线长度降序, 长轴优先编号
        return axes

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep(
                "stage1_hsv_mask", "1A.HSV绿分割",
                "BGR 转 HSV 后按绿色色相带提取绿点二值掩膜",
                details="Hue 通道将色彩与亮度解耦，35~85 的绿色色相带可稳定锁定芦笋茎干绿色，"
                        "S/V 双下限剔除传送带灰白反光与暗影低饱和噪点，彻底免疫底色干扰",
                parameters="h_low=35, h_high=85 (色相带，偏黄可下调至 30); s_min=40, v_min=40 (饱和度/明度门限)",
                pros_cons="优点: 速度极快 (<1ms) 且对底色变化天然免疫; 缺点: 强黄色灯光或蓝青色物料可能串色"
            ),
            PipelineStep(
                "stage2a_morph", "2A.腐蚀与膨胀",
                "闭运算弥合茎干断裂 + 开运算去散点碎屑 (数据源 1A)",
                details="芦笋茎干在 HSV 掩膜上因高光/阴影天然断裂，先做闭运算 (膨胀再腐蚀) 接缝保住完整长条，"
                        "再做开运算剔除孤立散点与颗粒碎屑，小于面积阈值的连通碎块直接丢弃；"
                        "纯开运算会把窄长条腐蚀碎裂，先闭后开是该步骤的关键次序",
                parameters="morph_close_k=7 (闭核, 先弥合断裂); morph_ksize=5 (开核, 后去散点); "
                           "min_blob_area=120 (碎块丢弃阈值 px, 0=不丢弃) —— 均为 2A 滑条可调",
                pros_cons="优点: 保住完整长条同时清除碎屑; 缺点: 闭核过大会黏连相邻贴合芦笋"
            ),
            PipelineStep(
                "stage2s_split", "2S.个体分离",
                "距离变换峰值种子 + minimax 泛洪竞争 + 浅鞍合并，切开黏连在一起的长条芦笋 (数据源 2A)",
                details="对 2A 掩膜做距离变换：每根长条的中心脊是局部峰值，黏连颈缩处是低谷；"
                        "峰值点聚合成种子簇后做 minimax 泛洪竞争 (每像素归属路径最高海拔最低的种子)，"
                        "相邻区域接触边即鞍部，鞍部低于两侧脊顶的深度浅于合并阈值 "
                        "(同一根被噪声鼓包分隔的碎片) 用并查集合并回同一根，"
                        "真黏连颈缩深度大则保留分离；"
                        "单根长条只有一个脊簇不会被切开，并排贴合的双条则获得两个种子被正确分离，"
                        "输出实例标签图供 3A 逐个体提取轴线",
                parameters="split_peak_k=7 (峰值检测窗口, 步骤 2S 滑条 3~21); "
                           "split_min_ridge=3 (脊峰最小半径 px, 低于此的噪声峰忽略); "
                           "split_merge_saddle=2.0 (浅鞍合并阈值 px, 0=不合并)",
                pros_cons="优点: 利用长条宽度特性自动分离黏连, 无需人工框选; 缺点: 交叉(X型)黏连脊线连通时保守不分离"
            ),
            PipelineStep(
                "stage3a_axis", "3A.轴线提取",
                "逐个体骨架拓扑提取轴线，一根一轴 (弯曲不拆段, 交叉共线配对)",
                details="对 2S 每个独立个体做 Zhang-Suen 细化得到单像素骨架 (即芦笋中轴线)："
                        "弯曲芦笋的骨架是单条路径，天然一根一轴不再被拆段；"
                        "骨架毛刺 (锯齿/叶柄伪分支) 按长度自动修剪；"
                        "真交叉黏连簇在交叉点按臂方向共线连续性配对 (并查集)，两两重组还原为各根轴线；"
                        "短于准入门槛的碎渣轴不输出不显示，轴线按长度降序编号 (A1/A2/...)",
                parameters="min_axis_len=80 (轴线准入门槛 px, 碎渣轴不输出, 3A 滑条 0~300); "
                           "毛刺修剪阈 max(12, 0.12×最长支路) px; 配对夹角阈 ~110°",
                pros_cons="优点: 弯曲/交叉场景一根一轴, 拓扑级鲁棒; 缺点: 密集多交叉簇配对可能保守合并"
            ),
            PipelineStep(
                "stage3_fuse", "3.融合",
                "A 系 2A 掩膜与 B 系 2B 清理边缘按位或并集，双特征分支在此汇合",
                details="将 A 系色彩路线的绿色掩膜与 B 系边缘路线的清理边缘做 bitwise_or 并集融合："
                        "色彩区域提供主体像素云，边缘分支补回 HSV 漏检的弱色茎干轮廓，"
                        "融合掩膜同时具备色彩覆盖率与边缘完整性",
                parameters="canny_lo=60, canny_hi=160 (B 分支 Canny 滞后阈值, 步骤 3 滑条 0~500)",
                pros_cons="优点: 双分支互补，HSV 漏检的弱色茎干可由边缘补回; 缺点: 强纹理背景会引入边缘噪声碎块"
            ),
            PipelineStep(
                "stage4_region", "4.区域成形",
                "闭运算弥合断裂，形成完整绿色连通区域",
                details="对 3 融合输出的掩膜做闭运算 (先膨胀后腐蚀)：弥合茎干上的高光断裂与叶隙孔洞，"
                        "使每根芦笋凝聚为一个致密连通区域",
                parameters="ksize=(5,5) 椭圆结构元 (断裂严重时可调大至 (7,7) 增强弥合能力)",
                pros_cons="优点: 区域致密完整，为外框与轴线提供高质量像素云; 缺点: 结构元过大会粘连相邻贴合芦笋"
            ),
            PipelineStep(
                "stage5_boxes", "5.外框提取",
                "连通域最小外接矩形框定每根芦笋候选外框",
                details="对每个绿色连通区域提取最小外接旋转矩形，按像素面积与长宽比双门限过滤，"
                        "保留细长杆状外框作为单根芦笋候选，碎屑圆斑与大面积粘连块被剔除",
                parameters="min_area_px=400 (最小面积); min_aspect=1.5 (最小长宽比，并排粘连时可上调)",
                pros_cons="优点: 每根芦笋一个外框，直观且稳定; 缺点: 多根紧密并排粘连时可能并入同一外框"
            ),
            PipelineStep(
                "stage6_axis", "6.轴线拟合",
                "框内绿色像素云鲁棒直线拟合得到主干轴线",
                details="对每个外框内的全部绿色像素做 cv2.fitLine (L2 鲁棒) 直线拟合，"
                        "最小二乘主方向即为主干轴线方向，轴向投影极值确定线段端点",
                parameters="distType=L2 (最小二乘); 拟合点数下限=30 (像素过少的框直接淘汰)",
                pros_cons="优点: 亚像素级轴向精度，对边缘锯齿不敏感; 缺点: 弯曲芦笋需折线分段才能精确贴合"
            ),
            PipelineStep(
                "stage7_stem", "7.主干确定",
                "单层假设下每个外框唯一确定一条主干",
                details="假定所有芦笋仅有一层、不考虑叠压遮挡，因此无需拓扑剥层仲裁："
                        "每个有效外框的拟合轴线直接作为该芦笋的主干，按物理长度门限最终确认",
                parameters="min_length_mm=60, max_length_mm=600 (物理长度门限，标称深度平面换算)",
                pros_cons="优点: 流程极简，无分层仲裁开销; 缺点: 严重叠压场景需升级至 B1 的剥层方案"
            ),
            PipelineStep(
                "stage8_midpoint", "8.中点定位",
                "主干轴向中点作为输出位置，X/Y 反投影解算且 Z 恒为 0",
                details="在主干轴线上取轴向投影中点作为芦笋中心位置，按标称深度反投影解算相机系 X/Y 坐标；"
                        "Z 轴按 F1 路线约定恒为 0 (纯 2D 平面定位，不消费深度图)",
                parameters="Z 恒为 0 (F1 路线约定); X/Y 反投影采用标称深度 nominal_z_mm 缩放",
                pros_cons="优点: 输出即主干中点，位置稳定不受笋尖朝向影响; 缺点: 无真实高度信息，抓取高度需上层规划"
            )
        ]

    def run(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated",
        nominal_z_mm: float = 640.0
    ) -> PipelineResult:
        t0 = time.perf_counter()
        h, w = color_bgr.shape[:2]
        step_images: Dict[str, Optional[np.ndarray]] = {}
        scale_2d = nominal_z_mm / self.fx   # 像素 -> mm (标称深度平面)

        # ---------------- 步骤 1: HSV 绿色分割 ----------------
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv,
                                 (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        green_pixels = int(cv2.countNonZero(green_mask))

        vis_1 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_1[green_mask > 0] = (80, 240, 120)
        self._draw_header(vis_1, f"STAGE 1: HSV GREEN MASK (H {self.h_low}~{self.h_high}) | Px: {green_pixels}",
                          (80, 240, 120))
        step_images["stage1_hsv_mask"] = vis_1

        # ---------------- 步骤 2A: 腐蚀与膨胀 (开运算去散点 + 小面积碎块丢弃, 数据源 1A 绿色掩膜) ----------------
        mask_open, removed_blobs = self._stage2a_morph(green_mask)

        vis_2a = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_2a[mask_open > 0] = (80, 240, 120)
        self._draw_header(vis_2a, f"STAGE 2A: MORPH CLOSE {max(1, int(self.morph_close_k)) | 1}"
                                  f" + OPEN {max(1, int(self.morph_ksize)) | 1}"
                                  f" | < {int(self.min_blob_area)}px Rejected: {removed_blobs}", (0, 220, 120))
        step_images["stage2a_morph"] = vis_2a

        palette = [(255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120), (255, 210, 40),
                   (180, 160, 255), (120, 255, 255), (255, 150, 150)]

        # ---------------- 步骤 2S: 个体分离 (距离变换峰值种子 + 分水岭, 切开黏连长条, 数据源 2A) ----------------
        instances, n_split = self._stage2s_split(mask_open)

        vis_2s = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        for lbl in range(1, n_split + 1):
            region = instances == lbl
            vis_2s[region] = (vis_2s[region] * 0.45
                              + np.array(palette[(lbl - 1) % len(palette)], dtype=np.float32) * 0.55
                              ).astype(np.uint8)
        self._draw_header(vis_2s, f"STAGE 2S: INSTANCE SPLIT (Watershed) | Individuals: {n_split}",
                          (255, 200, 40))
        step_images["stage2s_split"] = vis_2s

        # ---------------- 步骤 3A: 轴线提取 (A 分支, 数据源 2S 个体实例, 逐个体鲁棒直线拟合) ----------------
        axes_a = self._stage3a_axis(instances)

        vis_3a = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis_3a[mask_open > 0] = (vis_3a[mask_open > 0] * 0.5
                                 + np.array((40, 120, 60), dtype=np.float32)).astype(np.uint8)
        for i, ax in enumerate(axes_a):
            p1 = (int(ax["p1"][0]), int(ax["p1"][1]))
            p2 = (int(ax["p2"][0]), int(ax["p2"][1]))
            cv2.line(vis_3a, p1, p2, (255, 255, 255), 2)
            cx, cy = int(ax["centroid"][0]), int(ax["centroid"][1])
            cv2.circle(vis_3a, (cx, cy), 4, palette[i % len(palette)], -1)
            put_text(vis_3a, f"A{i + 1} L:{ax['len_px']:.0f}px",
                     (cx - 40, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        self._draw_header(vis_3a, f"STAGE 3A: AXIS EXTRACTION | Axes: {len(axes_a)}", (255, 200, 40))
        step_images["stage3a_axis"] = vis_3a

        # ---------------- 步骤 1B/2B: Canny 边缘提取与闭运算清理 (B 分支, 为步骤 3 融合供源) ----------------
        edges_b = self._stage1b_edges(color_bgr)
        edges_bridged = self._stage2b_edge_morph(edges_b)

        # ---------------- 步骤 3: 融合 (A 系 2A 掩膜 ∪ B 系 2B 清理边缘, 双分支按位或汇合) ----------------
        mask_fused = cv2.bitwise_or(mask_open, edges_bridged)

        vis_3 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_3[mask_open > 0] = (80, 240, 120)                          # A 系贡献 (绿)
        vis_3[(edges_bridged > 0) & (mask_open == 0)] = (250, 230, 90)  # B 系补充 (黄)
        self._draw_header(vis_3, f"STAGE 3: FUSE A-Mask + B-Edges | Fused px: {int(cv2.countNonZero(mask_fused))}",
                          (200, 200, 255))
        step_images["stage3_fuse"] = vis_3

        # ---------------- 步骤 4: 区域成形 (闭运算弥合断裂, 凝聚连通区域, 数据源 3 融合掩膜) ----------------
        k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_region = cv2.morphologyEx(mask_fused, cv2.MORPH_CLOSE, k5)

        vis_4 = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        vis_4[mask_region > 0] = (80, 240, 120)
        self._draw_header(vis_4, "STAGE 4: MORPH CLOSE REGION FORMING", (0, 220, 120))
        step_images["stage4_region"] = vis_4

        # ---------------- 步骤 5: 连通域外框提取 ----------------
        contours, _ = cv2.findContours(mask_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []  # (contour, minAreaRect, box_corners)
        for c in contours:
            if cv2.contourArea(c) < self.min_area_px:
                continue
            rect = cv2.minAreaRect(c)
            rw, rh = rect[1]
            if min(rw, rh) < 1e-3:
                continue
            if max(rw, rh) / min(rw, rh) < self.min_aspect:
                continue
            box_pts = np.array(cv2.boxPoints(rect), dtype=np.int32)
            boxes.append((c, rect, box_pts))

        vis_5 = color_bgr.copy()
        for i, (_, _, box_pts) in enumerate(boxes):
            cv2.polylines(vis_5, [box_pts], True, palette[i % len(palette)], 2)
        self._draw_header(vis_5, f"STAGE 5: GREEN FRAME BOXES | Found: {len(boxes)}", (255, 200, 40))
        step_images["stage5_boxes"] = vis_5

        # ---------------- 步骤 6+7: 轴线拟合与主干确定 (单层假设, 无叠压分层) ----------------
        stems: List[Dict] = []
        vis_6 = color_bgr.copy()
        vis_7 = color_bgr.copy()
        for c, rect, box_pts in boxes:
            fill = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(fill, [c], -1, 255, -1)
            ys, xs = np.nonzero(fill)
            if len(xs) < 30:
                continue
            pts = np.column_stack((xs, ys)).astype(np.float32)

            # 轴线拟合: 全量像素云 L2 鲁棒直线拟合
            vx_v, vy_v, x0_v, y0_v = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx_v[0]), float(vy_v[0])
            if vx < 0:
                vx, vy = -vx, -vy
            mean_pt = pts.mean(axis=0)
            proj = (pts - mean_pt) @ np.array([vx, vy])
            min_p, max_p = float(proj.min()), float(proj.max())
            len_px = max_p - min_p

            # 步骤 6 可视化: 拟合轴线段
            ax1 = (int(mean_pt[0] + min_p * vx), int(mean_pt[1] + min_p * vy))
            ax2 = (int(mean_pt[0] + max_p * vx), int(mean_pt[1] + max_p * vy))
            cv2.line(vis_6, ax1, ax2, (255, 255, 255), 2)

            # 步骤 7: 物理长度门限确认主干 (单层假设, 一个外框唯一一条主干)
            len_mm = len_px * scale_2d
            if not (self.min_length_mm <= len_mm <= self.max_length_mm):
                continue
            diam_px = float(min(rect[1]))
            diam_mm = diam_px * scale_2d
            mid_u = float(mean_pt[0] + (min_p + max_p) * 0.5 * vx)
            mid_v = float(mean_pt[1] + (min_p + max_p) * 0.5 * vy)
            yaw = float(np.degrees(np.arctan2(vy, vx)))
            if yaw > 90.0:
                yaw -= 180.0
            elif yaw < -90.0:
                yaw += 180.0

            stems.append({
                "mid_px": (mid_u, mid_v),
                "len_px": len_px, "len_mm": len_mm,
                "diam_px": diam_px, "diam_mm": diam_mm,
                "yaw": yaw, "axis": (vx, vy),
                "box_pts": box_pts, "contour": c,
            })

            # 步骤 7 可视化: 外框 + 主干 + 长度标签
            cv2.polylines(vis_7, [box_pts], True, (80, 240, 120), 2)
            cv2.line(vis_7, ax1, ax2, (255, 255, 255), 2)
            put_text(vis_7, f"L:{len_mm:.0f}mm D:{diam_mm:.1f}mm",
                     (int(mid_u) - 60, int(mid_v) - 8),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        self._draw_header(vis_6, f"STAGE 6: AXIS FITTING | Fitted: {len(boxes)}", (200, 200, 255))
        step_images["stage6_axis"] = vis_6
        self._draw_header(vis_7, f"STAGE 7: MAIN STEMS (Single-Layer) | Confirmed: {len(stems)}",
                          (80, 240, 120))
        step_images["stage7_stem"] = vis_7

        # ---------------- 步骤 8: 主干中点定位 (Z 恒为 0) ----------------
        from src.vision.asparagus_analyzer import AsparagusTarget, AsparagusAnalyzer

        stems.sort(key=lambda s: s["len_px"], reverse=True)
        targets: List[AsparagusTarget] = []

        for rank, s in enumerate(stems[:3]):
            mid_u, mid_v = s["mid_px"]
            vx, vy = s["axis"]

            # X/Y 按标称深度反投影解算, Z 恒为 0 (F1 路线约定)
            grip_x = (mid_u - self.cx) * nominal_z_mm / self.fx
            grip_y = (mid_v - self.cy) * nominal_z_mm / self.fy
            grip_z = 0.0

            if frame_transform is not None:
                p_h = frame_transform @ np.array([grip_x, grip_y, grip_z, 1.0])
                robot_x, robot_y, robot_z = float(p_h[0]), float(p_h[1]), float(p_h[2])
                v_robot = frame_transform[:3, :3] @ np.array([vx, vy, 0.0])
                robot_r = float(np.degrees(np.arctan2(v_robot[1], v_robot[0])))
                if robot_r > 90.0:
                    robot_r -= 180.0
                elif robot_r < -90.0:
                    robot_r += 180.0
            else:
                robot_x, robot_y, robot_z = grip_x, grip_y, grip_z
                robot_r = s["yaw"]

            targets.append(AsparagusTarget(
                id=rank + 1,
                center_px=(mid_u, mid_v),
                length_px=s["len_px"],
                diam_px=s["diam_px"],
                yaw_deg=round(s["yaw"], 1),
                axis_vector=(vx, vy),
                box_corners=s["box_pts"],
                contour=s["contour"],
                length_mm=round(s["len_mm"], 1),
                diam_mm=round(s["diam_mm"], 1),
                grip_x=round(grip_x, 1),
                grip_y=round(grip_y, 1),
                grip_z=round(grip_z, 1),
                z_top=0.0,
                rel_height_mm=0.0,
                robot_x=round(robot_x, 1),
                robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1),
                robot_r=round(robot_r, 1),
                is_topmost=(rank == 0),
                calibration_source=frame_calib_source
            ))

        dummy_analyzer = AsparagusAnalyzer(self.fx, self.fy, self.cx, self.cy)
        vis_8 = dummy_analyzer.draw_detections(color_bgr, targets, sel_target_idx=0)
        for t in targets:
            cv2.circle(vis_8, (int(t.center_px[0]), int(t.center_px[1])), 5, (0, 0, 255), -1)
            cv2.circle(vis_8, (int(t.center_px[0]), int(t.center_px[1])), 9, (255, 255, 255), 1)
        self._draw_header(vis_8, f"STAGE 8: STEM MIDPOINT (Z=0) | Targets: {len(targets)}",
                          (0, 255, 120))
        step_images["stage8_midpoint"] = vis_8

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=targets,
            elapsed_ms=round(elapsed_ms, 1),
            step_snapshots=step_images,
            extra_metrics={
                "green_pixels": green_pixels,
                "fused_pixels": int(cv2.countNonZero(mask_fused)),
                "individuals": n_split,
                "axes_early": len(axes_a),
                "boxes_found": len(boxes),
                "stems_confirmed": len(stems)
            }
        )

    def preview_stage1_hsv_mask(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 1 HSV 绿分割调参实时预览: 按滑条当前阈值叠加绿色掩膜高亮"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,
                           (self.h_low, self.s_min, self.v_min),
                           (self.h_high, self.s_high, self.v_high))
        vis = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
        vis[mask > 0] = (80, 240, 120)
        put_text(vis,
                 f"HSV TUNING  H:{self.h_low}~{self.h_high}  S:{self.s_min}~{self.s_high}"
                 f"  V:{self.v_min}~{self.v_high}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2)
        return vis

    def _draw_header(self, img: np.ndarray, text: str, color) -> None:
        """步骤快照左上角标准化标题栏"""
        cv2.rectangle(img, (12, 12), (680, 48), (20, 20, 20), -1)
        cv2.rectangle(img, (12, 12), (680, 48), color, 2)
        put_text(img, text, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)
