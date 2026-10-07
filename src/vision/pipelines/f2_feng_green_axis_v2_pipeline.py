"""
技术路线 F2：冯氏二代 —— 轮廓中心线三步流水线 (独立实现, 不继承 F1)
==========================================================================
(Feng's Green Axis V2 Pipeline: HSV Green Segmentation -> Contour Draw -> Centerline)
独立流水线: 处理核代码复制自 F1 (后续两条路线独立演化, 互不影响), 精简为四步 (无个体分离):
  1. HSV 色彩空间绿色分割 (处理核同 F1 的 1A)
  2. 绘制: 形态学清理 (先 CLOSE 弥合茎干断裂 -> 再 OPEN 去散点 -> 小面积碎块丢弃) 后,
     提取并绘制绿色区域外轮廓 (芦笋外轮廓)
  3. 中心线提取: 芦笋不是纯粹的直线而是弯曲的, 逐外轮廓连通域做 Zhang-Suen 骨架拓扑分组
     (细化 + 毛刺修剪 + 交叉臂共线配对), 每组洪泛归还相邻交叉点切割区像素桥接配对臂,
     再双 BFS 树直径追踪出从笋头到笋尖沿弯曲中心行走的有序折线中心线
     (approxPolyDP 简化), 弧长 < min_axis_len 的碎渣线不输出
  4. 单根识别: 骨架拓扑判定单根/多根 —— 同一轮廓拆出 >=2 条中心线 (交叉黏连) 或
     中心线外残留 >= 分叉臂长门槛的侧臂 (Y 形分叉) 判为多根异常,
     染红系 (红/淡红/粉) + 亮红中心线告警; 正常单根染绿 + 白色中心线
无个体分离: 交叉黏连的芦笋骨架呈 X 型, 交叉臂共线配对天然把中心线按根拆分。
不输出抓取目标 (targets 为空), 中心线结果呈现在快照与 extra_metrics 中。
"""

import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.text_rendering import put_text
from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry

# 轮廓/中心线着色调色板 (步骤 2 与步骤 3 共用)
F2_PALETTE = [(255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120),
              (255, 210, 40), (180, 160, 255), (120, 255, 255), (255, 150, 150)]


@PipelineRegistry.register("feng_green_axis_v2", "算法 F2: 冯氏二代 (Feng's Green Axis V2)")
class FengGreenAxisV2Pipeline(BaseAsparagusPipeline):
    """算法 F2：冯氏二代 —— HSV 绿分割 -> 外轮廓绘制 -> 骨架分组双 BFS 中心线追踪 (独立实现)"""

    name = "算法 F2: 冯氏二代 (Feng's Green Axis V2)"
    description = "HSV 绿分割 -> 外轮廓绘制 (形态学清理) -> 骨架拓扑分组 + 双 BFS 折线中心线 (弯曲一根一线, 交叉配对拆分)"

    # 步骤滑条声明: 步骤 2 为形态学滑条, 步骤 3 为中心线弧长准入门槛
    STEP_SLIDERS = {
        "stage1_hsv_mask": [
            {"label": "H 色相", "attr_low": "h_low", "attr_high": "h_high", "vmin": 0, "vmax": 179},
            {"label": "S 饱和", "attr_low": "s_min", "attr_high": "s_high", "vmin": 0, "vmax": 256},
            {"label": "V 明度", "attr_low": "v_min", "attr_high": "v_high", "vmin": 0, "vmax": 256},
        ],
        "stage2_draw": [
            {"label": "闭核", "attr": "morph_close_k", "vmin": 1, "vmax": 15},
            {"label": "开核", "attr": "morph_ksize", "vmin": 1, "vmax": 15},
            {"label": "最小面积", "attr": "min_blob_area", "vmin": 0, "vmax": 3000},
        ],
        "stage3_centerline": [
            {"label": "最小线长", "attr": "min_axis_len", "vmin": 0, "vmax": 300},
        ],
        "stage4_classify": [
            {"label": "分叉臂长", "attr": "fork_min_len", "vmin": 0, "vmax": 300},
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
        # 步骤 2 绘制参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.morph_close_k = 7        # 闭核尺寸: 先弥合 HSV 茎干断裂保住完整长条 (奇数化)
        self.morph_ksize = 5          # 开核尺寸: 后去散点碎屑 (使用时归一化为奇数)
        self.min_blob_area = 120      # 小面积连通碎块丢弃阈值 px (0=不丢弃)
        # 步骤 3 中心线提取参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.min_axis_len = 80.0      # 中心线弧长准入门槛 px (碎渣短线不输出不显示, 0=不过滤)
        # 步骤 4 单根识别参数 (Studio 单滑块可调, 持久化于 gui_settings.json)
        self.fork_min_len = 50.0      # 分叉侧臂准入长度 px (中心线外残留侧臂 >= 此值判分叉, 0=不检)

    # ---------------- 步骤 1 处理核与预览 ----------------
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

    # ---------------- 骨架拓扑处理核 (复制自 F1, 独立演化) ----------------
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

    def _skeleton_groups(self, crop: np.ndarray) -> Optional[Tuple[List[np.ndarray], np.ndarray]]:
        """骨架拓扑分组核心: Zhang-Suen 细化 -> 小支路毛刺修剪
        -> 交叉臂共线连续性配对 (并查集), 返回 (每组骨架点掩膜列表, 交叉点切割区掩膜);
        组掩膜不含交叉点切割区像素 (中心线追踪按组洪泛归还切割区像素以桥接配对臂);
        骨架退化 (极小碎域) 返回 None, 由调用方决定退化行为。
        弯曲芦笋骨架为单条路径不再拆段, 天然一根一线; 交叉黏连簇在交叉点按方向连续性两两配对"""
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

        # 终态拓扑分解: 交叉点 3x3 扩张切割后每段支路是简单路径 (切割区掩膜供洪泛桥接)
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

        # 配对分支并成组 (每组为一条中心线的骨架点集, 不含交叉点切割区像素)
        groups: Dict[int, np.ndarray] = {}
        for i in range(1, n_br):
            r = find(i)
            groups[r] = groups.get(r, np.zeros_like(skel, bool)) | (br_lbl == i)
        return list(groups.values()), cut

    # ---------------- 步骤 2 处理核: 形态学清理 + 外轮廓提取 ----------------
    def _stage2a_morph(self, green_mask: np.ndarray):
        """形态学清理处理核: 闭运算弥合茎干断裂 -> 开运算去散点 -> 小面积连通碎块丢弃
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

    def _stage2_draw(self, green_mask: np.ndarray):
        """步骤 2 处理核: 形态学清理 (CLOSE 弥合断裂 -> OPEN 去散点 -> 碎块丢弃) 后
        提取外轮廓。返回 (清理掩膜, 外轮廓列表, 被丢弃碎块数)"""
        mask_clean, removed = self._stage2a_morph(green_mask)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return mask_clean, contours, removed

    def preview_stage2_draw(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 2 绘制调参实时预览: 按滑条当前核尺寸与面积阈值提取并着色绘制外轮廓"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        _, contours, removed = self._stage2_draw(green_mask)
        vis = self._draw_contours_dim(color_bgr, contours)
        put_text(vis, f"2 DRAW TUNING  k={max(1, int(self.morph_close_k)) | 1}"
                      f"  min_area={int(self.min_blob_area)}  Contours: {len(contours)}"
                      f"  Rej: {removed}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)
        return vis

    def _draw_contours_dim(self, color_bgr: np.ndarray, contours) -> np.ndarray:
        """暗化原图上逐轮廓调色板填充 + 青色描边 (步骤 2 与预览共用)"""
        vis = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        for i, cnt in enumerate(contours):
            fill = np.zeros_like(vis)
            cv2.drawContours(fill, [cnt], -1, F2_PALETTE[i % len(F2_PALETTE)], -1)
            m = fill.max(axis=2) > 0
            vis[m] = (vis[m] * 0.45 + fill[m] * 0.55).astype(np.uint8)
            cv2.drawContours(vis, [cnt], -1, (255, 255, 0), 2)   # 青色外轮廓描边
        return vis

    # ---------------- 步骤 3 处理核: 骨架分组 + 双 BFS 中心线追踪 ----------------
    @staticmethod
    def _bfs_path(ptset: set, start: Tuple[int, int]):
        """8 连通 BFS 带父指针: 从 start 出发, 返回 (最远点, start->最远点的有序路径)"""
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        frontier = deque([start])
        far = start
        while frontier:
            cur = frontier.popleft()
            far = cur
            cx, cy = cur
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nxt = (cx + dx, cy + dy)
                    if nxt in ptset and nxt not in parent:
                        parent[nxt] = cur
                        frontier.append(nxt)
        path: List[Tuple[int, int]] = []
        node = far
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()
        return far, path

    def _trace_group_path(self, group_mask: np.ndarray, junction_mask: np.ndarray,
                          ox: int, oy: int) -> Optional[Tuple[List[Tuple[float, float]], float, bool]]:
        """单组骨架掩膜 -> 有序折线中心线 (全图坐标):
        洪泛归还与本组支路相邻的交叉点切割区像素 (桥接同组配对臂断缝, 不串入他组支路)
        -> 双 BFS 树直径取最长路径 (笋头到笋尖) -> 8 连通弧长 -> approxPolyDP 折线简化
        -> 分叉检测: 直径路径之外残留骨架连通域像素数 >= fork_min_len 判 Y 形分叉
        返回 (折线 [(x,y),...], 弧长 px, 是否分叉) 或 None"""
        conn = group_mask.astype(bool).copy()
        jm = junction_mask.astype(bool)
        for _ in range(8):   # 交叉点切割区通常 2~4px, 洪泛收敛即止
            grown = cv2.dilate(conn.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
            add = grown & jm & ~conn
            if not add.any():
                break
            conn |= add
        ys, xs = np.nonzero(conn)
        if len(xs) < 8:
            return None
        ptset = set(zip(xs.tolist(), ys.tolist()))
        far_a, _ = self._bfs_path(ptset, next(iter(ptset)))
        far_b, path = self._bfs_path(ptset, far_a)
        if len(path) < 8:
            return None
        arc = 0.0   # 8 连通弧长: 直行 1, 对角 √2
        for (x1, y1), (x2, y2) in zip(path[:-1], path[1:]):
            arc += 1.41421356 if (x1 != x2 and y1 != y2) else 1.0
        pts = np.array(path, dtype=np.float32).reshape(-1, 1, 2)
        poly = cv2.approxPolyDP(pts, max(1.5, arc * 0.01), True).reshape(-1, 2)
        poly_full = [(float(px + ox), float(py + oy)) for px, py in poly]
        # 分叉检测: 直径路径之外残留骨架 (未修剪的侧臂) 最大连通域像素数达到门槛判分叉
        fork = False
        if float(self.fork_min_len) > 0:
            path_mask = np.zeros(conn.shape, bool)
            for px, py in path:
                path_mask[py, px] = True
            rest = (conn & ~path_mask).astype(np.uint8)
            if rest.any():
                n_r, _, r_stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
                for r in range(1, n_r):
                    if r_stats[r, cv2.CC_STAT_AREA] >= float(self.fork_min_len):
                        fork = True
                        break
        return poly_full, arc, fork

    def _stage_centerlines(self, mask_clean: np.ndarray) -> List[Dict]:
        """步骤 3 处理核: 逐外轮廓连通域骨架拓扑分组, 每组追踪有序折线中心线
        返回 [{"path", "len_px", "p1", "p2", "centroid", "area_px",
               "cc_id": 所属外轮廓连通域标签, "fork": 是否 Y 形分叉, "multi": 同域多线黏连}],
        按弧长降序"""
        lines: List[Dict] = []
        n_cc, cc_lbl, cc_stats, _ = cv2.connectedComponentsWithStats(mask_clean, connectivity=8)
        for lbl in range(1, n_cc):
            area = int(cc_stats[lbl, cv2.CC_STAT_AREA])
            if area < 30:          # 像素过少的碎域不参与追踪
                continue
            ox = int(cc_stats[lbl, cv2.CC_STAT_LEFT])
            oy = int(cc_stats[lbl, cv2.CC_STAT_TOP])
            w = int(cc_stats[lbl, cv2.CC_STAT_WIDTH])
            h = int(cc_stats[lbl, cv2.CC_STAT_HEIGHT])
            crop = (cc_lbl[oy:oy + h, ox:ox + w] == lbl).astype(np.uint8)
            result = self._skeleton_groups(crop)
            if result is None:     # 骨架退化 (极小碎域): 跳过
                continue
            group_masks, junctions = result
            for gm in group_masks:
                traced = self._trace_group_path(gm, junctions, ox, oy)
                if traced is None:
                    continue
                poly, arc, fork = traced
                lines.append({
                    "path": poly,
                    "len_px": arc,
                    "p1": poly[0],
                    "p2": poly[-1],
                    "centroid": poly[len(poly) // 2],
                    "area_px": area,
                    "cc_id": lbl,
                    "fork": fork,
                })
        lines = [ln for ln in lines if ln["len_px"] >= float(self.min_axis_len)]   # 碎渣线准入门槛
        lines.sort(key=lambda ln: ln["len_px"], reverse=True)   # 按弧长降序, 长线优先编号
        # 多根判定: 同一外轮廓连通域拆出 >=2 条中心线 => 交叉黏连多根
        cc_count: Dict[int, int] = {}
        for ln in lines:
            cc_count[ln["cc_id"]] = cc_count.get(ln["cc_id"], 0) + 1
        for ln in lines:
            ln["multi"] = cc_count[ln["cc_id"]] >= 2
        return lines

    def preview_stage3_centerline(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 3 中心线提取调参实时预览: 重算 1->2->3 链, 按最小线长门槛绘制中心线"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        mask_clean, contours, _ = self._stage2_draw(green_mask)
        lines = self._stage_centerlines(mask_clean)
        vis = self._draw_centerlines_dim(color_bgr, mask_clean, contours, lines)
        put_text(vis, f"3 CENTERLINE TUNING  min_len={self.min_axis_len:.0f}px  Lines: {len(lines)}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 200, 40), 2)
        return vis

    def _draw_centerlines_dim(self, color_bgr: np.ndarray, mask_clean: np.ndarray,
                              contours, lines: List[Dict]) -> np.ndarray:
        """暗化原图 + 绿掩膜淡染 + 青轮廓细线 + 白色折线中心线 (步骤 3 与预览共用)"""
        vis = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        vis[mask_clean > 0] = (vis[mask_clean > 0] * 0.5
                               + np.array((40, 120, 60), dtype=np.float32)).astype(np.uint8)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 1)   # 青色轮廓细线
        for i, ln in enumerate(lines):
            pts = np.array(ln["path"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (255, 255, 255), 2)
            mx, my = ln["centroid"]
            cv2.circle(vis, (int(mx), int(my)), 4, F2_PALETTE[i % len(F2_PALETTE)], -1)
            put_text(vis, f"A{i + 1} L:{ln['len_px']:.0f}px",
                     (int(mx) - 40, int(my) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return vis

    # ---------------- 步骤 4 处理核: 单根识别 (绿+白线 / 红系+亮红线告警) ----------------
    RED_FILLS = [(60, 60, 255), (115, 115, 255), (165, 135, 255)]   # 红 / 淡红 / 粉 (BGR, 逐异常株轮换)
    WARN_RED = (70, 70, 255)                                        # 异常中心线亮红告警色

    def _draw_classify_dim(self, color_bgr: np.ndarray, mask_clean: np.ndarray,
                           lines: List[Dict]) -> Tuple[np.ndarray, int, int]:
        """步骤 4 绘制: 暗化原图上按判定结果染色 ——
        正常单根: 绿色芦笋 + 白色中心线; 异常 (分叉/多根黏连): 红/淡红/粉芦笋 + 亮红中心线
        返回 (可视化图, 单根数, 异常数)"""
        vis = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        cc_lbl = cv2.connectedComponents(mask_clean, connectivity=8)[1] if lines else None
        # 第一遍: 按连通域一次性染株体色 (同一黏连域多条中心线共用一个区域)
        abnormal_cc: Dict[int, bool] = {}
        for ln in lines:
            cid = int(ln.get("cc_id", -1))
            abnormal_cc[cid] = abnormal_cc.get(cid, False) or bool(ln.get("multi") or ln.get("fork"))
        seen: set = set()
        red_i = 0
        for ln in lines:
            cid = int(ln.get("cc_id", -1))
            if cid < 0 or cc_lbl is None or cid in seen:
                continue
            seen.add(cid)
            m = cc_lbl == cid
            if not m.any():
                continue
            if abnormal_cc[cid]:
                col = self.RED_FILLS[red_i % len(self.RED_FILLS)]
                red_i += 1
            else:
                col = (80, 240, 120)   # 正常单根绿色株体
            vis[m] = (vis[m] * 0.40 + np.array(col, dtype=np.float32) * 0.60).astype(np.uint8)
        # 第二遍: 绘中心线与标注
        n_single = n_abn = 0
        for ln in lines:
            pts = np.array(ln["path"], dtype=np.int32).reshape(-1, 1, 2)
            mx, my = ln["centroid"]
            if ln.get("multi") or ln.get("fork"):
                n_abn += 1
                cv2.polylines(vis, [pts], False, self.WARN_RED, 3)
                tag = "MULTI" if ln.get("multi") else "FORK"
                put_text(vis, f"{tag} L:{ln['len_px']:.0f}px", (int(mx) - 40, int(my) - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.WARN_RED, 1)
            else:
                n_single += 1
                cv2.polylines(vis, [pts], False, (255, 255, 255), 2)
                put_text(vis, f"S L:{ln['len_px']:.0f}px", (int(mx) - 40, int(my) - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        return vis, n_single, n_abn

    def preview_stage4_classify(self, color_bgr: np.ndarray) -> np.ndarray:
        """步骤 4 单根识别调参实时预览: 重算 1->2->3->4 链, 按分叉臂长门槛染色"""
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        mask_clean, _, _ = self._stage2_draw(green_mask)
        lines = self._stage_centerlines(mask_clean)
        vis, n_single, n_abn = self._draw_classify_dim(color_bgr, mask_clean, lines)
        put_text(vis, f"4 SINGLE-STALK TUNING  fork_len={self.fork_min_len:.0f}px"
                      f"  Single: {n_single}  Multi/Fork: {n_abn}",
                 (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, self.WARN_RED, 2)
        return vis

    # ---------------- 快照标头 (复制自 F1) ----------------
    def _draw_header(self, img: np.ndarray, text: str, color) -> None:
        """步骤快照左上角标准化标题栏"""
        cv2.rectangle(img, (12, 12), (680, 48), (20, 20, 20), -1)
        cv2.rectangle(img, (12, 12), (680, 48), color, 2)
        put_text(img, text, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    def get_steps(self) -> List[PipelineStep]:
        return [
            PipelineStep(
                "stage1_hsv_mask", "1.HSV绿分割",
                "BGR 转 HSV 后按绿色色相带提取绿点二值掩膜",
                details="Hue 通道将色彩与亮度解耦，35~85 的绿色色相带可稳定锁定芦笋茎干绿色，"
                        "S/V 双下限剔除传送带灰白反光与暗影低饱和噪点，彻底免疫底色干扰",
                parameters="h_low=35, h_high=85 (色相带，偏黄可下调至 30); s_min=40, v_min=40 "
                           "(饱和度/明度门限, 步骤 1 滑条可调)",
                pros_cons="优点: 速度极快 (<1ms) 且对底色变化天然免疫; 缺点: 强黄色灯光或蓝青色物料可能串色"
            ),
            PipelineStep(
                "stage2_draw", "2.绘制",
                "形态学清理后提取并绘制芦笋外轮廓 (闭运算弥合断裂 + 开运算去散点 + 碎块丢弃)",
                details="绿分割掩膜上茎干因高光/阴影天然断裂，先闭运算 (膨胀再腐蚀) 接缝保住完整长条，"
                        "再开运算剔除孤立散点，小于面积阈值的连通碎块直接丢弃；"
                        "然后对清理后的掩膜提取外轮廓 (RETR_EXTERNAL) 并逐个着色绘制，"
                        "每个连通域的外轮廓即一根芦笋 (或一片黏连簇) 的完整边界",
                parameters="morph_close_k=7 (闭核, 先弥合断裂); morph_ksize=5 (开核, 后去散点); "
                           "min_blob_area=120 (碎块丢弃阈值 px, 0=不丢弃) —— 均为步骤 2 滑条可调",
                pros_cons="优点: 外轮廓是弯曲中心线与长度测量的直接数据源; "
                          "缺点: 黏连严重时多个体连成一片轮廓 (中心线步骤按骨架拓扑拆分)"
            ),
            PipelineStep(
                "stage3_centerline", "3.中心线提取",
                "逐外轮廓骨架拓扑分组, 双 BFS 追踪从笋头到笋尖沿弯曲中心行走的有序折线中心线",
                details="芦笋是弯曲的细长杆, 轴线不是直线而是中心线: 对每个外轮廓连通域做 "
                        "Zhang-Suen 细化得到单像素骨架 (即芦笋中轴), 小毛刺按长度自动修剪；"
                        "交叉黏连簇在交叉点按臂方向共线连续性配对 (并查集, 夹角 >~110° 才接)，"
                        "每组洪泛归还与本组支路相邻的交叉点切割区像素桥接配对臂 (不串入他组支路)；"
                        "组内双 BFS 取最长路径 (树直径) 即从一端到另一端的中心线，"
                        "8 连通弧长累计后 approxPolyDP 保形简化为有序折线；"
                        "弧长短于准入门槛的碎渣线不输出不显示，按弧长降序编号 (A1/A2/...)",
                parameters="min_axis_len=80 (中心线弧长准入门槛 px, 步骤 3 滑条 0~300); "
                           "毛刺修剪阈 max(12, 0.12×最长支路) px; 配对夹角阈 ~110°; 折线简化 eps=max(1.5, 弧长×1%)",
                pros_cons="优点: 弯曲芦笋一根一线且沿真实弯曲中心行走, 交叉黏连按拓扑拆分; "
                          "缺点: 密集多交叉簇配对可能保守合并"
            ),
            PipelineStep(
                "stage4_classify", "4.单根识别",
                "骨架拓扑判定单根/多根: 正常单根=绿色芦笋+白色中心线, 分叉或多根黏连=红系芦笋+亮红中心线告警",
                details="以骨架拓扑为分根判据: 同一外轮廓连通域经交叉臂配对拆出 >=2 条中心线 => "
                        "交叉黏连多根 (X 形, 标 MULTI); 单条中心线直径路径之外残留长度 >= 分叉臂长"
                        "门槛的侧臂 => 单株分叉 (Y 形, 标 FORK)。两类均判为异常多根: "
                        "株体染红/淡红/粉 (逐株轮换) 并绘亮红中心线告警; 正常单根株体染绿 + 白色中心线",
                parameters="fork_min_len=50 (分叉侧臂准入长度 px, 步骤 4 滑条 0~300, 0=不检分叉)",
                pros_cons="优点: 无需个体分离即可区分单根与黏连多根, 为抓取提供质量警示; "
                          "缺点: 交叉臂配对保守合并时整簇可能被记为一根"
            ),
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
        step_images: Dict[str, Optional[np.ndarray]] = {}

        # ---------------- 步骤 1: HSV 绿色分割 ----------------
        hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv,
                                 (self.h_low, self.s_min, self.v_min),
                                 (self.h_high, self.s_high, self.v_high))
        green_pixels = int(cv2.countNonZero(green_mask))

        vis_1 = (color_bgr.astype(np.float32) * 0.30).astype(np.uint8)
        vis_1[green_mask > 0] = (80, 240, 120)
        self._draw_header(vis_1, f"STAGE 1: HSV GREEN MASK (H {self.h_low}~{self.h_high})"
                                 f" | Px: {green_pixels}", (80, 240, 120))
        step_images["stage1_hsv_mask"] = vis_1

        # ---------------- 步骤 2: 绘制 (形态学清理 + 外轮廓提取与着色) ----------------
        mask_clean, contours, removed = self._stage2_draw(green_mask)

        vis_2 = self._draw_contours_dim(color_bgr, contours)
        self._draw_header(vis_2, f"STAGE 2: CONTOUR DRAW (CLOSE {max(1, int(self.morph_close_k)) | 1}"
                                 f" + OPEN {max(1, int(self.morph_ksize)) | 1}"
                                 f" | < {int(self.min_blob_area)}px Rej: {removed}"
                                 f") | Contours: {len(contours)}", (255, 255, 0))
        step_images["stage2_draw"] = vis_2

        # ---------------- 步骤 3: 中心线提取 (骨架分组 + 双 BFS 折线追踪) ----------------
        lines = self._stage_centerlines(mask_clean)

        vis_3 = self._draw_centerlines_dim(color_bgr, mask_clean, contours, lines)
        self._draw_header(vis_3, f"STAGE 3: CENTERLINE | Lines: {len(lines)}", (255, 200, 40))
        step_images["stage3_centerline"] = vis_3

        # ---------------- 步骤 4: 单根识别 (单根绿+白线 / 多根红系+亮红线告警) ----------------
        vis_4, n_single, n_abn = self._draw_classify_dim(color_bgr, mask_clean, lines)
        self._draw_header(vis_4, f"STAGE 4: SINGLE/MULTI (fork >= {self.fork_min_len:.0f}px)"
                                 f" | Single: {n_single} | Multi/Fork: {n_abn}", self.WARN_RED)
        step_images["stage4_classify"] = vis_4

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            targets=[],
            elapsed_ms=round(elapsed_ms, 1),
            step_snapshots=step_images,
            extra_metrics={
                "green_pixels": green_pixels,
                "contours": len(contours),
                "centerlines": len(lines),
                "singles": n_single,
                "multis": n_abn,
            }
        )
