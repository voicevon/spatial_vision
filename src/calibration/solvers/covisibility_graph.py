#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标靶共视连通图拓扑分析器 (Covisibility Graph Analyzer)
- 纯面向对象设计，单一职责处理多视角标靶共视网络建模与图论拓扑分析
- 连通分量检测 (BFS)、孤岛标靶识别、单图支撑关键桥梁 (Critical Bridges) 预警
- 确保 BA 平差前不存在拓扑断网，避免奇异矩阵
"""

from typing import Dict, List, Tuple, Optional, Any, Set
import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


class CovisibilityGraphError(Exception):
    """标靶共视连通图拓扑异常（如出现孤立子图、断网或约束退化）"""
    pass


class CovisibilityGraphAnalyzer:
    """
    共视连通性安全守门员 (Co-visibility Graph Connectivity Guard)
    提供标靶共视图连通度、关键桥梁及孤岛分析的核心算法
    """

    @staticmethod
    def analyze(frame_detections: List[Dict[int, np.ndarray]], 
                frame_names: Optional[List[str]] = None,
                origin_tag_id: int = 0,
                x_align_tag_id: int = 1) -> Dict[str, Any]:
        """
        全面审查标靶共视图拓扑健康度：
        1. 检查各 Tag 是否在一个连通分量内，防止误删导致孤岛断网引发奇异矩阵；
        2. 识别单视角关键桥梁 (Critical Bridges)，当桥梁仅由 1 张图片支撑时警示用户；
        3. 给出详细结构化拓扑报告。
        """
        all_tags = set()
        for tags in frame_detections:
            all_tags.update(tags.keys())

        if len(all_tags) == 0:
            return {
                "is_valid": False,
                "all_tags": [],
                "connected_tags": [],
                "unconnected_tags": [],
                "adj_list": {},
                "edge_counts": {},
                "critical_bridges": [],
                "valid_frames_count": 0,
                "components": [],
                "message": "有效标靶集合为空，无法建图！"
            }

        # 构建无向图与共视重叠帧数统计
        adj_list: Dict[int, Set[int]] = {t: set() for t in all_tags}
        edge_counts: Dict[Tuple[int, int], int] = {}

        for tags in frame_detections:
            t_ids = list(tags.keys())
            for i in range(len(t_ids)):
                for j in range(i + 1, len(t_ids)):
                    u, v = min(t_ids[i], t_ids[j]), max(t_ids[i], t_ids[j])
                    adj_list[u].add(v)
                    adj_list[v].add(u)
                    edge_counts[(u, v)] = edge_counts.get((u, v), 0) + 1

        # 识别关键单一支撑桥梁 (只有 1 帧同时观测到该标靶对)
        critical_bridges = [(u, v) for (u, v), count in edge_counts.items() if count == 1]

        # 寻找连通分量 (BFS)
        remaining = set(all_tags)
        components = []
        while remaining:
            root = next(iter(remaining))
            comp = set()
            q = [root]
            while q:
                curr = q.pop(0)
                if curr not in comp:
                    comp.add(curr)
                    q.extend(adj_list[curr] - comp)
            components.append(sorted(list(comp)))
            remaining -= comp

        # 确定主基准分量 (优先包含 x_align_tag_id 或 origin_tag_id)
        preferred_base = x_align_tag_id if x_align_tag_id in all_tags else (origin_tag_id if origin_tag_id in all_tags else min(all_tags))
        main_component = []
        for comp in components:
            if preferred_base in comp:
                main_component = comp
                break
        if not main_component and components:
            main_component = max(components, key=len)

        unconnected = sorted(list(all_tags - set(main_component)))
        is_valid = (len(components) == 1) and (len(all_tags) >= 2) and (len(frame_detections) >= 2)

        if not is_valid:
            if len(all_tags) < 2:
                msg = f"有效标靶总数不足 2 个 (仅检出 {list(all_tags)})，无法构建相对空间地图！"
            elif len(frame_detections) < 2:
                msg = f"有效多视角图像不足 2 张 (当前仅 {len(frame_detections)} 张)，无法构建空间刚体约束！"
            else:
                msg = (f"【严重风险】标靶共视图发生断裂！共发现 {len(components)} 个孤立分量。\n"
                       f"主连通网络: {main_component}\n"
                       f"失联断网标靶: {unconnected}\n"
                       f"提示：请检查 tag_observations.yaml，恢复连接上述失联标靶的关键视角观测 (设为 keep: true)。")
        else:
            msg = f"共视连通性检查完全通过！全部 {len(all_tags)} 个标靶处于统一连通拓扑网中。"

        return {
            "is_valid": is_valid,
            "all_tags": sorted(list(all_tags)),
            "connected_tags": main_component,
            "unconnected_tags": unconnected,
            "components": components,
            "adj_list": {k: sorted(list(v)) for k, v in adj_list.items()},
            "edge_counts": {f"{u}-{v}": c for (u, v), c in edge_counts.items()},
            "critical_bridges": critical_bridges,
            "valid_frames_count": len(frame_detections),
            "message": msg
        }

    @staticmethod
    def print_topology_report(covis_report: Dict[str, Any], stats: Dict[str, Any], valid_frames: List[str]):
        """打印详细共视拓扑分析诊断报告"""
        log.info("\n" + "=" * 70)
        log.info("           标靶共视连通图拓扑结构深度诊断报告")
        log.info("=" * 70)
        log.info(f" 状态评估: {'[ 连通健康 (PASS) ]' if covis_report['is_valid'] else '[ 断网告警 (FAIL) ]'}")
        log.info(f" 诊断说明: {covis_report['message']}")
        log.info(f" 参与帧数: {len(valid_frames)} 张图像满足 >= 2 标靶相对刚体约束")
        log.info("-" * 70)
        log.info(" [标靶节点与连通度 (Degree)]")
        for tid in covis_report.get("all_tags", []):
            neighbors = covis_report.get("adj_list", {}).get(tid, [])
            is_conn = tid in covis_report.get("connected_tags", [])
            status_str = "正常连通" if is_conn else "【断网孤立】"
            log.info(f"   - Tag #{tid:02d}: 相邻标靶 {neighbors} (度数: {len(neighbors)}) -> {status_str}")

        log.info("\n [共视桥梁与重叠帧数 (Edge Co-visibility Count)]")
        for edge_str, count in covis_report.get("edge_counts", {}).items():
            bridge_warn = " \033[93m[单图支撑关键桥梁 - 切勿剔除]\033[0m" if count == 1 else ""
            log.info(f"   - 标靶对 ({edge_str}): 在 {count} 帧图像中同时出现{bridge_warn}")

        if stats.get("excluded_items"):
            log.info("\n [当前已人工剔除的观测清单]")
            for exc in stats["excluded_items"]:
                note = f" (备注: {exc['note']})" if exc.get("note") else ""
                log.info(f"   - [{exc['image']}] Tag #{exc['tag_id']}{note}")
        log.info("=" * 70)


def print_topology_report(covis_report: Dict[str, Any], stats: Dict[str, Any], valid_frames: List[str]):
    """打印详细共视拓扑分析诊断报告"""
    CovisibilityGraphAnalyzer.print_topology_report(covis_report, stats, valid_frames)

