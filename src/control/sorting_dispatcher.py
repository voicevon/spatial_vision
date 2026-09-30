#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务分选与调度核心控制器 (SortingDispatcher)
============================================
负责衔接“视觉感知层事实”与“运动执行层任务”的核心纽带：
1. 工艺角色与槽位抽象: 自动从 Workspace 的 Smart ROI 中抽取 Source (进料) 与 Destination (落料槽);
2. 智能抓取挑选: 优先挑选最顶层 (is_topmost)、无遮挡、高置信度的物料;
3. 槽位分选路由: 根据物料品质分级 (Grade: A/B/C) 匹配可用目标槽;
4. 满溢监控防错: 追踪各槽位实时根数计数 (current_count)，达到容量上限 (capacity_max) 自动阻断或切换备用槽;
5. 生成抽象任务: 产出结构化 ScaraPickTask，交由 ScaraMotionPlanner 翻译执行。
"""

from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

from src.workspace.roi_manager import RoiDefinition
from src.vision.asparagus_analyzer import AsparagusTarget
from src.control.scara_motion_planner import ScaraPickTask
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DestinationSlot:
    """目的地落料槽运行状态模型"""
    slot_index: int
    roi_id: str
    name: str
    drop_x: float
    drop_y: float
    drop_z: float = 0.0
    drop_r: float = 0.0
    capacity_max: int = 5                # 该槽最大容纳根数
    current_count: int = 0               # 当前已落料根数
    target_grades: List[str] = field(default_factory=lambda: ["A", "B", "C"])  # 允许接纳的品级
    enabled: bool = True

    @property
    def is_full(self) -> bool:
        """是否已放满"""
        return self.current_count >= self.capacity_max

    @property
    def remaining_capacity(self) -> int:
        """剩余可用容量"""
        return max(0, self.capacity_max - self.current_count)


class SortingDispatcher:
    """分选业务调度器"""

    def __init__(
        self,
        rois: Optional[List[RoiDefinition]] = None,
        default_drop_x: float = 220.0,
        default_drop_y: float = 0.0,
        default_drop_z: float = 0.0,
    ):
        self.default_drop_x = float(default_drop_x)
        self.default_drop_y = float(default_drop_y)
        self.default_drop_z = float(default_drop_z)

        self.source_rois: List[RoiDefinition] = []
        self.destination_slots: Dict[int, DestinationSlot] = {}

        if rois:
            self.load_from_rois(rois)

    def load_from_rois(self, rois: List[RoiDefinition]):
        """从 Smart ROI 集合中解析并初始化源头与目的地槽位"""
        self.source_rois.clear()
        self.destination_slots.clear()

        dest_list: List[Tuple[int, DestinationSlot]] = []

        for r in rois:
            if not r.enabled:
                continue

            # 判定 Source: 显式声明 role="source" 或类别为 belt
            if r.role == "source" or (r.role == "general" and r.category == "belt"):
                self.source_rois.append(r)

            # 判定 Destination: 显式声明 role="destination" 或类别为 tray/wheel
            elif r.role == "destination" or (r.role == "general" and r.category in ("tray", "wheel")):
                binding = r.binding or {}
                slot_idx = int(binding.get("slot_index", len(dest_list)))
                cap_max = int(binding.get("capacity_max", 5))

                # 落料坐标优先取几何中心
                c_xyz = r.center_xyz_mm if r.center_xyz_mm else [self.default_drop_x, self.default_drop_y, self.default_drop_z]
                dx, dy, dz = float(c_xyz[0]), float(c_xyz[1]), float(c_xyz[2])

                # 允许接纳的品级
                raw_grades = binding.get("grades")
                if isinstance(raw_grades, list):
                    grades = [str(g).upper() for g in raw_grades]
                else:
                    grades = ["A", "B", "C"]

                slot = DestinationSlot(
                    slot_index=slot_idx,
                    roi_id=r.roi_id,
                    name=r.name,
                    drop_x=dx,
                    drop_y=dy,
                    drop_z=dz,
                    capacity_max=cap_max,
                    current_count=0,
                    target_grades=grades,
                    enabled=True,
                )
                dest_list.append((slot_idx, slot))

        # 按 slot_index 升序存入字典
        dest_list.sort(key=lambda x: x[0])
        for s_idx, s in dest_list:
            self.destination_slots[s_idx] = s

        log.info(
            "[Dispatcher] 装载就绪: 检出 %d 个源头区域 (Source), %d 个落料槽位 (Destination)",
            len(self.source_rois), len(self.destination_slots)
        )

    def select_best_target(self, targets: List[AsparagusTarget]) -> Optional[AsparagusTarget]:
        """
        在所有候选芦笋中，挑选最优、最安全的抓取目标。
        规则：
        1. 必须置信度达标 (confidence > 0.5)；
        2. 优先挑选最顶层且无遮挡的目标 (is_topmost=True)；
        3. 多根均在顶层时，挑选相对台面凸起高度最高、直径适宜、品质优秀的目标。
        """
        if not targets:
            return None

        # 基础置信度过滤
        valid_targets = [t for t in targets if t.confidence >= 0.5]
        if not valid_targets:
            return None

        # 优先过滤顶层目标
        topmost = [t for t in valid_targets if t.is_topmost]
        pool = topmost if topmost else valid_targets

        # 排序策略: 顶层优先 > 凸起高度最高 > 置信度高
        def score(t: AsparagusTarget) -> float:
            base_score = 1000.0 if t.is_topmost else 0.0
            base_score += float(t.rel_height_mm) * 10.0
            base_score += float(t.confidence) * 50.0
            if t.grade == "A":
                base_score += 30.0
            elif t.grade == "B":
                base_score += 15.0
            return base_score

        best = max(pool, key=score)
        return best

    def assign_destination_slot(self, target: AsparagusTarget) -> Optional[DestinationSlot]:
        """
        根据物料品质评级，路由匹配空闲且未放满的落料槽。
        若首选槽已满，寻找其他兼容该品级且有余量的备用槽。
        若全满，返回 None (触发产线满溢等待)。
        """
        # 若未配置任何专属目的地槽位，提供全局默认虚拟槽位
        if not self.destination_slots:
            return DestinationSlot(
                slot_index=0,
                roi_id="default_drop",
                name="默认全局落料点",
                drop_x=self.default_drop_x,
                drop_y=self.default_drop_y,
                drop_z=self.default_drop_z,
                capacity_max=999999,
                current_count=0,
            )

        t_grade = str(target.grade).upper() if target.grade else "A"

        # 1. 精确匹配当前品级且有余量的槽位
        candidates: List[DestinationSlot] = []
        for slot in self.destination_slots.values():
            if slot.enabled and not slot.is_full and t_grade in slot.target_grades:
                candidates.append(slot)

        if candidates:
            # 专属匹配策略：
            # 1. 专属性优先: target_grades 越专一权重越高 (如专收 A 级的优先于收 A/B/C 的通用槽)
            # 2. 聚类归并优先: 同等专属性下，优先连续填满已开箱的槽位
            # 3. 槽位编号升序优先
            return min(candidates, key=lambda s: (len(s.target_grades), 0 if s.current_count > 0 else 1, s.slot_index))

        # 2. 备用兜底：寻找接纳所有品级的通用溢出槽位
        for slot in self.destination_slots.values():
            if slot.enabled and not slot.is_full and set(["A", "B", "C"]).issubset(set(slot.target_grades)):
                return slot

        # 所有兼容槽位均满溢
        log.warning("[Dispatcher] 槽位满溢警报: 无法为品级 [%s] 找到未满目标槽！", t_grade)
        return None

    def dispatch_cycle(
        self,
        detected_targets: List[AsparagusTarget]
    ) -> Tuple[Optional[AsparagusTarget], Optional[DestinationSlot], Optional[ScaraPickTask]]:
        """
        运行一次完整的生产分选决策周拍 (Decision Cycle)。
        返回: (选中物料, 目标槽位, SCARA任务结构)
        若无物料或槽位已满，返回 None。
        """
        # 1. 挑选物料
        target = self.select_best_target(detected_targets)
        if not target:
            return None, None, None

        # 2. 匹配槽位
        slot = self.assign_destination_slot(target)
        if not slot:
            return target, None, None

        # 3. 构造任务
        task = ScaraPickTask(
            target_id=target.id,
            pick_x=float(target.robot_x),
            pick_y=float(target.robot_y),
            pick_z=float(target.robot_z),
            pick_r=float(target.robot_r),
            drop_x=float(slot.drop_x),
            drop_y=float(slot.drop_y),
            drop_z=float(slot.drop_z),
            drop_r=float(slot.drop_r),
            slot_index=slot.slot_index,
            tag_info=f"物料#{target.id}({target.grade}级) -> 槽位#{slot.slot_index + 1}[{slot.name}]",
        )
        return target, slot, task

    def confirm_placed(self, slot_index: int):
        """当下位机或机械臂回报落料动作完成时调用，推进槽位计数值"""
        if slot_index in self.destination_slots:
            slot = self.destination_slots[slot_index]
            slot.current_count += 1
            log.info(
                "[Dispatcher] 槽位 #%d [%s] 计数累加: %d / %d",
                slot_index + 1, slot.name, slot.current_count, slot.capacity_max
            )

    def reset_slot(self, slot_index: int):
        """人工或产线机构换箱后清零指定槽位"""
        if slot_index in self.destination_slots:
            self.destination_slots[slot_index].current_count = 0
            log.info("[Dispatcher] 槽位 #%d 计数已复位清零", slot_index + 1)

    def reset_all_slots(self):
        """清零所有槽位"""
        for s in self.destination_slots.values():
            s.current_count = 0
        log.info("[Dispatcher] 全局槽位计数均已复位")
