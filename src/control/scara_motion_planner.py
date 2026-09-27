#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCARA 机械臂轨迹与 G-code 规划器 (ScaraMotionPlanner)
=====================================================
将视觉感知层输出的 3D/6D 几何位姿 (Pick Pose) 与目标落料位姿 (Place Pose)
翻译为具体的 SCARA 机械臂 G-code 指令序列。

【架构原则】:
1. 属于控制与驱动层 (Motion & Driver Layer)，绝不侵入视觉感知层；
2. 明确区分拾取点 (Pick) 与落料点 (Place)，支持多槽位多态落料；
3. 严格遵循三段式防撞安全高度逻辑 (过渡安全高度 -> 垂直下探 -> 夹爪动作 -> 提刀过渡 -> 落料)。
"""

from typing import List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ScaraPickTask:
    """单个 SCARA 拾取放料任务定义"""
    target_id: int
    pick_x: float
    pick_y: float
    pick_z: float
    pick_r: float                        # 夹爪偏航对齐角 (deg)
    drop_x: float
    drop_y: float
    drop_z: float = 0.0
    drop_r: float = 0.0
    slot_index: Optional[int] = None     # 目标落料槽位索引 (如 0~7)
    tag_info: str = ""                   # 标定来源或描述


class ScaraMotionPlanner:
    """SCARA 机械臂轨迹与 G-code 规划器"""

    def __init__(
        self,
        safe_z: float = 80.0,
        feedrate_xy: int = 4000,
        feedrate_z: int = 1500,
        dwell_ms: int = 200,
        gripper_open_cmd: str = "M3",
        gripper_close_cmd: str = "M4",
    ):
        self.safe_z = float(safe_z)
        self.feedrate_xy = int(feedrate_xy)
        self.feedrate_z = int(feedrate_z)
        self.dwell_ms = int(dwell_ms)
        self.gripper_open_cmd = gripper_open_cmd
        self.gripper_close_cmd = gripper_close_cmd

    def plan(self, task: ScaraPickTask) -> str:
        """根据任务结构体生成防撞 G-code 指令文本"""
        lines: List[str] = []
        lines.append("; ==============================================================================")
        lines.append(f"; SCARA 抓取指令 (物料 #{task.target_id} -> 目标落料: X{task.drop_x:.1f} Y{task.drop_y:.1f})")
        if task.slot_index is not None:
            lines.append(f"; [调度路由] 目标槽位: #{task.slot_index + 1} (slot_index={task.slot_index})")
        if task.tag_info:
            lines.append(f"; [标定与状态] {task.tag_info}")
        lines.append("; ==============================================================================")
        lines.append("G90                     ; 绝对坐标模式")
        lines.append(f"G0 Z{self.safe_z:.1f} F{self.feedrate_xy}          ; 提升至安全过渡高度 (避免平移撞料)")
        lines.append(f"G0 X{task.pick_x:.2f} Y{task.pick_y:.2f} R{task.pick_r:.2f} F{self.feedrate_xy} ; 快速对准物料抓取中轴")
        lines.append(f"{self.gripper_open_cmd}                      ; 预先张开夹爪")
        lines.append(f"G1 Z{task.pick_z:.2f} F{self.feedrate_z}          ; 垂直平稳下探至夹持高度")
        lines.append(f"{self.gripper_close_cmd}                      ; 闭合夹爪牢固夹持")
        lines.append(f"G4 P{self.dwell_ms}                 ; 保压延时确保夹牢")
        lines.append(f"G0 Z{self.safe_z:.1f} F{self.feedrate_xy}          ; 提起物料脱离来料区")
        lines.append(f"G0 X{task.drop_x:.2f} Y{task.drop_y:.2f} R{task.drop_r:.2f} F{self.feedrate_xy} ; 移动至指定落料槽上方")
        lines.append(f"{self.gripper_open_cmd}                      ; 释放物料落料")
        lines.append(f"G0 Z{self.safe_z:.1f} F{self.feedrate_xy}          ; 提刀复位至安全过渡层")
        return "\n".join(lines)

    def plan_from_target(
        self,
        target_id: int,
        pick_x: float,
        pick_y: float,
        pick_z: float,
        pick_r: float,
        drop_x: float = 220.0,
        drop_y: float = 0.0,
        drop_z: float = 0.0,
        slot_index: Optional[int] = None,
        tag_info: str = "",
    ) -> str:
        """便捷调用入口：直接传入散列几何参数"""
        task = ScaraPickTask(
            target_id=target_id,
            pick_x=pick_x,
            pick_y=pick_y,
            pick_z=pick_z,
            pick_r=pick_r,
            drop_x=drop_x,
            drop_y=drop_y,
            drop_z=drop_z,
            slot_index=slot_index,
            tag_info=tag_info,
        )
        return self.plan(task)

    def generate_pick_gcode(
        self,
        target: Any,
        drop_x: float = 220.0,
        drop_y: float = 0.0,
        drop_z: float = 0.0,
        drop_r: float = 0.0,
        slot_index: Optional[int] = None,
    ) -> str:
        """从视觉目标 AsparagusTarget 提取几何位姿与标定来源并生成 G-code"""
        source = getattr(target, "calibration_source", "uncalibrated")
        if source == "tag_online":
            src_str = "AprilTag 在线外参标定 [高精可靠]"
        elif source == "tag_cached":
            src_str = "历史缓存标定矩阵 [离线兜底]"
        elif source == "hand_eye":
            src_str = "手工标定矩阵 [固定补偿]"
        else:
            src_str = "UNCALIBRATED 安全警告 (尚未执行标定，当前坐标以原点基准相对量输出)"

        task = ScaraPickTask(
            target_id=getattr(target, "target_id", 1),
            pick_x=getattr(target, "robot_x", 0.0),
            pick_y=getattr(target, "robot_y", 0.0),
            pick_z=getattr(target, "robot_z", 0.0),
            pick_r=getattr(target, "robot_r", 0.0),
            drop_x=drop_x,
            drop_y=drop_y,
            drop_z=drop_z,
            drop_r=drop_r,
            slot_index=slot_index,
            tag_info=src_str,
        )
        return self.plan(task)
