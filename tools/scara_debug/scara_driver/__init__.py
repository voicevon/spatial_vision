# -*- coding: utf-8 -*-
"""
scara_driver — Flux Loader SCARA 机械臂核心控制库

对外公开 API（外部视觉算法或自动化脚本直接导入）：

    from scara_driver import ScaraRobot, ScaraRobotConfig, Pose, JointAngles
    from scara_driver.comm import SerialTransceiver, MarlinProtocolHandler

内部实现模块（不对外暴露）：
    MarlinProtocolParser, GripperSubsystem, StepProfile
"""

from .models import JointAngles, LimitSwitchStatus, Pose
from .config import ScaraRobotConfig
from .kinematics import ScaraKinematics, ReachabilityError
from .comm import (
    AbstractTransceiver,
    SerialTransceiver,
    MarlinProtocolHandler,
)
from .subsystems import GripperSubsystem
from .robot import ScaraRobot
from .workflows import PickAndPlaceWorkflow, PickAndPlaceConfig
from .jog import JogController

__all__ = [
    # 领域模型
    "Pose",
    "JointAngles",
    "LimitSwitchStatus",
    # 配置
    "ScaraRobotConfig",
    # 运动学
    "ScaraKinematics",
    "ReachabilityError",
    # 通信
    "AbstractTransceiver",
    "SerialTransceiver",
    "MarlinProtocolHandler",
    # 子系统
    "GripperSubsystem",
    # 核心实体
    "ScaraRobot",
    # 工作流
    "PickAndPlaceWorkflow",
    "PickAndPlaceConfig",
    # 交互
    "JogController",
]
