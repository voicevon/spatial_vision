#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SCARA 抓取生产工作台 (SCARA Production Studio)
==============================================
专用于上料抓取工位的产线实时运行工作台：
1. 监听进料皮带 (Smart ROI: role='source') 芦笋物料;
2. 调度器 (SortingDispatcher) 动态决策目标落料槽位与防溢控制;
3. 轨迹规划器 (ScaraMotionPlanner) 实时输出防撞 G-code;
4. 驱动 SCARA 机械臂串口执行全自动搬运闭环。
"""

from tools.scara_production.app import ScaraProductionApp, main

__all__ = ["ScaraProductionApp", "main"]
