# -*- coding: utf-8 -*-
"""
工位基础设施与空间几何场景管理领域包 (Workspace & Spatial Scene Domain)
========================================================================
集中导出工位实体、工位管理器、标靶白名单配置及空间几何管理器：
- Workspace: 工位自包含领域实体
- WorkspaceManager: 工位生命周期仓储服务
- Tag 白名单与锚点管理函数
- CoordinateTreeManager / RoiSpaceManager: 空间几何场景
- audit_workspace / audit_all_workspaces: 健康体检
"""

from src.workspace.workspace_entity import Workspace
from src.workspace.workspace_manager import (
    WorkspaceManager,
    load_workspace_coordinate_manager,
    load_workspace_roi_manager,
)
from src.workspace.tag_whitelist_manager import (
    load_workspace_tag_whitelist,
    load_workspace_marker_size_mm,
    load_workspace_tag_config,
    save_workspace_tag_config,
    load_workspace_anchor_tags,
)
from src.workspace.coordinate_manager import CoordinateTreeManager
from src.workspace.roi_manager import RoiSpaceManager
from src.workspace.health_auditor import audit_workspace, audit_all_workspaces

__all__ = [
    "Workspace",
    "WorkspaceManager",
    "CoordinateTreeManager",
    "RoiSpaceManager",
    "audit_workspace",
    "audit_all_workspaces",
    "load_workspace_tag_whitelist",
    "load_workspace_marker_size_mm",
    "load_workspace_tag_config",
    "save_workspace_tag_config",
    "load_workspace_anchor_tags",
    "load_workspace_coordinate_manager",
    "load_workspace_roi_manager",
]
