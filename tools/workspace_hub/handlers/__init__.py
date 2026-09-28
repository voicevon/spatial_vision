# -*- coding: utf-8 -*-
"""
Workspace Hub 控制器事件处理器包 (Handlers Package)
=================================================
将巨型控制器拆解为独立的业务领域处理机：
- ModalHandler: 模态弹窗点击处理
- WorkspaceHandler: 工位增删改查、克隆与一致性自愈
"""

from tools.workspace_hub.handlers.modal_handler import ModalHandler
from tools.workspace_hub.handlers.workspace_handler import WorkspaceHandler

__all__ = [
    "ModalHandler",
    "WorkspaceHandler",
]
