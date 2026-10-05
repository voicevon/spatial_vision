# -*- coding: utf-8 -*-
"""
Workspace Hub 工位生命周期业务处理器 (WorkspaceHandler)
=====================================================
负责：
1. 工位创建 (支持中文别名与自动化目录骨架搭建)
2. 工位克隆 (深拷贝标定数据与相对坐标系树)
3. 工位重命名与备注说明即时修改
4. 工位安全删除与唯一工位防误删保护
5. 物理磁盘与元数据一致性自愈 (自动修复图片计数与统计指标)
6. 跨平台操作系统文件浏览器打开工位目录
"""

import os
import sys
import subprocess
from typing import Any

from src.ui.dialog_utils import (
    show_confirm_dialog, show_text_input_dialog, show_info_dialog,
    prompt_create_workspace_dialog
)
from src.devices.camera_service import CameraService
from src.workspace.health_auditor import audit_workspace, audit_all_workspaces


class WorkspaceHandler:
    """工位生命周期与数据一致性处理器"""

    def __init__(self, app: Any):
        self.app = app
        self.state = app.state
        self.workspace_mgr = app.workspace_mgr

    def handle_open_directory(self):
        """在系统资源管理器中打开工位目录"""
        ws = self.state.get_selected_workspace()
        if ws and os.path.exists(ws.workspace_dir):
            try:
                if sys.platform == "win32":
                    os.startfile(ws.workspace_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", ws.workspace_dir])
                else:
                    subprocess.run(["xdg-open", ws.workspace_dir])
                self.state.set_toast(f"已在资源管理器中打开: {ws.name}")
            except Exception as e:
                self.state.set_toast(f"打开目录异常: {e}")

    def handle_rename_workspace(self):
        """修改 Workspace 显示名称 (支持中文)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        new_name = show_text_input_dialog(
            "修改 Workspace 名称",
            f"请输入 Workspace【{ws.name}】的新显示名称\n(支持中文、英文、数字，如: 1号机台主标定):",
            initial=ws.name
        )
        if new_name and new_name != ws.name:
            self.state.rename_current_workspace(new_name)

    def handle_edit_description(self):
        """修改 Workspace 备注说明 (支持中文单行文本)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            return

        cur_desc = getattr(ws, "description", "") or ""
        new_desc = show_text_input_dialog(
            "修改工位备注",
            f"请输入工位【{ws.name}】的备注信息 (单行文本):",
            initial=cur_desc
        )
        if new_desc is not None:
            self.state.update_current_workspace_description(new_desc.strip())

    def handle_create_workspace(self):
        """新建 Workspace (支持中文名称与相机硬件/分辨率规格配置弹窗)"""
        idx = len(self.state.workspaces) + 1
        default_alias = f"Workspace_{idx}"

        devs = CameraService.probe_available_devices()
        form_res = prompt_create_workspace_dialog(
            default_alias=default_alias,
            available_devices=devs
        )
        if not form_res:
            self.state.set_toast("已取消新建 Workspace。")
            return

        chosen_name = form_res["alias"]
        cam_type = form_res["camera_type"]
        cam_serial = form_res["camera_serial"]
        cam_res = form_res["camera_resolution"]

        new_ws = self.workspace_mgr.create_workspace(
            alias=chosen_name,
            description=f"Workspace {chosen_name}",
            camera_type=cam_type,
            camera_serial=cam_serial,
            camera_resolution=cam_res
        )
        self.state.refresh_workspaces()
        target_idx = 0
        for i, s in enumerate(self.state.workspaces):
            if s.workspace_id == new_ws.workspace_id:
                target_idx = i
                break
        self.state.select_workspace_at_index(target_idx)
        self.state.set_toast(f"已新建工位【{new_ws.name}】: {cam_type} @ {new_ws.resolution_str}，按 [C] 开始采图！")

    def handle_clone_workspace(self):
        """克隆 Workspace (支持中文名称弹窗)"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何 Workspace，无法克隆！")
            return

        default_clone_name = f"{ws.name}_对照组"
        chosen_name = show_text_input_dialog(
            "克隆 Workspace",
            f"请输入克隆后的新 Workspace 名称 (基于原 Workspace【{ws.name}】):",
            initial=default_clone_name
        )
        if not chosen_name:
            return

        cloned = self.workspace_mgr.clone_workspace(ws.workspace_id, new_alias=chosen_name)
        if cloned:
            # 立即刷新 Workspace 列表
            self.state.refresh_workspaces()
            target_idx = 0
            for i, s in enumerate(self.state.workspaces):
                if s.workspace_id == cloned.workspace_id:
                    target_idx = i
                    break
            self.state.select_workspace_at_index(target_idx)
            self.state.set_toast(f"已成功克隆 Workspace: 【{cloned.name}】并定位至新 Workspace！")
        else:
            self.state.set_toast("克隆 Workspace 失败，请检查源目录！")

    def handle_delete_workspace(self):
        """删除当前选中的 Workspace"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何 Workspace，无法删除！")
            return

        if len(self.state.workspaces) <= 1:
            self.state.set_toast("至少需保留一个工位，禁止删除唯一工位！")
            return

        confirmed = show_confirm_dialog(
            "确认删除 Workspace",
            f"确定要永久删除工位【{ws.name}】吗？\n\n物理ID: {ws.workspace_id}\n此操作将删除该工位的所有图片和标定数据，不可恢复！"
        )
        if not confirmed:
            self.state.set_toast("已取消删除操作。")
            return

        ok, msg = self.workspace_mgr.delete_workspace(ws.workspace_id)
        if ok:
            self.state.refresh_workspaces()
            self.state.set_toast(f"已成功删除工位: 【{ws.name}】")
        else:
            self.state.set_toast(f"删除工位失败: {msg}")

    def handle_sync_data_consistency(self):
        """核验物理磁盘与元数据一致性，修剪幽灵标靶，重新扫描并自动自愈同步"""
        ws = self.state.get_selected_workspace()
        if not ws:
            self.state.set_toast("未选中任何工位，无法核验！")
            return

        # 调用领域层单工位健康体检与自愈
        res = audit_workspace(ws, auto_fix=True)

        # 刷新相册与列表数据
        self.state.gallery.load_current_workspace_images()
        self.state.gallery.load_prod_images()
        self.state.refresh_workspaces()

        if res["actions_taken"]:
            acts_str = "；".join(res["actions_taken"])
            self.state.set_toast(f"元数据自愈成功: {acts_str}")
        else:
            self.state.set_toast(f"一致性核验完成: 【{ws.name}】物理与元数据已是最新，无幽灵标靶")

    def handle_audit_all_workspaces(self):
        """全工位深度健康体检与自愈，输出详细报告并弹窗反馈"""
        self.state.set_toast("正在执行全工位健康体检与自愈，请稍候...")
        res = audit_all_workspaces(self.workspace_mgr, auto_fix=True)

        # 刷新当前选中的工位与列表视图
        self.state.gallery.load_current_workspace_images()
        self.state.gallery.load_prod_images()
        self.state.refresh_workspaces()

        # 弹出完成 Toast
        self.state.set_toast(
            f"全工位体检完成！自愈 {res['healed_workspaces']} 个工位，清理 {res['total_ghost_pruned']} 枚幽灵标靶"
        )

        # 弹出原生信息提示框展示报告摘要
        show_info_dialog("全工位健康体检与自愈报告", res["summary_text"])
