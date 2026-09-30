# -*- coding: utf-8 -*-
"""
Workspace Hub 标靶白名单与锚点状态机 (WhitelistState)
===================================================
管理工位 Tag 标靶白名单放行矩阵、物理边长与锚点世界坐标：
1. 工位 tag_whitelist.yaml 动态感知与内存缓存刷新
2. 30-Tag 矩阵交互放行、批量放行/清空与持久化
3. AprilTag 世界坐标物理锚点标注管理与单轴录入
4. 标靶物理名义边长 (marker_size_mm) 录入与校验
"""

import os
from typing import Any, Optional
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)


class WhitelistState:
    """标靶白名单与世界坐标锚点状态机"""

    def __init__(self, parent_hub_state: Any):
        self.hub = parent_hub_state

        # 白名单缓存与 mtime 感知
        self._whitelist_cache: dict = {}
        self._whitelist_cache_mtime: float = -1.0
        self._whitelist_cache_ws: str = ""

        # 白名单矩阵编辑状态
        self.whitelist_edit_mode = False
        self.whitelist_edit_ids: set[int] = set()

        # 锚点编辑状态
        self.anchor_mode = False
        self.anchor_map: dict = {}

        # 锚点独立弹窗状态
        self.anchor_modal_open = False
        self.anchor_modal_tag = -1
        self.anchor_modal_xyz = [0.0, 0.0, 0.0]
        self.anchor_modal_known = [False, False, False]
        self.anchor_axis_sel = -1
        self.anchor_axis_buf = ""

        # 标靶物理边长专属弹窗
        self.marker_size_modal_open = False
        self.marker_size_buf = ""

    def get_tag_whitelist(self) -> dict:
        """读取当前选中工位的 tag_whitelist.yaml (基于 mtime 自动感知外部编辑并刷新缓存)"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return {}
        path = self.hub.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        if not os.path.exists(path):
            return {}
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            return {}

        if (self._whitelist_cache_ws == ws.workspace_id
                and self._whitelist_cache
                and abs(mtime - self._whitelist_cache_mtime) < 1e-6):
            return self._whitelist_cache

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception:
            return {}

        self._whitelist_cache = data
        self._whitelist_cache_mtime = mtime
        self._whitelist_cache_ws = ws.workspace_id
        return data

    def refresh_whitelist_cache(self):
        """强制失效白名单缓存 (外部编辑器保存返回后立即刷新)"""
        self._whitelist_cache = {}
        self._whitelist_cache_mtime = -1.0
        self._whitelist_cache_ws = ""

    def ensure_tag_whitelist_file(self) -> str:
        """确保当前选中工位的白名单文件存在并有效"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return ""
        return self.hub.workspace_mgr.ensure_tag_whitelist(ws.workspace_id)

    def update_tag_anchor(self, tag_id: int, anchor_data: Any) -> tuple[bool, str]:
        """更新或清除当前工位中的 tag_anchors 物理坐标标注 (统一标准结构)"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return False, "未选择工位"
        ok, msg = self.hub.workspace_mgr.update_tag_anchor(ws.workspace_id, tag_id, anchor_data)
        if ok:
            self._reload_anchor_map()
            self.refresh_whitelist_cache()
        return ok, msg

    def enter_whitelist_edit(self):
        """进入编辑模式: 以当前 yaml 的 allowed_ids 为初始工作集合"""
        wl = self.get_tag_whitelist() or {}
        ids = set()
        for x in (wl.get("allowed_ids") or []):
            try:
                ids.add(int(x))
            except (TypeError, ValueError):
                continue
        self.whitelist_edit_ids = ids
        self.whitelist_edit_mode = True
        self.anchor_mode = False
        self._close_anchor_modal()

    def exit_whitelist_edit(self):
        """退出编辑模式 (写穿式保存, 无未保存残留)"""
        self.whitelist_edit_mode = False
        self.anchor_mode = False
        self._close_anchor_modal()

    def _save_whitelist_yaml(self):
        """写穿当前编辑集合到 tag_whitelist.yaml (保留其余字段, 更新 mtime 联动全链路缓存)"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return
        path = self.hub.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        doc = {}
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    doc = yaml.safe_load(f) or {}
            except Exception:
                doc = {}
        doc["workspace_id"] = ws.workspace_id
        doc["workspace_name"] = ws.name
        doc["allowed_ids"] = sorted(self.whitelist_edit_ids)
        if "tag_default_size_mm" not in doc:
            size_val = 35.5
            if os.path.isfile(ws.map_path):
                try:
                    with open(ws.map_path, "r", encoding="utf-8") as mf:
                        mdata = yaml.safe_load(mf) or {}
                    if mdata.get("marker_size_mm"):
                        size_val = float(mdata["marker_size_mm"])
                except Exception:
                    pass
            doc["tag_default_size_mm"] = size_val
        doc.setdefault("description", f"Workspace {ws.name} 标靶白名单配置")
        doc.setdefault("notes", "工位物理白名单恒启用 (名单内容即行为): allowed_ids 非空时仅放行名单内标靶 (权威约束)；留空 = 探索模式放行所有检测标靶")
        try:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        except Exception as e:
            log.warning(f"写回 tag_whitelist.yaml 失败: {e}")
            self.hub.set_toast(f"写回白名单失败: {e}")
            return
        self.refresh_whitelist_cache()

    def toggle_whitelist_id(self, tag_id: int) -> int:
        """切换芯片放行/拦截并写穿保存, 返回当前放行数"""
        if tag_id in self.whitelist_edit_ids:
            self.whitelist_edit_ids.discard(tag_id)
        else:
            self.whitelist_edit_ids.add(tag_id)
        self._save_whitelist_yaml()
        return len(self.whitelist_edit_ids)

    def whitelist_batch(self, action: str) -> int:
        """批量操作: 'all'=全量放行 0~29, 'clear'=清空 (探索模式); 返回当前放行数"""
        if action == "all":
            self.whitelist_edit_ids = set(range(30))
        elif action == "clear":
            self.whitelist_edit_ids = set()
        self._save_whitelist_yaml()
        return len(self.whitelist_edit_ids)

    def enter_anchor_mode(self):
        """进入锚点子模式: 芯片点击改为打开锚点弹窗"""
        self._reload_anchor_map()
        self.anchor_mode = True

    def exit_anchor_mode(self):
        self.anchor_mode = False
        self._close_anchor_modal()

    def _reload_anchor_map(self):
        """载入当前工位锚点 (Tag 锚点统一为工位独立数据, 未写穿前以当前标定为基础)"""
        from src.calibration.workspace_manager import load_workspace_anchor_tags, load_workspace_tag_anchors
        ws = self.hub.get_selected_workspace()
        if not ws:
            self.anchor_map = {}
            return
        m = load_workspace_anchor_tags(ws.workspace_dir)
        if m is None:
            m = load_workspace_tag_anchors(ws.workspace_dir)
        self.anchor_map = m or {}

    def get_anchor_map(self) -> dict:
        """获取当前工位完整的 AprilTag 物理锚点映射表 (优先取 anchor_tags.yaml，回退 tag_whitelist.yaml)"""
        self._reload_anchor_map()
        return self.anchor_map or {}

    def _persist_anchor_map(self) -> bool:
        """写穿当前工位锚点文件并同步写穿 tag_whitelist.yaml (统一采用标准结构)"""
        from src.calibration.workspace_manager import save_workspace_anchor_tags
        ws = self.hub.get_selected_workspace()
        if ws is None:
            return False
        ok = save_workspace_anchor_tags(ws, self.anchor_map)
        path = self.hub.workspace_mgr.get_tag_whitelist_path(ws.workspace_id)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg["tag_anchors"] = {
                    tid: {
                        "xyz_mm": [float(v) for v in a["xyz_mm"]],
                        "known": [bool(b) for b in a.get("known", [True, True, True])],
                        **({"frame_id": str(a["frame_id"])} if a.get("frame_id") else {}),
                    }
                    for tid, a in sorted(self.anchor_map.items())
                }
                # 自动将具有已知轴约束的 tag 加入 allowed_ids
                allowed_set = set(cfg.get("allowed_ids", []))
                for tid, a in self.anchor_map.items():
                    if any(a.get("known", [])):
                        allowed_set.add(tid)
                cfg["allowed_ids"] = sorted(list(allowed_set))
                if "tag_default_size_mm" not in cfg:
                    size_val = 35.5
                    if os.path.isfile(ws.map_path):
                        try:
                            with open(ws.map_path, "r", encoding="utf-8") as mf:
                                mdata = yaml.safe_load(mf) or {}
                            if mdata.get("marker_size_mm"):
                                size_val = float(mdata["marker_size_mm"])
                        except Exception:
                            pass
                    cfg["tag_default_size_mm"] = size_val
                with open(path, "w", encoding="utf-8") as f:
                    yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                self.refresh_whitelist_cache()
            except Exception as e:
                log.warning(f"同步 tag_whitelist.yaml tag_anchors 异常: {e}")
        return ok

    def _close_anchor_modal(self):
        self.anchor_modal_open = False
        self.anchor_modal_tag = -1
        self.anchor_axis_sel = -1
        self.anchor_axis_buf = ""

    def open_anchor_editor(self, tag_id: int):
        """打开 Tag 专属坐标编辑弹窗: 以工位统一锚点结构为草稿"""
        self._reload_anchor_map()
        entry = self.anchor_map.get(tag_id) if self.anchor_map else None
        if not entry:
            wl = self.get_tag_whitelist()
            anchors = wl.get("tag_anchors", {}) if isinstance(wl, dict) else {}
            cand = anchors.get(tag_id) or anchors.get(str(tag_id))
            if isinstance(cand, dict) and "xyz_mm" in cand:
                entry = cand
        if entry:
            self.anchor_modal_xyz = [float(v) for v in entry["xyz_mm"]]
            self.anchor_modal_known = [bool(b) for b in entry.get("known", [True, True, True])]
        else:
            self.anchor_modal_xyz = [0.0, 0.0, 0.0]
            self.anchor_modal_known = [False, False, False]
        self.anchor_modal_tag = tag_id
        self.anchor_modal_open = True
        self.anchor_axis_sel = 0
        self.anchor_axis_buf = ""

    def anchor_axis_select(self, axis: int):
        """选中输入轴 (自动提交上一轴缓冲)"""
        self._commit_axis_buffer()
        self.anchor_axis_sel = axis
        self.anchor_axis_buf = ""

    def _commit_axis_buffer(self):
        buf = self.anchor_axis_buf.strip()
        if self.anchor_axis_sel < 0 or not buf:
            self.anchor_axis_buf = ""
            return
        try:
            self.anchor_modal_xyz[self.anchor_axis_sel] = float(buf)
            self.anchor_modal_known[self.anchor_axis_sel] = True
        except ValueError:
            self.hub.set_toast(f"轴 {'XYZ'[self.anchor_axis_sel]} 数值无效: {buf}")
        self.anchor_axis_buf = ""

    def anchor_pad_key(self, label: str):
        """锚点键盘: 数字/小数点入缓冲, '-/+' 切换符号, '清空'/'退格' 编辑, '确认' 提交当前轴"""
        if label == "退格":
            self.anchor_axis_buf = self.anchor_axis_buf[:-1]
            return
        if label == "清空":
            self.anchor_axis_buf = ""
            return
        if label == "确认":
            self._commit_axis_buffer()
            return
        if label == "-/+":
            if self.anchor_axis_buf.startswith("-"):
                self.anchor_axis_buf = self.anchor_axis_buf[1:]
            else:
                self.anchor_axis_buf = "-" + self.anchor_axis_buf
            return
        if label == "." and "." in self.anchor_axis_buf:
            return
        self.anchor_axis_buf = (self.anchor_axis_buf + label)[-12:]

    def anchor_axis_clear(self, axis: int):
        """清除单轴已知标记 (支持部分已知)"""
        self.anchor_modal_known[axis] = False

    def anchor_known_count(self) -> int:
        return sum(1 for b in self.anchor_modal_known if b)

    def save_anchor_modal(self) -> tuple[bool, str]:
        """保存弹窗: 已知轴 ≥1 写穿工位锚点文件；全未知 = 从工位锚点中删除该条目"""
        self._commit_axis_buffer()
        tid = self.anchor_modal_tag
        n = self.anchor_known_count()
        self._reload_anchor_map()
        if n == 0:
            self.anchor_map.pop(tid, None)
            ok = self._persist_anchor_map()
            self._close_anchor_modal()
            return (ok, f"Tag #{tid} 锚点已清除") if ok else (False, "锚点写回失败")
        self.anchor_map[tid] = {"xyz_mm": [float(v) for v in self.anchor_modal_xyz],
                                "known": [bool(b) for b in self.anchor_modal_known]}
        ok = self._persist_anchor_map()
        self._close_anchor_modal()
        if ok:
            return True, f"Tag #{tid} 锚点已保存到本工位 ({n}/3 轴已知)"
        return False, "锚点写回失败"

    def clear_anchor_modal(self) -> tuple[bool, str]:
        """删除当前 Tag 的锚点并写穿工位锚点文件"""
        tid = self.anchor_modal_tag
        self._reload_anchor_map()
        self.anchor_map.pop(tid, None)
        ok = self._persist_anchor_map()
        self._close_anchor_modal()
        return (True, f"Tag #{tid} 锚点已清除") if ok else (False, "锚点写回失败")

    def cancel_anchor_modal(self):
        self._close_anchor_modal()

    def get_workspace_marker_size(self) -> Optional[float]:
        """获取当前工位显式配置的标靶物理边长 (mm)"""
        ws = self.hub.get_selected_workspace()
        if not ws:
            return None
        from src.calibration.workspace_manager import load_workspace_marker_size_mm
        return load_workspace_marker_size_mm(ws.workspace_dir)

    def open_marker_size_editor(self):
        """打开标靶物理边长专属编辑模态窗"""
        cur_sz = self.get_workspace_marker_size()
        self.marker_size_buf = f"{cur_sz:.3f}" if cur_sz and cur_sz > 0 else ""
        self.marker_size_modal_open = True

    def save_marker_size_modal(self) -> tuple[bool, str]:
        """提交保存标靶物理边长 (写穿工位 tag_whitelist.yaml)"""
        buf = self.marker_size_buf.strip()
        if not buf:
            return False, "标靶边长不能为空"
        try:
            val = float(buf)
        except ValueError:
            return False, f"非法数值: {buf}"
        if val <= 0:
            return False, "标靶边长必须大于 0 mm"
        ws = self.hub.get_selected_workspace()
        if not ws:
            return False, "未选择工位"
        ok, msg = self.hub.workspace_mgr.update_workspace_marker_size(ws.workspace_id, val)
        if ok:
            self.refresh_whitelist_cache()
            self.marker_size_modal_open = False
            self.marker_size_buf = ""
        return ok, msg

    def cancel_marker_size_modal(self):
        """取消标靶物理边长编辑"""
        self.marker_size_modal_open = False
        self.marker_size_buf = ""

    def marker_size_pad_key(self, label: str):
        """标靶边长键盘输入分发"""
        if label == "退格":
            self.marker_size_buf = self.marker_size_buf[:-1]
            return
        if label == "清空":
            self.marker_size_buf = ""
            return
        if label == "确认":
            ok, msg = self.save_marker_size_modal()
            self.hub.set_toast(msg)
            return
        if label == "." and "." in self.marker_size_buf:
            return
        if label in "0123456789.":
            self.marker_size_buf = (self.marker_size_buf + label)[-8:]
