# -*- coding: utf-8 -*-
"""
工位物理标靶准入与世界锚点配置管理 (Tag Whitelist Manager)
======================================================
管理工位级标靶单一真理源 (tag_whitelist.yaml):
1. 标靶放行白名单 (load_workspace_tag_whitelist)
2. 标靶物理边长权威来源 (load_workspace_marker_size_mm)
3. 标靶综合先验 Schema 读写与规范化清洗 (load_workspace_tag_config / save_workspace_tag_config)
4. 世界绝对坐标锚点提取 (load_workspace_anchor_tags)
"""

import os
from typing import Any, List, Optional, Dict
import yaml

from src.utils.logger import get_logger
from src.workspace.workspace_entity import Workspace

log = get_logger(__name__)


def load_workspace_tag_whitelist(workspace_dir: str) -> List[int]:
    """
    读取工位 tag_whitelist.yaml 的放行标靶 (工位级物理白名单, 恒启用 — 名单内容即行为):
    - tags 非空 → 返回 sorted(tags.keys()) (该工位检测过滤的权威约束)
    - tags 为空/文件缺失/解析失败 → 返回 [] (探索模式: 不加额外过滤)
    """
    cfg = load_workspace_tag_config(workspace_dir)
    tags = cfg.get("tags") or {}
    return sorted(list(tags.keys()))


def load_workspace_marker_size_mm(workspace_dir: str) -> Optional[float]:
    """
    读取工位 tag_whitelist.yaml 的 tag_default_size_mm (标靶物理边长, 唯一权威来源).

    规则 (FR-9.x):
      - 文件存在且 tag_default_size_mm 为正数 → 返回 float(mm)
      - 文件缺失 / 字段缺失 / 非正数 / 解析失败 → 返回 None
        (调用方需主动报错, 禁止默认 50.0 等兜底值)
    """
    path = os.path.join(workspace_dir, "tag_whitelist.yaml")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        log.warning(f"[WS] 解析工位标靶边长失败 ({path}): {e}")
        return None
    v = data.get("tag_default_size_mm")
    if v is None:
        # 自愈机制: 尝试从当前工位 tags_map.yaml 继承已标定的 marker_size_mm
        map_path = os.path.join(workspace_dir, "tags_map.yaml")
        if os.path.isfile(map_path):
            try:
                with open(map_path, "r", encoding="utf-8") as mf:
                    mdata = yaml.safe_load(mf) or {}
                mv = mdata.get("marker_size_mm")
                if mv is not None and float(mv) > 0:
                    f_val = float(mv)
                    data["tag_default_size_mm"] = f_val
                    try:
                        with open(path, "w", encoding="utf-8") as wf:
                            yaml.dump(data, wf, allow_unicode=True, default_flow_style=False, sort_keys=False)
                        log.info(f"[WS] 从 tags_map.yaml 自愈回填 tag_default_size_mm={f_val} mm 至 {path}")
                    except Exception as we:
                        log.warning(f"[WS] 自愈写回 tag_whitelist.yaml 失败: {we}")
                    return f_val
            except Exception as me:
                log.warning(f"[WS] 读取 tags_map.yaml 自愈标靶边长异常: {me}")
        return None
    try:
        f = float(v)
        if f <= 0:
            log.warning(f"[WS] 工位标靶边长非法 (≤0): {path} → {v}")
            return None
        return f
    except (ValueError, TypeError) as e:
        log.warning(f"[WS] 工位标靶边长字段非数值 ({path}): {v} ({e})")
        return None


def load_workspace_tag_config(workspace_dir: str) -> Dict[str, Any]:
    """
    【单一真理源】加载工位标靶综合先验配置 (收敛于 tag_whitelist.yaml.tags):
    - 返回标准 Schema 字典:
      {
          "workspace_id": str,
          "workspace_name": str,
          "tag_default_size_mm": float,
          "tags": Dict[int, Dict[str, Any]],
          "notes": str,
      }
    """
    ws_id = os.path.basename(os.path.normpath(workspace_dir))
    wl_path = os.path.join(workspace_dir, "tag_whitelist.yaml")

    wl_data: Dict[str, Any] = {}
    if os.path.exists(wl_path):
        try:
            with open(wl_path, "r", encoding="utf-8") as f:
                wl_data = yaml.safe_load(f) or {}
        except Exception as e:
            log.warning(f"[WS] 读取 tag_whitelist.yaml 异常 ({wl_path}): {e}")

    clean_tags: Dict[int, Dict[str, Any]] = {}

    if "tags" in wl_data and isinstance(wl_data["tags"], dict):
        for k, v in wl_data["tags"].items():
            try:
                tid = int(k)
                if not isinstance(v, dict):
                    v = {}
                clean_item = {}
                raw_xyz = v.get("xyz_mm")
                if raw_xyz is not None and isinstance(raw_xyz, (list, tuple)):
                    xyz_f = []
                    for val in raw_xyz[:3]:
                        if val is None or (isinstance(val, str) and val.strip().lower() in ("null", "none", "~", "nan", ".nan")):
                            xyz_f.append(None)
                        else:
                            try:
                                xyz_f.append(round(float(val), 4))
                            except (ValueError, TypeError):
                                xyz_f.append(None)
                    while len(xyz_f) < 3:
                        xyz_f.append(None)
                    if any(c is not None for c in xyz_f):
                        clean_item["xyz_mm"] = xyz_f
                clean_tags[tid] = clean_item
            except (ValueError, TypeError):
                continue

    size_val = wl_data.get("tag_default_size_mm")
    if size_val is not None:
        try:
            size_val = float(size_val)
        except Exception:
            size_val = 40.0
    else:
        size_val = 40.0

    return {
        "workspace_id": wl_data.get("workspace_id", ws_id),
        "workspace_name": wl_data.get("workspace_name", ws_id),
        "tag_default_size_mm": size_val,
        "tags": clean_tags,
        "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义)"
    }


def save_workspace_tag_config(
    workspace: Workspace,
    tags: Optional[Dict[int, Dict[str, Any]]] = None,
    tag_default_size_mm: Optional[float] = None
) -> bool:
    """
    【统一真理源唯一序列化出口】写穿工位标靶配置 (tag_whitelist.yaml):
    - 严格且仅输出 tags 单源字典，绝不接受 allowed_ids / anchor_tags 废弃胶水参数;
    - 未知轴在 YAML 中原生格式化为标准 null (跨语言/跨平台无歧义);
    - 物理自愈垃圾清除: 彻底删除孤立的旧 anchor_tags.yaml。
    """
    path = workspace.whitelist_path
    curr = load_workspace_tag_config(workspace.workspace_dir)

    # 1. 确定最终生效的 tags
    if tags is not None:
        curr_tags = dict(tags)
    else:
        curr_tags = dict(curr.get("tags") or {})

    # 2. 清洗规范化 tags
    clean_tags: Dict[int, Dict[str, Any]] = {}
    for tid, tcfg in curr_tags.items():
        try:
            t_int = int(tid)
            if not isinstance(tcfg, dict):
                tcfg = {}
            item = {}
            xyz = tcfg.get("xyz_mm")
            if xyz is not None and isinstance(xyz, (list, tuple)):
                xyz_out = [
                    round(float(c), 4) if (c is not None and str(c).strip().lower() not in ("null", "none", "~", "nan")) else None
                    for c in xyz[:3]
                ]
                while len(xyz_out) < 3:
                    xyz_out.append(None)
                if any(c is not None for c in xyz_out):
                    item["xyz_mm"] = xyz_out
            clean_tags[t_int] = item
        except (ValueError, TypeError):
            continue

    if tag_default_size_mm is not None and tag_default_size_mm > 0:
        curr["tag_default_size_mm"] = float(round(tag_default_size_mm, 3))

    # 3. 严格 Schema 序列化: 仅输出 tags 单一真理源
    clean_doc = {
        "workspace_id": workspace.workspace_id,
        "workspace_name": workspace.name or workspace.workspace_id,
        "tag_default_size_mm": curr["tag_default_size_mm"],
        "tags": {k: clean_tags[k] for k in sorted(clean_tags.keys())},
        "notes": "工位物理标靶单一真理源 (Tag 准入与世界锚点原子化统一定义)"
    }

    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(clean_doc, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        log.info(f"[WS] 统一工位标靶配置已写穿: {path} (总标靶 {len(clean_tags)} 枚)")
    except Exception as e:
        log.error(f"[WS] 写穿工位标靶配置失败 ({path}): {e}")
        return False

    # 物理自愈垃圾清除: 彻底删除孤立的旧 anchor_tags.yaml
    old_anchor_file = os.path.join(workspace.workspace_dir, "anchor_tags.yaml")
    if os.path.isfile(old_anchor_file):
        try:
            os.remove(old_anchor_file)
            log.info(f"[WS] 已删除冗余独立锚点文件: {old_anchor_file}")
        except Exception as e:
            log.warning(f"[WS] 删除旧 anchor_tags.yaml 失败: {e}")

    return True


def load_workspace_anchor_tags(workspace_dir: str) -> Optional[Dict[int, Dict]]:
    """
    【统一真理源接口】读取工位世界坐标锚点 (收敛至 tag_whitelist.yaml 的 tags 字段):
    - 文件存在 → 返回所有具备非空 xyz_mm 的锚点字典 {int tag_id: {"xyz_mm": [f3], "known": [b3]}}
    - 若无任何锚点返回空字典 {}
    - 若整个文件不存在返回 None
    """
    wl_path = os.path.join(workspace_dir, "tag_whitelist.yaml")
    if not os.path.exists(wl_path):
        return None

    cfg = load_workspace_tag_config(workspace_dir)
    tags = cfg.get("tags") or {}
    anchors = {}
    for tid, tcfg in tags.items():
        xyz = tcfg.get("xyz_mm")
        if xyz is not None and isinstance(xyz, (list, tuple)):
            xyz_f = [float(c) if c is not None else 0.0 for c in xyz[:3]]
            known_b = [c is not None for c in xyz[:3]]
            while len(xyz_f) < 3:
                xyz_f.append(0.0)
                known_b.append(False)
            if any(known_b):
                anchors[int(tid)] = {
                    "xyz_mm": xyz_f[:3],
                    "known": known_b[:3]
                }
    return anchors
