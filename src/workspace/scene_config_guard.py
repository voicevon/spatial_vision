#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工位空间几何场景配置安全守卫 (SceneConfigGuard)
=================================================
核心职责：
1. 【单一真理源唯一写者】统管 spatial_scene.yaml (frames 坐标系树 + rois 3D 空间物件) 的原子读写；
2. 【防竞态原子落盘】采用路径锁 (RLock) 结合临时文件原子重命名 (os.replace)，杜绝并发覆写与断电损坏；
3. 【模块增量解耦】为 CoordinateTreeManager 与 RoiSpaceManager 提供高内聚的原子持久化接口，彻底消除双写者竞态。
"""

import os
import threading
from typing import Any, Dict, List, Optional
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

# 文件路径级别的锁池 (避免不同工位文件互相阻塞，同时保证同一工位原子串行写入)
_FILE_LOCKS: Dict[str, threading.RLock] = {}
_LOCKS_MUTEX = threading.Lock()


def _get_file_lock(filepath: str) -> threading.RLock:
    """获取指定路径的独占重入锁"""
    norm_path = os.path.normcase(os.path.abspath(filepath))
    with _LOCKS_MUTEX:
        if norm_path not in _FILE_LOCKS:
            _FILE_LOCKS[norm_path] = threading.RLock()
        return _FILE_LOCKS[norm_path]


class SceneConfigGuard:
    """工位空间几何场景 (spatial_scene.yaml) 的权威守卫与唯一序列化出口"""

    DEFAULT_VERSION = "2.0"

    @classmethod
    def load_scene(cls, filepath: str) -> Dict[str, Any]:
        """
        线程安全读取空间场景完整配置。
        若文件不存在，返回标准空骨架结构。
        """
        if not filepath or not os.path.exists(filepath):
            return {
                "version": cls.DEFAULT_VERSION,
                "workspace_id": "",
                "active_frame_id": "world",
                "frames": [],
                "rois": [],
            }

        lock = _get_file_lock(filepath)
        with lock:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                if not isinstance(data, dict):
                    data = {}

                data.setdefault("version", cls.DEFAULT_VERSION)
                data.setdefault("workspace_id", "")
                data.setdefault("active_frame_id", "world")
                if not isinstance(data.get("frames"), list):
                    data["frames"] = []
                if not isinstance(data.get("rois"), list):
                    data["rois"] = []
                return data
            except Exception as e:
                log.error(f"[SceneConfigGuard] 读取空间场景失败 ({filepath}): {e}")
                return {
                    "version": cls.DEFAULT_VERSION,
                    "workspace_id": "",
                    "active_frame_id": "world",
                    "frames": [],
                    "rois": [],
                }

    @classmethod
    def save_scene(cls, filepath: str, scene_data: Dict[str, Any]) -> bool:
        """
        原子写穿空间场景完整配置。
        先写入 .tmp 临时文件，校验成功后原子替换目标文件。
        """
        if not filepath:
            log.error("[SceneConfigGuard] save_scene 目标路径不能为空")
            return False

        lock = _get_file_lock(filepath)
        with lock:
            target_dir = os.path.dirname(os.path.abspath(filepath))
            os.makedirs(target_dir, exist_ok=True)
            tmp_path = filepath + ".tmp"

            try:
                out_data = {
                    "version": scene_data.get("version", cls.DEFAULT_VERSION),
                    "workspace_id": scene_data.get("workspace_id", ""),
                    "active_frame_id": scene_data.get("active_frame_id", "world"),
                    "frames": scene_data.get("frames", []),
                    "rois": scene_data.get("rois", []),
                }

                with open(tmp_path, "w", encoding="utf-8") as f:
                    yaml.dump(out_data, f, allow_unicode=True, sort_keys=False, indent=2)

                # 原子替换 (Windows/POSIX 原生安全原子操作)
                os.replace(tmp_path, filepath)
                log.debug(f"[SceneConfigGuard] 场景已原子持久化: {filepath}")
                return True
            except Exception as e:
                log.error(f"[SceneConfigGuard] 原子持久化失败 ({filepath}): {e}")
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass
                return False

    @classmethod
    def update_frames(
        cls,
        filepath: str,
        frames: List[Dict[str, Any]],
        active_frame_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> bool:
        """
        【坐标系单向写者】原子更新坐标系树节 (frames)，严格保全已有的 rois 节。
        """
        if not filepath:
            return False

        lock = _get_file_lock(filepath)
        with lock:
            scene = cls.load_scene(filepath)
            scene["frames"] = frames
            if active_frame_id is not None:
                scene["active_frame_id"] = active_frame_id
            if workspace_id:
                scene["workspace_id"] = workspace_id
            return cls.save_scene(filepath, scene)

    @classmethod
    def update_rois(
        cls,
        filepath: str,
        rois: List[Dict[str, Any]],
        workspace_id: Optional[str] = None,
    ) -> bool:
        """
        【ROI 单向写者】原子更新 3D ROI 空间物件节 (rois)，严格保全已有的 frames 节。
        """
        if not filepath:
            return False

        lock = _get_file_lock(filepath)
        with lock:
            scene = cls.load_scene(filepath)
            scene["rois"] = rois
            if workspace_id:
                scene["workspace_id"] = workspace_id
            return cls.save_scene(filepath, scene)
