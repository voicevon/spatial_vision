# -*- coding: utf-8 -*-
"""
全局多相机资产注册表 (CameraRegistry)
====================================
统一管理 data/cameras/ 下的全部物理相机档案：
- RealSense 相机采用硬件原生唯一序列号建档: rs_<serial>
- USB 相机采用持久化短 UUID 建档: usb_<uuid8>
- 为未来跨网络同步与 NetCamera (net_<device_id>) 预留统一契约
"""

import os
import re
import shutil
import time
import uuid
from typing import List, Optional, Tuple

from src.cameras.camera_entity import CameraProfile, IntrinsicsRecord, format_resolution
from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_CAMERAS_ROOT = os.path.join(PROJECT_ROOT, "data", "cameras")


def _sanitize_id(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(raw).strip())
    return cleaned.strip("_")


class CameraRegistry:
    """全局多相机资产仓储"""

    def __init__(self, cameras_root: Optional[str] = None):
        self.cameras_root = os.path.abspath(cameras_root or DEFAULT_CAMERAS_ROOT)
        os.makedirs(self.cameras_root, exist_ok=True)

    def list_cameras(self) -> List[CameraProfile]:
        """列出全部已登记相机（按 camera_type 与名称稳定排序）"""
        if not os.path.isdir(self.cameras_root):
            return []
        cameras: List[CameraProfile] = []
        for entry in sorted(os.listdir(self.cameras_root)):
            full = os.path.join(self.cameras_root, entry)
            if os.path.isdir(full):
                prof = CameraProfile.load(full)
                if prof is not None:
                    cameras.append(prof)
        cameras.sort(key=lambda c: (c.camera_type, c.name.lower(), c.camera_id))
        return cameras

    def get_camera(self, camera_id: str) -> Optional[CameraProfile]:
        if not camera_id or not str(camera_id).strip():
            return None
        cam_dir = os.path.join(self.cameras_root, str(camera_id).strip())
        if not os.path.isdir(cam_dir):
            return None
        return CameraProfile.load(cam_dir)

    def find_by_hardware(
        self,
        camera_type: str,
        hardware_serial: str = "",
        usb_index: Optional[int] = None,
    ) -> Optional[CameraProfile]:
        """根据物理硬件标识查找已登记相机"""
        ctype = str(camera_type).strip().lower()
        serial = str(hardware_serial or "").strip()
        for cam in self.list_cameras():
            if cam.camera_type.lower() != ctype:
                continue
            if serial and cam.hardware_serial == serial:
                return cam
            if ctype == "usb" and not serial and usb_index is not None and cam.usb_index == int(usb_index):
                return cam
        return None

    def generate_camera_id(self, camera_type: str, hardware_serial: str = "") -> str:
        """生成稳定、规范的相机唯一 ID"""
        ctype = str(camera_type).strip().lower()
        serial = _sanitize_id(hardware_serial)
        if ctype == "realsense":
            return f"rs_{serial}" if serial else f"rs_{uuid.uuid4().hex[:8]}"
        if ctype == "net_camera":
            return f"net_{serial}" if serial else f"net_{uuid.uuid4().hex[:8]}"
        return f"usb_{uuid.uuid4().hex[:8]}"

    def register_camera(
        self,
        name: str,
        camera_type: str = "realsense",
        hardware_serial: str = "",
        usb_index: Optional[int] = None,
        description: str = "",
        camera_id: Optional[str] = None,
    ) -> CameraProfile:
        """登记新相机（若同物理序列号或同 camera_id 已存在则直接返回既有档案）"""
        ctype = str(camera_type).strip().lower()
        serial = str(hardware_serial or "").strip()

        if camera_id:
            existing = self.get_camera(camera_id)
            if existing is not None:
                return existing
        elif serial or (ctype == "usb" and usb_index is not None):
            existing = self.find_by_hardware(ctype, hardware_serial=serial, usb_index=usb_index)
            if existing is not None:
                return existing

        cid = _sanitize_id(camera_id) if camera_id else self.generate_camera_id(ctype, serial)
        # 避免极小概率碰撞
        while os.path.exists(os.path.join(self.cameras_root, cid)):
            cid = f"{cid}_{uuid.uuid4().hex[:4]}"

        now = time.strftime("%Y-%m-%d %H:%M:%S")
        cam_dir = os.path.join(self.cameras_root, cid)
        profile = CameraProfile(
            camera_id=cid,
            name=(name or cid).strip(),
            camera_dir=cam_dir,
            camera_type=ctype,
            hardware_serial=serial,
            usb_index=int(usb_index) if usb_index is not None else None,
            description=description,
            created_at=now,
            updated_at=now,
        )
        profile.save_meta()
        log.info(f"[CameraRegistry] 已登记相机档案: {cid} ({profile.name}, type={ctype})")
        return profile

    def rename_camera(self, camera_id: str, new_name: str) -> bool:
        prof = self.get_camera(camera_id)
        if prof is None or not new_name or not new_name.strip():
            return False
        prof.name = new_name.strip()
        return prof.save_meta()

    def delete_camera(self, camera_id: str) -> bool:
        prof = self.get_camera(camera_id)
        if prof is None:
            return False
        try:
            shutil.rmtree(prof.camera_dir)
            log.info(f"[CameraRegistry] 已删除相机档案: {camera_id}")
            return True
        except Exception as e:
            log.error(f"[CameraRegistry] 删除相机档案失败 ({camera_id}): {e}")
            return False

    def resolve_intrinsics_record(
        self,
        camera_id: str,
        target_resolution: Optional[Tuple[int, int]] = None,
    ) -> Optional[IntrinsicsRecord]:
        """
        根据 camera_id 查找内参档案：
        1. 若指定 target_resolution=(w, h)，优先精确匹配 <w>x<h>/intrinsics.yaml；
        2. 否则回退返回该相机下任意已标定档案（按高分辨率优先，供调用方做等比仿射缩放）。
        """
        prof = self.get_camera(camera_id)
        if prof is None:
            return None

        if target_resolution is not None:
            exact_key = format_resolution(target_resolution[0], target_resolution[1])
            rec = prof.load_intrinsics(exact_key)
            if rec is not None:
                return rec

        for res_key in prof.list_resolutions():
            rec = prof.load_intrinsics(res_key)
            if rec is not None:
                return rec
        return None
