# -*- coding: utf-8 -*-
"""
全局物理相机实体与分辨率内参档案模型 (CameraProfile & IntrinsicsRecord)
====================================================================
两层物理目录契约 (data/cameras/<camera_id>/):
  ├── camera_meta.yaml        # 相机物理身份元数据
  └── <width>x<height>/       # 分辨率子档案
      ├── raw_images/         # 棋盘格原图 (intr_XXXX.png，最大序号防覆盖自增)
      └── intrinsics.yaml     # K, D, RMSE, 标定板参数, 每帧误差与标定时间
"""

import glob
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import yaml

from src.utils.logger import get_logger

log = get_logger(__name__)

IMAGE_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
INTR_IMG_PATTERN = re.compile(r"^intr_(\d+)\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)
RES_DIR_PATTERN = re.compile(r"^(\d+)x(\d+)$")


def format_resolution(width: int, height: int) -> str:
    return f"{int(width)}x{int(height)}"


def parse_resolution(res_str: str) -> Optional[Tuple[int, int]]:
    m = RES_DIR_PATTERN.match(str(res_str).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


@dataclass
class IntrinsicsRecord:
    """单个分辨率档案下的相机内参标定结果"""
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    rmse: float = 0.0
    valid_images_count: int = 0
    total_images_count: int = 0
    pattern_size: List[int] = field(default_factory=lambda: [9, 6])
    square_size_mm: float = 25.0
    per_image_errors: List[Dict[str, Any]] = field(default_factory=list)
    calibrated_at: str = ""

    @property
    def camera_matrix(self) -> List[List[float]]:
        return [
            [float(self.fx), 0.0, float(self.cx)],
            [0.0, float(self.fy), float(self.cy)],
            [0.0, 0.0, 1.0],
        ]

    @property
    def resolution_str(self) -> str:
        return format_resolution(self.width, self.height)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "fx": float(self.fx),
            "fy": float(self.fy),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "camera_matrix": self.camera_matrix,
            "dist_coeffs": [float(x) for x in list(self.dist_coeffs)[:5]],
            "rmse": float(self.rmse),
            "valid_images_count": int(self.valid_images_count),
            "total_images_count": int(self.total_images_count),
            "pattern_size": [int(self.pattern_size[0]), int(self.pattern_size[1])],
            "square_size_mm": float(self.square_size_mm),
            "per_image_errors": list(self.per_image_errors),
            "calibrated_at": self.calibrated_at or time.strftime("%Y-%m-%d %H:%M:%S"),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntrinsicsRecord":
        cm = data.get("camera_matrix")
        if isinstance(cm, list) and len(cm) == 3 and all(isinstance(r, list) and len(r) == 3 for r in cm):
            fx = float(cm[0][0])
            fy = float(cm[1][1])
            cx = float(cm[0][2])
            cy = float(cm[1][2])
        else:
            fx = float(data.get("fx", 0.0))
            fy = float(data.get("fy", 0.0))
            cx = float(data.get("cx", 0.0))
            cy = float(data.get("cy", 0.0))

        dist_raw = data.get("dist_coeffs") or [0.0, 0.0, 0.0, 0.0, 0.0]
        dist = [float(x) for x in list(dist_raw)[:5]]
        while len(dist) < 5:
            dist.append(0.0)

        pat_raw = data.get("pattern_size") or [9, 6]
        pat = [int(pat_raw[0]), int(pat_raw[1])] if len(pat_raw) >= 2 else [9, 6]

        return cls(
            width=int(data.get("width", 1920)),
            height=int(data.get("height", 1080)),
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            dist_coeffs=dist,
            rmse=float(data.get("rmse", 0.0)),
            valid_images_count=int(data.get("valid_images_count", 0)),
            total_images_count=int(data.get("total_images_count", 0)),
            pattern_size=pat,
            square_size_mm=float(data.get("square_size_mm", 25.0)),
            per_image_errors=list(data.get("per_image_errors") or []),
            calibrated_at=str(data.get("calibrated_at", "")),
        )


@dataclass
class CameraProfile:
    """全局单台物理相机档案模型"""
    camera_id: str                 # 全局唯一标识 (如 rs_141722070123 / usb_a1b2c3d4)
    name: str                      # 友好别名 (如 产线1号D435)
    camera_dir: str                # 物理目录 data/cameras/<camera_id>
    camera_type: str = "realsense" # 'realsense' | 'usb' (未来预留 'net_camera')
    facing: str = "rear"           # 'rear' (后置正常) | 'front' (前置左右镜像预览)
    hardware_serial: str = ""      # RealSense 硬件序列号或未来设备唯一标识
    usb_index: Optional[int] = None
    description: str = ""
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_front_facing(self) -> bool:
        return str(self.facing).strip().lower() == "front"

    @property
    def meta_path(self) -> str:
        return os.path.join(self.camera_dir, "camera_meta.yaml")

    def resolution_dir(self, resolution: str) -> str:
        return os.path.join(self.camera_dir, str(resolution).strip())

    def raw_images_dir(self, resolution: str) -> str:
        return os.path.join(self.resolution_dir(resolution), "raw_images")

    def intrinsics_path(self, resolution: str) -> str:
        return os.path.join(self.resolution_dir(resolution), "intrinsics.yaml")

    def ensure_dirs(self, resolution: Optional[str] = None):
        os.makedirs(self.camera_dir, exist_ok=True)
        if resolution:
            os.makedirs(self.raw_images_dir(resolution), exist_ok=True)

    def list_resolutions(self) -> List[str]:
        """列出该相机下已存在的分辨率子档案（按宽降序排列）"""
        if not os.path.isdir(self.camera_dir):
            return []
        res_list: List[Tuple[int, int, str]] = []
        for entry in os.listdir(self.camera_dir):
            full = os.path.join(self.camera_dir, entry)
            if os.path.isdir(full):
                parsed = parse_resolution(entry)
                if parsed:
                    res_list.append((parsed[0], parsed[1], entry))
        res_list.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [t[2] for t in res_list]

    def list_images(self, resolution: str) -> List[str]:
        """返回指定分辨率下所有棋盘格原图绝对路径（按文件名排序）"""
        img_dir = self.raw_images_dir(resolution)
        if not os.path.isdir(img_dir):
            return []
        files: List[str] = []
        for ext in IMAGE_EXTS:
            files.extend(glob.glob(os.path.join(img_dir, ext)))
        return sorted(files)

    def next_image_path(self, resolution: str) -> str:
        """按最大序号自增生成下一张原图路径 (intr_XXXX.png)，防止删帧覆盖"""
        self.ensure_dirs(resolution)
        img_dir = self.raw_images_dir(resolution)
        max_num = 0
        for fname in os.listdir(img_dir):
            m = INTR_IMG_PATTERN.match(fname)
            if m:
                max_num = max(max_num, int(m.group(1)))
        return os.path.join(img_dir, f"intr_{max_num + 1:04d}.png")

    def delete_image(self, resolution: str, image_path_or_name: str) -> bool:
        """删除指定分辨率下的单张原图"""
        target = (
            image_path_or_name
            if os.path.isabs(image_path_or_name)
            else os.path.join(self.raw_images_dir(resolution), image_path_or_name)
        )
        if os.path.isfile(target):
            try:
                os.remove(target)
                return True
            except Exception as e:
                log.warning(f"[CameraProfile] 删除图片失败 ({target}): {e}")
        return False

    def load_intrinsics(self, resolution: str) -> Optional[IntrinsicsRecord]:
        """读取指定分辨率的内参档案，不存在则返回 None"""
        p = self.intrinsics_path(resolution)
        if not os.path.isfile(p):
            return None
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not data:
                return None
            return IntrinsicsRecord.from_dict(data)
        except Exception as e:
            log.warning(f"[CameraProfile] 读取内参档案失败 ({p}): {e}")
            return None

    def save_intrinsics(self, resolution: str, record: IntrinsicsRecord) -> bool:
        """原子写入指定分辨率的内参档案"""
        self.ensure_dirs(resolution)
        p = self.intrinsics_path(resolution)
        tmp_p = p + ".tmp"
        try:
            with open(tmp_p, "w", encoding="utf-8") as f:
                yaml.safe_dump(record.to_dict(), f, allow_unicode=True, sort_keys=False)
            os.replace(tmp_p, p)
            self.updated_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self.save_meta()
            return True
        except Exception as e:
            log.error(f"[CameraProfile] 保存内参档案失败 ({p}): {e}")
            if os.path.exists(tmp_p):
                try:
                    os.remove(tmp_p)
                except OSError:
                    pass
            return False

    def save_meta(self) -> bool:
        self.ensure_dirs()
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if not self.created_at:
            self.created_at = now
        self.updated_at = now
        payload = {
            "camera_id": self.camera_id,
            "name": self.name,
            "camera_type": self.camera_type,
            "facing": "front" if self.is_front_facing else "rear",
            "hardware_serial": self.hardware_serial,
            "usb_index": self.usb_index,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        tmp_p = self.meta_path + ".tmp"
        try:
            with open(tmp_p, "w", encoding="utf-8") as f:
                yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)
            os.replace(tmp_p, self.meta_path)
            return True
        except Exception as e:
            log.error(f"[CameraProfile] 保存相机元数据失败 ({self.meta_path}): {e}")
            if os.path.exists(tmp_p):
                try:
                    os.remove(tmp_p)
                except OSError:
                    pass
            return False

    @classmethod
    def load(cls, camera_dir: str) -> Optional["CameraProfile"]:
        meta_path = os.path.join(camera_dir, "camera_meta.yaml")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            cam_id = str(data.get("camera_id") or os.path.basename(camera_dir)).strip()
            return cls(
                camera_id=cam_id,
                name=str(data.get("name") or cam_id),
                camera_dir=os.path.abspath(camera_dir),
                camera_type=str(data.get("camera_type") or "realsense"),
                facing="front" if str(data.get("facing") or "rear").strip().lower() == "front" else "rear",
                hardware_serial=str(data.get("hardware_serial") or ""),
                usb_index=data.get("usb_index"),
                description=str(data.get("description") or ""),
                created_at=str(data.get("created_at") or ""),
                updated_at=str(data.get("updated_at") or ""),
            )
        except Exception as e:
            log.warning(f"[CameraProfile] 加载相机档案失败 ({camera_dir}): {e}")
            return None
