#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
标靶观测清单与地图文件仓储管理器 (Manifest Repository)
- 纯面向对象单一职责设计，专注于 AprilTag 观测数据清单与空间地图的持久化序列化与反序列化
- 支持两阶段建图数据交换 (tag_observations.yaml)
- 历史人工筛选偏好继承 (keep / note)
- 磁盘图像变动检测与增量同步自愈
- 物理有效性前置过滤与整帧剔除旁路
- 空间地图 (tags_map.yaml) 导出持久化
"""

import os
import glob
import yaml
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import cv2

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def get_default_manifest_path() -> str:
    """获取当前激活工位的标定观测清单绝对路径"""
    try:
        from src.calibration.workspace_manager import WorkspaceManager
        return WorkspaceManager().get_current_workspace().calib_manifest_path
    except Exception:
        return os.path.join(PROJECT_ROOT, "config", "tag_observations.yaml")


class ManifestRepository:
    """
    标靶观测清单与地图文件仓储类
    负责 YAML 文件的读取、校验、增量同步与持久化存储
    """

    def __init__(self, builder: Optional[Any] = None):
        """
        :param builder: 宿主 TagMapBuilder 实例（提供图像检测、指标计算、角点精修与 PnP 校验接口）
        """
        self.builder = builder

    def export_manifest(self, 
                        image_paths: List[str], 
                        manifest_path: Optional[str] = None,
                        generate_visualized: bool = True) -> str:
        """
        两阶段建图流水线 - 阶段一：
        扫描多视角图像，提取标靶观测数据并导出为结构化审核清单 (YAML)
        - 自动保留用户此前修改过的 keep: false 剔除状态与人工备注 (note)
        - 自动输出单元方格分辨率 (Cell: WxH px)、中心坐标与面积
        - 支持同步生成高清图示化标注图片至 visualized/ 目录
        """
        if manifest_path is None:
            manifest_path = get_default_manifest_path()

        existing_prefs = {}
        existing_enabled = {}
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_manifest = yaml.safe_load(f) or {}
                for img_key, img_info in old_manifest.get("images", {}).items():
                    existing_enabled[img_key] = img_info.get("enabled", True)
                    for obs in img_info.get("observations", []):
                        tid = obs.get("tag_id")
                        keep = obs.get("keep", True)
                        note = obs.get("note", "")
                        existing_prefs[(img_key, tid)] = (keep, note)
                log.info(f"[+] 检测到已有审核清单，已成功加载 {len(existing_prefs)} 条历史人工保留/剔除标记")
            except Exception as e:
                log.warning(f"[WARN] 读取已有清单配置失败: {e}，将生成全新清单。")

        manifest_data = {
            "summary": {},
            "images": {}
        }
        total_obs = 0
        total_kept = 0
        total_excluded = 0

        vis_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "visualized")
        if generate_visualized:
            os.makedirs(vis_dir, exist_ok=True)

        for path in sorted(image_paths):
            base_name = os.path.basename(path)
            img = cv2.imread(path)
            if img is None:
                continue

            if self.builder is not None:
                detected = self.builder.detect_tags(img)
            else:
                detected = {}

            obs_list = []

            for tid in sorted(detected.keys()):
                corners = detected[tid]
                if self.builder is not None:
                    metrics = self.builder.compute_tag_metrics(corners)
                else:
                    metrics = {"cell_size_px": [0, 0], "center_px": [0.0, 0.0], "area_px": 0.0}
                
                # 保留历史人工决策
                keep_val, note_val = existing_prefs.get((base_name, tid), (True, ""))
                total_obs += 1
                if keep_val:
                    total_kept += 1
                else:
                    total_excluded += 1

                obs_entry = {
                    "tag_id": int(tid),
                    "keep": bool(keep_val),
                    "cell_size_px": metrics["cell_size_px"],
                    "center_px": metrics["center_px"],
                    "area_px": metrics["area_px"],
                    "note": str(note_val),
                    "corners": [[round(float(c[0]), 2), round(float(c[1]), 2)] for c in corners.reshape(4, 2)]
                }
                obs_list.append(obs_entry)

            annotated_name = os.path.splitext(base_name)[0] + "_annotated.png"
            annotated_path = os.path.join(vis_dir, annotated_name)
            
            if generate_visualized and self.builder is not None:
                ann_img = self.builder.render_annotated_frame(img, detected)
                cv2.imwrite(annotated_path, ann_img)

            manifest_data["images"][base_name] = {
                "file_name": base_name,
                "image_path": path.replace("\\", "/"),
                "annotated_path": annotated_path.replace("\\", "/"),
                "enabled": existing_enabled.get(base_name, True),
                "detected_count": len(obs_list),
                "observations": obs_list
            }

        manifest_data["summary"] = {
            "total_images": len(manifest_data["images"]),
            "total_observations": total_obs,
            "total_kept": total_kept,
            "total_excluded": total_excluded,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # 写入带有清晰中文使用指南的 YAML
        header_comments = (
            "# ==============================================================================\n"
            "# AprilTag 离线建图与 BA 平差观测数据清单 (Observations Manifest)\n"
            "# \n"
            "# 【人工审核与坏样本剔除指南】：\n"
            "# 1. 请在下方列表中查找需要剔除的劣质标靶观测（如透视大倾角、模糊、边缘拉丝畸变）；\n"
            "# 2. 将对应的 `keep: true` 修改为 `keep: false` 即可，求解器将自动忽略该项；\n"
            "# 3. 可在 `note: \"\"` 中记录剔除原因（如 \"大倾角发灰\"、\"遮挡严重\" 等）；\n"
            "# 4. 修改保存后，在控制台选择继续执行平差即可；\n"
            "# 5. 【拓扑连通性安全守门员】：系统会在平差前进行连通图校验，防止误删导致断网崩溃！\n"
            "# ==============================================================================\n\n"
        )
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(header_comments)
            yaml.dump(manifest_data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

        log.info(f"[OK] 观测清单已成功导出至: {manifest_path}")
        log.info(f"     统计: 共 {len(manifest_data['images'])} 张图像，{total_obs} 次标靶观测 (保留: {total_kept}, 排除: {total_excluded})")
        return manifest_path

    def load_manifest(self, 
                      manifest_path: Optional[str] = None) -> Tuple[List[Dict[int, np.ndarray]], List[str], Dict[str, Any]]:
        """
        两阶段建图流水线 - 阶段二：
        从审核清单中加载已审核的标靶观测数据，并过滤掉 keep: false 的坏样本。
        :return: (frame_detections, valid_frame_names, stats)
        """
        if manifest_path is None:
            manifest_path = get_default_manifest_path()

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"未找到观测清单文件: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        # 核心自愈机制：检测磁盘是否存在未登记的新图片 (如 view_0016~view_0022)，自动增量录入
        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        disk_files = sorted(glob.glob(os.path.join(manifest_dir, "view_*.png")))
        if not disk_files:
            disk_files = sorted(glob.glob(os.path.join(manifest_dir, "raw_images", "view_*.png")))
        if not disk_files:
            disk_files = sorted([
                p for p in glob.glob(os.path.join(manifest_dir, "*.png"))
                if not p.endswith("_annotated.png") and not p.endswith("_quiver.png")
            ])
        if not disk_files:
            disk_files = sorted([
                p for p in glob.glob(os.path.join(manifest_dir, "raw_images", "*.png"))
                if not p.endswith("_annotated.png") and not p.endswith("_quiver.png")
            ])
        existing_imgs = data.get("images", {})
        disk_bases = {os.path.basename(p) for p in disk_files}
        manifest_bases = set(existing_imgs.keys())
        
        # 若磁盘包含未在清单中出现的新文件，自动触发增量导出
        if disk_bases - manifest_bases:
            missing_count = len(disk_bases - manifest_bases)
            log.info(f"[*] 检测到采图目录新增 {missing_count} 张照片，正在自动增量同步录入清单与可视化...")
            self.export_manifest(disk_files, manifest_path=manifest_path, generate_visualized=True)
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        frame_detections = []
        valid_frame_names = []
        stats = {
            "total_images": len(data.get("images", {})),
            "total_observations": 0,
            "total_kept": 0,
            "total_excluded": 0,
            "total_excluded_frames": 0,
            "dropped_single_tag_frames": [],
            "excluded_items": []
        }

        manifest_dir = os.path.dirname(os.path.abspath(manifest_path))
        for img_name, img_info in data.get("images", {}).items():
            # 核心旁路拦截：若当前帧已被人工临时剔除 (enabled: false)，整帧所有观测直接旁路跳过
            if not img_info.get("enabled", True):
                stats["total_excluded_frames"] += 1
                for obs in img_info.get("observations", []):
                    stats["total_observations"] += 1
                    stats["total_excluded"] += 1
                    stats["excluded_items"].append({
                        "image": img_name,
                        "tag_id": int(obs["tag_id"]),
                        "note": "整张图片已被人工临时停用剔除"
                    })
                continue

            tags_in_frame = {}
            # 探测原图是否存在以执行实时亚像素精修 (优先记录路径，其次当前目录及 raw_images 目录)
            raw_img_path = img_info.get("image_path", os.path.join(manifest_dir, img_name))
            if not os.path.isabs(raw_img_path):
                raw_img_path = os.path.join(PROJECT_ROOT, raw_img_path)
            if not os.path.exists(raw_img_path):
                for candidate in [
                    os.path.join(manifest_dir, "raw_images", img_name),
                    os.path.join(manifest_dir, img_name),
                ]:
                    if os.path.exists(candidate):
                        raw_img_path = candidate
                        break
            gray_for_refine = None
            if os.path.exists(raw_img_path):
                raw_img = cv2.imread(raw_img_path)
                if raw_img is not None:
                    gray_for_refine = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

            for obs in img_info.get("observations", []):
                stats["total_observations"] += 1
                tid = int(obs["tag_id"])
                keep = obs.get("keep", True)
                if not keep:
                    stats["total_excluded"] += 1
                    stats["excluded_items"].append({
                        "image": img_name,
                        "tag_id": tid,
                        "note": obs.get("note", "")
                    })
                    continue

                corners = np.array(obs["corners"], dtype=np.float64)
                if gray_for_refine is not None and self.builder is not None and hasattr(self.builder, "refine_corners_subpix"):
                    corners = self.builder.refine_corners_subpix(gray_for_refine, corners)

                # 物理有效性前置过滤：过滤微小噪点伪标靶 (area < 120px^2) 或测距异常 (深度不在 150~2200mm)
                area = float(cv2.contourArea(corners.astype(np.float32)))
                z = 0.0
                succ = False
                if self.builder is not None and hasattr(self.builder, "solve_single_tag_pnp"):
                    succ, _, tv = self.builder.solve_single_tag_pnp(corners)
                    z = float(tv[2][0]) if succ else 0.0

                if area < 120.0 or (succ and (z > 2200.0 or z < 150.0)):
                    stats["total_excluded"] += 1
                    stats["excluded_items"].append({
                        "image": img_name,
                        "tag_id": tid,
                        "note": f"物理异常噪点自动拦截 (面积={area:.1f}px, 深度={z:.1f}mm)"
                    })
                    continue

                stats["total_kept"] += 1
                tags_in_frame[tid] = corners

            # 约束检查：一张图必须至少观测到 2 个 Tag 才能形成相对刚体约束
            if len(tags_in_frame) >= 2:
                frame_detections.append(tags_in_frame)
                valid_frame_names.append(img_name)
            else:
                stats["dropped_single_tag_frames"].append((img_name, len(tags_in_frame)))

        return frame_detections, valid_frame_names, stats

    @staticmethod
    def _sanitize_for_yaml(obj: Any) -> Any:
        """递归清洗数据结构，将 tuple 转换为 list，杜绝产生 !!python/tuple 等 YAML 私有标签"""
        if isinstance(obj, tuple):
            return [ManifestRepository._sanitize_for_yaml(x) for x in obj]
        elif isinstance(obj, list):
            return [ManifestRepository._sanitize_for_yaml(x) for x in obj]
        elif isinstance(obj, dict):
            return {k: ManifestRepository._sanitize_for_yaml(v) for k, v in obj.items()}
        return obj

    @staticmethod
    def save_map(map_data: Dict, output_path: str):
        """保存标靶地图至 YAML 文件. 必须显式传入路径 — 默认值已废弃 (Tag 地图必须写入工位沙盒)"""
        if not output_path:
            raise ValueError("save_map: output_path 不能为空 (Tag 地图必须写入工位沙盒, 不再回退全局 config/tags_map.yaml)")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sanitized = ManifestRepository._sanitize_for_yaml(map_data)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(sanitized, f, allow_unicode=True, sort_keys=False)
        log.info(f"[OK] 标靶空间立体地图已成功保存至: {output_path}")

    @staticmethod
    def load_map(map_path: str) -> Optional[Dict[str, Any]]:
        """从 YAML 文件加载标靶空间地图"""
        if not map_path or not os.path.exists(map_path):
            return None
        with open(map_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


