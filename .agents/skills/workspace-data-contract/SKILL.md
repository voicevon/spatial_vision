---
name: workspace-data-contract
description: Reference guide and data contract for spatial_vision workspace directories, three-dataset folder structure (intrinsics/calibration/production), tag whitelist & world anchor single source of truth, and spatial scene coordinate trees.
---

# Workspace Data Contract & Physical Layout

本项目工位 (Workspace) 采用自包含文件系统存储，严格遵循**单一真理源 (SSOT)**，绝不允许历史兼容别名与降级回退。

## 1. 工位根目录结构 (`data/workspaces/<workspace_id>/`)

```text
data/workspaces/<workspace_id>/
├── workspace_meta.yaml         # 工位元数据 (ID, 名称, 创建时间, 锁定相机类型与分辨率)
├── tag_whitelist.yaml          # 【标靶准入与世界锚点唯一真理源】
├── spatial_scene.yaml          # 【空间场景多坐标系树与 ROI 物件真理源】
├── intrinsics/                 # 【图集一：相机内参专区】
│   ├── raw_images/             # 内参标定原始采图 (intr_XXXX.png)
│   └── camera_intrinsics.yaml  # 工位内参报告 (fx, fy, cx, cy, k1, k2, p1, p2, k3)
├── calibration/                # 【图集二：外参建图专区】
│   ├── raw_images/             # 多视角外参采图 (calib_XXXX.png)
│   ├── visualized/             # BA 平差与重投影可视化结果
│   └── tags_map.yaml           # 解算产物: 标靶 3D 姿态与世界坐标地图
└── production/                 # 【图集三：生产业务专区】
    └── raw_images/             # 现场采图样本 (prod_XXXX.png)
```

## 2. 标靶准入与世界锚点契约 (`tag_whitelist.yaml`)

`tag_whitelist.yaml` 是物理标靶准入与世界坐标绝对锚点的唯一真理源（严禁回退或支持旧格式）：

```yaml
workspace_id: 20261004_175120_workspace_2
workspace_name: 现场工位A
tag_default_size_mm: 50.0       # 标靶物理边长 (唯一权威来源，缺失直接报错，严禁硬编码兜底)
tags:
  0:                            # 仅准入未定世界坐标标靶
  1:                            # 完整世界锚点
    xyz_mm: [0.0, 0.0, 0.0]
  2:                            # 局部约束锚点 (未定轴为 null)
    xyz_mm: [150.0, null, 0.0]
```

- **读取函数**：`src.workspace.tag_whitelist_manager`：
  - `load_workspace_tag_whitelist(ws_dir) -> List[int]`
  - `load_workspace_marker_size_mm(ws_dir) -> Optional[float]`
  - `load_workspace_anchor_tags(ws_dir) -> Optional[Dict[int, Dict]]`
  - `load_workspace_tag_config(ws_dir) -> Dict`
  - `save_workspace_tag_config(ws, tags=..., tag_default_size_mm=...) -> bool`

## 3. 空间场景契约 (`spatial_scene.yaml`)

管理多坐标系关系树 (`coordinate_frames`) 与 ROI 几何物件 (`regions_of_interest`)：
- 由 `ws.get_coordinate_manager()` (`CoordinateTreeManager`) 读写与自愈；
- 由 `ws.get_roi_manager()` (`RoiSpaceManager`) 统一管理。

## 4. 图像命名与防覆盖规范

三大图集遵循语义化文件命名前缀（4 位补零，从 1 起始）：
- **相机内参标定图集** (`intrinsics/raw_images/`): `intr_XXXX.png`
- **外参空间建图图集** (`calibration/raw_images/`): `calib_XXXX.png`
- **生产现场采样图集** (`production/raw_images/`): `prod_XXXX.png`

采图向导 (`capture_wizard.py`) 与 Hub 原地采图必须通过解析对应用途已有最大序列号自增，严禁直接使用 `len(glob)` 导致跳号删帧后覆盖已有图片。
