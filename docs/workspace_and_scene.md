# 工位沙盒、空间场景与 Workspace Hub

> **定位**：定义工位（Workspace）文件系统契约、三大图集命名规范、`tag_whitelist.yaml` 与 `spatial_scene.yaml` 真理源、多坐标系树与 3D Smart ROI 模型，以及 Workspace Hub 核心机制。  
> **核心源码**：`src/workspace/`、`tools/workspace_hub/`。

---

## 1. 工位沙盒物理目录契约

每个工位自包含全部标定与生产资产，位于 `data/workspaces/<workspace_id>/`：

```text
data/workspaces/<workspace_id>/
├── workspace_meta.yaml          # 工位元数据（ID、名称、绑定相机、锁定分辨率、图集计数与 BA 指标缓存）
├── tag_whitelist.yaml           # 【唯一真理源】标靶准入、物理边长 tag_default_size_mm 与世界锚点 xyz_mm
├── spatial_scene.yaml           # 【唯一真理源】多坐标系树 (frames) 与 3D Smart ROI 物件 (rois)
├── tags_map.yaml                # 【工位顶层产物】空间立体几何地图 T_world_from_tag（供标定与生产消费）
├── intrinsics/                  # 【专区一：相机内参】
│   ├── raw_images/              #   棋盘格/标定板原图 (intr_XXXX.png)
│   ├── camera_intrinsics.yaml   #   工位专属内参矩阵 K、畸变系数 D 与锁定分辨率
│   └── reports/                 #   内参标定评估报告
├── calibration/                 # 【专区二：外参建图】
│   ├── raw_images/              #   AprilTag 多视角原图 (calib_XXXX.png)
│   ├── tag_observations.yaml    #   离线观测清单与剔除标记
│   ├── reports/                 #   BA 平差与盲测质检报告
│   └── visualized/              #   重投影残差与棱柱可视化图
└── production/                  # 【专区三：生产采样】
    ├── raw_images/              #   现场工件采图样本 (prod_XXXX.png)
    └── reports/                 #   生产质检报告
```

### 1.1 硬件绑定与分辨率锁定 (SSOT)
- 分辨率解析优先级：`intrinsics/camera_intrinsics.yaml` 中的 `(width, height)` $\rightarrow$ `workspace_meta.yaml` 中的 `camera_resolution` $\rightarrow$ 缺省 `1920x1080`。
- **硬件锁定保护 (`is_hardware_locked`)**：工位一旦存在任意历史照片（`total_image_count > 0`），即锁定物理相机类型（`realsense` / `usb` / `mock`）与分辨率。
- **防呆校验 (`validate_image_resolution`)**：传入图像尺寸与工位锁定规格不符时直接拒绝，杜绝跨分辨率内参错配。

### 1.2 三大图集命名契约与防覆盖
| 图集专区 | 存储目录 | 命名格式 | 说明 |
| :--- | :--- | :--- | :--- |
| 相机内参 | `intrinsics/raw_images/` | `intr_XXXX.png` | 4 位补零，从 `0001` 起始 |
| 外参建图 | `calibration/raw_images/` | `calib_XXXX.png` | 4 位补零，从 `0001` 起始 |
| 生产采样 | `production/raw_images/` | `prod_XXXX.png` | 4 位补零，从 `0001` 起始 |

- 采图向导（`tools/capture/`）与 Hub 原地采图必须解析对应用途已有最大序列号后自增，**严禁**使用 `len(glob)` 计数导致删帧后覆盖旧图。
- 所有图像读写统一走 `src/utils/image_io.py`（`imread_unicode` / `imwrite_unicode`）。

---

## 2. 标靶白名单与世界锚点 (`tag_whitelist.yaml`)

由 `src/workspace/tag_whitelist_manager.py` 唯一读写，严禁支持旧格式或隐式边长回退：

```yaml
workspace_id: 20261004_175120_workspace_2
workspace_name: 现场工位A
tag_default_size_mm: 50.0        # 标靶物理边长唯一权威来源；缺失必须报错，严禁硬编码兜底
tags:
  0:                             # 仅准入，未定世界坐标
  1:                             # 完整世界锚点
    xyz_mm: [0.0, 0.0, 0.0]
  2:                             # 局部约束锚点（未定轴填 null）
    xyz_mm: [150.0, null, 0.0]
```

- **Tag 编号分段约定**：`world` 根系占 `0~9`，第 $k$ 个机构子坐标系占 `k*10 ~ k*10+9`。
- **关键 API**：`load_workspace_tag_whitelist`、`load_workspace_marker_size_mm`、`load_workspace_anchor_tags`、`load_workspace_tag_config`、`save_workspace_tag_config`。

---

## 3. 空间场景真理源 (`spatial_scene.yaml`)

`spatial_scene.yaml` 统一管理工位内的**多坐标系树 (`frames`)** 与 **3D Smart ROI (`rois`)**。

### 3.1 单一原子写者：`SceneConfigGuard`
- 路径：`src/workspace/scene_config_guard.py`
- 机制：维护文件路径级重入锁（`RLock`），写入时先写临时文件再 `os.replace` 原子重命名。
- 约束：`CoordinateTreeManager` 与 `RoiSpaceManager` 的全部持久化操作均委托给 `SceneConfigGuard`，彻底消除双写竞态与断电损坏。

### 3.2 多坐标系树 (`CoordinateTreeManager`)
- **根节点**：`world` 为唯一根节点（`parent_frame_id: None`，与 `tags_map.yaml` 对齐）。
- **子坐标系类型 (`FrameDefinition.type`)**：
  - `fixed_transform`：固定刚体变换（`translation_xyz_mm` + `rotation_rpy_deg`，`xyz` 外部旋转、单位毫米与度）；支持逐轴已知掩码 `known_dof`（6 维布尔）与先验位姿，供 `frame_extrinsic_solver` 反推外参。
  - `tag_bound`：绑定一个或多个 AprilTag ID（`tag_ids`）及局部偏移（`offset_xyz_mm`、`offset_rpy_deg`）。
- **级联求解**：`get_transform(target, source)` 自动沿树递归求解任意两帧的 $4\times 4$ 齐次变换矩阵，内置循环依赖防护。

### 3.3 3D Smart ROI (`RoiSpaceManager`)
每个 `RoiDefinition` 是挂载在指定 `frame_id` 下的 3D 有向长方体（OBB：`center_xyz_mm`、`size_xyz_mm`、`rotation_rpy_deg`），并承载生产工艺语义：

| 字段 | 取值与含义 |
| :--- | :--- |
| `category` | 物理部件类型：`belt`（皮带）、`wheel`（分离轮）、`tray`（托盘）、`general` |
| `role` | 工艺角色：`source`（来料抓取源）、`destination`（落料目标槽）、`keepout`（防撞禁区）、`general` |
| `target_intent` | 感知意图：`pose_pick`（位姿抓取）、`piece_count`（根数统计）、`occupancy`（有无检测）、`general` |
| `binding` | 下游通信与槽位映射（如 `{"slot_index": 0, "capacity_max": 5, "grades": ["A", "B"]}`），由 `SortingDispatcher` 消费 |
| `pipeline_override` | 可选的定制视觉管线 key（为空时沿用工位全局默认管线） |

---

## 4. 数据一致性审计与自愈 (`HealthAuditor`)

- 模块：`src/workspace/health_auditor.py`
- **职责**：
  1. 比对 `workspace_meta.yaml` 缓存计数与物理磁盘 `raw_images/` 实际文件数，在 Workspace Hub 提供一键刷新元数据自愈；
  2. 核验 `tags_map.yaml` 与 `tag_whitelist.yaml` 一致性，清除非法/越界标靶并统一通过 `ManifestRepository.save_map` 写回。

---

## 5. Workspace Hub 交互架构 (`tools/workspace_hub/`)

- 启动：`python -m tools.workspace_hub`（或 Dashboard `[1]`）。
- 架构：960×720 左右两栏 + 自适应动态页签区，遵循 `BaseCvApp` + 只读 `Renderer` + 子状态机（`states/gallery_state.py`、`whitelist_state.py`、`geometry_state.py`）。
  - **左栏**：工位树导航（工位根节点 + 机构坐标系子节点）+ 新建工位。
  - **右栏（工位宏观视图）**：Dashboard（运维条与健康大屏）、相机&内参（3×3 图集墙 + 一键内参求解）、外参建图（图集 + 采图向导 + 直通 Spatial Mapping Studio）、生产图集。
  - **右栏（机构微观视图）**：Tags 标靶与位姿（外参配置 + 标靶边长 + 10-Slot 放行矩阵）、ROI 物件（3D OBB 增删改查）。
- 性能优化：命中区缓存跳过静态帧重绘；`Workspace.load()` 优先信任元数据缓存；局部 Patch 贴图渲染中文。
