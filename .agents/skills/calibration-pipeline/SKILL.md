---
name: calibration-pipeline
description: Domain guide for spatial_vision camera calibration and 3D spatial mapping pipeline. Covers camera intrinsics, multi-view BA solver, Umeyama world datum alignment, and anchor conflict troubleshooting.
---

# Calibration & 3D Spatial Mapping Pipeline

本项目视觉几何与标定系统分为三大核心解算阶段。

## 1. 标定流水线三阶段

```text
[阶段一: 相机内参标定]
  采图 raw_images (棋盘格/标定板) ──> CameraIntrinsicsCalibrator ──> camera_intrinsics.yaml
                                                                        │
[阶段二: 多视角光束法平差 (BA)]                                            ▼
  采图 raw_images (AprilTag 多视角) ──> TagBAOptimizer ──> 相对三维立体地图 (点云/姿态)
                                                                        │
[阶段三: 世界基准系对齐 (World Alignment)]                                 ▼
  tag_whitelist.yaml (已知世界锚点) ──> WorldDatumAligner ──> 绝对世界系 tags_map.yaml
```

## 2. 核心求解器组件与职责

| 求解器模块 | 核心类 / 函数 | 职责与产物 |
| :--- | :--- | :--- |
| `src.calibration.solvers.camera_intrinsics_calibrator` | `CameraIntrinsicsCalibrator` | 标定板角点亚像素提取、内参矩阵 $K$ 与畸变系数 $D$ 解算、重投影误差统计 |
| `src.calibration.ba_optimizer` | `TagBAOptimizer` | 多视角观测图残差优化，联合优化多帧相机外参与各 AprilTag 相对空间位姿 |
| `src.calibration.solvers.world_datum_aligner` | `align_to_world_datum` | 使用 Umeyama 算法或水平面调平先验，将相对建图坐标系对齐到现场物理世界基准系 |

## 3. 常见异常与诊断排查 (Troubleshooting)

### ⚠️ `[ANCHOR CONFLICT FATAL]` (锚点几何严重冲突)
- **现象**：
  日志报错形如：`Tag #1 ⇋ Tag #2: 标称世界距离 100.0 mm, 视觉重构测距 500.0 mm (偏差 400.0 mm, 相对误差 80.0%)`。
- **根因**：
  `tag_whitelist.yaml` 中录入的锚点世界坐标间距，与相机视觉实测（基于已知标靶边长或多视角三角测量）等效几何距离相差过大。
- **排查路径**：
  1. 检查工位 `tag_whitelist.yaml`：确认高疑标靶的 `xyz_mm` 是否录入错误（如正负号颠倒、轴向混淆）；
  2. 检查现场物理张贴：确认实际标靶物理张贴间距是否与图纸尺寸严重不符；
  3. 检查标靶物理边长 `tag_default_size_mm` 是否与实际打印张贴尺寸一致。
