# spatial_vision

面向绿芦笋自动化上料与分拣产线的**工业 3D/2D 视觉感知、AprilTag 空间建图标定与多执行器控制平台**。

核心覆盖五大能力：
- **工位沙盒与空间场景 (`src/workspace`)**：多工位物理隔离、硬件与分辨率锁定、多坐标系树与 3D Smart ROI。
- **相机内参与空间建图 (`src/calibration`)**：标定板内参求解、多视角 AprilTag 16h5 两阶段 BA 平差、Umeyama 世界基准对齐与在线自定位。
- **视觉感知与几何测量 (`src/vision`)**：3D RGB-D 堆叠顶层抓取位姿解算、纯 2D 多管线对比框架、单根芦笋绿色有效长度与特征直径测量。
- **执行器调度与控制 (`src/control`, `src/devices`)**：Smart ROI 分选调度、SCARA 三段式防撞 G-code 规划与 Marlin 串口闭环、ESP32 8 通道分离轮 MQTT 节拍控制。
- **工业交互工具链 (`tools/`)**：基于 OpenCV + NumPy 的统一 Dashboard 与全套标定、测量、生产及硬件调试工作台。

---

## 快速上手

推荐环境：**Windows 10/11 x64**（兼顾 Linux），**Python 3.10+**。

```powershell
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动工业视觉总控 Dashboard
./run.ps1            # PowerShell
# 或
run.bat              # Windows CMD
# 或直接调用
python tools/gui_launcher.py
```

---

## 核心工具入口

除通过 Dashboard 卡片一键拉起外，常用工具支持命令行独立启动：

| 类别 | 工具 | 命令 | 说明 |
| :--- | :--- | :--- | :--- |
| **资产中枢** | Workspace Hub | `python -m tools.workspace_hub` | 工位 CRUD、全局相机绑定、两大图集、坐标系树与 3D ROI |
| **资产中枢** | Camera Hub | `python -m tools.camera_hub` | 全局多相机资产库、棋盘格实时角点采图与逐帧内参标定 |
| **采图标定** | 图像采集向导 | `python -m tools.capture` | 外参 (`calib_`) / 生产 (`prod_`) 双用途快速采集 |
| **采图标定** | 空间建图工作站 | `python -m tools.spatial_mapping_studio` | 样本审核、两阶段 BA 平差、残差剪枝与地图落盘 |
| **采图标定** | AprilTag 图纸生成 | `python tools/generate_apriltags.py` | 生成 16h5 (ID 0~29) 标靶高清图与 A4 排版 PDF |
| **视觉验证** | 顶层抓取位姿工作室 | `python -m tools.asparagus_pose_studio` | 堆叠顶层抓取位姿与纯 2D 多算法管线对比 |
| **视觉验证** | 单根芦笋测量工作室 | `python -m tools.asparagus_measure_studio` | ROI 内单根芦笋绿色有效长度与 20mm 处直径测量 |
| **生产运行** | 单根分拣运行时 | `python tools/sorter_runtime.py` | 2D 机架 4 标靶自动拉正、硬触发测量分级与串口闭环（支持 `--sim`/`--headless`） |
| **生产运行** | SCARA 抓取生产 | `python -m tools.scara_production` | Smart ROI 进料感知、分选调度与 SCARA 抓取闭环 |
| **生产运行** | 分离轮生产工作台 | `python -m tools.isolate_wheels_production` | 8 通道分离轮自动循环出料与 PPM 效能监控 |
| **硬件调试** | Robot 在线跟踪 | `python -m tools.tracker` | 实时 Tag 世界坐标解算与 SCARA 联动校偏 |
| **硬件调试** | RealSense 诊断 | `python tools/d435_viewer.py` | D435 双流实时预览、深度探针与快照抓拍 |
| **硬件调试** | SCARA 调试 | `python tools/scara_debug/app.py` | 串口点动、回零设零、舵机与 G-code 透传 |
| **硬件调试** | 分离轮 MQTT 调试 | `python tools/isolate_wheels_debug/app.py` | ESP32 分离轮设备发现、状态监视与单/多轴调试 |
| **硬件调试** | NetCamera 调试 | `python tools/net_camera_debug/app.py` | 手机网络摄像机 MQTT 远控与 RTSP 拉流预览 |
| **系统运维** | 环境诊断 | `python tools/diagnose_env.py` | 依赖库与相机驱动健康检查 |

---

## 自动化测试

```powershell
# 运行全量单元测试
python -m unittest discover -s tests -v

# 运行单个测试模块
python -m unittest tests.test_ba_optimizer
```

---

## 文档导航

| 文档 | 内容摘要 |
| :--- | :--- |
| **[architecture.md](docs/architecture.md)** | 分层 DAG 架构、`src/` 包职责、配置文件单一写者矩阵、工具链总览与 GUI 选型 ADR |
| **[requirements.md](docs/requirements.md)** | 系统边界、全子系统物理工况与硬件规格、运动控制与数据一致性安全红线 |
| **[workspace_and_scene.md](docs/workspace_and_scene.md)** | 工位目录契约、三大图集命名规范、`tag_whitelist.yaml` / `spatial_scene.yaml`、`SceneConfigGuard` 与 Workspace Hub |
| **[calibration_pipeline.md](docs/calibration_pipeline.md)** | 内参标定、多视角两阶段 BA、Umeyama 基准对齐、子系外参反推、在线自定位与 BA 误差物理机理 |
| **[vision_pipelines.md](docs/vision_pipelines.md)** | 3D RGB-D 堆叠抓取感知、纯 2D 多算法管线注册框架（F/A/B/C/H/Y）与单根芦笋几何测量 |
| **[control_and_production.md](docs/control_and_production.md)** | Smart ROI 分选调度、SCARA 三段式防撞规划与 Marlin 串口、分离轮 MQTT v1.3 与网络摄像机接入 |
| **[多摄变焦与标定稳定性机理.md](docs/knowledge/多摄变焦与标定稳定性机理.md)** | 跨项目通用原理：智能手机多摄内参漂移机理、编程陷阱与锁定控制 |
| **[AGENTS.md](AGENTS.md)** | 项目开发铁律（零向后兼容、`temp/` 隔离、交互式命名确认、禁自动提交、文档维护准则） |
