# Smart ROI 与多态生产架构设计方案 (Smart ROI & Multi-Modal Production Architecture)

- **版本**: v1.0
- **创建时间**: 2026-09-27
- **归档路径**: `docs/plans/2026-09-27-smart-roi-and-production-architecture-design.md`
- **状态**: 方案草案 / 待评审 (Proposal / Under Review)
- **关联模块**: `src/calibration/roi_manager.py`, `src/calibration/workspace_manager.py`, `src/vision/pipelines/`, `tools/asparagus_pose_studio/`, `tools/workspace_hub/`, `tools/tracker/`, `tools/gui_launcher.py`

---

## 1. 概述与背景 (Executive Summary)

### 1.1 现状与痛点识别

在 `flux_vision_3d` / `spatial_vision` 系统演进过程中，工业现场的生产需求呈现出高度多态与分散的特征：

1. **ROI 的空间形态与工艺角色差异极大**：
   - **单区域抓取**：上料进料皮带（Belt）通常是 1 个连续大面积空间，核心是识别物料（芦笋）的三维位姿、中心线与抓取点；
   - **多区域分选**：下料分离轮（Isolate Wheels）或分类料槽存在 **8 个连续托架/格槽（Pocket/Slot）**，核心并非抓取，而是**各区域独立统计物料根数（Counting）与满溢状态（Occupancy）**。
   - **工艺流向属性**：ROI 存在明确的工艺语义——**源头（Source，来料提取区）** 与 **目的地（Destination，落料暂存区）**。

2. **感知层越权插手控制层，破坏单一职责原则（SRP）**：
   - 现有数据结构 `AsparagusTarget` 中硬编码了 `generate_gcode(safe_z, drop_x=220, drop_y=0)` 方法；
   - 研发工具 `AsparagusPoseStudio` 界面中嵌入了 G-code 代码框与导出按钮；
   - **危害**：视觉模块被迫预设了单一落料点（`drop_x=220`），完全无法感知下游 8 个料槽的存在状态与根数计数；把“机械臂运动学指令”与“视觉特征提取”紧耦合，导致算法无法在分选场景中复用。

3. **生产闭环模糊，实验性工具承担生产角色**：
   - 现有的 `tools/tracker/app.py`（Robot 在线跟踪）本质上是早期单一链路的测试台，无法承载“SCARA 抓取上料”与“分离轮 8 托架 MQTT 节拍分选”两种完全不同的工业生产工艺。

### 1.2 本方案设计目标

- **职责解耦**：彻底剥离视觉算法对具体机器人 G-code / 硬件控制指令的依赖，感知层仅输出“空间客观事实（Facts）”。
- **Smart ROI 数据模型**：在现有 3D 几何 OBB 基础上，赋予 ROI 生产角色（`source`/`destination`）、业务意图（`pose_pick`/`piece_count`）与通信通道映射。
- **双层协同（Two-Tier Architecture）**：厘清 Workspace（宏观机台与生产模式）与 ROI（微观局部感知动作）的层级关系，同时支持“1对1”简单工位与“1对多”复杂工位。
- **工台矩阵重塑**：研发工作室（Studio）专注算法与几何验证；生产运行时（Production）按工艺分裂为专注的 SCARA 生产工作台与分离轮分选生产工作台。

---

## 2. 核心架构解耦：四层系统模型

系统严格划分为四层单向依赖架构，杜绝跨层逆向调用：

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. 物理空间层 (Spatial & ROI Foundation)                               │
│    资产: rois.yaml, frames.yaml, tags_map.yaml                         │
│    职责: 划定物理空间舞台与多坐标系拓扑 (世界系、皮带系、轮系、机械臂基座) │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 几何与空间约束
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. 视觉感知层 (Perception / 视觉算法)                                  │
│    组件: Vision Pipelines, AsparagusAnalyzer, 连通域/特征统计模块     │
│    职责: 坚持“只输出客观事实 (Facts)，绝不下发动作指令 (Commands)”     │
│          - 皮带 ROI: 输出芦笋实体列表 [{id, pick_xyz, yaw, grade...}] │
│          - 8槽 ROI: 输出各槽计数与状态 [count_0, count_1 ... count_7]   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 纯感知事实 (DTO / Counts)
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. 业务调度层 (Business Coordination)  ★【核心纽带】                    │
│    组件: Production Dispatcher (分选调度与工艺状态机)                  │
│    职责: 统筹全局业务逻辑、槽位决策与防溢控制                           │
│          - 规则: "芦笋 #1 直径15mm -> 1级品 -> 分配至 2号托架"         │
│          - 状态: "2号托架已达上限 -> 启动溢出策略 / 告警"              │
│          - 产出: 抽象任务指令 (Pick-Place Pair 或 Batch Counts JSON)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ 结构化任务指令
┌───────────────────────────────────▼────────────────────────────────────┐
│ 4. 运动与驱动层 (Motion & Device Drivers)                              │
│    组件: MotionPlanner (轨迹规划), RobotSerial (串口), MqttClient (网络)│
│    职责: 将抽象任务翻译为物理硬件协议并执行                             │
│          - SCARA: 生成安全高度过渡、进出点、G-code 串口发送            │
│          - 分离轮: 组装 {"cmd":"load", "counts":[...]} MQTT 发布       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 数据建模：Workspace 与 ROI 的双层容器体系

针对“生产意图到底属于 Workspace 还是属于 ROI”的问题，确立**双层容器（Two-Tier Container）**划分机制：

### 3.1 职责划分对比表

| 维度 | Workspace 级别 (工位/机台全局) | ROI 级别 (局部感知窗口/通道) |
| :--- | :--- | :--- |
| **物理对应** | 整台自动化机台 (Machine / Station) | 机台上的某个工装位、检测视窗、料槽 (Zone / Slot) |
| **模式/意图** | **`production.mode` (生产工作流模式)**<br>- `scara_pick_and_place` (SCARA 拾取放料工位)<br>- `isolate_wheels_sorting` (分离轮 8 槽分选工位)<br>- `standalone_inspection` (独立品控质检工位) | **`intent` (局部感知动作意图)**<br>- `pose_pick` (提取 3D 位姿与抓取点)<br>- `piece_count` (统计目标根数)<br>- `occupancy` (仅判断有货/无货) |
| **硬件接口** | **全局通信总线配置**<br>- SCARA 机械臂串口端口、波特率<br>- 分离轮 MQTT Broker IP、Topic 根路径 | **逻辑通道映射 (Channel Binding)**<br>- `slot_index: 0~7` (映射至协议数组下标)<br>- `drop_id: "A1"` (对应物理料槽代号) |
| **工艺角色** | 统筹协调策略 (Dispatcher Policy) | **`role` (区域工艺角色)**<br>- `source` (来料抓取区)<br>- `destination` (落料受控区)<br>- `keepout` (防撞禁区) |

### 3.2 配置文件规范更新

#### (1) `workspace_meta.yaml`（增强生产工作流属性）
```yaml
version: "1.1"
workspace_id: "20260915_bench_default"
name: "主产线 1号分选工作站"
description: "包含上料进料皮带与 8 托架分离轮的复合工位"

# --- 新增: 全局生产工作流与驱动规范 ---
production:
  mode: "isolate_wheels_sorting"        # 推荐生产模式: scara_pick | isolate_wheels_sorting
  active_gui: "isolate_wheels_production" # 默认关联启动的生产工作台
  cycle_interval_ms: 100                # 生产感知刷新节拍 (ms)
  
  # 关联硬件驱动配置 (按需加载)
  hardware:
    mqtt:
      broker: "voicevon.vicp.io"
      port: 1883
      client_id: "flux_sorter_01"
      topic_command: "flux/loader/device1/cmd"
      topic_state: "flux/loader/device1/state"
    serial_scara:
      port: "COM11"
      baudrate: 115200
      safe_z_mm: 80.0
```

#### (2) `rois.yaml`（升级为 Smart ROI 规范）
```yaml
version: "1.1"
workspace_id: "20260915_bench_default"

rois:
  # ----------------------------------------------------
  # 案例 1: 进料皮带 (抓取源头)
  # ----------------------------------------------------
  - roi_id: "roi_infeed_belt"
    name: "主进料输送带"
    frame_id: "world"
    category: "belt"
    enabled: true
    
    # 空间几何体 (已有底座)
    geometry:
      type: "box"
      center_xyz_mm: [0.0, 300.0, 50.0]
      size_xyz_mm: [200.0, 600.0, 100.0]
      rotation_rpy_deg: [0.0, 0.0, 0.0]
    visual_color_rgb: [0, 255, 128]

    # 【新增: Smart ROI 生产意图与角色】
    role: "source"                      # source: 进料提取区
    target_intent: "pose_pick"          # 业务意图: 抓取位姿提取
    pipeline_override: "feng_green_axis_v2" # 绑定的推荐算法
    min_confidence: 0.65                # 检出阈值

  # ----------------------------------------------------
  # 案例 2: 分离轮 8 托架阵列 (受控目的地)
  # ----------------------------------------------------
  - roi_id: "roi_wheel_slot_0"
    name: "分离轮-1号托架"
    frame_id: "frame_wheel"
    category: "tray"
    enabled: true
    geometry:
      type: "box"
      center_xyz_mm: [50.0, 10.0, 20.0]
      size_xyz_mm: [40.0, 260.0, 45.0]
      rotation_rpy_deg: [0.0, 0.0, 0.0]
    visual_color_rgb: [255, 165, 0]

    # 【新增: Smart ROI 生产意图与角色】
    role: "destination"                 # destination: 落料受控区
    target_intent: "piece_count"        # 业务意图: 根数统计
    binding:
      channel: "mqtt_loader"            # 通信通道代号
      slot_index: 0                     # 映射至 counts[0]
      capacity_max: 5                   # 满溢阈值 (根数)

  # ... roi_wheel_slot_1 至 slot_7 类似配置 (slot_index: 1~7)
```

---

## 4. 工具矩阵重构：研发工作室 vs 生产运行台

在 `tools/gui_launcher.py` 中，彻底理顺**研发评估工台（Studio）**与**生产运行时（Production）**的职责边界：

### 4.1 全局工具矩阵全景 (研、产、调、维 四维清晰划分)

在 Dashboard 控制中心 (`tools/gui_launcher.py`) 中，各个工具按照 **“空间资产 (A) → 研发调优 (B) → 产线生产 (C) → 硬件单体调试与运维 (D)”** 建立严格的层级矩阵，绝不混淆概念：

```
========================= A — Workspace 空间与资产 =========================
[1] Workspace Hub               : 工位沙盒管理、Tag 白名单、多坐标系拓扑、Smart ROI 空间集合配置
[3] 空间建图工作站 (Mapping Studio): 多视角稳健 BA 平差解算、立体建图 tags_map.yaml 生成

======================== B — 视觉算法研发工台 (R&D Studio) =================
[2] 图像采集向导 (Capture Wizard): 标定图集 (calibration) 与现场生产样本 (production) 交互采集
[4] 芦笋位姿工作室 (Pose Studio) : 纯粹算法试验台 (多流水线对比、参数精调、3D 抓取向量几何验证)
                                 ★【彻底剥离 G-code 导出与生成，专注感知本身】

======================= C — 产线实时生产运行 (Production Runtimes) =========
[5] SCARA 抓取生产工作台         : 原“Robot 在线跟踪”重构升级，进料皮带实时视觉感知 + 抓取闭环
[O] 分离轮分选生产工作台         : 8 托架多 ROI 实时切片识别 + 根数统计 + MQTT 节拍聚合闭环
    (Production 专攻“相机+算法+调度+通信”全自动闭环，支持产线大屏或无头流水线)

====================== D — 硬件单体底层调试与运维 (Diagnostics & Tools) =====
[6] RealSense 硬件诊断探针       : 双流画面纯硬件诊断、空间毫米级深度探针取样 (纯预览不落盘)
[7] SCARA 机械臂点动调试         : 轴点动 (W/S/A/D)、回零设零、MKS 串口指令与原生 G-code 透传
[8] Isolator WHEELS 调试        : 分离轮 ESP32 专属 MQTT 调试 (设备发现、状态监视、8 托架数量手动下发)
[9] NetCamera 调试               : 手机网络相机 RTSP 实时画面拉流、MQTT 遥控与推流参数调试
[T] 系统环境诊断与自动化测试     : 深度检查 Python/OpenCV/驱动环境，运行 tests/ 全量测试套件守门
[P] 安装/更新项目依赖            : 一键基于 requirements.txt 同步和锁定全局 Python 依赖环境
[W] AprilTag 标靶图纸生成        : 一键批量生成 ID 0~29 标靶高清图像与 100% 物理尺寸 A4 打印 PDF
```

> **特别澄清：单体调试（Group D）与 生产运行（Group C）的边界**：
> - **单体调试工具（如 `[8] Isolator WHEELS 调试`、`[7] SCARA 机械臂调试`）**：解决的是**“执行机构单机通不通”**的问题（手动点按步进器、手动发单条报文、排查电机与硬件接线），是底层的维护与验机手段，**必须完整保留，绝不可裁剪**；
> - **生产运行工作台（Group C）**：解决的是**“视觉感知 + 调度逻辑 + 硬件控制”三位一体的自动化闭环流水线**。单体调试正常是生产运行能够顺利启动的前提。

### 4.2 芦笋位姿工作室（Studio）职责净化方案

1. **移除项**：
   - 移除 `self.gcode_text` 状态变量；
   - 移除界面右上角的 `[导出G-code [E]]` 按钮与 G-code 代码预览框；
   - 移除对 `export_gcode_file()` 的调用。
2. **强化项（回归纯粹几何与算法评估）**：
   - 保留并强化 3D 视口中的抓取位姿矢量绘制（Pick Vector: $X, Y, Z$ 坐标中心、法向轴、偏航角 $Yaw$）；
   - 保留夹爪虚拟开合包围盒（纯几何碰撞示意，非机器指令）；
   - 保留批量解算报表（评估分割精度、检出率、解算耗时）。

---

## 5. 局部实现细节与代码解耦方案

### 5.1 视觉目标类（`AsparagusTarget`）纯净化

从 `src/vision/asparagus_analyzer.py` 中剥离所有运动控制逻辑，使其回归纯粹的 **DTO（数据传输对象）**：

```python
# 改造后的 AsparagusTarget: 纯感知事实，无任何控制指令
@dataclass
class AsparagusTarget:
    id: int
    # 图像几何
    center_px: Tuple[float, float]
    length_px: float
    width_px: float
    angle_deg: float
    
    # 世界/机械臂基准系几何 (mm / deg)
    world_x: float
    world_y: float
    world_z: float
    yaw_deg: float
    
    # 物理品质指标
    length_mm: float
    diam_mm: float
    straightness_ratio: float = 1.0       # 直度
    grade: str = "A"                      # 品质等级: A(一级) / B(二级) / C(次品)
    
    is_topmost: bool = False              # 是否为无遮挡最顶层
    confidence: float = 1.0
    
    # 彻底删除: def generate_gcode(...)
```

### 5.2 运动规划与 G-code 生成器独立封装

在 `src/control/` 下新建专属的运动规划器，单向消费感知层输出：

```python
# 文件: src/control/scara_motion_planner.py
class ScaraMotionPlanner:
    """SCARA 轨迹与 G-code 指令规划器 (独立于视觉感知层)"""
    
    def __init__(self, safe_z_mm: float = 80.0, feedrate_xy: int = 4000, feedrate_z: int = 1500):
        self.safe_z = safe_z_mm
        self.f_xy = feedrate_xy
        self.f_z = feedrate_z

    def plan_pick_and_place(self, pick_pose: Tuple[float, float, float, float],
                            place_pose: Tuple[float, float, float]) -> str:
        """
        根据拾取位姿 (X, Y, Z, Yaw) 与目标落料位姿 (X, Y, Z) 生成完整 G-code 序列
        """
        px, py, pz, pyaw = pick_pose
        dx, dy, dz = place_pose
        
        gcode = [
            "; --- SCARA Pick & Place 动作指令 ---",
            "G90",
            f"G0 Z{self.safe_z:.1f} F{self.f_xy}",
            f"G0 X{px:.2f} Y{py:.2f} R{pyaw:.2f} F{self.f_xy}",
            "M3",                                           # 夹爪预张开
            f"G1 Z{pz:.2f} F{self.f_z}",                    # 平稳下探
            "M4",                                           # 夹爪闭合
            "G4 P200",                                      # 夹持保压
            f"G0 Z{self.safe_z:.1f} F{self.f_xy}",          # 提升至安全高度
            f"G0 X{dx:.2f} Y{dy:.2f} R0.00 F{self.f_xy}",  # 移至落料槽上方
            "M3",                                           # 释放落料
            f"G0 Z{self.safe_z:.1f} F{self.f_xy}"
        ]
        return "\n".join(gcode)
```

### 5.3 业务调度层机制（`SortingDispatcher`）

衔接“1个皮带 Source”与“8个料槽 Destination”的业务调度逻辑：

```python
# 概念逻辑说明: 业务调度器核心
class SortingDispatcher:
    def __init__(self, workspace_rois: List[RoiDefinition]):
        self.source_rois = [r for r in workspace_rois if r.role == "source"]
        self.dest_rois = sorted([r for r in workspace_rois if r.role == "destination"],
                                key=lambda r: r.binding.get("slot_index", 0))

    def evaluate_cycle(self, detected_targets: List[AsparagusTarget],
                       slot_counts: List[int]) -> Optional[Tuple[AsparagusTarget, int]]:
        """
        输入: 进料区检测出的芦笋列表, 各槽位当前计数列表
        输出: 匹配出的 (抓取目标物料, 目标落料槽索引)
        """
        if not detected_targets:
            return None
            
        # 1. 优先抓取最顶层且置信度最高的物料
        best_target = max([t for t in detected_targets if t.is_topmost],
                          key=lambda x: x.confidence, default=detected_targets[0])
                          
        # 2. 根据品质等级匹配目标槽位，并检查容量上限 (Capacity)
        target_slot_idx = self._find_available_slot(best_target.grade, slot_counts)
        if target_slot_idx is None:
            log.warning("所有匹配槽位已满，等待产线换料或触发满溢报警！")
            return None
            
        return (best_target, target_slot_idx)
```

---

## 6. 演进路线图 (Roadmap & Next Steps)

| 阶段 | 核心任务 | 交付成果与验收标准 |
| :--- | :--- | :--- |
| **Phase 1: 数据模型升级与 Studio 职责净化** | 1. 扩展 `RoiDefinition`（增加 `role`, `target_intent`, `binding`）<br>2. 升级 `rois.yaml` 读写解析<br>3. 从 `AsparagusTarget` 剥离 `generate_gcode`<br>4. 从 `AsparagusPoseStudio` 移除 G-code 按钮与文本框 | `rois.yaml` 支持生产属性；Pose Studio 恢复纯粹算法验证台，界面清爽无报错。 |
| **Phase 2: 核心调度与控制解耦** | 1. 在 `src/control/` 建立 `ScaraMotionPlanner`<br>2. 建立 `src/control/sorting_dispatcher.py` 调度骨架<br>3. 单元测试验证：感知与规划完全解耦 | 单元测试通过，纯几何位姿可正确转化为动态落料点的 G-code。 |
| **Phase 3: 多态 Production GUI 落地** | 1. 重构原 `tools/tracker` 为 `tools/scara_production`<br>2. 建设 `tools/isolate_wheels_production` 8槽分选看板<br>3. 更新 `tools/gui_launcher.py` 目录卡片 | Dashboard 启动台具备清晰的四组卡片，两套生产运行时各自独立流畅运行。 |

---

## 7. 结论

本设计方案将原先交织在一起的“空间坐标”、“视觉算法”、“槽位分配”与“机械控制”彻底解耦：
- **Workspace** 统领全局生产模式；
- **Smart ROI** 明确局部感知意图；
- **视觉层** 专心看清世界；
- **调度与控制层** 统筹业务与机械执行。

此结构既消除了代码重复，又彻底规避了单一职责倒挂带来的维护灾难，为后续系统扩展提供了极其坚固的架构支撑。
