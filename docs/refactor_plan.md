# flux_vision_3d 重构计划

> 更新：2026-09-17。范围：全仓库代码审查（宏观 A1-A7 / 中观 M1-M5 / 微观 C1-C6）后的分级执行计划。

## 1. 决策记录

| 决策项 | 结论 |
|---|---|
| 优先级 | P0、P1 先行；P2 排后 |
| SCARA 命名 | src/ 硬件语境（机械臂本体描述）**保留 SCARA**；UI/工具层统一 Robot |
| 依赖 | matplotlib 已删；bleak / ultralytics 保留（后续 FR 预留） |
| 大文件 | flux_vision_3d.rar 已删除，`*.rar *.zip *.7z` 已入 .gitignore |
| 新类 | 已批准并创建：`src/utils/logger.py`（Logger）、`src/calibration/tag_detector.py`（TagDetector）、`src/calibration/camera_service.py`（CameraService）、`src/calibration/prism_renderer.py`（PrismRenderer） |

## 2. 已完成（P0，本轮）

| 项 | 内容 | 验证 |
|---|---|---|
| 清障 | 删 rar（590.8MB）、删 matplotlib 依赖、清 10 个过期 .pyc | - |
| Logger 基建 | `src/utils/logger.py` 幂等配置；src/ 9 文件 66 处 print 全部迁移 | 全量回归 OK |
| 吞异常修复 | ba_optimizer L300/L536、verification_visualizer L204/L352 → `log.warning` | 全量回归 OK |
| TagDetector 收编 | 三处重复检测实现（offline_engine / tag_map_builder / tag_capture_wizard）统一为 `src/calibration/tag_detector.py`；引擎保留委托属性，外部引用零破坏；精修失败记 warning | 96/96 tests OK |
| 过期测试修正 | test_tag_offline_studio 的 `LAUNCH_AR` → `LAUNCH_TRACKER`（旧命名遗留） | 96/96 tests OK |

## 3. P1 计划（下一批执行）

### 3.1 CameraService — 三套相机管理统一 ✅
- 现状：tracker/camera_controller.py、src/calibration/camera_streamer.py、wizard 内嵌 pipeline 三套 RealSense/USB 管理，各自启停、分辨率、曝光逻辑。
- 目标：新建 `src/calibration/camera_service.py`，向导/跟踪器/流式工具全部委托；GUI 状态（下拉选项等）留在工具层。
- 完成：CameraService 实现 RealSense/USB 双物理后端 + 分级回退链 + 内参回调；camera_controller/camera_streamer/wizard 三处委托收编。
- 验收：96/96 tests OK；ruff F821 通过。

### 3.2 棱柱渲染统一 ✅
- 现状：两套四棱柱绘制实现（src/calibration/verification_visualizer.py 与 wizard/tracker 内嵌投影绘制），样式不一致。
- 目标：抽 `src/calibration/prism_renderer.py`，纯函数式（输入内参/位姿/尺寸，输出绘制调用），verification_visualizer 与 tracker 复用。
- 完成：draw_prism 纯函数 + 三套配色常量（THEORY/OBSERVED/MAPPING）；verification_visualizer 双柱、tracker renderer、wizard/builder 建图四处统一委托。
- 验收：96/96 tests OK；ruff F821 通过。

### 3.3 窗口管理统一 ✅
- 现状：src/utils/window_helper.py（GUI 逻辑寄生核心库）与各工具散布的 cv2.namedWindow/moveWindow 双轨。
- 目标：window_helper 迁至 tools/（核心库去 GUI 化）；各工具统一经 helper 设置窗口。
- 完成：`src/utils/window_helper.py` 迁至 `tools/window_helper.py`，三处引用（studio/app、tag_manifest_reviewer、tag_capture_wizard）import 更新；原文件已删除。
- 验收：96/96 tests OK；ruff F821 通过。

### 3.5 config 收编 ✅
- 现状：多处绕过 config_guard 直接 `yaml.safe_load("config.yaml")`（builder、wizard、tracker 等）。
- 目标：统一经 `src/utils/config_guard` 提供的读取入口，白名单/检测参数单点加载。
- 完成：robot_serial / tag_map_builder / tag_capture_wizard / tag_super_extractor / tag_manager / diagnose_tag_frame 六处读取统一走 `load_raw_config`；yaml 写回（wizard save_config、tag_manager 白名单）保留原样；diagnose_tag_frame 死 yaml import 已删。

### 3.6 UI 硬编码收敛 ✅
- tracker/app.py 锚点文案 (0,0,405)/(0,520,196) 等改为从地图读取。
- 完成：PLANE_Z_CHOICES 拆为基础档位常量 + 锚点档位（从 tags_map 的 anchor_positions 动态注入）；锚点 " (Tag N)" 标注与启动横幅文案均动态生成，含锚点地图与空地图双路径冒烟验证。

### 3.4 上帝文件拆分 ✅（本轮执行完毕）

| 原文件 | 拆分后 | 拆出模块 |
|---|---|---|
| studio_renderer.py 1632 | 534 | studio_ui_common (122, 常量+绘制函数) / studio_frame_list (300, StudioFrameListMixin) / studio_center_view (332, StudioCenterViewMixin) / studio_inspector (442, StudioInspectorMixin) |
| studio/app.py 1150 | 754 | studio_events (206, StudioEventMixin) / studio_workflows (191, StudioWorkflowMixin) / studio_app_meta (13, PROJECT_ROOT 常量) |
| studio_state.py 869 | 487 | studio_data_actions (405, StudioDataActionsMixin) |
| tag_map_builder.py 815 | 547 | builder_workflow (297, CLI 流程编排, 0 新类)；原文件保留 main() 薄壳入口兼容 cli_menu 子进程调用 |
| ba_optimizer.py 877 | 747 | ba_report (178, compute_3d_uncertainties + export_diagnostic_report 纯函数, 类内委托, 0 新类) |

- 新增 6 个无状态 Mixin 类（已批准），全部方法逐字搬移、签名不变、零调用点变化；app.py 经 self.ui_renderer / data 委托访问不受影响。
- 验证：unittest 96/96 全绿；ruff F821 全项目通过；tag_map_builder CLI --help 与 tracker/studio 入口 import 冒烟通过。

## 4. P2（本轮执行完毕）

| 项 | 处置结果 |
|---|---|
| F401 未使用 import | 拆分遗留 26 文件 + tests 7 文件全部手工 Edit 清除；studio_renderer 5 个下拉常量重导出误删后已恢复（studio_events 经它转导入）。教训：单文件 F401 需先核查跨模块重导出链 |
| tools print 甄别迁移 | 585 处 / 19 文件：迁移 163 处诊断 print → Logger（级别按内容判定，冗余 [INFO]/[WARN]/[ERROR] 前缀剥离）；保留 ~420 处（cli_menu 菜单 UI、报告表格、json 数据输出、input 提示、\r 同行刷新） |
| tools 吞异常甄别 | 60 处 `except: pass`：20 处真异常加 `log.warning`；40 处保留 + 注释（热路径瞬态/可选依赖探测/清理容错/用户中断） |
| 魔法数字收敛 | TagDetector 亚像素精修 6 常量（窗口率 0.06/夹紧 3~9/迭代 40/精度 0.001px/漂移阈值 2.5px）；BA 两阶段统一收敛容差 `_BA_CONVERGE_TOL` |
| 根目录子目录 9→6 | scratch/ 与 temp/ 已删除（无需迁入 data/temp）；.ruff_cache 清理并加入 .gitignore；现存 config data docs src tests tools 共 6 个 |
| temp/ 脚本甄别 | 4 个有效回归收编为标准单测：tests/test_pnp_flip_regression.py、test_builder_pnp_regression.py、test_hp_detect_regression.py、test_prism_normal_regression.py；check_world_z / smoke_recog_overlay / scratch 诊断脚本与 png 产物删除 |

- 验证：unittest 100/100 全绿（原 96 + 新收编 4）；ruff F821 全项目通过；F401 仅剩 7 处有意重导出（src/utils/__init__.py 2 处 + studio_renderer 5 常量）。

## 5. 不做事项

- SCARA→Robot 全量替换：src/ 硬件语境保留（决策记录 #2）。
- cli_menu / env_utils 等独立小工具的深度重构：无功能痛点，仅随 P2 print 批次顺带清理。
