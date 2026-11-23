---
name: gui-tool-development
description: Architecture guide for OpenCV-based interactive tools in spatial_vision. Covers BaseCvApp lifecycle, Renderer read-only separation, CameraService hardware encapsulation, GUI settings isolation, and unit test patterns.
---

# GUI Tool Development Guide (OpenCV Tools)

本项目交互式桌面工具（`tools/` 目录下各 Studio / Wizard / Hub）基于 OpenCV 实现，遵循统一的架构分层规范。

## 1. 架构分层约定

```text
tools/<tool_name>/
├── app.py / <tool_name>.py    # 控制器: 继承 BaseCvApp, 管理业务状态、事件分发与调度
├── renderer.py                # 渲染器: 只读状态引用, 画布合成与按钮命中表 (buttons) 生成
└── states/ 或 handlers/        # 复杂工具的状态机与模态弹窗处理器 (如 workspace_hub)
```

### 核心分层准则：
1. **硬件操作完全委托 `CameraService`**：
   - 严禁在 `tools/` 代码中直接 `import pyrealsense2` 或直接操作 `rs.sensor`；
   - 相机启停、多档位降级、曝光调节、自动曝光切换，必须统一通过 `src.devices.camera_service.CameraService`。
2. **渲染器 (Renderer) 纯只读**：
   - `Renderer` 仅持有主控制器状态的只读引用，负责绘制与返回每帧按钮命中表 `[(btn_id, (x1, y1, x2, y2), payload)]`；
   - 业务状态修改一律由控制器在 `on_click` / `_handle_action` 中执行。
3. **视觉规范统一**：
   - 颜色、边框、交互样式统一引用 `src.ui.gui_theme.GuiTheme`，禁止硬编码散落 BGR 数值。

## 2. 状态持久化与测试隔离 (零污染原则)

- GUI 偏好统一存储在 `config/gui_settings.json`，各工具按 `APP_ID` 拥有独立子树。
- **单测铁律**：测试中构造带持久化的 App 时，**绝对禁止读写真实 `config/gui_settings.json`**：
  - 构造函数若支持 `settings_file` 参数，必须传入临时文件路径；
  - 模块级 `GUI_SETTINGS_FILE` 变量，必须在 `setUp` 中使用 `patch.object(mod, 'GUI_SETTINGS_FILE', tmp_settings)` 重定向并在 `tearDown` 恢复。

## 3. 基础测试模板

```python
import tempfile, unittest
from unittest.mock import patch, MagicMock

class TestMyToolApp(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.tmp_settings = os.path.join(self.temp_dir, "gui_settings.json")
        # 隔离 GUI 偏好
        p = patch.object(my_tool_mod, "GUI_SETTINGS_FILE", self.tmp_settings)
        p.start()
        self.addCleanup(p.stop)

    def test_smoke(self):
        app = MyToolApp(settings_file=self.tmp_settings)
        canvas = app.render()
        self.assertIsNotNone(canvas)
```
