---
name: run-tests
description: Run the spatial_vision unit test suite (unittest) and verify results and working-tree cleanliness. Use after any code change, refactor, or before declaring a task done.
---

# Run Tests Protocol

## 1. 命令

- 单文件：`python -m unittest tests/test_xxx.py`
- 全量回归：`python -m unittest discover tests`（约 1 分钟，需用后台模式运行并等待完成通知，不要轮询刷屏）
- 工作目录必须是项目根目录 `spatial_vision/`。

## 2. 结果判定

- 以输出末尾的 `Ran N tests ... OK` 为准；出现 `FAILED` / `ERROR` 必须定位并修复，禁止通过删除或弱化断言来“变绿”。
- 日志里的 `[ERROR]` / `[WARNING]` 行不等于失败（部分用例会故意触发，如锚点几何冲突），以 unittest 汇总结论为准。
- Windows 控制台为 GBK，日志中文可能显示乱码，属显示问题，不影响判定；需要读中文时查看后台任务的 log 文件。

## 3. 跑完后的工作区清洁检查（必做）

1. 执行 `git status -s`，确认改动仅包含本次任务预期的文件。
2. 若 `config/gui_settings.json` 等运行时配置被改动，说明某个测试写了真实配置，属于**测试隔离缺陷**：
   - 先 `git checkout config/gui_settings.json` 还原；
   - 再定位并修复该测试（用 `patch.object` 将模块级 `GUI_SETTINGS_FILE` 重定向到临时目录），不要把这类改动混入提交。
3. 临时脚本只能放在 `temp/`（见 AGENTS.md），用完即删。

## 4. 新增/修改功能时

- 新功能、边界条件、Bug 修复必须附带单测；测试依赖的文件、目录一律使用 `tempfile`，严禁依赖真实 `data/` 或 `config/` 内容。
