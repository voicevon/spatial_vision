---
name: refactor-protocol
description: Step-by-step protocol for refactoring spatial_vision code (splitting large modules, moving responsibilities, removing deprecated interfaces) while honoring AGENTS.md. Use when the user asks to refactor a file, class, or function.
---

# Refactor Protocol

准则以根目录 [AGENTS.md](file:///d:/Software/asp_flux/spatial_vision/AGENTS.md) 为准，本 Skill 只定义执行步骤。

## 1. 诊断（先读后动）

1. 通读目标文件，并用 `grep_search` 找出所有调用点与测试引用。
2. 列出具体坏味道及位置（职责混杂、硬件/SDK 抽象泄漏、重复定义、边界缺陷），按风险排序。
3. 给出方案与风险，等用户确认后再改动；用户意图不明时用 `ask_question` 澄清。

## 2. 命名决策

- 拆分出的新文件、类、函数的命名一律走 `ask_question`：3~5 个候选 + 理由，用户点选后才能落地。
- 严禁擅自重命名现有标识符。

## 3. 实施规则

- **不留兼容层**：旧接口调用处全量迁移，然后物理删除旧实现；不保留别名、fallback、双轨读取、为旧调用方做的 re-export 垫片。
- **单一真理源**：同一规则、配置、逻辑只在一处定义（如硬件操作只走 `CameraService`，工位标靶配置只走 `tag_whitelist.yaml`）。
- 保留与改动无关的注释和 docstring。
- 临时脚本、实验输出只放 `temp/`。

## 4. 验证

1. 先跑相关单测，再按 `run-tests` Skill 跑全量回归。
2. 为重构中修复的缺陷或新增的边界行为补单测。
3. `git status -s` 确认只有预期文件被改动，运行时配置（如 `config/gui_settings.json`）不得混入。

## 5. 收尾汇报

用简表对比“问题 → 重构后做法”，给出测试结果（N tests OK），并列出尚未处理的遗留项。
