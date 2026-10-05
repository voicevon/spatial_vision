---
name: code-review
description: Conduct a systematic, rigorous code review for Git diffs, staged changes, pull requests, or specified source files. Evaluates architectural compliance, logic correctness, concurrency/resource safety, performance bottlenecks, and style.
---

# Code Review Protocol

Use this skill whenever the user asks for a code review, checks staged/uncommitted changes, audits a pull request, or evaluates code quality (e.g. "帮我做下代码评审", "审查最近的代码", "Review this PR/diff", "/review").

## 1. Scope Identification

1. **Detect Target Changes**:
   - For uncommitted/staged working tree changes: run `git status` and `git diff HEAD` (or staged diff `git diff --cached`).
   - For branch/PR comparisons: run `git diff main...HEAD` (or the relevant target branch).
   - For specific commits: run `git show <commit_hash>`.
   - For explicitly specified files: read and inspect the full content and recent diffs of those files.
2. **Context Gathering**:
   - Check project-level architecture constraints in [AGENTS.md](file:///d:/Software/asp_flux/spatial_vision/AGENTS.md) and related rule files.
   - Check if changes touch key interfaces, state machines, or algorithmic solvers.

## 2. Review Dimensions

Evaluate code against the following dimensions:

### A. 项目架构准则审查 (严格对齐 AGENTS.md)
- **零向后兼容 (Zero Backward Compatibility)**：核查是否引入任何废弃字段分支、fallback 回滚逻辑或内存别名胶水；必须单一真理源 (SSOT)。
- **临时文件收敛隔离 (Temp File Isolation)**：核查排查脚本、实验图象、分析导出是否严格且仅存放在 `temp/`；业务与测试代码严禁依赖 `temp/`。

### B. Correctness & Edge Cases
- **Logic & Flow**: Check for off-by-one errors, inverted conditions, unreachable code, unhandled edge cases (empty lists/dicts, `None` values, zero division).
- **Domain & Math Specifics**:
  - Matrix shapes and dimensionality consistency.
  - Coordinate frames, rotation conventions (quaternion vs rotation vector vs Euler), and metric units.
  - Invariants of transformation matrices (e.g., proper SO(3) / SE(3) validity).
- **Concurrency & State**:
  - Thread safety in GUI / hub state managers.
  - Proper task lifecycle handling (no orphaned background threads or event loops).

### C. Resource & Performance
- **Resource Management**:
  - Always prefer explicit context managers (`with`) for files, network sockets, camera streams, and locks.
  - Ensure hardware/device handles and GPU memory are explicitly released upon failure or teardown.
- **Efficiency**:
  - Avoid redundant deep copies or buffer allocations in high-frequency loops.
  - Avoid blocking calls on the UI thread or inside event loops.

### D. Testing & Type Safety
- **Type Annotations**: Comprehensive typing for function signatures and public APIs.
- **Test Coverage**: Ensure novel features, edge cases, and bug fixes are accompanied by unit tests.

### E. 命名审查原则 (严格对齐 AGENTS.md)
- **严禁擅自修改**：任何文件名、类名、函数名、变量名优化，严禁直接在代码中重命名或代用户做主；
- **提供 3~5 候选**：针对待优化命名，必须列出 3 至 5 个备选方案并附带推荐理由与侧重点；
- **交互式点选确认**：必须使用 `ask_question` 工具以单选/多选交互式选择题呈现，经用户明确点选确认后方可实施修改。

## 3. Output Format

Present the review results in clear, actionable markdown:

```markdown
### 📋 评审概述 (Overview)
- **改动范围**: [简要说明改动的模块与主要意图]
- **总体结论**: ✅ 通过 / ⚠️ 建议修改后合并 / ❌ 阻断性问题需重构

---

### 🔍 关键发现 (Findings by Severity)

#### 🔴 P0 - 阻断性问题 (Blocker / Must Fix)
> 严重逻辑缺陷、数据损坏隐患、违反核心架构准则（如残留向下兼容代码、临时文件污染源码目录）。
- **[文件路径:行号]**: 问题描述及影响。
  \`\`\`diff
  - 待修改代码
  + 建议修改代码
  \`\`\`

#### 🟡 P1 - 重要建议 (Major / Should Fix)
> 性能隐患、异常/边界条件缺失、潜在竞态、未妥善释放的资源。
- **[文件路径:行号]**: 改进说明与示例。

#### 🟢 P2 - 细节与代码风格 (Minor / Nitpick)
> 类型注解优化、文档与注释完善度、局部微调。
- **[文件路径:行号]**: 优化建议。

#### 🏷️ 命名优化建议 (Naming Proposals - 供用户选择)
> 规则：严禁直接修改，必须给出 3~5 个备选名字供用户裁定。
- **待优化项**: `[当前文件名 / 类名 / 函数名 / 变量名]`（位置：`path/to/file.py:L123`）
  - **当前问题**: 为什么现有命名不够合理/精准
  - **推荐候选方案 (请选择)**:
    1. `candidate_name_1` - [推荐理由/侧重点]
    2. `candidate_name_2` - [推荐理由/侧重点]
    3. `candidate_name_3` - [推荐理由/侧重点]
    4. `candidate_name_4` (可选) - [推荐理由/侧重点]
    5. `candidate_name_5` (可选) - [推荐理由/侧重点]

---

### 💡 综合建议与下一步行动 (Next Steps)
- [列出开发者合并/运行测试前需执行的动作]
```
