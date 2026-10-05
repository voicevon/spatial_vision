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

### A. Architectural Integrity & Project Rules (Strict)
- **Zero Backward Compatibility**:
  - Verify that no deprecated fields, legacy fallback switches, or compatibility shims are being added or retained.
  - Verify that data models adhere strictly to the Single Source of Truth (SSOT).
- **Temporary File Isolation**:
  - Verify that debug scripts, scratch notebooks, and temporary image/data outputs strictly reside under the project root's `temp/` folder.
  - Verify that production pipelines and unit tests NEVER depend on files inside `temp/`.

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

### E. 命名审查原则 (Naming Principles - 严格遵循)
- **严禁擅自直接修改命名**：
  - 针对**文件名、类名、函数名、变量名**，若审查中认为命名不合理或有优化空间，**严禁在代码中直接重命名或强行替换**。
- **必须提供 3~5 个候选名称供用户决策**：
  - 必须以“建议与候选方案”的形式提出，原则上为每个待优化项推荐 **3 个乃至 5 个备选名称**。
  - 需附带简要的语义分析或推荐理由。
- **交互式选择题确认机制 (Interactive Multiple-Choice Prompting)**：
  - 命名优化的确认步骤**必须使用交互式选择题（调用 `ask_question` 工具）**呈现给用户。
  - 将每个待优化项作为一道选择题，清晰列出 3~5 个候选选项供用户点选确认，严禁要求用户手动打字输入或擅自代用户做决定。用户选择后方可执行重命名。

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
