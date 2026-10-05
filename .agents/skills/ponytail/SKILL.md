---
name: ponytail
description: Prevent over-engineering and premature abstraction. Enforce YAGNI, maximize reuse of existing codebase and standard libraries, and write the minimum code necessary.
---

# Ponytail Protocol — 拒绝过度工程与极简极效设计

> "Think like the laziest senior dev in the room. The best code is the code you never have to write, debug, or maintain."

Ponytail 是面向 AI 辅助编程与工程落地的**反过度工程 (Anti-Overengineering) 规则集**。它强制要求在编写任何新代码、设计新架构前，按阶梯递进评估必要性。

---

## 核心优先阶梯 (The Priority Ladder)

在动手编写任何代码、创建新文件或引入新抽象之前，必须依次通过以下四级检验：

```text
[1. 必要性检验 (YAGNI)] ──(必须存在)──> [2. 存量复用检验] ──(无现成可用)──> [3. 原生能力检验] ──(无原生支持)──> [4. 极简直接实现]
         │                                      │                                    │
    (非当务之急)                           (已有相似实现)                        (标准库已提供)
         ↓                                      ↓                                    ↓
     直接不做                                直接复用/调用                        直接调用标准库
```

### 1. Does this need to exist? (必要性与 YAGNI 原则)
- **拒绝为假想未来买单**：严禁实现“以后可能会用到”的灵活性、参数、回调或配置项。只解决当前明确的工程诉求。
- **杜绝空头抽象**：只有 1 个实现类时，严禁提前抽象基类、协议类或工厂方法。
- **杜绝冗余装饰**：不要为一行简单赋值或查找编写冗长的包装方法（Wrapper）。

### 2. Does it already exist in the codebase? (优先复用既有模块)
- **搜索既有资产**：在实现新功能前，必须先在 `src/` 与 `tools/` 中查找是否有已验证的工具函数或领域类。
  - 路径与文件读写：优先复用既有 `image_io`、`config_guard` 等；
  - 视口与 GUI 组件：优先复用 `src/ui/` 中的通用主题与原生组件，严禁在页面中重复手写按钮/弹窗样式；
  - 几何解算与空间变换：优先调用 `src/calibration/solvers/` 与领域管理器，严禁散落手写变换矩阵逻辑。

### 3. Can the standard library or platform handle it? (优先标准库与原生能力)
- **零新依赖原则**：严禁因单一小功能随意 `pip install` 引入重量级第三方依赖。
- 优先依靠 Python 标准库（`math`, `os`, `pathlib`, `dataclasses`, `typing`, `json`, `yaml`）与核心基础设施（`OpenCV`, `NumPy`, `SciPy`）。

### 4. Write the minimum amount of code necessary (最小代码量与平铺设计)
- **扁平优于嵌套 (Flat is better than nested)**：优先使用简单的顶层函数与平铺的数据类 (`dataclass`)，减少不必要的类嵌套。
- **直观优于精巧**：不要为了所谓“优雅”使用过于隐晦的元编程、深层闭包或复杂继承链。
- **就地解决**：若改动仅有 2~3 行，直接在调用点清晰实现，避免为了抽函数而新增大量胶水文件。

---

## 与 spatial_vision 项目原则 (`AGENTS.md`) 的强协同

Ponytail 协议与本项目的核心准则天然契合：

1. **零向后兼容包袱 (No Backward Compatibility)**：
   - 绝不保留兼容废弃特性的 fallback、alias 或历史字段；
   - 废弃代码直接物理删除，杜绝代码库“既要旧又要新”的臃肿堆积。
2. **单一真理源 (SSOT)**：
   - 数据与状态必须只有唯一确切的持有者，严禁搞“多路同步”或“双轨制并存”。
3. **临时文件唯一出口 (`temp/`)**：
   - 实验脚本、中间分析日志随用随删，严禁污染业务代码库。
