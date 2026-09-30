# 架构规则：严禁向后兼容 (No Backward Compatibility)

在本项目的所有功能实现与重构中，**严禁引入任何向后兼容性代码**：

1. **零兼容分支 (Zero Compatibility Branches)**：
   - 杜绝在代码中编写 `if old_field is None: try legacy_field`。
   - 杜绝支持已废弃的历史配置格式（如 `tag_anchors`、`whitelist_tag_ids`、双锚点 `origin_xyz_mm/align_xyz_mm`）。

2. **零内存兼容补丁 (Zero In-Memory Compatibility Patches)**：
   - 严禁在加载数据后为了迎合旧组件而向字典中挂载别名视图（如 `data["tag_anchors"] = data["anchor_tags"]`）。
   - 调用方必须直接适配官方标准字段，不满足要求的调用方代码直接重构。

3. **物理剔除冗余字段与废弃接口 (Hard Removal of Deprecated Interfaces)**：
   - 废弃的旧文件（如 `anchor_tags.yaml`）直接由单真理源（`tag_whitelist.yaml`）接管，不再保留双源读取。
   - 废弃函数直接删除，单测同步更新为直接断言新规范。

4. **临时与实验文件收敛准则 (Single Fixed Temp Directory)**：
   - 所有一次性调试脚本、临时实验图象、数据分析导出，**唯一固定存放在根目录 `temp/` 下**；
   - 严禁在根目录或业务目录（`src/`、`tools/`、`tests/`、`data/`）下随意创建临时文件或创建其他同义目录（如 `scratch/`、`reports/` 等）；
   - `temp/` 被 `.gitignore` 永久忽略，用完即弃，正式业务代码绝不依赖 `temp/` 内容。
