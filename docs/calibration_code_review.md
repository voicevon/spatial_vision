# `src/calibration` 代码审查（算法为重点）

审查范围：`solvers/`（BA、世界对齐、子系外参、里程碑、共视图）、`verification/ba_report.py`、`manifest_repository.py`。
严重度：**P0** = 结果错误/系统性偏差，**P1** = 精度或鲁棒性问题，**P2** = 性能，**P3** = 规范/卫生（含违反 AGENTS.md 的项）。

---

## 一、P0 — 会直接导致结果错误或系统性偏差

### P0-1 BA 世界系 Z 轴 = 基准 Tag 的物理法向 → 高度（Z）系统性斜坡误差
- [ba_optimizer.py:L135](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L135) 把 `base_static_id` 的位姿固定为 `I`（规范自由度锚点）。这样一来，BA 坐标系的 Z 轴**严格等于这枚贴纸的物理法向**。
- [world_datum_aligner.py:L366-L370](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L366-L370) 的 `planar_2d` 分支只解 `yaw`，默认「BA Z 轴指向天」。
- 后果：贴纸倾斜 θ 时，距离 L 处的 Z 误差 ≈ `L·sinθ`。0.5° × 1 m ≈ **8.7 mm**。从锚点残差看，Z 误差会随 XY 位置**线性变化**（呈斜坡状）。这很可能就是上次「Tag 高度偏差」的根因之一。
- `planar_leveled` 分支本可以修正这个问题，但启用条件过严（[L321-L324](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L321-L324)）：要求 ≥3 枚已知 Z 的锚点，**且**它们的世界 Z 极差 ≤ 2 mm。只要台面上的锚点设计高度不同，调平就不会启用，静默退回 `planar_2d`。
- **建议**：把调平改成通用回归：对已知 Z 的锚点求解 `s·(R_tilt·p)_z + t_z = z_w`（roll/pitch/t_z 共 3 个参数，≥3 个点即可），不再要求共面、等高。
  更彻底的做法见 P1-1：做逐轴掩码的 7-DoF 联合最小二乘。另外，`planar_2d` 下应**输出警告**：「roll/pitch 未约束，Z 精度取决于基准 Tag 平贴度」。

### P0-2 BA 内的基线先验与 marker 尺寸先验互相冲突，并会累积改写状态
- [ba_optimizer.py:L257-L263](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L257-L263) 在残差里加了 `(dist_est - real_dist) * 5.0`。但尺度已经由 `obj_points`（marker 边长）唯一确定。两个尺度源不一致时，优化器会**非均匀地扭曲**整张地图来折中。而且量纲混用（mm×5 和 px 放在一起），这一项还会进入 Cauchy 核（f_scale 单位为 px）。
- 优化后 [L473](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L473) 又做了一次后验整体缩放。整体缩放本身是精确的（缩放全部 Tag + 相机 + marker 尺寸，投影不变），**前面那个残差先验就是多余且有害的**。
- [L482](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L482)、[L494](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L494) 会改写 `self.marker_size_mm`。同一实例第二次调用 `optimize` 时尺度会**累乘**，`self.obj_points` 却不更新，状态前后不一致。
- **建议**：删除残差里的基线项，只保留后验相似缩放。`marker_size_mm` 作为结果返回，不要回写实例。

### P0-3 两点对齐时绕连线的旋转不可观测，代码掩盖了而不是求解了
- [frame_extrinsic_solver.py:L77-L118](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L77-L118)：只用两个中心点，绕 A→B 轴的转角（1 个自由度）**根本无解**。`align_vectors_3d` 取最小测地旋转，相当于暗中假设「子系绕该轴的姿态与父系最接近」。结果取决于父系的朝向，是任意的。
- 文档说「法向残差严格为 0，消灭假超差」。实际上法向残差为 0 正是因为这个方向没有约束，而不是精度高。
- [L382-L385](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L382-L385)：未提供 `x_axis_local_xyz_mm` 时，用**实测距离**当名义距离，RMSE 恒为 0，质检形同虚设。
- **建议**：BA 已经给出每枚 Tag 的完整 6DoF 位姿（`T_w_t` 的旋转），应该用上：
  - 用 Tag 法向 / Tag 旋转约束绕轴角（例如两枚 Tag 法向的平均方向与子系名义法向对齐），或者
  - 位置 + 姿态联合配准：一枚 Tag 就能定 6DoF，两枚以上为超定。
  - 如果坚持只用中心点：两点模式必须在结果里标注「绕轴旋转 1DoF 未约束」，并引入显式先验（如保持子系 Z 与父系 Z 共向），不能静默给出数值。

### P0-4 世界锚点过滤失败时，会把子坐标系的局部坐标当成世界坐标
- [world_datum_aligner.py:L503](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L503)：`active_anchors = world_anchors if world_anchors else anchor_tags`。没有 world 锚点时，会把 `xyz_mm` 为**子系局部坐标**的 Tag（≥10）直接当世界真值去解相似变换，得到一个看似成功、实际错误的世界系。
- 这同时违反 AGENTS.md「严禁 fallback」。**建议**：`world_anchors` 为空时直接报错。

---

## 二、P1 — 精度与鲁棒性

### P1-1 世界对齐是分步估计（先尺度、再偏航、再平移），没有联合最优
- [world_datum_aligner.py:L244-L383](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L244-L383)：
  - 尺度取各点对比值的中位数，没有按基线长度加权。
  - 偏航对所有点对等权做圆均值（[L367](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L367)），短基线点对的角度噪声大，却与长基线等权。
  - 平移在 R、s 固定后逐轴取均值。
- `planar` 分支的 2D 尺度距离 `d_b` 用的是**调平前**的 BA XY（[L261](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L261)）。Tag 之间高差大时，Z 会混进 XY 距离。
- Umeyama 分支只用三轴全知的锚点，部分已知锚点（只知 Z）的信息被丢弃。
- **建议**：统一成一个求解器：参数为 `s, R(3), t(3)`，残差为逐锚点逐轴的 `known` 掩码残差，用 Umeyama / 2D Procrustes 的闭式解作初值，再用 `least_squares` 精修（毫秒级）。重力先验可以作为 roll/pitch 的弱先验项，而不是写死。三个分支即可合并，P0-1 也一并解决。
- 2D 偏航的闭式最优解（若暂时保留分支）：对去中心后的点，`yaw = atan2(Σ(b×w), Σ(b·w))`，天然按距离加权。

### P1-2 共线 / 退化判据用的是绝对阈值 1e-2 mm，形同虚设
- [world_datum_aligner.py:L304](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L304)、[L329](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L329)：`sv[1] > 1e-2`（单位 mm）。三点近似共线（例如横向偏差 2 mm）时仍会走 Umeyama，绕该线的转角极度病态。
- [frame_extrinsic_solver.py:L121-L157](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L121-L157) 的 Kabsch 完全没有退化检查。
- **建议**：改用相对判据，如 `sv[1]/sv[0] > 0.05`，并加物理下限（如 `sv[1]/√n > 30 mm`）。

### P1-3 BA 初值：单 Tag IPPE 有翻转二义性，且只取第一次看到的那一帧
- [ba_optimizer.py:L169-L185](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L169-L185)：未知 Tag 的初值取**遍历顺序中第一个**看到它的相机。IPPE_SQUARE 在小目标或近正对时有两个解，容易翻转，后续用多 Tag PnP 传递时会把错误放大。
- [L158](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L158)：多 Tag PnP 只有 1 枚已知 Tag（4 点共面）时同样存在二义性。
- **建议**：
  - 每枚 Tag 选面积最大、最正对的观测作初值，或者用 `solvePnPGeneric` 取两解，按多帧重投影一致性挑选。
  - 多 Tag PnP 改为 `SOLVEPNP_SQPNP` 或 `solvePnPRansac`，对初值中的粗差更稳。

### P1-4 不确定度计算的量纲和坐标系都不对
- [ba_report.py:L26-L60](file:///d:/Software/asp_flux/spatial_vision/src/calibration/verification/ba_report.py#L26-L60)：`res_stage2.jac` 是 scipy 中**已经乘过鲁棒核和权重**的「修正雅可比」，却配上**未加权**的 `rmse_px²` 作为 σ²，两者不配套。
- 不确定度是在 BA 尺度、BA 坐标系下算的，经过基线缩放或世界对齐（s≠1、R≠I）后**没有做 `s²·R Σ Rᵀ` 变换**，报告中的 σx/σy/σz 与报告中的世界坐标不对应。
- 基准 Tag 的 σ=0 是规范自由度造成的假象，其它 Tag 的 σ 实际上是「相对基准 Tag」的不确定度。
- **建议**：σ² 用 `2·cost/(m−n)`（与 jac 配套），协方差按相似变换传播，并在报告里注明「相对基准 Tag」。

### P1-5 离群检测与加权的细节
- [ba_optimizer.py:L383](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L383)：`obs_weights <= 0.05` 永远不会成立，因为 `compute_observation_weight` 的下限钳在 0.1。这是死代码。
- 鲁棒核作用在**加权后**的残差上（f_scale 为 px），低权重观测在进核之前就被缩小了，等于双重降权，`f_scale` 的物理意义也被扭曲。建议在核内只做鲁棒处理，权重改为表达 1/σ 的观测噪声模型。
- Cauchy 按 x、y 分量各自生效，而不是按角点或整次观测。可以接受，但离群判定是按整次观测（4 角点均值），两者口径不一致。
- [L459](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L459) 日志把 `nfev` 写成「迭代次数」，实为函数评估次数。

### P1-6 共视图的「关键桥梁」定义不对
- [covisibility_graph.py:L72](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/covisibility_graph.py#L72)：「只被 1 帧共视的边」不等于图论上的桥。这条边可能有其它路径冗余，而真正的单点故障（割边、割点）反而不会被报出。
- 连通只是必要条件。建议在 **帧-Tag 二部图**上用 Tarjan 求割点和桥，并统计每枚 Tag 的有效观测数和视角分布（基线角）。

### P1-7 子坐标系里程碑的若干隐患
- 世界对齐失败时，阶段 A 静默改用 marker 名义尺度的相对坐标（[multiframe_milestone_solver.py:L437-L440](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/multiframe_milestone_solver.py#L437-L440)）。marker 打印误差 1% 时，1 m 跨度会被误判为 10 mm 超差。建议改为显式标注「尺度未校准」或直接跳过。
- [L502](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/multiframe_milestone_solver.py#L502) 会**就地改写** `frame.calibration_spec`，副作用可能被持久化。
- M1 没有 RMSE 门限，只要有 Tag 就判合格（[L368](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/multiframe_milestone_solver.py#L368)）。
- Tag 区间按 `frame_id` 中的数字 ×10 推断（[L196-L204](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/multiframe_milestone_solver.py#L196-L204)、[L480-L487](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/multiframe_milestone_solver.py#L480-L487)），是隐式约定加两处重复实现。

---

## 三、P2 — 性能

### P2-1 BA 用稠密有限差分雅可比，规模稍大就很慢
- `least_squares` 没有传 `jac` 或 `jac_sparsity`（[ba_optimizer.py:L336](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L336)）。每次求雅可比都要对 n 个参数各做一次完整的 Python 循环和 `projectPoints`，并构造 m×n 稠密矩阵。30 帧 × 20 Tag 约 300 个参数，每次雅可比就要约 300 次全量评估。
- **建议**：传 `jac_sparsity`（每 8 个残差只依赖 1 个相机 + 1 个 Tag，共 12 个参数），配合 `tr_solver='lsmr'`，可提速 1 到 2 个数量级。进一步可以用 `cv2.projectPoints` 返回的 jacobian 拼出解析雅可比。
- 残差函数应向量化：预先把观测展平成数组，避免每次调用都重建 dict 和做 `np.linalg.inv`。

---

## 四、P3 — 规范与卫生（AGENTS.md「不做向后兼容」）

| 位置 | 问题 |
| --- | --- |
| [world_datum_aligner.py:L88-L133](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L88-L133) | 接受 `coords`/`position_mm` 别名；文档仍写着「兼容旧格式」，但代码并不支持（文档与代码不符） |
| [world_datum_aligner.py:L479](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L479) | 文档写「旧双锚点格式自动归一化」，已失效 |
| [world_datum_aligner.py:L505-L513](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L505-L513) | 预检时 `solve_similarity_from_anchors` 被重复调用一次，异常被吞掉 |
| [world_datum_aligner.py:L700-L711](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/world_datum_aligner.py#L700-L711) | X 轴对齐 Tag 缺失时自动降级到最远 Tag（静默兜底） |
| [frame_extrinsic_solver.py:L249-L254](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L249-L254) | 方法名别名：`multi_tag_registration`/`registration_3d`/`two_tag_alignment`/`single_tag` |
| [frame_extrinsic_solver.py:L351-L352](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L351-L352)、[L377-L378](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L377-L378) | 双字段回退 `origin_*`/`tag_a_*`；L377 的 `[:3]` 只作用在 `or` 右侧（运算符优先级 bug） |
| [frame_extrinsic_solver.py:L309-L319](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L309-L319) | 3 点配准静默降级为 2 点定轴（叠加 P0-3） |
| [frame_extrinsic_solver.py:L295-L297](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/frame_extrinsic_solver.py#L295-L297) | 地图中缺失的 Tag 被静默跳过，没有计入报错信息 |
| [multiframe_milestone_solver.py:L217-L222](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/multiframe_milestone_solver.py#L217-L222) | 「实验性兼容」死代码 |
| [manifest_repository.py:L195-L207](file:///d:/Software/asp_flux/spatial_vision/src/calibration/manifest_repository.py#L195-L207) | 4 级 glob 回退目录 |
| [ba_optimizer.py:L524](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L524) | `tag_family` 硬编码为 `DICT_APRILTAG_16h5` |
| [ba_optimizer.py:L298](file:///d:/Software/asp_flux/spatial_vision/src/calibration/solvers/ba_optimizer.py#L298) | 函数内重复 `import math` |
| 两处 `_rotation_to_rpy_deg` | BA 与 aligner 各有一份拷贝，应统一使用 `coordinate_manager.rot_mat_to_rpy_deg`（约定相同，均为外旋 xyz） |
| [manifest_repository.py:L291](file:///d:/Software/asp_flux/spatial_vision/src/calibration/manifest_repository.py#L291) | 深度 150~2200 mm、面积 120 px² 为魔法常数，应进入配置 |

已核对**无问题**的部分：Umeyama 公式及其反射修正、Kabsch 反射修正、Rodrigues 最小旋转（含 180° 分支）、RPY 约定（全系统统一为外旋 xyz）、不确定度的参数索引偏移、两点对齐的 RMSE（= |Δd|/2）。

---

## 五、建议修复顺序

1. **P0-1 + P1-1**：统一世界对齐求解器（掩码 7-DoF LSQ，加重力弱先验），直接针对 Tag 高度偏差。
2. **P0-4**：删除 `active_anchors` 兜底。
3. **P0-2**：删除 BA 内的基线残差，停止回写 `marker_size_mm`。
4. **P0-3**：子系外参引入 Tag 姿态约束，两点模式显式标注欠定。
5. **P2-1**：加 `jac_sparsity`（改动小、收益大）。
6. 其余 P1 / P3 项随重构一并清理。
