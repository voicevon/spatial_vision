"""
芦笋 3D 视觉特征分析与最顶层目标解算模块 (工业级黑帽暗缝实例切分与顶层解算)
=======================================================================
核心职责：
  1. 3D 深度浮凸主导 + 植物色自适应宽容度前景提取 (彻底免疫黑色/深色工作台)
  2. 工作台面最小二乘平面拟合，解算物料相对工作台的真实凸起净高度 (Relative Height)
  3. 基于水平黑帽暗缝 (Black-Hat Seam Detection) 与形态学切分的多根贴合芦笋实例分离 (Instance Separation)
  4. 基于 fitLine 主成分直线拟合，高精度解算芦笋中轴倾角 (Yaw deg) 与沿轴中心线
  5. 芦笋物理尺寸解算：物理长度 (Length mm)、截面直径 (Diameter mm)
  6. 掩膜中轴脊线抗噪深度采样 (Z_top, Z_center)
  7. 锁定最上层可抓取目标 (Topmost Pickable Target) 并输出 SCARA 夹爪位姿 (X, Y, Z, R)
  8. 集成 AprilTag 多标靶在线相机外参定位 (TagLocalizer)，三级标定降级链
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.utils.text_rendering import put_text


@dataclass
class AsparagusTarget:
    """单根芦笋感知与几何特征数据"""
    id: int
    center_px: Tuple[float, float]       # 图像像素中心 (u, v)
    length_px: float                     # 像素长度
    diam_px: float                       # 像素直径
    yaw_deg: float                       # 轴线在水平面的倾角 [-90°, 90°]，即夹爪目标旋转角 R
    axis_vector: Tuple[float, float]     # 沿长轴的归一化方向向量 (vx, vy)
    box_corners: np.ndarray              # 最小外接矩形 4 个角点像素坐标
    contour: np.ndarray                  # 轮廓多边形
    
    # 3D 物理尺寸 (单位: mm)
    length_mm: float                     # 物理实际长度 (经 3D 欧氏反投影恢复)
    diam_mm: float                       # 物理平均直径
    
    # 相机坐标系下的物理空间点 (单位: mm, 原点为镜头光心)
    grip_x: float                        # 相机系 X
    grip_y: float                        # 相机系 Y
    grip_z: float                        # 相机系 Z (沿光轴的镜头绝对深度，如 530mm)
    z_top: float                         # 芦笋顶面绝对深度 (越小越靠近相机)
    rel_height_mm: float                 # 相对工作台面的凸起净高度 (越大越在上层, 如 +45mm)
    
    # 机械臂 SCARA 世界抓取坐标系 (基座法兰原点, 运动面平行于传送带)
    robot_x: float = 0.0                 # SCARA 抓取 X (mm)
    robot_y: float = 0.0                 # SCARA 抓取 Y (mm)
    robot_z: float = 0.0                 # SCARA 垂直下探高度 (mm, 相对传送带台面)
    robot_r: float = 0.0                 # SCARA 夹爪末端偏航角 (deg)
    
    is_topmost: bool = False             # 是否被判定为最顶层目标
    calibration_source: str = "uncalibrated"  # 标定来源: "tag_online" | "tag_cached" | "hand_eye" | "uncalibrated"

    # 物理品质分级指标 (纯感知属性)
    straightness_ratio: float = 1.0      # 直度/弯曲比 (中轴直线距离/实际沿线积分长度)
    grade: str = "A"                     # 品质等级: A(一级/直且匀称), B(二级/轻微弯曲), C(次品/严重弯曲或残损)
    confidence: float = 1.0              # 检出综合置信度 (0.0 ~ 1.0)


class AsparagusAnalyzer:
    def __init__(self, fx: float = 909.12, fy: float = 907.46, cx: float = 647.46, cy: float = 377.51):
        """
        初始化分析器
        :param fx, fy, cx, cy: 彩色相机内参矩阵参数
        """
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        
        # 尺寸与滤波超参数 (匹配真实多根堆叠细长/特长与粗干芦笋)
        self.min_area = 600              # 最小像素面积
        self.min_length_mm = 60.0        # 最小物理长度 (mm, 涵盖短切与端头)
        self.max_length_mm = 550.0       # 最大物理长度 (mm, 充分容纳 400~500mm 特长芦笋)
        self.min_diam_mm = 5.0           # 最小物理直径 (mm, 涵盖笋尖)
        self.max_diam_mm = 65.0          # 最大物理直径 (mm, 涵盖大直径特级笋及并排局部接触)
        self.min_aspect_ratio = 1.8      # 最小长宽比 (兼顾倾斜投影与局部遮挡)
        self.table_margin_mm = 8.0       # 相对工作台面的凸起门限 (mm, 排除底板杂散反光，放行扁平下压笋)

        # 手眼标定矩阵 (Eye-to-Hand: 相机坐标系 -> SCARA 基座坐标系, 4x4 齐次矩阵)
        # 作为 AprilTag 在线定位不可用时的回退方案
        self.t_cam_to_scara: Optional[np.ndarray] = None
        self.is_hand_eye_calibrated: bool = False

        # AprilTag 在线相机定位器 (优先级最高的标定来源)
        self.tag_localizer = None  # type: Optional["TagLocalizer"]
        self.last_valid_tag_transform: Optional[np.ndarray] = None  # 历史锁定外参缓存
        self.last_tag_info: dict = {}  # 上一帧 AprilTag 定位的诊断信息

        # CV 视觉识别流水线调试图与中间过程缓存
        self.vis_stage1: Optional[np.ndarray] = None
        self.vis_dist: Optional[np.ndarray] = None       # 欧氏距离变换场 (Radius Energy Map)
        self.vis_peaks: Optional[np.ndarray] = None      # 垂向极大值峰脊线 (Transverse Ridge Peaks)
        self.vis_stage2: Optional[np.ndarray] = None
        self.vis_stage3: Optional[np.ndarray] = None
        self.last_pipeline_targets: List[AsparagusTarget] = []

    def set_hand_eye_matrix(self, t_matrix: Optional[np.ndarray]):
        """
        设置或更新相机到 SCARA 机械臂基座的手眼标定矩阵 (4x4)
        自动校验矩阵有效性；若仍为单位阵或非合法矩阵，则标记为未标定安全模式
        """
        if t_matrix is not None:
            mat = np.array(t_matrix, dtype=float)
            if mat.shape == (4, 4):
                self.t_cam_to_scara = mat
                is_identity = np.allclose(mat, np.eye(4), atol=1e-3)
                self.is_hand_eye_calibrated = not is_identity
                return
        self.t_cam_to_scara = None
        self.is_hand_eye_calibrated = False

    def set_tag_localizer(self, localizer):
        """
        设置 AprilTag 在线相机定位器实例
        :param localizer: TagLocalizer 实例 (已加载 tags_map.yaml)
        """
        self.tag_localizer = localizer

    def _resolve_calibration(self, color_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
        """
        三级标定降级链：解算当前帧的最优坐标变换矩阵与标定来源
        优先级：AprilTag 在线定位 > AprilTag 历史缓存 > 手工 SVD 标定 > 未标定防撞
        :return: (transform_matrix_4x4 或 None, calibration_source 字符串)
        """
        # 第一级：AprilTag 在线定位 (每帧实时 PnP 解算)
        if self.tag_localizer is not None:
            try:
                success, t_cam_to_world, info = self.tag_localizer.localize_camera(color_bgr)
                self.last_tag_info = info
                if success and t_cam_to_world is not None:
                    self.last_valid_tag_transform = t_cam_to_world.copy()
                    return t_cam_to_world, "tag_online"
            except Exception:
                pass  # 定位器异常不应中断主流程

            # 第二级：AprilTag 历史缓存外参 (标靶暂时不可见时沿用上帧)
            if self.last_valid_tag_transform is not None:
                return self.last_valid_tag_transform, "tag_cached"

        # 第三级：手工 SVD 点触标定矩阵 (config.yaml 中的 t_cam_to_scara)
        if self.is_hand_eye_calibrated and self.t_cam_to_scara is not None:
            return self.t_cam_to_scara, "hand_eye"

        # 第四级：完全未标定 — 启用防撞保护模式
        return None, "uncalibrated"

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float):
        """动态更新内参"""
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def fit_table_plane(self, depth_mm: np.ndarray) -> Optional[np.ndarray]:
        """
        使用最小二乘法鲁棒拟合工作台背景平面方程: Z_table(x, y) = a*x + b*y + d
        消除相机俯视时轻微俯仰/横滚角度引起的整幅图像渐变倾斜
        """
        h, w = depth_mm.shape
        # 实测工作台底板距离主要分布在 610mm ~ 670mm 之间
        valid = (depth_mm >= 610) & (depth_mm <= 670)
        if np.count_nonzero(valid) < 1500:
            # 备用方案：取深度较大（后 20% 分位）的有效点作为底板候选
            non_zero = depth_mm[depth_mm > 400]
            if len(non_zero) < 1000:
                return None
            p80 = np.percentile(non_zero, 80)
            valid = (depth_mm >= p80 - 20) & (depth_mm <= p80 + 30)
        
        # 降采样快速拟合
        step = 8
        y_grid, x_grid = np.indices((h, w))
        x_sub = x_grid[::step, ::step][valid[::step, ::step]]
        y_sub = y_grid[::step, ::step][valid[::step, ::step]]
        z_sub = depth_mm[::step, ::step][valid[::step, ::step]]
        
        if len(z_sub) < 300:
            return None
            
        A = np.column_stack([x_sub, y_sub, np.ones_like(x_sub)])
        plane_coeff, _, _, _ = np.linalg.lstsq(A, z_sub, rcond=None)
        return plane_coeff

    def get_table_tilt_angles(self, plane_coeff: Optional[np.ndarray]) -> Tuple[float, float, float]:
        """
        根据传送带拟合平面反算当前相机的物理安装倾角
        返回: (pitch_deg 俯仰角, roll_deg 横滚角, total_tilt_deg 综合空间倾角)
        """
        if plane_coeff is None:
            return (0.0, 0.0, 0.0)
        a, b, d = float(plane_coeff[0]), float(plane_coeff[1]), float(plane_coeff[2])
        if abs(d) < 1e-6:
            return (0.0, 0.0, 0.0)
        roll_deg = float(np.degrees(np.arctan2(a * self.fx, d)))
        pitch_deg = float(np.degrees(np.arctan2(b * self.fy, d)))
        norm_val = np.sqrt(1.0 + (a * self.fx / d)**2 + (b * self.fy / d)**2)
        total_tilt = float(np.degrees(np.arccos(1.0 / norm_val)))
        return (round(pitch_deg, 1), round(roll_deg, 1), round(total_tilt, 1))

    def compute_relative_height(self, depth_mm: np.ndarray, plane_coeff: Optional[np.ndarray]) -> np.ndarray:
        """
        计算每个像素相对工作台面的凸起净高度 (mm)
        Height_rel(x, y) = Z_table(x, y) - Z_actual(x, y)
        """
        h, w = depth_mm.shape
        y_grid, x_grid = np.indices((h, w))
        
        if plane_coeff is not None:
            table_z = plane_coeff[0] * x_grid + plane_coeff[1] * y_grid + plane_coeff[2]
        else:
            valid_depths = depth_mm[depth_mm > 400]
            table_z = np.median(valid_depths) if len(valid_depths) > 0 else 640.0
            
        rel_h = np.where(depth_mm > 0, table_z - depth_mm, 0.0)
        return np.maximum(0.0, rel_h)

    def render_height_map(self, depth_mm: np.ndarray, plane_coeff: Optional[np.ndarray], max_h_mm: float = 80.0) -> np.ndarray:
        """
        渲染传送带校准相对高度热力图 (Table-Relative Height / Elevation Map)
        原理：
          以拟合出的传送带黑色平面作为 Z=0 基准参考系，彻底消除相机大倾角俯视引起的底板倾斜渐变！
          - 台面表面 (rel_h <= 2.5mm)：统一置为平坦无畸变的深冷色基准底板；
          - 凸起物料 (rel_h > 2.5mm)：按相对台面垂直净高度 [0 ~ max_h_mm] 映射为渐变热力色彩：
            * 0~15mm: 青绿 (低层物料/细支)
            * 15~35mm: 翠绿/草绿 (底层芦笋)
            * 35~55mm: 明黄/橙色 (中层芦笋)
            * 55~80mm+: 鲜艳暖红/亮红 (最顶层优先抓取芦笋)
          并在画面右侧绘制精美的高度色标尺 (Height Color Bar)，标注毫米物理刻度。
        """
        h, w = depth_mm.shape
        rel_h = self.compute_relative_height(depth_mm, plane_coeff)
        
        # 归一化映射 (0 ~ max_h_mm) -> (0 ~ 255)
        clamped_h = np.clip(rel_h - 2.5, 0.0, max_h_mm)
        norm_h = (clamped_h / max_h_mm * 255.0).astype(np.uint8)
        
        # 使用 COLORMAP_TURBO 进行高动态对比色彩映射
        h_color = cv2.applyColorMap(norm_h, cv2.COLORMAP_TURBO)
        
        # 无效深度点 (0mm 盲区) 设为深灰色
        h_color[depth_mm == 0] = (20, 20, 20)
        
        # 相对台面 <= 2.5mm 的台面背景区域，赋予均匀的暗深蓝色底色 (消除相机倾角渐变)
        table_bg_mask = (depth_mm > 0) & (rel_h <= 2.5)
        h_color[table_bg_mask] = (35, 20, 10)
        
        # 在画面右上角/右侧绘制高度色阶标尺 (Height Color Bar)
        bar_w = 18
        bar_h = min(220, int(h * 0.45))
        bar_x = w - 85
        bar_y = 50
        
        # 生成垂直颜色阶梯 (从上到下: max_h_mm -> 0mm)
        gradient = np.linspace(255, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
        gradient_img = np.repeat(gradient, bar_w, axis=1)
        bar_bgr = cv2.applyColorMap(gradient_img, cv2.COLORMAP_TURBO)
        
        # 叠加标尺半透明背景框
        cv2.rectangle(h_color, (bar_x - 10, bar_y - 28), (w - 10, bar_y + bar_h + 20), (15, 15, 15), -1)
        cv2.rectangle(h_color, (bar_x - 10, bar_y - 28), (w - 10, bar_y + bar_h + 20), (70, 70, 70), 1)
        
        # 标题
        put_text(h_color, "Height(mm)", (bar_x - 6, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        
        # 标尺贴图与边框
        h_color[bar_y:bar_y + bar_h, bar_x:bar_x + bar_w] = bar_bgr
        cv2.rectangle(h_color, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (200, 200, 200), 1)
        
        # 标尺刻度与文本
        steps = [
            (0.0, f"+{int(max_h_mm)}mm"),
            (0.25, f"+{int(max_h_mm * 0.75)}mm"),
            (0.5, f"+{int(max_h_mm * 0.5)}mm"),
            (0.75, f"+{int(max_h_mm * 0.25)}mm"),
            (1.0, "0(台面)")
        ]
        for ratio, text in steps:
            curr_y = bar_y + int(ratio * bar_h)
            cv2.line(h_color, (bar_x + bar_w, curr_y), (bar_x + bar_w + 4, curr_y), (255, 255, 255), 1)
            put_text(h_color, text, (bar_x + bar_w + 7, curr_y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
            
        return h_color

    def extract_stage1_foreground(self, color_bgr: np.ndarray, depth_mm: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        【阶段 1】前景物料提取 (ExG 超绿 + 传送带 ROI 约束)
        :return: (fg_mask_full, (roi_x1, roi_x2, roi_y1, roi_y2))
        """
        h, w = color_bgr.shape[:2]
        # 传送带核心作业区域 ROI (自动剔除左右两侧支架和反光区域)
        roi_x1 = int(w * 0.35)
        roi_x2 = int(w * 0.81)
        roi_y1 = int(h * 0.02)
        roi_y2 = int(h * 0.98)

        roi_bgr = color_bgr[roi_y1:roi_y2, roi_x1:roi_x2].astype(np.float32)
        b, g, r = roi_bgr[:, :, 0], roi_bgr[:, :, 1], roi_bgr[:, :, 2]

        # 超绿特征 ExG 与亮度/色差联合约束
        exg = 2.0 * g - r - b
        gray = cv2.cvtColor(color_bgr[roi_y1:roi_y2, roi_x1:roi_x2], cv2.COLOR_BGR2GRAY)
        
        # 纯植物绿/嫩黄绿提取：ExG > 10 或 G 显著大于 B 且具有基本亮度
        fg_roi = ((exg > 10.0) | ((g > b * 1.05) & (gray > 42))).astype(np.uint8) * 255

        # 微弱开运算消除极微小反光毛刺 (严禁大闭运算，坚决保护并排芦笋之间的缝隙！)
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_roi = cv2.morphologyEx(fg_roi, cv2.MORPH_OPEN, k_open)

        # 若有深度，排除非工作台深度区域 (350mm ~ 780mm)
        if depth_mm is not None:
            depth_roi = depth_mm[roi_y1:roi_y2, roi_x1:roi_x2]
            valid_depth = (depth_roi >= 350) & (depth_roi <= 780)
            # 仅在有深度的区域进行深度约束
            fg_roi = np.where((depth_roi > 0) & (~valid_depth), 0, fg_roi)

        fg_full = np.zeros((h, w), dtype=np.uint8)
        fg_full[roi_y1:roi_y2, roi_x1:roi_x2] = fg_roi

        # 阶段 1 可视化渲染：原图压暗 + 绿色荧光高亮物料 + ROI 引导框
        vis = (color_bgr.astype(np.float32) * 0.45).astype(np.uint8)
        green_layer = vis.copy()
        green_layer[fg_full > 0] = (40, 235, 90)
        vis = cv2.addWeighted(green_layer, 0.65, vis, 0.35, 0)
        
        # 绘制传送带作业 ROI
        cv2.rectangle(vis, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.putText(vis, "CONVEYOR WORKSPACE ROI", (roi_x1 + 8, roi_y1 + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)

        # 统计前景像素面积
        fg_pixels = int(np.count_nonzero(fg_full))
        badge = f"STAGE 1: FOREGROUND (ExG+ROI) | Pixels: {fg_pixels}"
        cv2.rectangle(vis, (12, 12), (520, 48), (20, 20, 20), -1)
        cv2.rectangle(vis, (12, 12), (520, 48), (0, 235, 90), 2)
        put_text(vis, badge, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2)

        self.vis_stage1 = vis
        return fg_full, (roi_x1, roi_x2, roi_y1, roi_y2)

    def extract_stage2_spines(
        self,
        color_bgr: np.ndarray,
        fg_mask: np.ndarray,
        roi_box: Tuple[int, int, int, int],
        depth_mm: Optional[np.ndarray] = None,
        nominal_z_mm: float = 640.0
    ) -> List[dict]:
        """
        【阶段 2】独立脊线骨架与单体验证 (Spine & Ridge Tracing)
        原理：
          在欧氏距离变换场 (Distance Transform) 中，不论多根芦笋如何并排挨着，
          每一根芦笋的中轴线上都是截面半径的局部极大值峰（Ridge）。
          沿垂向 (Y 轴) 提取局部极大值峰线，横向 (X 轴) 桥接，即可彻底切分并排粘连物料！
        :return: 候选芦笋脊线字典列表
        """
        h, w = color_bgr.shape[:2]
        roi_x1, roi_x2, roi_y1, roi_y2 = roi_box
        fg_roi = fg_mask[roi_y1:roi_y2, roi_x1:roi_x2]

        if np.count_nonzero(fg_roi) < 200:
            self.vis_stage2 = color_bgr.copy()
            return []

        # 1. 距离变换 (计算前景像素到背景边界的最短欧氏距离，值即代表截面半径)
        dist = cv2.distanceTransform(fg_roi, cv2.DIST_L2, 5)

        # 生成【距离变换场】可视化图 (COLORMAP_TURBO 热力图，直观展现物料半径能量分布与贴合鞍部)
        vis_d = (color_bgr.astype(np.float32) * 0.25).astype(np.uint8)
        dist_norm = np.clip(dist / 28.0 * 255.0, 0, 255).astype(np.uint8)
        dist_color = cv2.applyColorMap(dist_norm, cv2.COLORMAP_TURBO)
        roi_patch = vis_d[roi_y1:roi_y2, roi_x1:roi_x2]
        fg_bool = fg_roi > 0
        roi_patch[fg_bool] = dist_color[fg_bool]
        vis_d[roi_y1:roi_y2, roi_x1:roi_x2] = roi_patch
        cv2.rectangle(vis_d, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_d, (12, 12), (560, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_d, (12, 12), (560, 48), (40, 230, 240), 2)
        put_text(vis_d, "CV: DISTANCE TRANSFORM (Radius Field & Seam Valleys)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (40, 230, 240), 2)
        self.vis_dist = vis_d

        # 2. 垂向局部极大值提取 (沿垂直芦笋长轴跨度方向做非极大值抑制 NMS，在贴合处天然出现凹陷谷底，只在物料中轴取峰值)
        kernel_v = np.ones((7, 1), np.uint8)
        dist_dil_v = cv2.dilate(dist, kernel_v)
        peaks = (dist == dist_dil_v) & (dist >= 4.0)

        # 生成【极大值峰脊线】可视化图 (点亮非极大值抑制后的中轴峰值点阵，证明并排缝隙处的数学解耦)
        vis_p = (color_bgr.astype(np.float32) * 0.35).astype(np.uint8)
        cnts_fg, _ = cv2.findContours(fg_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cf in cnts_fg:
            cv2.drawContours(vis_p, [cf + np.array([roi_x1, roi_y1])], -1, (60, 110, 75), 1)
        py, px = np.where(peaks)
        for y_pt, x_pt in zip(py, px):
            gx, gy = x_pt + roi_x1, y_pt + roi_y1
            cv2.drawMarker(vis_p, (gx, gy), (0, 255, 255), cv2.MARKER_CROSS, 4, 1)
        cv2.rectangle(vis_p, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 200, 255), 2)
        cv2.rectangle(vis_p, (12, 12), (580, 48), (20, 20, 20), -1)
        cv2.rectangle(vis_p, (12, 12), (580, 48), (0, 255, 255), 2)
        put_text(vis_p, "CV: TRANSVERSE NMS RIDGE PEAKS (De-coupling Seams)",
                 (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
        self.vis_peaks = vis_p

        # 3. 沿芦笋主轴方向横向形态学闭运算桥接 (形成连续中轴骨架)
        k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
        peaks_connected = cv2.morphologyEx(peaks.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k_h)

        cnts, _ = cv2.findContours(peaks_connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        scale = (nominal_z_mm / self.fx) if depth_mm is None else None

        candidates = []
        # 可视化图底图
        vis = color_bgr.copy()
        overlay = vis.copy()

        # 调色盘区分不同芦笋实例
        palette = [
            (255, 120, 40), (40, 230, 240), (240, 100, 220), (80, 240, 120),
            (255, 210, 40), (120, 160, 255), (200, 255, 80), (255, 80, 140)
        ]

        cand_id = 1
        for c in cnts:
            pts = c.reshape(-1, 2)
            if len(pts) < 8:
                continue

            rect = cv2.minAreaRect(c)
            (rcx, rcy), (rw, rh), _ = rect
            l_approx = max(rw, rh)
            if l_approx < 80:
                continue

            # 主轴拟合
            [vx, vy, x0, y0] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx[0]), float(vy[0])
            if abs(vx) < 0.45:
                continue  # 芦笋应大致平行传送带输送方向
            if vx < 0:
                vx, vy = -vx, -vy

            # 投影计算脊线长度
            proj = np.dot(pts - np.array([rcx, rcy]), np.array([vx, vy]))
            min_p, max_p = float(np.min(proj)), float(np.max(proj))
            len_px = float(max_p - min_p)
            if len_px < 100:
                continue

            # 在脊线上多点采样距离场半径
            sampled_radii = []
            spine_pts_img = []
            for s in np.linspace(min_p * 0.15, max_p * 0.85, 16):
                sx = int(round(rcx + s * vx))
                sy = int(round(rcy + s * vy))
                if 0 <= sx < fg_roi.shape[1] and 0 <= sy < fg_roi.shape[0]:
                    sampled_radii.append(float(dist[sy, sx]))
                    spine_pts_img.append((sx + roi_x1, sy + roi_y1))

            if not sampled_radii:
                continue

            avg_rad = float(np.median(sampled_radii))
            diam_px = float(avg_rad * 2.0)

            # 图像绝对中心
            global_cx = float(rcx + roi_x1)
            global_cy = float(rcy + roi_y1)

            # 物理尺度换算与先验尺寸过滤
            if depth_mm is not None:
                # 采样脊线上的真实深度
                sample_depths = []
                for (px_x, px_y) in spine_pts_img:
                    if 0 <= px_x < w and 0 <= px_y < h:
                        d_val = depth_mm[px_y, px_x]
                        if 350 <= d_val <= 780:
                            sample_depths.append(d_val)
                z_ref = float(np.median(sample_depths)) if len(sample_depths) >= 4 else nominal_z_mm
                local_scale = z_ref / self.fx
            else:
                z_ref = nominal_z_mm
                local_scale = scale

            l_mm = float(len_px * local_scale)
            d_mm = float(diam_px * local_scale)

            # 物理尺寸合规性校验：充分支持 6mm ~ 48mm 粗细 (涵盖特级粗笋 35mm)，长度 > 100mm
            if not (6.0 <= d_mm <= 48.0 and 100.0 <= l_mm <= 600.0):
                continue

            yaw_deg = float(np.degrees(np.arctan2(vy, vx)))
            if yaw_deg > 90.0: yaw_deg -= 180.0
            elif yaw_deg < -90.0: yaw_deg += 180.0

            # 构造紧凑外接定向矩形角点
            u_vec = np.array([vx, vy])
            v_vec = np.array([-vy, vx])
            half_l = len_px * 0.5
            half_w = max(diam_px * 0.5, 4.0)

            c_pt = np.array([global_cx, global_cy])
            p1 = c_pt - half_l * u_vec - half_w * v_vec
            p2 = c_pt + half_l * u_vec - half_w * v_vec
            p3 = c_pt + half_l * u_vec + half_w * v_vec
            p4 = c_pt - half_l * u_vec + half_w * v_vec
            box_corners = np.array([p1, p2, p3, p4], dtype=np.int32)

            color_theme = palette[(cand_id - 1) % len(palette)]

            # 阶段 2 可视化绘制：半透明定向外框 + 脊线中轴 + 采样半径圈
            cv2.fillPoly(overlay, [box_corners], color_theme)
            cv2.polylines(vis, [box_corners], True, color_theme, 2)
            
            # 白色高亮中心脊线
            sp_p1 = (int(global_cx - half_l * vx), int(global_cy - half_l * vy))
            sp_p2 = (int(global_cx + half_l * vx), int(global_cy + half_l * vy))
            cv2.line(vis, sp_p1, sp_p2, (255, 255, 255), 2)

            # 采样圆指示
            for (px_x, px_y) in spine_pts_img[::4]:
                cv2.circle(vis, (px_x, px_y), max(2, int(diam_px * 0.5)), (255, 255, 200), 1)

            # 实例标签
            tag_str = f"#{cand_id} D:{d_mm:.1f} L:{l_mm:.0f}"
            put_text(vis, tag_str, (int(global_cx - 30), int(global_cy - half_w - 6)),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            candidates.append({
                'id': cand_id,
                'center_px': (global_cx, global_cy),
                'length_px': len_px,
                'diam_px': diam_px,
                'length_mm': round(l_mm, 1),
                'diam_mm': round(d_mm, 1),
                'yaw_deg': round(yaw_deg, 1),
                'axis_vector': (vx, vy),
                'box_corners': box_corners,
                'z_ref': z_ref,
                'spine_pts': spine_pts_img
            })
            cand_id += 1

        # 混合半透明图层
        vis = cv2.addWeighted(overlay, 0.25, vis, 0.75, 0)
        badge2 = f"STAGE 2: SPINES & RIDGES | Identified Instances: {len(candidates)}"
        cv2.rectangle(vis, (12, 12), (540, 48), (20, 20, 20), -1)
        cv2.rectangle(vis, (12, 12), (540, 48), (40, 230, 240), 2)
        put_text(vis, badge2, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 240), 2)

        self.vis_stage2 = vis
        return candidates

    def estimate_stage3_poses(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        spines: List[dict],
        plane_coeff: Optional[np.ndarray] = None,
        frame_transform: Optional[np.ndarray] = None,
        frame_calib_source: str = "uncalibrated"
    ) -> List[AsparagusTarget]:
        """
        【阶段 3】位姿解算与排名前三位 (Top 3) 可抓取物料输出
        :return: 严格按优先级排序的排名前三位 AsparagusTarget 列表
        """
        targets: List[AsparagusTarget] = []

        for sp in spines:
            cx_val, cy_val = sp['center_px']
            vx_val, vy_val = sp['axis_vector']
            length_px = sp['length_px']
            diam_px = sp['diam_px']
            length_mm = sp['length_mm']
            diam_mm = sp['diam_mm']
            yaw_deg = sp['yaw_deg']
            box_corners = sp['box_corners']
            z_ref = sp['z_ref']

            if depth_mm is not None:
                # 采样沿中轴脊线的顶层深度
                spine_pts = sp.get('spine_pts', [])
                valid_ds = []
                for (px_x, px_y) in spine_pts:
                    if 0 <= px_x < color_bgr.shape[1] and 0 <= px_y < color_bgr.shape[0]:
                        dv = depth_mm[px_y, px_x]
                        if 350 <= dv <= 780:
                            valid_ds.append(dv)
                
                if len(valid_ds) >= 3:
                    z_top = float(np.percentile(valid_ds, 15))
                    z_med = float(np.median(valid_ds))
                else:
                    z_top = z_ref
                    z_med = z_ref

                # 计算相机坐标系下的 3D 抓取中心点 (X, Y, Z)
                grip_x = float((cx_val - self.cx) * z_med / self.fx)
                grip_y = float((cy_val - self.cy) * z_med / self.fy)
                grip_z = float(z_top)

                # 计算相对工作台的凸起净高度 (mm)
                if plane_coeff is not None:
                    table_z_local = plane_coeff[0] * cx_val + plane_coeff[1] * cy_val + plane_coeff[2]
                    rel_height_mm = float(table_z_local - z_top)
                else:
                    rel_height_mm = float(640.0 - z_top)
            else:
                z_top = 0.0
                z_med = z_ref
                grip_x = float((cx_val - self.cx) * z_ref / self.fx)
                grip_y = float((cy_val - self.cy) * z_ref / self.fy)
                grip_z = 0.0
                rel_height_mm = 0.0

            # SCARA 抓取坐标系转换
            if frame_transform is not None:
                p_cam_h = np.array([grip_x, grip_y, grip_z, 1.0])
                p_robot_h = frame_transform @ p_cam_h
                robot_x = float(p_robot_h[0])
                robot_y = float(p_robot_h[1])
                robot_z = float(p_robot_h[2])

                r_mat = frame_transform[:3, :3]
                v_cam = np.array([vx_val, vy_val, 0.0])
                v_robot = r_mat @ v_cam
                r_rad = np.arctan2(v_robot[1], v_robot[0])
                robot_r = float(np.degrees(r_rad))
                if robot_r > 90.0: robot_r -= 180.0
                elif robot_r < -90.0: robot_r += 180.0
            else:
                robot_x = float(grip_x)
                robot_y = float(grip_y)
                robot_z = float(rel_height_mm)
                robot_r = float(yaw_deg)

            target = AsparagusTarget(
                id=sp['id'],
                center_px=(cx_val, cy_val),
                length_px=length_px,
                diam_px=diam_px,
                yaw_deg=round(yaw_deg, 1),
                axis_vector=(vx_val, vy_val),
                box_corners=box_corners,
                contour=box_corners,
                length_mm=round(length_mm, 1),
                diam_mm=round(diam_mm, 1),
                grip_x=round(grip_x, 1),
                grip_y=round(grip_y, 1),
                grip_z=round(grip_z, 1),
                z_top=round(z_top, 1),
                rel_height_mm=round(rel_height_mm, 1),
                robot_x=round(robot_x, 1),
                robot_y=round(robot_y, 1),
                robot_z=round(robot_z, 1),
                robot_r=round(robot_r, 1),
                is_topmost=False,
                calibration_source=frame_calib_source
            )
            targets.append(target)

        # 排序：优先按相对台面凸起高度降序；纯 2D 时按面积和居中度排序
        if len(targets) > 0:
            if depth_mm is not None:
                targets.sort(key=lambda t: t.rel_height_mm, reverse=True)
            else:
                targets.sort(key=lambda t: (t.length_px * t.diam_px), reverse=True)

            # 严格保留排名前三位 (Top 3)
            targets = targets[:3]
            for rank_i, t in enumerate(targets):
                t.id = rank_i + 1
                t.is_topmost = (rank_i == 0)

        self.last_pipeline_targets = targets
        self.vis_stage3 = self.draw_detections(color_bgr, targets, sel_target_idx=0)
        return targets

    def analyze(
        self,
        color_bgr: np.ndarray,
        depth_mm: Optional[np.ndarray],
        stages: Tuple[bool, bool, bool] = (True, True, True)
    ) -> List[AsparagusTarget]:
        """
        三阶段透明流水线端到端解算入口
        :param stages: (run_stage1, run_stage2, run_stage3) 是否执行各阶段
        :return: 最终排名前三位的识别目标 (若未执行阶段 3 则返回空列表)
        """
        run_s1, run_s2, run_s3 = stages

        # 标定与平面拟合准备
        frame_transform, frame_calib_source = self._resolve_calibration(color_bgr)
        plane_coeff = self.fit_table_plane(depth_mm) if depth_mm is not None else None

        # 阶段 1：前景物料提取
        if not run_s1:
            self.vis_stage1 = None
            self.vis_stage2 = None
            self.vis_stage3 = None
            return []

        fg_mask, roi_box = self.extract_stage1_foreground(color_bgr, depth_mm)

        # 阶段 2：独立脊线骨架与单体验证
        if not run_s2:
            self.vis_stage2 = None
            self.vis_stage3 = None
            return []

        nominal_z = 640.0
        if plane_coeff is not None and abs(plane_coeff[2]) > 300:
            nominal_z = float(plane_coeff[2])

        spines = self.extract_stage2_spines(
            color_bgr, fg_mask, roi_box, depth_mm=depth_mm, nominal_z_mm=nominal_z
        )

        # 阶段 3：位姿解算与 Top 3 输出
        if not run_s3:
            self.vis_stage3 = None
            return []

        targets = self.estimate_stage3_poses(
            color_bgr, depth_mm, spines,
            plane_coeff=plane_coeff,
            frame_transform=frame_transform,
            frame_calib_source=frame_calib_source
        )

        return targets

    def draw_detections(self, image: np.ndarray, targets: List[AsparagusTarget], sel_target_idx: int = 0) -> np.ndarray:
        """
        在图像上绘制排名前三位的芦笋目标：
        高亮当前选中/顶层的芦笋 (荧光光晕、加粗双线轮廓、夹爪准星、详细数据卡片)，
        其余备选目标以对比色标注编号与简明尺寸。
        """
        annotated = image.copy()
        if not targets:
            return annotated

        # 确保选中序号合法
        sel_idx = max(0, min(sel_target_idx, len(targets) - 1))

        # 1. 针对选中的目标先绘制半透明发光填充遮罩 (高亮可选取的芦笋)
        sel_t = targets[sel_idx]
        overlay = annotated.copy()
        cv2.fillPoly(overlay, [sel_t.box_corners], (0, 230, 110))
        cv2.addWeighted(overlay, 0.28, annotated, 0.72, 0, annotated)

        # 2. 依次绘制各目标 (先画非选中的，后画高亮选中的，保证高亮图层置顶)
        draw_order = [i for i in range(len(targets)) if i != sel_idx] + [sel_idx]

        for idx in draw_order:
            t = targets[idx]
            is_sel = (idx == sel_idx)
            is_top = t.is_topmost

            cx_int, cy_int = int(t.center_px[0]), int(t.center_px[1])
            vx, vy = t.axis_vector

            if is_sel:
                # 高亮选中的目标：双层发光外框
                cv2.polylines(annotated, [t.box_corners], True, (0, 255, 120), 4)
                cv2.polylines(annotated, [t.box_corners], True, (255, 255, 255), 1)

                # 中心主轴线 (亮黄粗线)
                half_len = int(t.length_px * 0.46)
                p1 = (int(cx_int - half_len * vx), int(cy_int - half_len * vy))
                p2 = (int(cx_int + half_len * vx), int(cy_int + half_len * vy))
                cv2.line(annotated, p1, p2, (0, 255, 255), 3)

                # 夹爪开合面线 (亮红)
                half_diam = int(max(18, t.diam_px * 0.85))
                perp_p1 = (int(cx_int - half_diam * (-vy)), int(cy_int - half_diam * vx))
                perp_p2 = (int(cx_int + half_diam * (-vy)), int(cy_int + half_diam * vx))
                cv2.line(annotated, perp_p1, perp_p2, (0, 50, 255), 3)

                # 抓取瞄准十字准星与瞄准光圈
                cv2.circle(annotated, (cx_int, cy_int), 14, (0, 255, 120), 2)
                cv2.circle(annotated, (cx_int, cy_int), 6, (0, 0, 255), -1)
                cv2.drawMarker(annotated, (cx_int, cy_int), (255, 255, 255), cv2.MARKER_CROSS, 20, 2)

                # 详细数据悬浮卡片 (包含用户关心的：直径、长度、方向、高度)
                badge = f"TOP #1 [最优选取]" if is_top else f"#{t.id} [当前选取]"
                line1 = f"{badge}  D:{t.diam_mm}mm  L:{t.length_mm}mm"
                h_str = f"H:+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else "H:--"
                line2 = f"方向:{t.yaw_deg}deg  高度:{h_str}"
                line3 = f"SCARA ({t.robot_x}, {t.robot_y}, {t.robot_z}) R:{t.robot_r}"

                card_w, card_h = 290, 72
                bx1 = max(10, min(annotated.shape[1] - card_w - 10, cx_int - card_w // 2))
                by1 = max(10, cy_int - card_h - 22)
                bx2, by2 = bx1 + card_w, by1 + card_h

                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (18, 22, 28), -1)
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 120), 2)
                put_text(annotated, line1, (bx1 + 10, by1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 120), 2)
                put_text(annotated, line2, (bx1 + 10, by1 + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 255), 1)
                put_text(annotated, line3, (bx1 + 10, by1 + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)

            else:
                # 备选目标：清晰天蓝色框与紧凑标签
                cv2.polylines(annotated, [t.box_corners], True, (240, 180, 40), 2)

                # 轴线与十字
                half_len = int(t.length_px * 0.40)
                p1 = (int(cx_int - half_len * vx), int(cy_int - half_len * vy))
                p2 = (int(cx_int + half_len * vx), int(cy_int + half_len * vy))
                cv2.line(annotated, p1, p2, (200, 200, 200), 2)
                cv2.drawMarker(annotated, (cx_int, cy_int), (240, 180, 40), cv2.MARKER_CROSS, 12, 1)

                h_val = f"+{t.rel_height_mm}mm" if t.rel_height_mm > 0 else "--"
                label = f"#{t.id} D:{t.diam_mm} L:{t.length_mm} R:{t.yaw_deg} H:{h_val}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
                lx = max(6, min(annotated.shape[1] - lw - 10, cx_int - lw // 2))
                ly = max(lh + 6, cy_int - 12)
                cv2.rectangle(annotated, (lx - 4, ly - lh - 4), (lx + lw + 4, ly + 4), (20, 20, 20), -1)
                cv2.rectangle(annotated, (lx - 4, ly - lh - 4), (lx + lw + 4, ly + 4), (240, 180, 40), 1)
                put_text(annotated, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 220, 160), 1)

        return annotated

    def diagnose(self, color_bgr: np.ndarray, depth_mm: np.ndarray) -> str:
        """
        现场感知即时诊断工具：
        当未检测到有效芦笋时，全流程逐步检查并输出详细根因分析与调整建议
        """
        lines = []
        lines.append("【芦笋 3D 感知系统即时诊断报告】")
        h, w = depth_mm.shape
        
        # 1. 检查深度数据流健康度
        valid_depth = depth_mm[depth_mm > 0]
        valid_ratio = len(valid_depth) / (h * w) * 100
        lines.append(f"1. 深度点云健康度: 有效深度像素占比 {valid_ratio:.1f}% ({len(valid_depth)}/{h*w})")
        if len(valid_depth) < 5000:
            lines.append("   [!] 严重警告: 深度点云极为稀疏 (<5000点)！可能原因: 黑色传送带反光散射/激光散斑被遮挡/距离超出相机量程。")
            lines.append("   [建议] 检查相机镜头前有无异物，按 [L] 确认激光器已开启，或微调激光功率。")
            return "\n".join(lines)
            
        d_min = np.min(valid_depth)
        d_max = np.max(valid_depth)
        d_med = np.median(valid_depth)
        lines.append(f"   - 深度分布: 最近 {d_min}mm, 中位数 {d_med:.0f}mm, 最远 {d_max}mm")
        
        # 2. 传送带底板拟合诊断
        plane_coeff = self.fit_table_plane(depth_mm)
        if plane_coeff is None:
            lines.append("2. 传送带底板平面拟合: 【失败】")
            lines.append(f"   [!] 未能在 610~670mm 区间或高位深层找到足够密集的基准点。当前视野中位深度为 {d_med:.0f}mm。")
            lines.append("   [建议] 当前相机安装高度可能偏离 640mm 标准高度，建议检查机台物理安装或微调拟合区间。")
        else:
            pitch, roll, tilt = self.get_table_tilt_angles(plane_coeff)
            lines.append(f"2. 传送带底板平面拟合: 【成功】 (Pitch={pitch:+.1f}°, Roll={roll:+.1f}°, 综合倾角={tilt:.1f}°)")
            
        # 3. 前景物料提取与相对凸起高度
        rel_h = self.compute_relative_height(depth_mm, plane_coeff)
        h_max = float(np.max(rel_h))
        h_p98 = float(np.percentile(rel_h[rel_h > 0], 98)) if np.count_nonzero(rel_h > 0) > 100 else 0.0
        lines.append(f"3. 相对底板净凸起高度: 最高点={h_max:.1f}mm, 98分位凸起={h_p98:.1f}mm (门限: >={self.table_margin_mm}mm)")
        if h_p98 < self.table_margin_mm:
            lines.append(f"   [!] 警告: 视野内几乎无高于台面 {self.table_margin_mm}mm 的凸起物体！传送带上可能未放置物料，或物料过于贴平。")
            return "\n".join(lines)
            
        # 4. 植物色域与连通域切分
        contours = self.segment_and_separate(color_bgr, depth_mm, rel_h)
        lines.append(f"4. 实例分割与黑帽暗缝切分: 提取出 {len(contours)} 个候选轮廓 (最小面积门限: {self.min_area}px)")
        if len(contours) == 0:
            lines.append("   [!] 警告: 前景掩膜提取为空！可能原因: 物料颜色不满足嫩绿/黄绿植物色域，或光照严重过曝/欠曝。")
            return "\n".join(lines)
            
        # 5. 逐个候选轮廓的几何过滤诊断
        lines.append("5. 候选轮廓几何规格过滤详情:")
        for idx, cnt in enumerate(contours[:6]):  # 最多打印前 6 个
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                lines.append(f"   - 轮廓 #{idx+1}: 面积 {area:.0f}px < {self.min_area}px [过滤: 碎片杂质]")
                continue
            [vx, vy, x0, y0] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            pts = cnt.reshape(-1, 2).astype(float)
            diff = pts - np.array([float(x0[0]), float(y0[0])])
            proj_len = np.dot(diff, np.array([float(vx[0]), float(vy[0])]))
            proj_wid = np.dot(diff, np.array([-float(vy[0]), float(vx[0])]))
            l_px = float(np.max(proj_len) - np.min(proj_len))
            d_px = float(np.max(proj_wid) - np.min(proj_wid))
            asp = l_px / max(1.0, d_px)
            if asp < self.min_aspect_ratio:
                lines.append(f"   - 轮廓 #{idx+1}: 长宽比 {asp:.2f} < {self.min_aspect_ratio} [过滤: 非细长棒体/块状杂物]")
                continue
            # 深度采样
            c_mask = np.zeros(color_bgr.shape[:2], dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            valid_d = depth_mm[(c_mask > 0) & (depth_mm > 350) & (depth_mm < 700)]
            if len(valid_d) == 0:
                lines.append(f"   - 轮廓 #{idx+1}: 区域内无有效深度数据 [过滤: 测距盲区]")
                continue
            z_med = float(np.median(valid_d))
            l_mm = float(l_px * z_med / self.fx)
            d_mm = float(d_px * z_med / self.fx)
            if not (self.min_length_mm <= l_mm <= self.max_length_mm):
                lines.append(f"   - 轮廓 #{idx+1}: 估算长度 {l_mm:.1f}mm 超出范围 [{self.min_length_mm}, {self.max_length_mm}]mm [过滤]")
                continue
            if not (self.min_diam_mm <= d_mm <= self.max_diam_mm):
                lines.append(f"   - 轮廓 #{idx+1}: 估算直径 {d_mm:.1f}mm 超出范围 [{self.min_diam_mm}, {self.max_diam_mm}]mm [过滤]")
                continue
            lines.append(f"   - 轮廓 #{idx+1}: 【合规目标】 L={l_mm:.1f}mm, D={d_mm:.1f}mm")
            
        lines.append("------------------------------------------------------------")
        lines.append("【调试建议】")
        lines.append("  1. 若芦笋偏长或偏粗，系统已将长径放宽至 550mm / 65mm。")
        lines.append("  2. 如自动识别暂未锁定，您可直接用【鼠标左键点击】画面上任意芦笋，")
        lines.append("     准心将锁定该点，按 [G] 键将直接输出该示教点的防撞抓取 G-code！")
        return "\n".join(lines)
