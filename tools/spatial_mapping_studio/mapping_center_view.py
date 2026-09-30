"""
空间建图工作站 - 中栏视口渲染 Mixin (MappingCenterViewMixin)
================================================================================
承载 MappingRenderer 的中栏绘制分区：
1. render_center_viewport: 高清工作视口, 等比居中自适应渲染与双下拉控制菜单
2. overlay_visual_elements: 标靶标注、3D 双棱柱与残差矢量叠加渲染
仅包含纯绘制方法, 不持有任何状态; 通过 self 依赖宿主 MappingRenderer 的其他方法,
由 MRO 解析跨分区调用。
"""

import os
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.ui.text_rendering import put_text
from src.ui.gui_theme import GuiTheme
from src.ui.gui_components import draw_dropdown_button, draw_dashboard_button

BA_VIEW_OPTIONS = GuiTheme.BA_VIEW_OPTIONS
OBS_VIEW_OPTIONS = GuiTheme.OBS_VIEW_OPTIONS


class MappingCenterViewMixin:
    """中栏视口渲染 Mixin (由宿主类 MappingRenderer 组合)"""

    def render_center_viewport(self, app: Any, canvas: np.ndarray, x: int, y: int, w: int, h: int):
        """中栏：高清工作视口，等比居中自适应渲染"""
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (14, 15, 18), -1)

        if not app.data_mgr.image_files:
            put_text(canvas, "未扫描到采图图像 (当前工位 raw_images/ 为空)", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1, cv2.LINE_AA)
            return

        cur_file = app.data_mgr.image_files[app.data_mgr.current_img_idx]
        bgr = cv2.imread(cur_file)
        if bgr is None:
            put_text(canvas, f"读取图像文件失败: {cur_file}", (x + 100, y + h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
            return

        disp_frame = bgr.copy()
        base_name = os.path.basename(cur_file)
        meta = app.data_mgr.frame_metrics_cache.get(base_name, {})
        obs_list = meta.get("observations", [])

        # 叠加标靶与 3D 双棱柱
        self.overlay_visual_elements(app, disp_frame, obs_list, meta.get("is_excluded", False), meta=meta,
                                     panel_rect=(x, y, w, h))

        # 视口等比与平移缩放渲染 (委托给 viewport 控制器)
        frame_h, frame_w = disp_frame.shape[:2]
        rois = app.viewport.compute_viewport_render_rois((x, y, w, h), frame_w, frame_h)
        if rois is not None:
            (src_x1, src_y1, src_x2, src_y2), (dst_x1, dst_y1, dst_x2, dst_y2) = rois
            src_roi = disp_frame[src_y1:src_y2, src_x1:src_x2]
            dst_w = dst_x2 - dst_x1
            dst_h = dst_y2 - dst_y1
            if dst_w > 0 and dst_h > 0 and src_roi.size > 0:
                interp = cv2.INTER_LINEAR if app.viewport.zoom_level > 1.2 else cv2.INTER_AREA
                resized_roi = cv2.resize(src_roi, (dst_w, dst_h), interpolation=interp)
                canvas[dst_y1:dst_y2, dst_x1:dst_x2] = resized_roi

        # 视口外边框
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (55, 60, 70), 1)

        # 视口左上角：所有控件单行水平排列 (BA 理论 / 实测识别 / 绘制ROI物件 / 绘制XY平面 / Z轴高度选择)
        row_y1 = y + 8
        row_y2 = row_y1 + 22
        cursor_x = x + 12

        # 1. BA 理论下拉框
        ba_x1, ba_x2 = cursor_x, cursor_x + 148
        cur_ba_label = dict(BA_VIEW_OPTIONS).get(app.ba_view_mode, "3D 翡翠绿棱柱")
        is_ba_open = (app.active_dropdown == "BA_VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (ba_x1, row_y1, ba_x2, row_y2), cur_ba_label,
                             is_open=is_ba_open, mouse_pos=app.mouse_pos, prefix="BA理论: ")
        app.dropdown_boxes["BA_VIEW_DROPDOWN"] = {
            "rect": (ba_x1, row_y1, ba_x2, row_y2),
            "options": BA_VIEW_OPTIONS,
            "active_key": app.ba_view_mode
        }
        app.gui_buttons.append(("TOGGLE_BA_VIEW_DROPDOWN", (ba_x1, row_y1, ba_x2, row_y2), "BA_VIEW_DROPDOWN"))
        cursor_x = ba_x2 + 8

        # 2. 实测识别下拉框
        obs_x1, obs_x2 = cursor_x, cursor_x + 148
        cur_obs_label = dict(OBS_VIEW_OPTIONS).get(app.obs_view_mode, "3D 科技天蓝棱柱")
        is_obs_open = (app.active_dropdown == "OBS_VIEW_DROPDOWN")
        draw_dropdown_button(canvas, (obs_x1, row_y1, obs_x2, row_y2), cur_obs_label,
                             is_open=is_obs_open, mouse_pos=app.mouse_pos, prefix="实测识别: ")
        app.dropdown_boxes["OBS_VIEW_DROPDOWN"] = {
            "rect": (obs_x1, row_y1, obs_x2, row_y2),
            "options": OBS_VIEW_OPTIONS,
            "active_key": app.obs_view_mode
        }
        app.gui_buttons.append(("TOGGLE_OBS_VIEW_DROPDOWN", (obs_x1, row_y1, obs_x2, row_y2), "OBS_VIEW_DROPDOWN"))
        cursor_x = obs_x2 + 8

        # 3. 绘制 ROI 物件按钮
        draw_roi_active = getattr(app, "draw_roi_mode", False)
        draw_roi_txt = "✓ 绘制ROI物件" if draw_roi_active else "绘制ROI物件"
        draw_roi_accent = (0, 215, 90) if draw_roi_active else None
        draw_roi_w = 110
        draw_roi_x1, draw_roi_x2 = cursor_x, cursor_x + draw_roi_w
        draw_dashboard_button(canvas, (draw_roi_x1, row_y1, draw_roi_x2, row_y2), draw_roi_txt,
                              mouse_pos=app.mouse_pos, accent=draw_roi_accent)
        app.gui_buttons.append(("DRAW_ROI_OBJECT", (draw_roi_x1, row_y1, draw_roi_x2, row_y2), "DRAW_ROI_OBJECT"))
        cursor_x = draw_roi_x2 + 8

        # 4. 绘制 XY 平面按钮
        xy_on = app.data_mgr.show_xy_plane_on
        xy_lbl = "√ XY平面" if xy_on else "绘制XY平面"
        xy_accent = (0, 255, 180) if xy_on else None
        xy_w = 88
        xy_x1, xy_x2 = cursor_x, cursor_x + xy_w
        draw_dashboard_button(canvas, (xy_x1, row_y1, xy_x2, row_y2), xy_lbl,
                              mouse_pos=app.mouse_pos, accent=xy_accent)
        app.gui_buttons.append(("TOGGLE_DRAW_XY_PLANE", (xy_x1, row_y1, xy_x2, row_y2), "TOGGLE_DRAW_XY_PLANE"))
        cursor_x = xy_x2 + 8

        # 5. Z 轴特殊点下拉框
        z_w = 120
        z_x1, z_x2 = cursor_x, cursor_x + z_w
        cur_z_lbl = app.get_current_plane_z_label() if hasattr(app, "get_current_plane_z_label") else "Z轴特殊点"
        is_z_open = (app.active_dropdown == "PLANE_Z_DROPDOWN")
        draw_dropdown_button(canvas, (z_x1, row_y1, z_x2, row_y2), cur_z_lbl,
                             is_open=is_z_open, mouse_pos=app.mouse_pos,
                             theme_color=(0, 220, 255) if xy_on else (140, 160, 180))
        plane_opts = [(str(val) if val is not None else "NONE", lbl)
                      for val, lbl in app.get_plane_z_options()] if hasattr(app, "get_plane_z_options") else []
        active_z_key = str(app.plane_z) if (xy_on and hasattr(app, "plane_z")) else "NONE"
        app.dropdown_boxes["PLANE_Z_DROPDOWN"] = {
            "rect": (z_x1, row_y1, z_x2, row_y2),
            "options": plane_opts,
            "active_key": active_z_key
        }
        app.gui_buttons.append(("TOGGLE_PLANE_Z_DROPDOWN", (z_x1, row_y1, z_x2, row_y2), "PLANE_Z_DROPDOWN"))

    def overlay_visual_elements(
        self,
        app: Any,
        disp_frame: np.ndarray,
        observations: List[Dict[str, Any]],
        is_frame_excluded: bool,
        meta: Optional[Dict[str, Any]] = None,
        panel_rect: Optional[Tuple[int, int, int, int]] = None
    ):
        """依据 ba_view_mode 与 obs_view_mode 双独立维度解耦渲染，剔除标靶显著打红叉"""
        ba_mode = app.ba_view_mode
        obs_mode = app.obs_view_mode

        # 画布鼠标坐标 -> 原始帧坐标 (悬停展开标靶详情, 高密度场景防遮挡)
        mouse_frame = None
        if panel_rect is not None and getattr(app, "mouse_pos", None):
            try:
                fh, fw = disp_frame.shape[:2]
                img_rect = app.viewport.compute_image_rect(panel_rect, fw, fh)
                ix1, iy1, ix2, iy2 = img_rect[0], img_rect[1], img_rect[2], img_rect[3]
                if ix2 > ix1 and iy2 > iy1:
                    mfx = (app.mouse_pos[0] - ix1) / float(ix2 - ix1) * fw
                    mfy = (app.mouse_pos[1] - iy1) / float(iy2 - iy1) * fh
                    if 0 <= mfx < fw and 0 <= mfy < fh:
                        mouse_frame = (mfx, mfy)
            except Exception:
                mouse_frame = None

        obj_pts = []
        img_pts = []
        valid_obs = []

        # 1. 第一阶段：绘制实测观测标注（有效标靶记录用于 PnP，剔除标靶绘制红叉审核标记）
        obs_map = {}
        for obs in observations:
            tid = obs["tag_id"]
            obs_map[tid] = obs
            pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
            keep = obs.get("keep", True) and not is_frame_excluded

            # 剔除状态下在标靶实测位置绘制鲜红显著的大叉号 (打叉审核模式)
            if not keep:
                cv2.line(disp_frame, (pts[0][0], pts[0][1]), (pts[2][0], pts[2][1]), (0, 0, 235), 3, cv2.LINE_AA)
                cv2.line(disp_frame, (pts[1][0], pts[1][1]), (pts[3][0], pts[3][1]), (0, 0, 235), 3, cv2.LINE_AA)
                cv2.polylines(disp_frame, [pts], isClosed=True, color=(40, 40, 180), thickness=2, lineType=cv2.LINE_AA)
                cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                put_text(disp_frame, f"Tag #{tid} [EXCL]", (cx - 42, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 240), 2, cv2.LINE_AA)
            else:
                # 收集参与三维相机位姿解算的已知有效标靶
                w_c = app.data_mgr.get_tag_world_corners(tid)
                if w_c is not None:
                    obj_pts.append(w_c)
                    img_pts.append(np.array(obs["corners"], dtype=np.float64))
                    valid_obs.append(obs)

        rendered_tids = set()

        # 2. 第二阶段：解算当前相机位姿 (PnP)
        rvec = None
        tvec = None
        success = False
        if len(obj_pts) >= 1:
            obj_flat = np.concatenate(obj_pts, axis=0)
            img_flat = np.concatenate(img_pts, axis=0)
            rvec, tvec, success = app.pnp_solver.solve_pnp(obj_flat, img_flat)

        # 若当前无足够有效点 (如标靶全被剔除)，尝试复用 meta 缓存的相机外参
        if not success and meta is not None:
            rvec_c = meta.get("rvec")
            tvec_c = meta.get("tvec")
            if rvec_c is not None and tvec_c is not None:
                rvec = np.array(rvec_c, dtype=np.float64)
                tvec = np.array(tvec_c, dtype=np.float64)
                success = True

        # 3. 第三阶段：3D 棱柱与残差立体渲染 (无论标靶是否被剔除，只要开启 ba_mode=='3d'，绿色 BA 理论棱柱全量呈现！)
        if success:
            R_c_w, _ = cv2.Rodrigues(rvec)
            T_c_w = np.eye(4, dtype=np.float64)
            T_c_w[:3, :3] = R_c_w
            T_c_w[:3, 3] = tvec.flatten()

            need_3d = (ba_mode == "3d" or obs_mode == "3d")
            if need_3d:
                # 收集候选标靶：
                # (a) 当前帧观测到的所有标靶 (不论保留还是已剔除)
                # (b) 如果开启了 ba_mode == "3d"，还包含地图中已建图的其余已知标靶
                candidate_tids = list(obs_map.keys())
                if ba_mode == "3d":
                    tags_dict = (app.data_mgr.tags_map_data or {}).get("tags", {})
                    for m_tid in tags_dict.keys():
                        if m_tid not in obs_map:
                            candidate_tids.append(m_tid)

                h_f, w_f = disp_frame.shape[:2]
                # FR-9.6 世界系位姿元数据 (平差锚定后每枚标靶的 XYZ 与 RPY)
                tags_meta = (app.data_mgr.tags_map_data or {}).get("tags", {})
                for tid in candidate_tids:
                    T_w_t = app.data_mgr.get_tag_transform(tid)
                    if T_w_t is None:
                        continue

                    # 计算标靶在当前相机系下的理论位姿
                    T_c_t = T_c_w @ T_w_t
                    t_tag_center = T_c_t[:3, 3]

                    # 标靶必须位于相机正前方
                    if t_tag_center[2] <= 50.0:
                        continue

                    # 理论 BA 位姿 (翡翠绿)
                    r_tag, t_tag = None, None
                    if ba_mode == "3d":
                        r_tag, _ = cv2.Rodrigues(T_c_t[:3, :3])
                        t_tag = t_tag_center.reshape((3, 1))

                    obs = obs_map.get(tid)
                    obs_r, obs_t = None, None
                    c_arr = None
                    succ_single = False
                    is_kept = False

                    if obs is not None:
                        c_arr = np.array(obs["corners"], dtype=np.float64).reshape((4, 2))
                        is_kept = obs.get("keep", True) and not is_frame_excluded
                        # 仅在有效保留且 obs_mode=='3d' 下才计算并显示实测蓝色棱柱
                        if is_kept and obs_mode == "3d":
                            # 传入地图理论法向, 消除 IPPE 平面二义性 180° 翻转
                            succ_single, obs_r, obs_t = app.pnp_solver.solve_single_tag_pnp(
                                c_arr, expected_z_cam=T_c_t[:3, :3][:, 2])

                    # 如果既不画理论绿色棱柱，也不画实测蓝色棱柱，跳过
                    if r_tag is None and obs_r is None:
                        continue

                    # 若当前标靶未检出 (纯理论)，检查理论中心是否在像面可视范围内
                    if obs is None and r_tag is not None:
                        p_center, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), r_tag, t_tag, app.pnp_solver.camera_matrix, app.pnp_solver.dist_coeffs)
                        cu, cv = p_center.reshape(-1)
                        if not (-80 <= cu <= w_f + 80 and -80 <= cv <= h_f + 80):
                            continue

                    err_mm = 0.0
                    if t_tag is not None and succ_single and obs_t is not None:
                        err_mm = float(np.linalg.norm(t_tag - obs_t))
                    err_px = (meta or {}).get("tag_errors", {}).get(tid, 0.2)

                    # 悬停命中检测 (帧坐标, 48px 半径): 实测以观测角点中心, 纯理论以投影中心
                    tag_center_f = None
                    if c_arr is not None:
                        tag_center_f = (float(np.mean(c_arr[:, 0])), float(np.mean(c_arr[:, 1])))
                    elif r_tag is not None:
                        p_c, _ = cv2.projectPoints(np.array([[0.0, 0.0, 0.0]]), r_tag, t_tag,
                                                   app.pnp_solver.camera_matrix, app.pnp_solver.dist_coeffs)
                        tag_center_f = (float(p_c.reshape(-1)[0]), float(p_c.reshape(-1)[1]))
                    is_hovered = (mouse_frame is not None and tag_center_f is not None
                                  and (mouse_frame[0] - tag_center_f[0]) ** 2 + (mouse_frame[1] - tag_center_f[1]) ** 2 < 48.0 ** 2)

                    # 状态提示文案
                    status_hint = None
                    if obs is not None and not is_kept:
                        status_hint = "[BA理论:实测已剔除]"
                    elif obs is None:
                        status_hint = "[BA理论:未检出/遮挡]"

                    app.visualizer.render_tag_dual_prisms(
                        img=disp_frame,
                        ba_rvec=r_tag if ba_mode == "3d" else None,
                        ba_tvec=t_tag if ba_mode == "3d" else None,
                        obs_rvec=obs_r if (obs_mode == "3d" and succ_single) else None,
                        obs_tvec=obs_t if (obs_mode == "3d" and succ_single) else None,
                        tag_id=tid,
                        err_px=err_px,
                        err_mm=err_mm,
                        observed_corners=c_arr,
                        tag_status_hint=status_hint,
                        world_position_mm=(tags_meta.get(tid) or {}).get("position_mm"),
                        world_rpy_deg=(tags_meta.get(tid) or {}).get("rpy_deg"),
                        ba_center_xyz=(t_tag_center.tolist() if r_tag is not None else None),
                        obs_center_xyz=(obs_t.flatten().tolist() if obs_t is not None else None),
                        hovered=is_hovered
                    )
                    rendered_tids.add(tid)

            # 4. 2D 理论重投影框与残差矢量
            if ba_mode == "2d" and len(valid_obs) > 0:
                proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, app.pnp_solver.camera_matrix, app.pnp_solver.dist_coeffs)
                proj_flat = proj_pts.reshape((-1, 2))
                for i in range(len(valid_obs)):
                    p4 = proj_flat[i * 4:(i + 1) * 4].astype(np.int32)
                    cv2.polylines(disp_frame, [p4], isClosed=True, color=(0, 210, 255), thickness=1, lineType=cv2.LINE_AA)

                if obs_mode == "2d" and hasattr(app.visualizer, "draw_reprojection_vectors"):
                    app.visualizer.draw_reprojection_vectors(disp_frame, img_flat, proj_flat, scale_factor=40.0)

        # 4. 保底渲染：对所有提取到但未被 3D 棱柱覆盖的有效保留标靶，保底绘制 2D 实测角点多边形与编号标签
        if obs_mode != "off":
            for obs in observations:
                if not obs.get("keep", True) or is_frame_excluded:
                    continue
                tid = obs["tag_id"]
                if obs_mode == "2d" or tid not in rendered_tids:
                    pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
                    cv2.polylines(disp_frame, [pts], isClosed=True, color=(0, 230, 80), thickness=2, lineType=cv2.LINE_AA)
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    in_map = (app.data_mgr.get_tag_world_corners(tid) is not None)
                    tag_lbl = f"Tag #{tid}" if in_map else f"Tag #{tid} [未入图]"
                    put_text(disp_frame, tag_lbl, (cx - 38, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 230, 80), 2, cv2.LINE_AA)
                    rendered_tids.add(tid)

        # 5. 若处于病因切片诊断模式，叠加视野内预测但实测漏检的标靶框 (橙黄色矩形与 Tag 标注)
        if getattr(app, "show_frame_diagnostics", False):
            diag = getattr(app.data_mgr, "current_diagnostics", {})
            missing = diag.get("missing_projected_tags", []) or diag.get("missing_theoretical_tags", [])
            for m in missing:
                tid = m.get("tag_id")
                c_pts = m.get("proj_corners") or m.get("predicted_corners")
                if c_pts is not None and len(c_pts) == 4:
                    pts_i = np.array(c_pts, dtype=np.int32)
                    cv2.polylines(disp_frame, [pts_i], isClosed=True, color=(0, 140, 255), thickness=2, lineType=cv2.LINE_AA)
                    mcx, mcy = int(np.mean(pts_i[:, 0])), int(np.mean(pts_i[:, 1]))
                    put_text(disp_frame, f"? Tag #{tid} [漏检预测]", (mcx - 45, mcy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 2, cv2.LINE_AA)

        # 6. 世界 XY 平面透视网格与 Z 轴特殊点辅助线叠加 (移植自在线跟踪)
        if app.data_mgr.show_xy_plane_on and success and rvec is not None and tvec is not None:
            self.draw_xy_plane_overlay(app, disp_frame, rvec, tvec)

        # 7. 3D ROI 空间物件投影绘制 (黄色半透明长方体覆盖)
        if getattr(app, "draw_roi_mode", False) and success and rvec is not None and tvec is not None:
            self._draw_roi_cuboids_overlay(app, disp_frame, rvec, tvec)

    def _draw_roi_cuboids_overlay(self, app: Any, disp_frame: np.ndarray,
                                  rvec: np.ndarray, tvec: np.ndarray):
        """遍历当前工位所有 ROI, 用当前帧 BA 外参投影 8 角点到图像, 绘制黄色半透明长方体。
        前置条件: app.roi_mgr / app.coord_mgr / app.pnp_solver.camera_matrix 已就绪。
        失败 (无 ROI / 无 BA 位姿 / ROI 未解算) 时静默跳过, 不弹 toast。"""
        roi_mgr = getattr(app, "roi_mgr", None)
        coord_mgr = getattr(app, "coord_mgr", None)
        if roi_mgr is None or coord_mgr is None:
            return
        pnp_solver = getattr(app, "pnp_solver", None)
        if pnp_solver is None or getattr(pnp_solver, "camera_matrix", None) is None:
            return

        K = pnp_solver.camera_matrix
        dist = getattr(pnp_solver, "dist_coeffs", None)

        # 由 rvec/tvec 组合出 4x4 T_world_from_cam 的逆, 用于筛选相机背后的角点
        R_c_w, _ = cv2.Rodrigues(rvec)
        T_world_from_cam = np.eye(4, dtype=np.float64)
        T_world_from_cam[:3, :3] = R_c_w
        T_world_from_cam[:3, 3] = tvec.flatten()
        T_cam_from_world = np.linalg.inv(T_world_from_cam)

        roi_list = roi_mgr.list_rois()
        if not roi_list:
            return

        h, w = disp_frame.shape[:2]
        # 8 角点 -> 12 条边 (长方体拓扑)
        _EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
                  (4, 5), (5, 6), (6, 7), (7, 4),
                  (0, 4), (1, 5), (2, 6), (3, 7)]

        for roi in roi_list:
            if not getattr(roi, "enabled", True):
                continue
            obb = roi_mgr.get_roi_world_obb(roi.roi_id, coord_mgr)
            if obb is None or not obb.get("is_resolved", False):
                continue

            world_corners = obb["corners_8x3"].astype(np.float64)
            # 1) 相机后方点过滤: 计算相机系深度, 仅保留 z>0 的角点
            homo = np.hstack([world_corners, np.ones((8, 1), dtype=np.float64)])
            cam_pts = (T_cam_from_world @ homo.T).T[:, :3]
            front_mask = cam_pts[:, 2] > 1e-3
            if not np.any(front_mask):
                continue

            # 2) 8 角点统一投影到图像像素坐标
            proj_pts, _ = cv2.projectPoints(
                world_corners.reshape(-1, 1, 3), rvec, tvec, K, dist
            )
            img_pts = proj_pts.reshape(8, 2)
            pts_int = np.round(img_pts).astype(np.int32)

            # 3) 仅保留前置角点: 用 front_mask 过滤后的多边形填充 + 可见边描线
            visible_idx = np.where(front_mask)[0]
            if len(visible_idx) >= 3:
                # 用 cv2.convexHull 求可视角点的凸包 (近似可视面)
                hull_pts = pts_int[visible_idx]
                try:
                    hull = cv2.convexHull(hull_pts)
                    if hull is not None and len(hull) >= 3:
                        overlay = disp_frame.copy()
                        cv2.fillPoly(overlay, [hull], color=(0, 220, 255), lineType=cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.30, disp_frame, 0.70, 0, disp_frame)
                except cv2.error:
                    pass

            # 4) 12 条边全部描线 (含被遮挡的背面边, 便于辨识整体形状)
            for i, j in _EDGES:
                pt1 = tuple(int(v) for v in pts_int[i])
                pt2 = tuple(int(v) for v in pts_int[j])
                # 裁剪到画面外视为不可见
                if not (-50 <= pt1[0] <= w + 50 and -50 <= pt1[1] <= h + 50):
                    if not (-50 <= pt2[0] <= w + 50 and -50 <= pt2[1] <= h + 50):
                        continue
                cv2.line(disp_frame, pt1, pt2, color=(0, 220, 255), thickness=2, lineType=cv2.LINE_AA)

            # 5) 中心十字 + 名称 (黄色)
            center_world = obb["center_world"].astype(np.float64)
            center_cam = (T_cam_from_world @ np.append(center_world, 1.0))[:3]
            if center_cam[2] > 1e-3:
                center_proj, _ = cv2.projectPoints(
                    center_world.reshape(-1, 1, 3), rvec, tvec, K, dist
                )
                cx, cy = center_proj.reshape(2)
                cx_i, cy_i = int(round(cx)), int(round(cy))
                if 0 <= cx_i <= w and 0 <= cy_i <= h:
                    cv2.drawMarker(disp_frame, (cx_i, cy_i), color=(0, 255, 255),
                                   markerType=cv2.MARKER_CROSS, markerSize=12, thickness=1, line_type=cv2.LINE_AA)
                    label = f"{roi.name} ({roi.roi_id})"
                    put_text(disp_frame, label, (cx_i + 8, cy_i - 8),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2, cv2.LINE_AA)

    def draw_xy_plane_overlay(self, app: Any, canvas: np.ndarray, rvec: np.ndarray, tvec: np.ndarray):
        """世界 XY 平面透视网格叠加 (移植自在线跟踪):
        支持两组垂直平行线网格 + 三轴加粗高亮 (X红 / Y绿 / Z蓝) + 向上箭头 + 原点标记 + 特殊标靶等高红线
        """
        if not app.data_mgr.show_xy_plane_on:
            return
        if rvec is None or tvec is None:
            return

        R, _ = cv2.Rodrigues(rvec)
        t_flat = np.asarray(tvec, dtype=np.float64).reshape(3)
        K = app.pnp_solver.camera_matrix
        h_f, w_f = canvas.shape[:2]
        ext = getattr(app, "PLANE_EXTENT_MM", 600)
        step = getattr(app, "PLANE_STEP_MM", 100)
        z0 = float(getattr(app, "plane_z", 0.0))
        plane_z_max = getattr(app, "PLANE_Z_MM", 600)

        COL_GRAY = (90, 95, 105)
        COL_RED = (60, 60, 245)
        COL_GREEN = (50, 220, 100)
        COL_BLUE = (245, 150, 50)  # BGR 格式高亮科技蓝
        COL_WHITE = (220, 220, 220)

        def _project(p_w):
            p_cam = R @ np.asarray(p_w, dtype=np.float64).reshape(3) + t_flat
            if p_cam[2] <= 1e-6:
                return None
            uv = K @ p_cam
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            return (u, v) if (0 <= u < w_f and 0 <= v < h_f) else None

        def _seg(p0, p1, color, thick):
            """长线段沿线采样投影连线 (自动处理出画与近裁剪)"""
            prev = None
            for k in range(25):
                s = k / 24.0
                p = (p0[0] + (p1[0] - p0[0]) * s,
                     p0[1] + (p1[1] - p0[1]) * s,
                     p0[2] + (p1[2] - p0[2]) * s)
                uv = _project(p)
                if uv is not None and prev is not None:
                    cv2.line(canvas, prev, uv, color, thick, cv2.LINE_AA)
                prev = uv

        # 1. 平行线网格 (绘制高度 z0): 平行于 X 轴与平行于 Y 轴两组
        for i in range(-ext, ext + 1, step):
            _seg((-ext, i, z0), (ext, i, z0), COL_GRAY, 1)
            _seg((i, -ext, z0), (i, ext, z0), COL_GRAY, 1)

        # 2. 坐标轴加粗高亮: X 红 / Y 绿 (随平面高度 z0) / Z 蓝 (0→600mm)
        _seg((-ext, 0, z0), (ext, 0, z0), COL_RED, 3)
        _seg((0, -ext, z0), (0, ext, z0), COL_GREEN, 3)
        _seg((0, 0, 0), (0, 0, plane_z_max), COL_BLUE, 4)

        # 3. Tag 等高辅助红线: 当平面高度与某已知标靶中心 Z 重合且该标靶不在原点时,
        #    平移一条红色 X 轴穿过该标靶 (如 Z=196 平面过 Tag 1); Tag 0 在原点, 主 X 轴已穿过
        tags_dict = (app.data_mgr.tags_map_data or {}).get("tags", {})

        for tid, t_info in (tags_dict or {}).items():
            mat = t_info.get("transform_matrix")
            if mat and len(mat) == 4:
                c_x, c_y, c_z = float(mat[0][3]), float(mat[1][3]), float(mat[2][3])
                if abs(c_z - z0) < 2.0 and (abs(c_x) > 1.0 or abs(c_y) > 1.0):
                    _seg((c_x - ext, c_y, z0), (c_x + ext, c_y, z0), COL_RED, 2)
                    uv = _project((c_x + ext, c_y, z0))
                    if uv is not None:
                        put_text(canvas, f"X (Tag {tid})", (uv[0] + 6, uv[1] - 8),
                                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_RED, 2, cv2.LINE_AA)

        # 4. Z 轴高度刻度 (每 100mm) + 顶端箭头
        for hz in range(100, plane_z_max, 100):
            tp = _project((0, 0, hz))
            if tp is not None:
                cv2.line(canvas, (tp[0] - 5, tp[1]), (tp[0] + 5, tp[1]), COL_BLUE, 2, cv2.LINE_AA)
                put_text(canvas, str(hz), (tp[0] + 8, tp[1] - 6),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_BLUE, 1, cv2.LINE_AA)

        p_top = _project((0, 0, plane_z_max))
        p_base = _project((0, 0, 0))
        if p_top is not None and p_base is not None:
            d = np.array(p_top, dtype=np.float64) - np.array(p_base, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n > 24:
                d /= n
                perp = np.array([-d[1], d[0]])
                tip = np.array(p_top, dtype=np.float64)
                wing = 14.0 * d
                arrow = np.array([tip, tip - wing + 6.0 * perp, tip - wing - 6.0 * perp], dtype=np.int32)
                cv2.fillPoly(canvas, [arrow], COL_BLUE)

        for label, p, col in (("X", (ext + 50, 0, z0), COL_RED),
                              ("Y", (0, ext + 50, z0), COL_GREEN),
                              ("Z", (0, 0, plane_z_max + 50), COL_BLUE),
                              ("0", (0, 0, 0), COL_WHITE)):
            uv = _project(p)
            if uv is not None:
                put_text(canvas, label, (uv[0] + 6, uv[1] - 8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)
