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

        # 3. 绘制 ROI 物件 — dropdown 触发器 (点击展开 ROI CheckList 浮层)
        roi_mgr = getattr(app, "roi_mgr", None)
        roi_count = len(roi_mgr.list_rois()) if roi_mgr else 0
        roi_panel_open = (app.active_dropdown == "ROI_LIST_PANEL")
        roi_enabled_count = sum(1 for r in roi_mgr.list_rois() if r.enabled) if roi_mgr else 0
        if roi_count > 0:
            roi_btn_lbl = f"ROI物件 {roi_enabled_count}/{roi_count}"
        else:
            roi_btn_lbl = "ROI物件"
        draw_roi_w = 110
        draw_roi_x1, draw_roi_x2 = cursor_x, cursor_x + draw_roi_w
        draw_dropdown_button(canvas, (draw_roi_x1, row_y1, draw_roi_x2, row_y2), roi_btn_lbl,
                             is_open=roi_panel_open, mouse_pos=app.mouse_pos,
                             theme_color=(0, 215, 90) if roi_panel_open else (140, 160, 180))
        app.dropdown_boxes["ROI_LIST_PANEL"] = {
            "rect": (draw_roi_x1, row_y1, draw_roi_x2, row_y2),
            "options": [],
            "active_key": "",
        }
        app.gui_buttons.append(("TOGGLE_ROI_LIST_PANEL", (draw_roi_x1, row_y1, draw_roi_x2, row_y2), "ROI_LIST_PANEL"))
        cursor_x = draw_roi_x2 + 8

        # 4. 绘制坐标系 —— dropdown 触发器 (点击展开坐标系基准切换与 CheckList 浮层)
        coord_mgr = getattr(app, "coord_mgr", None)
        visibility = getattr(app, "coord_frame_visibility", {})
        coord_panel_open = (app.active_dropdown == "COORD_FRAME_PANEL")
        ref_fid = getattr(app, "active_reference_frame_id", "world")
        if coord_mgr is not None:
            all_frames = coord_mgr.list_frames()
            vis_count = sum(1 for f in all_frames if visibility.get(f.frame_id, False))
            total_count = len(all_frames)
            if ref_fid == "world":
                coord_btn_lbl = f"基准:世界 ({vis_count}/{total_count})"
            else:
                f_obj = coord_mgr.get_frame(ref_fid)
                f_name = f_obj.name if f_obj and f_obj.name else ref_fid
                coord_btn_lbl = f"基准:{f_name}★"
        else:
            coord_btn_lbl = "坐标系"
        coord_w = 135
        coord_x1, coord_x2 = cursor_x, cursor_x + coord_w
        btn_theme = (0, 240, 220) if ref_fid != "world" else ((0, 220, 255) if coord_panel_open else (140, 160, 180))
        draw_dropdown_button(canvas, (coord_x1, row_y1, coord_x2, row_y2), coord_btn_lbl,
                             is_open=coord_panel_open, mouse_pos=app.mouse_pos,
                             theme_color=btn_theme)
        app.dropdown_boxes["COORD_FRAME_PANEL"] = {
            "rect": (coord_x1, row_y1, coord_x2, row_y2),
            "options": [],
            "active_key": "",
        }
        app.gui_buttons.append(("TOGGLE_COORD_FRAME_PANEL", (coord_x1, row_y1, coord_x2, row_y2), "COORD_FRAME_PANEL"))


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

                    # 若当前设置了非世界基准坐标系，计算局部相对位置与名义偏差
                    ref_fid = getattr(app, "active_reference_frame_id", "world")
                    coord_mgr = getattr(app, "coord_mgr", None)
                    ref_pos = None
                    nom_pos = None
                    nom_err = None
                    if ref_fid != "world" and coord_mgr is not None:
                        T_ref_w, is_res = coord_mgr.get_transform("world", ref_fid)
                        if is_res:
                            pw = (tags_meta.get(tid) or {}).get("position_mm")
                            if pw is not None and len(pw) >= 3:
                                ref_pos = (T_ref_w @ np.array([pw[0], pw[1], pw[2], 1.0], dtype=np.float64))[:3].tolist()
                            ref_frame = coord_mgr.get_frame(ref_fid)
                            if ref_frame and getattr(ref_frame, "calibration_spec", None):
                                ref_tags = ref_frame.calibration_spec.get("reference_tags", {})
                                if tid in ref_tags or str(tid) in ref_tags:
                                    nom_pos = ref_tags.get(tid) or ref_tags.get(str(tid))
                                    if nom_pos is not None and ref_pos is not None:
                                        nom_err = float(np.linalg.norm(np.array(ref_pos[:3]) - np.array(nom_pos[:3])))

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
                        hovered=is_hovered,
                        ref_frame_id=ref_fid,
                        ref_position_mm=ref_pos,
                        nominal_local_xyz=nom_pos,
                        nominal_error_mm=nom_err
                    )
                    rendered_tids.add(tid)

            # 4. 2D 理论重投影框与残差矢量 (理论统一翡翠绿)
            if ba_mode == "2d" and len(valid_obs) > 0:
                proj_pts, _ = cv2.projectPoints(obj_flat, rvec, tvec, app.pnp_solver.camera_matrix, app.pnp_solver.dist_coeffs)
                proj_flat = proj_pts.reshape((-1, 2))
                for i in range(len(valid_obs)):
                    p4 = proj_flat[i * 4:(i + 1) * 4].astype(np.int32)
                    cv2.polylines(disp_frame, [p4], isClosed=True, color=(0, 230, 80), thickness=2, lineType=cv2.LINE_AA)

                if obs_mode == "2d" and hasattr(app.visualizer, "draw_reprojection_vectors"):
                    app.visualizer.draw_reprojection_vectors(disp_frame, img_flat, proj_flat, scale_factor=40.0)

        # 4. 保底渲染：对所有提取到但未被 3D 棱柱覆盖的有效保留标靶，保底绘制 2D 实测角点多边形与编号标签 (实测统一科技天蓝)
        if obs_mode != "off":
            for obs in observations:
                if not obs.get("keep", True) or is_frame_excluded:
                    continue
                tid = obs["tag_id"]
                if obs_mode == "2d" or tid not in rendered_tids:
                    pts = np.array(obs["corners"], dtype=np.int32).reshape((-1, 2))
                    cv2.polylines(disp_frame, [pts], isClosed=True, color=(245, 180, 30), thickness=2, lineType=cv2.LINE_AA)
                    cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))
                    in_map = (app.data_mgr.get_tag_world_corners(tid) is not None)
                    tag_lbl = f"Tag #{tid}" if in_map else f"Tag #{tid} [未入图]"
                    put_text(disp_frame, tag_lbl, (cx - 38, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 180, 30), 2, cv2.LINE_AA)
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

        # 6. 坐标系可见性绘制：XY 平面（正象限）+ X/Y/Z 三轴正向箭头
        coord_mgr_local = getattr(app, "coord_mgr", None)
        if coord_mgr_local is not None and success and rvec is not None and tvec is not None:
            self._draw_visible_coord_frames(app, disp_frame, rvec, tvec)

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

        coord_mgr = getattr(app, "coord_mgr", None)
        ref_fid = getattr(app, "active_reference_frame_id", "world")
        T_w_ref = np.eye(4, dtype=np.float64)
        if ref_fid != "world" and coord_mgr is not None:
            T_w_f, is_res = coord_mgr.get_frame_to_world(ref_fid)
            if is_res:
                T_w_ref = T_w_f

        def _to_world(p_loc):
            if ref_fid == "world":
                return p_loc
            homo = np.array([p_loc[0], p_loc[1], p_loc[2], 1.0], dtype=np.float64)
            return (T_w_ref @ homo)[:3]

        def _project(p_w):
            p_cam = R @ np.asarray(p_w, dtype=np.float64).reshape(3) + t_flat
            if p_cam[2] <= 1e-6:
                return None
            uv = K @ p_cam
            u, v = int(round(uv[0] / uv[2])), int(round(uv[1] / uv[2]))
            return (u, v) if (0 <= u < w_f and 0 <= v < h_f) else None

        def _seg(p0, p1, color, thick):
            """长线段沿线采样投影连线 (自动处理出画与近裁剪)"""
            p0_w = _to_world(p0)
            p1_w = _to_world(p1)
            prev = None
            for k in range(25):
                s = k / 24.0
                p = (p0_w[0] + (p1_w[0] - p0_w[0]) * s,
                     p0_w[1] + (p1_w[1] - p0_w[1]) * s,
                     p0_w[2] + (p1_w[2] - p0_w[2]) * s)
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
            tp = _project(_to_world((0, 0, hz)))
            if tp is not None:
                cv2.line(canvas, (tp[0] - 5, tp[1]), (tp[0] + 5, tp[1]), COL_BLUE, 2, cv2.LINE_AA)
                put_text(canvas, str(hz), (tp[0] + 8, tp[1] - 6),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.40, COL_BLUE, 1, cv2.LINE_AA)

        p_top = _project(_to_world((0, 0, plane_z_max)))
        p_base = _project(_to_world((0, 0, 0)))
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

        # 5. 坐标轴端点标签 (X/Y/Z/0)
        origin_label = f"0 ({ref_fid})" if ref_fid != "world" else "0"
        for label, p, col in (("X", (ext + 50, 0, z0), COL_RED),
                              ("Y", (0, ext + 50, z0), COL_GREEN),
                              ("Z", (0, 0, plane_z_max + 50), COL_BLUE),
                              (origin_label, (0, 0, 0), COL_WHITE)):
            uv = _project(_to_world(p))
            if uv is not None:
                put_text(canvas, label, (uv[0] + 6, uv[1] - 8),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)

    def _draw_coord_frames_overlay(
        self,
        app: Any,
        disp_frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_len_mm: float = 120.0,
    ):
        """遍历 coord_mgr 所有已解算坐标系，绘制 X+/Y+/Z+ 三轴正方向箭头。

        颜色约定 (BGR):
          X+ 轴 → 蓝红色  (0,  60, 220)
          Y+ 轴 → 青绿色  (40, 200,  40)
          Z+ 轴 → 科技蓝  (220, 140, 30)
        箭头长度 = axis_len_mm (世界单位 mm)，自动按 ROI 平均尺寸自适应。
        仅绘制正方向（原点 → +轴端），背后的轴（相机深度 ≤ 0）静默跳过。
        """
        coord_mgr = getattr(app, "coord_mgr", None)
        if coord_mgr is None:
            return
        pnp_solver = getattr(app, "pnp_solver", None)
        if pnp_solver is None:
            return
        K = getattr(pnp_solver, "camera_matrix", None)
        if K is None:
            return
        dist = getattr(pnp_solver, "dist_coeffs", None)

        h_f, w_f = disp_frame.shape[:2]
        R_c_w, _ = cv2.Rodrigues(rvec)
        t_flat = np.asarray(tvec, dtype=np.float64).flatten()

        # 三轴颜色 (BGR)
        AXIS_COLORS = [
            ((0, 60, 220),   "X"),   # X+ 红
            ((40, 200, 40),  "Y"),   # Y+ 绿
            ((220, 140, 30), "Z"),   # Z+ 蓝
        ]
        # 三轴世界方向单位向量
        AXIS_DIRS = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        def _proj_world(pt_w):
            """世界坐标点 → 图像像素，返回 None 表示相机背面或出界"""
            p_cam = R_c_w @ np.asarray(pt_w, dtype=np.float64) + t_flat
            if p_cam[2] <= 1e-3:
                return None
            pts, _ = cv2.projectPoints(
                np.asarray(pt_w, dtype=np.float64).reshape(1, 1, 3),
                rvec, tvec, K, dist
            )
            u, v = int(round(pts[0, 0, 0])), int(round(pts[0, 0, 1]))
            # 允许稍微出界（让长箭头末端也能绘出）
            if -60 <= u <= w_f + 60 and -60 <= v <= h_f + 60:
                return (u, v)
            return None

        def _draw_arrow(p0_uv, p1_uv, color, label: str):
            """在图像上绘制从 p0→p1 的箭头 + 轴标签"""
            if p0_uv is None or p1_uv is None:
                return
            # 主干
            cv2.line(disp_frame, p0_uv, p1_uv, color, 2, cv2.LINE_AA)
            # 箭头头部 (fillPoly 三角形)
            d = np.array(p1_uv, dtype=np.float64) - np.array(p0_uv, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n < 8:
                return
            d /= n
            perp = np.array([-d[1], d[0]])
            tip = np.array(p1_uv, dtype=np.float64)
            wing_len = min(10.0, n * 0.25)
            arrow_head = np.array([
                tip,
                tip - wing_len * d + (wing_len * 0.45) * perp,
                tip - wing_len * d - (wing_len * 0.45) * perp,
            ], dtype=np.int32)
            cv2.fillPoly(disp_frame, [arrow_head], color)
            # 轴标签
            lx = int(p1_uv[0] + d[0] * 6 + 4)
            ly = int(p1_uv[1] + d[1] * 6 + 4)
            put_text(disp_frame, label, (lx, ly),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)

        frames = coord_mgr.list_frames()
        for frame in frames:
            # 只绘制已就绪的坐标系
            T_w_f, is_resolved = coord_mgr.get_frame_to_world(frame.frame_id)
            if not is_resolved:
                continue

            # 坐标系原点（世界坐标）
            origin_w = T_w_f[:3, 3]
            # 坐标系三轴正方向（世界坐标中的方向向量）
            R_w_f = T_w_f[:3, :3]   # 坐标系三列 = X/Y/Z 在世界系的方向

            origin_uv = _proj_world(origin_w)

            for (color, label), axis_dir_local in zip(AXIS_COLORS, AXIS_DIRS):
                # 轴终点：从原点沿该轴正方向延伸 axis_len_mm
                tip_w = origin_w + R_w_f @ (axis_dir_local * axis_len_mm)
                tip_uv = _proj_world(tip_w)
                _draw_arrow(origin_uv, tip_uv, color, label)

            # 坐标系 ID 标注（原点旁）
            if origin_uv is not None:
                name_lbl = frame.name or frame.frame_id
                put_text(disp_frame, f"[{name_lbl}]",
                         (origin_uv[0] + 14, origin_uv[1] - 14),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 230), 1, cv2.LINE_AA)

    def _draw_visible_coord_frames(
        self,
        app: Any,
        disp_frame: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        axis_len_mm: float = 120.0,
        grid_ext_mm: float = 400.0,
        grid_step_mm: float = 100.0,
    ):
        """遍历 coord_frame_visibility 勾选的坐标系，各自绘制：
          - 正象限 XY 平面网格 (X: 0→+ext, Y: 0→+ext)
          - X+/Y+/Z+ 三轴正向箭头（各轴仅正方向，负方向不画）
          - 坐标系名称标注

        颜色约定 (BGR): X=红(0,60,220) / Y=绿(40,200,40) / Z=蓝橙(220,140,30)
        网格颜色: 半透明灰 (80,85,95)
        """
        coord_mgr = getattr(app, "coord_mgr", None)
        if coord_mgr is None:
            return
        visibility = getattr(app, "coord_frame_visibility", {})
        pnp_solver = getattr(app, "pnp_solver", None)
        if pnp_solver is None:
            return
        K = getattr(pnp_solver, "camera_matrix", None)
        if K is None:
            return
        dist = getattr(pnp_solver, "dist_coeffs", None)

        # ROI 物件也在这里统一绘制（保持渲染顺序）
        roi_mgr = getattr(app, "roi_mgr", None)
        if roi_mgr is not None:
            self._draw_roi_cuboids_overlay(app, disp_frame, rvec, tvec)

        h_f, w_f = disp_frame.shape[:2]
        R_c_w, _ = cv2.Rodrigues(rvec)
        t_flat = np.asarray(tvec, dtype=np.float64).flatten()

        # 轴颜色与方向
        AXIS_COLORS = [
            ((0, 60, 220),   "X"),
            ((40, 200, 40),  "Y"),
            ((220, 140, 30), "Z"),
        ]
        AXIS_DIRS = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]
        COL_GRID = (80, 85, 95)

        def _proj_w(pt_w):
            """世界坐标点 → 图像像素，相机背面返回 None"""
            p_cam = R_c_w @ np.asarray(pt_w, dtype=np.float64) + t_flat
            if p_cam[2] <= 1e-3:
                return None
            pts, _ = cv2.projectPoints(
                np.asarray(pt_w, dtype=np.float64).reshape(1, 1, 3),
                rvec, tvec, K, dist
            )
            u, v = int(round(pts[0, 0, 0])), int(round(pts[0, 0, 1]))
            if -80 <= u <= w_f + 80 and -80 <= v <= h_f + 80:
                return (u, v)
            return None

        def _seg_w(p0_w, p1_w, color, thick, n_samples=20):
            """世界坐标两端点 → 采样投影连线（自动处理近裁剪）"""
            p0 = np.asarray(p0_w, dtype=np.float64)
            p1 = np.asarray(p1_w, dtype=np.float64)
            prev = None
            for k in range(n_samples + 1):
                s = k / float(n_samples)
                uv = _proj_w(p0 + (p1 - p0) * s)
                if uv is not None and prev is not None:
                    cv2.line(disp_frame, prev, uv, color, thick, cv2.LINE_AA)
                prev = uv

        def _draw_arrow_uv(p0_uv, p1_uv, color, label):
            """图像坐标箭头 + 轴标签"""
            if p0_uv is None or p1_uv is None:
                return
            cv2.line(disp_frame, p0_uv, p1_uv, color, 2, cv2.LINE_AA)
            d = np.array(p1_uv, dtype=np.float64) - np.array(p0_uv, dtype=np.float64)
            n = float(np.linalg.norm(d))
            if n < 8:
                return
            d /= n
            perp = np.array([-d[1], d[0]])
            tip = np.array(p1_uv, dtype=np.float64)
            wl = min(10.0, n * 0.25)
            head = np.array([tip,
                             tip - wl * d + wl * 0.45 * perp,
                             tip - wl * d - wl * 0.45 * perp], dtype=np.int32)
            cv2.fillPoly(disp_frame, [head], color)
            lx = int(p1_uv[0] + d[0] * 6 + 4)
            ly = int(p1_uv[1] + d[1] * 6 + 4)
            put_text(disp_frame, label, (lx, ly),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 2, cv2.LINE_AA)

        for frame in coord_mgr.list_frames():
            if not visibility.get(frame.frame_id, False):
                continue
            T_w_f, is_resolved = coord_mgr.get_frame_to_world(frame.frame_id)
            if not is_resolved:
                continue

            origin_w = T_w_f[:3, 3]
            R_w_f = T_w_f[:3, :3]

            ext = grid_ext_mm
            step = grid_step_mm

            # ---- 正象限 XY 平面网格 ----
            # Y 方向平行线 (固定 y, x 从 0 扫到 ext)
            y_vals = list(range(0, int(ext) + 1, int(step)))
            for iy in y_vals:
                p0_local = np.array([0.0,  iy, 0.0])
                p1_local = np.array([ext,  iy, 0.0])
                p0_w = origin_w + R_w_f @ p0_local
                p1_w = origin_w + R_w_f @ p1_local
                _seg_w(p0_w, p1_w, COL_GRID, 1)
            # X 方向平行线 (固定 x, y 从 0 扫到 ext)
            x_vals = list(range(0, int(ext) + 1, int(step)))
            for ix in x_vals:
                p0_local = np.array([ix, 0.0, 0.0])
                p1_local = np.array([ix,  ext, 0.0])
                p0_w = origin_w + R_w_f @ p0_local
                p1_w = origin_w + R_w_f @ p1_local
                _seg_w(p0_w, p1_w, COL_GRID, 1)

            # ---- 三轴正向箭头 ----
            origin_uv = _proj_w(origin_w)
            for (color, lbl), axis_dir in zip(AXIS_COLORS, AXIS_DIRS):
                tip_w = origin_w + R_w_f @ (axis_dir * axis_len_mm)
                tip_uv = _proj_w(tip_w)
                _draw_arrow_uv(origin_uv, tip_uv, color, lbl)

            # ---- 坐标系名称标注 ----
            if origin_uv is not None:
                name_lbl = frame.name or frame.frame_id
                put_text(disp_frame, f"[{name_lbl}]",
                         (origin_uv[0] + 14, origin_uv[1] - 14),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 210, 230), 1, cv2.LINE_AA)
