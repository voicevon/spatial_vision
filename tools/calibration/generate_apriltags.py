#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AprilTag 16h5 标靶高清生成与排版工具
- 生成 ID 00 ~ 29 的高清标靶图片 (PNG)
- 生成严格 1:1 比例的 A4 PDF 文件，共 3 页，每页 10 个标靶 (0~9, 10~19, 20~29)
- 每个标靶物理尺寸 40.0 mm x 40.0 mm，带有 50.0 mm x 50.0 mm 剪切外框
- 每个标靶均配备四向几何对齐十字架 (便于剪裁后与治具/机械臂基准十字线精准对齐粘贴)
- 顶部配备 100.0 mm 物理打印校验尺，供游标卡尺测量验证 100% 打印无缩放
"""

import os
import sys
import cv2
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm
from reportlab.lib import colors

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.ui.text_rendering import measure_text, put_text
from src.utils.logger import get_logger

log = get_logger(__name__)


def generate_tags(output_dir: str = "data/apriltags_16h5",
                  tag_count: int = 30,
                  tag_pixel_size: int = 800,
                  border_bits: int = 2):
    """
    生成高清 AprilTag 16h5 标靶 (PNG 格式) 并排版为高精度严格 1:1 的 A4 PDF 文件。
    
    参数:
        output_dir: 输出目录
        tag_count: 标靶总数 (默认 30，对应 ID 00 ~ 29，每页 10 个，共 3 页)
        tag_pixel_size: 单个标靶高分辨率位图的像素尺寸 (默认 800x800，对应 40mm)
        border_bits: AprilTag 黑色边框宽度 (默认 2)
    """
    os.makedirs(output_dir, exist_ok=True)
    raw_marker_dir = os.path.join(output_dir, "_raw_markers")
    os.makedirs(raw_marker_dir, exist_ok=True)
    
    # 获取 OpenCV 内置的 AprilTag 16h5 字典
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
    
    generated_png_files = []
    raw_marker_files = []
    log.info(f"[*] 开始生成 AprilTag 16h5 标靶 (ID 00 ~ {tag_count - 1:02d})...")

    # 40mm 对应 tag_pixel_size (800px), 50mm 剪裁框对应 1000px
    # 每毫米对应像素: 800 / 40.0 = 20 px/mm
    px_per_mm = tag_pixel_size / 40.0
    cut_pixel_size = int(50.0 * px_per_mm)  # 1000 px
    margin_to_cut = (cut_pixel_size - tag_pixel_size) // 2  # 100 px (5mm)
    outer_pad = 80  # 剪裁框外侧留白 (用于画十字延伸线及文本)

    # 准备总览网格图 (5 列 x 6 行)
    cols = 5
    rows = (tag_count + cols - 1) // cols
    card_w = 400
    card_h = 460
    grid_img = np.ones((rows * card_h, cols * card_w, 3), dtype=np.uint8) * 255

    for tag_id in range(tag_count):
        # 1. 生成纯标靶正方形黑白图像 (严格 1:1)
        marker_img = cv2.aruco.generateImageMarker(dictionary, tag_id, tag_pixel_size, borderBits=border_bits)
        raw_marker_path = os.path.join(raw_marker_dir, f"raw_marker_{tag_id:02d}.png")
        cv2.imwrite(raw_marker_path, marker_img)
        raw_marker_files.append(raw_marker_path)
        
        # 2. 转换为带 50mm 剪裁外框、十字对齐线及文字标注的独立 PNG 卡片
        card_total_w = cut_pixel_size + 2 * outer_pad
        card_total_h = cut_pixel_size + 2 * outer_pad + 100
        card = np.ones((card_total_h, card_total_w, 3), dtype=np.uint8) * 255
        
        # 50mm 剪切框在卡片中的左上角坐标
        cut_x1 = outer_pad
        cut_y1 = outer_pad
        cut_x2 = cut_x1 + cut_pixel_size
        cut_y2 = cut_y1 + cut_pixel_size
        
        # 40mm 标靶在卡片中的左上角坐标 (在 50mm 剪切框内严格居中)
        tag_x1 = cut_x1 + margin_to_cut
        tag_y1 = cut_y1 + margin_to_cut
        tag_x2 = tag_x1 + tag_pixel_size
        tag_y2 = tag_y1 + tag_pixel_size
        
        # 贴入标靶
        marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        card[tag_y1:tag_y2, tag_x1:tag_x2] = marker_bgr
        
        # 绘制 40mm 标靶边界细线 (180, 180, 180)
        cv2.rectangle(card, (tag_x1, tag_y1), (tag_x2, tag_y2), (180, 180, 180), 2)
        
        # 绘制 50mm 剪切外框 (深灰细线，提示可用剪刀剪切)
        cv2.rectangle(card, (cut_x1, cut_y1), (cut_x2, cut_y2), (100, 100, 100), 2)
        
        # 绘制四向对齐十字架 (红线，从 40mm 边缘贯穿 50mm 剪切框并向外微延，方便对齐粘贴)
        cx = tag_x1 + tag_pixel_size // 2
        cy = tag_y1 + tag_pixel_size // 2
        cross_ext = 40  # 向外延伸 40px (2mm)
        # 上臂
        cv2.line(card, (cx, cut_y1 - cross_ext), (cx, tag_y1), (0, 0, 230), 2)
        # 下臂
        cv2.line(card, (cx, tag_y2), (cx, cut_y2 + cross_ext), (0, 0, 230), 2)
        # 左臂
        cv2.line(card, (cut_x1 - cross_ext, cy), (tag_x1, cy), (0, 0, 230), 2)
        # 右臂
        cv2.line(card, (tag_x2, cy), (cut_x2 + cross_ext, cy), (0, 0, 230), 2)

        # 在 50mm 剪切框右上角内侧标注微型 ID (防止剪切后无法识别)
        put_text(card, f"#{tag_id:02d}", (cut_x2 - 110, cut_y1 + 45),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 80, 80), 2, cv2.LINE_AA)

        # 底部完整信息标注
        if tag_id == 0:
            label = f"Tag #00 [SCARA Origin Anchor] (Tag: 40mm, Cut: 50mm)"
        elif tag_id == 1:
            label = f"Tag #01 [World +X Axis Ref] (Tag: 40mm, Cut: 50mm)"
        else:
            label = f"Tag #{tag_id:02d} (16h5 | Tag: 40x40mm, Cut: 50x50mm)"
            
        put_text(card, label, (cut_x1, cut_y2 + 65), 
                 cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2, cv2.LINE_AA)
        
        out_png_path = os.path.join(output_dir, f"tag16h5_id_{tag_id:02d}.png")
        cv2.imwrite(out_png_path, card)
        generated_png_files.append(out_png_path)
        
        # 填充到 PNG 总览图
        r = tag_id // cols
        c = tag_id % cols
        small_card = cv2.resize(card, (card_w, card_h), interpolation=cv2.INTER_AREA)
        grid_img[r * card_h:(r + 1) * card_h, c * card_w:(c + 1) * card_w] = small_card

    # 保存 PNG 总览大图
    grid_png_path = os.path.join(output_dir, "apriltags_16h5_all_grid.png")
    cv2.imwrite(grid_png_path, grid_img)
    log.info(f"[OK] 成功生成 {tag_count} 个独立高清标靶 PNG 文件: tag16h5_id_00.png ~ tag16h5_id_{tag_count - 1:02d}.png")
    log.info(f"[OK] 成功生成总览排版 PNG: {grid_png_path}")

    # =========================================================================
    # 生成高精度、严格 1:1 比例的 A4 打印 PDF 文件 (3 页，每页 10 个标靶)
    # =========================================================================
    pdf_path = os.path.join(output_dir, "apriltags_16h5_grid_a4.pdf")
    build_a4_pdf(pdf_path, raw_marker_files, tag_count=tag_count)
    log.info(f"[OK] 成功生成严格 1:1 A4 排版 PDF 文件 (3页，每页10个): {pdf_path}")

    # 清理临时 raw_markers
    for f in raw_marker_files:
        try:
            os.remove(f)
        except OSError:
            pass  # 清理容错：临时文件删除失败不影响成品交付
    try:
        os.rmdir(raw_marker_dir)
    except OSError:
        pass  # 清理容错：临时目录删除失败不影响成品交付

    return pdf_path


def build_a4_pdf(pdf_path: str, raw_marker_files: list, tag_count: int = 30):
    """
    使用 ReportLab 构建高精度、绝对严格 1:1 长宽比的 A4 PDF 标靶纸。
    - 总页数: 3 页
      - 第 1 页: ID 00 ~ 09 (共 10 个)
      - 第 2 页: ID 10 ~ 19 (共 10 个)
      - 第 3 页: ID 20 ~ 29 (共 10 个)
    - 每页排版: 2 列 x 5 行 = 10 个标靶
    - 标靶物理尺寸: 40.0 mm x 40.0 mm (黑色实体)
    - 剪切外框尺寸: 50.0 mm x 50.0 mm (外框放大到 50mm 作为剪切边界，四周各有 5.0mm 留白)
    - 全标靶对齐十字架: 每个标靶均带有四向红线十字对齐线，方便剪裁后与治具/机械臂十字基准线精确对齐粘贴
    - 每页顶部配备 100.0 mm 物理打印校验尺，供游标卡尺测量验证 100% 打印无失真
    """
    page_w, page_h = A4  # 210mm x 297mm
    c = canvas.Canvas(pdf_path, pagesize=A4)
    
    tags_per_page = 10
    cols = 2
    rows = 5
    num_pages = (tag_count + tags_per_page - 1) // tags_per_page  # 3 页
    
    # 尺寸参数
    tag_size_mm = 40.0
    tag_draw_size = tag_size_mm * mm
    cut_size_mm = 50.0
    cut_draw_size = cut_size_mm * mm
    margin_to_cut = (cut_draw_size - tag_draw_size) / 2.0  # 5.0 mm
    
    margin_x = 12.0 * mm
    grid_w = page_w - 2 * margin_x         # 210 - 24 = 186.0 mm
    col_w = grid_w / cols                  # 186 / 2 = 93.0 mm
    
    top_grid_y = page_h - 23.0 * mm        # 顶部留 23mm 放置标题、说明与 100mm 校验尺
    bottom_grid_y = 6.0 * mm               # 底部留 6mm 边距
    grid_h = top_grid_y - bottom_grid_y    # 268.0 mm
    row_h = grid_h / rows                  # 268 / 5 = 53.6 mm

    for page_idx in range(num_pages):
        start_id = page_idx * tags_per_page
        end_id = min(start_id + tags_per_page, tag_count)
        
        # ---------------- 1. 顶部标题、页码与说明 ----------------
        c.setFont("Helvetica-Bold", 11)
        c.drawCentredString(page_w / 2.0, page_h - 8.5 * mm, 
                            f"AprilTag 16h5 Calibration Sheet (ID {start_id:02d} - {end_id - 1:02d}) [Page {page_idx + 1}/{num_pages}]")
        
        c.setFont("Helvetica", 7.0)
        c.setFillColor(colors.HexColor("#333333"))
        c.drawCentredString(page_w / 2.0, page_h - 12.0 * mm, 
                            "Print Setting: Set scale to '100% / Actual Size' (Do NOT 'Fit to Paper'). Aspect ratio: Strictly 1:1.")

        # ---------------- 2. 100.0 mm 打印比例校验标尺 ----------------
        scale_len_mm = 100.0
        scale_x1 = (page_w - scale_len_mm * mm) / 2.0
        scale_x2 = scale_x1 + scale_len_mm * mm
        scale_y = page_h - 17.5 * mm
        
        c.setStrokeColor(colors.black)
        c.setLineWidth(0.8)
        # 主水平线与端点垂直卡线
        c.line(scale_x1, scale_y, scale_x2, scale_y)
        c.line(scale_x1, scale_y - 2.5 * mm, scale_x1, scale_y + 2.5 * mm)
        c.line(scale_x2, scale_y - 2.5 * mm, scale_x2, scale_y + 2.5 * mm)
        # 中间 50mm 刻度线
        c.line(scale_x1 + 50.0 * mm, scale_y - 1.5 * mm, scale_x1 + 50.0 * mm, scale_y + 1.5 * mm)
        
        # 10mm 细刻度
        c.setLineWidth(0.4)
        for tick_idx in range(1, 10):
            if tick_idx == 5:
                continue
            tx = scale_x1 + tick_idx * 10.0 * mm
            c.line(tx, scale_y - 1.0 * mm, tx, scale_y + 1.0 * mm)

        c.setFont("Helvetica-Bold", 6.2)
        c.setFillColor(colors.black)
        c.drawCentredString(page_w / 2.0, scale_y + 1.2 * mm, "|<--- 100.0 mm Scale Verification Bar (Measure with Caliper) --->|")

        # ---------------- 3. 本页 10 个标靶网格排版 (2 列 x 5 行) ----------------
        for idx in range(start_id, end_id):
            local_idx = idx - start_id
            r = local_idx // cols
            col_idx = local_idx % cols
            
            cell_x = margin_x + col_idx * col_w
            # ReportLab 原点在左下角，第 0 行在最上方
            cell_y = top_grid_y - (r + 1) * row_h
            
            # 单元格轻微参考分割虚线 (极淡浅灰)
            c.setStrokeColor(colors.HexColor("#EAEAEA"))
            c.setLineWidth(0.2)
            c.setDash(2, 3)
            c.rect(cell_x, cell_y, col_w, row_h, stroke=1, fill=0)
            c.setDash()  # 恢复实线
            
            # 50.0 mm x 50.0 mm 剪切外框定位 (靠左居中偏左放置，右侧留空间标文字)
            # col_w = 93mm, cut_draw_size = 50mm, 预留左边距 6mm，右侧有 37mm 放置文字信息
            cut_x = cell_x + 6.0 * mm
            cut_y = cell_y + (row_h - cut_draw_size) / 2.0
            
            # 40.0 mm 标靶在 50.0 mm 剪切框内严格居中
            tag_x = cut_x + margin_to_cut
            tag_y = cut_y + margin_to_cut
            
            cx = tag_x + tag_draw_size / 2.0
            cy = tag_y + tag_draw_size / 2.0
            
            # 绘制严格 1:1 正方形 AprilTag 图片
            marker_file = raw_marker_files[idx]
            c.drawImage(marker_file, tag_x, tag_y, 
                        width=tag_draw_size, height=tag_draw_size, 
                        preserveAspectRatio=True)
            
            # 标靶 40mm 黑色外框边缘极细线 (方便卡尺测量验证)
            c.setStrokeColor(colors.HexColor("#777777"))
            c.setLineWidth(0.3)
            c.rect(tag_x, tag_y, tag_draw_size, tag_draw_size, stroke=1, fill=0)
            
            # 绘制 50.0 mm x 50.0 mm 剪切外框 (深灰色细线，带剪切角标提示)
            c.setStrokeColor(colors.HexColor("#555555"))
            c.setLineWidth(0.5)
            c.rect(cut_x, cut_y, cut_draw_size, cut_draw_size, stroke=1, fill=0)
            
            # 绘制每一个标靶的四向对齐十字架 (红线，用于对齐和粘贴)
            # 穿过 5mm 白边，并向 50mm 剪裁框外侧微延 2.0mm
            c.setStrokeColor(colors.HexColor("#D32F2F"))
            c.setLineWidth(0.6)
            ext_mm = 2.0 * mm
            # 上臂
            c.line(cx, tag_y + tag_draw_size, cx, cut_y + cut_draw_size + ext_mm)
            # 下臂
            c.line(cx, tag_y, cx, cut_y - ext_mm)
            # 左臂
            c.line(tag_x, cy, cut_x - ext_mm, cy)
            # 右臂
            c.line(tag_x + tag_draw_size, cy, cut_x + cut_draw_size + ext_mm, cy)
            
            # 剪切框右上角内侧微型标号 (剪下后仍可清晰辨认 ID)
            c.setFont("Helvetica-Bold", 6.5)
            c.setFillColor(colors.HexColor("#555555"))
            c.drawRightString(cut_x + cut_draw_size - 1.5 * mm, cut_y + cut_draw_size - 3.2 * mm, f"#{idx:02d}")
            
            # 剪切框左上角剪切提示小标记
            c.setFont("Helvetica", 5.0)
            c.setFillColor(colors.HexColor("#888888"))
            c.drawString(cut_x + 1.2 * mm, cut_y + cut_draw_size - 3.0 * mm, "CUT 50mm")
            
            # 右侧详细说明区域 (cut_x + cut_draw_size + 4mm)
            text_x = cut_x + cut_draw_size + 4.5 * mm
            text_center_y = cell_y + row_h / 2.0
            
            c.setFillColor(colors.black)
            c.setFont("Helvetica-Bold", 9.0)
            if idx == 0:
                c.drawString(text_x, text_center_y + 12.0 * mm, "Tag #00")
                c.setFont("Helvetica-Bold", 7.2)
                c.setFillColor(colors.HexColor("#D32F2F"))
                c.drawString(text_x, text_center_y + 7.5 * mm, "[SCARA Origin]")
            elif idx == 1:
                c.drawString(text_x, text_center_y + 12.0 * mm, "Tag #01")
                c.setFont("Helvetica-Bold", 7.2)
                c.setFillColor(colors.HexColor("#1976D2"))
                c.drawString(text_x, text_center_y + 7.5 * mm, "[Ref +X Axis]")
            else:
                c.drawString(text_x, text_center_y + 10.0 * mm, f"Tag #{idx:02d}")
                c.setFont("Helvetica", 6.8)
                c.setFillColor(colors.HexColor("#666666"))
                c.drawString(text_x, text_center_y + 6.0 * mm, "AprilTag 16h5")
            
            c.setFont("Helvetica", 6.5)
            c.setFillColor(colors.HexColor("#333333"))
            c.drawString(text_x, text_center_y + 0.5 * mm, f"Tag: 40.0 x 40.0 mm")
            c.drawString(text_x, text_center_y - 4.5 * mm, f"Cut: 50.0 x 50.0 mm")
            
            c.setFont("Helvetica", 5.8)
            c.setFillColor(colors.HexColor("#888888"))
            c.drawString(text_x, text_center_y - 9.5 * mm, "+ Crosshair Alignment")
            c.drawString(text_x, text_center_y - 14.0 * mm, "Border: 5.0 mm margin")

        c.showPage()
        
    c.save()


if __name__ == "__main__":
    generate_tags()
