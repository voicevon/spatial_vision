#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
建图 CLI 流程编排 (Builder Workflow)
====================================
从 tag_map_builder.py 拆出的命令行流程编排层：
  - interactive_workflow: 交互式两阶段建图与人工质量审核工作流
  - main: CLI 参数解析与运行模式分发（仅导出清单 / 深度诊断 / 静默求解 / 交互控制台）
纯几何与 BA 求解逻辑保留在 TagMapBuilder 类内 (tag_map_builder.py)。
"""

import os
import sys
import glob
import yaml
import argparse
from typing import Dict, List, Tuple, Optional, Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.calibration.solvers.covisibility_graph import CovisibilityGraphAnalyzer
from tools.calibration.tag_map_builder import TagMapBuilder
from src.utils.logger import get_logger

log = get_logger(__name__)




def interactive_workflow(args, builder: TagMapBuilder, image_paths: List[str], baseline_pair: Optional[Tuple[int, int, float]]):
    """交互式两阶段建图与人工质量审核工作流"""
    manifest_path = args.manifest
    vis_dir = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), "visualized")

    # 1. 若清单不存在，先自动导出
    if not os.path.exists(manifest_path):
        log.info(f"[*] 首次运行，正在自动提取并生成观测数据清单: {manifest_path} ...")
        builder.repository.export_manifest(image_paths=image_paths, manifest_path=manifest_path)

    while True:
        # 加载清单与状态
        try:
            frame_detections, valid_frames, stats = builder.repository.load_manifest(manifest_path=manifest_path)
            covis_report = CovisibilityGraphAnalyzer.analyze(frame_detections, valid_frames, args.origin_id, args.x_axis_id)
        except Exception as e:
            log.warning(f"读取或解析观测清单异常: {e}")
            covis_report = {"is_valid": False, "message": str(e), "all_tags": [], "critical_bridges": []}
            stats = {"total_images": 0, "total_observations": 0, "total_kept": 0, "total_excluded": 0, "excluded_items": []}

        # 终端横幅
        print("\n" + "=" * 76)
        print("     【AprilTag 两阶段人工审核与 BA 空间平差控制台】(Tag Map Builder)")
        print("=" * 76)
        print(f" 采图目录: {args.image_dir} ({len(image_paths)} 帧)")
        print(f" 审核清单: {manifest_path}")
        print(f" 观测统计: 共 {stats['total_observations']} 次标靶检出 | 保留: {stats['total_kept']} | 人工剔除: {stats['total_excluded']}")

        if covis_report["is_valid"]:
            status_tag = f"\033[92m[ 连通网健康 (PASS) ]\033[0m"
            print(f" 拓扑状态: {status_tag} 全部 {len(covis_report['all_tags'])} 个标靶完全连通 (有效参与: {len(valid_frames)} 帧)")
        else:
            status_tag = f"\033[91m[ 拓扑断网告警 (FAIL) ]\033[0m"
            print(f" 拓扑状态: {status_tag} {covis_report['message']}")

        if covis_report.get("critical_bridges"):
            print(f" 关键桥梁: 发现 {len(covis_report['critical_bridges'])} 对标靶仅由单图连接: {covis_report['critical_bridges']} (注意不要剔除桥梁帧)")

        print("-" * 76)
        print(" 【第一阶段：数据审核与拓扑评估】")
        print("   [1] 启动交互审核画板 (鼠标点击标靶直接剔除/恢复，实时拓扑安全红绿灯)")
        print("   [2] 查看共视拓扑结构深度诊断报告 (节点度数、共视重叠帧数、关键桥梁)")
        print("")
        print(" 【辅助维护工具集 (无固定顺序，按需调用)】")
        print("   [3] 刷新/重新扫描图像并更新清单 (自动保留已有 keep: false 与备注)")
        print("   [4] 在系统资源管理器中打开可视化标注图目录 (visualized/) 查看像元标牌")
        print("   [5] (或 [E]) 在系统文本编辑器中直接编辑清单 (tag_observations.yaml)")
        print("")
        print(" 【最终收官：终审求解与成果导出】")
        print("   [6] (或 [B]) 立即执行 BA 全局平差优化并导出地图 (config/tags_map.yaml)")
        print("----------------------------------------------------------------------------")
        print("   [Q] 退出")
        print("=" * 76)

        try:
            choice = input("请输入操作编号 [1-6, E, B, Q]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            log.error("\n[退出] 操作已终止。")
            break

        if choice == '1':
            try:
                from tools.calibration.tag_manifest_reviewer import TagManifestReviewer
                log.info("\n[启动] 正在启动 AprilTag 观测样本交互审核画板...")
                reviewer = TagManifestReviewer(manifest_path=manifest_path, builder=builder)
                reviewer.run()
            except Exception as e:
                log.warning(f"\n启动交互画板失败或当前环境无显示服务: {e}")
                log.warning("已自动回退到调用系统文本编辑器打开 YAML 清单。")
                abs_manifest = os.path.abspath(manifest_path)
                try:
                    if sys.platform == "win32":
                        os.startfile(abs_manifest)
                    elif sys.platform == "darwin":
                        import subprocess
                        subprocess.run(["open", abs_manifest])
                    else:
                        import subprocess
                        subprocess.run(["xdg-open", abs_manifest])
                except Exception as e:
                    log.warning(f"调用系统编辑器打开清单失败: {abs_manifest}: {e}")
                try:
                    input("\n修改保存完毕后，请按回车键刷新...")
                except (EOFError, KeyboardInterrupt):
                    pass  # 用户中断/输入流关闭时直接继续，属预期路径

        elif choice == '2':
            CovisibilityGraphAnalyzer.print_topology_report(covis_report, stats, valid_frames)
            try:
                input("\n按回车键返回菜单...")
            except (EOFError, KeyboardInterrupt):
                pass  # 用户中断/输入流关闭时直接继续，属预期路径

        elif choice == '3':
            log.info(f"\n[*] 正在重新扫描所有标定图像并刷新清单 (历史人工标记将安全保留)...")
            builder.repository.export_manifest(image_paths=image_paths, manifest_path=manifest_path)
            try:
                input("\n清单刷新完成，按回车键继续...")
            except (EOFError, KeyboardInterrupt):
                pass  # 用户中断/输入流关闭时直接继续，属预期路径

        elif choice == '4':
            os.makedirs(vis_dir, exist_ok=True)
            log.info(f"\n[浏览] 正在打开可视化标注目录: {vis_dir} ...")
            try:
                if sys.platform == "win32":
                    os.startfile(vis_dir)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", vis_dir])
                else:
                    import subprocess
                    subprocess.run(["xdg-open", vis_dir])
            except Exception as e:
                log.warning(f"打开目录失败: {e}")
            try:
                input("\n按回车键返回菜单...")
            except (EOFError, KeyboardInterrupt):
                pass  # 用户中断/输入流关闭时直接继续，属预期路径

        elif choice in ('5', 'E'):
            abs_manifest = os.path.abspath(manifest_path)
            log.info(f"\n[打开] 正在尝试调用系统默认编辑器打开: {abs_manifest} ...")
            try:
                if sys.platform == "win32":
                    os.startfile(abs_manifest)
                elif sys.platform == "darwin":
                    import subprocess
                    subprocess.run(["open", abs_manifest])
                else:
                    import subprocess
                    subprocess.run(["xdg-open", abs_manifest])
            except Exception as e:
                log.warning(f"无法自动拉起编辑器: {e}，请手动打开该文件。")
            try:
                input("\n修改保存完毕后，请按回车键重新读取并刷新拓扑连通状态...")
            except (EOFError, KeyboardInterrupt):
                pass  # 用户中断/输入流关闭时直接继续，属预期路径

        elif choice in ('6', 'B'):
            if not covis_report["is_valid"]:
                log.warning(f"\n\033[91m[拦截] 共视拓扑检查未通过，禁止执行 BA 优化！\033[0m")
                log.info(f"原因: {covis_report['message']}")
                log.info("请先选择选项 [1] 打开交互画板，恢复失联标靶的关键视角后再继续。")
                try:
                    input("\n按回车键返回菜单...")
                except (EOFError, KeyboardInterrupt):
                    pass  # 用户中断/输入流关闭时直接继续，属预期路径
                continue

            try:
                tags_map = builder.optimize_bundle_adjustment(
                    frame_detections=frame_detections,
                    active_frame_names=valid_frames,
                    origin_tag_id=args.origin_id,
                    x_align_tag_id=args.x_axis_id,
                    baseline_pair=baseline_pair
                )
                builder.save_map(tags_map, args.output)
                log.info(f"\n\033[92m[SUCCESS] 标靶空间地图优化求解大功告成！已成功输出至: {args.output}\033[0m")
                break
            except Exception as e:
                log.warning(f"\n\033[91mBA 平差求解失败: {e}\033[0m")
                try:
                    input("\n按回车键返回菜单...")
                except (EOFError, KeyboardInterrupt):
                    pass  # 用户中断/输入流关闭时直接继续，属预期路径

        elif choice in ('Q', '0'):
            log.error("\n[退出] 已退出建图工具。")
            break


def main():
    parser = argparse.ArgumentParser(description="AprilTag 16h5 多标靶离线两阶段建图与 BA 平差工具")
    # 基准标靶 ID 与地图输出路径已下沉至工位沙盒 (origin_tag_id / x_axis_tag_id / map_path)
    def_origin_id = 0
    def_x_axis_id = 28
    def_image_dir = ""
    def_manifest = ""
    def_output = ""
    try:
        from src.workspace.workspace_manager import WorkspaceManager
        current_ws = WorkspaceManager().get_current_workspace()
        def_image_dir = current_ws.calib_raw_images_dir
        def_manifest = current_ws.calib_manifest_path
        def_output = current_ws.map_path
        # 注: 基准 ID 若工位未配置, 由 BA 求解器自动选择 (该变量仅 CLI 兼容占位)
    except Exception as e:
        log.warning(f"获取当前工位失败 (将以默认值运行): {e}")

    parser.add_argument("--image_dir", type=str, default=def_image_dir, help="多视角标定图片目录")
    parser.add_argument("--manifest", type=str, default=def_manifest, help="观测数据审核清单路径")
    parser.add_argument("--marker_size", type=float, default=None, help="标靶黑白边框名义边长 (mm), 缺省从工位 tag_whitelist.yaml.tag_default_size_mm 读取")
    parser.add_argument("--origin_id", type=int, default=def_origin_id, help="SCARA 原点锚定标靶 ID")
    parser.add_argument("--x_axis_id", type=int, default=def_x_axis_id, help="世界 X 轴对齐基准标靶 ID (默认与 config.yaml 一致)")
    parser.add_argument("--baseline_pair", nargs=3, type=float, metavar=('TAG_A', 'TAG_B', 'DIST_MM'),
                        help="双标靶基线尺度校准参数: TAG_A TAG_B 真实距离(mm), 例如: --baseline_pair 1 5 620.5")
    parser.add_argument("--output", type=str, default=def_output, help="导出的图谱文件路径")
    parser.add_argument("--export-manifest", action="store_true", help="仅扫描图像导出观测数据清单并退出")
    parser.add_argument("--solve-manifest", action="store_true", help="直接读取清单执行 BA 平差 (非交互批处理)")
    parser.add_argument("--inspect", action="store_true", help="仅执行连通性深度诊断并输出报告后退出")
    parser.add_argument("--no-interactive", action="store_true", help="静默非交互模式运行")
    args = parser.parse_args()

    # 寻找图像文件
    patterns = [os.path.join(args.image_dir, ext) for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp")]
    image_paths = []
    for pat in patterns:
        image_paths.extend(glob.glob(pat))

    if not image_paths and not os.path.exists(args.manifest):
        log.warning(f"[!] 目录 '{args.image_dir}' 下未找到任何标定图像，且未找到已有清单 '{args.manifest}'！")
        log.warning(f"[*] 提示：请使用相机采集覆盖多标靶的图像放入该目录后重试。")
        sys.exit(1)

    # 标靶物理边长: 显式传入优先, 否则从当前工位 tag_whitelist.yaml.tag_default_size_mm 读取
    if args.marker_size is None:
        try:
            from src.workspace.workspace_manager import load_workspace_marker_size_mm
            loaded = load_workspace_marker_size_mm(current_ws.workspace_dir) if current_ws else None
        except Exception:
            loaded = None
        if loaded is None:
            log.error("未指定标靶物理边长 (--marker_size) 且当前工位 tag_whitelist.yaml.tag_default_size_mm 缺失或非法.")
            sys.exit(1)
        args.marker_size = loaded

    builder = TagMapBuilder(marker_size_mm=args.marker_size)

    baseline_pair = None
    if args.baseline_pair is not None:
        baseline_pair = (int(args.baseline_pair[0]), int(args.baseline_pair[1]), float(args.baseline_pair[2]))

    # 单独模式 1: 仅导出清单
    if args.export_manifest:
        builder.repository.export_manifest(image_paths=image_paths, manifest_path=args.manifest)
        log.info("[OK] 清单导出完毕，已退出。")
        return

    # 单独模式 2: 仅深度诊断
    if args.inspect:
        if not os.path.exists(args.manifest):
            builder.repository.export_manifest(image_paths=image_paths, manifest_path=args.manifest)
        f_det, v_frames, stats = builder.repository.load_manifest(manifest_path=args.manifest)
        report = CovisibilityGraphAnalyzer.analyze(f_det, v_frames, args.origin_id, args.x_axis_id)
        CovisibilityGraphAnalyzer.print_topology_report(report, stats, v_frames)
        return

    # 单独模式 3: 直接静默求解
    if args.solve_manifest or args.no_interactive or (not sys.stdin.isatty()):
        if not os.path.exists(args.manifest):
            builder.repository.export_manifest(image_paths=image_paths, manifest_path=args.manifest)
        tags_map = builder.build_map_from_manifest(
            manifest_path=args.manifest,
            origin_tag_id=args.origin_id,
            x_align_tag_id=args.x_axis_id,
            baseline_pair=baseline_pair
        )
        builder.save_map(tags_map, args.output)
        return

    # 默认模式: 启动交互式审核与建图控制台
    interactive_workflow(args, builder, image_paths, baseline_pair)


if __name__ == "__main__":
    main()
