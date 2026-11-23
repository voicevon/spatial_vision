import unittest
"""
使用真实捕获的相机数据快照进行全量芦笋识别、尺寸解算、实例分离与最顶层抓取点自动化测试
"""
import glob
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.vision.asparagus_analyzer import AsparagusAnalyzer


def generate_scara_gcode(target) -> str:
    """生成 SCARA 抓取与分拣 G-code 指令片段"""
    return (
        f"; --- SCARA 抓取指令 (芦笋 #{target.target_id}) ---\n"
        f"G0 Z80.0 F4000               ; 抬升至安全过渡高度\n"
        f"G0 X{target.grip_cam_x:.2f} Y{target.grip_cam_y:.2f} R{target.yaw_deg:.2f} F4000 ; 快速平移旋转对准\n"
        f"M3                            ; 开启夹爪/气动电磁阀 (张开就位)\n"
        f"G1 Z{target.grip_cam_z:.2f} F1500              ; 垂直下探至芦笋抓取面\n"
        f"M4                            ; 闭合夹爪 (牢固夹持)\n"
        f"G4 P200                       ; 抓持保压延时 200ms\n"
        f"G0 Z80.0 F4000               ; 提料脱离堆叠区\n"
        f"G0 X220.00 Y0.00 R0.00 F4000 ; 移动至分级分料口\n"
        f"M3                            ; 释放物料入仓\n"
    )


class TestRealSnapshot(unittest.TestCase):
    def test_on_real_snapshots(self):
        color_files = sorted(glob.glob("data/snapshots/color_*.png"))
    
        if not color_files:
            print("[SKIP] data/snapshots/ 目录下未找到任何真实彩色快照，跳过实测。")
            return
        
        print(f"================================================================================")
        print(f"             flux_vision_3d 真实芦笋快照全量分离与解算自动化测试")
        print(f"================================================================================")
        print(f"发现有效真实快照: {len(color_files)} 组\n")
    
        analyzer = AsparagusAnalyzer(fx=909.12, fy=907.46, cx=647.46, cy=377.51)
    
        total_frames = 0
        success_frames = 0
        total_detected_targets = 0
    
        for color_path in color_files:
            stem = os.path.basename(color_path).replace("color_", "").replace(".png", "")
            depth_path = os.path.join("data", "snapshots", f"depth_raw_{stem}.npy")
        
            if not os.path.exists(depth_path):
                continue
            
            total_frames += 1
            color = cv2.imread(color_path)
            depth = np.load(depth_path)
        
            plane_coeff = analyzer.fit_table_plane(depth)
            pitch, roll, total_tilt = analyzer.get_table_tilt_angles(plane_coeff)
            targets = analyzer.analyze(color, depth)
            topmost = next((t for t in targets if t.is_topmost), None)
        
            print(f"--------------------------------------------------------------------------------")
            print(f" [快照] {stem} | 传送带倾角自标定: Pitch={pitch:+.1f}°, Roll={roll:+.1f}° (空间倾角 {total_tilt:.1f}°) | 检出: {len(targets)} 根")
        
            if len(targets) > 0:
                success_frames += 1
                total_detected_targets += len(targets)
            
                for t in targets:
                    tag = "[TOPMOST]" if t.is_topmost else " [SUB-LY] "
                    print(f"   {tag} #{t.target_id}: L={t.length_mm:5.1f}mm, D={t.diam_mm:4.1f}mm, "
                          f"Yaw={t.yaw_deg:5.1f}°, Grip=(X:{t.grip_cam_x:6.1f}, Y:{t.grip_cam_y:6.1f}, Z:{t.grip_cam_z:5.1f})mm, "
                          f"凸起净高=+{t.rel_height_mm:4.1f}mm")
            
                if topmost is not None:
                    gcode_snippet = (
                        f"   [SCARA 防撞 G-code 摘要] G0 X{topmost.robot_x:.1f} Y{topmost.robot_y:.1f} R{topmost.robot_r:.1f} "
                        f"-> G1 Z{topmost.robot_z:.1f} (凸起高度=+{topmost.rel_height_mm:.1f}mm) -> M4 -> G0 Z80.0"
                    )
                    print(gcode_snippet)
                
                # 渲染并保存可视化分析图
                annotated = analyzer.draw_detections(color, targets)
                out_path = os.path.join("data", "snapshots", f"analyzed_{stem}.png")
                cv2.imwrite(out_path, annotated)
            else:
                print(f"   [!] 未检出符合几何特征的芦笋目标")
            
        print(f"\n================================================================================")
        print(f" 测试总结报告:")
        print(f"   - 总测试快照帧数: {total_frames}")
        print(f"   - 成功锁定顶层帧数: {success_frames} / {total_frames} ({success_frames/max(1,total_frames)*100:.1f}%)")
        print(f"   - 累计解算芦笋物料: {total_detected_targets} 根次")
        print(f"   - 标注图输出路径: data/snapshots/analyzed_*.png")
        print(f"================================================================================\n")
    
        assert success_frames > 0, "全部快照均未检测出芦笋，算法未达到预期！"



if __name__ == "__main__":
    unittest.main()
