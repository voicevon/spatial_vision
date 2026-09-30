import os
import sys
import shutil
import tempfile
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.workspace.coordinate_manager import (
    CoordinateTreeManager,
    FrameDefinition,
    rpy_deg_to_rot_mat,
    make_transform_matrix,
)
from src.workspace.roi_manager import (
    RoiSpaceManager,
    RoiDefinition,
)
from src.workspace.workspace_manager import (
    WorkspaceManager,
    Workspace,
    load_workspace_coordinate_manager,
    load_workspace_roi_manager,
)


def test_frame_hierarchy_and_transforms():
    """测试坐标系树层级变换与点云转换"""
    coord_mgr = CoordinateTreeManager(workspace_id="test_ws")

    # 1. 默认根节点应为 world
    frames = coord_mgr.list_frames()
    assert len(frames) == 1
    assert frames[0].frame_id == "world"

    # 2. 添加相对坐标系 frame_a: 相对于 world 平移 [100, 0, 0]
    frame_a = FrameDefinition(
        frame_id="frame_a",
        name="A工位",
        parent_frame_id="world",
        type="fixed_transform",
        translation_xyz_mm=[100.0, 0.0, 0.0],
        rotation_rpy_deg=[0.0, 0.0, 0.0],
    )
    assert coord_mgr.add_frame(frame_a) is True

    # 3. 添加二级相对坐标系 frame_b: 相对于 frame_a 平移 [0, 50, 0]
    frame_b = FrameDefinition(
        frame_id="frame_b",
        name="B滑台",
        parent_frame_id="frame_a",
        type="fixed_transform",
        translation_xyz_mm=[0.0, 50.0, 0.0],
        rotation_rpy_deg=[0.0, 0.0, 90.0],  # 沿 Z 轴旋转 90度
    )
    assert coord_mgr.add_frame(frame_b) is True

    # 验证 frame_b 到 world 的变换矩阵
    T_w_b, is_res = coord_mgr.get_frame_to_world("frame_b")
    assert is_res is True
    # 平移应该为 [100, 50, 0]
    assert np.allclose(T_w_b[:3, 3], [100.0, 50.0, 0.0], atol=1e-3)

    # 验证点云转换: 在 frame_b 原点处的点 [0, 0, 0]，变换到 world 应当在 [100, 50, 0]
    pts_b = np.array([[0.0, 0.0, 0.0]])
    pts_w = coord_mgr.transform_points(pts_b, "frame_b", "world")
    assert np.allclose(pts_w[0], [100.0, 50.0, 0.0], atol=1e-3)


def test_tag_bound_frame():
    """测试基于 AprilTag 绑定的动标坐标系"""
    coord_mgr = CoordinateTreeManager(workspace_id="test_ws")

    # 模拟地图中 Tag 15 的世界位姿 (平移 [200, 300, 50])
    T_world_tag15 = make_transform_matrix(np.eye(3), [200.0, 300.0, 50.0])
    coord_mgr.set_tags_map({15: T_world_tag15})

    # 添加动标坐标系: 绑定 Tag 15, 局部偏移 [0, 0, 10]
    frame_slider = FrameDefinition(
        frame_id="frame_slider",
        name="动态滑块",
        parent_frame_id="world",
        type="tag_bound",
        tag_id=15,
        offset_xyz_mm=[0.0, 0.0, 10.0],
    )
    assert coord_mgr.add_frame(frame_slider) is True

    T_w_slider, is_res = coord_mgr.get_frame_to_world("frame_slider")
    assert is_res is True
    assert np.allclose(T_w_slider[:3, 3], [200.0, 300.0, 60.0], atol=1e-3)


def test_cycle_detection():
    """测试坐标系拓扑循环引用防护与无效父级校验"""
    coord_mgr = CoordinateTreeManager(workspace_id="test_ws")

    # 1. 尝试挂载到不存在的父级
    f_invalid = FrameDefinition(frame_id="f_invalid", name="Invalid", parent_frame_id="non_existent")
    assert coord_mgr.add_frame(f_invalid) is False

    # 2. 正常添加 f1 挂在 world, f2 挂在 f1
    f1 = FrameDefinition(frame_id="f1", name="F1", parent_frame_id="world")
    assert coord_mgr.add_frame(f1) is True

    f2 = FrameDefinition(frame_id="f2", name="F2", parent_frame_id="f1")
    assert coord_mgr.add_frame(f2) is True

    # 3. 试图将 f1 的父级改为 f2，形成环路 (f1 -> f2 -> f1)
    f1_bad = FrameDefinition(frame_id="f1", name="F1", parent_frame_id="f2")
    assert coord_mgr.add_frame(f1_bad) is False


def test_roi_world_obb_and_point_filtering():
    """测试 ROI 空间 OBB 求解与点云过滤"""
    coord_mgr = CoordinateTreeManager(workspace_id="test_ws")

    # 定义挂载在相对坐标系上的 ROI
    frame_conveyor = FrameDefinition(
        frame_id="frame_conveyor",
        name="输送机",
        parent_frame_id="world",
        type="fixed_transform",
        translation_xyz_mm=[500.0, 200.0, 100.0],
        rotation_rpy_deg=[0.0, 0.0, 0.0],
    )
    coord_mgr.add_frame(frame_conveyor)

    roi_mgr = RoiSpaceManager(workspace_id="test_ws")
    # ROI 中心在局部 [0, 0, 0]，尺寸 [100, 200, 50] (dx, dy, dz)
    # 则在世界系下中心为 [500, 200, 100]，范围 X:[450, 550], Y:[100, 300], Z:[75, 125]
    roi_belt = RoiDefinition(
        roi_id="roi_belt",
        name="皮带表面",
        frame_id="frame_conveyor",
        category="belt",
        center_xyz_mm=[0.0, 0.0, 0.0],
        size_xyz_mm=[100.0, 200.0, 50.0],
    )
    roi_mgr.add_roi(roi_belt)

    obb = roi_mgr.get_roi_world_obb("roi_belt", coord_mgr)
    assert obb is not None
    assert np.allclose(obb["center_world"], [500.0, 200.0, 100.0], atol=1e-3)
    assert obb["corners_8x3"].shape == (8, 3)

    # 生成测试点云 (世界系)
    pts_world = np.array([
        [500.0, 200.0, 100.0],  # 正中心 -> 盒内
        [460.0, 150.0, 90.0],   # 盒内
        [600.0, 200.0, 100.0],  # X 超界 (600 > 550) -> 盒外
        [500.0, 350.0, 100.0],  # Y 超界 (350 > 300) -> 盒外
    ])

    mask = roi_mgr.get_point_cloud_mask(pts_world, "roi_belt", coord_mgr)
    assert np.array_equal(mask, [True, True, False, False])

    filtered_inside = roi_mgr.filter_point_cloud(pts_world, "roi_belt", coord_mgr, keep_inside=True)
    assert len(filtered_inside) == 2

    filtered_outside = roi_mgr.filter_point_cloud(pts_world, "roi_belt", coord_mgr, keep_inside=False)
    assert len(filtered_outside) == 2


def test_yaml_persistence_and_workspace_clone():
    """测试 YAML 文件持久化与 Workspace 生命周期克隆"""
    tmp_root = tempfile.mkdtemp(prefix="test_ws_root_")
    try:
        ws_mgr = WorkspaceManager(workspaces_dir=tmp_root)
        ws_a = ws_mgr.create_workspace(alias="工位A")

        # 检查初始 spatial_scene.yaml 是否已建立
        assert os.path.exists(ws_a.spatial_scene_path)

        # 向工位 A 写入自定义坐标系和 ROI
        coord_mgr = load_workspace_coordinate_manager(ws_a)
        coord_mgr.add_frame(FrameDefinition(
            frame_id="frame_custom",
            name="自定义系",
            translation_xyz_mm=[10.0, 20.0, 30.0],
        ))
        coord_mgr.save()

        roi_mgr = load_workspace_roi_manager(ws_a)
        roi_mgr.add_roi(RoiDefinition(
            roi_id="roi_test_01",
            name="测试ROI",
            frame_id="frame_custom",
            role="source",
            target_intent="pose_pick",
            binding={"channel": "mqtt_loader", "slot_index": 2},
            pipeline_override="feng_green_axis_v2",
            min_confidence=0.75,
        ))
        roi_mgr.save()

        # 克隆工位 A -> 工位 B
        ws_b = ws_mgr.clone_workspace(ws_a.workspace_id, new_alias="工位B")
        assert ws_b is not None
        assert os.path.exists(ws_b.spatial_scene_path)

        # 验证工位 B 是否完整继承了坐标系和 ROI (包括 Smart ROI 扩展属性)
        coord_mgr_b = load_workspace_coordinate_manager(ws_b)
        assert coord_mgr_b.get_frame("frame_custom") is not None

        roi_mgr_b = load_workspace_roi_manager(ws_b)
        roi_b = roi_mgr_b.get_roi("roi_test_01")
        assert roi_b is not None
        assert roi_b.role == "source"
        assert roi_b.target_intent == "pose_pick"
        assert roi_b.binding.get("slot_index") == 2
        assert roi_b.pipeline_override == "feng_green_axis_v2"
        assert abs(roi_b.min_confidence - 0.75) < 1e-4

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    print(">>> 正在运行 test_frame_hierarchy_and_transforms...")
    test_frame_hierarchy_and_transforms()
    print("√ test_frame_hierarchy_and_transforms 通过")

    print(">>> 正在运行 test_tag_bound_frame...")
    test_tag_bound_frame()
    print("√ test_tag_bound_frame 通过")

    print(">>> 正在运行 test_cycle_detection...")
    test_cycle_detection()
    print("√ test_cycle_detection 通过")

    print(">>> 正在运行 test_roi_world_obb_and_point_filtering...")
    test_roi_world_obb_and_point_filtering()
    print("√ test_roi_world_obb_and_point_filtering 通过")

    print(">>> 正在运行 test_yaml_persistence_and_workspace_clone...")
    test_yaml_persistence_and_workspace_clone()
    print("√ test_yaml_persistence_and_workspace_clone 通过")

    print("\n===============================")
    print("所有多坐标系与 ROI 空间测试均 100% 通过！")
    print("===============================")

