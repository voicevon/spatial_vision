# -*- coding: utf-8 -*-
"""
方案 B 标靶配置单一真理源防腐与完整性质检测试 (Tag Config Anti-Corruption & Integrity Tests)
验证机制:
1. Schema 严格白名单与垃圾字段清洗 (输入 frame_id / 杂质属性在写穿时被彻底过滤);
2. 工位单一真理源验证 (不再存在孤立的 anchor_tags.yaml);
3. 物理锚点自动收录进白名单准入集合;
4. 真实工位配置纯净度扫描。
"""

import os
import shutil
import tempfile
import unittest
import yaml
from src.workspace.workspace_manager import (
    Workspace,
    load_workspace_tag_config,
    save_workspace_tag_config,
    load_workspace_anchor_tags,
    save_workspace_anchor_tags,
)


class TestTagConfigIntegrity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_tag_config_")
        self.ws_dir = os.path.join(self.test_dir, "20260930_100000_Site_A")
        os.makedirs(self.ws_dir, exist_ok=True)
        self.ws = Workspace(
            workspace_id="20260930_100000_Site_A",
            name="Site_A",
            workspace_dir=self.ws_dir
        )

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_schema_pruning_and_anti_corruption(self):
        """测试防腐机制: 注入脏数据 (frame_id, 杂质键, 重复 allowed_ids) 会被自动清洗剔除"""
        dirty_anchors = {
            0: {
                "xyz_mm": [0.0, 0.0, 0.0],
                "known": [True, True, False],
                "frame_id": "world",          # 非法补丁字段
                "garbage_attr": "junk_data",   # 随机注入垃圾
            },
            14: {
                "coords": [-33.0, 490.0, 0.0], # 兼容字段
                "known": [True, True, True],
                "frame_id": "frame_sub_1",     # 非法补丁字段
            }
        }
        dirty_allowed = [0, 0, 14, 14, 5, "5", "abc"]  # 包含重复与脏字符

        # 执行保存
        ok = save_workspace_tag_config(
            self.ws,
            allowed_ids=dirty_allowed,
            anchor_tags=dirty_anchors,
            tag_default_size_mm=45.0
        )
        self.assertTrue(ok)

        # 检查物理落盘的 YAML 纯净度
        wl_path = self.ws.whitelist_path
        self.assertTrue(os.path.exists(wl_path))
        with open(wl_path, "r", encoding="utf-8") as f:
            raw_yaml_text = f.read()
            doc = yaml.safe_load(raw_yaml_text)

        # 1. 验证没有任何垃圾字段写入文件
        self.assertNotIn("frame_id", raw_yaml_text, "YAML 绝对禁止出现 frame_id 字段！")
        self.assertNotIn("garbage_attr", raw_yaml_text, "未知杂质属性必须被物理剔除！")
        self.assertNotIn("tag_anchors", doc, "旧 tag_anchors 字段已被规范的 anchor_tags 替代")

        # 2. 验证白名单已彻底去重清洗
        self.assertEqual(doc["allowed_ids"], [0, 5, 14])

        # 3. 验证锚点数据结构严格规范与 null 序列化 (彻底剔除 known 冗余键)
        anchors = doc["anchor_tags"]
        self.assertIn(0, anchors)
        self.assertIn(14, anchors)
        self.assertNotIn("known", anchors[0], "物理 YAML 中已彻底剔除冗余的 known 字段！")
        self.assertNotIn("known", anchors[14], "物理 YAML 中已彻底剔除冗余的 known 字段！")
        self.assertEqual(anchors[0]["xyz_mm"], [0.0, 0.0, None])
        self.assertIn("null", raw_yaml_text, "未知轴在物理 YAML 中应自动表现为 null 关键字")
        self.assertEqual(anchors[14]["xyz_mm"], [-33.0, 490.0, 0.0])

    def test_null_axis_syntax_and_automatic_inference(self):
        """测试 YAML 中直接以 null/None/~ 表示未知轴，无需 known 字段即可自动推导"""
        sample_yaml = """
workspace_id: test_null_ws
workspace_name: test_null_ws
tag_default_size_mm: 40.0
allowed_ids: [0, 1, 2]
anchor_tags:
  0:
    xyz_mm: [0.0, 0.0, null]
  1:
    xyz_mm: [100.0, 200.0, 300.0]
  2:
    xyz_mm: [null, null, 50.0]
"""
        wl_path = self.ws.whitelist_path
        with open(wl_path, "w", encoding="utf-8") as f:
            f.write(sample_yaml)

        # 读取并验证算法消费端获取到的数据结构 (安全 float + known mask)
        loaded = load_workspace_anchor_tags(self.ws_dir)
        self.assertIn(0, loaded)
        self.assertEqual(loaded[0]["known"], [True, True, False])
        self.assertEqual(loaded[0]["xyz_mm"], [0.0, 0.0, 0.0])

        self.assertIn(1, loaded)
        self.assertEqual(loaded[1]["known"], [True, True, True])
        self.assertEqual(loaded[1]["xyz_mm"], [100.0, 200.0, 300.0])

        self.assertIn(2, loaded)
        self.assertEqual(loaded[2]["known"], [False, False, True])
        self.assertEqual(loaded[2]["xyz_mm"], [0.0, 0.0, 50.0])

    def test_single_source_self_healing_delete_old_file(self):
        """【无向后兼容原则】旧 anchor_tags.yaml 不再作为有效源读取，写回时自动物理清除孤立文件"""
        old_anchor_file = os.path.join(self.ws_dir, "anchor_tags.yaml")
        with open(old_anchor_file, "w", encoding="utf-8") as f:
            f.write("anchor_tags:\n  1:\n    xyz_mm: [0, 520, null]\n")

        # 验证不再向后兼容读取旧 anchor_tags.yaml，调用端只能从 tag_whitelist.yaml 获取
        loaded = load_workspace_anchor_tags(self.ws_dir)
        self.assertEqual(loaded, {}, "旧 anchor_tags.yaml 不再作为有效源读取")

        # 触发一次标准写穿，验证残留的孤立旧文件被物理彻底清除
        save_workspace_tag_config(self.ws, anchor_tags={1: {"xyz_mm": [0, 520, None]}})
        self.assertFalse(os.path.exists(old_anchor_file), "独立的 anchor_tags.yaml 必须被物理清理彻底删除！")
        self.assertTrue(os.path.exists(self.ws.whitelist_path))

    def test_all_workspaces_schema_v2_purity(self):
        """扫描现存真实工位目录，确保 100% 符合方案 B 纯净契约，绝无孤立 anchor_tags.yaml"""
        ws_root = os.path.join(os.path.dirname(__file__), "..", "data", "workspaces")
        if not os.path.isdir(ws_root):
            return

        for ws_name in os.listdir(ws_root):
            ws_path = os.path.join(ws_root, ws_name)
            if not os.path.isdir(ws_path):
                continue
            # 1. 验证不存在独立的 anchor_tags.yaml
            stray_anchor_file = os.path.join(ws_path, "anchor_tags.yaml")
            self.assertFalse(
                os.path.exists(stray_anchor_file),
                f"工位 [{ws_name}] 仍残留独立的 anchor_tags.yaml，违反方案 B 单真理源契约！"
            )

            # 2. 验证 tag_whitelist.yaml 格式规范
            wl_file = os.path.join(ws_path, "tag_whitelist.yaml")
            if os.path.exists(wl_file):
                with open(wl_file, "r", encoding="utf-8") as f:
                    txt = f.read()
                    doc = yaml.safe_load(txt) or {}
                self.assertNotIn("frame_id", txt, f"工位 [{ws_name}] 的 tag_whitelist.yaml 包含 frame_id 补丁！")
                self.assertNotIn("tag_anchors", doc, f"工位 [{ws_name}] 仍包含旧 tag_anchors 冗余字段")
                self.assertIn("allowed_ids", doc)
                self.assertIn("anchor_tags", doc)


if __name__ == "__main__":
    unittest.main()
