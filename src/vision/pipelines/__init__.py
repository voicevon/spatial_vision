"""
芦笋感知算法流水线模块初始化
"""

from src.vision.pipelines.base_pipeline import BaseAsparagusPipeline, PipelineResult, PipelineStep
from src.vision.pipelines.registry import PipelineRegistry

# 显式导入各算法流水线以触发自动注册 (导入顺序 = GUI 下拉显示顺序)
# F1 置于首位：冯氏寻找法为默认首选路线；F2 (冯氏二代) 紧随其后
import src.vision.pipelines.f1_green_medial_axis_pipeline
import src.vision.pipelines.f2_exgr_watershed_pipeline
import src.vision.pipelines.a_ridge_tracing_pipeline
import src.vision.pipelines.b1_polarity_scanline_pipeline
import src.vision.pipelines.b2_segment_topology_pipeline
import src.vision.pipelines.c1_skeleton_thinning_pipeline
import src.vision.pipelines.c2_frangi_vesselness_pipeline
import src.vision.pipelines.y1_stub_pipeline

# H 系列：非机器学习经典算法流水线
import src.vision.pipelines.h1_sato_pipeline
import src.vision.pipelines.h2_skeleton_pipeline
import src.vision.pipelines.h3_fast_marching_pipeline
import src.vision.pipelines.h4_graph_tracing_pipeline
import src.vision.pipelines.h5_phase_congruency_pipeline
import src.vision.pipelines.h6_steerable_pipeline

__all__ = [
    "BaseAsparagusPipeline",
    "PipelineResult",
    "PipelineStep",
    "PipelineRegistry",
]
