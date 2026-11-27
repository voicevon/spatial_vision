# -*- coding: utf-8 -*-
"""
空间平差优化、外参对齐与单目内参求解器包 (Bundle Adjustment, Extrinsic & Intrinsics Solvers)
"""

from .camera_intrinsics_calibrator import (
    IntrinsicsCalibrationResult,
    calibrate_camera_from_images,
    calibrate_camera_profile,
    detect_chessboard_corners,
)
from .chessboard_guide import (
    GuidePose,
    PoseAlignmentFeedback,
    ProjectedGuideMesh,
    default_guide_poses,
    evaluate_alignment,
    extract_detected_quad_norm,
    project_guide_mesh,
)

__all__ = [
    "IntrinsicsCalibrationResult",
    "calibrate_camera_from_images",
    "calibrate_camera_profile",
    "detect_chessboard_corners",
    "GuidePose",
    "PoseAlignmentFeedback",
    "ProjectedGuideMesh",
    "default_guide_poses",
    "evaluate_alignment",
    "extract_detected_quad_norm",
    "project_guide_mesh",
]
