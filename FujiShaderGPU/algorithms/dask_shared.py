"""
FujiShaderGPU/algorithms/dask_shared.py

後方互換性のための再エクスポートハブ。
全アルゴリズム実装は個別の _impl_*.py モジュールに分離済み。

Phase 1: 共通基盤 (_base, _nan_utils, _global_stats, _normalization)
Phase 2: 大型アルゴリズム6種 (_impl_rvi, _impl_npr_edges,
         _impl_ambient_occlusion, _impl_openness, _impl_specular,
         _impl_fractal_anomaly)
Phase 3: 残留アルゴリズム8種 (_impl_hillshade, _impl_slope,
         _impl_visual_saliency, _impl_lrm, _impl_atmospheric_scattering,
         _impl_multiscale_terrain,
         _impl_curvature, _impl_experimental)
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple  # noqa: F401
from abc import ABC, abstractmethod  # noqa: F401
import cupy as cp  # noqa: F401
import numpy as np  # noqa: F401
import dask.array as da  # noqa: F401

# ---------------------------------------------------------------------------
# Phase 1: 共通基盤の再エクスポート
# ---------------------------------------------------------------------------

# --- _base.py ---
from ._base import (
    Constants,
    DaskAlgorithm,
    classify_resolution,
    get_gradient_scale_factor,
)

# --- _nan_utils.py ---
from ._nan_utils import (
    handle_nan_with_gaussian,
    handle_nan_with_uniform,
    handle_nan_for_gradient,
    _normalize_spatial_radii,
    _resolve_spatial_radii_weights,
    _combine_multiscale_dask,
    _smooth_for_radius,
    _radius_to_downsample_factor,
    _downsample_nan_aware,
    _upsample_to_shape,
    restore_nan,
)

# --- _global_stats.py ---
from ._global_stats import (
    determine_optimal_downsample_factor,
    compute_global_stats,
    apply_global_normalization,
)

# --- _normalization.py ---
from ._normalization import (
    rvi_stat_func,
    rvi_norm_func,
    npr_stat_func,
    lrm_stat_func,
    tpi_norm_func,
)

# ---------------------------------------------------------------------------
# Phase 2: 大型アルゴリズムの再エクスポート
# ---------------------------------------------------------------------------

# --- _impl_rvi.py ---
from ._impl_rvi import (
    high_pass, compute_rvi_efficient_block, multiscale_rvi, RVIAlgorithm,
)

# --- _impl_npr_edges.py ---
from ._impl_npr_edges import compute_npr_edges_block, NPREdgesAlgorithm

# --- _impl_ambient_occlusion.py ---
from ._impl_ambient_occlusion import (
    compute_ambient_occlusion_block, compute_ambient_occlusion_spatial_block,
    AmbientOcclusionAlgorithm,
)

# --- _impl_openness.py ---
from ._impl_openness import (
    compute_openness_vectorized, compute_openness_spatial_block,
    OpennessAlgorithm,
)

# --- _impl_specular.py ---
from ._impl_specular import (
    compute_specular_block, compute_specular_spatial_block,
    SpecularAlgorithm,
)

# --- _impl_fractal_anomaly.py ---
from ._impl_fractal_anomaly import (
    compute_roughness_multiscale, compute_fractal_dimension_block,
    fractal_stat_func, FractalAnomalyAlgorithm,
)

# ---------------------------------------------------------------------------
# Phase 3: 残留アルゴリズムの再エクスポート
# ---------------------------------------------------------------------------

# --- _impl_hillshade.py ---
from ._impl_hillshade import (
    compute_hillshade_block, compute_hillshade_spatial_block,
    HillshadeAlgorithm,
)

# --- _impl_slope.py ---
from ._impl_slope import (
    compute_slope_block, compute_slope_spatial_block,
    SlopeAlgorithm,
)

# --- _impl_visual_saliency.py ---
from ._impl_visual_saliency import (
    _compress_saliency_feature, visual_saliency_stat_func,
    compute_visual_saliency_block, VisualSaliencyAlgorithm,
)

# --- _impl_lrm.py ---
from ._impl_lrm import compute_lrm_block, LRMAlgorithm

# --- _impl_atmospheric_scattering.py ---
from ._impl_atmospheric_scattering import (
    compute_atmospheric_scattering_block,
    compute_atmospheric_scattering_spatial_block,
    AtmosphericScatteringAlgorithm,
)

# --- _impl_multiscale_terrain.py ---
from ._impl_multiscale_terrain import MultiscaleDaskAlgorithm

# --- _impl_curvature.py ---
from ._impl_curvature import compute_curvature_block, CurvatureAlgorithm

# --- _impl_experimental.py ---
from ._impl_experimental import (
    compute_scale_space_surprise_block, ScaleSpaceSurpriseAlgorithm,
    compute_multi_light_uncertainty_block,
    compute_multi_light_uncertainty_spatial_block,
    MultiLightUncertaintyAlgorithm,
)

# ---------------------------------------------------------------------------
# アルゴリズムレジストリ
# ---------------------------------------------------------------------------

ALGORITHMS = {
    'rvi': RVIAlgorithm(),
    'hillshade': HillshadeAlgorithm(),
    'slope': SlopeAlgorithm(),
    'specular': SpecularAlgorithm(),
    'atmospheric_scattering': AtmosphericScatteringAlgorithm(),
    'multiscale_terrain': MultiscaleDaskAlgorithm(),
    'curvature': CurvatureAlgorithm(),
    'visual_saliency': VisualSaliencyAlgorithm(),
    'npr_edges': NPREdgesAlgorithm(),
    'ambient_occlusion': AmbientOcclusionAlgorithm(),
    'lrm': LRMAlgorithm(),
    'openness': OpennessAlgorithm(),
    'fractal_anomaly': FractalAnomalyAlgorithm(),
    'scale_space_surprise': ScaleSpaceSurpriseAlgorithm(),
    'multi_light_uncertainty': MultiLightUncertaintyAlgorithm(),
}
