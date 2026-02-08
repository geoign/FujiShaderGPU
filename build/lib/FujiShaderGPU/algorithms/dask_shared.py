"""
FujiShaderGPU/algorithms/dask_shared.py
"""
from __future__ import annotations
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import cupy as cp
import numpy as np
import dask.array as da
from cupyx.scipy.ndimage import gaussian_filter, uniform_filter, maximum_filter, minimum_filter, convolve, binary_dilation, median_filter
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)
from .common.spatial_mode import determine_spatial_radii

class Constants:
    DEFAULT_GAMMA = 1/2.2
    DEFAULT_AZIMUTH = 315
    DEFAULT_ALTITUDE = 45
    MAX_DEPTH = 150
    NAN_FILL_VALUE_POSITIVE = -1e6
    NAN_FILL_VALUE_NEGATIVE = 1e6

# 1. 繧医ｊ隧ｳ邏ｰ縺ｪ隗｣蜒丞ｺｦ蛻・｡樣未謨ｰ・域里蟄倥・classify_resolution繧堤ｽｮ縺肴鋤縺茨ｼ・
def classify_resolution(pixel_size: float) -> str:
    """
    隗｣蜒丞ｺｦ繧貞・鬘橸ｼ医ｈ繧願ｩｳ邏ｰ縺ｪ蛻・｡橸ｼ・
    Returns: 'ultra_high', 'very_high', 'high', 'medium', 'low', 'very_low', 'ultra_low'
    """
    if pixel_size <= 0.5:
        return 'ultra_high'
    elif pixel_size <= 1.0:
        return 'very_high'
    elif pixel_size <= 2.5:
        return 'high'
    elif pixel_size <= 5.0:
        return 'medium'
    elif pixel_size <= 15.0:
        return 'low'
    elif pixel_size <= 30.0:
        return 'very_low'
    else:
        return 'ultra_low'

# 2. 隗｣蜒丞ｺｦ縺ｫ蠢懊§縺溷鏡驟阪せ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ菫よ焚繧定ｨ育ｮ励☆繧矩未謨ｰ・域眠隕剰ｿｽ蜉・・
def get_gradient_scale_factor(pixel_size: float, algorithm: str = 'default') -> float:
    """
    隗｣蜒丞ｺｦ縺ｫ蠢懊§縺溷鏡驟阪せ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ菫よ焚繧定ｿ斐☆
    菴手ｧ｣蜒丞ｺｦ縺ｻ縺ｩ螟ｧ縺阪↑菫よ焚繧定ｿ斐＠縲∝鏡驟阪ｒ陬懈ｭ｣縺吶ｋ
    """
    if algorithm == 'npr_edges':
        # NPR繧ｨ繝・ず逕ｨ縺ｮ菫よ焚・医ｈ繧顔ｩ肴･ｵ逧・↑繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ・・
        if pixel_size <= 1.0:
            return 1.0
        elif pixel_size <= 5.0:
            return 1.5
        elif pixel_size <= 10.0:
            return 2.5
        elif pixel_size <= 30.0:
            return 4.0
        else:
            return 6.0
    elif algorithm == 'visual_saliency':
        # Visual Saliency逕ｨ縺ｮ菫よ焚・医ｈ繧頑而縺医ａ縺ｪ繧ｹ繧ｱ繝ｼ繝ｪ繝ｳ繧ｰ・・
        if pixel_size <= 1.0:
            return 1.0
        elif pixel_size <= 5.0:
            return 1.2
        elif pixel_size <= 10.0:
            return 1.5
        elif pixel_size <= 30.0:
            return 2.0
        else:
            return 2.5
    else:
        # 繝・ヵ繧ｩ繝ｫ繝医・菫よ焚
        return cp.sqrt(max(1.0, pixel_size))
    
###############################################################################
# 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝蝓ｺ蠎輔け繝ｩ繧ｹ縺ｨ蜈ｱ騾壹う繝ｳ繧ｿ繝ｼ繝輔ぉ繝ｼ繧ｹ
###############################################################################

class DaskAlgorithm(ABC):
    """蝨ｰ蠖｢隗｣譫舌い繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ蝓ｺ蠎輔け繝ｩ繧ｹ"""
    
    @abstractmethod
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        """繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ繝｡繧､繝ｳ蜃ｦ逅・"""
        pass
    
    @abstractmethod
    def get_default_params(self) -> dict:
        """繝・ヵ繧ｩ繝ｫ繝医ヱ繝ｩ繝｡繝ｼ繧ｿ繧定ｿ斐☆"""
        pass

###############################################################################
# NaN蜃ｦ逅・・繝ｦ繝ｼ繝・ぅ繝ｪ繝・ぅ髢｢謨ｰ
###############################################################################

def handle_nan_with_gaussian(block: cp.ndarray, sigma: float, mode: str = 'nearest') -> Tuple[cp.ndarray, cp.ndarray]:
    """NaN繧定・・縺励◆繧ｬ繧ｦ繧ｷ繧｢繝ｳ繝輔ぅ繝ｫ繧ｿ蜃ｦ逅・"""
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return gaussian_filter(block, sigma=sigma, mode=mode), nan_mask
    
    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)
    
    smoothed_values = gaussian_filter(filled * valid, sigma=sigma, mode=mode)
    smoothed_weights = gaussian_filter(valid, sigma=sigma, mode=mode)
    smoothed = cp.where(smoothed_weights > 0, smoothed_values / smoothed_weights, 0)
    
    return smoothed, nan_mask

def handle_nan_with_uniform(block: cp.ndarray, size: int, mode: str = 'nearest') -> Tuple[cp.ndarray, cp.ndarray]:
    """NaN繧定・・縺励◆uniform_filter蜃ｦ逅・"""
    nan_mask = cp.isnan(block)
    if not nan_mask.any():
        return uniform_filter(block, size=size, mode=mode), nan_mask
    
    filled = cp.where(nan_mask, 0, block)
    valid = (~nan_mask).astype(cp.float32)
    
    sum_values = uniform_filter(filled * valid, size=size, mode=mode)
    sum_weights = uniform_filter(valid, size=size, mode=mode)
    mean = cp.where(sum_weights > 0, sum_values / sum_weights, 0)
    
    return mean, nan_mask

def handle_nan_for_gradient(block: cp.ndarray, scale: float = 1.0, 
                          pixel_size: float = 1.0) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """NaN繧定・・縺励◆蜍ｾ驟崎ｨ育ｮ・"""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block
    
    dy, dx = cp.gradient(filled * scale, pixel_size, edge_order=2)
    return dy, dx, nan_mask


def _normalize_spatial_radii(radii: Optional[List[int]], pixel_size: float) -> List[int]:
    """Normalize user-provided radii or auto-derive stable defaults."""
    if radii is None:
        return determine_spatial_radii(pixel_size=pixel_size)
    out: List[int] = []
    for r in radii:
        try:
            rv = int(round(float(r)))
        except (TypeError, ValueError):
            continue
        if rv > 0:
            out.append(rv)
    if not out:
        return determine_spatial_radii(pixel_size=pixel_size)
    # Keep user order while dropping duplicates.
    seen = set()
    ordered = []
    for v in out:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def _combine_multiscale_dask(
    responses: List[da.Array],
    *,
    weights: Optional[List[float]] = None,
    agg: str = "mean",
) -> da.Array:
    """Combine per-radius dask responses with optional weighted mean."""
    if not responses:
        raise ValueError("responses must not be empty")
    if len(responses) == 1:
        return responses[0]

    stacked = da.stack(responses, axis=0)
    agg_norm = str(agg or "mean").lower()
    if agg_norm == "stack":
        return stacked
    if agg_norm == "max":
        return da.max(stacked, axis=0)
    if agg_norm == "min":
        return da.min(stacked, axis=0)
    if agg_norm == "sum":
        return da.sum(stacked, axis=0)

    if isinstance(weights, (list, tuple)) and len(weights) == len(responses):
        w = np.asarray(weights, dtype=np.float32)
        if np.isfinite(w).all() and w.sum() > 0:
            w = w / w.sum()
            out = responses[0] * float(w[0])
            for i in range(1, len(responses)):
                out = out + responses[i] * float(w[i])
            return out
    return da.mean(stacked, axis=0)


def _smooth_for_radius(block: cp.ndarray, radius: float) -> cp.ndarray:
    """NaN-aware gaussian smoothing controlled by spatial radius."""
    r = max(1.0, float(radius))
    if r <= 1.0:
        return block
    sigma = max(0.5, r / 2.0)
    smoothed, _ = handle_nan_with_gaussian(block, sigma=sigma, mode="nearest")
    return smoothed
    
def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    """NaN菴咲ｽｮ繧貞ｾｩ蜈・"""
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result

###############################################################################
# 繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險医Θ繝ｼ繝・ぅ繝ｪ繝・ぅ
###############################################################################
def determine_optimal_downsample_factor(
    data_shape: Tuple[int, int],
    algorithm_name: str = None,
    target_pixels: int = 500000,  # 逶ｮ讓吶ヴ繧ｯ繧ｻ繝ｫ謨ｰ・・000x1000・・
    min_factor: int = 5,
    max_factor: int = 100,
    algorithm_complexity: Dict[str, float] = None) -> int:
    """
    繝・・繧ｿ繧ｵ繧､繧ｺ縺ｨ繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ迚ｹ諤ｧ縺ｫ蝓ｺ縺･縺・※譛驕ｩ縺ｪ繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚繧呈ｱｺ螳・
    
    Parameters:
    -----------
    data_shape : Tuple[int, int]
        蜈･蜉帙ョ繝ｼ繧ｿ縺ｮ蠖｢迥ｶ (height, width)
    algorithm_name : str
        繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝蜷搾ｼ郁､・尅蠎ｦ縺ｮ隱ｿ謨ｴ逕ｨ・・
    target_pixels : int
        繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν蠕後・逶ｮ讓吶ヴ繧ｯ繧ｻ繝ｫ謨ｰ
    min_factor : int
        譛蟆上ム繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚
    max_factor : int
        譛螟ｧ繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚
    algorithm_complexity : Dict[str, float]
        繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺斐→縺ｮ隍・尅蠎ｦ菫よ焚・医ョ繝輔か繝ｫ繝医・蜀・鳩霎樊嶌・・
    
    Returns:
    --------
    int : 譛驕ｩ縺ｪ繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚
    """
    # 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ隍・尅蠎ｦ菫よ焚・郁ｨ育ｮ励さ繧ｹ繝医′鬮倥＞縺ｻ縺ｩ螟ｧ縺阪＞蛟､・・
    if algorithm_complexity is None:
        algorithm_complexity = {
            'rvi': 1.2,                    # 繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ蜃ｦ逅・
            'hillshade': 0.8,              # 蜊倡ｴ斐↑蜍ｾ驟崎ｨ育ｮ・
            'slope': 0.8,                  # 蜊倡ｴ斐↑蜍ｾ驟崎ｨ育ｮ・
            'specular': 1.5,               # 繝ｩ繝輔ロ繧ｹ險育ｮ励′驥阪＞
            'atmospheric_scattering': 0.9,
            'multiscale_terrain': 1.5,     # 繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ蜃ｦ逅・
            'frequency_enhancement': 1.3,   # FFT蜃ｦ逅・
            'curvature': 1.0,              # 2谺｡蠕ｮ蛻・
            'visual_saliency': 1.4,        # 繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ迚ｹ蠕ｴ謚ｽ蜃ｺ
            'npr_edges': 1.1,              # 繧ｨ繝・ず讀懷・
            'ambient_occlusion': 2.0,      # 譛繧りｨ育ｮ励さ繧ｹ繝医′鬮倥＞
            'lrm': 1.1,                    # 繧ｬ繧ｦ繧ｷ繧｢繝ｳ繝輔ぅ繝ｫ繧ｿ
            'openness': 1.8,               # 螟壽婿蜷第爾邏｢
            'fractal_anomaly': 1.6,        # 繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ蝗槫ｸｰ險育ｮ・
        }
    
    # 迴ｾ蝨ｨ縺ｮ繝斐け繧ｻ繝ｫ謨ｰ
    current_pixels = data_shape[0] * data_shape[1]
    
    # 蝓ｺ譛ｬ縺ｮ繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚・亥ｹｳ譁ｹ譬ｹ縺ｧ險育ｮ暦ｼ・
    base_factor = cp.sqrt(current_pixels / target_pixels).get()
    
    # 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ隍・尅蠎ｦ縺ｧ隱ｿ謨ｴ
    complexity = algorithm_complexity.get(algorithm_name, 1.0)
    adjusted_factor = base_factor * complexity
    
    # 謨ｴ謨ｰ蛹悶＠縺ｦ遽・峇蜀・↓蜿弱ａ繧・
    downsample_factor = int(cp.clip(adjusted_factor, min_factor, max_factor))
    
    # 繝・・繧ｿ縺悟ｰ上＆縺・ｴ蜷医・菫よ焚繧貞ｰ上＆縺上☆繧・
    if current_pixels < 1_000_000:  # 1M繝斐け繧ｻ繝ｫ譛ｪ貅
        downsample_factor = min(downsample_factor, 2)
    elif current_pixels < 10_000_000:  # 10M繝斐け繧ｻ繝ｫ譛ｪ貅
        downsample_factor = min(downsample_factor, 4)
    return downsample_factor

def compute_global_stats(gpu_arr: da.Array, 
                        stat_func: callable,
                        algorithm_func: callable,
                        algorithm_params: dict,
                        downsample_factor: int = None,  # None縺ｮ蝣ｴ蜷医・閾ｪ蜍墓ｱｺ螳・
                        depth: int = None,
                        algorithm_name: str = None) -> Tuple[Any, ...]:
    """
    繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ縺励◆繝・・繧ｿ縺ｧ邨ｱ險磯㍼繧定ｨ育ｮ励☆繧句・騾夐未謨ｰ
    
    Parameters:
    -----------
    gpu_arr : da.Array
        蜈･蜉帙ョ繝ｼ繧ｿ
    stat_func : callable
        邨ｱ險磯㍼繧定ｨ育ｮ励☆繧矩未謨ｰ縲・uPy驟榊・繧貞女縺大叙繧翫∫ｵｱ險磯㍼縺ｮ繧ｿ繝励Ν繧定ｿ斐☆
    algorithm_func : callable
        繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ蜃ｦ逅・未謨ｰ・域ｭ｣隕丞喧縺ｪ縺励ヰ繝ｼ繧ｸ繝ｧ繝ｳ・・
    algorithm_params : dict
        繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ繝代Λ繝｡繝ｼ繧ｿ
    downsample_factor : int
        繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ菫よ焚
    depth : int
        map_overlap縺ｮdepth・・one縺ｮ蝣ｴ蜷医・閾ｪ蜍戊ｨ育ｮ暦ｼ・
    
    Returns:
    --------
    邨ｱ險磯㍼縺ｮ繧ｿ繝励Ν
    """
    # downsample_factor縺梧欠螳壹＆繧後※縺・↑縺・ｴ蜷医・閾ｪ蜍墓ｱｺ螳・
    if downsample_factor is None:
        downsample_factor = determine_optimal_downsample_factor(
            gpu_arr.shape,
            algorithm_name=algorithm_name
        )

    # 繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ
    downsampled = gpu_arr[::downsample_factor, ::downsample_factor]
    
    # depth縺ｮ隱ｿ謨ｴ
    if depth is not None:
        depth_small = max(1, depth // downsample_factor)
    else:
        depth_small = 1
    
    # 繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν迚医〒繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝繧貞ｮ溯｡鯉ｼ域ｭ｣隕丞喧縺ｪ縺暦ｼ・
    params_small = algorithm_params.copy()
    
    # 繧ｹ繧ｱ繝ｼ繝ｫ邉ｻ繝代Λ繝｡繝ｼ繧ｿ縺ｮ隱ｿ謨ｴ縺悟ｿ・ｦ√↑蝣ｴ蜷・
    for key in ['scale', 'sigma', 'radius', 'kernel_size']:
        if key in params_small and params_small[key] is not None:
            if isinstance(params_small[key], list):
                if key in ['radius', 'kernel_size']:
                    params_small[key] = [max(1, int(s/downsample_factor)) for s in params_small[key]]
                else:
                    params_small[key] = [max(1, s/downsample_factor) for s in params_small[key]]
            else:
                if key in ['radius', 'kernel_size']:
                    params_small[key] = max(1, int(params_small[key]/downsample_factor))
                else:
                    params_small[key] = max(1, params_small[key]/downsample_factor)
    
    result_small = downsampled.map_overlap(
        algorithm_func,
        depth=depth_small,
        boundary='reflect',
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        **params_small
    ).compute()
    
    # 邨ｱ險磯㍼繧定ｨ育ｮ・
    stats = stat_func(result_small)

    return stats

def apply_global_normalization(block: cp.ndarray, 
                              norm_func: callable,
                              stats: Tuple[Any, ...],
                              nan_mask: cp.ndarray = None) -> cp.ndarray:
    """
    繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險磯㍼繧剃ｽｿ縺｣縺ｦ豁｣隕丞喧繧帝←逕ｨ縺吶ｋ蜈ｱ騾夐未謨ｰ
    
    Parameters:
    -----------
    block : cp.ndarray
        蜃ｦ逅・☆繧九ヶ繝ｭ繝・け
    norm_func : callable
        豁｣隕丞喧髢｢謨ｰ縲・block, stats, nan_mask)繧貞女縺大叙繧翫∵ｭ｣隕丞喧縺輔ｌ縺溘ヶ繝ｭ繝・け繧定ｿ斐☆
    stats : tuple
        繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險磯㍼
    nan_mask : cp.ndarray
        NaN繝槭せ繧ｯ・医が繝励す繝ｧ繝ｳ・・
    
    Returns:
    --------
    豁｣隕丞喧縺輔ｌ縺溘ヶ繝ｭ繝・け
    """
    if nan_mask is None:
        nan_mask = cp.isnan(block)
    
    normalized = norm_func(block, stats, nan_mask)
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣・・-1縺ｮ遽・峇縺ｮ蝣ｴ蜷医・縺ｿ・・
    valid_normalized = normalized[~nan_mask]
    if len(valid_normalized) > 0 and cp.min(valid_normalized) >= 0 and cp.max(valid_normalized) <= 1:
        normalized = cp.power(normalized, Constants.DEFAULT_GAMMA)
    
    # NaN菴咲ｽｮ繧貞ｾｩ蜈・
    normalized = restore_nan(normalized, nan_mask)
    
    return normalized.astype(cp.float32)


###############################################################################
# 蜷・い繝ｫ繧ｴ繝ｪ繧ｺ繝逕ｨ縺ｮ邨ｱ險医・豁｣隕丞喧髢｢謨ｰ
###############################################################################

# RVI逕ｨ
def rvi_stat_func(data: cp.ndarray) -> Tuple[float]:
    """RVI normalization scale from robust absolute percentile."""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        abs_valid = cp.abs(valid_data)
        scale = float(cp.percentile(abs_valid, 80))
        if scale > 1e-9:
            return (scale,)
        # Fallback for near-constant tiles/arrays.
        return (float(cp.std(valid_data)) if float(cp.std(valid_data)) > 1e-9 else 1.0,)
    return (1.0,)

def rvi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """RVI逕ｨ縺ｮ豁｣隕丞喧"""
    scale_global = stats[0]
    if scale_global > 0:
        normalized = block / scale_global
        return cp.clip(normalized, -1, 1)
    return cp.zeros_like(block)

# FrequencyEnhancement逕ｨ
def freq_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """蜻ｨ豕｢謨ｰ蠑ｷ隱ｿ逕ｨ縺ｮ邨ｱ險磯㍼險育ｮ暦ｼ域怙蟆丞､繝ｻ譛螟ｧ蛟､・・"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        return (float(cp.min(valid_data)), float(cp.max(valid_data)))
    return (0.0, 1.0)

def freq_norm_func(block: cp.ndarray, stats: Tuple[float, float], nan_mask: cp.ndarray) -> cp.ndarray:
    """蜻ｨ豕｢謨ｰ蠑ｷ隱ｿ逕ｨ縺ｮ豁｣隕丞喧"""
    min_val, max_val = stats
    if max_val > min_val:
        return (block - min_val) / (max_val - min_val)
    return cp.full_like(block, 0.5)

# NPREdges逕ｨ
def npr_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """NPR繧ｨ繝・ず逕ｨ縺ｮ邨ｱ險磯㍼險育ｮ暦ｼ亥鏡驟阪・繝代・繧ｻ繝ｳ繧ｿ繧､繝ｫ・・"""
    # 邁｡譏鍋噪縺ｫ蜍ｾ驟阪ｒ險育ｮ・
    dy, dx = cp.gradient(data)
    gradient_mag = cp.sqrt(dx**2 + dy**2)
    valid_grad = gradient_mag[~cp.isnan(gradient_mag)]
    
    if len(valid_grad) > 0:
        return (float(cp.percentile(valid_grad, 70)), 
                float(cp.percentile(valid_grad, 90)))
    return (0.1, 0.3)

def lrm_stat_func(data: cp.ndarray) -> Tuple[float]:
    """LRM??????????MAD????"""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) == 0:
        return (1.0,)

    med = cp.median(valid_data)
    abs_dev = cp.abs(valid_data - med)
    mad = float(cp.median(abs_dev))
    if mad > 1e-9:
        return (1.4826 * mad,)

    p90 = float(cp.percentile(abs_dev, 90))
    if p90 > 1e-9:
        return (p90,)
    return (1.0,)

def tpi_norm_func(block: cp.ndarray, stats: Tuple[float], nan_mask: cp.ndarray) -> cp.ndarray:
    """TPI/LRM逕ｨ縺ｮ豁｣隕丞喧"""
    max_abs = stats[0]
    if max_abs > 0:
        return cp.clip(block / max_abs, -1, 1)
    return cp.zeros_like(block)

###############################################################################
# 2.1. RVI (Ridge-Valley Index) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def high_pass(block: cp.ndarray, *, sigma: float) -> cp.ndarray:
    """CuPy 縺ｧ繧ｬ繧ｦ繧ｷ繧｢繝ｳ縺ｼ縺九＠蠕後↓蟾ｮ蛻・ｒ蜿悶ｋ鬮倪叢ass 繝輔ぅ繝ｫ繧ｿ・・aN蟇ｾ蠢懶ｼ・"""
    # NaN繝槭せ繧ｯ蜃ｦ逅・
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        # NaN繧貞捉蝗ｲ縺ｮ蛟､縺ｧ蝓九ａ繧具ｼ井ｸ譎ら噪縺ｫ・・
        filled = cp.where(nan_mask, 0, block)
        # 譛牙柑縺ｪ繝斐け繧ｻ繝ｫ縺ｮ繝槭せ繧ｯ繧ゆｽ懈・
        valid_mask = (~nan_mask).astype(cp.float32)
        
        # 繧ｬ繧ｦ繧ｷ繧｢繝ｳ繝輔ぅ繝ｫ繧ｿ繧貞､縺ｨ譛牙柑繝槭せ繧ｯ縺ｮ荳｡譁ｹ縺ｫ驕ｩ逕ｨ
        blurred_values = gaussian_filter(filled * valid_mask, sigma=sigma, mode="nearest", truncate=4.0)
        blurred_weights = gaussian_filter(valid_mask, sigma=sigma, mode="nearest", truncate=4.0)
        
        # 驥阪∩莉倥″蟷ｳ蝮・〒NaN鬆伜沺繧定・・縺励◆縺ｼ縺九＠
        blurred = cp.where(blurred_weights > 0, blurred_values / blurred_weights, 0)
    else:
        blurred = gaussian_filter(block, sigma=sigma, mode="nearest", truncate=4.0)
    
    result = block - blurred
    
    # NaN菴咲ｽｮ繧貞ｾｩ蜈・
    result = restore_nan(result, nan_mask)
        
    return result

def compute_rvi_efficient_block(block: cp.ndarray, *, 
                               radii: List[int] = [4, 16, 64], 
                               weights: Optional[List[float]] = None) -> cp.ndarray:
    """蜉ｹ邇・噪縺ｪRVI險育ｮ暦ｼ医Γ繝｢繝ｪ譛驕ｩ蛹也沿・・"""
    nan_mask = cp.isnan(block)
    
    if weights is None:
        weights = cp.array([1.0 / len(radii)] * len(radii), dtype=cp.float32)
    else:
        if not isinstance(weights, cp.ndarray):
            weights = cp.array(weights, dtype=cp.float32)
        if len(weights) != len(radii):
            raise ValueError(f"Length of weights ({len(weights)}) must match length of radii ({len(radii)})")
    
    # 邨先棡繧偵う繝ｳ繝励Ξ繝ｼ繧ｹ縺ｧ邏ｯ遨搾ｼ医Γ繝｢繝ｪ蜉ｹ邇・髄荳奇ｼ・
    rvi_combined = None
    
    for i, (radius, weight) in enumerate(zip(radii, weights)):
        if radius <= 1:
            mean_elev, _ = handle_nan_with_gaussian(block, sigma=1.0, mode='nearest')
        else:
            kernel_size = 2 * radius + 1
            mean_elev, _ = handle_nan_with_uniform(block, size=kernel_size, mode='reflect')
        
        # 蟾ｮ蛻・ｒ險育ｮ・
        diff = weight * (block - mean_elev)
        
        if rvi_combined is None:
            rvi_combined = diff
        else:
            rvi_combined += diff
        
        # 繝｡繝｢繝ｪ繧貞叉蠎ｧ縺ｫ隗｣謾ｾ
        del mean_elev, diff
    
    # NaN蜃ｦ逅・
    rvi_combined = restore_nan(rvi_combined, nan_mask)
    
    return rvi_combined


def multiscale_rvi(gpu_arr: da.Array, *, 
                   radii: List[int], 
                   weights: Optional[List[float]] = None) -> da.Array:
    """蜉ｹ邇・噪縺ｪ繝槭Ν繝√せ繧ｱ繝ｼ繝ｫRVI・・ask迚茨ｼ・"""
    
    if not radii:
        raise ValueError("At least one radius value is required")
    
    # 譛螟ｧ蜊雁ｾ・↓蝓ｺ縺･縺・※depth繧定ｨｭ螳夲ｼ・aussian繧医ｊ繧ょ､ｧ蟷・↓蟆上＆縺・ｼ・
    max_radius = max(radii)
    depth = max_radius * 2 + 1  # 蜊雁ｾ・・2蛟・1縺ｫ螟画峩
    
    # 蜊倅ｸ縺ｮmap_overlap縺ｧ蜈ｨ繧ｹ繧ｱ繝ｼ繝ｫ繧定ｨ育ｮ暦ｼ亥柑邇・噪・・
    result = gpu_arr.map_overlap(
        compute_rvi_efficient_block,
        depth=depth,
        boundary="reflect",
        dtype=cp.float32,
        meta=cp.empty((0, 0), dtype=cp.float32),
        radii=radii,
        weights=weights
    )
    
    return result


class RVIAlgorithm(DaskAlgorithm):
    """Ridge-Valley Index繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝・亥柑邇・噪螳溯｣・ｼ・"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:        
        # 譁ｰ縺励＞蜉ｹ邇・噪縺ｪ蜊雁ｾ・・繝ｼ繧ｹ
        radii = params.get('radii', None)
        weights = params.get('weights', None)
        
        # 閾ｪ蜍墓ｱｺ螳・
        if radii is None:
            pixel_size = params.get('pixel_size', 1.0)
            radii = self._determine_optimal_radii(pixel_size)
        max_radius = max(radii)
        rvi = multiscale_rvi(gpu_arr, radii=radii, weights=weights)
        
        # Prefer externally supplied global stats (tile backend computes once).
        stats = params.get("global_stats", None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) >= 1
            and float(stats[0]) > 1e-9
        )
        if not stats_ok:
            num_blocks = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if num_blocks > 1:
                stats = compute_global_stats(
                    rvi,
                    rvi_stat_func,
                    compute_rvi_efficient_block,
                    {'radii': radii, 'weights': weights},
                    params.get('downsample_factor', None),
                    depth=max_radius * 2 + 1
                )
            else:
                # Single-tile fallback; avoids per-tile stat variability seams.
                stats = (1.0,)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 1 and float(stats[0]) > 1e-9):
            stats = (1.0,)
        
        # 豁｣隕丞喧繧帝←逕ｨ
        return rvi.map_blocks(
            lambda block: apply_global_normalization(block, rvi_norm_func, stats),
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
    
    def _determine_optimal_radii(self, pixel_size: float) -> List[int]:
        """繝斐け繧ｻ繝ｫ繧ｵ繧､繧ｺ縺ｫ蝓ｺ縺･縺・※譛驕ｩ縺ｪ蜊雁ｾ・ｒ豎ｺ螳・"""
        # 螳滉ｸ也阜縺ｮ霍晞屬・医Γ繝ｼ繝医Ν・峨ｒ繝斐け繧ｻ繝ｫ縺ｫ螟画鋤
        target_distances = [5, 20, 80, 320]  # 繝｡繝ｼ繝医Ν蜊倅ｽ・
        radii = []
        
        for dist in target_distances:
            radius = int(dist / pixel_size)
            # 迴ｾ螳溽噪縺ｪ遽・峇縺ｫ蛻ｶ髯・
            radius = max(2, min(radius, 256))
            radii.append(radius)
        
        # 驥崎､・ｒ蜑企勁縺励※繧ｽ繝ｼ繝・
        radii = sorted(list(set(radii)))
        
        # 譛螟ｧ4縺､縺ｾ縺ｧ縺ｫ蛻ｶ髯・
        if len(radii) > 4:
            # 蟇ｾ謨ｰ逧・↓蛻・ｸ・☆繧九ｈ縺・↓驕ｸ謚・
            # NumPy繧剃ｽｿ逕ｨ縺励※CPU荳翫〒險育ｮ暦ｼ亥ｰ上＆縺ｪ驟榊・縺ｪ縺ｮ縺ｧGPU霆｢騾√・繧ｪ繝ｼ繝舌・繝倥ャ繝峨ｒ驕ｿ縺代ｋ・・
            indices = np.logspace(0, np.log10(len(radii)-1), 4).astype(int)
            radii = [radii[int(i)] for i in indices]
        
        return radii
    
    def get_default_params(self) -> dict:
        return {
            'mode': 'radius',  # 繝・ヵ繧ｩ繝ｫ繝医・蜉ｹ邇・噪縺ｪ蜊雁ｾ・Δ繝ｼ繝・
            'radii': None,     # None縺ｮ蝣ｴ蜷医・閾ｪ蜍墓ｱｺ螳・
            'weights': None,   # None縺ｮ蝣ｴ蜷医・蝮・ｭ蛾㍾縺ｿ
            'sigmas': None,    # 蠕捺擂繝｢繝ｼ繝臥畑・井ｺ呈鋤諤ｧ・・
            'agg': 'mean',     # 蠕捺擂繝｢繝ｼ繝臥畑・井ｺ呈鋤諤ｧ・・
            'auto_sigma': False,  # 蠕捺擂縺ｮsigma閾ｪ蜍墓ｱｺ螳壹・辟｡蜉ｹ蛹・
        }

###############################################################################
# 2.2. Hillshade 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_hillshade_block(block: cp.ndarray, *, azimuth: float = Constants.DEFAULT_AZIMUTH, 
                           altitude: float = Constants.DEFAULT_ALTITUDE, z_factor: float = 1.0,
                           pixel_size: float = 1.0) -> cp.ndarray:
    """1繝悶Ο繝・け縺ｫ蟇ｾ縺吶ｋHillshade險育ｮ・"""
    # NaN繝槭せ繧ｯ繧剃ｿ晏ｭ・
    nan_mask = cp.isnan(block)
    
    # 譁ｹ菴崎ｧ偵→鬮伜ｺｦ隗偵ｒ繝ｩ繧ｸ繧｢繝ｳ縺ｫ螟画鋤
    azimuth_rad = cp.radians(azimuth)
    altitude_rad = cp.radians(altitude)
    
    # 蜍ｾ驟崎ｨ育ｮ暦ｼ井ｸｭ螟ｮ蟾ｮ蛻・ｼ・ NaN繧貞性繧蝣ｴ蜷医・髫｣謗･蛟､縺ｧ陬憺俣
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=z_factor, pixel_size=pixel_size)
    
    # 蜍ｾ驟阪→蛯ｾ譁懆ｧ・
    slope = cp.arctan(cp.sqrt(dx**2 + dy**2))
    
    # 繧｢繧ｹ繝壹け繝茨ｼ域万髱｢譁ｹ菴搾ｼ・
    aspect = cp.arctan2(-dy, dx)  # 蛹励ｒ0ﾂｰ縺ｨ縺吶ｋ蠎ｧ讓咏ｳｻ
    
    # 蜈画ｺ舌・繧ｯ繝医Ν縺ｨ縺ｮ隗貞ｺｦ蟾ｮ
    aspect_diff = aspect - azimuth_rad
    
    # Hillshade險育ｮ暦ｼ・ambertian reflectance model・・
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect_diff)
    
    # 0-255縺ｮ遽・峇縺ｫ豁｣隕丞喧・・illshade縺ｯ萓句､也噪縺ｫ0-255蜃ｺ蜉幢ｼ・
    hillshade = cp.clip(hillshade, -1, 1)
    hillshade = ((hillshade + 1) / 2 * 255).astype(cp.float32)
    
    # NaN蜃ｦ逅・
    hillshade = restore_nan(hillshade, nan_mask)
    
    return hillshade

class HillshadeAlgorithm(DaskAlgorithm):
    """Hillshade繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        azimuth = params.get('azimuth', Constants.DEFAULT_AZIMUTH)
        altitude = params.get('altitude', Constants.DEFAULT_ALTITUDE)
        z_factor = params.get('z_factor', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        multiscale = params.get('multiscale', False)
        radii = params.get('radii', [1])  # common multiscale parameter (pixels)
        weights = params.get('weights', None)
        agg = params.get('agg', 'mean')
        mode = str(params.get("mode", "local")).lower()

        if mode == "spatial":
            radii = _normalize_spatial_radii(radii, pixel_size)
            multiscale = True
        else:
            if not isinstance(radii, (list, tuple)) or len(radii) == 0:
                radii = [1]
            radii = [max(1.0, float(r)) for r in radii]
            multiscale = bool(multiscale or len(radii) > 1)

        if multiscale and len(radii) > 1:
            # 繝槭Ν繝√せ繧ｱ繝ｼ繝ｫHillshade
            results = []
            for radius in radii:
                sigma = max(0.5, float(radius) / 3.0)
                # 縺ｾ縺壹せ繝繝ｼ繧ｸ繝ｳ繧ｰ
                if sigma > 1:
                    depth = int(4 * sigma)
                    smoothed = gpu_arr.map_overlap(
                        lambda x, *, sigma=sigma: gaussian_filter(x, sigma=sigma, mode='nearest'),
                        depth=depth,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32)
                    )
                else:
                    smoothed = gpu_arr
                
                # Hillshade險育ｮ・
                hs = smoothed.map_overlap(
                    compute_hillshade_block,
                    depth=1,
                    boundary='reflect',
                    dtype=cp.float32,
                    meta=cp.empty((0, 0), dtype=cp.float32),
                    azimuth=azimuth,
                    altitude=altitude,
                    z_factor=z_factor,
                    pixel_size=pixel_size
                )
                results.append(hs)
            
            # 髮・ｴ・
            stacked = da.stack(results, axis=0)
            if agg == "stack":
                return stacked
            elif agg == "mean":
                if isinstance(weights, (list, tuple)) and len(weights) == len(radii):
                    w = np.asarray(weights, dtype=np.float32)
                    if np.isfinite(w).all() and w.sum() > 0:
                        w = w / w.sum()
                        w_da = da.from_array(w.astype(np.float32), chunks=(len(radii),))
                        return da.sum(stacked * w_da[:, None, None], axis=0)
                return da.mean(stacked, axis=0)
            elif agg == "min":
                return da.min(stacked, axis=0)
            elif agg == "max":
                return da.max(stacked, axis=0)
            else:
                return da.mean(stacked, axis=0)
        else:
            # 蜊倅ｸ繧ｹ繧ｱ繝ｼ繝ｫHillshade
            return gpu_arr.map_overlap(
                compute_hillshade_block,
                depth=1,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                azimuth=azimuth,
                altitude=altitude,
                z_factor=z_factor,
                pixel_size=pixel_size
            )
    
    def get_default_params(self) -> dict:
        return {
            'azimuth': Constants.DEFAULT_AZIMUTH,
            'altitude': Constants.DEFAULT_ALTITUDE,
            'z_factor': 1.0,
            'pixel_size': 1.0,
            'multiscale': False,
            'radii': None,
            'weights': None,
            'agg': 'mean',
            'mode': 'local',
        }

###############################################################################
# 2.3. Slope 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝・域僑蠑ｵ萓具ｼ・
###############################################################################

###############################################################################
# 2.8. Visual Saliency (隕冶ｦ夂噪鬘戊送諤ｧ) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################
def _compress_saliency_feature(feature: cp.ndarray) -> cp.ndarray:
    """Tile-stable feature compression without block-global normalization."""
    return cp.log1p(cp.clip(feature, 0.0, None)).astype(cp.float32)


def visual_saliency_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """Global robust range for saliency normalization."""
    valid_data = data[~cp.isnan(data)]
    if valid_data.size == 0:
        return (0.0, 1.0)
    return (float(cp.percentile(valid_data, 0.5)),
            float(cp.percentile(valid_data, 99.5)))


def compute_visual_saliency_block(
    block: cp.ndarray,
    *,
    scales: List[float] = [2, 4, 8, 16],
    pixel_size: float = 1.0,
    normalize: bool = True,
    norm_min: float = None,
    norm_max: float = None,
) -> cp.ndarray:
    """Itti-style saliency (intensity + orientation conspicuity) for DEM."""
    nan_mask = cp.isnan(block)
    if nan_mask.any():
        fill = cp.nanmean(block)
        fill = cp.where(cp.isfinite(fill), fill, 0.0)
        work = cp.where(nan_mask, fill, block).astype(cp.float32)
    else:
        work = block.astype(cp.float32, copy=False)

    use_scales = [max(0.5, float(s)) for s in scales]
    if len(use_scales) < 4:
        use_scales = [2.0, 4.0, 8.0, 16.0]

    # Intensity center-surround maps (Itti c-s differences).
    c_indices = [0, 1]
    deltas = [2, 3]
    intensity_maps: List[cp.ndarray] = []
    for ci in c_indices:
        for d in deltas:
            si = ci + d
            if si >= len(use_scales):
                continue
            c_map = gaussian_filter(work, sigma=use_scales[ci], mode='nearest')
            s_map = gaussian_filter(work, sigma=use_scales[si], mode='nearest')
            fm = cp.abs(c_map - s_map)
            intensity_maps.append(_compress_saliency_feature(fm))

    if intensity_maps:
        I = cp.mean(cp.stack(intensity_maps, axis=0), axis=0)
    else:
        I = cp.zeros_like(work, dtype=cp.float32)

    # Orientation conspicuity approximation with 4 orientation channels.
    ori_maps: List[cp.ndarray] = []
    orientations = [0.0, cp.pi / 4, cp.pi / 2, 3 * cp.pi / 4]
    for sigma in use_scales[:3]:
        sm = gaussian_filter(work, sigma=sigma, mode='nearest')
        gy, gx = cp.gradient(sm)
        mag = cp.sqrt(gx * gx + gy * gy) + 1e-8
        theta = cp.arctan2(gy, gx)
        for o in orientations:
            resp = mag * cp.maximum(cp.cos(2.0 * (theta - o)), 0.0)
            ori_maps.append(_compress_saliency_feature(resp))

    if ori_maps:
        O = cp.mean(cp.stack(ori_maps, axis=0), axis=0)
    else:
        O = cp.zeros_like(work, dtype=cp.float32)

    sal = 0.5 * (I + O)

    if normalize:
        if norm_min is None or norm_max is None:
            norm_min, norm_max = visual_saliency_stat_func(sal)
        if norm_max > norm_min:
            result = (sal - norm_min) / (norm_max - norm_min)
        else:
            result = cp.zeros_like(sal)
        result = cp.clip(result, 0, 1)
    else:
        result = sal

    result = restore_nan(result, nan_mask)
    return result.astype(cp.float32)


class VisualSaliencyAlgorithm(DaskAlgorithm):
    """Visual saliency based on Itti-style conspicuity maps."""

    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [2, 4, 8, 16])
        max_scale = max(scales)

        stats = params.get('global_stats', None)
        stats_ok = isinstance(stats, (tuple, list)) and len(stats) >= 2
        if not stats_ok:
            # In tile backend each call receives one tile chunk. Per-tile stats would
            # reintroduce seam artifacts, so avoid local-stat fallback there.
            num_blocks = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if num_blocks > 1:
                stats = compute_global_stats(
                    gpu_arr,
                    visual_saliency_stat_func,
                    compute_visual_saliency_block,
                    {
                        'scales': scales,
                        'pixel_size': params.get('pixel_size', 1.0),
                        'normalize': False,
                    },
                    downsample_factor=params.get('downsample_factor', None),
                    depth=int(max_scale * 8),
                )
            else:
                stats = (0.0, 1.0)

        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2):
            stats = (0.0, 1.0)

        return gpu_arr.map_overlap(
            compute_visual_saliency_block,
            depth=int(max_scale * 8),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales,
            pixel_size=params.get('pixel_size', 1.0),
            normalize=True,
            norm_min=stats[0],
            norm_max=stats[1],
        )

    def get_default_params(self) -> dict:
        return {
            'scales': [2, 4, 8, 16],
            'pixel_size': 1.0,
            'downsample_factor': None,
            'verbose': False,
        }

###############################################################################
# 2.9. NPR Edges (髱槫・螳溽噪繝ｬ繝ｳ繝繝ｪ繝ｳ繧ｰ霈ｪ驛ｭ邱・ 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_npr_edges_block(block: cp.ndarray, *, edge_sigma: float = 1.0,
                          threshold_low: float = 0.1, threshold_high: float = 0.3,
                          pixel_size: float = 1.0) -> cp.ndarray:
    """NPR繧ｹ繧ｿ繧､繝ｫ縺ｮ霈ｪ驛ｭ邱壽歓蜃ｺ・域隼濶ｯ迚・2・・"""
    nan_mask = cp.isnan(block)
    resolution_class = classify_resolution(pixel_size)
    
    # 隗｣蜒丞ｺｦ縺ｫ蠢懊§縺溘せ繝繝ｼ繧ｸ繝ｳ繧ｰ
    if resolution_class in ['ultra_high', 'very_high']:
        adaptive_sigma = 0.5
    elif resolution_class in ['high', 'medium']:
        adaptive_sigma = 1.0
    elif resolution_class == 'low':  # 10m
        adaptive_sigma = 0.5  # 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ繧貞ｼｱ繧√ｋ
    else:  # very_low, ultra_low
        adaptive_sigma = 0.3
    
    # edge_sigma縺梧・遉ｺ逧・↓謖・ｮ壹＆繧後※縺・ｋ蝣ｴ蜷医・縺昴ｌ繧剃ｽｿ逕ｨ
    if edge_sigma != 1.0:
        adaptive_sigma = edge_sigma
    
    # 繝弱う繧ｺ髯､蜴ｻ・域怙蟆城剞縺ｫ・・
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
        if adaptive_sigma > 0.1:
            smoothed = gaussian_filter(filled, sigma=adaptive_sigma, mode='nearest')
        else:
            smoothed = filled
    else:
        if adaptive_sigma > 0.1:
            smoothed = gaussian_filter(block, sigma=adaptive_sigma, mode='nearest')
        else:
            smoothed = block
    
    # 譁ｹ豕・: Sobel繝輔ぅ繝ｫ繧ｿ繧剃ｽｿ逕ｨ縺励◆蜍ｾ驟崎ｨ育ｮ暦ｼ・ixel_size縺ｫ萓晏ｭ倥＠縺ｪ縺・ｼ・
    # Sobel繧ｫ繝ｼ繝阪Ν
    sobel_x = cp.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=cp.float32) / 8.0
    sobel_y = cp.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=cp.float32) / 8.0
    
    # 逡ｳ縺ｿ霎ｼ縺ｿ縺ｫ繧医ｋ蜍ｾ驟崎ｨ育ｮ・
    dx = convolve(smoothed, sobel_x, mode='nearest')
    dy = convolve(smoothed, sobel_y, mode='nearest')
    
    # 蜍ｾ驟阪・螟ｧ縺阪＆・域ｨ咎ｫ伜ｷｮ縺ｨ縺励※謇ｱ縺・ｼ・
    gradient_mag = cp.sqrt(dx**2 + dy**2)
    
    # 譁ｹ豕・: 隗｣蜒丞ｺｦ驕ｩ蠢懷梛縺ｮ蜍ｾ驟榊ｼｷ隱ｿ
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        # 菴手ｧ｣蜒丞ｺｦ縺ｧ縺ｯ螻謇逧・↑讓咎ｫ伜ｷｮ繧堤峩謗･險育ｮ・
        # 3x3霑大ｍ縺ｧ縺ｮ譛螟ｧ讓咎ｫ伜ｷｮ繧定ｨ育ｮ・
        local_max = maximum_filter(smoothed, size=3, mode='nearest')
        local_min = minimum_filter(smoothed, size=3, mode='nearest')
        local_range = local_max - local_min
        
        # 蜍ｾ驟阪→螻謇遽・峇縺ｮ邨・∩蜷医ｏ縺・
        gradient_mag = cp.maximum(gradient_mag, local_range * 0.3)
    
    gradient_dir = cp.arctan2(dy, dx)
    
    # 驕ｩ蠢懃噪縺ｪ髢ｾ蛟､險ｭ螳・
    valid_grad = gradient_mag[~nan_mask] if nan_mask.any() else gradient_mag.ravel()
    if len(valid_grad) > 0:
        # 邨ｱ險磯㍼縺ｫ蝓ｺ縺･縺城明蛟､
        grad_std = cp.std(valid_grad)
        grad_mean = cp.mean(valid_grad)
        
        # 隗｣蜒丞ｺｦ縺ｫ蠢懊§縺滄明蛟､謌ｦ逡･
        if resolution_class in ['low', 'very_low', 'ultra_low']:
            # 菴手ｧ｣蜒丞ｺｦ・壼ｹｳ蝮・､繧貞渕貅悶↓
            base_threshold = grad_mean
            threshold_range = grad_std * 1.5
        else:
            # 鬮倩ｧ｣蜒丞ｺｦ・壹ヱ繝ｼ繧ｻ繝ｳ繧ｿ繧､繝ｫ繝吶・繧ｹ
            base_threshold = cp.percentile(valid_grad, 50)
            threshold_range = cp.percentile(valid_grad, 90) - base_threshold
        
        # 繝ｦ繝ｼ繧ｶ繝ｼ謖・ｮ壹・髢ｾ蛟､縺ｧ隱ｿ謨ｴ
        actual_threshold_low = base_threshold + threshold_range * threshold_low * 0.5
        actual_threshold_high = base_threshold + threshold_range * threshold_high
        
        # 譛蟆城明蛟､繧剃ｿ晁ｨｼ・亥ｮ悟・縺ｫ逋ｽ縺・判蜒上ｒ髦ｲ縺撰ｼ・
        min_threshold = grad_mean * 0.1
        actual_threshold_low = cp.maximum(actual_threshold_low, min_threshold)
        actual_threshold_high = cp.maximum(actual_threshold_high, min_threshold * 2)
    else:
        actual_threshold_low = 0.1
        actual_threshold_high = 0.3
    
    # 髱樊怙螟ｧ蛟､謚大宛・育ｰ｡譏鍋沿・・
    angle = gradient_dir * 180.0 / cp.pi
    angle[angle < 0] += 180
    
    nms = gradient_mag.copy()
    
    # 8譁ｹ蜷代〒縺ｮ髱樊怙螟ｧ蛟､謚大宛・亥・縺ｮ繧ｳ繝ｼ繝峨→蜷後§・・
    # 0蠎ｦ縺ｨ180蠎ｦ譁ｹ蜷・
    shifted_pos = cp.roll(gradient_mag, 1, axis=1)
    shifted_neg = cp.roll(gradient_mag, -1, axis=1)
    mask = ((angle < 22.5) | (angle >= 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 45蠎ｦ譁ｹ蜷・
    shifted_pos = cp.roll(cp.roll(gradient_mag, 1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, -1, axis=0), 1, axis=1)
    mask = ((angle >= 22.5) & (angle < 67.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 90蠎ｦ譁ｹ蜷・
    shifted_pos = cp.roll(gradient_mag, 1, axis=0)
    shifted_neg = cp.roll(gradient_mag, -1, axis=0)
    mask = ((angle >= 67.5) & (angle < 112.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 135蠎ｦ譁ｹ蜷・
    shifted_pos = cp.roll(cp.roll(gradient_mag, -1, axis=0), -1, axis=1)
    shifted_neg = cp.roll(cp.roll(gradient_mag, 1, axis=0), 1, axis=1)
    mask = ((angle >= 112.5) & (angle < 157.5))
    nms = cp.where(mask & ((gradient_mag < shifted_pos) | (gradient_mag < shifted_neg)), 0, nms)
    
    # 繝繝悶Ν繧ｹ繝ｬ繝・す繝ｧ繝ｫ繝・
    strong = nms > actual_threshold_high
    weak = (nms > actual_threshold_low) & (nms <= actual_threshold_high)
    
    # 繧ｨ繝・ず縺ｮ蠑ｷ隱ｿ・・PR繧ｹ繧ｿ繧､繝ｫ・・
    edges = cp.zeros_like(nms)
    edges[strong] = 1.0
    edges[weak] = 0.5
    
    # 繝偵せ繝・Μ繧ｷ繧ｹ蜃ｦ逅・ｼ域磁邯壽ｧ縺ｮ謾ｹ蝟・ｼ・
    for _ in range(3):  # 3蝗槭↓蠅励ｄ縺励※謗･邯壹ｒ蠑ｷ蛹・
        dilated = cp.maximum(
            cp.maximum(
                cp.roll(edges, 1, axis=0),
                cp.roll(edges, -1, axis=0)
            ),
            cp.maximum(
                cp.roll(edges, 1, axis=1),
                cp.roll(edges, -1, axis=1)
            )
        )
        
        # 蟇ｾ隗呈婿蜷代ｂ霑ｽ蜉
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, 1, axis=0), 1, axis=1))
        dilated = cp.maximum(dilated, cp.roll(cp.roll(edges, -1, axis=0), -1, axis=1))
        
        edges = cp.where(weak & (dilated > 0.5), 1.0, edges)
    
    # 蠕悟・逅・ｼ夊ｧ｣蜒丞ｺｦ縺ｫ蠢懊§縺溘お繝・ず縺ｮ螟ｪ縺戊ｪｿ謨ｴ
    if resolution_class in ['low', 'very_low', 'ultra_low']:
        # 菴手ｧ｣蜒丞ｺｦ縺ｧ縺ｯ繧ｨ繝・ず繧貞ｰ代＠螟ｪ縺上☆繧・
        structure = cp.ones((3, 3))
        edges_binary = edges > 0.5
        edges_dilated = binary_dilation(edges_binary, structure=structure).astype(cp.float32)
        edges = cp.where(edges_dilated, cp.maximum(edges, 0.8), edges)
    
    # 繧ｨ繝・ず蠑ｷ蠎ｦ繧定ｪｿ謨ｴ
    edges = edges * 0.8
    
    # 霈ｪ驛ｭ邱壹ｒ蜿崎ｻ｢・磯ｻ堤ｷ壹〒謠冗判・・
    result = 1.0 - edges
    
    # 繧ｳ繝ｳ繝医Λ繧ｹ繝郁ｪｿ謨ｴ・医お繝・ず繧偵ｈ繧願ｦ九ｄ縺吶￥・・
    result = cp.clip(result, 0.2, 1.0)
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN蜃ｦ逅・
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class NPREdgesAlgorithm(DaskAlgorithm):
    """NPR霈ｪ驛ｭ邱壹い繝ｫ繧ｴ繝ｪ繧ｺ繝・域隼濶ｯ迚茨ｼ・"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        edge_sigma = params.get('edge_sigma', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        threshold_low = params.get('threshold_low', 0.1)
        threshold_high = params.get('threshold_high', 0.3)
        
        # depth繧・縺ｫ蠅励ｄ縺呻ｼ・obel繝輔ぅ繝ｫ繧ｿ縺ｨ閹ｨ蠑ｵ蜃ｦ逅・・縺溘ａ・・
        depth = 3
        
        if edge_sigma != 1.0:
            depth = max(depth, int(edge_sigma * 4 + 2))
        
        return gpu_arr.map_overlap(
            compute_npr_edges_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            edge_sigma=edge_sigma,
            threshold_low=threshold_low,
            threshold_high=threshold_high,
            pixel_size=pixel_size,
        )
    
    def get_default_params(self) -> dict:
        return {
            'edge_sigma': 1.0,
            'threshold_low': 0.1,
            'threshold_high': 0.3,
            'pixel_size': 1.0
        }

###############################################################################
# 2.10. Atmospheric Perspective (螟ｧ豌鈴□霑第ｳ・ 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################



###############################################################################
# 2.11. Ambient Occlusion (迺ｰ蠅・・驕ｮ阡ｽ) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_ambient_occlusion_block(block: cp.ndarray, *, 
                                    num_samples: int = 16,
                                    radius: float = 10.0,
                                    intensity: float = 1.0,
                                    pixel_size: float = 1.0) -> cp.ndarray:
    """繧ｹ繧ｯ繝ｪ繝ｼ繝ｳ遨ｺ髢鍋腸蠅・・驕ｮ阡ｽ・・SAO・峨・蝨ｰ蠖｢迚茨ｼ磯ｫ倬溘・繧ｯ繝医Ν蛹也沿・・"""
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # 繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ譁ｹ蜷代ｒ莠句燕險育ｮ・
    angles = cp.linspace(0, 2 * cp.pi, num_samples, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
    
    # 霍晞屬縺ｮ繧ｵ繝ｳ繝励Μ繝ｳ繧ｰ
    r_factors = cp.array([0.25, 0.5, 0.75, 1.0])
    
    # 蜈ｨ繧ｵ繝ｳ繝励Ν轤ｹ縺ｮ蠎ｧ讓吶ｒ莠句燕險育ｮ暦ｼ医・繧ｯ繝医Ν蛹厄ｼ・
    occlusion_total = cp.zeros((h, w), dtype=cp.float32)
    sample_count = cp.zeros((h, w), dtype=cp.float32)  # 譛牙柑縺ｪ繧ｵ繝ｳ繝励Ν謨ｰ繧偵き繧ｦ繝ｳ繝・
    
    # 繝舌ャ繝∝・逅・〒鬮倬溷喧
    for r_factor in r_factors:
        r = radius * r_factor
        
        # 蜈ｨ譁ｹ蜷代・螟我ｽ阪ｒ荳蠎ｦ縺ｫ險育ｮ暦ｼ医ヴ繧ｯ繧ｻ繝ｫ蜊倅ｽ阪↓螟画鋤・・
        # 菫ｮ豁｣: radius縺ｯ繝斐け繧ｻ繝ｫ蜊倅ｽ阪→縺励※謇ｱ縺・ｼ・ixel_size縺ｧ髯､邂励＠縺ｪ縺・ｼ・
        dx_all = cp.round(r * directions[:, 0]).astype(int)
        dy_all = cp.round(r * directions[:, 1]).astype(int)
        
        for i in range(num_samples):
            # CuPy驟榊・縺九ｉ蛟句挨縺ｮ蛟､繧貞叙蠕励＠縺ｦ譏守､ｺ逧・↓int縺ｫ螟画鋤
            dx = int(dx_all[i])
            dy = int(dy_all[i])
            
            if dx == 0 and dy == 0:
                continue
            
            # 蠢・ｦ√↑譁ｹ蜷代・縺ｿ繝代ョ繧｣繝ｳ繧ｰ
            pad_left = max(0, -dx)
            pad_right = max(0, dx)
            pad_top = max(0, -dy)
            pad_bottom = max(0, dy)
            
            # 繝代ョ繧｣繝ｳ繧ｰ・・dge mode繧剃ｽｿ逕ｨ・・
            padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                        mode='edge')
            
            # 繧ｷ繝輔ヨ
            start_y = pad_top + dy
            start_x = pad_left + dx
            shifted = padded[start_y:start_y+h, start_x:start_x+w]
            
            # 鬮倥＆縺ｮ蟾ｮ縺ｨ驕ｮ阡ｽ隗貞ｺｦ
            height_diff = shifted - block
            # 菫ｮ豁｣: 螳滄圀縺ｮ霍晞屬・医Γ繝ｼ繝医Ν・峨ｒ菴ｿ逕ｨ
            distance = r * pixel_size
            occlusion_angle = cp.arctan(height_diff / distance)
            
            # 豁｣縺ｮ隗貞ｺｦ縺ｮ縺ｿ繧帝・阡ｽ縺ｨ縺励※謇ｱ縺・ｼ井ｿｮ豁｣: 繧医ｊ驕ｩ蛻・↑驕ｮ阡ｽ縺ｮ險育ｮ暦ｼ・
            # 隗貞ｺｦ繧・-1縺ｮ遽・峇縺ｫ豁｣隕丞喧・域怙螟ｧ45蠎ｦ繧・縺ｨ縺吶ｋ・・
            max_angle = cp.pi / 4  # 45蠎ｦ
            occlusion = cp.maximum(0, occlusion_angle) / max_angle
            occlusion = cp.minimum(occlusion, 1.0)  # 1繧定ｶ・∴縺ｪ縺・ｈ縺・↓繧ｯ繝ｪ繝・・
            
            # 霍晞屬縺ｫ繧医ｋ貂幄｡ｰ・井ｿｮ豁｣: 繧医ｊ邱ｩ繧・°縺ｪ貂幄｡ｰ・・
            distance_factor = 1.0 - (r_factor * 0.3)  # 0.5縺九ｉ0.3縺ｫ螟画峩
            
            # 驕ｮ阡ｽ縺ｮ邏ｯ遨搾ｼ・aN繧帝勁螟厄ｼ・
            valid = ~(cp.isnan(shifted) | nan_mask)
            occlusion_total += cp.where(valid, 
                                      occlusion * distance_factor,
                                      0)
            sample_count += cp.where(valid, 1.0, 0)
    
    # 豁｣隕丞喧・井ｿｮ豁｣: 譛牙柑縺ｪ繧ｵ繝ｳ繝励Ν謨ｰ縺ｧ髯､邂暦ｼ・
    # 繧ｼ繝ｭ髯､邂励ｒ髦ｲ縺・
    sample_count = cp.maximum(sample_count, 1.0)
    
    # 蟷ｳ蝮・・阡ｽ繧定ｨ育ｮ暦ｼ井ｿｮ豁｣: 縺吶〒縺ｫ0-1縺ｮ遽・峇・・
    mean_occlusion = occlusion_total / sample_count
    
    # AO縺ｮ險育ｮ暦ｼ井ｿｮ豁｣: 繧医ｊ逶ｴ謗･逧・↑險育ｮ暦ｼ・
    # 驕ｮ阡ｽ縺悟､壹＞縺ｻ縺ｩ證励￥縺ｪ繧具ｼ・縺ｫ霑代▼縺擾ｼ・
    ao = 1.0 - mean_occlusion * intensity
    ao = cp.clip(ao, 0, 1)
    
    # 繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ・・aN閠・・・・
    if nan_mask.any():
        filled_ao = cp.where(nan_mask, 1.0, ao)  # NaN鬆伜沺縺ｯ譏弱ｋ縺擾ｼ磯・阡ｽ縺ｪ縺暦ｼ・
        ao = gaussian_filter(filled_ao, sigma=1.0, mode='nearest')
    else:
        ao = gaussian_filter(ao, sigma=1.0, mode='nearest')
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣
    result = cp.power(ao, Constants.DEFAULT_GAMMA)
    
    # NaN蜃ｦ逅・
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)


class AmbientOcclusionAlgorithm(DaskAlgorithm):
    """迺ｰ蠅・・驕ｮ阡ｽ繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        num_samples = params.get('num_samples', 16)
        radius = params.get('radius', 10.0)
        intensity = params.get('intensity', 1.0)
        pixel_size = params.get('pixel_size', 1.0)
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            responses = []
            for r in radii:
                r_use = float(max(1, int(round(float(r)))))
                responses.append(
                    gpu_arr.map_overlap(
                        compute_ambient_occlusion_block,
                        depth=int(r_use + 1),
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        num_samples=num_samples,
                        radius=r_use,
                        intensity=intensity,
                        pixel_size=pixel_size,
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        
        # 菫ｮ豁｣: radius繧偵ヴ繧ｯ繧ｻ繝ｫ蜊倅ｽ阪→縺励※謇ｱ縺・・縺ｧ縲｝ixel_size縺ｧ髯､邂励＠縺ｪ縺・
        # 繝ｦ繝ｼ繧ｶ繝ｼ縺梧欠螳壹☆繧脚adius縺ｯ譌｢縺ｫ繝斐け繧ｻ繝ｫ蜊倅ｽ・
        return gpu_arr.map_overlap(
            compute_ambient_occlusion_block,
            depth=int(radius + 1),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            num_samples=num_samples,
            radius=radius,
            intensity=intensity,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'num_samples': 16,
            'radius': 10.0,     # 繝斐け繧ｻ繝ｫ蜊倅ｽ阪・謗｢邏｢蜊雁ｾ・
            'intensity': 1.0,
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }

###############################################################################
# 2.13. LRM (Local Relief Model) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_lrm_block(block: cp.ndarray, *, kernel_size: int = 25,
                     pixel_size: float = 1.0,
                     std_global: float = None,
                     normalize: bool = True) -> cp.ndarray:  # ????
    """??????????????????????????????????"""
    # NaN???????????
    nan_mask = cp.isnan(block)

    # ??????????????????????????aN?????
    sigma = kernel_size / 3.0
    trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')

    # ????????????
    lrm = block - trend

    result = lrm
    if normalize:
        if std_global is not None and std_global > 1e-9:
            scale = float(std_global)
        else:
            scale = lrm_stat_func(lrm)[0]
            if not cp.isfinite(scale) or scale <= 1e-9:
                scale = 1.0

        # ?????????????????????
        result = cp.tanh(lrm / (2.5 * scale))

    # NaN????
    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)

class LRMAlgorithm(DaskAlgorithm):
    """螻謇襍ｷ莨上Δ繝・Ν繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        kernel_size = params.get('kernel_size', 25)

        stats = params.get('global_stats', None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) == 1
            and float(stats[0]) > 1e-9
        )
        if not stats_ok:
            # ???????????
            stats = compute_global_stats(
                gpu_arr,
                lambda data: lrm_stat_func(compute_lrm_block(data, kernel_size=kernel_size, normalize=False)),
                compute_lrm_block,
                {'kernel_size': kernel_size, 'normalize': False},
                downsample_factor=None,
                depth=int(kernel_size * 2)
            )
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 1 and float(stats[0]) > 1e-9):
            stats = (1.0,)

        return gpu_arr.map_overlap(
            compute_lrm_block,
            depth=int(kernel_size * 2),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            kernel_size=kernel_size,
            std_global=stats[0],
            normalize=True
        )
    
    def get_default_params(self) -> dict:
        return {
            'kernel_size': 25,  # 繝医Ξ繝ｳ繝蛾勁蜴ｻ縺ｮ繧ｫ繝ｼ繝阪Ν繧ｵ繧､繧ｺ
        }

###############################################################################
# 2.14. Openness (髢句ｺｦ) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝 - 邁｡譏馴ｫ倬溽沿
###############################################################################
# 蜉ｹ邇・噪縺ｪ繝吶け繝医Ν蛹也沿
def compute_openness_vectorized(block: cp.ndarray, *, 
                              openness_type: str = 'positive',
                              num_directions: int = 16,
                              max_distance: int = 50,
                              pixel_size: float = 1.0) -> cp.ndarray:
    """髢句ｺｦ縺ｮ險育ｮ暦ｼ域怙驕ｩ蛹也沿・・"""
    h, w = block.shape
    nan_mask = cp.isnan(block)
    
    # 譁ｹ蜷代・繧ｯ繝医Ν縺ｮ莠句燕險育ｮ・
    angles = cp.linspace(0, 2 * cp.pi, num_directions, endpoint=False)
    directions = cp.stack([cp.cos(angles), cp.sin(angles)], axis=1)
    
    # 蛻晄悄蛹・
    init_val = -cp.pi/2 if openness_type == 'positive' else cp.pi/2
    max_angles = cp.full((h, w), init_val, dtype=cp.float32)
    
    # 霍晞屬繧ｵ繝ｳ繝励Ν繧剃ｺ句燕險育ｮ暦ｼ域紛謨ｰ蛟､縺ｫ・・
    distances = cp.unique(cp.linspace(0.1, 1.0, 10) * max_distance).astype(int)
    distances = distances[distances > 0]  # 0繧帝勁螟・
    
    # 繝代ョ繧｣繝ｳ繧ｰ蛟､縺ｮ豎ｺ螳・
    pad_value = Constants.NAN_FILL_VALUE_POSITIVE if openness_type == 'positive' else Constants.NAN_FILL_VALUE_NEGATIVE
    
    for r in distances:
        # 蜈ｨ譁ｹ蜷代・繧ｪ繝輔そ繝・ヨ繧剃ｸ蠎ｦ縺ｫ險育ｮ・
        offsets = cp.round(r * directions).astype(int)
        
        for offset in offsets:
            offset_x, offset_y = offset
            # CuPy驟榊・隕∫ｴ繧単ython int縺ｫ譏守､ｺ逧・↓螟画鋤
            offset_x = int(offset_x)
            offset_y = int(offset_y)
            
            if offset_x == 0 and offset_y == 0:
                continue
            
            # 繝代ョ繧｣繝ｳ繧ｰ繧ｵ繧､繧ｺ縺ｮ險育ｮ暦ｼ育ｰ｡貎斐↓・・
            pad_left = max(0, -offset_x)
            pad_right = max(0, offset_x)
            pad_top = max(0, -offset_y)
            pad_bottom = max(0, offset_y)
            
            # 繝代ョ繧｣繝ｳ繧ｰ
            if nan_mask.any():
                padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                            mode='constant', constant_values=pad_value)
            else:
                padded = cp.pad(block, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                            mode='edge')
            
            # 莉･荳九・螟画峩縺ｪ縺・
            # 繧ｷ繝輔ヨ・育ｰ｡貎斐↑險倩ｿｰ・・
            start_y = pad_top + offset_y
            start_x = pad_left + offset_x
            shifted = padded[start_y:start_y+h, start_x:start_x+w]
            
            # 隗貞ｺｦ險育ｮ励→譖ｴ譁ｰ
            angle = cp.arctan((shifted - block) / (r * pixel_size))
            
            if openness_type == 'positive':
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.maximum(max_angles, angle), max_angles)
            else:
                valid = ~(cp.isnan(angle) | nan_mask)
                max_angles = cp.where(valid, cp.minimum(max_angles, angle), max_angles)
    
    # 髢句ｺｦ縺ｮ險育ｮ励→豁｣隕丞喧
    openness = (cp.pi/2 - max_angles if openness_type == 'positive' 
                else cp.pi/2 + max_angles)
    openness = cp.clip(openness / (cp.pi/2), 0, 1)
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣縺ｨNaN蜃ｦ逅・
    result = cp.power(openness, Constants.DEFAULT_GAMMA)
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class OpennessAlgorithm(DaskAlgorithm):
    """髢句ｺｦ繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝・育ｰ｡譏馴ｫ倬溽沿・・"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        max_distance = params.get('max_distance', 50)
        openness_type = params.get('openness_type', 'positive')
        num_directions = params.get('num_directions', 16)
        pixel_size = params.get('pixel_size', 1.0)
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            responses = []
            for r in radii:
                max_dist = int(max(2, round(float(r))))
                responses.append(
                    gpu_arr.map_overlap(
                        compute_openness_vectorized,
                        depth=max_dist + 1,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        openness_type=openness_type,
                        num_directions=num_directions,
                        max_distance=max_dist,
                        pixel_size=pixel_size,
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)

        return gpu_arr.map_overlap(
            compute_openness_vectorized,  # 繝吶け繝医Ν蛹也沿繧剃ｽｿ逕ｨ
            depth=max_distance+1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            openness_type=openness_type,
            num_directions=num_directions,
            max_distance=max_distance,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'openness_type': 'positive',  # 'positive' or 'negative'
            'num_directions': 16,  # 謗｢邏｢譁ｹ蜷第焚・亥ｰ代↑縺上＠縺ｦ鬮倬溷喧・・
            'max_distance': 50,    # 譛螟ｧ謗｢邏｢霍晞屬・医ヴ繧ｯ繧ｻ繝ｫ・・
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }

def compute_slope_block(block: cp.ndarray, *, unit: str = 'degree',
                       pixel_size: float = 1.0) -> cp.ndarray:
    """1繝悶Ο繝・け縺ｫ蟇ｾ縺吶ｋ蜍ｾ驟崎ｨ育ｮ・"""
    # NaN繝槭せ繧ｯ繧剃ｿ晏ｭ・
    nan_mask = cp.isnan(block)
    
    # 蜍ｾ驟崎ｨ育ｮ暦ｼ・aN閠・・・・
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    # 蜍ｾ驟阪・螟ｧ縺阪＆
    slope_rad = cp.arctan(cp.sqrt(dx**2 + dy**2))
    
    # 蜊倅ｽ榊､画鋤
    if unit == 'degree':
        slope = cp.degrees(slope_rad)
    elif unit == 'percent':
        slope = cp.tan(slope_rad) * 100
    else:  # radians
        slope = slope_rad
    
    # NaN蜃ｦ逅・
    slope = restore_nan(slope, nan_mask)
    
    return slope.astype(cp.float32)


def compute_slope_spatial_block(
    block: cp.ndarray,
    *,
    unit: str = "degree",
    pixel_size: float = 1.0,
    radius: float = 4.0,
) -> cp.ndarray:
    smoothed = _smooth_for_radius(block, radius)
    return compute_slope_block(smoothed, unit=unit, pixel_size=pixel_size)

class SlopeAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        unit = params.get('unit', 'degree')
        pixel_size = params.get('pixel_size', 1.0)
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(2, int(float(radius) * 2 + 1))
                responses.append(
                    gpu_arr.map_overlap(
                        compute_slope_spatial_block,
                        depth=depth,
                        boundary="reflect",
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        unit=unit,
                        pixel_size=pixel_size,
                        radius=float(radius),
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)

        return gpu_arr.map_overlap(
            compute_slope_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            unit=unit,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'unit': 'degree',  # 'degree', 'percent', 'radians'
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }

###############################################################################
# 2.3. Specular (驥大ｱ槫・豐｢蜉ｹ譫・ 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_specular_block(block: cp.ndarray, *, roughness_scale: float = 50.0,
                          shininess: float = 20.0, pixel_size: float = 1.0,
                          light_azimuth: float = Constants.DEFAULT_AZIMUTH, light_altitude: float = Constants.DEFAULT_ALTITUDE) -> cp.ndarray:
    """驥大ｱ槫・豐｢蜉ｹ譫懊・險育ｮ暦ｼ・ook-Torrance繝｢繝・Ν縺ｮ邁｡逡･迚茨ｼ・"""
    # NaN繝槭せ繧ｯ繧剃ｿ晏ｭ・
    nan_mask = cp.isnan(block)
    
    # 豕慕ｷ壹・繧ｯ繝医Ν縺ｮ險育ｮ・
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    normal = cp.stack([-dx, -dy, cp.ones_like(dx)], axis=-1)
    normal = normal / cp.linalg.norm(normal, axis=-1, keepdims=True)
    
    # 繝ｩ繝輔ロ繧ｹ縺ｮ險育ｮ暦ｼ亥ｱ謇逧・↑讓咎ｫ倥・蛻・淵・・
    kernel_size = max(3, int(roughness_scale))
    
    # NaN蟇ｾ蠢懊・繝ｩ繝輔ロ繧ｹ險育ｮ・
    if nan_mask.any():
        filled = cp.where(nan_mask, 0, block)
        valid = (~nan_mask).astype(cp.float32)
        
        # 蟷ｳ蝮・→蟷ｳ蝮・ｺ御ｹ励ｒ險育ｮ暦ｼ・aN閠・・・・
        mean_values = uniform_filter(filled * valid, size=kernel_size, mode='constant')
        mean_weights = uniform_filter(valid, size=kernel_size, mode='constant')
        mean_filter = cp.where(mean_weights > 0, mean_values / mean_weights, 0)
        
        sq_values = uniform_filter((filled**2) * valid, size=kernel_size, mode='constant')
        mean_sq_filter = cp.where(mean_weights > 0, sq_values / mean_weights, 0)
    else:
        # 蟷ｳ蝮・→蟷ｳ蝮・ｺ御ｹ励ｒ險育ｮ・
        mean_filter = uniform_filter(block, size=kernel_size, mode='constant')
        mean_sq_filter = uniform_filter(block**2, size=kernel_size, mode='constant')
    
    # 讓呎ｺ門￥蟾ｮ = sqrt(E[X^2] - E[X]^2)
    roughness = cp.sqrt(cp.maximum(mean_sq_filter - mean_filter**2, 0))
    
    # 繝ｩ繝輔ロ繧ｹ繧呈ｭ｣隕丞喧・医ｈ繧企←蛻・↑遽・峇縺ｫ・・
    roughness_valid = roughness[~nan_mask] if nan_mask.any() else roughness
    if len(roughness_valid) > 0 and cp.max(roughness_valid) > 0:
        roughness = roughness / cp.max(roughness_valid)
        # 譛蟆丞､繧定ｨｭ螳壹＠縺ｦ螳悟・縺ｪ髀｡髱｢蜿榊ｰ・ｒ髦ｲ縺・
        roughness = cp.clip(roughness, 0.1, 1.0)
    else:
        roughness = cp.full_like(block, 0.5)
    
    # 蜈画ｺ先婿蜷・
    light_az_rad = cp.radians(light_azimuth)
    light_alt_rad = cp.radians(light_altitude)
    light_dir = cp.array([
        cp.sin(light_az_rad) * cp.cos(light_alt_rad),
        -cp.cos(light_az_rad) * cp.cos(light_alt_rad),
        cp.sin(light_alt_rad)
    ])
    
    # 隕也ｷ壽婿蜷托ｼ育悄荳翫°繧会ｼ・
    view_dir = cp.array([0, 0, 1])
    
    # 繝上・繝輔・繧ｯ繝医Ν
    half_vec = (light_dir + view_dir) / cp.linalg.norm(light_dir + view_dir)
    
    # 繧ｹ繝壹く繝･繝ｩ繝ｼ險育ｮ暦ｼ医ラ繝・ヨ遨阪ｒ豁｣縺励￥險育ｮ暦ｼ・
    n_dot_h = cp.sum(normal * half_vec.reshape(1, 1, 3), axis=-1)
    n_dot_h = cp.clip(n_dot_h, 0, 1)
    
    # 繧医ｊ遨上ｄ縺九↑謖・焚繧剃ｽｿ逕ｨ
    exponent = shininess * (1.0 - roughness * 0.8)  # roughness縺碁ｫ倥＞縺ｻ縺ｩ謖・焚繧剃ｸ九￡繧・
    specular = cp.power(n_dot_h, exponent)
    
    # 繝・ぅ繝輔Η繝ｼ繧ｺ謌仙・繧りｿｽ蜉・亥ｮ悟・縺ｪ鮟偵ｒ髦ｲ縺撰ｼ・
    n_dot_l = cp.sum(normal * light_dir.reshape(1, 1, 3), axis=-1)
    n_dot_l = cp.clip(n_dot_l, 0, 1)
    diffuse = n_dot_l * 0.3  # 繝・ぅ繝輔Η繝ｼ繧ｺ謌仙・繧・0%
    
    # 蜷域・
    result = diffuse + specular * 0.7
    result = cp.clip(result, 0, 1)
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣・医ｈ繧頑・繧九￥縺吶ｋ縺溘ａ縲√ぎ繝ｳ繝槫､繧定ｪｿ謨ｴ・・
    result = cp.power(result, 0.7)  # Constants.DEFAULT_GAMMA縺ｮ莉｣繧上ｊ縺ｫ0.7繧剃ｽｿ逕ｨ
    
    # NaN蜃ｦ逅・
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

def compute_specular_spatial_block(
    block: cp.ndarray,
    *,
    roughness_scale: float = 50.0,
    shininess: float = 20.0,
    pixel_size: float = 1.0,
    light_azimuth: float = Constants.DEFAULT_AZIMUTH,
    light_altitude: float = Constants.DEFAULT_ALTITUDE,
    radius: float = 4.0,
) -> cp.ndarray:
    smoothed = _smooth_for_radius(block, radius)
    return compute_specular_block(
        smoothed,
        roughness_scale=roughness_scale,
        shininess=shininess,
        pixel_size=pixel_size,
        light_azimuth=light_azimuth,
        light_altitude=light_altitude,
    )


class SpecularAlgorithm(DaskAlgorithm):
    """驥大ｱ槫・豐｢蜉ｹ譫懊い繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        roughness_scale = params.get('roughness_scale', 50.0)
        shininess = params.get('shininess', 20.0)
        pixel_size = params.get('pixel_size', 1.0)
        light_azimuth = params.get('light_azimuth', Constants.DEFAULT_AZIMUTH)
        light_altitude = params.get('light_altitude', Constants.DEFAULT_ALTITUDE)
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(int(roughness_scale), int(float(radius) * 2 + 1))
                responses.append(
                    gpu_arr.map_overlap(
                        compute_specular_spatial_block,
                        depth=depth,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        roughness_scale=roughness_scale,
                        shininess=shininess,
                        pixel_size=pixel_size,
                        light_azimuth=light_azimuth,
                        light_altitude=light_altitude,
                        radius=float(radius),
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)

        return gpu_arr.map_overlap(
            compute_specular_block,
            depth=int(roughness_scale),
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            roughness_scale=roughness_scale,
            shininess=shininess,
            pixel_size=pixel_size,
            light_azimuth=light_azimuth,
            light_altitude=light_altitude
        )
    
    def get_default_params(self) -> dict:
        return {
            'roughness_scale': 20.0,
            'shininess': 10.0,
            'light_azimuth': Constants.DEFAULT_AZIMUTH,
            'light_altitude': Constants.DEFAULT_ALTITUDE,
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }

###############################################################################
# 2.4. Atmospheric Scattering (螟ｧ豌玲淵荵ｱ蜈・ 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_atmospheric_scattering_block(block: cp.ndarray, *, 
                                       scattering_strength: float = 0.5,
                                       intensity: Optional[float] = None,
                                       pixel_size: float = 1.0) -> cp.ndarray:
    """螟ｧ豌玲淵荵ｱ縺ｫ繧医ｋ繧ｷ繧ｧ繝ｼ繝・ぅ繝ｳ繧ｰ・・ayleigh謨｣荵ｱ縺ｮ邁｡逡･迚茨ｼ・"""
    # intensity 縺ｯ scattering_strength 縺ｮ繧ｨ繧､繝ｪ繧｢繧ｹ・亥ｾ梧婿莠呈鋤・・
    if intensity is not None:
        scattering_strength = intensity
        
    # NaN繝槭せ繧ｯ繧剃ｿ晏ｭ・
    nan_mask = cp.isnan(block)
    
    # 豕慕ｷ夊ｨ育ｮ・
    dy, dx, nan_mask = handle_nan_for_gradient(block, scale=1, pixel_size=pixel_size)
    
    slope = cp.sqrt(dx**2 + dy**2)
    
    # 螟ｩ鬆りｧ抵ｼ域ｳ慕ｷ壹→蝙ら峩譁ｹ蜷代・縺ｪ縺呵ｧ抵ｼ・
    zenith_angle = cp.arctan(slope)
    
    # 螟ｧ豌励・蜴壹＆・育ｰ｡逡･蛹厄ｼ壼､ｩ鬆りｧ偵↓豈比ｾ具ｼ・
    air_mass = 1.0 / (cp.cos(zenith_angle) + 0.001)  # 繧ｼ繝ｭ髯､邂怜屓驕ｿ
    
    # Rayleigh謨｣荵ｱ縺ｮ霑台ｼｼ
    scattering = 1.0 - cp.exp(-scattering_strength * air_mass)
    
    # 髱偵∩縺後°縺｣縺滓淵荵ｱ蜈峨ｒ陦ｨ迴ｾ・亥腰荳繝√Ε繝ｳ繝阪Ν縺ｪ縺ｮ縺ｧ譏主ｺｦ縺ｮ縺ｿ・・
    ambient = 0.4 + 0.6 * scattering
    
    # 騾壼ｸｸ縺ｮHillshade縺ｨ邨・∩蜷医ｏ縺・
    azimuth_rad = cp.radians(Constants.DEFAULT_AZIMUTH)
    altitude_rad = cp.radians(Constants.DEFAULT_ALTITUDE)
    aspect = cp.arctan2(-dy, dx)
    
    hillshade = cp.cos(altitude_rad) * cp.cos(slope) + \
                cp.sin(altitude_rad) * cp.sin(slope) * cp.cos(aspect - azimuth_rad)
    
    # 謨｣荵ｱ蜈峨→逶ｴ謗･蜈峨・蜷域・
    result = ambient * 0.3 + hillshade * 0.7
    result = cp.clip(result, 0, 1)
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN蜃ｦ逅・
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

def compute_atmospheric_scattering_spatial_block(
    block: cp.ndarray,
    *,
    scattering_strength: float = 0.5,
    intensity: Optional[float] = None,
    pixel_size: float = 1.0,
    radius: float = 4.0,
) -> cp.ndarray:
    smoothed = _smooth_for_radius(block, radius)
    return compute_atmospheric_scattering_block(
        smoothed,
        scattering_strength=scattering_strength,
        intensity=intensity,
        pixel_size=pixel_size,
    )


class AtmosphericScatteringAlgorithm(DaskAlgorithm):
    """螟ｧ豌玲淵荵ｱ蜈峨い繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scattering_strength = params.get('scattering_strength', 0.5)
        intensity = params.get('intensity', None)
        pixel_size = params.get('pixel_size', 1.0)
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(2, int(float(radius) * 2 + 1))
                responses.append(
                    gpu_arr.map_overlap(
                        compute_atmospheric_scattering_spatial_block,
                        depth=depth,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        scattering_strength=scattering_strength,
                        intensity=intensity,
                        pixel_size=pixel_size,
                        radius=float(radius),
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)
        
        return gpu_arr.map_overlap(
            compute_atmospheric_scattering_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scattering_strength=scattering_strength,
            intensity=intensity,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'scattering_strength': 0.5,
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }

###############################################################################
# 2.5. Multiscale Terrain (繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ蝨ｰ蠖｢) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

class MultiscaleDaskAlgorithm(DaskAlgorithm):
    """繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ蝨ｰ蠖｢繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [1, 10, 50, 100])
        weights = params.get('weights', None)
        
        downsample_factor = params.get('downsample_factor', None)
        if downsample_factor is None:
            # 閾ｪ蜍墓ｱｺ螳・
            downsample_factor = determine_optimal_downsample_factor(
                gpu_arr.shape,
                algorithm_name='multiscale_terrain'
            )
            
        if weights is None:
            # 繝・ヵ繧ｩ繝ｫ繝茨ｼ壹せ繧ｱ繝ｼ繝ｫ縺ｫ蜿肴ｯ比ｾ九☆繧矩㍾縺ｿ
            weights = [1.0 / s for s in scales]
        
        # 驥阪∩繧呈ｭ｣隕丞喧
        weights = cp.array(weights, dtype=cp.float32)
        weights = weights / weights.sum()
        
        # 譛螟ｧ繧ｹ繧ｱ繝ｼ繝ｫ縺ｫ蝓ｺ縺･縺・※蜈ｱ騾壹・depth繧呈ｱｺ螳・
        max_scale = max(scales)
        common_depth = min(int(4 * max_scale), Constants.MAX_DEPTH)
        
        # 邵ｮ蟆冗沿縺ｧ邨ｱ險磯㍼繧定ｨ育ｮ・
        downsampled = gpu_arr[::downsample_factor, ::downsample_factor]
        
        # 邵ｮ蟆冗沿縺ｧ繝槭Ν繝√せ繧ｱ繝ｼ繝ｫ蜃ｦ逅・
        results_small = []

        # 蜈ｱ騾壹・depth繧定ｨ育ｮ暦ｼ郁ｿｽ蜉・・
        max_scale_small = max([max(1, s // downsample_factor) for s in scales])
        common_depth_small = min(int(4 * max_scale_small), Constants.MAX_DEPTH)

        def create_weighted_combiner(weights):
            def weighted_combine_for_stats(*blocks):
                nan_mask = cp.isnan(blocks[0])
                result = cp.zeros_like(blocks[0])
                
                for i, block in enumerate(blocks):
                    valid = ~cp.isnan(block)
                    result[valid] += block[valid] * weights[i]
                
                result[nan_mask] = cp.nan
                return result
            return weighted_combine_for_stats
        
        for i, scale in enumerate(scales):
            scale_small = max(1, scale // downsample_factor)
            
            def compute_detail_small(block, *, scale):
                # scale=1縺ｧ繧よ怙蟆城剞縺ｮ繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ繧帝←逕ｨ
                smoothed, nan_mask = handle_nan_with_gaussian(block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            
            detail_small = downsampled.map_overlap(
                compute_detail_small,
                depth=common_depth_small,  # 蜈ｱ騾壹・depth繧剃ｽｿ逕ｨ
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                scale=scale_small
            )
            results_small.append(detail_small)
            
        # 繝ｫ繝ｼ繝怜ｾ後↓荳蠎ｦ縺縺大粋謌・
        combined_small = da.map_blocks(
            create_weighted_combiner(weights),
            *results_small,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        ).compute()

        # 邨ｱ險磯㍼繧定ｨ育ｮ・
        valid_data = combined_small[~cp.isnan(combined_small)]
        if len(valid_data) > 0:
            norm_min = float(cp.percentile(valid_data, 5))
            norm_max = float(cp.percentile(valid_data, 95))
        else:
            norm_min, norm_max = 0.0, 1.0
        
        if params.get('verbose', False):
            print(f"Multiscale Terrain global stats: min={norm_min:.3f}, max={norm_max:.3f}")
        
        # Step 2: 繝輔Ν繧ｵ繧､繧ｺ縺ｧ蜃ｦ逅・
        results = []
        for scale in scales:
            def compute_detail_with_smooth(block, *, scale):
                # scale=1縺ｧ繧よ怙蟆城剞縺ｮ繧ｹ繝繝ｼ繧ｸ繝ｳ繧ｰ繧帝←逕ｨ
                smoothed, nan_mask = handle_nan_with_gaussian(block, sigma=max(scale, 0.5), mode='nearest')
                detail = block - smoothed
                detail = restore_nan(detail, nan_mask)
                return detail
            
            detail = gpu_arr.map_overlap(
                compute_detail_with_smooth,
                depth=common_depth,
                boundary='reflect',
                dtype=cp.float32,
                meta=cp.empty((0, 0), dtype=cp.float32),
                scale=scale
            )
            results.append(detail)
        
        # 驥阪∩莉倥″蜷域・縺ｨ繧ｰ繝ｭ繝ｼ繝舌Ν豁｣隕丞喧
        def weighted_combine_and_normalize(*blocks):
            """繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險磯㍼繧剃ｽｿ逕ｨ縺励※豁｣隕丞喧"""
            nan_mask = cp.isnan(blocks[0])
            result = cp.zeros_like(blocks[0])
            
            for i, block in enumerate(blocks):
                valid = ~cp.isnan(block)
                result[valid] += block[valid] * weights[i]
            
            # 繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險磯㍼縺ｧ豁｣隕丞喧
            if norm_max > norm_min:
                result = (result - norm_min) / (norm_max - norm_min)
                result = cp.clip(result, 0, 1)
            else:
                result = cp.full_like(result, 0.5)
            
            # 繧ｬ繝ｳ繝櫁｣懈ｭ｣
            result = cp.power(result, Constants.DEFAULT_GAMMA)
            
            # NaN菴咲ｽｮ繧貞ｾｩ蜈・
            result[nan_mask] = cp.nan
            
            return result.astype(cp.float32)
        
        combined = da.map_blocks(
            weighted_combine_and_normalize,
            *results,
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32)
        )
        return combined
    
    def get_default_params(self) -> dict:
        return {
            'scales': [1, 10, 50, 100],
            'weights': None,
            'downsample_factor': None,       # 繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚
            'verbose': False               # 繝・ヰ繝・げ蜃ｺ蜉・
        }

###############################################################################
# 2.6. Frequency Enhancement (蜻ｨ豕｢謨ｰ蠑ｷ隱ｿ) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def enhance_frequency_block(block: cp.ndarray, *, target_frequency: float = 0.1,
                          bandwidth: float = 0.05, enhancement: float = 2.0,
                          normalize: bool = True,  # 霑ｽ蜉
                          norm_min: float = None,   # 霑ｽ蜉
                          norm_max: float = None) -> cp.ndarray:  # 霑ｽ蜉
    """迚ｹ螳壼捉豕｢謨ｰ謌仙・縺ｮ蠑ｷ隱ｿ"""
    # NaN繝槭せ繧ｯ繧剃ｿ晏ｭ・
    nan_mask = cp.isnan(block)
    
    # NaN繧貞ｹｳ蝮・､縺ｧ荳譎ら噪縺ｫ蝓九ａ繧・
    if nan_mask.any():
        block_filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        block_filled = block
    
    # 遯馴未謨ｰ繧帝←逕ｨ・亥｢・阜縺ｧ縺ｮ荳埼｣邯壽ｧ繧定ｻｽ貂幢ｼ・
    window_y = cp.hanning(block.shape[0])[:, None]
    window_x = cp.hanning(block.shape[1])[None, :]
    window = window_y * window_x
    windowed_block = block_filled * window
    
    # 2D FFT
    fft = cp.fft.fft2(windowed_block)
    freq_x = cp.fft.fftfreq(block.shape[0])
    freq_y = cp.fft.fftfreq(block.shape[1])
    freq_grid = cp.sqrt(freq_x[:, None]**2 + freq_y[None, :]**2)
    
    # 繝舌Φ繝峨ヱ繧ｹ繝輔ぅ繝ｫ繧ｿ
    filter_mask = cp.exp(-((freq_grid - target_frequency)**2) / (2 * bandwidth**2))
    filter_mask = 1 + (enhancement - 1) * filter_mask
    
    # 繝輔ぅ繝ｫ繧ｿ驕ｩ逕ｨ
    filtered_fft = fft * filter_mask
    enhanced = cp.real(cp.fft.ifft2(filtered_fft))
    
    if normalize and norm_min is not None and norm_max is not None:
        # 繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險磯㍼縺ｧ豁｣隕丞喧
        if norm_max > norm_min:
            enhanced = (enhanced - norm_min) / (norm_max - norm_min)
        else:
            enhanced = cp.full_like(enhanced, 0.5)
        # 豁｣隕丞喧蠕後ｂ繧ｬ繝ｳ繝櫁｣懈ｭ｣繧帝←逕ｨ
        result = cp.power(enhanced, Constants.DEFAULT_GAMMA)
    else:
        # 豁｣隕丞喧縺ｪ縺励・蝣ｴ蜷医・縺昴・縺ｾ縺ｾ霑斐☆
        result = enhanced
    
    # NaN蜃ｦ逅・
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class FrequencyEnhancementAlgorithm(DaskAlgorithm):
    """蜻ｨ豕｢謨ｰ蠑ｷ隱ｿ繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        # 邨ｱ險磯㍼繧定ｨ育ｮ・
        stats = compute_global_stats(
            gpu_arr,
            freq_stat_func,
            enhance_frequency_block,
            {
                'target_frequency': params.get('target_frequency', 0.1),
                'bandwidth': params.get('bandwidth', 0.05),
                'enhancement': params.get('enhancement', 2.0),
                'normalize': False
            },
            downsample_factor=None,
            depth=32
        )
        
        return gpu_arr.map_overlap(
            enhance_frequency_block,
            depth=32,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            target_frequency=params.get('target_frequency', 0.1),
            bandwidth=params.get('bandwidth', 0.05),
            enhancement=params.get('enhancement', 2.0),
            normalize=True,
            norm_min=stats[0],
            norm_max=stats[1]
        )
    
    def get_default_params(self) -> dict:
        return {
            'target_frequency': 0.1,  # 0-0.5縺ｮ遽・峇
            'bandwidth': 0.05,
            'enhancement': 2.0
        }

###############################################################################
# 2.7. Curvature (譖ｲ邇・ 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_curvature_block(block: cp.ndarray, *, curvature_type: str = 'mean',
                          pixel_size: float = 1.0) -> cp.ndarray:
    """譖ｲ邇・ｨ育ｮ暦ｼ亥ｹｳ蝮・峇邇・√ぎ繧ｦ繧ｹ譖ｲ邇・∝ｹｳ髱｢繝ｻ譁ｭ髱｢譖ｲ邇・ｼ・"""
    # NaN繝槭せ繧ｯ繧剃ｿ晏ｭ・
    nan_mask = cp.isnan(block)
    
    # NaN繧帝團謗･蛟､縺ｧ荳譎ら噪縺ｫ蝓九ａ繧・
    if nan_mask.any():
        filled = cp.where(nan_mask, cp.nanmean(block), block)
    else:
        filled = block
    
    # 1谺｡蠕ｮ蛻・
    dy, dx = cp.gradient(filled, pixel_size, edge_order=2)
    
    # 2谺｡蠕ｮ蛻・
    dyy, dyx = cp.gradient(dy, pixel_size, edge_order=2)
    dxy, dxx = cp.gradient(dx, pixel_size, edge_order=2)
    
    if curvature_type == 'mean':
        # 蟷ｳ蝮・峇邇・
        p = dx
        q = dy
        r = dxx
        s = (dxy + dyx) / 2  # 蟇ｾ遘ｰ諤ｧ縺ｮ縺溘ａ蟷ｳ蝮・ｒ蜿悶ｋ
        t = dyy
        
        denominator = cp.power(1 + p**2 + q**2, 1.5)
        numerator = (1 + q**2) * r - 2 * p * q * s + (1 + p**2) * t
        
        curvature = -numerator / (2 * denominator + 1e-10)
        
    elif curvature_type == 'gaussian':
        # 繧ｬ繧ｦ繧ｹ譖ｲ邇・
        curvature = (dxx * dyy - dxy**2) / cp.power(1 + dx**2 + dy**2, 2)
        
    elif curvature_type == 'planform':
        # 蟷ｳ髱｢譖ｲ邇・ｼ育ｭ蛾ｫ倡ｷ壹・譖ｲ邇・ｼ・
        curvature = -2 * (dx**2 * dxx + 2 * dx * dy * dxy + dy**2 * dyy) / \
                   (cp.power(dx**2 + dy**2, 1.5) + 1e-10)
                   
    else:  # profile
        # 譁ｭ髱｢譖ｲ邇・ｼ域怙螟ｧ蛯ｾ譁懈婿蜷代・譖ｲ邇・ｼ・
        curvature = -2 * (dx**2 * dyy - 2 * dx * dy * dxy + dy**2 * dxx) / \
                   ((dx**2 + dy**2) * cp.power(1 + dx**2 + dy**2, 0.5) + 1e-10)
    
    # 譖ｲ邇・・蜿ｯ隕門喧・域ｭ｣雋縺ｧ濶ｲ蛻・￠・・
    # 豁｣縺ｮ譖ｲ邇・ｼ亥・・峨ｒ譏弱ｋ縺上∬ｲ縺ｮ譖ｲ邇・ｼ亥・・峨ｒ證励￥
    curvature_normalized = cp.tanh(curvature * 100)  # 諢溷ｺｦ隱ｿ謨ｴ
    result = (curvature_normalized + 1) / 2  # 0-1縺ｫ豁｣隕丞喧
    
    # 繧ｬ繝ｳ繝櫁｣懈ｭ｣
    result = cp.power(result, Constants.DEFAULT_GAMMA)
    
    # NaN蜃ｦ逅・
    result = restore_nan(result, nan_mask)
    
    return result.astype(cp.float32)

class CurvatureAlgorithm(DaskAlgorithm):
    """譖ｲ邇・い繝ｫ繧ｴ繝ｪ繧ｺ繝"""
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        curvature_type = params.get('curvature_type', 'mean')
        pixel_size = params.get('pixel_size', 1.0)
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            def _curv_spatial(block: cp.ndarray, *, radius: float, curvature_type: str, pixel_size: float) -> cp.ndarray:
                smoothed = _smooth_for_radius(block, radius)
                return compute_curvature_block(smoothed, curvature_type=curvature_type, pixel_size=pixel_size)

            responses = []
            for radius in radii:
                depth = max(3, int(float(radius) * 2 + 2))
                responses.append(
                    gpu_arr.map_overlap(
                        _curv_spatial,
                        depth=depth,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        radius=float(radius),
                        curvature_type=curvature_type,
                        pixel_size=pixel_size,
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)

        return gpu_arr.map_overlap(
            compute_curvature_block,
            depth=2,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            curvature_type=curvature_type,
            pixel_size=pixel_size
        )
    
    def get_default_params(self) -> dict:
        return {
            'curvature_type': 'mean',  # 'mean', 'gaussian', 'planform', 'profile'
            'pixel_size': 1.0,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }

###############################################################################
# 2.15. Fractal Anomaly (繝輔Λ繧ｯ繧ｿ繝ｫ逡ｰ蟶ｸ讀懷・) 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
###############################################################################

def compute_roughness_multiscale(
    block: cp.ndarray,
    radii: List[int],
    window_mult: int = 3,
    detrend: bool = True,
) -> cp.ndarray:
    """Compute per-scale roughness maps for fractal-style analysis."""
    nan_mask = cp.isnan(block)
    sigmas = []

    for r in radii:
        sigma = max(0.8, float(r * window_mult) / 6.0)
        if detrend:
            # Remove local trend first, then measure residual energy at the same scale.
            # This avoids slope-driven pseudo-fractal spikes.
            if nan_mask.any():
                trend, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
                residual = block - trend
                local_energy, _ = handle_nan_with_gaussian(
                    residual ** 2, sigma=sigma, mode='nearest'
                )
            else:
                trend = gaussian_filter(block, sigma=sigma, mode='nearest')
                residual = block - trend
                local_energy = gaussian_filter(residual ** 2, sigma=sigma, mode='nearest')
            rough = cp.sqrt(cp.maximum(local_energy, 1e-8))
        else:
            if nan_mask.any():
                local_mean, _ = handle_nan_with_gaussian(block, sigma=sigma, mode='nearest')
                local_mean_sq, _ = handle_nan_with_gaussian(
                    block ** 2, sigma=sigma, mode='nearest'
                )
            else:
                local_mean = gaussian_filter(block, sigma=sigma, mode='nearest')
                local_mean_sq = gaussian_filter(block ** 2, sigma=sigma, mode='nearest')
            variance = local_mean_sq - local_mean ** 2
            rough = cp.sqrt(cp.maximum(variance, 0.0))
        sigmas.append(rough)

    return cp.stack(sigmas, axis=-1)


def compute_fractal_dimension_block(block: cp.ndarray, *, 
                                  radii: List[int] = [4, 8, 16, 32, 64],
                                  normalize: bool = True,
                                  mean_global: float = None,
                                  std_global: float = None,
                                  relief_p10: float = None,
                                  relief_p75: float = None,
                                  smoothing_sigma: float = 1.2,
                                  despeckle_threshold: float = 0.35,
                                  despeckle_alpha_max: float = 0.30,
                                  detail_boost: float = 0.35) -> cp.ndarray:
    """Compute fractal anomaly from detrended multiscale roughness."""
    nan_mask = cp.isnan(block)

    # 1) Detrended multiscale roughness (stable under strong regional slope).
    sigmas = compute_roughness_multiscale(block, radii, window_mult=3, detrend=True)

    # 2) Scale-law fit in log space.
    fit_scales = cp.asarray(radii, dtype=cp.float32)
    log_scales = cp.log(fit_scales)
    log_sigmas = cp.log(cp.maximum(sigmas, 1e-5))

    scale_w = cp.sqrt(fit_scales)
    scale_w = scale_w / cp.sum(scale_w)
    w3 = scale_w.reshape(1, 1, -1)

    mean_log_scale = cp.sum(log_scales * scale_w)
    mean_log_sigma = cp.sum(log_sigmas * w3, axis=2)

    log_scales_broadcast = log_scales.reshape(1, 1, -1)
    cov = cp.sum(
        (log_scales_broadcast - mean_log_scale)
        * (log_sigmas - mean_log_sigma[:, :, None])
        * w3,
        axis=2,
    )
    var_log_scale = cp.sum(((log_scales - mean_log_scale) ** 2) * scale_w)

    beta = cov / (var_log_scale + 1e-10)

    y_fit = mean_log_sigma[:, :, None] + beta[:, :, None] * (log_scales_broadcast - mean_log_scale)
    ss_res = cp.sum(((log_sigmas - y_fit) ** 2) * w3, axis=2)
    ss_tot = cp.sum(((log_sigmas - mean_log_sigma[:, :, None]) ** 2) * w3, axis=2)
    r2 = cp.clip(1.0 - ss_res / (ss_tot + 1e-10), 0.0, 1.0)

    # 3) Tile-invariant feature components (avoid per-tile robust scaling seams).
    rmse = cp.sqrt(cp.maximum(ss_res, 0.0))
    beta_dev = cp.clip(beta - 1.2, -4.0, 4.0)
    rmse_comp = cp.log1p(cp.maximum(rmse, 0.0))

    # Relief confidence: suppress unstable behavior over near-flat/low-signal areas.
    roughness = cp.mean(sigmas, axis=2)
    valid_rough = roughness[~nan_mask]
    if (
        relief_p10 is not None
        and relief_p75 is not None
        and np.isfinite(relief_p10)
        and np.isfinite(relief_p75)
        and float(relief_p75) > float(relief_p10)
    ):
        r_p10 = float(relief_p10)
        r_p75 = float(relief_p75)
    elif valid_rough.size > 0:
        r_p10 = float(cp.percentile(valid_rough, 10))
        r_p75 = float(cp.percentile(valid_rough, 75))
    else:
        r_p10, r_p75 = 0.0, 1.0
    relief_conf = cp.clip((roughness - r_p10) / max(r_p75 - r_p10, 1e-6), 0.0, 1.0)

    # 4) Raw anomaly: macro-scale first, with controlled local detail injection.
    raw_feature = 0.75 * beta_dev + 0.45 * rmse_comp
    fine_i = 0
    coarse_i = min(2, log_sigmas.shape[2] - 1)
    fine_ratio = log_sigmas[:, :, fine_i] - log_sigmas[:, :, coarse_i]
    max_i = log_sigmas.shape[2] - 1
    macro_i = max(max_i - 2, 0)
    macro_ratio = log_sigmas[:, :, max_i] - log_sigmas[:, :, macro_i]
    raw_feature = raw_feature + 0.35 * macro_ratio * relief_conf
    raw_feature = raw_feature + float(detail_boost) * 0.18 * fine_ratio * relief_conf

    # 5) Confidence-aware denoising.
    smooth = max(0.0, float(smoothing_sigma))
    feat_smooth = raw_feature
    if smooth > 0:
        if nan_mask.any():
            feat_smooth, _ = handle_nan_with_gaussian(raw_feature, sigma=smooth, mode='nearest')
        else:
            feat_smooth = gaussian_filter(raw_feature, sigma=smooth, mode='nearest')
    alpha_r2 = cp.clip((r2 - 0.35) / 0.6, 0.0, 1.0)
    # Raise base alpha so local texture survives confidence smoothing.
    alpha = 0.50 + 0.50 * (alpha_r2 * relief_conf)
    feature_out = alpha * raw_feature + (1.0 - alpha) * feat_smooth

    # 6) Global normalization with smooth saturation to [-1, 1].
    if normalize and mean_global is not None and std_global is not None:
        if std_global > 1e-6:
            Z = (feature_out - mean_global) / std_global
            result = cp.tanh(Z / 2.5)
        else:
            result = cp.zeros_like(feature_out)
    else:
        result = feature_out

    # 7) Remove isolated impulses only where confidence and relief are both low.
    thr = max(0.05, float(despeckle_threshold))
    alpha_max = float(despeckle_alpha_max)
    med = median_filter(result, size=3, mode='nearest')
    thr_map = thr * (0.7 + 1.1 * alpha)
    despeckle_mask = (
        (cp.abs(result - med) > thr_map)
        & (alpha < alpha_max)
        & (relief_conf < 0.45)
    )
    result = cp.where(despeckle_mask, med, result)

    result = restore_nan(result, nan_mask)

    return result.astype(cp.float32)


def fractal_stat_func(data: cp.ndarray) -> Tuple[float, float]:
    """Compute robust center/scale for fractal-anomaly normalization."""
    valid_data = data[~cp.isnan(data)]
    if len(valid_data) > 0:
        p05 = float(cp.percentile(valid_data, 5))
        p95 = float(cp.percentile(valid_data, 95))
        if p95 > p05:
            center = 0.5 * (p05 + p95)
            scale = max(0.5 * (p95 - p05), 1e-6)
            return (center, scale)

        center = float(cp.median(valid_data))
        abs_dev = cp.abs(valid_data - center)
        mad = float(cp.median(abs_dev))
        scale = 1.4826 * mad if mad > 1e-9 else 0.5
        return (center, max(scale, 1e-6))
    return (0.0, 0.5)
class FractalAnomalyAlgorithm(DaskAlgorithm):
    """繝輔Λ繧ｯ繧ｿ繝ｫ逡ｰ蟶ｸ讀懷・繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝
    
    蝨ｰ蠖｢縺ｮ繝輔Λ繧ｯ繧ｿ繝ｫ谺｡蜈・ｒ險育ｮ励＠縲∫ｵｱ險育噪縺ｫ逡ｰ蟶ｸ縺ｪ鬆伜沺繧呈､懷・縺励∪縺吶・
    - 豁｣縺ｮ蛟､・域・繧九＞・・ 繝輔Λ繧ｯ繧ｿ繝ｫ谺｡蜈・′鬮倥＞ = 逡ｰ蟶ｸ縺ｫ隍・尅縺ｪ蝨ｰ蠖｢
    - 雋縺ｮ蛟､・域囓縺・ｼ・ 繝輔Λ繧ｯ繧ｿ繝ｫ谺｡蜈・′菴弱＞ = 逡ｰ蟶ｸ縺ｫ蟷ｳ貊代↑蝨ｰ蠖｢
    - 0莉倩ｿ托ｼ井ｸｭ髢楢牡・・ 蜈ｸ蝙狗噪縺ｪ蝨ｰ蠖｢繝代ち繝ｼ繝ｳ
    """
    
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        radii = params.get('radii', None)
        pixel_size = params.get('pixel_size', 1.0)
        smoothing_sigma = float(params.get('smoothing_sigma', 1.2))
        despeckle_threshold = float(params.get('despeckle_threshold', 0.35))
        despeckle_alpha_max = float(params.get('despeckle_alpha_max', 0.30))
        detail_boost = float(params.get('detail_boost', 0.35))
        relief_p10 = params.get('relief_p10', None)
        relief_p75 = params.get('relief_p75', None)
        
        # 蜊雁ｾ・・閾ｪ蜍墓ｱｺ螳・
        if radii is None:
            radii = self._determine_optimal_radii(pixel_size)
        
        # 譛菴・縺､縺ｮ繧ｹ繧ｱ繝ｼ繝ｫ繧堤｢ｺ菫晢ｼ亥盾閠・ｮ溯｣・・繝・ヵ繧ｩ繝ｫ繝医→蜷後§・・
        if len(radii) < 5:
            radii = [4, 8, 16, 32, 64]
        
        max_radius = max(radii)
        # window_mult=3繧定・・縺励◆depth
        depth = max_radius * 3 + 1
        
        # 繧ｰ繝ｭ繝ｼ繝舌Ν邨ｱ險磯㍼繧定ｨ育ｮ暦ｼ亥ｹｳ蝮・→讓呎ｺ門￥蟾ｮ・・

        stats = params.get('global_stats', None)
        stats_ok = (
            isinstance(stats, (tuple, list))
            and len(stats) >= 2
            and float(stats[1]) > 1e-9
        )
        if not stats_ok:
            num_blocks = int(np.prod(gpu_arr.numblocks)) if hasattr(gpu_arr, "numblocks") else 1
            if num_blocks > 1:
                stats = compute_global_stats(
                    gpu_arr,
                    fractal_stat_func,
                    compute_fractal_dimension_block,
                    {
                        'radii': radii,
                        'normalize': False,
                        'smoothing_sigma': smoothing_sigma,
                        'despeckle_threshold': despeckle_threshold,
                        'despeckle_alpha_max': despeckle_alpha_max,
                        'detail_boost': detail_boost,
                    },
                    downsample_factor=params.get('downsample_factor', None),
                    depth=depth,
                    algorithm_name='fractal_anomaly'
                )
            else:
                stats = (0.0, 0.5)
        if not (isinstance(stats, (tuple, list)) and len(stats) >= 2 and float(stats[1]) > 1e-9):
            stats = (0.0, 0.5)
        if relief_p10 is None and relief_p75 is None:
            if isinstance(stats, (tuple, list)) and len(stats) >= 4:
                relief_p10 = float(stats[2])
                relief_p75 = float(stats[3])

        mean_D, std_D = float(stats[0]), float(stats[1])
        
        # 繝輔Ν繧ｵ繧､繧ｺ縺ｧ蜃ｦ逅・ｼ域ｭ｣隕丞喧縺ゅｊ・・
        return gpu_arr.map_overlap(
            compute_fractal_dimension_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            radii=radii,
            normalize=True,
            mean_global=mean_D,
            std_global=std_D,
            relief_p10=relief_p10,
            relief_p75=relief_p75,
            smoothing_sigma=smoothing_sigma,
            despeckle_threshold=despeckle_threshold,
            despeckle_alpha_max=despeckle_alpha_max,
            detail_boost=detail_boost,
        )
    
    def _determine_optimal_radii(self, pixel_size: float) -> List[int]:
        """隗｣蜒丞ｺｦ縺ｫ蝓ｺ縺･縺・※譛驕ｩ縺ｪ蜊雁ｾ・ｒ豎ｺ螳・"""
        resolution_class = classify_resolution(pixel_size)
        
        if resolution_class == 'ultra_high':
            # 0.5m莉･荳・- 繧医ｊ蠎・＞遽・峇縺ｫ諡｡蠑ｵ
            base_radii = [4, 8, 16, 32, 64, 96]
        elif resolution_class == 'very_high':
            # 1m - 繧医ｊ蠎・＞遽・峇縺ｫ諡｡蠑ｵ
            base_radii = [4, 8, 16, 24, 32, 48]
        elif resolution_class == 'high':
            # 2.5m
            base_radii = [4, 8, 16, 32, 48, 64]
        elif resolution_class == 'medium':
            # 5m
            base_radii = [3, 6, 12, 24, 36, 48]
        elif resolution_class == 'low':
            # 10-15m - 螟ｧ蟷・↓諡｡蠑ｵ
            base_radii = [2, 4, 8, 16, 24, 32]
        else:
            # 30m莉･荳・
            base_radii = [2, 4, 8, 12, 16, 24]
        
        # 繝｡繝｢繝ｪ蛻ｶ邏・ｒ閠・・縺励※譛螟ｧ6縺､縺ｾ縺ｧ縺ｫ蛻ｶ髯撰ｼ・竊・縺ｫ蠅怜刈・・
        if len(base_radii) > 6:
            indices = cp.linspace(0, len(base_radii)-1, 6).astype(int).get()
            base_radii = [base_radii[int(i)] for i in indices]
        
        return base_radii
    
    def get_default_params(self) -> dict:
        return {
            'radii': None,  # None縺ｮ蝣ｴ蜷医・閾ｪ蜍墓ｱｺ螳・
            'pixel_size': 1.0,
            'downsample_factor': None,  # 邨ｱ險郁ｨ育ｮ玲凾縺ｮ繝繧ｦ繝ｳ繧ｵ繝ｳ繝励Ν菫よ焚
            'smoothing_sigma': 1.2,
            'despeckle_threshold': 0.35,
            'despeckle_alpha_max': 0.30,
            'detail_boost': 0.35,
        }

###############################################################################
# 2.13. Experimental Algorithms
###############################################################################

def compute_scale_space_surprise_block(
    block: cp.ndarray,
    *,
    scales: List[float],
    enhancement: float = 2.0,
    normalize: bool = True,
) -> cp.ndarray:
    """Scale-Space Surprise Map: 繧ｹ繧ｱ繝ｼ繝ｫ髢薙〒縺ｮ迚ｹ蠕ｴ螟牙喧驥上ｒ蠑ｷ隱ｿ"""
    nan_mask = cp.isnan(block)
    return kernel_scale_space_surprise(
        block,
        scales=scales,
        enhancement=enhancement,
        normalize=normalize,
        nan_mask=nan_mask,
    )


class ScaleSpaceSurpriseAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        scales = params.get('scales', [1.0, 2.0, 4.0, 8.0, 16.0])
        enhancement = float(params.get('enhancement', 2.0))
        normalize = bool(params.get('normalize', True))
        depth = int(max(1, cp.ceil(max(scales) * 3).item())) + 1

        return gpu_arr.map_overlap(
            compute_scale_space_surprise_block,
            depth=depth,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            scales=scales,
            enhancement=enhancement,
            normalize=normalize,
        )

    def get_default_params(self) -> dict:
        return {
            'scales': [1.0, 2.0, 4.0, 8.0, 16.0],
            'enhancement': 2.0,
            'normalize': True,
        }


def compute_multi_light_uncertainty_block(
    block: cp.ndarray,
    *,
    azimuths: List[float],
    altitude: float = 45.0,
    z_factor: float = 1.0,
    uncertainty_weight: float = 0.7,
    pixel_size: float = 1.0,
) -> cp.ndarray:
    """Multi-light Uncertainty Shading: 螟壽婿菴咲・譏弱〒荳咲｢ｺ螳滓ｧ繧貞庄隕門喧"""
    nan_mask = cp.isnan(block)
    return kernel_multi_light_uncertainty(
        block,
        azimuths=azimuths,
        altitude=altitude,
        z_factor=z_factor,
        uncertainty_weight=uncertainty_weight,
        pixel_size=pixel_size,
        nan_mask=nan_mask,
    )


def compute_multi_light_uncertainty_spatial_block(
    block: cp.ndarray,
    *,
    azimuths: List[float],
    altitude: float = 45.0,
    z_factor: float = 1.0,
    uncertainty_weight: float = 0.7,
    pixel_size: float = 1.0,
    radius: float = 4.0,
) -> cp.ndarray:
    smoothed = _smooth_for_radius(block, radius)
    return compute_multi_light_uncertainty_block(
        smoothed,
        azimuths=azimuths,
        altitude=altitude,
        z_factor=z_factor,
        uncertainty_weight=uncertainty_weight,
        pixel_size=pixel_size,
    )


class MultiLightUncertaintyAlgorithm(DaskAlgorithm):
    def process(self, gpu_arr: da.Array, **params) -> da.Array:
        azimuths = params.get('azimuths', [315.0, 45.0, 135.0, 225.0])
        altitude = float(params.get('altitude', 45.0))
        z_factor = float(params.get('z_factor', 1.0))
        uncertainty_weight = float(params.get('uncertainty_weight', 0.7))
        pixel_size = float(params.get('pixel_size', 1.0))
        mode = str(params.get("mode", "local")).lower()
        radii = _normalize_spatial_radii(params.get("radii"), pixel_size)
        weights = params.get("weights", None)
        agg = params.get("agg", "mean")

        if mode == "spatial":
            responses = []
            for radius in radii:
                depth = max(2, int(float(radius) * 2 + 1))
                responses.append(
                    gpu_arr.map_overlap(
                        compute_multi_light_uncertainty_spatial_block,
                        depth=depth,
                        boundary='reflect',
                        dtype=cp.float32,
                        meta=cp.empty((0, 0), dtype=cp.float32),
                        azimuths=azimuths,
                        altitude=altitude,
                        z_factor=z_factor,
                        uncertainty_weight=uncertainty_weight,
                        pixel_size=pixel_size,
                        radius=float(radius),
                    )
                )
            return _combine_multiscale_dask(responses, weights=weights, agg=agg)

        return gpu_arr.map_overlap(
            compute_multi_light_uncertainty_block,
            depth=1,
            boundary='reflect',
            dtype=cp.float32,
            meta=cp.empty((0, 0), dtype=cp.float32),
            azimuths=azimuths,
            altitude=altitude,
            z_factor=z_factor,
            uncertainty_weight=uncertainty_weight,
            pixel_size=pixel_size,
        )

    def get_default_params(self) -> dict:
        return {
            'azimuths': [315.0, 45.0, 135.0, 225.0],
            'altitude': 45.0,
            'z_factor': 1.0,
            'uncertainty_weight': 0.7,
            'mode': 'local',
            'radii': None,
            'weights': None,
        }


###############################################################################
# 2.14. 繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝繝ｬ繧ｸ繧ｹ繝医Μ
###############################################################################

ALGORITHMS = {
    'rvi': RVIAlgorithm(),
    'hillshade': HillshadeAlgorithm(),
    'slope': SlopeAlgorithm(),
    'specular': SpecularAlgorithm(),
    'atmospheric_scattering': AtmosphericScatteringAlgorithm(),
    'multiscale_terrain': MultiscaleDaskAlgorithm(),
    'frequency_enhancement': FrequencyEnhancementAlgorithm(),
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

# 譁ｰ縺励＞繧｢繝ｫ繧ｴ繝ｪ繧ｺ繝縺ｮ霑ｽ蜉萓・
# class AspectAlgorithm(DaskAlgorithm):
#     def process(self, gpu_arr: da.Array, **params) -> da.Array:
#         # 譁憺擇譁ｹ菴阪・險育ｮ・
#         pass
#     def get_default_params(self) -> dict:
#         return {'unit': 'degree', 'north_up': True}
# 
# ALGORITHMS['aspect'] = AspectAlgorithm()


