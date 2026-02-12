"""
FujiShaderGPU/algorithms/tile_shared.py
Windows/macOS向けタイルベース処理用の共通基盤

NOTE: 大部分のアルゴリズムは dask_shared.py のクラスを
tile/dask_bridge.py 経由で tile パスから利用しています。
このファイルには tile 専用の TileAlgorithm 基底クラスと、
共有カーネルに直接委譲する軽量アルゴリズムのみ残しています。
"""
import cupy as cp
from abc import ABC, abstractmethod
from typing import Dict, Any
from .common.kernels import (
    scale_space_surprise as kernel_scale_space_surprise,
    multi_light_uncertainty as kernel_multi_light_uncertainty,
)


class TileAlgorithm(ABC):
    """地形解析アルゴリズムの基底クラス"""

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """デフォルトパラメータを返す"""
        pass

    @abstractmethod
    def process(self, dem_gpu: cp.ndarray, **params) -> cp.ndarray:
        """
        GPU上でアルゴリズムを実行

        Parameters
        ----------
        dem_gpu : cp.ndarray
            GPU上のDEMデータ
        **params : dict
            アルゴリズム固有のパラメータ

        Returns
        -------
        cp.ndarray
            処理結果（GPU上）
        """
        pass


class ScaleSpaceSurpriseAlgorithm(TileAlgorithm):
    """スケール間での局所特徴変化を可視化"""

    def get_default_params(self):
        return {
            "scales": [1.0, 2.0, 4.0, 8.0, 16.0],
            "enhancement": 2.0,
            "normalize": True,
        }

    def process(self, dem_gpu, **params):
        p = self.get_default_params()
        p.update(params)
        return kernel_scale_space_surprise(
            dem_gpu,
            scales=p["scales"],
            enhancement=float(p["enhancement"]),
            normalize=bool(p["normalize"]),
        )


class MultiLightUncertaintyAlgorithm(TileAlgorithm):
    """複数方位ライトでの不確実性を重ねた陰影"""

    def get_default_params(self):
        return {
            "azimuths": [315.0, 45.0, 135.0, 225.0],
            "altitude": 45.0,
            "z_factor": 1.0,
            "uncertainty_weight": 0.7,
        }

    def process(self, dem_gpu, **params):
        p = self.get_default_params()
        p.update(params)
        return kernel_multi_light_uncertainty(
            dem_gpu,
            azimuths=p["azimuths"],
            altitude=float(p["altitude"]),
            z_factor=float(p["z_factor"]),
            uncertainty_weight=float(p["uncertainty_weight"]),
            pixel_size=float(p.get("pixel_size", 1.0)),
        )
