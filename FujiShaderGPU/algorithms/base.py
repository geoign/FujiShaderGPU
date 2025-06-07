"""
FujiShaderGPU/algorithms/base.py
"""

from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    """すべての処理アルゴリズムの基底クラス"""
    
    @abstractmethod
    def process(self, dem_gpu, **params):
        """GPU上でDEMデータを処理"""
        pass
    
    @abstractmethod
    def get_default_params(self):
        """デフォルトパラメータを返す"""
        pass
