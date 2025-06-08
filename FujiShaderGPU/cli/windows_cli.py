"""
FujiShaderGPU/cli/windows_cli.py
Windows環境用CLI - タイルベース処理の実装
"""
import os
from typing import List
import argparse
from .base import BaseCLI


class WindowsCLI(BaseCLI):
    """Windows環境向けCLI実装（タイルベース処理）"""
    
    def get_description(self) -> str:
        return "富士シェーダーGPU - 高速地形解析ツール (Windows/タイルベース処理)"
    
    def get_epilog(self) -> str:
        return """
使用例:
  # RVI（Ridge-Valley Index）計算
  fujishadergpu input.tif output.tif
  
  # Hillshade計算
  fujishadergpu input.tif output.tif --algorithm hillshade
  
  # マルチスケールRVI
  fujishadergpu input.tif output.tif --sigma 50,100,200
  
  # GPUタイプを指定
  fujishadergpu input.tif output.tif --gpu-type rtx4070
  
  # COG生成のみ（既存タイルから）
  fujishadergpu dummy.tif output.tif --cog-only --tmp-dir existing_tiles

注意: Windows環境では一部のアルゴリズムが利用できません。
      Linux環境ではより多くのアルゴリズムと高速な処理が可能です。
"""
    
    def get_supported_algorithms(self) -> List[str]:
        """Windows環境でサポートされているアルゴリズム"""
        return [
            "rvi_gaussian",
            "hillshade",
            "atmospheric_scattering",
            "composite_terrain",
            "curvature",
            "frequency_enhancement",
            "visual_saliency"
        ]
    
    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        """Windows固有の引数を追加"""
        # タイル処理関連
        parser.add_argument(
            "--tile-size",
            type=int,
            help="タイルサイズ（未指定時は自動検出）"
        )
        
        parser.add_argument(
            "--padding",
            type=int,
            help="タイル境界のパディング（未指定時は自動計算）"
        )
        
        parser.add_argument(
            "--max-workers",
            type=int,
            help="並列処理数（未指定時は自動検出）"
        )
        
        parser.add_argument(
            "--nodata-threshold",
            type=float,
            default=1.0,
            help="NoDataスキップ閾値 (default: 1.0)"
        )
        
        # GPU関連
        parser.add_argument(
            "--gpu-type",
            choices=["rtx4070", "t4", "l4", "a100", "auto"],
            default="auto",
            help="GPU種別指定 (default: auto)"
        )
        
        # モード関連
        parser.add_argument(
            "--single-scale",
            action="store_true",
            help="シングルスケールモードを強制"
        )
        
        parser.add_argument(
            "--no-auto-scale",
            action="store_true",
            help="自動スケール分析を無効化"
        )
        
        # アルゴリズム固有パラメータの追加
        parser.add_argument(
            "--azimuth",
            type=float,
            default=315.0,
            help="太陽の方位角 (度, default: 315, Hillshadeで使用)"
        )
        
        parser.add_argument(
            "--altitude",
            type=float,
            default=45.0,
            help="太陽の高度角 (度, default: 45, Hillshadeで使用)"
        )
        
        parser.add_argument(
            "--color-mode",
            choices=["warm", "cool", "grayscale"],
            default="warm",
            help="カラーモード (default: warm, Hillshadeで使用)"
        )
        
        parser.add_argument(
            "--cog-only",
            action="store_true",
            help="既存タイルからCOG生成のみ実行"
        )
    
    def _validate_platform_args(self, args: argparse.Namespace):
        """Windows固有の引数検証"""
        # COG生成のみモードの場合の特別処理
        if hasattr(args, 'cog_only') and args.cog_only:
            if not os.path.exists(args.tmp_dir):
                self.parser.error(f"--cog-only モードではタイルディレクトリが必要です: {args.tmp_dir}")
            # 入力ファイルチェックをスキップするためのフラグ
            args._skip_input_check = True
        
        # アルゴリズム固有の検証
        if args.algorithm == "rvi_gaussian":
            # RVIの場合、sigmaが未指定なら自動決定モードとする
            if not args.sigma_list and not hasattr(args, 'no_auto_scale'):
                self.logger.info("sigmaが未指定のため、地形解析による自動決定を行います")
    
    def execute(self, args: argparse.Namespace):
        """タイルベース処理を実行"""
        # 設定の確認
        from ..config.system_config import check_gdal_environment
        check_gdal_environment()
        
        # パラメータの準備
        params = self.get_common_params(args)
        
        # Windows固有パラメータの追加
        params.update({
            'tile_size': args.tile_size,
            'padding': args.padding,
            'max_workers': args.max_workers,
            'nodata_threshold': args.nodata_threshold,
            'gpu_type': args.gpu_type,
            'multiscale_mode': not args.single_scale,
            'auto_scale_analysis': not args.no_auto_scale,
            'cog_only': args.cog_only,
        })
        
        # ログ出力
        if args.cog_only:
            self.logger.info("=== COG生成のみモード ===")
            self.logger.info(f"タイルディレクトリ: {args.tmp_dir}")
            self.logger.info(f"出力: {args.output}")
        else:
            self.logger.info(f"入力: {args.input}")
            self.logger.info(f"出力: {args.output}")
            self.logger.info(f"アルゴリズム: {args.algorithm}")
            self.logger.info(f"モード: {'マルチスケール' if params['multiscale_mode'] else 'シングルスケール'}")
            if args.sigma_list:
                self.logger.info(f"Sigma値: {args.sigma_list}")
        
        # 処理の実行
        try:
            from ..core.tile_processor import process_dem_tiles
            
            # COG生成のみモードの場合、入力パスを調整
            if args.cog_only:
                params['input_path'] = params['input_path'] if os.path.exists(params['input_path']) else "dummy_input.tif"
            
            # アルゴリズム固有パラメータの準備
            algo_params = {}
            if args.algorithm == "hillshade":
                algo_params.update({
                    'azimuth': args.azimuth,
                    'altitude': args.altitude,
                    'color_mode': args.color_mode,
                })
            
            process_dem_tiles(
                input_cog_path=params['input_path'],
                output_cog_path=params['output_path'],
                tmp_tile_dir=params['tmp_dir'],
                algorithm=params['algorithm'],  # アルゴリズムを追加
                tile_size=params['tile_size'],
                padding=params['padding'],
                sigma=params['sigma_list'][0] if params['sigma_list'] else 10.0,
                max_workers=params['max_workers'],
                nodata_threshold=params['nodata_threshold'],
                gpu_type=params['gpu_type'],
                multiscale_mode=params['multiscale_mode'],
                pixel_size=params['pixel_size'],
                auto_scale_analysis=params['auto_scale_analysis'],
                cog_only=params['cog_only'],
                **algo_params  # アルゴリズム固有パラメータを展開
            )
            
            self.logger.info("✓ 処理完了")
            
        except Exception as e:
            self.logger.error(f"✗ エラー: {e}")
            raise