"""
FujiShaderGPU/cli/windows_cli.py
Windows環境用CLI - タイルベース処理の実装
"""
import os
from typing import List
import argparse
from .base import BaseCLI
from ..core.tile_processor import DEFAULT_ALGORITHMS


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
  
  # Spatial RVI（半径を手動指定）
  fujishadergpu input.tif output.tif --algorithm rvi --mode spatial --radii 4,16,64 --weights 0.5,0.3,0.2
  
  # GPUタイプを指定
  fujishadergpu input.tif output.tif --gpu-type rtx4070
  
  # COG生成のみ（既存タイルから）
  fujishadergpu dummy.tif output.tif --cog-only --tmp-dir existing_tiles

注意: 現在は Windows/Linux とも同一のアルゴリズム名を利用できます。
      主な違いはバックエンド実装（Windows: タイル処理 / Linux: Dask-CUDA）です。
"""
    
    def get_supported_algorithms(self) -> List[str]:
        """Windows環境でサポートされているアルゴリズム"""
        return list(DEFAULT_ALGORITHMS.keys())
    
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

        parser.add_argument(
            "--nodata",
            type=str,
            default=None,
            help="NoData値を明示指定 (例: 0, -9999, nan)"
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

        parser.add_argument(
            "--mode",
            choices=["local", "spatial"],
            default="local",
            help="計算モード: local(近傍) / spatial(半径積算). spatialで半径未指定時はYAMLプリセットを使用"
        )

        parser.add_argument(
            "--radii",
            type=str,
            help="Spatial半径(px)を明示指定 (例: 4,16,64)。未指定時はpixel sizeに応じてYAML自動選択"
        )

        parser.add_argument(
            "--weights",
            type=str,
            help="Spatial重み (例: 0.5,0.3,0.2)。未指定時はYAML重み/等重みを自動適用"
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
            "--z-factor",
            type=float,
            default=None,
            help="Hillshade vertical exaggeration (default: 1.0)"
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
        parser.add_argument(
            "--cog-backend",
            choices=["internal", "external", "auto"],
            default="internal",
            help="COG生成バックエンド (default: internal)"
        )
        parser.add_argument(
            "--gdal-bin-dir",
            type=str,
            default=None,
            help="外部GDALのbinディレクトリ (例: C:\\Program Files\\GDAL)"
        )

        # Experimental algorithms
        parser.add_argument(
            "--surprise-scales",
            type=str,
            help="Scale-Space Surprise のスケール。カンマ区切り (例: 1,2,4,8,16)"
        )
        parser.add_argument(
            "--surprise-enhancement",
            type=float,
            default=2.0,
            help="Scale-Space Surprise の強調係数 (default: 2.0)"
        )
        parser.add_argument(
            "--ml-azimuths",
            type=str,
            help="Multi-lightの方位角。カンマ区切り (例: 315,45,135,225)"
        )
        parser.add_argument(
            "--uncertainty-weight",
            type=float,
            default=0.7,
            help="Multi-light uncertainty の重み (default: 0.7)"
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
        if args.algorithm == "rvi" and not hasattr(args, 'no_auto_scale'):
            self.logger.info("radii未指定時は地形解析による自動スケール決定を行います")
        if args.cog_backend == "external" and not args.gdal_bin_dir:
            self.logger.warning(
                "--cog-backend external では --gdal-bin-dir の明示指定を推奨します"
            )
    
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
            'cog_backend': args.cog_backend,
            'gdal_bin_dir': args.gdal_bin_dir,
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
            self.logger.info(f"空間モード: {args.mode}")
        
        # 処理の実行
        try:
            from ..core.tile_processor import process_dem_tiles
            
            # COG生成のみモードの場合、入力パスを調整
            if args.cog_only:
                params['input_path'] = params['input_path'] if os.path.exists(params['input_path']) else "dummy_input.tif"

            radii_list = None
            weights_list = None
            if getattr(args, "radii", None):
                try:
                    radii_list = [int(v.strip()) for v in args.radii.split(",") if v.strip()]
                except ValueError:
                    self.parser.error("Invalid --radii format. Use comma-separated integers: 4,16,64")
            if getattr(args, "weights", None):
                try:
                    weights_list = [float(v.strip()) for v in args.weights.split(",") if v.strip()]
                except ValueError:
                    self.parser.error("Invalid --weights format. Use comma-separated numbers: 0.5,0.3,0.2")
            
            # アルゴリズム固有パラメータの準備
            algo_params = {}
            if args.algorithm == "hillshade":
                algo_params.update({
                    'azimuth': args.azimuth,
                    'altitude': args.altitude,
                    'color_mode': args.color_mode,
                    'z_factor': args.z_factor,
                })
            elif args.algorithm == "scale_space_surprise":
                if args.surprise_scales:
                    algo_params['scales'] = [float(s.strip()) for s in args.surprise_scales.split(",")]
                algo_params['enhancement'] = args.surprise_enhancement
            elif args.algorithm == "multi_light_uncertainty":
                if args.ml_azimuths:
                    algo_params['azimuths'] = [float(a.strip()) for a in args.ml_azimuths.split(",")]
                algo_params['altitude'] = args.altitude
                algo_params['uncertainty_weight'] = args.uncertainty_weight

            algo_params['mode'] = args.mode
            if radii_list:
                algo_params['radii'] = radii_list
            if weights_list:
                algo_params['weights'] = weights_list
            
            process_dem_tiles(
                input_cog_path=params['input_path'],
                output_cog_path=params['output_path'],
                tmp_tile_dir=params['tmp_dir'],
                algorithm=params['algorithm'],  # アルゴリズムを追加
                tile_size=params['tile_size'],
                padding=params['padding'],
                sigma=10.0,
                max_workers=params['max_workers'],
                nodata_threshold=params['nodata_threshold'],
                nodata_override=self._parse_nodata_override(args.nodata),
                gpu_type=params['gpu_type'],
                multiscale_mode=params['multiscale_mode'],
                pixel_size=params['pixel_size'],
                auto_scale_analysis=params['auto_scale_analysis'],
                cog_only=params['cog_only'],
                cog_backend=params['cog_backend'],
                gdal_bin_dir=params['gdal_bin_dir'],
                **algo_params  # アルゴリズム固有パラメータを展開
            )
            
            self.logger.info("✓ 処理完了")
            
        except Exception as e:
            self.logger.error(f"✗ エラー: {e}")
            raise

    @staticmethod
    def _parse_nodata_override(raw_value):
        if raw_value is None:
            return None
        text = str(raw_value).strip().lower()
        if text in {"nan", "+nan", "-nan"}:
            return float("nan")
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid --nodata value: {raw_value}") from exc
