"""
FujiShaderGPU/cli/linux_cli.py
Linux環境用CLI - Dask-CUDA処理の実装
"""
from typing import List, Optional
import argparse
from .base import BaseCLI

class LinuxCLI(BaseCLI):
    """Linux環境向けCLI実装（Dask-CUDA処理）"""
    
    def get_description(self) -> str:
        return """富士シェーダーGPU - 超高速地形解析ツール (Linux/Dask-CUDA処理)
        
巨大DEM (200,000×200,000px COG) から各種地形解析を実行し、
Cloud-Optimized GeoTIFF として書き出します。"""
    
    def get_epilog(self) -> str:
        return """
    使用例:
    # RVI: 地形を解析して半径を自動決定（推奨・高速）
    fujishadergpu input.tif output.tif
    
    # RVI: 手動で半径を指定（新方式・高速）
    fujishadergpu input.tif output.tif --radii 4,16,64,256
    
    # RVI: 従来のsigma指定（互換性のため残存）
    fujishadergpu input.tif output.tif --sigma 10,20,40 --use-sigma-mode
    
    # その他のアルゴリズムは変更なし
    fujishadergpu input.tif output.tif --algo hillshade
    
    # 大きなチャンクサイズを指定
    fujishadergpu input.tif output.tif --algo rvi --chunk 4096

    利用可能な全アルゴリズム:
    rvi, hillshade, slope, tpi, lrm, openness, specular,
    atmospheric_scattering, multiscale_terrain, frequency_enhancement,
    curvature, visual_saliency, npr_edges, atmospheric_perspective,
    ambient_occlusion
    """
    
    def get_supported_algorithms(self) -> List[str]:
        """Linux環境でサポートされているアルゴリズム（全て）"""
        return [
            "rvi", "hillshade", "slope", "tpi", "lrm", "openness",
            "specular", "atmospheric_scattering", "multiscale_terrain",
            "frequency_enhancement", "curvature", "visual_saliency",
            "npr_edges", "atmospheric_perspective", "ambient_occlusion"
        ]
    
    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        """Linux固有の引数を追加"""
        # Dask処理関連
        parser.add_argument(
            "--chunk",
            type=int,
            help="チャンク幅 (px)。未指定時は自動決定"
        )
        
        parser.add_argument(
            "--memory-fraction",
            type=float,
            # default=0.8,  # この行を削除
            default=0.5,  # より保守的なデフォルト値に変更
            help="GPU メモリ使用率 (default: 0.5)"  # ヘルプテキストも更新
        )
        
        # マルチスケール処理
        parser.add_argument(
            "--agg",
            choices=["mean", "min", "max", "sum", "stack"],
            default="mean",
            help="複数スケールの集約方法 (default: mean)"
        )
        
        # RVI固有（新しいオプション）
        parser.add_argument(
            "--radii",
            type=str,
            help="RVIの計算半径（ピクセル）。カンマ区切り (例: 4,16,64,256)"
        )
        
        parser.add_argument(
            "--weights",
            type=str,
            help="各半径の重み。カンマ区切り (例: 0.4,0.3,0.2,0.1)"
        )
        
        parser.add_argument(
            "--auto-radii",
            action="store_true",
            default=True,
            help="半径を地形解析により自動決定 (RVIのみ, default: True)"
        )
        
        parser.add_argument(
            "--no-auto-radii",
            action="store_true",
            help="半径自動決定を無効化 (RVIのみ)"
        )
        
        parser.add_argument(
            "--use-sigma-mode",
            action="store_true",
            help="従来のsigmaベースモードを使用 (RVIのみ, 非推奨)"
        )
        
        # Hillshade固有
        parser.add_argument(
            "--azimuth",
            type=float,
            default=315,
            help="太陽の方位角 (度, default: 315)"
        )
        
        parser.add_argument(
            "--altitude",
            type=float,
            default=45,
            help="太陽の高度角 (度, default: 45)"
        )
        
        parser.add_argument(
            "--z-factor",
            type=float,
            default=1.0,
            help="垂直誇張率 (default: 1.0)"
        )
        
        parser.add_argument(
            "--multiscale",
            action="store_true",
            help="マルチスケールHillshadeを実行"
        )
        
        # Slope固有
        parser.add_argument(
            "--unit",
            choices=["degree", "percent", "radians"],
            default="degree",
            help="勾配の単位 (default: degree)"
        )
        
        # Curvature固有
        parser.add_argument(
            "--curvature-type",
            choices=["mean", "gaussian", "planform", "profile"],
            default="mean",
            help="曲率の種類 (default: mean)"
        )
        
        # TPI/Openness共通
        parser.add_argument(
            "--radius",
            type=int,
            default=10,
            help="解析半径 (ピクセル, default: 10)"
        )
        
        # LRM固有
        parser.add_argument(
            "--kernel-size",
            type=int,
            default=25,
            help="トレンド除去のカーネルサイズ (default: 25)"
        )
        
        # Openness固有
        parser.add_argument(
            "--openness-type",
            choices=["positive", "negative"],
            default="positive",
            help="開度のタイプ (default: positive)"
        )
        
        parser.add_argument(
            "--num-directions",
            type=int,
            default=16,
            help="探索方向数 (default: 16)"
        )
        
        parser.add_argument(
            "--max-distance",
            type=int,
            default=50,
            help="最大探索距離 (ピクセル, default: 50)"
        )
        
        # 汎用強度パラメータ
        parser.add_argument(
            "--intensity",
            type=float,
            default=1.0,
            help="効果の強度 (default: 1.0, 複数のアルゴリズムで使用)"
        )

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """引数をパース（基底クラスをオーバーライド）"""
        parsed_args = super().parse_args(args)
        
        # radiiのパース（属性の存在チェックを追加）
        if hasattr(parsed_args, 'radii') and parsed_args.radii:
            try:
                parsed_args.radii_list = [int(r.strip()) for r in parsed_args.radii.split(",")]
            except ValueError:
                self.parser.error("無効なradii形式です。カンマ区切りの整数を指定してください: 4,16,64,256")
        else:
            parsed_args.radii_list = None
        
        # weightsのパース（属性の存在チェックを追加）
        if hasattr(parsed_args, 'weights') and parsed_args.weights:
            try:
                parsed_args.weights_list = [float(w.strip()) for w in parsed_args.weights.split(",")]
            except ValueError:
                self.parser.error("無効なweights形式です。カンマ区切りの数値を指定してください: 0.4,0.3,0.2,0.1")
        else:
            parsed_args.weights_list = None
        
        return parsed_args
    
    def _validate_platform_args(self, args: argparse.Namespace):
        # auto-radiiフラグの処理
        if hasattr(args, 'auto_radii') and hasattr(args, 'no_auto_radii'):
            args.auto_radii = args.auto_radii and not args.no_auto_radii
        
        # auto-sigmaフラグの処理（互換性）
        if hasattr(args, 'auto_sigma') and hasattr(args, 'no_auto_sigma'):
            args.auto_sigma = args.auto_sigma and not args.no_auto_sigma
        
        # RVIでradii/sigmaが未指定かつ自動決定も無効の場合エラー
        if args.algorithm == "rvi":
            # use_sigma_modeのデフォルト値を設定
            if not hasattr(args, 'use_sigma_mode'):
                args.use_sigma_mode = False
                
            if args.use_sigma_mode:
                # 従来モード
                # sigma_listはbase.pyのparse_argsで設定されるので、そちらを使用
                if not getattr(args, 'sigma_list', None) and not getattr(args, 'auto_sigma', False):
                    self.parser.error("sigmaモードではsigmaを指定するか、--auto-sigmaを有効にしてください")
            else:
                # 新モード（デフォルト）
                # radiiが指定されているかチェック（radii_listはまだ存在しない）
                if not getattr(args, 'radii', None) and not getattr(args, 'auto_radii', True):
                    self.parser.error("radiiを指定するか、--auto-radiiを有効にしてください")

    def execute(self, args: argparse.Namespace):
        """Dask-CUDA処理を実行"""
        # パラメータの準備
        params = self.get_common_params(args)
        
        # ログ出力
        self.logger.info(f"=== Dask-CUDA地形解析 ===")
        self.logger.info(f"入力: {args.input}")
        self.logger.info(f"出力: {args.output}")
        self.logger.info(f"アルゴリズム: {args.algorithm}")
        
        # デフォルト値の設定
        if not hasattr(args, 'use_sigma_mode'):
            args.use_sigma_mode = False
        if not hasattr(args, 'auto_radii'):
            args.auto_radii = True
        if not hasattr(args, 'auto_sigma'):
            args.auto_sigma = False
        if not hasattr(args, 'radii'):
            args.radii = None
        if not hasattr(args, 'radii_list'):
            args.radii_list = getattr(args, 'radii_list', None)
        if not hasattr(args, 'weights'):
            args.weights = None
        if not hasattr(args, 'weights_list'):
            args.weights_list = getattr(args, 'weights_list', None)

        # アルゴリズム固有パラメータの準備
        algo_params = {}
        
        # ... 既存のアルゴリズム固有パラメータ処理 ...
        # 共通パラメータ
        if hasattr(args, 'intensity'):
            algo_params['intensity'] = args.intensity

        # Hillshade固有
        if args.algorithm == 'hillshade':
            if hasattr(args, 'azimuth'):
                algo_params['azimuth'] = args.azimuth
            if hasattr(args, 'altitude'):
                algo_params['altitude'] = args.altitude
            if hasattr(args, 'z_factor'):
                algo_params['z_factor'] = args.z_factor
            if hasattr(args, 'multiscale'):
                algo_params['multiscale'] = args.multiscale

        # Slope固有
        elif args.algorithm == 'slope':
            if hasattr(args, 'unit'):
                algo_params['unit'] = args.unit

        # Curvature固有
        elif args.algorithm == 'curvature':
            if hasattr(args, 'curvature_type'):
                algo_params['curvature_type'] = args.curvature_type

        # TPI固有
        elif args.algorithm == 'tpi':
            if hasattr(args, 'radius'):
                algo_params['radius'] = args.radius

        # LRM固有
        elif args.algorithm == 'lrm':
            if hasattr(args, 'kernel_size'):
                algo_params['kernel_size'] = args.kernel_size

        # Openness固有
        elif args.algorithm == 'openness':
            if hasattr(args, 'radius'):
                algo_params['radius'] = args.radius
            if hasattr(args, 'openness_type'):
                algo_params['openness_type'] = args.openness_type
            if hasattr(args, 'num_directions'):
                algo_params['num_directions'] = args.num_directions
            if hasattr(args, 'max_distance'):
                algo_params['max_distance'] = args.max_distance

        # Ambient Occlusion固有
        elif args.algorithm == 'ambient_occlusion':
            if hasattr(args, 'num_samples'):
                algo_params['num_samples'] = args.num_samples
            if hasattr(args, 'radius'):
                algo_params['radius'] = args.radius
        
        # RVI用ログ出力
        if args.algorithm == "rvi":
            if args.use_sigma_mode:
                self.logger.info("モード: Sigma（従来方式）")
                if params['sigma_list']:
                    self.logger.info(f"Sigma値: {params['sigma_list']}")
            else:
                self.logger.info("モード: Radius（高速方式・自動決定）")
        
        # 処理の実行
        try:
            from ..core.dask_processor import run_pipeline
            
            # RVIパラメータの安全な処理
            if args.algorithm == "rvi":
                if args.use_sigma_mode:
                    # Sigmaモード
                    rvi_params = {
                        'sigmas': params['sigma_list'],
                        'radii': None,
                        'weights': None,
                        'auto_sigma': getattr(args, 'auto_sigma', False),
                        'auto_radii': False,
                    }
                else:
                    # Radiusモード（デフォルト）
                    rvi_params = {
                        'sigmas': None,
                        'radii': args.radii_list,  # 手動指定がある場合のみ
                        'weights': args.weights_list,
                        'auto_sigma': False,
                        'auto_radii': args.radii_list is None,  # radiiが指定されていない場合は自動決定
                    }
            else:
                # RVI以外のアルゴリズム
                rvi_params = {}
            
            run_pipeline(
                src_cog=params['input_path'],
                dst_cog=params['output_path'],
                algorithm=args.algorithm,
                agg=getattr(args, 'agg', 'mean'),
                chunk=getattr(args, 'chunk', None),
                show_progress=params['show_progress'],
                memory_fraction=getattr(args, 'memory_fraction', 0.5),
                **rvi_params,
                **algo_params
            )
            
        except Exception as e:
            self.logger.error(f"処理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
            raise
        