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
        
        # radiiのパース
        if parsed_args.radii:
            try:
                parsed_args.radii_list = [int(r.strip()) for r in parsed_args.radii.split(",")]
            except ValueError:
                self.parser.error("無効なradii形式です。カンマ区切りの整数を指定してください: 4,16,64,256")
        else:
            parsed_args.radii_list = None
        
        # weightsのパース
        if parsed_args.weights:
            try:
                parsed_args.weights_list = [float(w.strip()) for w in parsed_args.weights.split(",")]
            except ValueError:
                self.parser.error("無効なweights形式です。カンマ区切りの数値を指定してください: 0.4,0.3,0.2,0.1")
        else:
            parsed_args.weights_list = None
        
        return parsed_args
    
    def _validate_platform_args(self, args: argparse.Namespace):
        """Linux固有の引数検証"""
        # auto-radiiフラグの処理
        if hasattr(args, 'auto_radii') and hasattr(args, 'no_auto_radii'):
            args.auto_radii = args.auto_radii and not args.no_auto_radii
        
        # auto-sigmaフラグの処理（互換性）
        if hasattr(args, 'auto_sigma') and hasattr(args, 'no_auto_sigma'):
            args.auto_sigma = args.auto_sigma and not args.no_auto_sigma
        
        # RVIでradii/sigmaが未指定かつ自動決定も無効の場合エラー
        if args.algorithm == "rvi":
            if args.use_sigma_mode:
                # 従来モード
                if not args.sigma_list and not args.auto_sigma:
                    self.parser.error("sigmaモードではsigmaを指定するか、--auto-sigmaを有効にしてください")
            else:
                # 新モード（デフォルト）
                if not args.radii_list and not args.auto_radii:
                    self.parser.error("radiiを指定するか、--auto-radiiを有効にしてください")

    def execute(self, args: argparse.Namespace):
        """Dask-CUDA処理を実行"""
        # パラメータの準備
        params = self.get_common_params(args)
        
        # アルゴリズム固有パラメータの準備
        algo_params = {}
        
        if args.algorithm == "hillshade":
            algo_params.update({
                'azimuth': args.azimuth,
                'altitude': args.altitude,
                'z_factor': args.z_factor,
                'multiscale': args.multiscale,
            })
            if args.multiscale and args.sigma_list:
                algo_params['sigmas'] = args.sigma_list
                
        elif args.algorithm == "slope":
            algo_params['unit'] = args.unit
            
        elif args.algorithm == "curvature":
            algo_params['curvature_type'] = args.curvature_type
            
        elif args.algorithm == "tpi":
            algo_params['radius'] = args.radius
            
        elif args.algorithm == "lrm":
            algo_params['kernel_size'] = args.kernel_size
            
        elif args.algorithm == "openness":
            algo_params.update({
                'openness_type': args.openness_type,
                'num_directions': args.num_directions,
                'max_distance': args.max_distance,
            })
            
        elif args.algorithm in ["specular", "atmospheric_scattering", "ambient_occlusion",
                              "atmospheric_perspective", "npr_edges", "frequency_enhancement"]:
            if hasattr(args, 'intensity'):
                algo_params['intensity'] = args.intensity
        
        # ログ出力
        self.logger.info(f"=== Dask-CUDA地形解析 ===")
        self.logger.info(f"入力: {args.input}")
        self.logger.info(f"出力: {args.output}")
        self.logger.info(f"アルゴリズム: {args.algorithm}")
        
        if args.algorithm == "rvi":
            if args.use_sigma_mode:
                self.logger.info("モード: Sigma（従来方式）")
                if args.sigma_list:
                    self.logger.info(f"Sigma値: {args.sigma_list}")
                elif args.auto_sigma:
                    self.logger.info("Sigmaは地形解析により自動決定されます")
            else:
                self.logger.info("モード: Radius（高速方式）")
                if args.radii_list:
                    self.logger.info(f"半径: {args.radii_list}")
                elif args.auto_radii:
                    self.logger.info("半径は地形解析により自動決定されます")
        
        # 処理の実行
        try:
            from ..core.dask_processor import run_pipeline
            
            run_pipeline(
                src_cog=params['input_path'],
                dst_cog=params['output_path'],
                algorithm=args.algorithm,
                sigmas=params['sigma_list'] if args.use_sigma_mode else None,
                radii=getattr(args, 'radii_list', None),
                agg=args.agg,
                chunk=args.chunk,
                show_progress=params['show_progress'],
                auto_sigma=args.auto_sigma if args.use_sigma_mode else False,
                auto_radii=args.auto_radii if not args.use_sigma_mode else False,
                memory_fraction=args.memory_fraction,
                weights=getattr(args, 'weights_list', None),
                **algo_params
            )
            
        except ImportError as e:
            self.logger.error(f"Dask-CUDA環境が利用できません: {e}")
            self.logger.error("Linux環境用の依存関係をインストールしてください:")
            self.logger.error("pip install 'FujiShaderGPU[linux]'")
            raise
        except Exception as e:
            self.logger.error(f"処理中にエラーが発生しました: {e}")
            raise