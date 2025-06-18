"""
FujiShaderGPU/cli/linux_cli.py
Linux環境用CLI - Dask-CUDA処理の実装
"""
from typing import List, Optional
import os, argparse, rasterio, GPUtil
import numpy as np
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
    ambient_occlusion, fractal_anomaly
    """
    
    def get_supported_algorithms(self) -> List[str]:
        """Linux環境でサポートされているアルゴリズム（全て）"""
        return [
            "rvi", "hillshade", "slope", "tpi", "lrm", "openness",
            "specular", "atmospheric_scattering", "multiscale_terrain",
            "frequency_enhancement", "curvature", "visual_saliency",
            "npr_edges", "atmospheric_perspective", "ambient_occlusion",
            "fractal_anomaly"
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
            default=0.4,  # より保守的なデフォルト値に変更
            help="GPU メモリ使用率 (default: 0.5)"  # ヘルプテキストも更新
        )

        parser.add_argument(
            "--pixel_size",
            type=float,
            default=1.0,
            help="ピクセルサイズ（メートル単位, default: 1.0）"
        )

        parser.add_argument(
            "--verbose",
            action="store_true",
            help="詳細なログ出力を有効化"
        )
        
        # マルチスケール処理
        parser.add_argument(
            "--agg",
            choices=["mean", "min", "max", "sum", "stack"],
            default="mean",
            help="複数スケールの集約方法 (default: mean)"
        )
        
        # RVI固有
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

        parser.add_argument(
            "--sigmas",
            type=str,
            help="マルチスケールHillshadeのsigma値。カンマ区切り (例: 1,2,4,8)"
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

        # Frequency Enhancement固有
        parser.add_argument(
            "--target-frequency",
            type=float,
            default=0.1,
            help="強調する周波数 (0-0.5の範囲, default: 0.1)"
        )

        parser.add_argument(
            "--bandwidth",
            type=float,
            default=0.05,
            help="周波数帯域幅 (default: 0.05)"
        )

        parser.add_argument(
            "--enhancement",
            type=float,
            default=2.0,
            help="強調の強度 (default: 2.0)"
        )

        # Specular固有
        parser.add_argument(
            "--roughness-scale",
            type=float,
            default=20.0,
            help="ラフネス計算のスケール (default: 20.0)"
        )

        parser.add_argument(
            "--shininess",
            type=float,
            default=10.0,
            help="光沢の強さ (default: 10.0)"
        )

        parser.add_argument(
            "--light-azimuth",
            type=float,
            default=315,
            help="光源の方位角 (度, default: 315)"
        )

        parser.add_argument(
            "--light-altitude",
            type=float,
            default=45,
            help="光源の高度角 (度, default: 45)"
        )

        # Atmospheric Scattering固有
        parser.add_argument(
            "--scattering-strength",
            type=float,
            default=0.5,
            help="大気散乱の強度 (default: 0.5)"
        )

        # Multiscale Terrain固有
        parser.add_argument(
            "--scales",
            type=str,
            help="マルチスケール地形のスケール。カンマ区切り (例: 1,10,50,100)"
        )

        parser.add_argument(
            "--mst-weights",  # weightsと衝突を避けるため
            type=str,
            help="マルチスケール地形の重み。カンマ区切り (例: 0.4,0.3,0.2,0.1)"
        )

        # Visual Saliency固有
        parser.add_argument(
            "--vs-scales",  # scalesと衝突を避けるため
            type=str,
            help="視覚的顕著性のスケール。カンマ区切り (例: 2,4,8,16)"
        )

        parser.add_argument(
            "--use-global-stats",
            action="store_true",
            default=True,
            help="グローバル統計を使用 (default: True)"
        )

        parser.add_argument(
            "--no-global-stats",
            action="store_true",
            help="グローバル統計を無効化"
        )

        parser.add_argument(
            "--downsample-factor",
            type=int,
            default=20,
            help="ダウンサンプル係数 (default: 20)"
        )

        # NPR Edges固有
        parser.add_argument(
            "--edge-sigma",
            type=float,
            default=1.0,
            help="エッジ検出のぼかし強度 (default: 1.0)"
        )

        parser.add_argument(
            "--threshold-low",
            type=float,
            default=0.2,
            help="エッジ検出の下限閾値 (default: 0.2)"
        )

        parser.add_argument(
            "--threshold-high",
            type=float,
            default=0.5,
            help="エッジ検出の上限閾値 (default: 0.5)"
        )

        # Atmospheric Perspective固有
        parser.add_argument(
            "--depth-scale",
            type=float,
            default=1000.0,
            help="深度スケール (default: 1000.0)"
        )

        parser.add_argument(
            "--haze-strength",
            type=float,
            default=0.7,
            help="ヘイズの強度 (default: 0.7)"
        )

        # Ambient Occlusion固有（num-samplesのみ追加）
        parser.add_argument(
            "--num-samples",
            type=int,
            default=16,
            help="AO計算のサンプル数 (default: 16)"
        )

        # Fractal Anomaly固有（新規追加）
        parser.add_argument(
            "--fractal-radii",
            type=str,
            help="フラクタル異常検出の計算半径。カンマ区切り (例: 2,4,8,16,32)"
        )

        parser.add_argument(
            "--auto-fractal-radii",
            action="store_true",
            default=True,
            help="フラクタル半径を解像度から自動決定 (default: True)"
        )

        parser.add_argument(
            "--no-auto-fractal-radii",
            action="store_true",
            help="フラクタル半径の自動決定を無効化"
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
        
        # scalesのパース（multiscale_terrain用）
        if hasattr(parsed_args, 'scales') and parsed_args.scales:
            try:
                parsed_args.scales_list = [float(s.strip()) for s in parsed_args.scales.split(",")]
            except ValueError:
                self.parser.error("無効なscales形式です。カンマ区切りの数値を指定してください: 1,10,50,100")
        else:
            parsed_args.scales_list = None

        # mst_weightsのパース（multiscale_terrain用）
        if hasattr(parsed_args, 'mst_weights') and parsed_args.mst_weights:
            try:
                parsed_args.mst_weights_list = [float(w.strip()) for w in parsed_args.mst_weights.split(",")]
            except ValueError:
                self.parser.error("無効なmst-weights形式です。カンマ区切りの数値を指定してください: 0.4,0.3,0.2,0.1")
        else:
            parsed_args.mst_weights_list = None

        # vs_scalesのパース（visual_saliency用）
        if hasattr(parsed_args, 'vs_scales') and parsed_args.vs_scales:
            try:
                parsed_args.vs_scales_list = [float(s.strip()) for s in parsed_args.vs_scales.split(",")]
            except ValueError:
                self.parser.error("無効なvs-scales形式です。カンマ区切りの数値を指定してください: 2,4,8,16")
        else:
            parsed_args.vs_scales_list = None

        # sigmasのパース（Hillshade用）
        if hasattr(parsed_args, 'sigmas') and parsed_args.sigmas and parsed_args.algorithm == 'hillshade':
            try:
                parsed_args.sigmas_list = [float(s.strip()) for s in parsed_args.sigmas.split(",")]
            except ValueError:
                self.parser.error("無効なsigmas形式です。カンマ区切りの数値を指定してください: 1,2,4,8")
        else:
            parsed_args.sigmas_list = None

        # fractal_radiiのパース（fractal_anomaly用）
        if hasattr(parsed_args, 'fractal_radii') and parsed_args.fractal_radii:
            try:
                parsed_args.fractal_radii_list = [int(r.strip()) for r in parsed_args.fractal_radii.split(",")]
            except ValueError:
                self.parser.error("無効なfractal-radii形式です。カンマ区切りの整数を指定してください: 2,4,8,16,32")
        else:
            parsed_args.fractal_radii_list = None

        return parsed_args
    
    def _validate_platform_args(self, args: argparse.Namespace):
        # auto-radiiフラグの処理
        if hasattr(args, 'auto_radii') and hasattr(args, 'no_auto_radii'):
            args.auto_radii = args.auto_radii and not args.no_auto_radii
        
        # auto-sigmaフラグの処理（互換性）
        if hasattr(args, 'auto_sigma') and hasattr(args, 'no_auto_sigma'):
            args.auto_sigma = args.auto_sigma and not args.no_auto_sigma

        # use_global_statsフラグの処理
        if hasattr(args, 'use_global_stats') and hasattr(args, 'no_global_stats'):
            args.use_global_stats = args.use_global_stats and not args.no_global_stats

        # auto_fractal_radiiフラグの処理
        if hasattr(args, 'auto_fractal_radii') and hasattr(args, 'no_auto_fractal_radii'):
            args.auto_fractal_radii = args.auto_fractal_radii and not args.no_auto_fractal_radii
        
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
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"]="0.80"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__SPILL"]="0.85"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__PAUSE"]="0.90"
        os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TERMINATE"]="0.98"
        os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT"]="30s"
        os.environ["DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP"]="60s"
        os.environ["DASK_DISTRIBUTED__DEPLOY__LOST_WORKER_TIMEOUT"]="60s"

        # 追加: RMM環境変数
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_memory_gb = gpus[0].memoryTotal / 1024
        else:
            gpu_memory_gb = 40  # デフォルトA100想定
        if gpu_memory_gb >= 40:
            os.environ["RMM_ALLOCATOR"]="pool"
            os.environ["RMM_POOL_SIZE"]="35GB"  # A100の場合
            os.environ["RMM_MAXIMUM_POOL_SIZE"]="38GB"  # VRAMの95%程度

        # パラメータの準備
        params = self.get_common_params(args)

        # pixel_sizeの自動取得（指定されていない場合）
        if not hasattr(args, 'pixel_size') or args.pixel_size is None:
            with rasterio.open(params['input_path']) as src:
                # CRSをチェック
                if src.crs and src.crs.is_geographic:
                    # 地理座標系（緯度経度）の場合
                    # データの中心緯度を取得
                    bounds = src.bounds
                    center_lat = (bounds.bottom + bounds.top) / 2
                    
                    # 緯度1度あたりの距離（メートル）
                    # 地球の半径を使用した近似計算
                    lat_rad = np.radians(center_lat)
                    meters_per_degree_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + \
                                        1.175 * np.cos(4 * lat_rad) - 0.0023 * np.cos(6 * lat_rad)
                    meters_per_degree_lon = 111412.84 * np.cos(lat_rad) - \
                                        93.5 * np.cos(3 * lat_rad) + 0.118 * np.cos(5 * lat_rad)
                    
                    # ピクセルサイズを度からメートルに変換
                    pixel_size_x_deg = abs(src.transform[0])
                    pixel_size_y_deg = abs(src.transform[4])
                    
                    pixel_size_x_m = pixel_size_x_deg * meters_per_degree_lon
                    pixel_size_y_m = pixel_size_y_deg * meters_per_degree_lat
                    
                    # 平均値を使用（通常はほぼ同じ値）
                    args.pixel_size = (pixel_size_x_m + pixel_size_y_m) / 2
                    
                    self.logger.info(f"地理座標系を検出: 中心緯度 {center_lat:.2f}°")
                    self.logger.info(f"ピクセルサイズ: {pixel_size_x_deg:.6f}° x {pixel_size_y_deg:.6f}°")
                    self.logger.info(f"メートル換算: {args.pixel_size:.2f}m")
                    
                else:
                    # 投影座標系の場合（すでにメートル単位など）
                    pixel_size_x = abs(src.transform[0])
                    pixel_size_y = abs(src.transform[4])
                    
                    # 単位を確認（CRSがある場合）
                    if src.crs:
                        units = src.crs.linear_units
                        if units and units.lower() != 'metre' and units.lower() != 'meter':
                            self.logger.warning(f"座標系の単位が'{units}'です。メートル単位として扱います。")
                    
                    args.pixel_size = pixel_size_x
                    self.logger.info(f"投影座標系: ピクセルサイズ {args.pixel_size:.2f}m")
        else:
            # ユーザーが明示的に指定した場合
            self.logger.info(f"ユーザー指定のピクセルサイズ: {args.pixel_size}m")

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
        if hasattr(args, 'pixel_size'):
            algo_params['pixel_size'] = args.pixel_size
        if hasattr(args, 'verbose'):
            algo_params['verbose'] = args.verbose

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
            if hasattr(args, 'sigmas_list') and args.sigmas_list:
                algo_params['sigmas'] = args.sigmas_list

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
        
        # Frequency Enhancement固有
        elif args.algorithm == 'frequency_enhancement':
            if hasattr(args, 'target_frequency'):
                algo_params['target_frequency'] = args.target_frequency
            if hasattr(args, 'bandwidth'):
                algo_params['bandwidth'] = args.bandwidth
            if hasattr(args, 'enhancement'):
                algo_params['enhancement'] = args.enhancement

        # Specular固有
        elif args.algorithm == 'specular':
            if hasattr(args, 'roughness_scale'):
                algo_params['roughness_scale'] = args.roughness_scale
            if hasattr(args, 'shininess'):
                algo_params['shininess'] = args.shininess
            if hasattr(args, 'light_azimuth'):
                algo_params['light_azimuth'] = args.light_azimuth
            if hasattr(args, 'light_altitude'):
                algo_params['light_altitude'] = args.light_altitude

        # Atmospheric Scattering固有
        elif args.algorithm == 'atmospheric_scattering':
            if hasattr(args, 'scattering_strength'):
                algo_params['scattering_strength'] = args.scattering_strength

        # Multiscale Terrain固有
        elif args.algorithm == 'multiscale_terrain':
            if hasattr(args, 'scales_list') and args.scales_list:
                algo_params['scales'] = args.scales_list
            if hasattr(args, 'mst_weights_list') and args.mst_weights_list:
                algo_params['weights'] = args.mst_weights_list

        # Visual Saliency固有
        elif args.algorithm == 'visual_saliency':
            if hasattr(args, 'vs_scales_list') and args.vs_scales_list:
                algo_params['scales'] = args.vs_scales_list
            if hasattr(args, 'use_global_stats'):
                # no_global_statsとの処理
                use_global = args.use_global_stats and not getattr(args, 'no_global_stats', False)
                algo_params['use_global_stats'] = use_global
            if hasattr(args, 'downsample_factor'):
                algo_params['downsample_factor'] = args.downsample_factor

        # NPR Edges固有
        elif args.algorithm == 'npr_edges':
            if hasattr(args, 'edge_sigma'):
                algo_params['edge_sigma'] = args.edge_sigma
            if hasattr(args, 'threshold_low'):
                algo_params['threshold_low'] = args.threshold_low
            if hasattr(args, 'threshold_high'):
                algo_params['threshold_high'] = args.threshold_high

        # Atmospheric Perspective固有
        elif args.algorithm == 'atmospheric_perspective':
            if hasattr(args, 'depth_scale'):
                algo_params['depth_scale'] = args.depth_scale
            if hasattr(args, 'haze_strength'):
                algo_params['haze_strength'] = args.haze_strength

        # Fractal Anomaly固有
        elif args.algorithm == 'fractal_anomaly':
            if hasattr(args, 'fractal_radii_list') and args.fractal_radii_list:
                algo_params['radii'] = args.fractal_radii_list

        # RVI用ログ出力
        if args.algorithm == "rvi":
            if args.use_sigma_mode:
                # Sigmaモード
                rvi_params = {
                    'mode': 'sigma',
                    'sigmas': params['sigma_list'],
                    'radii': None,
                    'weights': None,
                    'auto_sigma': getattr(args, 'auto_sigma', False),
                    'auto_radii': False,
                    'agg': getattr(args, 'agg', 'mean'),
                }
            else:
                # Radiusモード（デフォルト）
                rvi_params = {
                    'mode': 'radius',
                    'sigmas': None,
                    'radii': args.radii_list,
                    'weights': args.weights_list,
                    'auto_sigma': False,
                    'auto_radii': args.radii_list is None,
                    'agg': getattr(args, 'agg', 'mean'),
                }
        
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
        