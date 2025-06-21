# 富士シェーダーGPU (FujiShaderGPU) 
## 役割
ギガピクセルサイズのDEMをGPUで処理し、地形表現ラスタを生成するオープンソースソフトウェア。

## 作者
Fumihiko IKEGAMI (Ikegami GeoResearch): https://github.com/geoign/FujiShaderGPU

## 実行環境
- CUDAが使えるLinux環境、特にGoogle ColabのA100 GPUランタイム(8 vCPU / 50GB DRAM / 200GB Storage)を想定。
- Windows環境のために、dask-cudaに依存しないtileベースの実装も保有。

## IO構造
- **入力データ**  
  GDALで読むことのできるDEMファイル。特にギガピクセルサイズの航空レーザー測量によるDEMをCloud Optimized GeoTIFFファイルにしたものを想定。
- **出力データ**  
  地形表現ラスタデータをCloud Optimized GeoTIFFファイルとして出力する。

## ソフトウェアを構成するファイルの機能と依存関係
現在のところ100%Pythonスクリプトで構成されている。

### FujiShaderGPU/algorithms/dask_algorithms.py  <!-- updated -->
*Linux 版 “Dask × CuPy” 並列バックエンドで動く **18 本**の地形解析アルゴリズム集。

---

#### 共通基盤
- **`Constants`** — ガンマ値・太陽方位/高度・NaN 埋め値・演算 ε などを集中管理。  
- **`DaskAlgorithm`** (ABC) — `process()` / `get_default_params()` を強制するインターフェース。  
- **ユーティリティ**  
  - 欠損補完 & フィルタ `handle_nan_*`, 勾配 `handle_nan_for_gradient`, NaN 復元。  
  - int16 変換 `compute_gamma_corrected_range`, `normalize_and_convert_to_int16`.  
  - **ダウンサンプル＋統計→正規化パイプライン**  
    `determine_optimal_downsample_factor` → `compute_global_stats` → `apply_global_normalization` で  
    *「サンプリング版で統計 → 元解像度へ一括適用」* を共通化。  
  - 地形特性で自動半径／σ決定  
    `determine_optimal_radii`, `determine_optimal_sigmas`.  

---

#### 実装アルゴリズム（18）
| Key | クラス | 主要パラメータ例 | 一行概要 |
|-----|--------|-----------------|----------|
| `rvi` | **RVIAlgorithm** | `radii, weights` | Ridge-Valley Index・半径重み自動決定 |
| `hillshade` | **HillshadeAlgorithm** | `azimuth, altitude, multiscale` | 陰影起伏（単/多スケール） |
| `slope` | **SlopeAlgorithm** | `unit, normalize` | 勾配角/％/rad |
| `aspect` | **AspectAlgorithm** | `unit` | 方位角マップ *(新規)* |
| `specular` | **SpecularAlgorithm** | `roughness_scale` | 金属光沢風ハイライト |
| `atmospheric_scattering` | **AtmosphericScatteringAlgorithm** | `scattering_factor` | 空気散乱による霞 |
| `multiscale_terrain` | **MultiscaleDaskAlgorithm** | `layers, weights` | 複合地形表現 *(新規)* |
| `frequency_enhancement` | **FrequencyEnhancementAlgorithm** | `target_frequency` | FFT バンドパス強調 |
| `curvature` | **CurvatureAlgorithm** | `curvature_type` | 平均/ガウス曲率 |
| `visual_saliency` | **VisualSaliencyAlgorithm** | `scales, pixel_size` | 解像度適応型顕著性 |
| `npr_edges` | **NPREdgesAlgorithm** | `edge_sigma, thresholds` | 非写実輪郭抽出 |
| `atmospheric_perspective` | **AtmosphericPerspectiveAlgorithm** | `depth_scale` | 遠近感トーン |
| `ambient_occlusion` | **AmbientOcclusionAlgorithm** | `radius, num_samples` | SSAO 風環境光遮蔽 |
| `tpi` | **TPIAlgorithm** | `radius` | Topographic Position Index |
| `lrm` | **LRMAlgorithm** | `kernel_size` | Local Relief Model |
| `openness` | **OpennessAlgorithm** | `max_distance` | 正負開度 (16 方位) |
| `fractal_anomaly` | **FractalAnomalyAlgorithm** | `radii, weights` | フラクタル度異常検出 |
| `frequency_enhancement` | **FrequencyEnhancementAlgorithm** | `target_frequency` | FFT 周波数強調 |

---

#### 外部依存
`cupy`, `cupyx.scipy.ndimage / fft`, `dask.array`, `numpy`, `logging`, `abc`, `typing`.  

---

#### 他モジュールとの連携
`core.dask_processor.run_pipeline()` が `ALGORITHMS[key]` レジストリ経由で各 `*Algorithm.process()` を呼び、  
巨大 COG DEM を GPU 分散処理 → int16 タイル → COG 書き出しの全体フローに組み込む。  


### FujiShaderGPU/core/dask_processor.py  <!-- updated -->
*GPU クラスタ構築から DEM 読み込み・アルゴリズム実行・COG 出力までを一括管理する **メインパイプライン**。

---

#### 主な責務
1. **Dask-CUDA クラスタ生成 (`make_cluster`)**  
   - RMMプールを CuPy に紐付け、メモリしきい値を `dask.config` へ一括設定。  
   - GPU/DRAM 容量・Google Colab 判定で挙動を動的に変更。  

2. **入力 DEM を Dask→CuPy へロード (`run_pipeline`)**  
   - COG を *lazy* チャンク読み込み (`rioxarray.open_rasterio` → `map_blocks(cp.asarray)`)。  
   - 解像度/総ピクセル数から **自動チャンクサイズ** を決定。  

3. **地形特性の高速サンプリング解析 (`analyze_terrain_characteristics`)**  
   - 高さ統計・勾配・曲率・FFT を 1 % サンプルで算出。  
   - `determine_optimal_radii/sigmas` と連携し **RVI / Fractal などの自動パラメータ決定** を実装。  

4. **アルゴリズム実行フェーズ**  
   - `ALGORITHMS[key].process()` を呼出し、結果を CuPy→NumPy 戻し。  
   - 小規模データは `client.persist`; 大規模 (>20 GB) はストリーム計算。  

5. **COG 出力ユーティリティ**  
   - **直接 COG 書込** (`_write_cog_da_original`) *GDAL ≥ 3.8*。  
   - **チャンク逐次書込** (`_write_cog_da_chunked_impl`)：VRAM/DRAM残量で自動選択。  
   - **GDAL CLI フォールバック** (`_fallback_cog_write` / `build_cog_with_overviews`)。  
   - `get_cog_options` で dtype 別 PREDICTOR など最適化パラメータを生成。  

6. **後処理 & クリーンアップ**  
   - `cleanup_gpu_memory`, `client.shutdown()`, `cluster.close()` で GPU/worker を確実に解放。  

---

#### 主要関数・クラス
| 関数 | 役割 | 備考 |
|------|------|------|
| `get_optimal_chunk_size` | GPU VRAM に合わせたチャンクピクセル数 | 512 整列 |
| `make_cluster` | `LocalCUDACluster` + `Client` 構築 | RMM 初期化・ダッシュボードリンク出力 |
| `analyze_terrain_characteristics` | 1 % サンプルで統計＋FFT | slope/curvature/周波数ピーク |
| `write_cog_da_chunked` | データ容量⇆RAM と比較し書込モード自動選択 | 大容量は chunked |
| `run_pipeline` | **エンドツーエンド実行** | 引数でアルゴリズム・radii/sigmas 等を制御 |

---

#### 外部依存
`cupy`, `dask.array`, `dask_cuda`, `distributed`, `rmm`, `rasterio`, `rioxarray`, `xarray`, `GPUtil`, `psutil`, `GDAL CLI`, `tqdm`, `logging`, `subprocess`.  

---

#### 他モジュールとの連携
- **`algorithms.dask_algorithms`** — `ALGORITHMS` レジストリ経由で各 *Algorithm* を呼出し。  
- **`determine_optimal_radii/sigmas`** を利用し RVI 等の *auto* パラメータを取得。  
- パイプライン結果は **`write_cog_da_*`** 関数で COG 出力され、後続のビューワ／ウェブ配信へ直接利用可能。  


### FujiShaderGPU/algorithms/tile_algorithms.py

- **主要機能 (タイル単位GPUアルゴリズム集)**  
  - `RVIGaussianAlgorithm` — Ridge-Valley Index（単一/マルチスケール）  
  - `AtmosphericScatteringAlgorithm` — 多重TPI＋天空可視性による大気散乱表現  
  - `CompositeTerrainAlgorithm` — Hillshade・大気散乱・RVI 等を重み付け合成しトーンマッピング  
  - `CurvatureAlgorithm` — マルチスケール平均/ガウス曲率の色付け表示  
  - `FrequencyEnhancementAlgorithm` — FFTバンドパス強調＋任意でヒルシェード合成  
  - `HillshadeAlgorithm` — 方位・高度・色モード可変の GPU Hillshade  
  - `VisualSaliencyAlgorithm` — 勾配＆センターサラウンド差分による顕著性強調  

- **共通ベース**  
  - `TileAlgorithm` (ABC) — `get_default_params()` と `process()` を規定し、各アルゴリズムが継承  

- **主な外部依存**  
  - **GPU 配列/FFT/フィルタ**: `cupy`, `cupyx.scipy.ndimage`, `cupyx.scipy.fft`  
  - **標準**: `abc`, `typing`

- **内部依存関係・呼び出し**  
  - `CompositeTerrainAlgorithm` が本ファイル内の `HillshadeAlgorithm`, `AtmosphericScatteringAlgorithm`, `RVIGaussianAlgorithm` を再利用  
  - `FrequencyEnhancementAlgorithm` → `HillshadeAlgorithm`（オプションで陰影付加）  
  - 全クラスは CuPy 配列を受け取り RGB またはスカラーを返し、上位タイルパイプラインから呼び出される想定

- **ファイルの役割**  
  Windows/macOS のローカル GPU 環境向けに、**1 タイル単位で実行できる地形可視化アルゴリズムを提供**。  
  Dask 分散版（`dask_algorithms.py`）に対し、こちらはスタンドアロン/小規模データ処理の高速実装を担う。


### FujiShaderGPU/cli/linux_cli.py
*Linux 環境用 Dask-CUDA ラッパー CLI。巨大 DEM を GPU 分散パイプラインへ橋渡しするフロントエンド。

---

#### 主な責務
1. **引数パーサ生成 (`_add_platform_specific_args`)**  
   - 共通 (`--chunk`, `--memory-fraction`, `--pixel_size`, `--verbose`) と  
     **18 アルゴリズム**それぞれの専用オプションを一括定義。  
2. **引数解析 (`parse_args`)**  
   - カンマ区切り文字列 → `list[int|float]` 変換。  
   - Radii/Weights/Scales 他、存在チェックで安全に追加属性を付与。  
3. **プラットフォーム検証 (`_validate_platform_args`)**  
   - *auto* フラグ整合性、RVI の半径／σモード依存関係を確認。  
4. **実行 (`execute`)**  
   - **GPU メモリ依存 RMM プール・Dask 環境変数** を自動設定。  
   - 入力 COG の CRS を判定し **度→メートル換算で `pixel_size` 自動推定**。  
   - アルゴリズム別 `algo_params` と RVI 専用パラメータを組み立て、  
     `core.dask_processor.run_pipeline()` を呼び出す。  
   - 例外をロギングして再送出。  

---

#### サブ機能・ユーティリティ
- **`get_description` / `get_epilog`** — CLI の概要・使用例を日本語で提供。  
- **`get_supported_algorithms`** — Linux 版で利用可能なアルゴリズム一覧を返却。  
- **CRS 依存ピクセルサイズ推定** — 緯度依存の地球半径式で度→m を近似。  

---

#### 外部依存
`argparse`, `os`, `rasterio`, `GPUtil`, `numpy`, `typing`,  
**内部**: `BaseCLI`, `core.dask_processor.run_pipeline`.  

---

#### 他モジュールとの連携
- **`core.dask_processor`** — 実際の Dask-CUDA パイプラインを実行。  
- **`algorithms.dask_algorithms`** — CLI オプションが `process()` の引数へ渡る。  


### FujiShaderGPU/core/tile_processor.py

- **主要機能 (タイルベース処理パイプライン)**  
  - `_load_algorithm()` ― `algorithms/tile_algorithms.py` から動的にアルゴリズムをロード。  
    - `rvi_gaussian` は関数実装をラップした特殊クラスで処理。  
  - `process_single_tile()` ― 1 タイルを GPU で計算して GeoTIFF に書き出す。  
    1. COG からウィンドウ読み込み → NoData マスク → `cupy` 転送  
    2. 選択アルゴリズム実行（単一／マルチスケール RVI ほか）  
    3. 結果を CPU へ戻し、コア領域を GeoTIFF で保存  
  - `process_dem_tiles()` ― DEM 全体をタイル分割し `ThreadPoolExecutor` で並列実行。  
    - ピクセルサイズ自動検出・スケール解析 → GPU 設定取得 (`get_gpu_config`)  
    - 進捗／エラー管理しながらタイルを生成 → `_build_vrt_and_cog_ultra_fast()` で最終 COG 合成  
  - `resume_cog_generation()` ― 既存タイルから COG だけを再構築・検証。  

- **主な外部依存**  
  - GPU 数値計算 / フィルタ: `cupy`, `cupyx.scipy.ndimage`  
  - 画像 I/O: `rasterio` (+ `Window`, `Affine`)  
  - 並列処理: `concurrent.futures.ThreadPoolExecutor`  
  - 汎用: `numpy`, `math`, `glob`, `shutil`, `os`, `logging`  

- **内部依存関係**  
  - GPUメモリ管理: `core.gpu_memory.gpu_memory_pool`  
  - GPU設定: `config.system_config.get_gpu_config`  
  - DEMメタ解析: `io.raster_info.detect_pixel_size_from_cog`  
  - スケール解析: `utils.scale_analysis.analyze_terrain_scales`, `_get_default_scales`  
  - NoData高速処理: `utils.nodata_handler._handle_nodata_ultra_fast`  
  - COG/VRT ビルド & 検証: `io.cog_builder._build_vrt_and_cog_ultra_fast`, `io.cog_validator._validate_cog_for_qgis`  
  - アルゴリズム群: `algorithms.tile_algorithms.*`（デフォルト 6 種＋RVI 特殊ケース）  

- **ファイルの役割**  
  Windows/macOS 向けに **巨大 DEM を GPU+マルチスレッドでタイル処理**し、  
  軽量かつ高速に最終 COG を生成するワークフローを提供する。

### FujiShaderGPU/io/cog_builder.py

- **主要機能 (VRT / COG 超高速ビルド)**  
  - `_build_vrt_and_cog_ultra_fast()` — タイル一式から  
    1) **VRT 統合** → 2) **COG 生成** → 3) **オーバービュー作成** を一気通貫で実行。  
  - `_create_vrt_ultra_fast()` & `_create_vrt_command_line_ultra()` — Python API と `gdalbuildvrt` CLI を自動切替して最速で VRT を構築。  
  - `_create_cog_ultra_fast()` — GDAL “COG” ドライバが使える環境では直接 COG 出力 (QGIS 最適化オプション付き)。  
  - `_create_cog_gtiff_ultra_fast()` — ドライバが無い場合は GeoTIFF → 手動オーバービュー → ファイル移動で擬似 COG を生成。  
  - `_create_qgis_optimized_overviews()` / `_create_overviews_gdal_api()` — `gdaladdo` 失敗時に API へフォールバックしつつ多段階ピラミッドを作成。  

- **主な外部依存**  
  - **GDAL Python & CLI**: `osgeo.gdal`, `gdalbuildvrt`, `gdaladdo`, `gdal_translate`  
  - **OS/ユーティリティ**: `os`, `glob`, `shutil`, `subprocess`, `time`  
  - **型ヒント**: `typing.List`  
  - **内部設定**: `config.gdal_config._configure_gdal_ultra_performance` — VRAM量に合わせた GDAL キャッシュ＆スレッド数を設定。  

- **内部依存関係・呼び出し元**  
  - `core.tile_processor._build_vrt_and_cog_ultra_fast()` ほかで呼び出され、タイル処理後の最終 COG 化を担当。  
  - `_build_vrt_and_cog_ultra_fast()` 内から前述の VRT/COG/Ovr 各関数を順次呼び出し。  

- **ファイルの役割**  
  富士シェーダーGPU における **“仕上げ工程”** — 膨大な GPU 生成タイルを高速に統合し、  
  QGIS 表示に最適化された Cloud Optimized GeoTIFF を数十秒〜数分で完成させるユーティリティ。


### FujiShaderGPU/cli/windows_cli.py

- **主要機能 (Windows 向け CLI ラッパ)**  
  - `WindowsCLI` クラスが **タイルベース GPU 処理** を Windows 用に公開。  
  - `get_supported_algorithms()` で利用可能アルゴリズム 7 種（rvi_gaussian, hillshade など）を制限提示 
  - `_add_platform_specific_args()` がタイル制御・GPU選択・Hillshade 専用色設定など Windows 固有フラグを追加 
  - `_validate_platform_args()` で `--cog-only` ディレクトリ検証と RVI パラメータ自動決定モードの通知 
    1. `config.system_config.check_gdal_environment()` で GDAL 環境確認  
    2. `get_common_params()` から共通設定取得＋Windows 追加パラメータを統合  
    3. Hillshade などアルゴリズム固有パラメータを組み立て  
    4. `core.tile_processor.process_dem_tiles()` を呼び出し実際の GPU タイル処理／COG 生成を実行 

- **主な外部依存**  
  - 標準: `argparse`, `os`, `logging`  
  - 型定義: `typing.List`  

- **内部依存関係**  
  - 共通 CLI 機能を持つ `cli/base.py::BaseCLI` を継承 
  - 処理コア: `core.tile_processor.process_dem_tiles()` にパラメータを委譲 
  - GDAL／システム環境チェック: `config.system_config.check_gdal_environment()`  

- **ファイルの役割**  
  富士シェーダーGPU を **Windows ターミナルから簡便に起動**させるエントリーポイント。  
  Windows 限定オプションでタイル分割サイズや GPU 種別を最適化しつつ、  
  タイル処理パイプライン (tile_processor) に橋渡しして巨大 DEM を高速 COG へ変換する。


### FujiShaderGPU/config/system_config.py

- **主要機能 (システム自動最適化)**  
  - `get_gpu_config()` — GPU 名と VRAM 量を検出し、タイルサイズ・並列ワーカー数・パディング・バッチサイズ・プリフェッチ数を GPU タイプ別（RTX 4070 / T4 / L4 / A100）に自動設定。Colab 環境ではメモリ制限に合わせてさらに縮小調整を行う。
  - `detect_optimal_system_config()` — CPU コア数、システム RAM、CUDA GPU の名称・VRAM・演算能力、Google Colab 判定などを詳細に取得し、最適化レベル（standard〜ultra）を分類。  
  - `check_gdal_environment()` — GDAL バージョンと COG / GTiff ドライバの有無をチェックし、QGIS 表示最適化 (512×512 ブロック・ZSTD 圧縮・多段階 Overview) をガイドする。 

- **主な外部依存**  
  - **GPU 情報** : `cupy`  
  - **システム統計** : `multiprocessing`, `psutil`, `math`  
  - **GIS ライブラリ** : `osgeo.gdal`  

- **内部依存関係・呼び出し元**  
  - `core.tile_processor`, `core.dask_processor` などが `get_gpu_config()` を呼び出し、タイルベース / Dask-CUDA パイプラインのパラメータを決定。  
  - CLI 層（例 : `cli/windows_cli.py`）は実行前に `check_gdal_environment()` を実行して GDAL 環境を確認。

- **ファイルの役割**  
  富士シェーダーGPU 全体の **ハードウェア＆GDAL キャパシティ検出エンジン**。システムごとの差異を吸収し、GPU パワーと GDAL 機能に合わせた安全かつ高速な既定設定を提供する基盤ユーティリティ。


### FujiShaderGPU/utils/scale_analysis.py

- **主要機能 (地形スケール自動解析)**  
  - `analyze_terrain_scales()` — 入力 COG の中央 8k² サンプルを取得し、GPU/CuPy バッチで **RVI 分散** を高速評価して最適スケール距離と重みを抽出。CPU/Scipy フォールバックあり。 :contentReference[oaicite:0]{index=0}  
  - `_analyze_scale_variances_ultra_fast()` — CuPy + `cupyx.scipy.ndimage.gaussian_filter` により複数 σ を並列適用し分散を算出。
  - `_select_optimal_scales_enhanced()` — 正規化分散の上位ピークを取得し、距離逆数で指数重みを決定（最大 5 スケール）。
  - `_analyze_scale_variances_scipy_fast()` / `_get_default_scales()` — Scipy CPU 版と安全デフォルト値。

- **主な外部依存**  
  - GPU 数値計算: `cupy`, `cupyx.scipy.ndimage`  
  - CPU 数値計算: `numpy`, `scipy.ndimage`（任意）  
  - ラスター I/O: `rasterio`  
  - 型定義: `typing`  

- **内部依存関係・呼び出し元**  
  - `core.tile_processor.process_dem_tiles()` ほかが本関数を呼び、アルゴリズム群（RVI, Hillshade 合成など）に渡すスケール距離 & ウェイトを決定。  

- **ファイルの役割**  
  DEM の微地形〜広域地形を **自動で最適スケール抽出**し、富士シェーダーGPU 全体のマルチスケール表現を強化するキーユーティリティ。


### FujiShaderGPU/cli/base.py

- **主要機能 (共通 CLI フレームワーク)**  
  - `BaseCLI` 抽象基底クラスを定義し、Linux/Windows CLI 実装に共通の  
    - **引数パーサ生成** (`_create_parser`)  
        - 必須: `input`, `output`  
        - 共通オプション: `--algorithm`, `--sigma`, `--pixel-size`, `--tmp-dir`, `--log-level`, `--no-progress`, `--force`  
        - 各派生クラスが `_add_platform_specific_args` で固有オプションを追加  
    - **引数前処理 & 検証** (`parse_args`)  
        - `sigma` をカンマ区切り → `sigma_list` へ変換し型チェック  
        - ログ設定と `_validate_platform_args` 呼び出し  
    - **実行テンプレート** (`run`)  
        - 入出力ファイル存在チェック → 派生 `execute()` を起動  
    - **共通パラメータ辞書** (`get_common_params`) — CLI 層→コア層への橋渡し  
  - ABC メソッド: `get_description`, `get_epilog`, `get_supported_algorithms`, `_add_platform_specific_args`, `_validate_platform_args`, `execute`

- **主な外部依存**  
  - `argparse`, `logging`, `os`  
  - 型 & 抽象化: `typing`, `abc`  

- **内部依存関係**  
  - `cli/linux_cli.py::LinuxCLI` と `cli/windows_cli.py::WindowsCLI` が本クラスを継承し、  
    プラットフォーム固有オプション／実行ロジックを実装。

- **ファイルの役割**  
  富士シェーダーGPU CLI シリーズの **コア・テンプレート**。  
  各 OS 向けサブクラスがこの基盤を拡張して一貫した UX とパラメータ受け渡しを実現する。


### FujiShaderGPU/io/cog_validator.py

- **主要機能 (COG 品質チェック)**  
  - `_validate_cog_for_qgis(cog_path)` — GDAL で COG を開き、  
    1. サイズ・バンド数・ブロック形状、  
    2. オーバービュー階層 & 圧縮方式、  
    3. メタデータ `LAYOUT` を確認し COG 準拠可否を判定。  
  - 上記情報から **QGIS 最適化スコア (0–100)** を算出し、タイル化やオーバービュー不足など改善提案をコンソールに表示。  
  - スコア ≥ 60 を合格として `True/False` を返却。例外時は `False`。

- **主な外部依存**  
  - `os` — ファイルサイズ取得  
  - `osgeo.gdal` — COG メタ情報の取得とバンド／オーバービュー検査

- **内部依存関係・呼び出し元**  
  - `core.tile_processor.resume_cog_generation()` や CLI 層が最終 COG を検証する際に利用。  

- **ファイルの役割**  
  生成された Cloud Optimized GeoTIFF が **QGIS で高速表示できる品質基準を満たすかを自動評価**し、ユーザへ改善ポイントを提示するユーティリティ。


### FujiShaderGPU/config/gdal_config.py

- **主要機能 (GDAL 超高速チューニング)**  
  - `_configure_gdal_ultra_performance(gpu_config)`  
    - `gpu_config["system_info"]` 内の **GPU 名・搭載メモリ** に応じて GDAL 環境変数を自動最適化。  
    - キャッシュサイズ、データセットプール、スワスサイズ、HTTP/2、多スレッド設定など約 15 項目を `os.environ` と `gdal.SetConfigOption` で一括設定。  
    - A100 / L4 / T4 / RTX4070 クラスとメモリ容量を判定し、32 GB〜4 GB まで段階的にパラメータを調整。 

- **主な外部依存**  
  - `os` — 環境変数設定  
  - `osgeo.gdal` — ランタイムで GDAL オプションを変更  

- **内部依存関係・呼び出し元**  
  - `core.tile_processor` と `io.cog_builder` が処理開始時に呼び出し、VRT/COG 生成の I/O 性能を最大化。  
  - `config.system_config.get_gpu_config()` が生成した GPU 構成を受け取って動作。  

- **ファイルの役割**  
  富士シェーダーGPU の **GDAL I/O を GPU/メモリ性能に最適化**し、巨大ファイル処理をボトルネック無く実行できるようにする環境設定ユーティリティ。


### FujiShaderGPU/__main__.py

- **主要機能 (統一エントリーポイント)**  
  - `main()` が **実行 OS を自動判定**（Linux / Windows / macOS）し、  
    - **Linux** → `cli.linux_cli.LinuxCLI`（Dask-CUDA パイプライン）  
    - **Windows / macOS** → `cli.windows_cli.WindowsCLI`（タイルベース GPU パイプライン）  
  - Linux で依存ライブラリ不足時はインストール手順を提示して終了。  
  - 取得した CLI インスタンスに対して `run()` を呼び出し、KeyboardInterrupt / 例外を捕捉して適切に終了コードを返す。

- **主な外部依存**  
  - 標準: `sys`, `platform`, `warnings`

- **内部依存関係**  
  - `cli.linux_cli.LinuxCLI`, `cli.windows_cli.WindowsCLI` — OS 別の処理フローを委譲。  

- **ファイルの役割**  
  FujiShaderGPU 全体の **単一コマンド起動点**。利用者は `python -m FujiShaderGPU` だけで実行環境に最適な CLI を自動選択できる。


### FujiShaderGPU/utils/nodata_handler.py

- **主要機能 (NoData 高速補間)**  
  - `_handle_nodata_ultra_fast(dem_tile, mask_nodata)`  
    - **NoData 割合 < 10 %** → 単純 0 置換で最速処理。  
    - **10–50 % & SciPy 利用可** → `distance_transform_edt` で最近傍埋め戻し（高速内挿）。  
    - **それ以外 / SciPy 無** → 有効ピクセル平均で埋め戻し。  
    - 全パスでコピーを返し元配列を破壊しない。

- **主な外部依存**  
  - 必須 : **NumPy**（配列操作）  
  - 任意 : **SciPy** (`ndimage.gaussian_filter`, `distance_transform_edt`, `uniform_filter`) — 高速補間ルートで使用。  

- **内部依存関係・呼び出し元**  
  - `core.tile_processor.process_single_tile()` がタイル読込後に呼び出し、GPU 処理前に NoData を除去 / 埋め戻し（= I/O ボトルネック軽減）。  

- **ファイルの役割**  
  DEM タイルごとの **NoData ピクセルをミリ秒単位で処理**する軽量ユーティリティ。  
  NoData 量と SciPy の有無に応じて最適な補間手法を自動選択し、GPU ステージへクリーンな標高データを供給する。


### FujiShaderGPU/io/raster_info.py

- **主要機能 (ピクセルサイズ自動検出)**
  - `detect_pixel_size_from_cog(input_cog_path)`  
    - COG を `rasterio` で開き **Transform と CRS** から X/Y ピクセル幅を取得。  
    - 地理座標系なら緯度中心でメートル換算し、投影系ならそのまま値を使用。  
    - X・Y の平均を **メートル単位のピクセルサイズ** として返す（例外時は 0.5 m）。  
    - コンソールに緯度・換算結果を表示してデバッグを支援。

- **主な外部依存**  
  - **rasterio** — COG メタデータ取得  
  - **math** — 度→ラジアン変換と cos 計算

- **内部依存関係・呼び出し元**  
  - `config.system_config.get_gpu_config()` や `core.tile_processor.process_dem_tiles()` で **自動ピクセルサイズ推定**に利用され、最適タイル寸法やスケール解析の初期値を提供。  

- **ファイルの役割**  
  DEM/DSM の **空間解像度を即時判定**し、富士シェーダーGPU 全体のパラメータ自動化を支える軽量ユーティリティ。


### FujiShaderGPU/core/gpu_memory.py

- **主要機能 (GPUメモリプール管理)**
  - `get_gpu_context()` — スレッドローカルに **CuPy** の既定メモリプール & ピン留めプールを保持し返却。  
    - **RMM** が利用可能なら `rmm_cupy_allocator` を読み込み、CuPy のアロケータを RMM に自動切替。
  - `gpu_memory_pool()` — 上記プールを `with` で使えるコンテキストマネージャとして公開。  
    - ブロック終了時に `cp.cuda.Stream.null.synchronize()` で GPU 処理完了を待機し、`free_all_blocks()` でメモリを完全解放。

- **主な外部依存**
  - **cupy** — GPU 配列 & メモリプール API  
  - **rmm**（任意）— NVIDIA RAPIDS Memory Manager  
  - 標準ライブラリ: `contextlib`, `threading`

- **内部依存関係・呼び出し元**
  - `core.tile_processor.process_single_tile()` などが `with gpu_memory_pool():` で GPU メモリを安全に確保・開放。
  - `core.dask_processor` 側でも RMM / CuPy アロケータ切替を行っており、本モジュールと合わせて GPU メモリ管理を統一。  

- **ファイルの役割**
  FujiShaderGPU 全体で **GPU メモリの確実な割当てとリーク防止** を担う軽量ユーティリティ。長時間のバッチ処理でもフラグメントを最小化し、VRAM を効率利用できる基盤となる。


### FujiShaderGPU/algorithms/__init__.py

- **主要機能 (プラットフォーム別 API リエクスポート)**
  - 実行 OS を `platform.system()` で判定し、  
    - **Linux** → `dask_algorithms` から `ALGORITHMS` 辞書と `DaskAlgorithm` 基底を公開  
    - **Windows/macOS** → `tile_algorithms` から各 `*Algorithm` クラスを個別公開  
  - `__all__` を動的に更新して、利用者が `from algorithms import *` した際に適切なシンボルだけを提供。

- **主な外部依存**
  - `platform` — OS 検出

- **内部依存関係・呼び出し元**
  - `algorithms.dask_algorithms`, `algorithms.tile_algorithms` への条件付きインポート。
  - 他モジュール（`core.*`, `cli.*` など）は **OS 非依存で `algorithms` パッケージを参照**でき、適切な実装が自動的に解決される。

- **ファイルの役割**
  FujiShaderGPU のアルゴリズム層に **単一の統一インターフェース**を提供し、  
  コード側は Platform 分岐を意識せず `algorithms` から必要なクラス／辞書を直接インポート可能にする。


### FujiShaderGPU/core/__init__.py

- **主要機能 (コア API リエクスポート)**
  - 実行 OS を判定し、以下を `__all__` に動的登録して公開する:
    - **共通**: `gpu_memory_pool`
    - **Windows / macOS**: `process_dem_tiles`, `resume_cog_generation`
    - **Linux**: `run_pipeline`, `make_cluster` 
  - 目的は、クライアントコードが `from core import *` で **OS に応じた最適パイプライン**を自動で取得できるようにすること。

- **主な外部依存**
  - `platform` — OS 判定のみ

- **内部依存関係・呼び出し元**
  - `core.gpu_memory.gpu_memory_pool`, `core.tile_processor`, `core.dask_processor` を条件付きインポートし再公開。
  - CLI 層や他モジュールは **OS 非依存で `core` パッケージを参照**でき、適切な実装が解決される。


### FujiShaderGPU/utils/types.py

- **主要機能 (結果データ型定義)**
  - `TileResult(NamedTuple)` ― タイル処理結果を保持する不変データ構造  
    - `tile_y`, `tile_x` ― タイル座標  
    - `success` ― 成否フラグ  
    - `filename` ― 出力ファイルパス (任意)  
    - `error_message`, `skipped_reason` ― 失敗・スキップ理由 (任意)

- **主な外部依存**  
  - `typing.NamedTuple`, `Optional`

- **内部依存関係・呼び出し元**  
  - `core.tile_processor.process_dem_tiles()` などで並列ワーカーが返す結果型として利用され、処理統計やログ出力に活用。

- **ファイルの役割**  
  富士シェーダーGPU の **タイル処理結果を型安全に表現**し、ワークフロー全体で一貫した結果伝達を実現するユーティリティ型。


### FujiShaderGPU/config/__init__.py

- **主要機能**  
  - パッケージ初期化のみ。モジュールレベルの処理やシンボル再公開は行わず、単に `FujiShaderGPU/config` ディレクトリを **Python パッケージとして認識させるためのプレースホルダー** に留まる。

- **主な外部依存**  
  - なし（docstring 以外のコードは存在しない）。

- **内部依存関係・役割**  
  - `config` 配下の各サブモジュール（`gdal_config.py`, `system_config.py` など）をまとめる **名前空間パッケージ**として機能。  
  - アプリ全体で `import FujiShaderGPU.config ...` の形で設定ユーティリティを呼び出す際のエントリポイントとなる。


### FujiShaderGPU/utils/__init__.py

- **主要機能**
  - 何もロジックを含まない **プレースホルダー**。`FujiShaderGPU/utils` を Python パッケージとして認識させるだけの空モジュール。

- **主な外部依存**
  - なし。

- **内部依存関係・役割**
  - `utils` 配下のユーティリティ群（`scale_analysis.py`, `nodata_handler.py`, `types.py` など）をまとめる名前空間として機能。


### FujiShaderGPU/io/__init__.py

- **主要機能**
  - 実装コードはなく、`FujiShaderGPU/io` ディレクトリを **Python パッケージとして認識させるための空モジュール**に留まる。

- **主な外部依存**
  - なし。

- **内部依存関係・役割**
  - `io` 配下の I/O ユーティリティ（`cog_builder.py`, `cog_validator.py`, `raster_info.py` など）をまとめる **名前空間パッケージ**として機能。


### FujiShaderGPU/cli/__init__.py

- **主要機能**
  - 実装ロジックはなく、`FujiShaderGPU/cli` ディレクトリを **Python パッケージとして認識させるためのプレースホルダー**に留まる。 

- **主な外部依存**
  - なし。  

- **内部依存関係・役割**
  - `cli` 配下のエントリースクリプト群（`linux_cli.py`, `windows_cli.py`, `base.py` など）をまとめる **名前空間パッケージ**として機能。


### FujiShaderGPU/__init__.py

- **主要機能**
  - トップレベルのパッケージ初期化のみ。コードは空の docstring に留まり、サブモジュールの再公開や副作用処理は一切行わない。

- **主な外部依存**
  - なし。

- **内部依存関係・役割**
  - FujiShaderGPU 全体を **Python パッケージとして認識させるプレースホルダー**。  
    他モジュールは通常 `import FujiShaderGPU` で名前空間が解決される。
