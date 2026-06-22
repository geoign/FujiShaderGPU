• # FujiShaderGPU コードレビュー・レポート

  対象: C:\Users\fikeg\OneDrive\Dev\Python\FujiShaderGPU
  実施日: 2026-06-22
  レビュー範囲: FujiShaderGPU/ 配下の主要実装、CLI、Dask/Tile バックエンド、アルゴリズム実装、IO/COG 生成、設定・ユーティリティ
  実施した確認: 静的コードレビュー、主要ファイル読解、python -m compileall -q FujiShaderGPU tests
  compileall 結果: 構文エラーなし

  ———

  ## 総評

  FujiShaderGPU は、Dask-CUDA バックエンドと Windows/macOS 向け Tile バックエンドをかなり丁寧に分離しつつ、アルゴリズム実装を共有化している高度なコー
  ドベースです。特に以下は良い設計です。

  - Dask/Tile 両バックエンドでアルゴリズム名を揃えている。
  - NoData を NaN に統一する方針が明確。
  - 大半径処理を overview/coarse field に逃がす設計により、VRAM 爆発を抑えようとしている。
  - algorithms/_norm_stats.py にグローバル正規化統計を集約し、Tile/Dask の出力差を減らそうとしている。
  - COG 出力の staging / direct chunk write / overview 生成にかなり実運用上の配慮がある。

  一方で、破壊的な一時ディレクトリ処理、CLI 引数が Linux/Dask 側で無効になる問題、正規化統計の関数参照ミス、整数 NoData 設計上の値衝突など、実運用で
  大きく影響する不具合がいくつかあります。以下に重要度順にまとめます。

  ———

  ## Critical

  ### 1. --tmp-dir . や任意既存ディレクトリ指定で作業ディレクトリ全削除の危険

  ファイル: FujiShaderGPU/core/tile_processor.py
  該当箇所: 647-684

  def _resolve_writable_tmp_dir(...):
      ...
      if requested.is_absolute():
          candidates.append(requested)
      else:
          ...
          candidates.append(safe_abspath(Path.cwd() / requested))

      ...
      for candidate in candidates:
          try:
              if candidate.exists():
                  if candidate.is_dir():
                      shutil.rmtree(candidate)
                  else:
                      candidate.unlink()
              candidate.mkdir(parents=True, exist_ok=False)

  問題:

  tmp_tile_dir に既存ディレクトリが指定されると、無条件に shutil.rmtree(candidate) されます。特に危険なのは以下です。

  fujishadergpu input.tif output.tif --tmp-dir .
  fujishadergpu input.tif output.tif --tmp-dir ..
  fujishadergpu input.tif output.tif --tmp-dir C:\Users\fikeg\OneDrive\Dev

  --tmp-dir . の場合、safe_abspath(Path.cwd() / requested) がカレントディレクトリを指し、プロジェクト全体を削除し得ます。

  影響:

  - ユーザーのソースコード、データ、Git リポジトリを消す可能性がある。
  - Windows/macOS Tile バックエンドの CLI オプションとして露出しているため、誤操作で発生しやすい。

  修正案:

  - tmp_dir の削除は、明確に FujiShaderGPU が作った専用サブディレクトリだけに限定する。
  - .、..、ルート、ホーム、リポジトリルート、既存の非空ディレクトリは拒否する。
  - 既存ディレクトリを使う場合は、中身を消さず、run ごとの UUID サブディレクトリを作る。

  例:

  def _is_dangerous_tmp_path(path: Path) -> bool:
      resolved = path.absolute()
      dangerous = {
          Path.cwd().absolute(),
          Path.home().absolute(),
          resolved.anchor and Path(resolved.anchor),
      }
      return resolved in dangerous or path.name in {"", ".", ".."}

  if candidate.exists():
      if _is_dangerous_tmp_path(candidate):
          raise ValueError(f"Refusing to use dangerous tmp dir: {candidate}")
      if any(candidate.iterdir()):
          candidate = candidate / f"fujishadergpu_{uuid.uuid4().hex}"
  candidate.mkdir(parents=True, exist_ok=False)

  ———

  ## High

  ### 2. Linux/Dask CLI の --pixel-size が実際には反映されない

  ファイル: FujiShaderGPU/cli/linux_cli.py
  該当箇所: 132-151

  algo_params = build_algo_params(args)
  ...
  run_pipeline(
      src_cog=params["input_path"],
      dst_cog=params["output_path"],
      algorithm=args.algorithm,
      chunk=getattr(args, "chunk", None),
      show_progress=params["show_progress"],
      auto_radii=auto_radii,
      memory_fraction=getattr(args, "memory_fraction", None),
      nodata_override=parse_nodata_override(getattr(args, "nodata", None)),
      output_dtype=getattr(args, "output_dtype", "float32"),
      output_range=parse_output_range(getattr(args, "output_range", None)),
      **algo_params,
  )

  ファイル: FujiShaderGPU/core/dask_processor.py
  該当箇所: 1098-1102

  px_m_x, px_m_y, pixel_size_m, is_geo, lat_center = _detect_metric_scales_from_dataarray(dem)
  params['pixel_size'] = float(pixel_size_m)
  params.setdefault('pixel_scale_x', float(px_m_x))
  params.setdefault('pixel_scale_y', float(px_m_y))
  params.setdefault('is_geographic_dem', bool(is_geo))

  問題:

  Linux CLI 側で _resolve_pixel_size(args, ...) により args.pixel_size を設定していますが、その値を run_pipeline() に渡していません。

  さらに、仮に algo_params["pixel_size"] として渡したとしても、run_pipeline() 側で params['pixel_size'] = float(pixel_size_m) により自動検出値で上書
  きされます。

  影響:

  - ユーザーが --pixel-size を指定しても Linux/Dask バックエンドでは無効。
  - 勾配・傾斜・陰影・半径換算など、物理スケール依存の結果が想定と異なる。
  - Windows/Tile 側では pixel_size が process_dem_tiles() に渡されるため、バックエンド間で挙動がズレる。

  修正案:

  run_pipeline() に明示的な pixel_size: Optional[float] = None を追加し、CLI から渡す。

  def run_pipeline(..., pixel_size: Optional[float] = None, **algo_params):
      ...
      px_m_x, px_m_y, detected_pixel_size_m, is_geo, lat_center = ...
      if pixel_size is not None:
          params["pixel_size"] = float(pixel_size)
      else:
          params["pixel_size"] = float(detected_pixel_size_m)

  Linux CLI 側:

  run_pipeline(
      ...,
      pixel_size=args.pixel_size,
      **algo_params,
  )

  ———

  ### 3. ambient_occlusion / openness のグローバル正規化統計が参照ミスで計算されない

  ファイル: FujiShaderGPU/algorithms/_norm_stats.py
  該当箇所: 29-46

  _NORM_STAT_SPECS = {
      ...
      "ambient_occlusion": ("_impl_ambient_occlusion",
                           "compute_ambient_occlusion_block",
                           "robust_unsigned_stretch_stat_func"),
      "openness": ("_impl_openness", "compute_openness_vectorized",
                  "robust_unsigned_stretch_stat_func"),
  }

  問題:

  robust_unsigned_stretch_stat_func は以下に定義されています。

  - FujiShaderGPU/algorithms/_global_stats.py
  - FujiShaderGPU/algorithms/_normalization.py

  しかし、_impl_ambient_occlusion.py と _impl_openness.py には定義・import されていません。

  そのため _compute_norm_stats_tiled() で以下が失敗します。

  mod = __import__(f"FujiShaderGPU.algorithms.{spec[0]}", fromlist=[spec[1]])
  stat_func = getattr(mod, spec[2])

  例:

  getattr(_impl_ambient_occlusion, "robust_unsigned_stretch_stat_func")

  は AttributeError になります。

  影響:

  - ambient_occlusion / openness の global_stats が注入されない。
  - apply_display_stretch_dask(result, params.get("global_stats")) が no-op になる。
  - コメント・設計上は [p1, p99] -> [0,1] の contrast stretch を期待しているが、実際には適用されず、白飛び・コントラスト不足が起きる。
  - Tile 側の direct path も params.get("global_stats") に依存しているため同様に効かない。

  修正案:

  どちらかに統一してください。

  案 A: _norm_stats.py で stat 関数の module も持つようにする。

  _NORM_STAT_SPECS = {
      "ambient_occlusion": (
          "_impl_ambient_occlusion",
          "compute_ambient_occlusion_block",
          "_global_stats",
          "robust_unsigned_stretch_stat_func",
      ),
  }

  案 B: _impl_ambient_occlusion.py / _impl_openness.py に re-export する。

  from ._global_stats import robust_unsigned_stretch_stat_func

  ただし、責務としては _global_stats.py を直接参照する案 A の方が自然です。

  ———

  ### 4. signed int16 出力で DN=0 が実データ値 0 と NoData を兼ねてしまう

  ファイル: FujiShaderGPU/io/output_encoding.py
  該当箇所: 8-15, 105-128

  # Integer encodings reserve **0 for NoData**
  ...
  # signed int16
  a_coef = maxpos / a            # value 0 -> DN 0 (== NoData)
  b_coef = 0.0
  dn_min, dn_max = -maxpos, maxpos

  ファイル: FujiShaderGPU/core/dask_processor.py
  該当箇所: 489-490

  out_band = out_ds.GetRasterBand(1)
  out_band.SetNoDataValue(nodata_val)

  ファイル: FujiShaderGPU/core/tile_processor.py
  該当箇所: 869-872

  if _quantize_qp is not None and _quantize_dtype is not None:
      result_core = quantize_array(result_core, _quantize_qp, _quantize_dtype)
      output_nodata = 0.0

  問題:

  signed な正規化アルゴリズム、特に以下では値 0 が普通に大量発生します。

  - topousm_fast
  - fractal_anomaly

  しかし signed int16 の設計では、

  - 実データ値 0
  - NoData

  の両方が DN=0 になります。

  GeoTIFF の NoData tag に 0 を設定すると、多くの GIS / QGIS / rasterio の masked read では 平坦地・ゼロ異常値が NoData としてマスクされます。

  コメントでは「visually negligible」とありますが、実際には解析値として 0 は意味のある値であり、表示上も透明穴や欠損として扱われる可能性があります。

  影響:

  - signed int16 出力で有効ピクセルが欠損扱いされる。
  - QGIS 等で透明化・穴あき表示になり得る。
  - downstream 処理で統計や histogram が壊れる。
  - Dask/Tile 両バックエンド共通。

  修正案:

  選択肢:

  1. signed int16 では NoData を -32768 に固定し、データ範囲を [-32767, 32767] にする。
  2. signed int16 では NoData tag を設定しない代わりに mask band を使う。
  3. signed int16 への量子化を禁止または warning ではなく error にする。
  4. uint8 と同じくゼロ値を offset して NoData=0 と衝突しない設計にする。

  最も単純なのは 1 です。

  if signed and dt == "int16":
      nodata = -32768
      dn_min, dn_max = -32767, 32767

  ———

  ### 5. cog_builder.py の .replace(".vrt", "_files.txt") がパス全体を破壊する

  ファイル: FujiShaderGPU/io/cog_builder.py
  該当箇所: 67, 504

  file_list_path = vrt_path.replace(".vrt", "_files.txt")

  問題:

  str.replace() はパス文字列中のすべての .vrt を置換します。例えば:

  C:\data\.vrt_cache\tiles.vrt

  は以下になります。

  C:\data\_files.txt_cache\tiles_files.txt

  ディレクトリ名まで破壊されます。

  影響:

  - VRT 作成用 file list が意図しない場所に作成される。
  - そのディレクトリが存在しなければ失敗。
  - 存在してしまうと別の場所に一時ファイルを作り、cleanup も不安定。

  修正案:

  Path を使ってファイル名だけを変える。

  _vrt = Path(vrt_path)
  file_list_path = str(_vrt.with_name(f"{_vrt.stem}_files.txt"))

  ———

  ## Medium

  ### 6. Tile バックエンドの「小さい DEM では spatial から local に fallback」が実質発火しない

  ファイル: FujiShaderGPU/core/tile_processor.py
  該当箇所: 1010-1024, 1299-1314

  先に auto radii が注入されます。

  elif (
      algorithm in AUTO_SPATIAL_RADII_ALGOS
      and not algo_params.get("radii")
      and _mode_now == "spatial"
  ):
      ...
      algo_params["radii"] = _auto_r
      ...
      algo_params["weights"] = _auto_w

  その後、小さい DEM fallback 判定をしています。

  user_radii_specified = ("radii" in algo_params) and (algo_params.get("radii") is not None)
  user_weights_specified = ("weights" in algo_params) and (algo_params.get("weights") is not None)

  if (
      requested_mode == "spatial"
      and not user_radii_specified
      and not user_weights_specified
      and min(int(width), int(height)) <= 1024
  ):
      algo_params["mode"] = "local"

  問題:

  user_radii_specified / user_weights_specified は「ユーザーが指定したか」ではなく「現在 params に存在するか」を見ています。auto radii 注入後なので常
  に true になり、fallback が発火しません。

  影響:

  - ARCHITECTURE.md の設計と実装がズレる。
  - 小さい DEM で意図せず spatial mode のまま処理される。
  - パフォーマンス・出力の期待値に影響。

  修正案:

  auto 注入前に user 指定フラグを保存する。

  _user_radii_specified = algo_params.get("radii") is not None
  _user_weights_specified = algo_params.get("weights") is not None

  # auto radii injection ...

  if requested_mode == "spatial" and not _user_radii_specified and not _user_weights_specified:
      ...

  ———

  ### 7. Linux CLI の Zarr 入力サポートが _resolve_pixel_size() で壊れる可能性

  ファイル: FujiShaderGPU/cli/linux_cli.py
  該当箇所: 71-103

  def _resolve_pixel_size(self, args: argparse.Namespace, input_path: str):
      ...
      with rasterio.open(input_path) as src:
          ...

  問題:

  Dask パイプライン本体は .zarr 入力をサポートしています。

  - core/dask_io.py::is_zarr_path
  - load_input_dataarray()
  - run_pipeline() 側の Zarr 分岐

  しかし Linux CLI は実行前に無条件で rasterio.open(input_path) します。.zarr 入力の場合、ここで失敗します。

  影響:

  - 実装上は Zarr をサポートしているのに CLI から使えない。
  - README/ARCHITECTURE の Zarr サポート説明とズレる。

  修正案:

  from ..core.dask_io import is_zarr_path

  if is_zarr_path(input_path):
      self.logger.info("Zarr input: pixel size will be detected from xarray/rioxarray metadata in run_pipeline")
      return

  また、前述の --pixel-size 修正と合わせて、CLI 側の pixel-size 解決を run_pipeline に統合するとよいです。

  ———

  ### 8. gdal.DontUseExceptions() が import 時にグローバル状態を変更している

  ファイル:

  - FujiShaderGPU/io/cog_builder.py:24
  - FujiShaderGPU/config/system_config.py:18

  gdal.DontUseExceptions()

  問題:

  GDAL の例外設定はプロセスグローバルです。モジュール import 時に DontUseExceptions() を呼ぶと、他のモジュール・利用者コードが gdal.UseExceptions()
  を期待していても無効化されます。

  影響:

  - GDAL エラーが Python 例外にならず、None 返り値や silent failure になりやすい。
  - ライブラリとして FujiShaderGPU を import しただけで、呼び出し元アプリの GDAL 挙動を変える。
  - デバッグ難度が上がる。

  修正案:

  - import 時のグローバル変更を避ける。
  - GDAL 操作の直前に局所的に設定し、finally で戻す。
  - 可能であれば UseExceptions() に寄せ、戻り値 None のチェックを徹底する。

  ———

  ### 9. gdal.BuildVRT() の戻り値をチェックせず、dataset も明示 close していない

  ファイル: FujiShaderGPU/io/cog_builder.py
  該当箇所: 217-227

  vrt_options = gdal.BuildVRTOptions(...)
  gdal.BuildVRT(vrt_path, tile_files, options=vrt_options)
  logger.info("Python VRT: %.1fs", time.time() - start)

  問題:

  gdal.BuildVRT() は失敗時に None を返し得ます。現在は結果を確認していません。また成功時も dataset handle を変数に受けず close/flush していません。

  影響:

  - VRT 作成失敗を検知できず、その後の COG 生成で分かりにくいエラーになる。
  - Windows では handle が残るとファイル削除・上書きが失敗しやすい。
  - VRT が完全に flush されないリスク。

  修正案:

  vrt_ds = gdal.BuildVRT(vrt_path, tile_files, options=vrt_options)
  if vrt_ds is None:
      raise RuntimeError(f"BuildVRT failed: {vrt_path}")
  vrt_ds = None

  ———

  ### 10. mutable default argument が残っている

  該当箇所:

  - FujiShaderGPU/algorithms/_impl_topousm_fast.py:50

  def compute_topousm_fast_efficient_block(block: cp.ndarray, *,
                                 radii: List[int] = [4, 16, 64],

  - FujiShaderGPU/algorithms/_impl_fractal_anomaly.py:81

  def compute_fractal_dimension_block(block, *, radii=[4, 8, 16, 32, 64],

  - FujiShaderGPU/algorithms/_impl_visual_saliency.py:80

  def compute_visual_saliency_block(block, *, scales=[2, 4, 8, 16], radii=None,

  問題:

  現状、関数内で radii / scales を破壊的に変更していないため即座に壊れる可能性は高くありません。しかし Python の mutable default は将来の変更でバグに
  なりやすく、レビュー・lint でも指摘される典型です。

  修正案:

  def compute_topousm_fast_efficient_block(..., radii: Optional[List[int]] = None, ...):
      if radii is None:
          radii = [4, 16, 64]

  ———

  ### 11. openness の inner loop で GPU scalar を float() 化しており同期コストが出る

  ファイル: FujiShaderGPU/algorithms/_impl_openness.py
  該当箇所: 103

  phys_dist = max(float(cp.sqrt(phys_dx ** 2 + phys_dy ** 2)), 1e-9)

  問題:

  phys_dx / phys_dy は Python float です。ここで cp.sqrt() を使うと CuPy scalar が作られ、float(...) で GPU → CPU 同期が発生します。これは
  num_directions * len(distances) の inner loop 内なので、小さい同期でも積み重なります。

  ambient_occlusion 側は同様の箇所で np.hypot() を使っており、その方が適切です。

  修正案:

  phys_dist = max(float(np.hypot(phys_dx, phys_dy)), 1e-9)

  ———

  ### 12. package __init__.py が import 失敗を握りつぶしている

  ファイル: FujiShaderGPU/algorithms/__init__.py
  該当箇所: 5-10, 13-49

  try:
      from .dask_registry import ALGORITHMS, DaskAlgorithm
      __all__.extend(["ALGORITHMS", "DaskAlgorithm"])
  except Exception:
      pass
  ...
  except Exception:
      pass

  ファイル: FujiShaderGPU/core/__init__.py
  該当箇所: 10-31

  try:
      from .gpu_memory import gpu_memory_pool
      __all__.append("gpu_memory_pool")
  except ImportError:
      pass

  問題:

  except Exception: pass は実装上のバグ、依存関係の不整合、循環 import を隠します。

  影響:

  - from FujiShaderGPU.algorithms import ALGORITHMS が失敗しても原因が見えない。
  - 利用者からは「名前が存在しない」だけに見える。
  - CI や packaging の問題を早期検出しにくい。

  修正案:

  - 最低限 warning log を出す。
  - ImportError だけ捕捉し、その他の例外は再送出する。

  except ImportError as exc:
      logger.warning("Optional import failed: %s", exc)

  ———

  ## Low / Design Concerns

  ### 13. compute_fractal_anomaly._determine_optimal_radii() が小さな処理に CuPy を使う

  ファイル: FujiShaderGPU/algorithms/_impl_fractal_anomaly.py
  該当箇所: 384-386

  indices = cp.linspace(0, len(base)-1, 6).astype(int).get()
  base = [base[int(i)] for i in indices]

  問題:

  この処理は数個の Python list index を作るだけです。CuPy を使うと CUDA context 初期化や .get() 転送が発生します。

  現状の実害:

  base は各分岐で 6 個なので if len(base) > 6: は現在ほぼ発火しません。ただし将来 base が増えた場合に無駄な GPU 使用になります。

  修正案:

  indices = np.linspace(0, len(base) - 1, 6).astype(int)
  base = [base[int(i)] for i in indices]

  ———

  ### 14. resume_cog_generation() の削除コマンド表示が Windows 向けでない

  ファイル: FujiShaderGPU/core/tile_processor.py
  該当箇所: 末尾付近

  logger.info(f"Delete command: rm -rf {tmp_tile_dir}")

  問題:

  Windows/macOS fallback として使われる Tile backend で、Windows ユーザーに rm -rf を表示します。

  修正案:

  platform に応じて表示を変えるか、「手動で削除してください」に留める。

  PowerShell なら:

  Remove-Item -Recurse -Force -LiteralPath "..."

  ただし、削除コマンドを出す場合は誤削除を避けるためパス検証が必要です。

  ———

  ### 15. core/tile_io.py の thread-local rasterio reader が明示 close されない

  ファイル: FujiShaderGPU/core/tile_io.py
  該当箇所: 9-24

  _thread_local = threading.local()

  def _get_thread_reader(input_cog_path: str):
      reader = getattr(_thread_local, "reader", None)
      ...
      reader = rasterio.open(input_cog_path, "r")
      _thread_local.reader = reader

  問題:

  ThreadPoolExecutor の worker thread が終了すれば最終的に GC されますが、明示 close ではありません。

  影響:

  - Windows で入力ファイルの上書き・削除が遅れる可能性。
  - 長時間プロセスで複数 input を処理すると reader が残りやすい。

  修正案:

  - Tile 処理終了時に worker thread 内 reader を close する仕組みを入れる。
  - または per-tile open/close に戻す代わりに、worker initializer/finalizer 相当の管理を明示する。

  ———

  ## 推奨修正優先順位

  ### 最優先

  1. _resolve_writable_tmp_dir() の無条件 rmtree を廃止
  2. Linux/Dask CLI の --pixel-size を正しく反映
  3. _NORM_STAT_SPECS の ambient_occlusion / openness stat 関数参照を修正
  4. signed int16 の NoData=0 衝突設計を見直す

  ### 次点

  5. .replace(".vrt", "_files.txt") を Path.with_name() に修正
  6. Tile 小 DEM fallback 判定を「ユーザー指定フラグ」で行う
  7. Zarr 入力時に Linux CLI が rasterio.open() しないようにする
  8. GDAL import 時の global DontUseExceptions() をやめる

  ### 品質改善

  9. mutable default argument を除去
  10. gdal.BuildVRT() の戻り値チェック・close
  11. openness inner loop の cp.sqrt → np.hypot
  12. except Exception: pass の削減

  ———

  ## 簡易パッチ例

  ### _NORM_STAT_SPECS 修正例

  # algorithms/_norm_stats.py
  _NORM_STAT_SPECS = {
      ...
      "ambient_occlusion": (
          "_impl_ambient_occlusion",
          "compute_ambient_occlusion_block",
          "_global_stats",
          "robust_unsigned_stretch_stat_func",
      ),
      "openness": (
          "_impl_openness",
          "compute_openness_vectorized",
          "_global_stats",
          "robust_unsigned_stretch_stat_func",
      ),
  }

  その場合、読み出し側も変更:

  if len(spec) == 3:
      impl_mod_name, block_name, stat_name = spec
      stat_mod_name = impl_mod_name
  else:
      impl_mod_name, block_name, stat_mod_name, stat_name = spec

  impl_mod = __import__(f"FujiShaderGPU.algorithms.{impl_mod_name}", fromlist=[block_name])
  stat_mod = __import__(f"FujiShaderGPU.algorithms.{stat_mod_name}", fromlist=[stat_name])
  block_func = getattr(impl_mod, block_name)
  stat_func = getattr(stat_mod, stat_name)

  ———

  ### Linux --pixel-size 修正例

  # cli/linux_cli.py
  run_pipeline(
      ...,
      pixel_size=args.pixel_size,
      **algo_params,
  )

  # core/dask_processor.py
  def run_pipeline(..., pixel_size: Optional[float] = None, **algo_params):
      ...
      px_m_x, px_m_y, detected_pixel_size_m, is_geo, lat_center = _detect_metric_scales_from_dataarray(dem)

      if pixel_size is not None:
          params["pixel_size"] = float(pixel_size)
      else:
          params["pixel_size"] = float(detected_pixel_size_m)

  ———

  ### .vrt file list path 修正例

  from pathlib import Path

  _vrt_path = Path(vrt_path)
  file_list_path = str(_vrt_path.with_name(f"{_vrt_path.stem}_files.txt"))

  ———

  ## 追加で入れるべきテスト

  ### 1. 危険 tmp-dir 拒否テスト

  def test_tmp_dir_refuses_cwd(tmp_path, monkeypatch):
      monkeypatch.chdir(tmp_path)
      with pytest.raises(ValueError):
          _resolve_writable_tmp_dir(".", "out.tif", "in.tif")

  ### 2. --pixel-size が Linux run_pipeline に渡るテスト

  LinuxCLI.execute() を monkeypatch して run_pipeline 呼び出し引数に pixel_size があることを確認。

  ### 3. AO/Openess の global_stats 注入テスト

  def test_ao_norm_spec_stat_func_resolves():
      from FujiShaderGPU.algorithms._norm_stats import _NORM_STAT_SPECS
      # spec の stat func が import/getattr できること

  ### 4. signed int16 NoData 衝突テスト

  def test_signed_int16_zero_not_used_as_nodata_for_valid_zero():
      qp = quantize_params(-1.176, 1.176, "int16")
      arr = np.array([np.nan, 0.0], dtype=np.float32)
      out = quantize_array(arr, qp, "int16")
      assert out[0] != out[1]

  現在の実装ではこのテストは失敗するはずです。

  ———

  ## 結論

  現状の FujiShaderGPU は、アーキテクチャ自体はかなり成熟しています。ただし、実運用で致命的になり得る箇所が数点あります。

  特に --tmp-dir の無条件削除 は、ユーザーデータを消す危険があるため最優先で修正すべきです。次に、Linux/Dask の --pixel-size 無効化、AO/Openess の
  global stats 参照ミス、signed int16 NoData 衝突が出力品質・利用者体験に大きく関わります。

  これらを直せば、Dask/Tile の出力整合性と安全性はかなり改善するはずです。