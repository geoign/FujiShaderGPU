# FujiShaderGPU コードレビューレポート

**対象プロジェクト:** `C:\\Users\\fikeg\\OneDrive\\Dev\\Python\\FujiShaderGPU`  
**レビュー日:** 2026-06-15  
**レビュー範囲:** コアパイプライン（Dask / tile 両バックエンド）、アルゴリズム実装、I/O、CLI、設定、前処理、テスト  
**レビュー手法:** 静的コード精読、テスト結果確認（`pytest`、`ruff`）、アーキテクチャ把握  

> 本レポートは、実行時に実際に発生するすべての挙動を GPU 実機で検証したものではありません。特に CUDA 関連の問題については、コード上のリスクとして挙げています。実際の修正前・修正後は必ずテスト環境（できれば対象 GPU）で動作確認してください。

---

## 1. 総合評価

FujiShaderGPU は、大規模 DEM の GPU 可視化という困難な領域に対して、**Dask-CUDA バックエンド（Linux）とタイルバックエンド（Windows/macOS）を共通化した設計**を持つ、成熟したパイプラインです。特に以下の点は高く評価できます。

- アルゴリズム実装を `algorithms/_impl_*.py` に集約し、Dask / tile 両方から呼び出す構成
- 大半径演算を COG オーバービュー 1 回読み出しで済ませる **Unified Overview Coarse Source** 設計
- CLI 引数・パラメータ変換を `cli/args.py` に集約した **single source of truth**
- NoData を NaN に統一し、整数出力時に 0 を NoData とする一貫したポリシー
- コンテナ環境（cgroup）を考慮した CPU / メモリ検出

一方、以下のような課題が散見されました。

- 例外処理が全体的に**包括的すぎてデバッグが困難**
- GDAL の例外モデルと rasterio / 将来の GDAL 4 が**競合する可能性**
- `scales` を使うアルゴリズムと `radii` を使うアルゴリズムの**パディング計算が混在・不整合**
- タイルバックエンドの「spatial」モードが**実質 Dask にフォールバック**している点の非明示性
- ビルドアーティファクト・カバレッジレポートなどの**リポジトリ汚染**

重大なクラッシュバグは見当たりませんが、上記の潜在的な不具合・非効率を放置すると、大規模ラスター処理時に予期せぬ結果や OOM、tile 境界のシームが発生するリスクがあります。

---

## 2. 重大度別の問題点

### 2.1 高重大度（信頼性・正確性に関わる）

#### 2.1.1 GDAL 例外モデルの混在と将来の互換性

`FujiShaderGPU/io/cog_builder.py`、`FujiShaderGPU/io/cog_validator.py`、`FujiShaderGPU/config/system_config.py` で以下が呼ばれています。

```python
gdal.DontUseExceptions()
```

これは**プロセス全体に影響するグローバル設定**です。

- rasterio は多くの箇所で GDAL エラーを Python 例外として扱うことを前提に設計されており、混在により「GDAL は失敗したが rasterio は気づいていない」状態が生じ得ます。
- GDAL 4 では `gdal.UseExceptions()` がデフォルトになる方向です。`DontUseExceptions()` は将来の互換性リスクです。
- `gdal.Open(...)` が `None` を返しても例外が飛ばないため、後続の `ds.GetRasterBand(1)` などで `AttributeError` が出るか、あるいは気づかずに不正な結果になる可能性があります。

**修正方針:**

- 全体を `gdal.UseExceptions()` に移行し、GDAL 呼び出しは明示的な `try/except` で囲む。
- 移行が困難な場合は、最低限 `gdal.DontUseExceptions()` を呼ぶ箇所を 1 箇所に集約し、全 `gdal.Open` 結果が `None` でないことを確認するラッパーを導入する。

#### 2.1.2 包括的すぎる `except Exception`

全体を通して `except Exception:` が数十箇所あり、特に以下が問題です。

| ファイル | 例 | リスク |
|---------|-----|--------|
| `FujiShaderGPU/algorithms/__init__.py` | Dask / tile アルゴリズムの import 全体を `except Exception` で囲む | CuPy/Dask の破損インストール時に「アルゴリズムが見つかりません」という不明瞭なエラーになる |
| `FujiShaderGPU/core/dask_processor.py` | パイプライン各所 | CUDA OOM、GDAL エラー、Dask 通信エラーなどの本質的な原因が隠蔽される |
| `FujiShaderGPU/core/tile_processor.py` | パディング計算・オーバービュー読み出し・統計注入 | 失敗しても警告ログのみで処理が続行し、結果にシームや誤った正規化が入る |
| `FujiShaderGPU/__main__.py` | 最上位で `except Exception` | スタックトレースが表示されず、サポート対応・デバッグが困難 |

**修正方針:**

- キャッチする例外を特定の型に絞る：`ImportError`、`ModuleNotFoundError`、`RuntimeError`、`cp.cuda.memory.OutOfMemoryError`、`subprocess.CalledProcessError`、`rasterio.errors.RasterioIOError` など。
- 予期しない例外は `logging.exception(...)` でトレースバックを残し、再 raise するか、`--verbose` オプションで表示する。

```python
# 推奨例
try:
    from ..algorithms.dask_registry import ALGORITHMS
except ImportError as exc:
    logger.warning("Dask registry not available: %s", exc)
    raise  # または明示的に無効化する理由を記録
```

#### 2.1.3 `scales` 系アルゴリズムのパディング計算が `radii` を参照してしまう

`FujiShaderGPU/core/tile_processor.py` の `_required_padding_for_algorithm` で、`multiscale_terrain` / `visual_saliency` / `scale_space_surprise` のパディング計算に `_unified_radii(default)` が使われています。

```python
def _unified_radii(default):
    rs = algo_params.get("radii")   # ← scales 系アルゴリズムでは空!
    ...
```

これらのアルゴリズムはユーザー入力を `algo_params["scales"]` に保持するため、**ユーザーが `--scales` / `--vs-scales` / `--surprise-scales` を指定してもパディング計算に無視されます**。結果として：

- 大きなスケールを指定したのにパディングが不足 → **タイル境界のシーム**
- 小さなスケールを指定したのに不要に大きなパディング → **VRAM 消費・処理遅延**

同様に、`_norm_stats.py` の `_norm_stat_max_scale` も `radii` / `kernel_size` しか見ていないため、`scales` 系アルゴリズムのグローバル統計のマージンが不足する可能性があります。

**修正方針:**

- アルゴリズム名に応じて `algo_params.get("scales")` または `algo_params.get("radii")` を使うように統一する。
- `_norm_stat_max_scale` も `scales` を考慮する。

```python
def _norm_stat_max_scale(merged: dict) -> float:
    vals = []
    for key in ("radii", "scales", "fractal_radii", "vs_scales", "surprise_scales"):
        v = merged.get(key)
        if isinstance(v, (list, tuple)) and v:
            vals.append(max(float(x) for x in v))
    ...
```

#### 2.1.4 Dask パイプラインでも `scales` / `radii` が混在

`FujiShaderGPU/core/dask_processor.py` の hybrid overview 生成（visual_saliency / scale_space_surprise / fractal_anomaly）は、`_radii = params.get("radii")` を参照しています。

```python
if algorithm in _HYBRID_PFX and params.get("radii") and ...:
    _radii = [float(r) for r in (params.get("radii") or [])]
```

`visual_saliency` / `scale_space_surprise` は `params["scales"]` を使うため、ユーザーが `--vs-scales 2,4,8,32` を指定しても hybrid overview はデフォルト `[2,4,8,16]` に対して生成されます。結果：

- ユーザーの `32` に対する coarse field が生成されない
- 不要な `16` の coarse field が生成される
- 大きなスケールが full-res の single tile Dask 配列で処理され、VRAM 圧迫

**修正方針:**

- hybrid 生成時にアルゴリズムに応じて `params.get("scales")` を優先する。

```python
if algorithm in ("visual_saliency", "scale_space_surprise"):
    user_scales = params.get("scales") or params.get("radii")
else:
    user_scales = params.get("radii")
```

#### 2.1.5 地理座標系のメートル換算が Linux CLI と `raster_info.py` で不一致

`FujiShaderGPU/cli/linux_cli.py` の `_resolve_pixel_size` は緯度に応じた詳細なメートル換算式を使っています。

```python
meters_per_degree_lat = 111132.92 - 559.82 * np.cos(2 * lat_rad) + ...
meters_per_degree_lon = 111412.84 * np.cos(lat_rad) - 93.5 * np.cos(3 * lat_rad) + ...
```

一方、`FujiShaderGPU/io/raster_info.py` の `metric_pixel_scales_from_metadata` は単純な定数を使っています。

```python
meters_per_degree_lat = 111_320.0
meters_per_degree_lon = meters_per_degree_lat * max(1e-6, abs(math.cos(math.radians(lat_center))))
```

これにより、**同じ入力 DEM に対して Linux バックエンドと Windows タイルバックエンドで検出される `pixel_size` が最大 0.2〜0.5% ずれます**。可視化用途では許容範囲かもしれませんが、数値的に一致させるべきです。

**修正方針:**

- 地理座標系のメートル換算を `FujiShaderGPU/algorithms/common/geo_scales.py` など 1 箇所に集約し、CLI と `raster_info.py` 両方から呼び出す。

#### 2.1.6 タイルバックエンドの「spatial」モードが実質 Dask 依存

`FujiShaderGPU/algorithms/tile/dask_bridge.py` の `_process_direct` は、spatial モードで多くのアルゴリズムに対して `_FallbackToDask()` を投げ、シングルタイルの Dask 配列として処理します。

```python
if str(p.get("mode", "local")).lower() == "spatial" and class_name in {
    "HillshadeAlgorithm", "SlopeAlgorithm", "SpecularAlgorithm",
    "AtmosphericScatteringAlgorithm", "CurvatureAlgorithm",
    "AmbientOcclusionAlgorithm", "OpennessAlgorithm",
    "FractalAnomalyAlgorithm",
}:
    raise _FallbackToDask()
```

`MultiscaleDaskAlgorithm`、`VisualSaliencyAlgorithm`、`NPREdgesAlgorithm`（spatial 時）、`ScaleSpaceSurpriseAlgorithm` も Dask フォールバックです。

`pyproject.toml` の `windows` extra には `dask[array]` が含まれているため動作はしますが、**Windows/macOS ユーザーにとって「tile backend = Dask-free」という印象を与えてしまう**のは誤解を招きます。Dask が未インストールの環境ではこれらのアルゴリズムが使えません。

**修正方針:**

- README / ドキュメントに「Windows/macOS でも spatial モードの一部アルゴリズムには dask[array] が必要」と明記する。
- 長期的には `_direct_*` 実装を拡充し、Dask フォールバックを減らす。

#### 2.1.7 一時ディレクトリが黙って削除される

`FujiShaderGPU/core/tile_processor.py` の `_resolve_writable_tmp_dir` は、既存の `tiles_tmp` を無条件で `shutil.rmtree` します。

```python
if candidate.exists():
    if candidate.is_dir():
        shutil.rmtree(candidate)
    else:
        candidate.unlink()
```

ユーザーが誤って重要なディレクトリ名を指定した場合、**データ消失のリスク**があります。また、デフォルト名 `tiles_tmp` でも実行のたびに消えるため、中断後の再開（resume）ができません（`--cog-only` は別途存在しますが、デフォルト動作で消えてしまいます）。

**修正方針:**

- 削除前にログ警告を出し、内容が期待するタイルファイルのみであることを確認する。
- デフォルトでは `tiles_tmp_<pid>_<timestamp>` などユニーク名を使い、衝突を避ける。
- `--force-tmp-cleanup` のような明示的オプションを設け、通常は上書き確認またはエラーにする。
EOF

### 2.2 中重大度（保守性・性能・信頼性）

#### 2.2.1 巨大モジュールの責務が混在

| モジュール | 行数 | 責務 | 推奨分割 |
|-----------|------|------|----------|
| `core/tile_processor.py` | ~1,644 | タイル分割・パディング・統計注入・並列実行・VRAM 抑制・COG 統合・検証 | `tile_grid.py`, `tile_scheduler.py`, `tile_stats.py`, `cog_consolidator.py` |
| `core/dask_processor.py` | ~1,585 | GDAL 設定・クラスタ起動・入力読み込み・半径解決・統計注入・量子化・COG/Zarr 書き込み・クリーンアップ | `dask_io.py`, `dask_write.py`, `dask_quantization.py`, `dask_cleanup.py` |
| `io/dem_preprocess.py` | ~972 | NoData 検出・push-pull fill・ストリーミング・overview 書き込み | `preprocess_detect.py`, `preprocess_fill.py`, `preprocess_stream.py` |
| `algorithms/_nan_utils.py` | ~878 | 平滑化・勾配・リサンプリング・overview サンプリング | `filters.py`, `overview_sampler.py`, `multiscale.py` |

**修正方針:**

- まず `core/tile_processor.py` と `core/dask_processor.py` から、I/O スケジューリング部分と COG 統合部分を切り出す。
- 分割時には、現行の API 維持のため `process_dem_tiles` / `run_pipeline` は thin wrapper に留める。

#### 2.2.2 `coarse_large_radius_response` 内の `.compute()`

`FujiShaderGPU/algorithms/_nan_utils.py` の `coarse_large_radius_response` は、unified overview 未使用時に `coarse.map_overlap(...).compute()` を呼びます。

```python
coarse_resp = coarse.map_overlap(
    block_fn, ...
).compute()
```

これは Dask グラフ構築中に同期的な計算を発生させるアンチパターンです。小さい coarse 配列なので現状は動作しますが、分散クライアント使用時に予期せぬブロッキングや二重計算を引き起こす可能性があります。

**修正方針:**

- `coarse` を `persist()` し、最終的な結果を `map_blocks` に渡す形に変更し、`.compute()` を削除する。
- または、unified overview path（`read_overview_coarse_dem`）を常に使うように統一し、この分岐を減らす。

#### 2.2.3 スレッド単位 rasterio リーダーのリーク

`FujiShaderGPU/core/tile_io.py` はスレッドローカルに `rasterio.DatasetReader` をキャッシュします。

```python
def _get_thread_reader(input_cog_path: str):
    reader = getattr(_thread_local, "reader", None)
    ...
    reader = rasterio.open(input_cog_path, "r")
    _thread_local.reader = reader
```

`ThreadPoolExecutor` 終了時に明示的な `.close()` 呼び出しがないため、GDAL ファイルハンドルが残る可能性があります。長時間連続実行や大量タイル処理時に `Too many open files` や Windows 上のファイルロック競合を引き起こす可能性があります。

**修正方針:**

- `ThreadPoolExecutor` シャットダウン時に、保存されているすべての reader を close するクリーンアップフックを追加する。
- または、`_get_thread_reader` を weakref ベースのキャッシュにし、明示的な close を義務付ける。

```python
import weakref
_readers: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

def close_all_thread_readers():
    for reader in list(_readers.values()):
        try:
            reader.close()
        except Exception:
            pass
```

#### 2.2.4 Dask 環境変数の設定タイミング

`FujiShaderGPU/cli/linux_cli.py` の `execute()` 内で Dask 用環境変数を設定しています。

```python
def execute(self, args):
    os.environ["DASK_DISTRIBUTED__WORKER__MEMORY__TARGET"] = "0.70"
    ...
    from ..core.dask_processor import run_pipeline
```

Dask がすでにどこかで import されていると、これらの環境変数は無視されます。

**修正方針:**

- `dask.config.set(...)` を `run_pipeline` 内または `make_cluster` 内で明示的に行う。
- または、`execute()` の先頭（他の Dask import より前）に設定を移動する。

#### 2.2.5 `GPUtil` の信頼性

`FujiShaderGPU/core/dask_cluster.py` と `FujiShaderGPU/cli/linux_cli.py` で `GPUtil.getGPUs()` を使っています。

- `GPUtil` はメンテナンスが止まっており、WSL / ヘッドレスコンテナ / 一部の NVIDIA ドライバー環境で誤検出・例外を起こすことが知られています。
- `gpus[0]` と CuPy device 0 が必ずしも同じ GPU ではありません。

**修正方針:**

- `cp.cuda.runtime.getDeviceProperties(0)` と `memGetInfo()` の情報を優先し、`GPUtil` はフォールバックのみにする。
- `cp.cuda.Device(0).use()` などでデバイスを明示的に選択してからメモリ情報を取得する。

#### 2.2.6 `rioxarray.open_rasterio` の非推奨

`FujiShaderGPU/core/dask_io.py` で `rioxarray.open_rasterio` を使用しています。

```python
return (
    rxr.open_rasterio(src_path, masked=True, chunks={'y': chunk, 'x': chunk}, lock=False)
    .squeeze()
    .astype('float32')
)
```

新しい rioxarray では `open_rasterio` が非推奨 / 削除される方向です。将来的に動作しなくなるリスクがあります。

**修正方針:**

- `xarray.open_dataset(src_path, engine="rasterio", chunks={...})` への移行を検討する。
- 移行時に CRS やチャンク挙動の差分をテストで確認する。

#### 2.2.7 `blur` アルゴリズムの整数出力範囲が未定義

`FujiShaderGPU/io/output_encoding.py` の `OUTPUT_VALUE_RANGES` に `blur` が含まれていません。

```python
OUTPUT_VALUE_RANGES: Dict[str, Tuple[float, float]] = {
    "topousm_fast": (-_NORM_HEADROOM, _NORM_HEADROOM),
    ...
    "slope": (0.0, 90.0),
    # "blur" がない
}
```

`--output-dtype int16/uint8` を `blur` と組み合わせると、`resolve_output_range` が `None` を返し、`_estimate_output_range` で中央窓のパーセンタイルから推定されます。全体ラスターの極値が含まれない可能性があり、**クリッピングによる情報欠落**が発生するリスクがあります。

**修正方針:**

- `blur` は「平滑化した標高」なので、入力 DEM の値域を metadata または事前統計から取得し、`OUTPUT_VALUE_RANGES` に動的に登録する。
- または、`blur` の整数出力を非推奨にし、CLI で警告を出す。

#### 2.2.8 `slope` の `percent` 単位で整数出力時のクリッピング

`resolve_output_range` は `slope` + `unit="percent"` の場合 `None` を返します。`--output-dtype int16/uint8` 時に中央窓推定になり、急斜面が clipping される可能性があります。

**修正方針:**

- 一般的な DEM で想定される最大勾配（例: 0〜10,000% またはピクセルサイズから推定）を固定範囲として提供する。
- または `--output-range` 必須にする警告を出す。

#### 2.2.9 定数の散在

以下のような値が各所にハードコードされています。

- `MAX_DEPTH = 150`
- `NORMAL_PERCENTILE = 99.0`
- overview levels `[2, 4, 8, 16, 32, 64, 128, 256]` / `[..., 512]`
- `BLOCKXSIZE = 512`
- `STRIPS_PER_WORKER = 6`
- `sample_max = 2048`

**修正方針:**

- `FujiShaderGPU/config/constants.py` を新設し、意味付きの定数に集約する。
- アルゴリズム別のデフォルト値も可能な限りそこから参照する。

#### 2.2.10 COG 検証結果が無視される

`FujiShaderGPU/io/cog_validator.py` の `_validate_cog_for_qgis` は `bool` を返しますが、`tile_processor.py` / `resume_cog_generation` では戻り値を確認していません。

```python
_validate_cog_for_qgis(output_cog_path)  # 戻り値を捨てている
```

検証スコアが 60 未満でも「完了」として扱われます。

**修正方針:**

- スコアが閾値未満の場合は `RuntimeError` を raise する。
- または、最低限 `logger.error` を出して `success` カウントに加算しないようにする。

```python
if not _validate_cog_for_qgis(output_cog_path):
    raise RuntimeError(f"COG validation failed for {output_cog_path}")
```

#### 2.2.11 `__main__.py` のエラー隠蔽

```python
except Exception as e:
    print(f"Execution failed: {e}")
    sys.exit(1)
```

スタックトレースが全く出ず、サポート時に原因を特定できません。

**修正方針:**

- `--verbose` / `-v` オプションを追加し、verbose 時は `logging.exception` でトレースバックを出力する。
- 常に `logger.exception` で記録し、UI 表示は簡潔にしてもログファイルには詳細を残す。

### 2.3 低重大度（改善推奨）

#### 2.3.1 コード重複

- `core/dask_processor.py` と `core/tile_processor.py` におおむね同じ「radii 解決 → 統計注入 → overview coarse source → 出力エンコーディング」の流れがある。
- `_compute_topousm_fast_overview_coarse_field` と `_compute_topousm_fast_overview_coarse_field_tile` がほぼ重複。
- `cog_builder.py` と `dask_processor.py` に `_get_overview_count` / `_assert_has_overviews` / `_build_zstd_overviews` が重複。

**修正方針:**

- `FujiShaderGPU/io/cog_common.py` などに統合する。

#### 2.3.2 未使用・死んだコード

- `FujiShaderGPU/algorithms/_global_stats.py` の `compute_global_stats` は `_norm_stats.py` に置き換えられた可能性があり、呼び出し元がなければ削除。
- `FujiShaderGPU/algorithms/_base.py` の `classify_resolution` は呼び出し元が見当たらない（`vulture` で確認推奨）。
- `FujiShaderGPU/algorithms/dask_shared.py` がまだ存在し、tile 側がそこからアルゴリズムクラスを import している。`_impl_*.py` への完全移行状況を整理する。

**修正方針:**

- `vulture` を実行し、未使用関数をリストアップして整理する。

#### 2.3.3 型ヒントの不足

多くの関数の引数・戻り値に型ヒントがありません。`pyproject.toml` では `disallow_untyped_defs = false` です。

**修正方針:**

- 新規コードから型ヒントを付与し、徐々に既存コードにも適用する。
- 特に `algo_params: dict` のような曖昧な引数は `TypedDict` または Pydantic/dataclass 化を検討する。

#### 2.3.4 テストカバレッジの穴

現状 15 ファイル程度のテストは主に単体・smoke です。不足している代表的なケース：

- タイルパイプラインの**実 COG 入力を使った end-to-end テスト**
- **Zarr 出力パス**の本格テスト（`test_zarr_io.py` は小さい）
- Dask-CUDA クラスタのライフサイクルテスト（CPU-only 環境では mock）
- GDAL 失敗モード（`gdaladdo` 不在、COG ドライバ不在）
- メモリプレッシャー / OOM 時の挙動
- macOS バックエンドが `WindowsCLI` を使うことの確認テスト
- `scales` 系アルゴリズムのパディング計算の回帰テスト

#### 2.3.5 ログ・プログレスの一貫性

- `core/dask_processor.py` では `tqdm` と `ProgressBar` と `progress()` が混在。
- 一部で f-string ログ、一部で `%s` ログが混在。

**修正方針:**

- logging では遅延評価のため `%s` 形式を統一する。
- プログレス表示は 1 つの抽象化クラスに集約する。

---

## 3. バグと思われる具体的なコード箇所

### 3.1 `_required_padding_for_algorithm` の `scales` 無視

`FujiShaderGPU/core/tile_processor.py` 内（約 line 237 付近）:

```python
elif algorithm == "visual_saliency":
    scales = _unified_radii([2, 4, 8, 16])  # ← scales パラメータを無視
```

同様に `multiscale_terrain`、`scale_space_surprise` でも同じパターン。

### 3.2 `_norm_stat_max_scale` の `scales` 無視

`FujiShaderGPU/algorithms/_norm_stats.py` 約 line 89 付近:

```python
def _norm_stat_max_scale(merged: dict) -> float:
    vals = []
    for key in ("radii", "scales"):  # ← "vs_scales" / "surprise_scales" / "fractal_radii" がない
```

### 3.3 Dask hybrid overview が `radii` を参照

`FujiShaderGPU/core/dask_processor.py` 約 line 1243 付近:

```python
_radii = [float(r) for r in (params.get("radii") or [])]
```

`visual_saliency` / `scale_space_surprise` では `params.get("scales")` を参照すべき。

### 3.4 地理座標のメートル換算不一致

`FujiShaderGPU/cli/linux_cli.py` と `FujiShaderGPU/io/raster_info.py` の計算式が異なる（前述）。

### 3.5 タイル spatial モードの Dask フォールバック

`FujiShaderGPU/algorithms/tile/dask_bridge.py` 約 line 426 付近:

```python
if str(p.get("mode", "local")).lower() == "spatial" and class_name in { ... }:
    raise _FallbackToDask()
```

### 3.6 一時ディレクトリの無条件削除

`FujiShaderGPU/core/tile_processor.py` 約 line 677 付近:

```python
if candidate.exists():
    if candidate.is_dir():
        shutil.rmtree(candidate)
```

### 3.7 `gdal.DontUseExceptions()` の多重呼び出し

`FujiShaderGPU/io/cog_builder.py`、`FujiShaderGPU/io/cog_validator.py`、`FujiShaderGPU/config/system_config.py` に散在。

### 3.8 包括的な `except Exception`

`FujiShaderGPU/algorithms/__init__.py` 約 line 9、49:

```python
except Exception:
    pass
```

### 3.9 `__main__.py` のトレースバック隠蔽

```python
except Exception as e:
    print(f"Execution failed: {e}")
    sys.exit(1)
```

### 3.10 `coarse_large_radius_response` 内の `.compute()`

`FujiShaderGPU/algorithms/_nan_utils.py` 約 line 385 付近:

```python
coarse_resp = coarse.map_overlap(
    block_fn, ...
).compute()
```

### 3.11 `OUTPUT_VALUE_RANGES` に `blur` なし

`FujiShaderGPU/io/output_encoding.py` 約 line 40 付近。

### 3.12 スレッド rasterio リーダーの close 漏れ

`FujiShaderGPU/core/tile_io.py` 全体。


---

## 4. 性能上の懸念

### 4.1 Dask 直接書き込みパスのメモリ推定

`write_cog_da_chunked` はホスト RAM / worker limit に基づいて直接書き込みか chunk 書き込みかを選択しますが、以下が気になります。

- `psutil` はホスト RAM を返すため、コンテナ limit での clamp は `worker_limit_gb` で行っていますが、worker limit が取れない場合に誤って direct path を選択しやすい。
- `single_worker_direct_cap = worker_limit_gb * 0.85 / 3.0` の分母 `3.0` は経験則です。アルゴリズムによって中間メモリは大きく変わるため、より保守的にするか、実測に基づくフィードバックを入れるべきです。

### 4.2 `cp.get_default_memory_pool().free_all_blocks()` の多用

`_norm_stats.py`、`_impl_specular.py`、`dask_processor.py` のクリーンアップで `free_all_blocks()` を呼んでいます。これはメモリを OS に返すため、次回の GPU メモリ確保時に再割り当てコストが発生し、断片化も進みます。特に RMM pool を使う場合は相性が悪く、パフォーマンスを低下させる可能性があります。

### 4.3 タイルバックエンドの GPU 直列化

`ThreadPoolExecutor` で複数ワーカーが CuPy の default stream に処理を投げても、同じストリーム上では GPU カーネルは直列実行されます。VRAM 使用率を下げる工夫はありますが、**GPU 利用率を上げる CUDA stream 並列化**がないため、タイルサイズが小さいほど GPU 使用率が下がります。

### 4.4 大半径の `map_overlap` での rechunk

`multiscale_topousm_fast` などで `depth = max_radius + 16` を指定します。チャンクサイズが depth より小さい場合、Dask は自動 rechunk します。大きな半径を指定すると、意図せず single-chunk 化や巨大な halo 読み出しが発生し、OOM の原因になります。`_required_padding_for_algorithm` の計算と整合性を保ちつつ、ユーザーに対して事前警告を強化すべきです。

---

## 5. 推奨される修正優先順位

### 優先度 1（すぐに対応すべき）

1. **`scales` / `radii` 混在の修正**
   - `_required_padding_for_algorithm` で `scales` 系パラメータを正しく参照
   - `_norm_stat_max_scale` に `scales` 系を追加
   - Dask hybrid overview 生成で `visual_saliency` / `scale_space_surprise` は `params.get("scales")` を参照
2. **一時ディレクトリの安全な扱い**
   - デフォルト名をユニーク化、または削除前の確認
3. **GDAL 例外モデルの見直し**
   - `gdal.DontUseExceptions()` を 1 箇所に集約、または `UseExceptions()` 移行計画
4. **包括的 `except Exception` の絞り込み**
   - 特に `algorithms/__init__.py`、`__main__.py`、コアパイプラインの重要箇所

### 優先度 2（短期間で対応すべき）

5. **地理座標メートル換算の共通化**
6. **スレッド rasterio リーダーのクリーンアップ**
7. **`coarse_large_radius_response` 内 `.compute()` の排除**
8. **`OUTPUT_VALUE_RANGES` に `blur` を追加**
9. **COG 検証結果を処理結果に反映**
10. **Dask 環境変数を `dask.config.set` に移行**
11. **`GPUtil` への依存を減らし CuPy 検出を優先**

### 優先度 3（中長期的な改善）

12. **巨大モジュールの分割**
13. **`rioxarray.open_rasterio` の移行検討**
14. **定数の集約**
15. **型ヒント・ドキュメントの強化**
16. **テストカバレッジ拡充（E2E tile、Zarr、GDAL 失敗モード）**
17. **CUDA stream 並列化の検討**
18. **Dask フォールバック削減のための tile 側 direct 実装拡充**

---

## 6. リポジトリクリーンアップ

以下は生成物・古いファイルです。リリース前に削除し、`.gitignore` に追加することを強く推奨します。

```text
build/
htmlcov/
FujiShaderGPU.egg-info/
**/__pycache__/
.ruff_cache/
.pytest_cache/
*.pyc
*.pyo
```

特に `build/lib/FujiShaderGPU/core/tile_processor.py` は、ソースより 400 行以上長い古いコードが残っており、レビュー・差分・パッケージングの混乱を招きます。

---

## 7. まとめ

FujiShaderGPU は **アーキテクチャレベルでは堅牢で、大規模 DEM 可視化に適した設計**を持っています。特に両バックエンドでアルゴリズム実装を共有し、大半径演算を COG overview で効率化する方針は優れています。

一方、**例外処理の甘さ、`radii`/`scales` パラメータの混在、地理座標換算の不一致、GDAL 例外モデルの競合、一時ディレクトリの安全面**など、改善すべき点が複数あります。これらを放置すると、特に大規模・地理座標系・カスタムスケール指定時に、シーム・OOM・誤った結果・デバッグ困難なエラーが発生する可能性があります。

優先度 1 の項目から順に対応し、テスト（特に実 COG を使った tile パイプラインの E2E テスト）で回帰を防ぐことをお勧めします。
