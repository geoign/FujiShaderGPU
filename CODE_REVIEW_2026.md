# FujiShaderGPU 包括的コードレビューレポート

**レビュー日:** 2026-06-17  
**レビューア:** Cline (AI Software Engineer)  
**対象範囲:** `FujiShaderGPU/` 全ソースコード（algorithms/, core/, io/, config/, utils/, cli/, tests/, tools/）

---

## 1. 全体評価

FujiShaderGPUは、GPU（CuPy/Dask-CUDA）を用いた大規模DEMラスタの地形可視化処理ソフトウェアとして、非常に高度な設計がなされています。アーキテクチャドキュメント（`ARCHITECTURE.md`）は629行に及び、設計意図が詳細に文書化されています。モジュール分割、NaNアウェアな処理、タイル/チャンクバックエンド間の出力パリティ、オーバービュー由来の大半径計算など、多くのベストプラクティスが実装されています。

しかし、コードを直接検証した結果、いくつかのバグ、非効率な点、および設計上の懸念が見つかりました。以下に重要度順に報告します。

---

## 2. バグ（Critical・High）

### BUG-1 [Critical] `cog_builder.py` — `.vrt` 文字列置換によるパス破壊

**ファイル:** `FujiShaderGPU/io/cog_builder.py`  
**行:** 67, 504

```python
file_list_path = vrt_path.replace(".vrt", "_files.txt")  # line 67
file_list_path = vrt_path.replace(".vrt", "_files.txt")  # line 504
```

`str.replace(".vrt", "_files.txt")` は最初の出現だけでなく**全ての出現**を置換します。一時ディレクトリのパスに `.vrt` が含まれる場合（例: `C:\data\.vrt_cache\tiles.vrt`）、結果は `C:\data\_files_cache\tiles_files.txt` となり、ファイルリストが正しい場所に作成されません。

興味深いことに、同じファイル内の `_create_cog_gtiff_ultra_fast`（行354-358）では、全く同じ問題を `Path.with_name()` を使って修正しています：

```python
_out_path = Path(output_cog_path)
temp_tiff_path = str(_out_path.with_name(f"{_out_path.stem}_temp{_out_path.suffix}"))
```

**修正方針:** 同じアプローチを適用：
```python
_vrt_path = Path(vrt_path)
file_list_path = str(_vrt_path.with_name(f"{_vrt_path.stem}_files.txt"))
```

---

### BUG-2 [High] `cog_builder.py` — モジュールインポート時のグローバル副作用

**ファイル:** `FujiShaderGPU/io/cog_builder.py`  
**行:** 24

```python
gdal.DontUseExceptions()
```

モジュールインポート時に `gdal.DontUseExceptions()` を呼ぶと、プロセス全体のGDAL例外設定がグローバルに変更されます。他のモジュールやライブラリがGDAL例外を有効にすることを期待している場合、エラーが黙殺されてバグの発見が困難になります。

**修正方針:** インポート時ではなく、実際にGDAL操作を行う関数内で設定する。または、コンテキストマネージャーを使用して関数終了後に元の設定を復元する。

---

### BUG-3 [High] 複数ファイル — 可変デフォルト引数（Mutable Default Argument）

**ファイル・行:**
- `FujiShaderGPU/algorithms/_impl_topousm_fast.py:51` — `radii: List[int] = [4, 16, 64]`
- `FujiShaderGPU/algorithms/_impl_fractal_anomaly.py:81` — `radii=[4, 8, 16, 32, 64]`
- `FujiShaderGPU/algorithms/_impl_fractal_anomaly.py:81` — `compute_fractal_dimension_block`
- `FujiShaderGPU/algorithms/_impl_visual_saliency.py:80` — `scales=[2, 4, 8, 16]`
- `FujiShaderGPU/algorithms/_impl_openness.py:81` — `radii=[4, 8, 16, 32, 64]`

Pythonの古典的な落とし穴です。デフォルト引数として可変オブジェクト（リスト）を使用すると、関数が呼ばれるたびに同じリストオブジェクトが再利用されます。現在は関数内でリストを変更していないため実害はありませんが、将来的なバグの温床となります。

**修正方針:**
```python
def compute_topousm_fast_efficient_block(block, *, radii=None, ...):
    if radii is None:
        radii = [4, 16, 64]
```

---

### BUG-4 [High] 複数 `__init__.py` — 例外の黙殺

**ファイル・行:**
- `FujiShaderGPU/__init__.py:9-10, 49-50` — `except Exception: pass`
- `FujiShaderGPU/core/__init__.py:12-32` — `except ImportError: pass`

```python
try:
    from .tile_processor import process_dem_tiles, resume_cog_generation
    __all__.extend(["process_dem_tiles", "resume_cog_generation"])
except ImportError:
    pass
```

構文エラーや循環インポートなど、実際のバグが完全に隠蔽されます。ユーザーは関数が存在しないことに気づくだけで、原因が全く分かりません。

**修正方針:** 最低限ログを出力する：
```python
except ImportError as e:
    logging.getLogger(__name__).warning("Import failed: %s", e)
```

---

### BUG-5 [Medium] `_impl_fractal_anomaly.py` — CPU計算にCuPyを使用

**ファイル:** `FujiShaderGPU/algorithms/_impl_fractal_anomaly.py`  
**行:** 348

```python
indices = cp.linspace(0, len(base)-1, 6).astype(int).get()
base = [base[int(i)] for i in indices]
```

単純なインデックス計算にCuPyを使用し、`.get()` でGPU→CPU転送を行っています。この時点ではGPUコンテキストが利用可能である可能性が高いですが、不必要なオーバーヘッドです。

**修正方針:**
```python
indices = np.linspace(0, len(base)-1, 6).astype(int)
```

---

### BUG-6 [Medium] `dask_processor.py` — `as_completed` イテレーションの潜在的レースコンディション

**ファイル:** `FujiShaderGPU/core/dask_processor.py`  
**行:** 553-570

```python
for fut in inflight:
    i, j = fut_meta.pop(fut)
    try:
        arr = fut.result()
    finally:
        del fut
    if write_err:
        raise write_err["e"]
    write_q.put((arr, col_off[j], row_off[i]))
    del arr
    done += 1
    if done % 10 == 0:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass
    pbar.update(1)
    _submit_next()
```

`write_err` のチェックが `fut.result()` の後に行われますが、ライタースレッドでエラーが発生した場合、次の `fut` の処理に進む前に例外を送出すべきです。現在のコードでは `fut.result()` が成功した場合でも、直前のライターエラーを遅延して検出する可能性があります。また、`del fut` の後に `write_err` をチェックしていますが、`finally` ブロックで `del fut` が実行されるため、`fut` は既に削除されている可能性があります。

実際には、`del fut` は `finally` ブロック内にあるため、例外が発生しても実行されます。その後 `write_err` がチェックされるため、ライターエラーは次のイテレーションの開始時に検出されます。これは数チャンク遅れのエラー検出となりますが、致命的ではありません。

**修正方針:** `fut.result()` の直後に `write_err` をチェックする：
```python
arr = fut.result()
if write_err:
    raise write_err["e"]
del fut
```

---

## 3. 非効率な点

### PERF-1 [Medium] DRY違反：バックエンド間の重複コード

**対象:**
- `tile_processor.py:506-556` (`_compute_topousm_fast_overview_coarse_field_tile`)
- `dask_processor.py:860-912` (`_compute_topousm_fast_overview_coarse_field`)

これら2つの関数は、COGオーバービューを読み取りTopoUSM Fast coarse fieldを計算するほぼ同一のコードです。両バックエンドで別々に実装されており、将来的な乖離のリスクがあります。

また、COGオプションの定義も `dask_processor.py:get_cog_options()` と `cog_builder.py:_create_cog_ultra_fast()` で重複しています。

**修正方針:** 共通のオーバービュー読み取り関数を `io/raster_info.py` または新しい `io/overview.py` に抽出し、両バックエンドから呼び出す。

---

### PERF-2 [Low] `dask_processor.py` — ハードコードされた `time.sleep(3)`

**ファイル:** `FujiShaderGPU/core/dask_processor.py`  
**行:** 1581

```python
time.sleep(3)
```

Daskワーカープロセスの終了を待つための固定スリープです。高速なシステムでは無駄な待機時間となり、低速なシステムでは不十分な可能性があります。

**修正方針:** `client.shutdown()` または `cluster.close()` のブロッキング動作に依存するか、プロセスの生存確認をポーリングする。

---

### PERF-3 [Low] `_impl_ambient_occlusion.py` — NaN領域のGaussian平滑化が無駄

**ファイル:** `FujiShaderGPU/algorithms/_impl_ambient_occlusion.py`  
**行:** 104-108

```python
if nan_mask.any():
    filled_ao = cp.where(nan_mask, 1.0, ao)
    ao = gaussian_filter(filled_ao, sigma=1.0, mode='nearest')
else:
    ao = gaussian_filter(ao, sigma=1.0, mode='nearest')
```

NaN領域を1.0で埋めてからGaussian平滑化を行っていますが、NaN領域の値は最終的に `restore_nan` で上書きされるため、NaN領域での平滑化計算は無駄です。ただし、境界近傍の有効ピクセルに影響を与えるため、完全に無駄というわけではありません。`handle_nan_with_gaussian` を使用すれば、より正確なNaNアウェアな平滑化が可能です。

**修正方針:** `handle_nan_with_gaussian` を使用して一貫性を確保する：
```python
ao, _ = handle_nan_with_gaussian(ao, sigma=1.0, mode='nearest')
```

---

### PERF-4 [Low] `_impl_openness.py` / `_impl_ambient_occlusion.py` — シフト演算の反復回数

**ファイル:** `FujiShaderGPU/algorithms/_impl_openness.py:86-117`  
**ファイル:** `FujiShaderGPU/algorithms/_impl_ambient_occlusion.py:67-96`

Openness は `num_directions × len(distances)` 回、AO は `4 × num_samples` 回の配列シフト演算をループで実行します。それぞれ `gaussian_filter` などのカーネル関数を使用せず、Pythonレベルのループで実行しているため、カーネル起動オーバーヘッドが累積します。

ただし、CuPyのブロードキャスト演算を使用しているため、GPU上では効率的に実行されます。カスタムCUDAカーネルを書けばさらに高速化できますが、現在の性能でも実用上は問題ないと判断します。

---

## 4. 数値的安定性・正確性

### NUM-1 [Medium] `_impl_fractal_anomaly.py` — 回帰係数のゼロ除算ガードが不十分

**ファイル:** `FujiShaderGPU/algorithms/_impl_fractal_anomaly.py`  
**行:** 140, 146

```python
beta = cov / (var_log_scale + 1e-10)
r2 = cp.clip(1.0 - ss_res / (ss_tot + 1e-10), 0.0, 1.0)
```

`var_log_scale` が0に近い場合（全てのスケールが同一の場合）、`beta` は非常に大きな値になります。また、`ss_tot` が0の場合（全ての粗さ値が同一）、`r2` は `1.0 - 0/1e-10 = 1.0` となり、完全な適合を示してしまいます。

これらはエッジケースであり、実際の地形DEMでは発生しにくいですが、理論的には問題があります。

**修正方針:** `var_log_scale` が極小値の場合は `beta = 0` とするガードを追加：
```python
if var_log_scale < 1e-6:
    beta = cp.zeros_like(cov)
else:
    beta = cov / var_log_scale
```

---

### NUM-2 [Low] `_impl_specular.py` — `half_vec` の正規化の安全性

**ファイル:** `FujiShaderGPU/algorithms/_impl_specular.py`  
**行:** 88

```python
half_vec = (light_dir + view_dir) / cp.linalg.norm(light_dir + view_dir)
```

`light_dir + view_dir` がゼロベクトルの場合、NaNが発生します。デフォルトパラメータ（azimuth=315, altitude=45）では `light_dir = [sin(315°)cos(45°), cos(315°)cos(45°), sin(45°)]`、`view_dir = [0, 0, 1]` となり、ゼロになることはありません。しかし、極端なパラメータ（altitude=0, azimuth=0など）でも安全性を保証すべきです。

**修正方針:** ノルムに最小値を設定：
```python
norm = cp.linalg.norm(light_dir + view_dir)
half_vec = (light_dir + view_dir) / max(norm, 1e-10)
```

---

### NUM-3 [Low] `kernels.py` — `scale_space_surprise` の正規化パーセンタイル

**ファイル:** `FujiShaderGPU/algorithms/common/kernels.py`  
**行:** 66-69

```python
lo = cp.percentile(valid, 5)
hi = cp.percentile(valid, 95)
if hi > lo:
    surprise = cp.clip((surprise - lo) / (hi - lo), 0, 1)
```

5-95パーセンタイルでクリップして `[0, 1]` に正規化していますが、`lo` と `hi` が非常に近い場合（フラットな地形など）、`hi - lo` が極小になり、予期せぬ増幅が発生する可能性があります。`hi > lo` のチェックがありますが、極端に小さい差は考慮されていません。

**修正方針:** 最小差のガードを追加：
```python
if hi > lo + 1e-10:
```

---

## 5. 設計・アーキテクチャの懸念

### DESIGN-1 [Medium] `_required_padding_for_algorithm` の複雑性

**ファイル:** `FujiShaderGPU/core/tile_processor.py`  
**行:** 190-346

この関数は157行に及び、アルゴリズムごとに異なるパディング計算ロジックを持ちます。多数の分岐とネストされた条件があり、メンテナンスが困難です。パディングが不十分だとタイルシームが発生し、過剰だとVRAMを無駄に消費します。

**修正方針:** 各アルゴリズムの `DaskAlgorithm` 基底クラスに `required_halo()` メソッドを追加し、アルゴリズム自身が必要なハローを宣言する設計に移行する。これにより `tile_processor.py` は単にそのメソッドを呼ぶだけになる。

---

### DESIGN-2 [Medium] テストカバレッジの不足

`tests/test_algorithm_smoke.py` は15アルゴリズム中2つ（`scale_space_surprise`, `multi_light_uncertainty`）しかスモークテストしていません。以下の重要なパスが未テストです：

- 量子化（int16/uint8エンコーディング）の正確性
- NaN伝播のエッジケース（全NaNタイル）
- オーバービュー読み取りのフォールバック
- 地理座標系DEMのピクセルスケール変換
- Dask/タイルバックエンド間の出力パリティ（数値的比較）
- COGオーバービュー構造の検証（部分的に `test_cog_overviews.py` でカバー）

**修正方針:** 
1. 全アルゴリズムのスモークテストマトリクスを作成
2. 量子化エンコーディングの往復テスト（float→int→float の誤差検証）
3. 既知の小さなDEMに対するDask/タイル出力の数値比較テスト
4. NaN含有入力のエッジケーステスト

---

### DESIGN-3 [Low] `restore_nan` のインプレース変更

**ファイル:** `FujiShaderGPU/algorithms/_nan_utils.py`  
**行:** 658-662

```python
def restore_nan(result: cp.ndarray, nan_mask: cp.ndarray) -> cp.ndarray:
    if nan_mask.any():
        result[nan_mask] = cp.nan
    return result
```

渡された配列をインプレースで変更します。呼び出し元が同じ配列を別の目的で使用している場合、予期せぬ副作用が発生します。現在の呼び出し元では問題ありませんが、将来的なバグの原因となります。

**修正方針:** ドキュメントでインプレース変更を明示するか、コピーを作成してから変更する。

---

### DESIGN-4 [Low] `blur` アルゴリズムのモード非対応

`blur` アルゴリズムはレジストリに登録されていますが、`_required_padding_for_algorithm` の `spatial_algorithms` セットに含まれておらず、spatial モードのサポートがありません。`ARCHITECTURE.md` の §11 では `blur` はlocal/spatialモードをサポートするアルゴリズムとしてリストされていませんが、他のドキュメントでは暗黙的にサポートを期待する記述があります。

**修正方針:** 意図的な設計であればドキュメントで明記し、そうでなければ spatial モードのサポートを追加する。

---

## 6. コード品質・スタイル

### STYLE-1 [Low] 型アノテーションの不整合

**ファイル:** `FujiShaderGPU/algorithms/_impl_openness.py:28-29`

```python
pixel_scale_x: float = None,
pixel_scale_y: float = None,
```

型アノテーションが `float` ですが、デフォルト値は `None` です。同様のパターンが `_impl_ambient_occlusion.py:29-30` など複数ファイルに存在します。

**修正方針:** `Optional[float] = None` とする。

---

### STYLE-2 [Low] 行の長すぎるコード

**ファイル:** `FujiShaderGPU/core/tile_processor.py:194` など

```python
coarse_dem=params.get("_overview_coarse_dem"),
coarse_decimation=params.get("_overview_decimation"), tile_origin=params.get("_tile_origin"), tile_full_shape=params.get("_tile_full_shape"),
```

1行に複数のキーワード引数が詰め込まれており、可読性が低いです。同様のパターンが `_impl_openness.py:194`、`_impl_ambient_occlusion.py:179` などでも見られます。

**修正方針:** 各引数を改行して配置する。

---

### STYLE-3 [Low] `f"{algorithm}"` の不要なf-string

複数ファイルで `f"{variable}"` のように、変数1つだけのf-stringが使用されています。これは `str(variable)` または変数そのものと同等です。

---

## 7. バックエンド間の一貫性

### CONSIST-1 [Good] 出力パリティの設計

アーキテクチャ上、Daskバックエンドとタイルバックエンド間の出力パリティが非常に念入りに設計されています：

- 共通の正規化統計（`_norm_stats.inject_global_stats`）
- 共通のオーバービュー由来大半径フィールド
- 共通の出力エンコーディング（`output_encoding.py`）
- タイル起源を考慮したグローバル座標サンプリング

これは素晴らしい設計であり、両バックエンドで一致する結果が期待できます。

---

### CONSIST-2 [Medium] TopoUSM Fast の統計サンプリングがDaskとタイルで異なる

**Dask:** `compute_topousm_fast_input_sample_stats` は中心窓から統計を推定  
**Tile:** `inject_global_stats` を通じて層化サンプリングで統計を計算

両者は異なるサンプリング戦略を使用しているため、同じDEMに対しても異なる正規化統計が得られる可能性があります。ただし、タイルバックエンドでは `inject_global_stats` が常に呼ばれるため、Dask側でのみフォールバック時の不一致が生じます。

**修正方針:** Daskバックエンドでも `inject_global_stats` の結果を優先し、フォールバック時のみ中心窓サンプリングを使用する（現状の動作だが、ログで明示する）。

---

## 8. セキュリティ・堅牢性

### SEC-1 [Low] `subprocess.run` のコマンドインジェクションリスク

**ファイル:** `FujiShaderGPU/io/cog_builder.py:87, 523-555`

`subprocess.run` は `check=True` で呼ばれており、引数はリスト形式で渡されているため、シェルインジェクションのリスクはありません。ただし、ファイルパスがユーザー入力から来る場合、特殊文字を含むパスが問題を起こす可能性があります。現在は `capture_output=True` でエラーが捕捉されているため、実害は低いです。

---

### SEC-2 [Low] 一時ディレクトリのクリーンアップ不備

**ファイル:** `FujiShaderGPU/core/tile_processor.py:1568-1572`

```python
except Exception as e:
    if os.path.exists(tmp_tile_dir):
        logger.error(f"An error occurred ({e}). Keeping the tile directory: {tmp_tile_dir}")
```

エラー時に一時ディレクトリを保持する設計は意図的（COG再生成のため）ですが、ディスク容量を消費し続ける可能性があります。特に大規模ラスタの場合、GB単位の一時ファイルが残ります。

**修正方針:** `--keep-tiles` オプションを追加し、デフォルトではクリーンアップする。または、古い一時ディレクトリを起動時に検出して削除する。

---

## 9. ドキュメントの正確性

### DOC-1 [Low] `ARCHITECTURE.md` のアルゴリズムリスト

`ARCHITECTURE.md` §4 のアルゴリズムカタログ（15アルゴリズム）は、`dask_registry.py` の `ALGORITHMS` 辞書と一致しています。`blur` アルゴリズムも含まれており、正確です。

---

### DOC-2 [Low] README.md の記載事項

READMEの内容は未確認ですが（今回のレビューでは読み込んでいません）、ARCHITECTURE.mdが非常に詳細なため、READMEとの整合性を確認することを推奨します。

---

## 10. 推奨される修正優先度

| 優先度 | ID | 問題 | 影響 |
|--------|----|----|------|
| **P0** | BUG-1 | `.vrt` 文字列置換 | 特定パスでファイルリスト破壊 |
| **P0** | BUG-2 | `gdal.DontUseExceptions()` | プロセス全体のGDAL例外無効化 |
| **P1** | BUG-3 | 可変デフォルト引数 | 将来的なバグの温床 |
| **P1** | BUG-4 | 例外の黙殺 | デバッグ困難性 |
| **P1** | DESIGN-2 | テストカバレッジ | リグレッション検出力不足 |
| **P2** | BUG-5 | CuPyの不要使用 | 微小な性能ロス |
| **P2** | BUG-6 | `as_completed` エラー検出遅延 | 数チャンク遅れのエラー |
| **P2** | PERF-1 | バックエンド間コード重複 | 保守性 |
| **P2** | NUM-1 | ゼロ除算ガード | エッジケースでの不正確な値 |
| **P2** | CONSIST-2 | TopoUSM Fast 統計サンプリング | バックエンド間の微細な不一致 |
| **P3** | DESIGN-1 | パディング計算の複雑性 | 保守性 |
| **P3** | NUM-2 | `half_vec` 正規化 | 極端なパラメータでのNaN |
| **P3** | NUM-3 | パーセンタイル正規化 | フラット地形での不安定性 |
| **P3** | PERF-2 | `time.sleep(3)` | 不要な待機 |
| **P3** | PERF-3 | AO のNaN平滑化 | 境界精度 |
| **P3** | SEC-2 | 一時ディレクトリクリーンアップ | ディスク容量 |
| **P4** | STYLE-1-3 | コードスタイル | 可読性 |

---

## 11. 特に優れている点

コードレビューの過程で、以下の設計・実装が特に優れていることを確認しました：

1. **NaNアウェアな処理の一貫性** — `handle_nan_with_gaussian`, `handle_nan_with_uniform`, `_downsample_nan_aware` など、NaNを適切に処理するユーティリティが共有され、全アルゴリズムで一貫して使用されている。

2. **オーバービュー由来の大半径計算** — COGのオーバービューから大半径の低周波成分を計算し、タイル/チャンクごとのハローを削減する設計は、大規模ラスタの処理効率とシームフリー性を両立する優れたアプローチ。

3. **タイル起源認識** — `tile_origin` / `tile_full_shape` パラメータによるグローバル座標サンプリングは、タイルバックエンドとDaskバックエンドの出力パリティを実現する鍵となる設計。

4. **コンテナ対応** — `container_cpu_count()`, `container_memory_available_gb()` など、cgroupを認識したリソース検出により、Docker/Kubernetes環境での正確なリソース管理が行われている。

5. **出力エンコーディングの設計** — float32/int16/uint8の量子化エンコーディングは、NoData=0のポリシー、符号付き/符号なしの対称エンコーディング、GDAL scale/offsetを埋め込まない設計など、実運用を考慮した堅実な設計。

6. **メモリ最適化** — specularアルゴリズムでのコンポーネント別法線計算、fractal_anomalyでのインプレース累積、COGライターのバックプレッシャー付きキューなど、GPU/CPU メモリ管理が徹底的に最適化されている。

---

## 12. 結論

FujiShaderGPUは、大規模DEM処理のための高度でよく設計されたソフトウェアです。アーキテクチャの意図は明確で、多くのベストプラクティスが実装されています。しかし、以下の点に早急に対処することを推奨します：

1. **`.vrt` パス置換バグ（BUG-1）** の即時修正
2. **`gdal.DontUseExceptions()`（BUG-2）** のスコープ制限
3. **可変デフォルト引数（BUG-3）** の一括修正
4. **例外の黙殺（BUG-4）** へのログ追加
5. **テストカバレッジ（DESIGN-2）** の拡充

これらの修正は比較的少量のコード変更で完了し、ソフトウェアの堅牢性と保守性が大幅に向上します。

---

*このレポートは、既存のレポートを参照せず、コードの直接検証のみに基づいて作成されました。*