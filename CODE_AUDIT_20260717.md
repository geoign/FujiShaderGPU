# FujiShaderGPU コード監査レポート

**監査日:** 2026-07-17
**対象:** `FujiShaderGPU/` パッケージ全ソース（約15,500行）、`tests/`、`tools/`、`pyproject.toml`
**方法:** モジュール別の並行静的監査（8系統）＋ 重要指摘は全て実コードで再検証。`build/` は古いコピーのため対象外。
**凡例:** 各指摘の末尾の ✅ は本監査で実コードを再読して裏付けたもの、🔍 はコード上は妥当だが実環境での挙動確認が望ましいもの。

---

## 1. 総評

アーキテクチャの方向性（NaN=NoData統一、overview活用の大半径分離、バックエンド間パリティの明示的な設計目標、cgroup考慮のメモリ管理）は堅実で、過去のレビュー（`CODE_REVIEW_2026.md` 等）で指摘された危険な tmp ディレクトリ処理などは既に修正済み。主経路でのサイレントなデータ破壊に相当する Critical は見つからなかった。

一方で以下の構造的弱点が残る:

1. **タイル↔Dask パリティの穴**が最も重要な問題群。topousm_fast の正規化統計の計算順序（H-1）、`--agg stack` の出力形状契約（H-5）、レンジ未解決時のフォールバック分岐など、「両バックエンドで同一出力」という設計目標を静かに破る箇所が複数ある。
2. **エラーパスの弱さ**。例外経路での writer スレッドリーク、オーバービュー再構築失敗時の出力喪失など、正常系は丁寧だが異常系で資源・データを失う経路が残る。
3. **環境変数まわりのデッドコード**。`RMM_*` は完全に読まれていない、`DASK_DISTRIBUTED__*` は import 済みの dask には効かない、「効いているように見えて効いていない」設定が運用者を誤解させる。
4. **新規6アルゴリズム（2026-07追加）周辺の仕上がり不足**。scale_drift の NaN 浸食（H-3）、structure_tensor の2倍計算（H-4）、統計フォールバックの不統一など、兄弟実装で確立済みの対策パターンが適用され忘れている箇所が目立つ。

**件数サマリ:** High 6件 / Medium 26件 / Low 40件強（本レポートに全件記載）

---

## 2. High（実コードで再検証済み）

### H-1 ✅ topousm_fast: tile バックエンドで global_stats が半径分割「後」に計算され、Dask と正規化スケールが不一致

`core/tile_processor.py:1210-1218` → `core/tile_processor.py:1459`

```python
algo_params["_topousm_fast_full_radii"] = list(_full_r)    # 1210: 書き込むが…
algo_params["_topousm_fast_full_weights"] = list(_full_w)  # 1211: …どこからも読まれない
algo_params["radii"] = _sr        # 1212: 小半径に差し替え
algo_params["weights"] = _sw
...
inject_global_stats(input_cog_path, algorithm, algo_params, is_zarr=False)  # 1459
```

- `_topousm_fast_full_radii` / `_topousm_fast_full_weights` はパッケージ全体で**書き込みのみ・読み取りゼロ**のデッド変数（Grepで全件確認）。コメント「Keep the full radii for the global normalization stat」の意図が配線されていない。
- `_compute_norm_stats_tiled`（`algorithms/_norm_stats.py:258-261`）は差し替え後の `params["radii"]`（小半径のみ）で統計を計算する。
- Dask 側は `core/dask_processor.py:1280` で `inject_global_stats` を呼び、overview 分割はその後（:1290-）。つまり **Dask=全半径で統計、tile=小半径のみで統計**。
- 影響: spatial モードで大半径（overview パス）を含む topousm_fast 実行時、tile 側の表示ストレッチが大半径成分を含まない分布から推定される。大半径成分は USM 振幅の主因なので、tile 出力が Dask 出力より過増幅される方向にパリティが崩れる。

**修正方向:** tile 側も `inject_global_stats` を半径分割より前に呼ぶ順序に変えるか、統計計算時だけ `_topousm_fast_full_radii/_weights` を `radii/weights` として渡す（デッド変数を実際に使う）。

---

### H-2 ✅ Dask: マルチ半径用チャンク縮小がデフォルト（auto radii）経路で一切発動しない

`core/dask_processor.py:1112-1122`（半径解決は :1230-1238 で後段）

```python
_n_radii = len(radii) if radii else 0   # run_pipeline の生引数。auto時は None
if _n_radii > 1:
    _shrink = (2.0 / (1.0 + min(_n_radii, 6))) ** 0.5
    ...
    chunk = _shrunk
```

コメントにある通り「multi-radius spatial run は半径ごとの応答を同時保持し per-block VRAM が数倍になる」ための縮小策だが、チェックが半径解決（`auto_spatial_profile`、:1231）より前に行われる。デフォルトの auto モード（`radii=None` → 最大6半径）では `_n_radii=0` となり**縮小が絶対に適用されない**。つまり最も一般的なマルチスケール実行が無防備で、大きなチャンク＋複数半径の同時保持により RMM プール OOM のリスクが残る。

**修正方向:** 縮小判定を半径解決後に移すか、auto 時は `len(auto_spatial_radii(short_side))` で判定する。

---

### H-3 ✅ scale_drift: coarse 経路の大スケール平滑場の NaN が有効画素に浸食する

`algorithms/_impl_scale_drift.py:149-155`（`_drift_combine_block`）、同:85-128（`_drift_vector`）、`algorithms/_nan_utils.py:400`

```python
def _drift_combine_block(block, *smooths, ...):
    nan_mask = cp.isnan(block)
    dx, dy = _drift_vector(list(smooths), scales, pair_w)   # smooths の NaN を無処理で渡す
```

- 大スケール（`4s+1 > MAX_DEPTH(=150)`、すなわち s≳38）の平滑場は `coarse_large_radius_response` 末尾の `da.where(da.isnan(gpu_arr), nan, upsampled)`（`_nan_utils.py:400`）で **NoData フットプリントに NaN が再マスク**される。
- `_drift_vector` は `cp.gradient(lo + hi)` → `gaussian_filter(gx*gx, sigma=w_sig)`（w_sig 最大 24 → 4σ≈96px）を掛けるため、NaN が **NoData 境界から最大 ~96px 内側の有効画素へ拡散**する。最後の `restore_nan` は元のフットプリントしか復元しないので、有効領域に NaN ホールが残る。
- 兄弟実装は対策済み: `_vs_combine_block`（`_impl_visual_saliency.py:187-190`）は `cp.where(cp.isnan(s), fillv, s)` で再充填、`_fractal_combine_block` も有限値に丸めている。scale_drift だけが未対処。
- デフォルト scales (2,4,8,16,32) では大スケール経路が発動しないため潜伏する。`--radii` で 38px 超を指定した spatial 実行で顕在化する。

**修正方向:** `_drift_combine_block` 冒頭で `_vs_combine_block` と同様に各 smooth の NaN をブロック充填値で置換してから `_drift_vector` に渡す。

---

### H-4 ✅ structure_tensor: u/v の2パス構成で全 GPU 計算がちょうど2倍、coarse DEM も二重生成

`algorithms/_impl_structure_tensor.py:232-237`、同:67-101

```python
u_fields = multiscale_response_fields(gpu_arr, radii, block_fn=st_component_block,
    coarse_cache={}, component='u', **common)
v_fields = multiscale_response_fields(gpu_arr, radii, block_fn=st_component_block,
    coarse_cache={}, component='v', **common)
```

- `st_component_block` は component が u でも v でも `_strike_uv` 全体（勾配 Gaussian 2本＋テンソル Gaussian 3本）を実行し、最後の1成分だけを使う。すなわち**全半径で GPU 計算がちょうど2倍**。
- `coarse_cache={}` を毎回新規に渡しているため、`da.coarsen` フォールバック時に共粗 DEM の persist（全解像度ラスタの読み出し＋縮約）が2回走る。`_nan_utils.py:355-367` のキャッシュは block_fn の結果を保持しないため、同一 dict の共有は安全。

**修正方向:** 1回の block_fn で (u, v) を2チャンネル場として同時に返す。少なくとも同一の `coarse_cache` dict を両パスに渡す。

---

### H-5 ✅ `--agg stack`: (C,H,W) と HxWxC の契約不一致で tile バックエンドの出力が壊れる

`algorithms/_nan_utils.py:180-183` → `algorithms/tile/dask_bridge.py:31-34, 492-497` → `core/tile_compute.py:125-136`、`core/tile_processor.py:927-932, 504-506`

- stack 出力は `da.stack(responses, axis=0)` により **(C,H,W) band-first**。tile 側は `_combine_direct` が `_FallbackToDask` を投げ、フォールバックが `result_da.compute()` の結果をそのまま返す。
- タイル側の後処理は全て HxWxC（channel-last）前提:
  - `apply_nodata_mask`（`tile_compute.py:131-133`）: `result_gpu[mask_gpu, :]` — (H,W) マスクを (C,H,W) の axis 0 に適用 → NoData 含有タイルで **IndexError**
  - コア切り出し（`tile_processor.py:927-932`）: `[core_y:..., core_x:..., :]` — band 軸を行として誤クロップ
  - `band_count = result_core.shape[-1]`（:504-506）→ 幅 W をバンド数と誤認
- Windows で `--agg stack`（`cli/args.py:52` で公開済み）を使うと、全アルゴリズムで例外または破損 COG になる。ブリッジ内の「writer contract が曖昧」という先送りコメント箇所そのものが顕在化した形。

**修正方向:** フォールバックで `result.ndim == 3` なら `cp.moveaxis(result, 0, -1)` に正規化して HxWxC 契約に合わせる（またはタイル側を band-first に統一）。どちらか一方に契約を確定させる。

---

### H-6 ✅ テストの環境ガード不統一: cupy/osgeo の直接 import でコレクション全体がクラッシュ

他の13ファイルは `pytest.importorskip("cupy")` 等でガードしているのに、以下4ファイルだけモジュールトップで直接 import:

- `tests/test_cog_overviews.py:5` — `from osgeo import gdal`
- `tests/test_visual_saliency_tile_stability.py:1` — `import cupy as cp`
- `tests/test_visual_saliency_normalization.py:1` — `import cupy as cp`
- `tests/test_fractal_anomaly_normalization.py:1` — `import cupy as cp`

GPU/GDAL の無い環境（CI lint、ドキュメント検証等）ではスキップではなく **collection error でスイート全体が起動しない**。pyproject の `dev` extra に cupy は含まれない。

併せて、`sys.path` ブートストラップの有無も不統一（9ファイルにあり、上記4件を含む8ファイルになし）。`conftest.py` は存在せず、pyproject にも `pythonpath` 設定がないため、パッケージ未インストール環境では半数のテストが import 不可。

**修正方向:** 全テストを `importorskip` に統一し、sys.path 挿入をやめて pyproject に `pythonpath = ["."]` を追加する。

---

## 3. Medium

### 3.1 core/tile

#### M-1 ✅ 数値 NoData と NaN が混在する入力で、元 NaN 画素が出力で再マスクされず有効値として漏洩

`core/tile_processor.py:827-830, 922`

`nodata` が数値（例: -9999）の場合 `_build_nodata_mask` は `np.isclose(data, nodata)` のみで、入力中の既存 NaN はマスクに含まれない。NaN 対応カーネルは void を fill して計算するため元 NaN 画素に有効値が「生える」。Dask 側は `da.where(da.isnan(gpu_arr), nan, result)`（`dask_processor.py:1431-1434`）で全 NaN を再マスクするので、出力フットプリントのパリティ違反でもある。副次的に `nodata_ratio` のスキップ判定も NaN をカウントしない。

**修正:** `mask_nodata |= np.isnan(dem_tile)` を合成する。

#### M-2 ✅ 正常終了時にタイル一時ディレクトリを一切削除しない

`core/tile_processor.py:1637-1658`

COG 生成・検証の成功後も `tmp_tile_dir` の全タイル（大規模 DEM では数十 GB の float32 GeoTIFF 群）が残置され、削除も残留通知のログもない。エラー時の保持は `--cog-only` 再開のため妥当だが、成功時に残す設計意図がコード・ログのどこにも表明されていない。

**修正:** 成功後にデフォルト削除（`--keep-tiles` で保持）、または最低限 info ログを出す。

#### M-3 `_infer_nodata_zero_from_border`: フル解像度の縁読み出し＋標高0=有効値の DEM 誤判定

`core/tile_processor.py:619-642`

- 上下左右4帯をフル解像度で読む。幅20万 px 級で `4×64×200,000×4B ≈ 205MB` の無駄な読み出し。`out_shape` で間引き読みすべき。
- 沿岸 DEM で海面=0.0 が有効な標高値の場合、縁の6割が海面なら `nodata=0` と推定され有効データ全体が NoData 化する。推定結果は info ログ1行のみで静かに適用される。

**修正:** 間引き読み出しに変更し、推定適用時は warning に格上げ。

#### M-4 🔍 `gpu_memory.py`: スレッド毎の `set_allocator` と「スレッドローカル」プールの実態不一致

`core/gpu_memory.py:11-28`

`cp.get_default_memory_pool()` はデバイス毎のプロセスグローバルプールを返すため、thread-local に保持しても全スレッド同一オブジェクト。rmm 存在環境では `hasattr` ガードがスレッドローカルなため各ワーカースレッドがプロセスグローバルな `cp.cuda.set_allocator` を再実行し、メインスレッド（デフォルトプール確保済み）とワーカー（RMM）でアロケータが混在し VRAM を二重計上しうる 🔍（Windows では `import rmm` が失敗するためデッドコード）。

**修正:** アロケータ設定はプロセス起動時1回のモジュールレベル初期化に分離。

### 3.2 core/dask

#### M-5 ✅ 「prefetch」メモリ設定は起動済みワーカーに効かない＋閾値の順序が逆転

`core/dask_processor.py:415-424`

```python
prefetch_config = {
    "distributed.worker.memory.pause": 0.90,
    "distributed.worker.memory.spill": 0.95,
}
...
with dask_config.set(prefetch_config):
```

- distributed の `WorkerMemoryManager` はワーカー起動時に config を読むため、クラスタ起動（:1078）後のこの設定は実行中ワーカーの挙動を変えない（インストール済み distributed 2025.5.1 で確認）。
- 仮に効いたとしても通常の順序は target < spill < pause < terminate。pause=0.90 < spill=0.95 で、spill 開始前に pause し、spill(0.95) == terminate(0.95) なのでスピルが間に合わず terminate されうる（値の逆転はコード上確定）。

**修正:** 削除するかクラスタ作成前に反映。残すなら spill < pause < terminate の順序に。

#### M-6 ✅ チャンク書き込みの例外経路で writer スレッドと GDAL ハンドルがリーク

`core/dask_processor.py:497-511, 553-592`

正常系では `write_q.put(None)`（:583）で writer を終了するが、`fut.result()` の例外や `raise write_err["e"]`（:560）では終了センチネルが送られず、`_writer` は `out_band` を掴んだまま `write_q.get()` で永遠にブロックする daemon スレッドとして残る。ライブラリとして同一プロセスで繰り返し実行するとスレッドと GDAL ハンドルが蓄積する。Windows では TemporaryDirectory の cleanup が共有違反で失敗し元の例外をマスクしうる。

**修正:** 例外経路でも `write_q.put(None)` + `write_q.join()` を保証し、残り future を `client.cancel()` する。

#### M-7 ✅ `_ensure_cog_has_overviews` が失敗時に出力ファイルを完全に喪失する

`core/dask_processor.py:263-274`

`dst` を `src`（.no_overviews.tif）にリネーム後、`build_cog_with_overviews` が例外を投げると finally で **src（＝元の出力）も削除**される。オーバービュー再構築に失敗しただけで計算済みの正常な COG 本体が消える。

**修正:** 失敗時は `src.replace(dst)` で復旧してから再送出する。

#### M-8 🔍 大きな CuPy 配列が `map_blocks` の kwargs としてタスクグラフに埋め込まれる

`core/dask_processor.py:1294-1411`、消費側 `algorithms/_impl_topousm_fast.py:291-301`

`_topousm_fast_coarse_field`、`_overview_coarse_dem`、`_fractal_large_fields` 等の具象 CuPy 配列（各〜16MB、visual_saliency は複数枚で計〜64-96MBの可能性）が map_blocks/map_overlap の kwargs 経由でタスクグラフのリテラルとして埋め込まれる。チャンク数が多い大規模ラスタではタスク配送時に繰り返しシリアライズされ、転送量とシリアライズ CPU が跳ね上がる懸念（HLG の層内共有の効き具合は dask バージョン依存 🔍。ストリーミング書き込み経路でチャンク毎に `client.compute(delayed_chunks[i,j])` を呼ぶ際にグラフ片＋リテラルを都度送る点は確定的）。

**修正:** `client.scatter(..., broadcast=True)` した Future を渡すか、1チャンクの dask 配列としてグラフに組み込む。

#### M-9 `_build_zstd_overviews` が `GDAL_NUM_THREADS=ALL_CPUS` を強制しコンテナ方針と矛盾

`core/dask_processor.py:292`

同ファイルの `get_cog_options`（:214-217）では「ALL_CPUS は CFS クォータを無視するため `container_cpu_count()` を使う」と明示しているのに、ここでは ALL_CPUS を強制。cgroup 制限下のコンテナでスロットリングを招く。

**修正:** `str(container_cpu_count())` に統一。

#### M-10 `dask_config.set` によるプロセスグローバル設定の恒久汚染

`core/dask_cluster.py:68-80`

コンテキストマネージャではなくモジュールレベルの `set` のため、クラスタ終了後もホストプロセスの dask グローバル設定（`array.rechunk.method=tasks`、worker memory 閾値）が書き換わったまま残る。ライブラリとして他の dask 処理と同居すると無関係なワークロードにまで適用される。`logging.getLogger('distributed.core').setLevel(WARNING)` も同様。

**修正:** run_pipeline 単位のコンテキストマネージャ化、または LocalCUDACluster/Client 引数で渡す。

### 3.3 io

#### M-11 ✅ `tempfile.mkstemp` のファイルディスクリプタリーク

`io/dem_preprocess.py:955-957`

```python
vrt_path = Path(tempfile.mkstemp(suffix=".mosaic.vrt", dir=str(tmp_parent))[1])
```

`mkstemp` は `(fd, path)` を返すが `os.close(fd)` していない。直前のストリップ生成（:924-925）やシリアルパス（:811-812）では `os.close(fd)` しているのにここだけ抜けている。Windows でハンドル残留の原因。

#### M-12 ストリーミング経路で出力親ディレクトリを作成しない

`io/dem_preprocess.py:315-352`（`_translate_to_cog`）vs `:379`

fast path は `dst_cog.parent.mkdir(parents=True, exist_ok=True)` しているが、ストリーミング経路には mkdir がない。`FUJISHADER_TMP_DIR`/`CPL_TMPDIR`/`TMPDIR` 設定環境（macOS は TMPDIR が常時設定）では、存在しない出力ディレクトリ指定時に**高コストな fill 処理の全部が終わった最後の Translate で失敗**する。

#### M-13 ワーカー GDAL キャッシュの下限 512MB が cgroup 予算を破る

`io/dem_preprocess.py:916-917`

`max(512, min(2048, cache_budget_mb // n_workers))` の下限が cgroup 予算を無効化。例: 利用可能 1GB・8ワーカー → 予算 409MB → 51MB/ワーカーのはずが 512MB×8=4GB を確保。cgroup OOM キルの危険。

#### M-14 ✅ `str.replace(".vrt", ...)` による一時ファイルパス破壊（旧レビュー BUG-1 が未修正）

`io/cog_builder.py:64` と `:509`

`str.replace` は全出現箇所を置換するため、一時ディレクトリ名に `.vrt` を含む場合（例: `out.vrt_tmp/tiles.vrt` → `out_files.txt_tmp/tiles_files.txt`）に存在しない親への open となり失敗する。:64 側は呼び出し元でフォールバックがあるが、**:509（external CLI 経路）にはフォールバックがなくジョブ全体が失敗**する。同ファイル :359-361 では同種の問題を `Path.with_name()` で回避しているのに、こちらは未修正。

**修正:** `Path(vrt_path).with_name(f"{stem}_files.txt")` に統一。

#### M-15 ✅ `or tiled` で COG 適合判定が無意味化

`io/cog_validator.py:79-84`

```python
if 'COG' in layout.upper() or tiled:
    cog_compliant = True
```

タイル化された任意の GeoTIFF（COG でなくても）が「COG-compliant」と判定され、スコアも 90 点に達して戻り値 True になる。呼び出し元（`tile_processor.py:1649, 1716`）は戻り値を無視しているため、検証失敗が処理を止めない点も併せて弱い。

**修正:** `LAYOUT=COG` メタデータの確認に限定し、戻り値を呼び出し側で扱う。

#### M-16 🔍 フォールバックのオーバービュー levels がラスタサイズ非適応

`io/cog_builder.py:104`

`levels = [2, 4, 8, ..., 256]` 固定。最小辺 < 256px のラスタで `BuildOverviews` が CE_Failure を返すと、GDAL API フォールバック経路全体が失敗する（GDAL が過大 level をエラーにするかは環境依存 🔍）。ラスタサイズで levels を切り詰めるべき。

#### M-17 レンジ解決失敗時のバックエンド間不一致（パリティ）

`io/output_encoding.py:124` が `None` を返すケース（slope `unit=percent`、`tv_decomposition` component=structure）で:

- tile: `core/tile_processor.py:1029-1033` → **float32 にフォールバック**（警告のみ）
- dask: `core/dask_processor.py:1454-1459` → パーセンタイル推定し**整数量子化を続行**

同一 CLI 要求で出力 dtype がバックエンド間で変わる。量子化の数式自体（rint/clip/NaN→0）は両者で一致していることを確認済み。

**修正:** 「None 時の推定ポリシー」を `output_encoding` 側で共通化する。

### 3.4 algorithms

#### M-18 hillshade の加重合成フォールバックが他アルゴリズムと不一致

`algorithms/_impl_hillshade.py:88-90, 113-125`

ユーザーが長さの合わない weights を渡すと、`_resolve_spatial_radii_weights` は 2**n プロファイルを返すが `weights is not None` なので元の無効リストが残り、`:113` のチェックで `da.mean` にフォールバックする。slope/curvature/AO/openness/atmospheric は解決済み weights を直接使うため 2**n 加重になる。同一入力でアルゴリズム間の挙動が分かれる。

**修正:** 他と同様に `radii, weights = _resolve_spatial_radii_weights(...)` と直接代入。

#### M-19 ✅ `_radius_to_downsample_factor` がブロック形状依存で、チャンク間・バックエンド間の出力が変わる

`algorithms/_nan_utils.py:546-557`（利用: openness, AO, `_smooth_for_radius` 経由で hillshade/slope/specular/atmospheric/curvature）

```python
block_factor = max(1.0, (block_pixels / 1_000_000.0) ** 0.5)
```

ダウンサンプル係数がブロック画素数の関数になっている。map_overlap ではブロック形状 = チャンク + 2×halo であり、端チャンク（端数）や Dask チャンク（4096²）と tile バックエンドのタイル（2048²+pad）で係数が変わり、**同一 radius でも ds_factor が2倍刻みで変わりうる**。ds_factor が変われば平滑の実効半径が変わるため、(a) 端チャンク境界でシーム、(b) tile↔dask パリティ不一致の両方がありうる。

**修正:** `block_factor` を廃止するか、グローバル定数のチャンクサイズに基づく決定論的な値に統一。少なくとも端チャンク/バックエンド間の差分を計測するテストを。

#### M-20 `_norm_stat_max_scale` が実効ハローを過小見積もり、境界汚染が統計に混入

`algorithms/_norm_stats.py:103-113`

`radii`/`scales`/`kernel_size` のみを見て、未指定時は 16.0。以下が考慮されない:

- openness の `max_distance`（デフォルト 50）、AO の `radius`（同 10）
- scale_drift: 平滑 4σ=128px + LK ウィンドウ ~96px ≈ **224px**
- tv_decomposition: 反復法の情報伝播 ~120px（`tv_scale`/`iterations` は対象キーに無い）
- structure_tensor: σ_i=16 → 4σ=**64px**（デフォルト radii はブロック内部で決まり見えない）
- ガウス系一般: ハローは ~4σ 必要なのに `margin = max_scale` と 4倍不足

p1/p99 のロバスト統計なので致命的ではないが、注入される `global_stats` の系統バイアス源。

**修正:** アルゴリズム毎に実効ハロー（ブロック関数の内部デフォルト込み）を返すフックを `_NORM_STAT_SPECS` に追加。

#### M-21 ✅ hillshade が (H,W,3) スタック法線で VRAM を3倍消費（specular は回避済みのパターン）

`algorithms/_impl_hillshade.py:43-44, 51`

```python
normal = cp.stack([-dz_d_east, -dz_d_north, cp.ones_like(dx)], axis=-1)
normal = normal / cp.linalg.norm(normal, axis=-1, keepdims=True)
```

`_impl_specular.py:42-45` のコメントに「スタック+正規化はピーク VRAM を3倍にし 12k チャンクで RMM プールを枯渇させた」とあり、specular/atmospheric は成分分離形に書き換え済み。hillshade だけが旧パターンのままで、大チャンクで specular より先に OOM しうる。計算結果は同一なので純粋なメモリ効率の問題。

**修正:** specular と同じ成分分離形（`inv_norm` スカラ場）に書き換える。

#### M-22 ✅ openness の物理距離計算でサンプル毎に GPU→ホスト同期が発生

`algorithms/_impl_openness.py:109-111`

```python
phys_dist = max(float(cp.sqrt(phys_dx ** 2 + phys_dy ** 2)), 1e-9)
```

`phys_dx/phys_dy` はホスト側の Python float なのに `cp.sqrt` に渡しており、0-d GPU 配列の生成と `float()` によるデバイス→ホスト同期が (方向数×距離数)（最大 ~160 回/ブロック）発生する。AO 側（`_impl_ambient_occlusion.py:93`）は同じ計算を `np.hypot` でホスト完結させている。

**修正:** `np.hypot(phys_dx, phys_dy)` に置き換える。

#### M-23 phase_congruency / tv_decomposition: 統計未注入時のフォールバックがブロック毎統計でシームが出る

`algorithms/_impl_phase_congruency.py:127-133, 189-197`、`algorithms/_impl_tv_decomposition.py:117-123, 162`

`inject_global_stats` は Zarr 入力で no-op（`_norm_stats.py:245-246`、確認済み）なので、**Zarr 入力ではノイズ閾値 T / tanh スケールがブロック毎に推定されシームが出る**。fractal_anomaly / visual_saliency は `compute_global_stats` で自前推定しており不統一。scale_drift / structure_tensor は統計欠落時に正規化を諦めて生値出力（COG ではストレッチ済み、Zarr では未ストレッチ＝別の非パリティ）。

**修正:** fractal/VS と同じく `compute_global_stats` フォールバックを追加。

#### M-24 frangi: 統計欠落フォールバックがストライド読みで全チャンク読み出しに陥る

`algorithms/_impl_frangi.py:192-199`

```python
sample = gpu_arr[::max(1, gpu_arr.shape[0] // 1024), ::max(1, ...)].compute()
```

このストライド参照は全チャンクをディスクから読み GPU へ転送させる。`_impl_topousm_fast.py:225-234` の docstring がまさにこのパターンを「very large rasters で stall する」として禁止し中央ウィンドウ方式に改めた経緯がある。frangi のフォールバック（Zarr / 統計注入失敗時）が同じ落とし穴に入っている。

#### M-25 tv_decomposition: `component='structure'` でも統計プリパスが走り、重い TV 求解が全て破棄される

`algorithms/_impl_tv_decomposition.py:111-114`、呼び出し側 `_norm_stats.py:258-261`

structure モードでは `global_stats` を一切参照しないのに、`_compute_norm_stats_tiled` は最大9タイル×最大140反復の TV 求解を全て無駄計算する。scale_drift の `drift_output='direction'` も同様に統計未使用で無駄。

**修正:** `_compute_norm_stats_tiled` 呼び出し前に component/output モードでスキップ。

#### M-26 ✅ npr_edges: ヒステリシス処理の8近傍が非対称（反对角の2方向が欠落）

`algorithms/_impl_npr_edges.py:169-176`

上下左右 + (+1,+1)/(-1,-1) 対角のみで、**(+1,-1)/(-1,+1) の反对角が無い**。NMS では4方向バケット全てを扱っているのにヒステリシスの連結性だけ6近傍になっており、反对角に細く繋がった weak edge が strong に昇格しない。直前のコメントで NMS の対角バケット向きをわざわざ修正しているので、こちらは書き漏らしと思われる。

**修正:** 反对角2方向の roll を追加（または `binary_dilation` 3x3 に置き換え）。

#### M-27 tile 直接パス: global_stats 未注入時のフォールバック統計がタイル単位でシームが出る

`algorithms/tile/dask_bridge.py:350-354`（topousm_fast）、`:398-405`（fractal_anomaly）

Dask 側は統計未注入時に全データの中央ウィンドウから推定するのに対し、tile 直接パスは**そのタイル自身の結果**から統計を計算する。`inject_global_stats` が失敗・省略された場合（Zarr 入力、プリパス失敗時）にタイル毎に正規化スケールが変わり、タイル境界の明るさムラが出る。加えて fractal は統計未取得時にブロック計算を normalize=False/True で**タイル毎に2回フル計算**する非効率もある。

**修正:** 統計未注入時は直接パスを使わず警告付きでフォールバック、またはプリパス必須化。少なくともログを出す。

#### M-28 `_direct_npr_edges` のハードコード既定値が正規デフォルトと不一致

`algorithms/tile/dask_bridge.py:367-369`

`threshold_low=0.1, threshold_high=0.3` とハードコードされているが、正規の既定値は `0.2/0.5`（`_impl_npr_edges.py:219-220`）。通常は `_merged_params` が `get_default_params()` を注入するため隠蔽されるが、その `except Exception: pass` でデフォルト取得がサイレント失敗した場合に限り、**Windows タイル直接パスだけが異なる閾値で計算**される。古いデフォルトの取り残し。

### 3.5 cli / config

#### M-29 `DASK_DISTRIBUTED__*` 環境変数がメインプロセスの dask 設定に反映されない

`cli/linux_cli.py:120-126`

dask が環境変数を config に取り込むのは `import dask` 時だけ。`linux_cli` はモジュール import 時点で `..algorithms.dask_registry` → `dask_shared` → `import dask.array` を経由して**既に dask を import 済み**（:25）。`execute()` で env を立てるのはその後なので、メインプロセス（client/scheduler）の `dask.config` には乗らない。nanny 経由で spawn されるワーカーは環境を継承して拾うため**ワーカーにだけ効く**半端な状態になる。`worker.memory.*` の4キーは `dask_cluster.py:68-78` の `dask_config.set` と重複。

**修正:** env 設定を dask import より前に移すか、`make_cluster` の `dask_config.set` にタイムアウト類も統一して env 経路を廃止。

#### M-30 ✅ `RMM_POOL_SIZE` / `RMM_ALLOCATOR` / `RMM_MAXIMUM_POOL_SIZE` が完全なデッドコード

`cli/linux_cli.py:129-135`

パッケージ全体を Grep してもこれらの env を読む箇所は存在しない（書き込みのみ、確認済み）。実際の RMM プールは `core/dask_cluster.py:84-89` が `LocalCUDACluster` に明示 kwarg で渡しており、上限の計算式も違う（linux_cli: プールの1.1倍 vs dask_cluster: `device_limit_gb * 0.97`）。運用者が「env で調整できる」と誤解する恐れがあり、`_rmm_gb` の計算自体も無駄。

**修正:** env 設定ブロックは削除し、プールサイズ調整は `make_cluster` 側の一本化された計算に寄せる。

#### M-31 ✅ `GDAL_SWATH_SIZE` の単位バグ（bytes 指定なのに MB の数値を渡している）

`config/gdal_config.py:50-57`

```python
"VSI_CACHE_SIZE": str(cache_bytes),
"GDAL_SWATH_SIZE": str(cache_mb),   # ← ここだけ MB の生数値
```

`GDAL_SWATH_SIZE` は bytes 解釈（デフォルト 10,000,000 bytes）。ここに `4096`〜`32768`（= 4〜32KB）を設定してしまうため、リモート COG 読み出しのレンジ結合が事実上潰れる。隣の `VSI_CACHE_SIZE`/`CPL_VSIL_CURL_CACHE_SIZE` は正しく `cache_bytes` を使っており、単位の不整合は明らか。

**修正:** `str(cache_bytes)` に修正。

#### M-32 Windows CLI が `show_progress` をパイプラインに渡さない（`--no-progress` が黙って無効）

`cli/base.py:140` で組み立て、`cli/linux_cli.py:157` は `run_pipeline(show_progress=...)` に渡すが、`cli/windows_cli.py:107-126` の `process_dem_tiles(...)` 呼び出しには含まれない。`process_dem_tiles` に同名パラメータは無く、`tile_processor.py` 内に tqdm/show_progress の参照も一切無い。Windows では `--no-progress` が受理されながら完全に無視され、長時間ジョブの進捗表示自体も存在しない。

**修正:** tile パイプラインに進捗コールバックを導入するか、Windows ではヘルプに「無効」と注記。

#### M-33 地理座標系 DEM の「度→メートル」換算式がバックエンド間で不一致（パリティ逸脱）

`cli/linux_cli.py:95-98` は WGS84 精密級数、Windows tile 経路（`core/tile_processor.py:891-892`）と Linux の Zarr 入力・下流自動検出（`io/raster_info.py:42-45`）は簡易式 `111_320.0 * cos(φ)`。緯度35°付近で pixel size が約 0.1〜0.4% ずれるため、メートル指定半径の換算が Linux(GeoTIFF CLI) と Windows/Zarr で微妙に異なる。

**修正:** 換算を `io/raster_info.py` の1関数に統一。

### 3.6 tests / tools

#### M-34 `tools/debug_fractal_anomaly.py` が本番ロジックの古い複製で、despeckle の適用ドメインが本番と乖離

`tools/debug_fractal_anomaly.py:76-147` vs `algorithms/_impl_fractal_anomaly.py:109-197`

本番は despeckle を**正規化後の値**に適用するのに対し、debug ツールは生の feature 値に適用し正規化自体をしない。しきい値 0.35 の意味が両者で異なるため、診断出力（D_out 統計、noise_ratio）は実パイプラインの挙動を再現していない。定数も全てハードコードで複製され、本番変更に追随しない。

#### M-35 `tools/install_gdal.py` が sudo 不在の非 root 環境で無言のまま権限失敗に進む

`tools/install_gdal.py:86-90, 175-176`

`return ["sudo"] if shutil.which("sudo") else []` — 非 root かつ sudo 無しでは空リストを返し、`apt-get` が権限エラーで失敗するが戻り値を一切チェックせず次に進む。最終的には "[FAILED]" に辿り着くが、原因が sudo 欠落なのか区別できない。

#### M-36 `tests/test_zarr_io.py` のラウンドトリップテストが値を一切検証していない

`tests/test_zarr_io.py:35-37`

書き出された zarr に変数が「どちらかの名前で存在する」ことしか検証しておらず、値・形状・dtype の一致を見ていない。Zarr I/O 専用テストがこれ1本しかないことを考えると実質的にスモーク以下。

---

## 4. Low（簡潔記載）

### 4.1 core/tile

- **L-1** `--mode local` フォールバックが半径自動注入により事実上デッドコード化（`tile_processor.py:1383-1397` vs `:1091-1106`）。「ユーザ指定かどうか」のフラグを自動注入前に保存すべき。
- **L-2** `spatial_algorithms` セットの不一致: `_required_padding_for_algorithm`（:340）には `"npr_edges"` があるが sanitize ゲート（:1133）には無い。セットを単一定数に集約すべき。
- **L-3** タイル毎の不要なホストコピー `dem_tile.astype(np.float32, copy=True)`（:854。reader は既に float32 の新規配列を返す）と二重 `np.isnan`（:829-830）。
- **L-4** topousm_fast 重複半径時にユーザ weights が静かに均等配分へ捨てられる（`tile_compute.py:56-76`）。sanitize 側（`tile_processor.py:142-155`）と挙動が違う点も不一致。
- **L-5** `tile_size<=0` / 負の padding の入力バリデーション欠如（:1533, :1560-1565）。
- **L-6** `resume_cog_generation`（:1722）が `rm -rf {tmp_tile_dir}` をログ提案: Windows で通用しない＋スペース入りパスをクォートしていない。
- **L-7** VRAM 推定のマジックナンバー `15.0` が `tile_processor.py:1298,1313` と `auto_tune.py:171` に重複定義。
- **L-8** `tile_io.py` のスレッドローカル reader が終了時に明示 close されない（executor スレッド終了＋GC 頼み）。
- **L-9** `process_single_tile` の `topousm_fast_weights` は呼出し側で常に `None` のデッド引数（`tile_processor.py:1112`）。
- **L-10** `apply_nodata_mask`（`tile_compute.py:125-136`）は halo 込み全面に fill してからコアを切り出す。コア切り出し後にコア内マスクへ適用すれば halo 分の無駄が省ける。

### 4.2 core/dask

- **L-11** `_select_chunk_temp_parent` のデッドな 0.75 項（`dask_processor.py:134`: `max(data_nbytes, int(data_nbytes * 0.75))` は常に前者）。意図は恐らく ×1.75（中間＋最終の二重占有）。
- **L-12** `_write_cog_da_original` の broad except（:374-376）が compute 自体の決定的失敗（GPU OOM 等）でも捕捉し、`_fallback_cog_write` で全量再計算→同じ所で再失敗。compute 段と書き込み段の例外を分離すべき。
- **L-13** `TqdmCallback`（:328-342）と `ProgressBar`（:790-791, `dask_io.py:69`）はローカルスケジューラ専用で distributed 下では進捗が出ない。`distributed.progress`（:1516 で既に使用中）に統一すべき。
- **L-14** pixel_size の no-op 再代入（:1251-1252）。
- **L-15** `agg='stack'` が Dask 側でも終盤の xarray ラップ（:1615）で不可解な ValueError になる。パイプライン冒頭で検証すべき。
- **L-16** `_build_cog_via_cog_driver` の tqdm リーク（:853-871。失敗経路で `pbar.close()` されない）。
- **L-17** クライアントプロセスでの CUDA コンテキスト初期化（`dask_cluster.py:27-31` の `memGetInfo`）が VRAM 見積もりに未計上。fork 開始方式では親のコンテキストがワーカー初期化を壊しうる 🔍（distributed デフォルトは spawn で無害）。pynvml 等コンテキストを作らない計測に。
- **L-18** 極小 device_limit 時に `rmm_max_gb > device_limit_gb` となりうる（`dask_cluster.py:55`）。
- **L-19** `GPUtil.getGPUs()` が未ガード（`dask_cluster.py:24-25`）。マルチ GPU は `gpus[0]` 固定で sizing。
- **L-20** `cluster._fujishader_mem` による私信属性の埋め込み（`dask_cluster.py:110-117`）。dask 側の実装変更で静かに効かなくなる脆さ。
- **L-21** `dask_io.py:45-56`: マルチバンド／1ピクセル幅入力が `.squeeze()` をすり抜け 3D/1D のまま下流へ流れ、分かりにくいエラーで失敗。ロード直後に `ndim == 2` を検証すべき。
- **L-22** `dask_io.py:52` の `lock=False` はシリアルフォールバック経路で GDAL ハンドル競合の余地 🔍。
- **L-23** Zarr 入力では CRS/ジオトランスフォームが保持されず COG 出力が無座標になりうる（`dask_io.py:33-43` + `dask_processor.py:480-488`）。

### 4.3 io

- **L-24** `_apply_band_fill` の冗長な `.astype(np.float32)` コピー（`dem_preprocess.py:762, 774, 779, 782`）。
- **L-25** NoData 検出用と平均用で粗グリッドを2回読む（`dem_preprocess.py:513-516` と `:547-549`）。
- **L-26** `_nan_aware_coarse_average` が k=1 で「NaN 無視平均」を喪失し単一ピクセルの NEAREST に退化（:149-166）。docstring と動作が乖離。
- **L-27** `_sample_coarse` の float64 一時配列3枚（約32B/px）と `_band_height` の `max(64, ...)` 下限による超広幅ラスタでの予算超過（:714-717, :270-279）。
- **L-28** 並列タスクが `surface`/`exterior_coarse`（計〜20MB）を毎回 pickle（:927-931）。initializer で一度だけ渡す設計に。
- **L-29** 無効な `nodata_override` の黙殺（:509）と `max(n_workers, n_workers * STRIPS_PER_WORKER)` のデッドコード（:907）。
- **L-30** 'all' モードで残余 NaN を無言で 0.0 置換（:776-779）。
- **L-31** GPU fill 失敗の broad except が info レベルで CPU フォールバックに流す（:111-112）。warning 以上で記録すべき。
- **L-32** `_ensure_output_nodata` の3点問題（`cog_builder.py:169-195`）: (a) 値の一致を見ない、(b) `IGNORE_COG_LAYOUT_BREAK=YES` 付き GA_Update で COG レイアウト保証を自ら壊す（`tile_processor.py:1646-1648` の方針コメントと矛盾）、(c) `SetNoDataValue` の戻り値を無視。
- **L-33** `_configure_gdal_ultra_performance` の GDAL_CACHEMAX がキャッシュ初回作成後では効かない可能性 🔍（`cog_builder.py:587` → `gdal_config.py:39-74`）。`gdal.SetCacheMax` を使うべき。
- **L-34** COG translate 失敗時に中途半端な出力ファイルが残る（`cog_builder.py:328-339, 560`）。
- **L-35** CLI 経路と Python API 経路で VRT オプション不一致（`cog_builder.py:70-82` vs `:214-222`: `-allow_projection_difference` の有無）。
- **L-36** `_detect_nodata_from_tiles` が先頭タイルの値を無検証で採用（`cog_builder.py:44-55`）。
- **L-37** `cog_validator.py:64`: オーバービュー幅0での ZeroDivisionError（外側の except で全体 False になる）。
- **L-38** `output_encoding.py:182-185`: ±inf が NoData にならず DN 最大値にクリップされる。`~np.isfinite(a)` に変えるのが安全。
- **L-39** `output_encoding.py:100-103`: `hi <= lo` の override が警告・エラーなく無視される。
- **L-40** `raster_info.py:73-75`: 失敗時の固定フォールバック 0.5m。radius 系パラメータの基礎になるため、検出失敗時に処理全体が誤スケールで静かに走る。
- **L-41** `raster_info.py:51-54`: CRS なし緯度経度データの誤判定に歯止めがない（度単位ピクセルがそのまま「メートル」として返る）。

### 4.4 algorithms

- **L-42** `robust_unsigned_stretch_stat_func` が `_global_stats.py:133-155` と `_normalization.py:44-57` に重複定義され縮退時の挙動が乖離（後者はデッドコードだが `__all__` で公開中）。一元化すべき。
- **L-43** tile バックエンドでタイル >2048px だと `coarsen_factor_for_shape` が F>1 を返し、overview 注入失敗時のフォールバックで `da.coarsen` がタイルローカルに走りシームが出うる（`_nan_utils.py:445-452`）。
- **L-44** `_combine_multiscale_dask` が負の重みをサニタイズしない（`_nan_utils.py:191-198`）。同ファイルの `_clean_normalized_weights` は非正値を0に丸めておりポリシー不統一。
- **L-45** `_resolve_spatial_radii_weights` が numpy 配列の weights を拒否し静かに自動プロファイルに差し替え（`_nan_utils.py:113, 120`）。
- **L-46** `_hybrid_combine_wrapper` 系の `int(round(s))` 辞書キーでスケール衝突の可能性（`_nan_utils.py:716-717, 766-769, 780`。banker's rounding で `round(2.5)==2`）。
- **L-47** 勾配の NaN 充填にグローバル `nanmean` を使用しデータ境界で人工的な崖（`_nan_utils.py:57`、`_impl_curvature.py:24`、`common/kernels.py:26, 92`）。`_downsample_nan_aware` では改良済みなのに勾配系は旧式のまま。
- **L-48** `_upsample_to_shape` / `pushpull_fill` の zoom 出力形状を切り詰めのみで処理（アンダーシュート未防御）🔍（`_nan_utils.py:643-655`、`_pyramid_fill.py:58, 73`）。
- **L-49** norm-stats ループ内の `cp.get_default_memory_pool().free_all_blocks()` は RMM 使用時に無効（`_norm_stats.py:207`、`_impl_specular.py:298`）。
- **L-50** `read_overview_coarse_dem` の返却デシメーションが実デシメーションと丸め分だけ乖離（`_nan_utils.py:804-820`。~0.1% 級のスケール誤差）。
- **L-51** `auto_spatial_profile` がユーザー radii を無検証で通す（`common/spatial_mode.py:97-101`。0・負値・重複がそのまま返る）。
- **L-52** `compute_global_stats` は中央窓のみで、オフセンターフットプリントで統計が偏る（`_global_stats.py:54-61`）。fractal_anomaly / visual_saliency / scale_space_surprise が依然使用。
- **L-53** openness で NoData 方向のレイが `dir_valid` にカウントされ境界が白く引っ張られる 🔍（`_impl_openness.py:84-87, 114-125`。pad_value=±1e6 が valid 扱い）。
- **L-54** `multi_light_uncertainty` で `azimuths` が空だと `cp.stack([])` で例外（`common/kernels.py:109-117`）。明示的ガードを。
- **L-55** scale_drift stats プリパス（direction）のコメントと実装の不一致＋参照実装のスケール clamp 不整合（`_impl_scale_drift.py:168-180`）。
- **L-56** npr_edges 勾配統計の対象半径上限（600）と coarse 切替閾値（≥256）の不整合、および `r_coarse` のキー衝突（`_impl_npr_edges.py:279, 206`、`_nan_utils.py:333`。例: radii=[40,320], fac=8 → r_coarse=40 で半径40の統計が半径320相当の応答に適用される）。
- **L-57** phase_congruency: Riesz 伝達関数の complex128 昇格の可能性 🔍（`_impl_phase_congruency.py:93-94`。cupy バージョン依存、complex128 だと FFT が倍精度で2倍のメモリ・演算）。`cp.complex64` に明示キャストを。FFT 周期境界の ringing や小チャンク時の波長未再クランプも同ファイルの既知の近似。
- **L-58** structure_tensor / frangi の参照ブロック実装で radii が非昇順のとき weights の対応が崩れる（`_impl_structure_tensor.py:141-145`、`_impl_frangi.py:99, 117-120`）。
- **L-59** lic: `flow_sigma` 未クランプでハロー予算超過の可能性＋積分ループのアロケーションチャーン（`_impl_lic.py:130, 78-91`。座標バッファの事前確保と `output=` 再利用で削減可能）。
- **L-60** グローバル統計ヘルパーまわりの重複読み出し（`_compute_fractal_relief_stats` と `_compute_norm_stats_tiled` が同一 stratified ウィンドウを別々に読み直す）と、ホスト計算への不要な GPU 利用（`_impl_experimental.py:196` の `cp.ceil(...).item()`、`_impl_fractal_anomaly.py:347-349` の `cp.linspace(...).get()`）。
- **L-61** 全 NaN チャンクで `cp.nanmean` が RuntimeWarning を吐く箇所のガード不統一（`_impl_experimental.py:68`、`common/kernels.py:26, 92`。VS/structure_tensor はガード済み）。
- **L-62** `_impl_topousm_fast.py:239`: 統計サンプル窓が max_radius > 1024 で footprint を包含できない（docstring と不整合、p99 ロバスト統計なので実害小）。
- **L-63** ミュータブルなデフォルト引数（`_impl_topousm_fast.py:51`、`_impl_fractal_anomaly.py:81`、`_impl_visual_saliency.py:80`。現状破壊的変更なし、実害なし）。
- **L-64** `_impl_visual_saliency.py:141,160,205,223`: `float(wvec[...])` を map 毎に呼び GPU→ホスト同期（回数は軽微）。
- **L-65** fractal_anomaly は Zarr 入力で `relief_p10/p75` が注入されず per-block パーセンタイルに落ちシームが出る（`_impl_fractal_anomaly.py:285-287`。docstring に既知と明記済み）。
- **L-66** `_direct_*` 内の spatial 分岐は全てデッドコード（`dask_bridge.py` の約100行。`_process_direct:426-432` が事前フォールバックするため到達不能）。削除するか到達不能の旨を明記。
- **L-67** フォールバック内の dask import エラーメッセージが到達不能（`dask_bridge.py:484-490`。tile アダプタは先に `dask_shared` を import するため）。
- **L-68** `_process_direct` のコメントがコードと矛盾（`dask_bridge.py:421-425`。fractal の spatial は直接パスを持たない）。
- **L-69** `tile_shared.py` の docstring 陳腐化＋重量連鎖 import（:5-8。re-export の2クラスは現在 Dask アダプタ）。
- **L-70** `algorithms/__init__.py` の tile 側公開リストが不完全（:24-57。2026-07 追加の6クラスが未 export）。
- **L-71** バックエンド間の hybrid overview 注入ゲート差異 🔍（Dask: `dask_processor.py:1321-1326` はモード非依存、tile: `tile_processor.py:1471-1476` は `mode == "spatial"` 必須。local モード＋大半径で両バックエンドの数値経路が異なる）。

### 4.5 cli / config / utils

- **L-72** Windows / macOS 分岐で ImportError が未捕捉（`__main__.py:35-44`。Linux 分岐だけ捕捉しており、cupy/osgeo 未導入だと素のトレースバック）。
- **L-73** CRS 未設定のラスタが「Projected CRS」と誤表示され単位不明のまま処理が進む（`linux_cli.py:107-116`）。
- **L-74** `--cog-only` 時の `input_path` 付け替えがデッドコード（`windows_cli.py:102-103`）。
- **L-75** 例外の三重報告（`linux_cli.py:166-169` で error+traceback+raise、`__main__.py:55-57` で print。`windows_cli.py:129-131` も同様の二重化）。
- **L-76** `--nodata` / `--output-range` の不正値が `parser.error` にならず生 ValueError で落ちる（`cli/args.py:212, 228`）。
- **L-77** ハードウェア検出（CuPy プローブ+cgroup 読み出し）の二重実行（`config/system_config.py:148`。`--cog-only` でも必ず呼ばれる）。
- **L-78** `get_gpu_config` の vram_gb 二重読み出し（デフォルト 0.0/8.0 の不整合）と GPU 未検出時の誤ラベル（`system_config.py:33,38`、`gpu_config_manager.py:60-73`）。
- **L-79** `GPUConfigManager` がモジュール import 時に YAML ファイル I/O（`gpu_config_manager.py:76`）。初回使用まで遅延する方が安全。
- **L-80** `resolve_tmp_dir` が env 指定パスの検証をしない（`utils/paths.py:42`。env が既存ファイルを指す場合に素の FileExistsError）。
- **L-81** `cli/base.py:122` の `os.path.exists` がリモート入力（`https://`、`/vsicurl/`、`s3://...zarr`）を一律拒否 🔍（`gdal_config.py:60-62` の HTTP チューニングと矛盾する可能性）。
- **L-82** `container_cpu_count` の `round()` が銀行丸め（`utils/cpu.py:103`。`math.ceil` 系が自然）。

### 4.6 tests / tools

- **L-83** 乱数シード未固定のテストが複数（`test_fractal_anomaly_normalization.py:47`、`test_topousm_fast_normalization.py:16`、`test_algorithm_smoke.py:21,37`、`test_local_spatial_modes.py:22`）。
- **L-84** `install_gdal.py:48`: `_verify()` が空出力で IndexError。
- **L-85** `install_gdal.py:79-83`: `_in_conda()` が CONDA_PREFIX のみでも真となり、install 先と verify 先が食い違いうる 🔍。
- **L-86** `install_gdal.py`: dry-run でも `_verify()` は実際に subprocess を実行（ヘルプ文と矛盾、副作用はなし）。
- **L-87** `test_cog_overviews.py:24,41`: ZSTD アサーションが GDAL ビルド依存の可能性 🔍。
- **L-88** `test_zarr_io.py:30`: `zarr_format=2` kwarg が xarray バージョン依存 🔍（pyproject の下限 `xarray>=2024.1` で使えるか要確認）。
- **L-89** `tools/debug_fractal_anomaly.py:70`: NoData 判定に `np.isclose`（rtol=1e-05）を使用し、nodata=-9999 では -9999±0.1 の実データも NaN 化。
- **L-90** `test_registry_cli_sync.py:40`: argparse のプライベート属性 `_actions` に依存。
- **L-91** `pyproject.toml:163-168`: addopts の `--cov` 指定で pytest-cov 未インストール時に pytest が起動不能。
- **L-92** `pyproject.toml:32`: `Operating System :: OS Independent` は実態不整合（cupy-cuda12x 必須）。`license = {text = "MIT"}` は setuptools>=77 で非推奨警告。

---

## 5. 検証の結果「誤報」と判明した指摘

### multi_light_uncertainty の符号付きピクセルスケール（High 候補として上がったがバグではない）

`algorithms/common/kernels.py:94-100` は `pixel_scale_y` を abs 化せず `cp.gradient` に渡しており、一見すると hillshade（`_impl_hillshade.py:35-42` が abs+sign 補正）と照明方向が逆転するように見える。実際に数式を展開すると:

- 符号付き spacing の `cp.gradient(f, step_y)` は「ジオトランスフォームの実座標方向の微分」を直接返す（北向上ラスタで step_y<0 なら dz/dnorth が得られる）。
- hillshade の abs+sign 補正も同じ物理微分を再構成する。
- 4通りの符号組合せ（pixel_scale_x/y の正負）全てで両者の法線ベクトルは**数学的に一致**する。

したがって照明の南北逆転は発生しない。ただし「符号付き spacing に依存するカーネル」と「abs+sign 補正するカーネル」が混在しているのは保守上の罠（将来の変更で片方だけ直すと壊れる）なので、規約の統一は Low として検討の価値あり。

---

## 6. 優先度の提案

1. **まず直すべき（出力の正しさに直結）:** H-1（topousm_fast パリティ）、H-5（agg=stack 破壊）、H-3（scale_drift NaN 浸食）、M-1（NaN 漏洩）、M-31（GDAL_SWATH_SIZE 単位）
2. **OOM・運用安定性:** H-2（チャンク縮小不発）、H-4（structure_tensor 2倍計算）、M-5（prefetch 設定）、M-13（キャッシュ下限）、M-21（hillshade VRAM）
3. **エラーパス:** M-6（writer リーク）、M-7（出力喪失）、M-11（fd リーク）、M-12（mkdir 欠落）
4. **テスト基盤:** H-6（importorskip/sys.path 統一）— CI を回せる状態にするのが他の修正の前提
5. **残り:** Medium/Low はパリティ関連（M-17, M-19, M-33, L-71）を優先し、デッドコード類はまとめて一掃するのが効率的

---

*本レポートは静的解析に基づく。🔍 の項目は実環境での挙動確認を推奨。GPU 必須のため本監査では実行テストは行っていない。*
