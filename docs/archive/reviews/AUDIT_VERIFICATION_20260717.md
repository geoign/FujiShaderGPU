# Kimi 3 監査レポートの独立検証と修正計画

**検証日:** 2026-07-17
**対象:** `CODE_AUDIT_20260717.md`(Kimi 3 による監査、High 6 / Medium 36 / Low 92 = 計134項目)
**方法:** 全項目を原典のコードで再検証した。High 5件(H-1〜H-5)は主検証者が直接ソースを追跡し、残る129項目はモジュール別の8系統の独立検証で確認した。「書き込みのみで読み手なし」型の主張は全パッケージのgrepで裏を取り、CuPyの型昇格やスレッド間のアロケータ共有など静的読解で決まらない主張は、実環境(Python 3.13 + CuPy 13.4.1 + dask/distributed 2025.5.1 + rioxarray 0.19.0)での実行で確認した。

---

## 1. 検証の総括

結論から言うと、この監査は採用に値する。134項目のうち完全な誤報は3件(L-22、L-57、L-62)にとどまり、High 6件と Medium 36件は M-16 を除く全件が事実として確認できた。引用行番号は全編を通じてほぼ正確で、「デッド変数」「順序逆転」「契約不一致」のような静的読解で完結する指摘は精度が高い。

誤りには傾向がある。3件の誤報はいずれも、実行や演算子単位の確認を要する主張だった。L-57 は CuPy のスカラ型昇格(complex64 に留まることを実行で確認)、L-62 は `max` を `min` と読み違えた式の解釈、L-22 は rioxarray の `lock=False` 実装(スレッドローカルなハンドル管理で安全)である。静的解析には見えない層で外した、という言い方ができる。逆に言えば、レポート自身が 🔍 を付けて実環境確認を促した箇所の自己評価は妥当だった。

一方で、検証の過程でレポートに載っていない問題が7件見つかった(§4)。うち N-2(地理座標系DEMの等方ピクセルスケール上書き)は、レポートが M-33 で指摘した換算式の 0.1〜0.4% のずれより一桁以上大きい、緯度35°で軸あたり約±10% のバックエンド間相違であり、修正計画ではこちらを主対象に据える。

なお、レポート冒頭の件数サマリ(Medium 26件 / Low 40件強)は実記載(36件 / 92件)と一致していない。実体の検証には影響しないが、レポートの自己集計は信用しないこと。

**判定の集計:**

| 区分 | 項目数 | CONFIRMED | PARTIAL(要訂正) | REFUTED |
|---|---|---|---|---|
| High | 6 | 6 | 0 | 0 |
| Medium | 36 | 34 | 2 (M-16, M-19) | 0 |
| Low | 92 | 76 | 13 | 3 (L-22, L-57, L-62) |
| §5 誤報判定 | 1 | 正しい(数学的に検証) | — | — |

---

## 2. 誤報と判定した指摘(採用しない)

- **L-22(`dask_io.py` の `lock=False` によるGDALハンドル競合)** — rioxarray 0.19.0 の `lock=False` は `URIManager` 経由でスレッドごとに独立した rasterio ハンドルを持つ(`rioxarray/_io.py:1119-1134`、`:256-263`)。docstring もこれを完全並列安全モードと明記しており、シリアルフォールバック経路に共有ハンドル競合は存在しない。
- **L-57(phase_congruency の complex128 昇格)** — `_impl_phase_congruency.py:93-94` の `1j * float32配列` は CuPy 13.4.1 の実行で complex64 になることを確認した。NEP-50 でも旧仕様でも complex128 には昇格しない。明示キャストは害のない注記にしかならない。付随して挙げられた FFT リンギングと小チャンク時の波長未再クランプは実在するが、レポート自身が既知の近似と認めている範囲であり、独立項目としない。
- **L-62(topousm_fast 統計窓が max_radius > 1024 で footprint を包含できない)** — 引用箇所 `_impl_topousm_fast.py:239` は `max(4096, max_radius * 4)` で、大半径ほど窓が「広がる」。`max` を `min` と読み違えたと思われる。ただし同型の実在する問題が別の場所にある(§4 の N-3)。
- **L-5 の後半(負の padding)** — 負の `--padding` は `tile_processor.py:1260-1265` の自動拡張(`padding < required_padding` で必ず ≥32 に補正)を通るため、窓計算に到達しない。`tile_size <= 0` の未検証(ZeroDivisionError)の方は実在し、そちらだけ採用する。

なお、レポート §5 が「誤報」と自己判定した multi_light_uncertainty の符号付きピクセルスケールは、`cp.gradient(f, -h) == -cp.gradient(f, h)`(中心差分・edge_order=2 の片側差分とも 1/h に線形)の恒等式から hillshade の abs+sign 補正と数学的に一致することを式展開で確認した。誤報判定は正しい。符号規約の混在を Low で統一検討という付記も妥当である。

---

## 3. 訂正付きで採用する指摘

内容の核は正しいが、機序・範囲・重大度のいずれかに訂正が要るもの。修正計画では訂正後の内容を基準にする。

| ID | 訂正内容 |
|---|---|
| M-4 | 実測で確認(プール=プロセス共有、`set_allocator`=プロセスグローバル、Windows では rmm import 不可)。混在は「スレッド別」ではなく「アロケータ切替の前後」で起きる。実害は rmm 導入済み Linux で tile バックエンドを使う場合に限られる |
| M-6 | writer スレッドのリークと future 未キャンセルは確定。ただし GDAL ハンドルは finally のクロージャ null 化で解放される。残る実害は「書き込み中のデータセット破棄」の競合窓(Windows で元例外をマスクしうる) |
| M-16 | Medium から **Low へ降格**。3段フォールバックの最深部でのみ発動し、GDAL が過大 level を実際にエラーにするかは未確認。levels のサイズ適応クランプ自体は1行なので修正はする |
| M-19 | ブロック形状依存の係数変動(端チャンク、tile 2048² vs Dask 4096²)は実在するが、全呼び出し点が sigma/距離/pixel_size を係数で補正するため「実効半径が変わる」は誤り。変わるのは近似の忠実度(エイリアシング、量子化、アップサンプルぼけ)で、シームの可視性は実測が要る |
| M-29 | 「ワーカーにだけ効く」はおそらく楽観的すぎる。distributed 2025.5.1 の nanny はワーカー側で `os.environ.update → config.refresh → dask.config.set(親のconfig)` の順に適用するため、親プロセスの陳腐化した config が env 由来の値を上書きし、env ブロックは全域で不活性の可能性が高い(バージョン依存、要実測) |
| M-31 | 単位バグ(bytes 指定に MB 値)は確定。ただし `GDAL_SWATH_SIZE` が効くのは `CopyWholeRaster`(COG 生成)のコピースワスであり、リモート COG のレンジ結合ではない(それは `CPL_VSIL_CURL_CHUNK_SIZE` の領分)。GDAL 内部の下限 ~1MB があるため実害は「意図した GB 級スワスが 1MB に潰れる」性能問題 |
| M-32 | `--no-progress` の無言無視は確定。ただし「進捗表示自体も存在しない」は誤りで、`tile_processor.py:1618-1622` が10タイルごとに進捗をログしている |
| M-33 | 換算式の相違(35°で0.1〜0.4%)は確認。ただし支配的なのはレポート未指摘の等方上書き(§4 N-2)であり、修正はそちらと一体で行う |
| H-6 | 件数は正確(未ガード4、bootstrap有9/無8、conftest なし、pythonpath なし)。「他の13はガード済み」は正しくは「11がimportorskip + 2はガード不要(GPU import なし)」。「dev extra に cupy がない」は字義通り真だが、cupy はコア依存なので `pip install .[dev]` では入る。クラッシュ条件は未インストールのリポジトリ直 pytest か CPU-only 環境 |
| L-17 | クライアントの CUDA コンテキストは「VRAM 見積もり未計上」ではない(コンテキスト生成後に free を読むので織り込み済み)。クライアントは他所でも CuPy を使うため pynvml 化の効果もない。実質却下に近いが、fork 懸念の注記のみ残す |
| L-19 | GPUtil は内部で Popen 失敗を握って `[]` を返し、空リストは処理済み。残るのはマルチ GPU で `gpus[0]` 固定の sizing のみ |
| L-23 | ジオトランスフォームは x/y 座標経由で通常保存され、自パッケージ産の Zarr は `spatial_ref` 座標で CRS も復元されうる。実害は外部産 Zarr に限定 |
| L-29 | (a) 非有限 nodata_override の黙殺は挙動として無害(NaN/inf は無条件マスク済み)でログ欠落のみ。(b) のデッドコードは確定 |
| L-37 | ovr幅0は GDAL が overview を ≥1px にクランプするため破損ファイルでのみ発生。外側 except で吸収済み。理論上の欠陥として最低優先 |
| L-40 | 「静かに」は不正確(warning は出る)。0.5m 固定で走り続ける実害は変わらず採用 |
| L-49 | RMM 下で no-op になるのは「rmm 導入済み Linux で tile バックエンドを使うプロセス」のみ。Dask 経路のクライアント側 `free_all_blocks` は有効 |
| L-51 | 無検証分岐は実在するが、`auto_spatial_profile` に radii を渡す呼び出し元が存在せず現状デッドブランチ。実ユーザ radii は `_normalize_spatial_radii` で検証済み |
| L-52 | 中央窓バイアスは実在するが、3アルゴリズムとも主経路は stratified 統計で、`compute_global_stats` は Zarr 入力・プリパス失敗時のフォールバックのみ |
| L-61 | ガード不統一は実在。ただし CuPy の all-NaN `nanmean` は RuntimeWarning を出さない(実測)ので症状の前提が崩れる。整理は cosmetic |
| L-65 | 現象は確定。ただし「docstring に既知と明記済み」は確認できず(該当 docstring なし) |
| L-66 | スパシャル分岐のデッドコード化は概ね確定(約75〜80行)。ただし npr_edges の spatial チェックは事前フォールバック集合に含まれないため現役、hillshade の combine ブロックも local マルチスケール経由で到達可能。「全て」は過大 |
| L-71 | ゲート非対称は実在するが、指摘された「local モード+大半径」は両バックエンドとも到達不能(強制 spatial 変換 + LOCAL_RADII=[1])。実在する分岐は地理座標系 DEM の扱い(§4 N-5) |
| L-87 | ZSTD アサーションの行は 41 のみ(24 は COG ドライバの skipif)。ビルド依存の本質は妥当 |
| L-91 | `pytest-cov` は dev extra に宣言済み(pyproject.toml:82)。刺さるのは `.[dev]` なしで pytest を打つ環境のみ(H-6 と同じシナリオなので無価値ではない) |

---

## 4. 検証で新たに判明した問題(レポート未記載)

検証の副産物として、以下を修正対象に追加する。

- **N-1: 本番の relief 統計プリパスにも `np.isclose` の rtol 既定値バグ** — `_impl_fractal_anomaly.py:403` は `atol=1e-6` を指定しつつ `rtol=1e-5` が既定のまま残り、nodata=-9999 で -9999±0.1 の実データも NoData 化する。レポートは同じ欠陥をデバッグツール(L-89)にのみ帰していた。
- **N-2: 地理座標系 GeoTIFF での等方ピクセルスケール上書き** — `dask_processor.py:1168-1183` は linux_cli が自動検出した精密換算値を「明示指定」として扱い、異方性(緯度35°で dx と dy は約10%異なる)を捨てて等方適用する。Windows tile 経路(`tile_processor.py:897-901`)と Zarr 経路は符号付き軸別スケールを保つため、これが度→m換算まわりの支配的なバックエンド間相違になる。M-33 の換算式統一と一体で修正する。
- **N-3: norm-stats 窓が半径 >~1024 の footprint を包含できない** — `_norm_stats.py:163-165` の `margin = min(max_scale, 1024)`、`tile = min(4096, ...)` により、L-62 が誤った場所に帰した問題がここには実在する。p1/p99 のロバスト統計なので実害は小さいが、M-20 の実効ハロー修正と同時に扱う。
- **N-4: `DASK_DISTRIBUTED__*` env はワーカーにも効いていない可能性** — §3 の M-29 訂正参照。nanny が親 config を最後に適用するため。M-29 の修正方針(env 経路の廃止)を後押しする材料。
- **N-5: hybrid overview 注入の実在する非対称は地理座標系ゲート** — tile 側(`tile_processor.py:1475`)は `not is_geographic_dem` を要求し、Dask 側(`dask_processor.py:1321-1326`)には地理条件がない。地理座標系 DEM + spatial + 大半径で、Dask は overview 経路、tile は打ち切りハロー経路と数値経路が分かれる。さらに L-43 の per-tile coarsen シームも、overview が注入されない地理座標系 DEM の tile 実行で常時到達可能になる。
- **N-6: ±inf の DN 化は Dask 側量子化にも存在し、-inf は DN 最小値になる** — L-38 の欠陥は `output_encoding.py:182-185` と `dask_processor.py:983-986` の両方にあり、修正は両側同時でなければ新たなパリティ違反を作る。
- **N-7: M-12 の発火条件は実質「Windows 全環境」** — `resolve_tmp_dir` は `TMP`/`TEMP` も参照する(`utils/paths.py:17-23`)ため、Windows では常に staging が出力ディレクトリと分離され、出力親ディレクトリ欠如が最終 Translate まで潜伏する。

---

## 5. 妥当な指摘の一覧

以下が「採用する指摘」の全量である。詳細な現象説明は原典 `CODE_AUDIT_20260717.md` を正とし、ここでは判定と重大度調整のみを重ねる。§3 に訂正があるものは訂正後の内容で読む。

### High(6件、全件採用)

| ID | 要旨 | 判定 |
|---|---|---|
| H-1 | topousm_fast: tile 側の global_stats が半径分割後の小半径で計算され Dask と不一致。`_topousm_fast_full_radii/_weights` はデッド変数 | 確定(直接検証) |
| H-2 | マルチ半径チャンク縮小が auto radii(デフォルト経路)で不発 | 確定(直接検証)。CLI 既定は spatial+auto なので主経路が無防備 |
| H-3 | scale_drift: coarse 経路の NaN が有効画素へ最大 ~96px 浸食 | 確定(直接検証)。visual_saliency の再充填パターン未適用は正確 |
| H-4 | structure_tensor: u/v 2パスで全 GPU 計算が2倍、coarse persist も二重 | 確定(直接検証) |
| H-5 | `--agg stack` の (C,H,W)↔HxWxC 契約不一致で tile 出力破壊 | 確定(直接検証)。Dask 側も終盤 ValueError(L-15)で、実質両バックエンド不良 |
| H-6 | テストの環境ガード・sys.path 不統一でコレクション全体が死ぬ | 確定(件数正確、§3 の注記あり) |

### Medium(35件採用、M-16 は Low へ降格)

| ID | 要旨 | 判定 |
|---|---|---|
| M-1 | 数値 NoData と NaN 混在で元 NaN 画素が有効値として漏洩(Dask とパリティ違反) | 確定。漏洩連鎖を dask_bridge 経由で末端まで確認 |
| M-2 | 正常終了時にタイル一時ディレクトリ未削除・無通知 | 確定。`--keep-tiles` 相当も存在せず |
| M-3 | 縁からの nodata=0 推定がフル解像度読み+海面0誤判定 | 確定。閾値は実際は「30%+非ゼロ画素あり」でも発火し、レポートより過激 |
| M-4 | スレッド毎 `set_allocator` とプールの実態不一致 | 確定(実測)。§3 訂正 |
| M-5 | prefetch メモリ設定が起動済みワーカーに無効+pause/spill 逆転 | 確定(distributed 2025.5.1 ソースで裏取り) |
| M-6 | チャンク書き込み例外経路で writer スレッドリーク+future 未キャンセル | 確定。§3 訂正(ハンドルは解放される) |
| M-7 | overview 再構築失敗で計算済み COG 本体を喪失 | 確定 |
| M-8 | 具象 CuPy 配列が map_blocks kwargs としてグラフに埋め込み | コード側確定(ストリーミング経路の per-chunk 再送は確実)。増幅率は要実測 |
| M-9 | `_build_zstd_overviews` の ALL_CPUS 強制 | 確定 |
| M-10 | `dask_config.set` のプロセスグローバル恒久汚染 | 確定。修正はクラスタ生成前を跨ぐ形にする必要あり(ワーカーは spawn 時に config を継承) |
| M-11 | `mkstemp` fd リーク | 確定。Windows で safe_unlink 5連敗→一時ファイル残留まで確認 |
| M-12 | ストリーミング経路の出力親ディレクトリ mkdir 欠落 | 確定+N-7(Windows 全環境で発火条件成立) |
| M-13 | ワーカー GDAL キャッシュ下限 512MB が cgroup 予算破り | 確定 |
| M-14 | `str.replace(".vrt", ...)` のパス破壊、external CLI 経路は回復不能 | 確定(トリガはやや特殊) |
| M-15 | `or tiled` で COG 判定が無意味化+戻り値無視 | 確定 |
| M-17 | レンジ未解決時の tile=float32 / dask=量子化続行の不一致 | 確定。量子化数式自体の一致も確認 |
| M-18 | hillshade の weights フォールバック不一致 | 確定(他9アルゴリズムとの差分を全件確認) |
| M-19 | `_radius_to_downsample_factor` のブロック形状依存 | PARTIAL 採用。§3 訂正(実効半径は補正済み、変わるのは近似忠実度) |
| M-20 | `_norm_stat_max_scale` の実効ハロー過小見積 | 確定(AO はデフォルトでは無害)。N-3 と同時修正 |
| M-21 | hillshade の (H,W,3) スタック法線で VRAM 3倍 | 確定(specular の対策コメントと不整合) |
| M-22 | openness のサンプル毎 GPU→ホスト同期(最大160回/ブロック) | 確定。`np.hypot` 一語の修正 |
| M-23 | phase_congruency / tv_decomposition の Zarr 統計フォールバックがブロック毎でシーム | 確定。アルゴリズム4群でポリシーが3種に割れていることも確認 |
| M-24 | frangi のストライド読みフォールバックが全チャンク読み出し | 確定(リポジトリ自身の禁止 docstring と矛盾) |
| M-25 | tv structure / scale_drift direction で統計プリパスが全て無駄 | 確定(最大9タイル×140反復の空振り) |
| M-26 | npr_edges ヒステリシスの反対角2方向欠落(6近傍) | 確定 |
| M-27 | tile 直接パスの統計フォールバックがタイル自身から計算(+fractal は2回計算) | 確定(発火は統計欠落時のみ) |
| M-28 | `_direct_npr_edges` の古い既定値 0.1/0.3 | 確定(実質発火不能、Low 相当) |
| M-29 | `DASK_DISTRIBUTED__*` env が import 済み dask に無効 | 確定+N-4(ワーカーにも効かない疑い) |
| M-30 | `RMM_*` env 3種が完全デッドコード | 確定(全パッケージ grep で裏取り) |
| M-31 | `GDAL_SWATH_SIZE` の単位バグ | 確定。§3 訂正(影響は COG 生成コピースワス) |
| M-32 | Windows で `--no-progress` が無言の no-op | 確定。§3 訂正 |
| M-33 | 度→m 換算式のバックエンド間不一致 | 確定+N-2(等方上書きが支配的) |
| M-34 | debug_fractal_anomaly が本番と適用ドメイン乖離 | 確定(正規化前 vs 正規化後、しきい値の意味が別物) |
| M-35 | install_gdal の sudo 不在時の無診断続行 | 確定(apt の stderr は出るが診断なし) |
| M-36 | test_zarr_io が値を一切検証しない | 確定(変数名すら or で許容) |

### Low(86件採用: 76件確定+訂正付き10件。L-22, L-57, L-62 は不採用、L-5 は半分採用)

コア/タイル: L-1, L-2, L-3, L-4, L-5(tile_size 検証のみ), L-6, L-7, L-8, L-9, L-10 — 全て確定。
コア/Dask: L-11〜L-16, L-18, L-20, L-21 確定。L-17(実質却下、注記のみ)、L-19(gpus[0] のみ)、L-23(外部産 Zarr に限定)。
io: L-24〜L-28, L-30〜L-36, L-38(+N-6), L-39, L-40, L-41 確定。L-29(b のみ)、L-37(最低優先)。L-41 は CRS なし DEM の頻度を考えると Medium 相当に格上げしてよい。
algorithms: L-42〜L-48, L-50, L-53〜L-56, L-58〜L-60, L-63〜L-70 確定(L-44/L-46/L-48/L-51 は潜在・防御のみ、L-60 の linspace は実際はデッドコード)。L-52, L-61, L-71 は §3 訂正付き。
cli/config/utils: L-72〜L-82 全て確定(L-82 は事実のみ、ceil 化は好み)。
tests/tools: L-83〜L-86, L-88〜L-90, L-92 確定。L-87, L-91 は §3 訂正付き。L-89 には N-1(本番側)を追加。

---

## 6. 修正計画

方針を先に置く。パリティ(tile↔Dask の同一出力)と出力の正しさに直結する修正を最初に固め、その検証を回すためのテスト基盤整備を同時に走らせる。OOM・性能・エラーパスはその後でよい。デッドコードと衛生は最後にまとめて掃除する。個々の修正は小さいが、統計フォールバックのポリシー統一(M-17/M-23/M-27)と `--agg stack` の契約確定(H-5/L-15)だけは先に設計判断が要る。

**検証環境の制約:** パリティ修正(P1 群)の確認には Linux/Dask 側の実行が要る。現在 Windows ローカルには dask_cuda/rmm がなく、GPU パリティテストは RunPod/vast 環境の確保後になる。コード修正と単体テスト整備はローカルで完結する。

### P0: テスト基盤(他の全修正の検証前提)

| # | 内容 | 対象 |
|---|---|---|
| P0-1 | 未ガード4ファイルに `pytest.importorskip` を導入し、全テストの sys.path 挿入を撤去、`conftest.py` または pyproject `pythonpath = ["."]` に一本化 | H-6 |
| P0-2 | `addopts` の `--cov` を外すか `pytest-cov` 必須を README/CI に明記 | L-91 |
| P0-3 | 乱数シード固定(4ファイル5箇所) | L-83 |
| P0-4 | test_zarr_io に値・形状・dtype のラウンドトリップ検証を追加、`zarr_format=2` の xarray 下限整合を確認 | M-36, L-88 |
| P0-5 | ZSTD アサーションにビルド依存 skip、argparse `_actions` 依存の除去 | L-87, L-90 |

### P1: 出力の正しさとパリティ(最優先)

| # | 内容 | 対象 |
|---|---|---|
| P1-1 | tile 側の `inject_global_stats` を topousm_fast 半径分割より前へ移動し、デッド変数 `_topousm_fast_full_radii/_weights` を削除 | H-1 |
| P1-2 | `--agg stack` の契約を確定する。推奨: 両バックエンドとも band-first (C,H,W) を正式契約とし、tile フォールバックで `cp.moveaxis(result, 0, -1)`、Dask 終盤の xarray ラップを ndim==3 対応に。工数を抑えるなら暫定で両 CLI とも stack を reject し、対応時期を明示 | H-5, L-15 |
| P1-3 | `_drift_combine_block` 冒頭で `_vs_combine_block` と同じ NaN 再充填を実装 | H-3 |
| P1-4 | `_build_nodata_mask` に `mask_nodata \|= np.isnan(dem_tile)` を合成(nodata_ratio 判定も同時に直る) | M-1 |
| P1-5 | 度→m 換算を `io/raster_info.py` の1関数(WGS84 精密級数)に統一し、`dask_processor.py:1168-1183` の等方上書きを軸別スケール保持に修正 | M-33, N-2 |
| P1-6 | `np.isclose` の rtol 既定値を本番(`_impl_fractal_anomaly.py:403`)とデバッグツールの両方で `rtol=0` に固定 | N-1, L-89 |
| P1-7 | `GDAL_SWATH_SIZE` を `str(cache_bytes)` に修正 | M-31 |
| P1-8 | npr_edges ヒステリシスに反対角2方向の roll を追加 | M-26 |
| P1-9 | 統計フォールバックのポリシーを1つに決めて4群を統一する。推奨: 統計未注入時(Zarr 入力・プリパス失敗)は全アルゴリズムとも「Dask 側は中央ウィンドウ推定(topousm_fast 方式)、tile 直接パスは使用せず警告付きフォールバック」。frangi のストライド読みも同時に中央ウィンドウ化 | M-17, M-23, M-24, M-27, L-65, L-52 |
| P1-10 | 量子化の ±inf を `~np.isfinite` / `~cp.isfinite` で NoData 化(encoding と dask 量子化の両側同時) | L-38, N-6 |
| P1-11 | hybrid overview 注入の地理座標系ゲートを両バックエンドで一致させる(方針決定: 地理 DEM で overview 経路を許すか、両方で禁止するか) | N-5, L-71, L-43 |

### P2: OOM・性能

| # | 内容 | 対象 |
|---|---|---|
| P2-1 | チャンク縮小判定を半径解決後へ移動(または auto 時は `auto_spatial_radii(short_side)` の長さで判定) | H-2 |
| P2-2 | `st_component_block` を (u,v) 同時返却の1パスに変更(2ch 場)、最低限 `coarse_cache` を共有 | H-4 |
| P2-3 | hillshade 法線を specular と同じ成分分離形に書き換え | M-21 |
| P2-4 | GDAL キャッシュ配分の下限を予算連動に(`max(64, min(2048, budget // workers))` 等) | M-13 |
| P2-5 | prefetch config ブロックを削除(残すならクラスタ生成前+target<spill<pause<terminate の順序) | M-5 |
| P2-6 | openness の距離計算を `np.hypot` に | M-22 |
| P2-7 | `_compute_norm_stats_tiled` を component/output モードでスキップ | M-25 |
| P2-8 | coarse field 群を `client.scatter(broadcast=True)` の Future 渡しに(ストリーミング経路優先) | M-8 |
| P2-9 | `_radius_to_downsample_factor` の block_factor を決定論化(グローバル定数チャンク基準)し、端チャンク/バックエンド差分の回帰テストを追加 | M-19 |
| P2-10 | `_norm_stat_max_scale` にアルゴリズム毎の実効ハロー(内部デフォルト込み)フックを追加し、norm-stats 窓の包含も確認 | M-20, N-3 |
| P2-11 | 縁 nodata 推定を `out_shape` 間引き読みに変え、適用時 warning へ格上げ | M-3 |

### P3: エラーパスと資源

| # | 内容 | 対象 |
|---|---|---|
| P3-1 | overview 再構築失敗時に `src.replace(dst)` で復旧してから再送出 | M-7 |
| P3-2 | writer 終了センチネルを finally で保証し、残 future を `client.cancel()` | M-6 |
| P3-3 | `mkstemp` の fd を `os.close` | M-11 |
| P3-4 | ストリーミング経路に `dst_cog.parent.mkdir(parents=True, exist_ok=True)` | M-12 |
| P3-5 | 成功時にタイル一時ディレクトリをデフォルト削除(`--keep-tiles` で保持)、最低限 info ログ | M-2 |
| P3-6 | `.vrt` 置換を `Path.with_name()` に統一(2箇所) | M-14 |
| P3-7 | 失敗時の中途出力削除、tqdm/pbar のリーク閉じ | L-34, L-16 |
| P3-8 | tile_size/padding の入力バリデーション(`parser.error` 化と併せて) | L-5, L-76 |

### P4: 設定・デッドコード衛生

| # | 内容 | 対象 |
|---|---|---|
| P4-1 | `RMM_*` env ブロック削除、`DASK_DISTRIBUTED__*` env 経路を廃止して `make_cluster` の `dask_config.set` に統一(クラスタ生成前スコープのコンテキスト化と併せて) | M-29, M-30, M-10, N-4 |
| P4-2 | `_build_zstd_overviews` を `container_cpu_count()` に統一 | M-9 |
| P4-3 | アロケータ設定をプロセス起動時1回のモジュール初期化へ | M-4 |
| P4-4 | COG validator を `LAYOUT=COG` 限定にし、呼び出し側で戻り値を処理 | M-15 |
| P4-5 | overview levels のラスタサイズ適応クランプ | M-16(降格後) |
| P4-6 | tile パイプラインへ show_progress 連携(最低限ヘルプに Windows 無効を注記) | M-32 |
| P4-7 | デッドコード一掃: `_direct_*` の到達不能 spatial 分岐(npr_edges の現役チェックは残す)、`_normalization.py` の重複 stat 関数、L-9/L-11/L-14/L-29b/L-74、`_determine_optimal_radii` の未達分岐 | L-66, L-67, L-42, L-9, L-11, L-14, L-29, L-74, L-60 |
| P4-8 | tools 整備: debug_fractal_anomaly を本番ロジック import 型に書き換え(または削除)、install_gdal の sudo 診断+戻り値チェック | M-34, M-35, L-84, L-85, L-86 |

### P5: 残り Low の一括処理

ファイル単位でまとめて処理する(各修正は独立で、順不同):

- **tile_processor / tile_compute / tile_io**: L-1(フラグ保存), L-2(セット集約), L-3(コピー削減), L-4(weights 再整列), L-6(rm -rf 提案の Windows 対応), L-7(定数集約), L-8(reader close), L-10(マスク適用順)
- **dask 系**: L-12(例外分離), L-13(progress 統一), L-18, L-20, L-21(ndim 検証), L-23(CRS 警告)
- **io 系**: L-24〜L-28, L-30〜L-33, L-35, L-36, L-39, L-40, **L-41(Medium 相当として優先度高め: CRS なし+度単位ピクセルの妥当性チェック)**
- **algorithms 系**: L-44〜L-48, L-50, L-53〜L-56, L-58, L-59, L-61, L-63, L-64, L-68〜L-70
- **cli/config**: L-72, L-73, L-75, L-77〜L-82
- **メタデータ**: L-92(classifier と license 形式)

### 検証ゲート

- P1 完了時: 同一 DEM(投影系・地理系・数値 nodata・NaN nodata の4種)で tile と Dask の出力を画素比較するパリティテストを新設し、以後のフェーズの回帰ゲートにする。
- P2 完了時: マルチ半径 auto 実行の per-block ピーク VRAM を計測し、H-2/H-4/M-21 の効果を数値で残す。
- 各フェーズは独立にコミット可能な粒度で分割してある。P1-2 と P1-9 のみ、着手前に契約の設計判断を固めること。
