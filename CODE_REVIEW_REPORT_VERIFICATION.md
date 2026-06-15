# CODE_REVIEW_REPORT.md 検証レポート（メタレビュー）

**対象:** `CODE_REVIEW_REPORT.md`（別 AI モデルによる FujiShaderGPU レビュー）
**検証日:** 2026-06-15
**検証手法:** レポートの全指摘を実コードと 1 件ずつ照合（`file:line` を確認）。パラメータの流れ（`cli/args.py` → `build_algo_params` → 各 `_impl_*`）、削除提案された関数の呼び出し元、git 追跡状態を実測。GPU 実行は未実施（静的検証）。
**結論の要約:** レポートは「それらしい」が、**根拠を実コードで追っていない指摘が複数含まれており、一部は事実誤認・削除すると壊れる提案**を含む。事実確認できた良い指摘もある。**鵜呑みは禁物**。

---

## 1. 総合判定

| 区分 | 件数の目安 | コメント |
|------|-----------|----------|
| ✅ 妥当（事実・方向性とも正しい） | 約 6 | tmp ディレクトリ削除、巨大モジュール、`except Exception`、COG 検証の戻り値無視 など |
| ⚠️ 部分的に妥当（事実は合うが重大度・因果が誇張/不正確） | 約 8 | GDAL 例外、scales/radii パディング、地理メートル換算、hybrid overview など |
| ❌ 誤り・要修正（事実誤認、または提案通りにやると壊れる） | 約 4 | `_norm_stat_max_scale` の scales 無視、デッドコード削除提案、rioxarray 非推奨、リポジトリ汚染 |

レポートの最大の問題は、**「CLI 引数名（`vs_scales` / `surprise_scales` / `fractal_radii`）」と「実際の params dict キー（`scales` / `radii`）」を取り違えている**点。これが scales/radii 系の指摘（2.1.3・2.1.4・3.2）の精度を大きく下げている。`build_algo_params`（[args.py:259-285](FujiShaderGPU/cli/args.py#L259-L285)）を読めば、`--vs-scales` → `p["scales"]`、`--surprise-scales` → `p["scales"]`、`--fractal-radii` → `p["radii"]` とマッピングされ、`vs_scales` 等のキーは params に存在しないことが分かる。レポートはこの 1 ファイルを追っていない。

---

## 2. 個別検証

### 2.1 高重大度とされた項目

#### 2.1.1 GDAL `DontUseExceptions()` の混在 — ⚠️ 部分的に妥当（重大度は誇張）

- **事実:** TRUE。3 箇所で確認（[cog_builder.py:24](FujiShaderGPU/io/cog_builder.py#L24)、[cog_validator.py:11](FujiShaderGPU/io/cog_validator.py#L11)、[system_config.py:21](FujiShaderGPU/config/system_config.py#L21)）。
- **誇張・誤りの点:**
  - 「rasterio が GDAL の失敗に気づかなくなる」という因果は**技術的に不正確**。rasterio は `osgeo.gdal` の Python 例外状態に依存せず、独自の Cython エラーハンドラ（CPLErr）で例外化する。`gdal.DontUseExceptions()` は `osgeo.gdal` バインディング側の設定で、rasterio の挙動には影響しない。
  - 「`gdal.Open` が None でも気づかず不正結果」も、少なくとも cog_validator では `if ds is None: return False`（[cog_validator.py:24](FujiShaderGPU/io/cog_validator.py#L24)）で既にガード済み。
- **妥当な点:** GDAL 4 で `UseExceptions()` がデフォルト化する方向は事実で、将来の衛生としての一元化提案は有効。
- **判定:** 高重大度ではなく **低〜中**。forward-compat の整理事項。

#### 2.1.2 包括的すぎる `except Exception` — ✅ 妥当

- **事実:** TRUE。[algorithms/__init__.py:9,49](FujiShaderGPU/algorithms/__init__.py#L9) は `except Exception: pass`、[__main__.py:55-57](FujiShaderGPU/__main__.py#L55) はトレースバックを潰して `print` のみ。
- **コメント:** import 失敗を握り潰すと「アルゴリズムが見つからない」という不明瞭な症状になる、という指摘は正しい。`__main__` の `--verbose` 提案も妥当。ただし「コアパイプライン各所」という一般化はやや誇張（多くの `except` はローカルにログ付きで処理されている）。
- **判定:** 妥当（中）。

#### 2.1.3 `_required_padding_for_algorithm` が scales を無視 — ⚠️ 部分的に妥当（重大度は状況依存）

- **事実:** `_unified_radii` は `algo_params.get("radii")` のみ参照（[tile_processor.py:228-235](FujiShaderGPU/core/tile_processor.py#L228-L235)）。`visual_saliency` / `multiscale_terrain` / `scale_space_surprise` はユーザー値を `params["scales"]` に持つため、`--vs-scales` 等を指定しても `_unified_radii` は**デフォルト値**（`[2,4,8,16]` 等）を返す。→ ユーザー scales がデフォルト最大値を超える場合のみ**パディング過小（シーム）**、下回る場合は過大。これは実在の不整合。
- **誇張の点:** 「scales を無視＝パディングが効かない」ではなく「**デフォルト scales で計算される**」。影響はユーザー scales がデフォルトを超えたときに限定。さらに visual_saliency は tile バックエンドで常に Dask フォールバック（[dask_bridge.py:451-457](FujiShaderGPU/algorithms/tile/dask_bridge.py#L451)）するため、per-tile halo の意味合いも単純でない。
- **判定:** 実在のバグだが**状況依存・中**。レポートの修正方針（アルゴ名に応じ scales/radii を切替）自体は妥当。

#### 2.1.4 Dask hybrid overview が radii を参照 — ⚠️ 因果が不正確（実害は効率のみ）

- **事実:** hybrid ブロックは `params.get("radii")` で**ゲートされている**（[dask_processor.py:1235-1243](FujiShaderGPU/core/dask_processor.py#L1235-L1243)）。`--vs-scales`/`--surprise-scales` 使用時は `params["radii"]` が空なので、**ブロックごとスキップされる**。
- **レポートの誤り:** 「デフォルト `[2,4,8,16]` に対して coarse field が生成される」「不要な `16` の field が生成される」は**誤り**。生成は走らず、アルゴリズムは single-block パスにフォールバックする。しかも compute 側は `radii` が無ければ `scales` を使う（[_impl_visual_saliency.py:252-253](FujiShaderGPU/algorithms/_impl_visual_saliency.py#L252-L253)、[_impl_experimental.py:177-178](FujiShaderGPU/algorithms/_impl_experimental.py#L177-L178)）ので**結果は正しい**。
- **真の問題:** ネイティブな scale 引数を使うと **overview 最適化が黙って無効化**される（＝大 scale が full-res 単一ブロックで処理され VRAM 圧迫）。これは正確性バグではなく**効率の問題**。
- **判定:** 方向性は有用だが因果説明が不正確。重大度は**中（効率）**。

#### 2.1.5 地理メートル換算の不一致 — ⚠️ 事実だが重大度は低

- **事実:** TRUE。[linux_cli.py:83-86](FujiShaderGPU/cli/linux_cli.py#L83-L86) は詳細式、[raster_info.py:42](FujiShaderGPU/io/raster_info.py#L42) は定数 `111_320.0`。なお tile_processor も同じ定数（[tile_processor.py:810](FujiShaderGPU/core/tile_processor.py#L810)）なので、**外れ値は linux_cli のみ**。
- **コメント:** 差は ~0.5%、地理座標 DEM 限定、可視化用途。レポート自身も「許容範囲かも」と書いている。**高重大度の枠に入れるのは過大**。共通化の提案自体は妥当。
- **判定:** 低（一貫性 nit）。

#### 2.1.6 spatial モードが実質 Dask 依存 — ✅ 妥当（重大度は低）

- **事実:** TRUE。[dask_bridge.py:426-432](FujiShaderGPU/algorithms/tile/dask_bridge.py#L426) で spatial の多くを `_FallbackToDask()`。`pyproject.toml` の `windows` extra に `dask[array]>=2024.4` あり（確認済み）なので動作はする。
- **判定:** ドキュメント整備の指摘として妥当（低）。

#### 2.1.7 一時ディレクトリの無条件削除 — ✅ 妥当（レポート中で最も価値のある指摘）

- **事実:** TRUE。[tile_processor.py:677-684](FujiShaderGPU/core/tile_processor.py#L677-L684) が既存候補を無条件 `shutil.rmtree`。
- **到達性も確認:** `--tmp-dir` は実在のユーザー CLI 引数（[base.py:41](FujiShaderGPU/cli/base.py#L41)、epilog 例にも `--tmp-dir existing_tiles`）。よって**ユーザーが重要ディレクトリを指定すると本当に消える**。さらに毎回消えるため resume も不可。
- **判定:** 妥当・**中〜高**。ユニーク名化／削除前確認の提案は適切。`--cog-only` が既存タイルを要求する一方で本関数が rmtree する点（[tile_processor.py:1279](FujiShaderGPU/core/tile_processor.py#L1279)）の相互作用は追加検証の価値あり。

### 2.2 中重大度とされた項目（抜粋）

| 項目 | 判定 | 要点 |
|------|------|------|
| 2.2.1 巨大モジュール | ✅ 妥当 | 行数ほぼ正確（tile 1644 / dask 1585 / dem_preprocess 972 / _nan_utils 878、実測一致）。分割は設計判断。 |
| 2.2.2 `.compute()` in coarse_large_radius_response | ✅ 事実 | [_nan_utils.py:385](FujiShaderGPU/algorithms/_nan_utils.py#L385) に存在。ただし coarse は `persist()` 済みの小配列（[_nan_utils.py:364](FujiShaderGPU/algorithms/_nan_utils.py#L364)）で影響軽微。低〜中。 |
| 2.2.3 thread rasterio リーダー leak | ⚠️ 低 | shutdown フック無しは事実だが、パス変更/再利用時に close される（[tile_io.py:18-22](FujiShaderGPU/core/tile_io.py#L18)）。1 実行あたり ~ワーカー数に有界。`Too many open files` は単一実行では考えにくい。Windows ファイルロックは一理。 |
| 2.2.4 Dask 環境変数のタイミング | ⚠️ 低〜中 | `os.environ` 設定後に import は事実（[linux_cli.py:108-126](FujiShaderGPU/cli/linux_cli.py#L108)）。実害は import 順依存。`dask.config.set` 化の提案は妥当。 |
| 2.2.5 GPUtil の信頼性 | ⚠️ 低〜中 | 使用は事実（[linux_cli.py:118](FujiShaderGPU/cli/linux_cli.py#L118)）。一般論として妥当。 |
| 2.2.6 rioxarray.open_rasterio 非推奨 | ❌ 誤り | 使用は事実（[dask_io.py:48](FujiShaderGPU/core/dask_io.py#L48)）だが、**`open_rasterio` は現行 rioxarray の標準 API で非推奨ではない**。「削除される方向」は事実誤認。移行は不要。 |
| 2.2.7 `blur` が OUTPUT_VALUE_RANGES に無い | ✅ 妥当 | [output_encoding.py:40-63](FujiShaderGPU/io/output_encoding.py#L40) に blur 無し → int 出力時に推定レンジ→クリップ懸念は正当（中）。 |
| 2.2.8 slope percent クリッピング | ✅ 妥当（軽微） | [output_encoding.py:101](FujiShaderGPU/io/output_encoding.py#L101) で degree/radian 以外は None。 |
| 2.2.9 定数の散在 | ⚠️ 一部不正確 | `MAX_DEPTH` は実際には `Constants.MAX_DEPTH` で集約済み（[tile_processor.py:215](FujiShaderGPU/core/tile_processor.py#L215)）。散在の主張は一部当たらない。スタイル nit。 |
| 2.2.10 COG 検証の戻り値無視 | ✅ 妥当 | [tile_processor.py:1566,1633](FujiShaderGPU/core/tile_processor.py#L1566) で bool を捨てている。ただし失敗は内部で `logger.error` 済み。低〜中。 |
| 2.2.11 `__main__` のエラー隠蔽 | ✅ 妥当 | 2.1.2 と重複。 |

### 2.3 低重大度とされた項目（抜粋）

#### 2.3.2 未使用・死んだコード — ❌ 誤り（提案通り削除すると壊れる）

- `compute_global_stats` は**使用中**：[_impl_experimental.py:190](FujiShaderGPU/algorithms/_impl_experimental.py#L190)、[_impl_fractal_anomaly.py:275](FujiShaderGPU/algorithms/_impl_fractal_anomaly.py#L275)、[_impl_visual_saliency.py:268](FujiShaderGPU/algorithms/_impl_visual_saliency.py#L268)。「呼び出し元がなければ削除」→ **呼び出し元あり**。
- `classify_resolution` も**使用中**：[_impl_fractal_anomaly.py:334](FujiShaderGPU/algorithms/_impl_fractal_anomaly.py#L334)、[_impl_npr_edges.py:38,304](FujiShaderGPU/algorithms/_impl_npr_edges.py#L38)。「見当たらない」は**誤り**。
- `dask_shared.py` は**現役のハブ**：`dask_registry` が `DaskAlgorithm` を import し、`dask/*` と `tile/*` の全アルゴリズムクラスがここから import している。「移行残りの leftover」という表現は不正確。
- **判定:** ❌。`vulture` を実際に走らせていれば防げた誤り。**この節の削除提案は実行しないこと。**

#### 2.3.1 / 2.3.3 / 2.3.4 / 2.3.5 — おおむね妥当（重複・型ヒント・テスト不足・ログ統一）。一般論として有効。

### 3. セクション 3「具体的バグ箇所」

- 多くは 2 章のクロスリファレンス。重要なのは **3.2 `_norm_stat_max_scale` が scales を無視 → ❌ 誤り**。実コード [_norm_stats.py:92](FujiShaderGPU/algorithms/_norm_stats.py#L92) は `for key in ("radii", "scales")` と**"scales" を含んでいる**。vs/sss/mst は params["scales"]、fractal は params["radii"] なので、この関数は**実際には全ケースを正しくカバー**している。レポートが挙げた `vs_scales`/`surprise_scales`/`fractal_radii` は params に存在しないキーで、提案修正はデッドキーを足すだけ。
- 3.1・3.3・3.4・3.5・3.6・3.7・3.8・3.9・3.10・3.11 は事実としては存在（前述の通り重大度・因果の補正が必要）。

### 4. 性能上の懸念

- 4.1 メモリ推定の経験則（分母 3.0 等）— 妥当な推測。
- 4.2 `free_all_blocks()` 多用 — 使用は事実だが**ファイル名が一部不正確**。実際は [_impl_fractal_anomaly.py:433](FujiShaderGPU/algorithms/_impl_fractal_anomaly.py#L433) / [_impl_npr_edges.py:354](FujiShaderGPU/algorithms/_impl_npr_edges.py#L354) / [_impl_specular.py:298](FujiShaderGPU/algorithms/_impl_specular.py#L298) / [_norm_stats.py:193](FujiShaderGPU/algorithms/_norm_stats.py#L193)（レポートは dask_processor を挙げるが該当せず）。さらに Linux は RMM allocator（[linux_cli.py:121](FujiShaderGPU/cli/linux_cli.py#L121)）使用時、`cp.get_default_memory_pool()` が実プールと別になり得るため、断片化の議論はやや単純化。方向性は一理。
- 4.3 CUDA stream 並列なし — 妥当。
- 4.4 大半径 `map_overlap` の rechunk → OOM 懸念 — 妥当。

### 5. リポジトリクリーンアップ（セクション 6）— ❌ 主張が不正確

- `build/`・`htmlcov/`・`*.egg-info/`・`__pycache__/`・`.ruff_cache/`・`.pytest_cache/` は**すべて既に `.gitignore` に記載済み**（[.gitignore:11,40,24,2,187,51](.gitignore#L11)）。
- `git ls-files` で `build/` と `*.egg-info` は**git に追跡されていない**ことを確認。よって「リポジトリ汚染」「.gitignore に追加を強く推奨」は**前提が誤り**（既に ignore 済み・未追跡）。
- 事実として `build/lib/.../tile_processor.py` は 2046 行でソース 1644 行より 402 行長い（「400 行以上」は正確）。ただし**未追跡のローカルビルド成果物**であり、git diff やパッケージングを汚さない。ローカル掃除としては有用だが、「汚染」は言い過ぎ。

---

## 6. レポートが見落としている／良かった点

**良い指摘（採用推奨）:**
- tmp ディレクトリの無条件 rmtree（2.1.7）— 最も価値あり。
- `except Exception` の握り潰しと `__main__` のトレースバック欠落（2.1.2）。
- `blur` の整数出力レンジ未定義（2.2.7）。
- COG 検証戻り値の無視（2.2.10）。
- scales/radii のパディング不整合（2.1.3、ただし状況依存）。

**注意すべき弱点:**
- **CLI 引数名と params キーの混同**で scales/radii 系の因果説明（2.1.3 後半・2.1.4・3.2）が崩れている。
- **デッドコード判定（2.3.2）が誤り**で、提案通り削除すると import エラーになる。
- **rioxarray 非推奨（2.2.6）は事実誤認**。
- **リポジトリ汚染（6）は既に対処済み**を未対処と誤認。
- 重大度の付け方が全体に高め（地理換算 0.5% を「高」に置く等）。

---

## 7. 推奨アクション（補正後の優先度）

1. **tmp ディレクトリ安全化**（ユニーク名 or 削除前確認）— 唯一データ損失に直結。最優先。
2. **scales/radii の一元化** — パディング（2.1.3）と hybrid overview ゲート（2.1.4）でアルゴリズム名に応じ `scales`/`radii` を切替。ただし「正確性バグ」ではなく主に**シーム/効率**問題として扱う。`_norm_stat_max_scale` は**既に正しい**ので触らない。
3. **`except Exception` の限定と `__main__` の `--verbose`** — デバッグ性向上。
4. `blur` レンジ追加・COG 検証戻り値の反映 — 小さく確実。
5. GDAL 例外は「forward-compat 整理」として低優先で。rasterio との競合論は割り引く。
6. **やらないこと:** 2.3.2 のデッドコード削除、rioxarray 移行、.gitignore 追記（いずれも不要 or 有害）。

> 総括: 元レポートは「アーキテクチャの長所」把握は的確で、tmp 削除・例外握り潰しなど**実装を読めば確実に当たる指摘**は良い。一方、**params の流れを追わずに書いた scales/radii 系の因果**と**実行ツール（vulture/git）を使わなかったデッドコード・リポジトリ汚染の指摘**に誤りが集中している。採用時は本検証の判定（✅/⚠️/❌）に従って取捨選択することを推奨する。
