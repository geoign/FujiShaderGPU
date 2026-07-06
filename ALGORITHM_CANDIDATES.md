# FujiShaderGPU アルゴリズム実装候補ストック

DEM地形可視化の新規アルゴリズム候補を、複数のAI・人間の提案から蓄積するドキュメント。
既存実装(hillshade / slope / curvature / openness / ambient_occlusion / specular /
atmospheric_scattering / topousm_fast / multiscale_terrain / blur / npr_edges /
visual_saliency / fractal_anomaly / scale_space_surprise / multi_light_uncertainty)
と数学的な「族」が重ならないものを優先する。

既存実装の共通項は「等方的ガウシアン+点単位統計+照明モデル」。したがって
**方向性(異方性)・位相/周波数領域・トポロジー・変分法・スケール間ダイナミクス**が空白地帯である。

## ステータス一覧

| ID | 名称 | 系統 | 判断 | 提案元 |
|----|------|------|------|--------|
| A1 | 構造テンソル異方性場 | CV/方向性 | **実装済み** (`structure_tensor`, 2026-07-06) | Claude Fable 5 (2026-07-06) |
| A2 | Frangi Vesselness | CV/医用画像 | **実装済み** (`frangi`, 2026-07-06) | Claude Fable 5 (2026-07-06) |
| A3 | LIC 流線テクスチャ | 科学可視化 | **実装済み** (`lic`, 2026-07-06) | Claude Fable 5 (2026-07-06) |
| B1 | Phase Congruency レリーフ | CV/位相 | **実装済み** (`phase_congruency`, 2026-07-06) | Claude Fable 5 (2026-07-06) |
| B2 | 方向性ウェーブレットエネルギー | 調和解析 | 先送り | Claude Fable 5 (2026-07-06) |
| C1 | Persistence(位相的持続性)マップ | TDA | 先送り | Claude Fable 5 (2026-07-06) |
| D1 | TV 構造-テクスチャ分解 | 変分法/PDE | **実装済み** (`tv_decomposition`, 2026-07-06) | Claude Fable 5 (2026-07-06) |
| E1 | Scale Drift(スケール漂流場) | オリジナル | **実装済み** (`scale_drift`, 2026-07-06) | Claude Fable 5 (2026-07-06) |
| E2 | Eigenterrain 疑似カラー | オリジナル/教師なし学習 | 先送り | Claude Fable 5 (2026-07-06) |
| E3 | 侵食時間レリーフ | オリジナル/PDE | 先送り | Claude Fable 5 (2026-07-06) |

### 実装メモ(2026-07-06、Claude Fable 5)

採用6件を tile / Dask 両バックエンドに実装済み(`_impl_structure_tensor.py` /
`_impl_frangi.py` / `_impl_lic.py` / `_impl_phase_congruency.py` /
`_impl_tv_decomposition.py` / `_impl_scale_drift.py`)。合成UTM DEMでのCLI
エンドツーエンド18ラン+ユニットテスト(数値的性質の検証込み)を通過。
設計時からの主な変更点:

- **HSV 3バンド出力は見送り**: Dask側COGライタが1バンド固定のため、両バックエンド
  同一出力の原則を優先。方向情報は単バンドモード(`--st-output orientation`、
  `--drift-output direction`、角度を[0,1)にマップ)で提供。ライタのマルチバンド
  対応後にHSV合成を追加予定。
- **A3 LIC のノイズは座標ハッシュではなく標高値ハッシュ**: タイル座標に依存しない
  ためタイル分割・バックエンド・チャンク割りに対して構成的にシームフリー。
  完全平坦地はテクスチャなし(流向も無いので問題なし)。
- **B1 の波長は ≤64px にクランプ**(FFT halo 2λ ≤ MAX_DEPTH=150 制約)。
  それ以上のスケールはオーバービュー経由の将来拡張として文書化。
  ノイズ閾値はKovesi原法の簡略版(最小スケール振幅の大域中央値+Rayleighモデル)。
- **E1 の LK 窓は σ≤24 にキャップ**(combine段のhalo予算)。ペア間の漂流は
  Δσ で正規化して合成。
- シーム検証: tile 512 vs 単一タイルで tv=ビット一致、lic p99.9=0.05%、
  structure_tensor / scale_drift はタイル境界で差分ゼロ(残差はNoData縁の
  ブロック内nanmean充填差のみ=既存アルゴリズムと同じ既知挙動)。

実装時の共通制約(全候補に適用):

- タイル/Dask 両バックエンドで同一出力になること。halo は `Constants.MAX_DEPTH = 150` px が上限。
  それを超えるサポートが必要なスケールは、既存の hybrid coarse path
  (COG オーバービューで粗計算→フル解像度に合成)に載せる。
- 可能な限り既存の `--mode spatial` / `--radii` / `--weights` 体系に統合する。
- NaN(NoData)対応は `_nan_utils.py` の流儀(nanmean 充填→計算→NaN 復元)に従う。
- 正規化は `_global_stats.py` によるグローバル統計を使い、タイル境界のシームを作らない。
- 新規アルゴリズムのファイル構成: `algorithms/_impl_<name>.py`(本体)+
  `algorithms/dask/<name>.py` + `algorithms/tile/<name>.py`(薄いラッパ)+
  `dask_registry.py` / CLI への登録(`tests/test_registry_cli_sync.py` が同期を検査する)。

---

## 採用候補(詳細)

### A1. 構造テンソル異方性場(Structure Tensor Fabric)

**提案: Claude Fable 5(2026-07-06 検討)/ 判断: 採用**

#### 概要

勾配の外積を平滑化した 2×2 構造テンソル

```
J_ρ = G_ρ * (∇z ∇zᵀ)   (G_ρ: 積分スケール ρ のガウシアン)
```

の固有値 λ1 ≥ λ2 と第一固有ベクトルから、各画素の
**支配方向 θ = ½·atan2(2J12, J11−J22)** と
**異方性強度(コヒーレンス) C = ((λ1−λ2)/(λ1+λ2+ε))²** を得る。
θ を Hue、C を Saturation、任意の明度成分(hillshade 等)を Value に割り当てた
HSV→RGB 出力にすると、断層系・氷河擦痕・褶曲軸・砂丘の走向・リニアメントの
「地形ファブリック」が色相として一望できる。既存アルゴリズムに「方向」を
第一級の出力とするものは皆無であり、最小コストで新しい次元を追加する。

θ は π 周期(0° と 180° は同じ走向)なので、Hue へは 2θ でマップする。

#### 出典

- Bigün, J. & Granlund, G.H. (1987) "Optimal Orientation Detection of Linear Symmetry." *ICCV 1987*.
- Knutsson, H. (1989) "Representing local structure using tensors." *SCIA 1989*.
- Weickert, J. (1999) "Coherence-Enhancing Diffusion Filtering." *IJCV* 31(2/3). — コヒーレンス定義の出典。
- 地形応用: 構造地質のリニアメント自動抽出文献多数(例: Koike et al. 1995, *Computers & Geosciences* のセグメントトレース法)。

#### 実装の方向性

- 部品はガウシアン微分(微分スケール σ)+要素ごとの積+ガウシアン平滑(積分スケール ρ)のみ。
  `cupyx.scipy.ndimage.gaussian_filter` で完結し、halo = ~4(σ+ρ) で MAX_DEPTH に収まりやすい。
- `--radii` を積分スケール ρ の系列として解釈し、マルチスケール合成はテンソルを
  radii 重みで加算平均してから固有値分解する(テンソルは線形に混ぜられるのが利点)。
- パラメータ案: `--derivative-sigma`(微分スケール、既定 1.0)、
  出力モード `--output hsv|coherence|orientation`(既定 hsv)。
- 出力: hsv モードは RGB 3バンド(既存はグレー1バンド主体なので、COG 書き出し側の
  3バンド対応を確認・拡張する。GEBCO パイプラインで RGB COG 実績あり)。
  coherence / orientation モードは1バンドで既存経路をそのまま使える。
- 固有値分解は 2×2 閉形式(atan2 と平方根)で書き、行列ルーチンは不要。
- A2・E1 と共有できるので `_impl_structure_tensor.py` に
  「ガウシアン微分+テンソル場+固有値解析」の共通部品を置く。

---

### A2. Frangi Vesselness(マルチスケール Hessian 固有値フィルタ)

**提案: Claude Fable 5(2026-07-06 検討)/ 判断: 採用**

#### 概要

医用画像の血管強調フィルタを DEM に転用する。スケール σ ごとに
スケール正規化 Hessian(σ²·H)の固有値 |λ1| ≤ |λ2| を取り、

```
R_B = λ1/λ2(ブロブ度),  S = √(λ1²+λ2²)(構造エネルギー)
V_σ = exp(−R_B²/2β²) · (1 − exp(−S²/2c²))   (λ2 の符号で尾根/谷を選別)
V   = max_σ V_σ
```

とする。スカラー曲率(既存 curvature)と違い**固有値の比**を使うため
コントラスト(比高)に依存せず、谷線・尾根線・旧河道・堤防・エスカー・ガリーの
**線状ネットワークだけ**が浮かび上がる。λ2 < 0 で尾根、λ2 > 0 で谷を抽出。
スケール正規化により σ を横断して線幅の違う構造を同じ強度で拾う。

#### 出典

- Frangi, A.F. et al. (1998) "Multiscale vessel enhancement filtering." *MICCAI 1998, LNCS 1496*.
- Sato, Y. et al. (1998) "Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images." *Medical Image Analysis* 2(2). — 同型の別定式化。
- Lindeberg, T. (1998) "Feature Detection with Automatic Scale Selection." *IJCV* 30(2). — σ² 正規化の根拠。
- 地形応用: 水路網抽出への Vesselness 適用例あり(例: Passalacqua et al. 2010, *JGR* の GeoNet はマルチスケール曲率+測地線で近い思想)。

#### 実装の方向性

- Hessian はガウシアン2階微分(gaussian_filter の order 指定)で得る。
  2×2 固有値は閉形式。halo = ~4σ_max。大きい σ は hybrid coarse path へ。
- `--radii` を σ 系列として解釈。スケール合成は Frangi 原法どおり max を既定とし、
  `--weights` 指定時は重み付き和も選べるようにする(既存体系との整合)。
- パラメータ案: `--feature ridge|valley|both`(既定 both: 尾根を正、谷を負にした
  発散型1バンド出力)、`--beta`(既定 0.5)、`--frangi-c`(既定: S の p95 の半分、
  グローバル統計プリパスで決定 — npr_edges の grad_stats と同じパターン)。
- 正規化: V は [0,1] に収まるので追加正規化は軽微。both モードは 0.5 中心の
  発散マップにして既存の gamma 処理に載せる。
- A1 と `_impl_structure_tensor.py` の微分部品を共有。

---

### A3. LIC 流線テクスチャ(Line Integral Convolution)

**提案: Claude Fable 5(2026-07-06 検討)/ 判断: 採用**

#### 概要

ホワイトノイズ画像を、DEM から導いたベクトル場(最急降下方向=水流方向、
または A1 の構造テンソル第一固有ベクトル=走向方向)に沿って線積分畳み込みする。
流線方向に相関を持つ「毛筆で撫でたような」テクスチャが得られ、排水パターン・
地形ファブリックが直感的に読める。hillshade と乗算合成すれば従来の陰影図に
流れの手触りを重ねた、既存のどれとも似ていない審美的な出力になる。
ベクトル場・ノイズ・積分長を変えるだけで表現の幅が広い(等高線方向 LIC も可能)。

#### 出典

- Cabral, B. & Leedom, L.C. (1993) "Imaging Vector Fields Using Line Integral Convolution." *SIGGRAPH '93*.
- Stalling, D. & Hege, H.-C. (1995) "Fast and Resolution Independent Line Integral Convolution." *SIGGRAPH '95*. — 高速化(流線再利用)。
- 地図学応用: Imhof 流のレリーフ表現に LIC を使う試みは散発的にあり(例: swisstopo 系の実験的レリーフ)。

#### 実装の方向性

- GPU では「各画素から前後 L ステップの RK2/オイラー積分でノイズをサンプル・平均」
  という素朴な並列実装が速い(Fast LIC の逐次最適化は GPU では不利)。
  ElementwiseKernel / RawKernel 1本で書ける。
- halo = ステップ長×ステップ数。`L ≤ MAX_DEPTH`(150px)を上限にクランプし、
  それ以上の積分長はオーバービュー上で実行して合成(hybrid coarse path)。
- 乱数はタイル座標からのハッシュベース(counter-based RNG, 例: Philox/squirrel noise)で
  生成し、タイル分割に依存しない決定的ノイズにする(シームと再現性の両立)。
- パラメータ案: `--vector-field flow|strike|contour`(既定 flow)、
  `--length`(積分半長 px、既定 20)、`--noise-scale`、
  `--composite hillshade|none`(既定 hillshade: 乗算合成した1バンドを出力)。
- ベクトル場の平滑化に A1 の構造テンソル(固有ベクトルは向きの ±180° 曖昧さに強い)を
  使うと品質が上がる — flow モードでも生勾配ではなくテンソル平滑後の場を推奨。

---

### B1. Phase Congruency レリーフ(モノジェニック信号)

**提案: Claude Fable 5(2026-07-06 検討)/ 判断: 採用**

#### 概要

特徴(エッジ・線)を「フーリエ成分の位相が揃う場所」として検出する。
勾配ベース(既存 npr_edges)と違い**振幅不変**: 比高数十 cm の低断層崖・段丘崖が、
山地の大起伏と同じ強度で検出される。低起伏地の活断層・微地形の可視化という、
古典アルゴリズムでは原理的に不可能な絵が出る。

2D への拡張はモノジェニック信号を使う。周波数領域の Riesz 変換
(H1 = iu/|u|, H2 = iv/|u|)をバンドパス(log-Gabor)した DEM に適用し、
偶成分 f と奇成分 (R1, R2) から局所振幅 A = √(f²+R1²+R2²) と局所位相を得て、
スケール横断で位相一致度 PC = Σ W·⌊A·ΔΦ − T⌋ / (ΣA + ε) を計算する。

#### 出典

- Morrone, M.C. & Owens, R.A. (1987) "Feature detection from local energy." *Pattern Recognition Letters* 6.
- Kovesi, P. (1999) "Image Features from Phase Congruency." *Videre* 1(3).— PC の実用定式化(ノイズ補償 T、重み W)。
- Kovesi, P. (2003) "Phase Congruency Detects Corners and Edges." *DICTA 2003*.
- Felsberg, M. & Sommer, G. (2001) "The Monogenic Signal." *IEEE Trans. Signal Processing* 49(12). — Riesz 変換による等方的直交信号。
- 地形応用: 断層崖検出への位相一致の適用は火星・月の地形研究に散見(振幅不変性が理由)。

#### 実装の方向性

- 唯一の FFT 系候補。タイルごとに halo 付きで cuFFT(cupy.fft)→ log-Gabor ×
  Riesz を周波数領域で乗算 → 逆 FFT。最低周波数のフィルタ波長が halo を決める:
  波長 ≲ 100px のスケールはタイル内で処理、それ以上は hybrid coarse path で
  オーバービューに委譲(オーバービュー上では同じ波長が小さい px 数になる)。
- `--radii` を log-Gabor の中心波長(px)系列として解釈。既定は 4,8,16,...,
  最大は MAX_DEPTH と DEM サイズから自動決定。
- パラメータ案: `--noise-t`(ノイズ閾値 T、グローバル統計プリパスで最小スケールの
  振幅分布から自動推定 — Kovesi 原法の median ベース推定)、`--sigma-onf`(log-Gabor
  帯域、既定 0.55)。
- 出力: PC ∈ [0,1] の1バンド。オプションで局所位相の符号による
  「凸(尾根様)/凹(谷様)」の発散マップ(`--feature-type edge|ridge|both`)。
- FFT のタイル境界: halo を波長の ~2 倍取り、窓関数はかけず halo 破棄で対応
  (reflect パディングと併用)。シーム検査は openness のシームテストの流儀に倣う。

---

### D1. TV 構造-テクスチャ分解(Total Variation Decomposition)

**提案: Claude Fable 5(2026-07-06 検討)/ 判断: 採用**

#### 概要

ROF モデル `min_u TV(u) + (λ/2)‖u−z‖²` を解くと、DEM が「輪郭(崖・遷急線)を保った
区分平滑成分 u」と「微細テクスチャ v = z − u」に分解される。v を描画すると
topousm(ガウシアン USM)と似た目的の絵になるが、決定的な違いとして
**急崖の周囲にハロー(オーバーシュート)が出ない**。ガウシアンは崖を鈍らせるため
残差に崖の亡霊が滲むが、TV は崖を u 側に保持するので、v は純粋な微細地形
(耕作痕・小崩壊・粗さの変化)だけになる。λ を段階的に変えれば
エッジ保存型のマルチスケール分解(TV スケール空間)にもなる。

#### 出典

- Rudin, L., Osher, S. & Fatemi, E. (1992) "Nonlinear total variation based noise removal algorithms." *Physica D* 60. — ROF モデル。
- Chambolle, A. & Pock, T. (2011) "A first-order primal-dual algorithm for convex problems with applications to imaging." *JMIV* 40(1). — GPU 向き主双対解法。
- Aujol, J.-F. et al. (2006) "Structure-Texture Image Decomposition — Modeling, Algorithms, and Parameter Selection." *IJCV* 67(1).
- Chan, T.F. & Esedoglu, S. (2005) "Aspects of total variation regularized L¹ function approximation." *SIAM J. Appl. Math.* 65(5). — TV-L1(コントラスト非依存のスケール選択性)。

#### 実装の方向性

- Chambolle-Pock 主双対法。1反復 = 前進差分×2 + 射影で、全て CuPy の
  要素演算+roll。反復数 N(既定 ~100-200)に対し情報伝播は高々 N px なので、
  **halo = N を MAX_DEPTH=150 でクランプ**すれば理論的に厳密なタイル整合が取れる
  (反復 PDE 系だがサポート有限なのが採用可能な理由)。
- 大きい構造スケール(λ 小)は伝播距離が足りなくなるため、hybrid coarse path で
  オーバービュー上で解いて合成する。`--radii` は「除去したい構造の目安スケール」として
  受け、λ に変換する(TV-L1 なら λ とスケールの関係が明確: 直径 < 2/λ の構造が消える)。
- パラメータ案: `--fidelity l2|l1`(既定 l1 — スケール選択がコントラスト非依存で
  地形向き)、`--iterations`(既定 150)、出力 `--component texture|structure`(既定 texture)。
- 出力: v は発散型(0 中心)1バンド。既存の percentile 正規化+gamma に載せる。
- 検証: 人工 DEM(段差+正弦波テクスチャ)で「段差が texture 側に漏れない」ことを
  ユニットテスト化。タイル境界一致テストは blur のパターンに倣う。

---

### E1. Scale Drift(スケール漂流場)

**提案: Claude Fable 5(2026-07-06 検討・オリジナル)/ 判断: 採用**

#### 概要

FujiShaderGPU オリジナル。ガウシアンスケール空間 L(x; σ_i) の**隣接レベル間で
オプティカルフロー**(Lucas-Kanade)を計算し、特徴がスケール増加とともに
「どちらへ動くか」というベクトル場(漂流場)を得る。

理論的背景はスケール空間の deep structure(Koenderink): 対称な地形では極値・稜線は
スケールを上げても動かないが、**非対称な地形(ケスタ、傾動地塊、非対称谷、
片側侵食の丘陵)では特徴点が系統的にドリフトする**。ドリフト方向を Hue、
大きさを Saturation/Value にした出力は「地形の非対称性=侵食・変形の方向履歴」を
色で示す。既存の scale_space_surprise がスケール間変化の**スカラー量**を取るのに
対し、これはその**ベクトル版**であり、方向情報を持つ点で本質的に異なる。
名称・定式化とも本検討によるオリジナルで、先行文献は未確認(実装時に要再調査)。

#### 出典

- (直接の先行研究なし — オリジナル。以下は理論的基盤)
- Koenderink, J.J. (1984) "The structure of images." *Biological Cybernetics* 50. — スケール空間の deep structure。
- Lindeberg, T. (1994) *Scale-Space Theory in Computer Vision*. Kluwer. — 極値のスケール間追跡(drift velocity の解析式 §8 付近)。
- Lucas, B.D. & Kanade, T. (1981) "An Iterative Image Registration Technique..." *IJCAI '81*.

#### 実装の方向性

- 各隣接スケール対 (σ_i, σ_{i+1}) について Lucas-Kanade 1 ステップ:
  `d_i = −(G_w * ∇L∇Lᵀ)⁻¹ (G_w * ∇L·L_t)`(L_t = L_{i+1} − L_i、窓 w ~ σ_i)。
  構造テンソル(A1 と同じ部品!)の逆行列を使うため `_impl_structure_tensor.py` を共有。
- 悪条件(λ2 ≈ 0、平坦地)では d を 0 に減衰させる(λ2/(λ2+ε) の重み)。
- スケール合成: `--radii` = σ 系列。各対のドリフトベクトルを `--weights` 由来の
  対重み(scale_space_surprise の pair-weights と同じ流儀)で加算。
- halo = ~5σ_max + LK 窓。大スケールは hybrid coarse path(既存の
  `_smooth_for_radius` / overview 機構をそのまま使える)。
- 出力モード案: `--output hsv|magnitude|divergence`。
  - hsv: 方向を Hue(こちらは 360° 周期なのでそのまま)、大きさを Sat にした RGB。
  - magnitude: |d| の1バンド(scale_space_surprise の対照として)。
  - divergence: ∇·d — ドリフトの湧き出し/吸い込みで、尾根の「押され方」を示す実験的指標。
- 検証: 人工の非対称ガウシアン丘(片側急・片側緩)でドリフトが緩斜面側を向くこと、
  対称丘でほぼゼロになることをユニットテスト化。
- 論文化の可能性あり。命名は "Scale-Drift Field" を仮とする。

---

## 先送り候補(概要のみ)

### B2. 方向性ウェーブレット(Gabor / Shearlet)エネルギー地図

**提案: Claude Fable 5(2026-07-06)/ 判断: 先送り(A1 と目的重複)**

N 方向 × M スケールの Gabor(または shearlet)フィルタバンク応答エネルギーで
方向性ファブリックを可視化する。出典: Jain & Farrokhnia (1991) *Pattern Recognition* 24(12);
Kutyniok & Labate (2012) *Shearlets*. Birkhäuser。
A1(構造テンソル)が同じ目的をより安価に達するため先送り。曲線状構造の
スケール-方向同時分解が必要になった時に再検討。

### C1. Persistence(位相的持続性)マップ

**提案: Claude Fable 5(2026-07-06)/ 判断: 先送り(GPU・タイル分割との相性)**

persistent homology でピーク/凹地の「プロミネンス」を全画素に与え、ノイズ起伏と
本質的地形単位を分離する。出典: Edelsbrunner, Letscher & Zomorodian (2002)
"Topological Persistence and Simplification." *Discrete Comput. Geom.* 28。
union-find 系の逐次処理が GPU/タイルと相性最悪。採る場合はオーバービュー上で
グローバル計算→フル解像度転写のハイブリッド構成。

### E2. Eigenterrain 疑似カラー

**提案: Claude Fable 5(2026-07-06・オリジナル)/ 判断: 先送り**

局所パッチ(例 16×16)を粗解像度で PCA(バッチ SVD)し、上位3主成分への射影を
RGB 化する教師なし地形テクスチャ埋め込み。eigenfaces(Turk & Pentland 1991)の
地形版。火山地・カルスト・地すべり地形が教師なしで色分けされる見込み。
基底学習という「状態」を持つため既存のステートレスなパイプライン設計と相性が悪く先送り。

### E3. 侵食時間レリーフ(Erosion-Time Relief)

**提案: Claude Fable 5(2026-07-06・オリジナル)/ 判断: 先送り**

平均曲率流ないし簡易 stream-power 則(E = K·A^m·S^n; Whipple & Tucker 1999, *JGR* 104)で
DEM を仮想侵食し、各画素の標高が閾値以上変化するまでの仮想時間をトーン化。
尾根の鋭さ・地形の若さの指標。長時間反復 PDE のためタイル境界整合が困難
(D1 と違い情報が流路沿いに長距離伝播する)。粗解像度実行が現実解だが優先度低。

---

## 追記テンプレート(新しい提案はこの形式で追加)

```markdown
### <ID>. <名称>

**提案: <AI名/人名> (<日付>)/ 判断: 検討中|採用|先送り**

#### 概要
(何を計算し、何が見えるか。既存アルゴリズムとの差別化を必ず1文入れる)

#### 出典
(原典論文・書籍。オリジナルなら理論的基盤を挙げ「オリジナル」と明記)

#### 実装の方向性
(halo/MAX_DEPTH=150 への収まり方、--radii 体系との統合、正規化、出力バンド構成)
```

---

*初版: 2026-07-06 — Claude Fable 5 による検討に基づく。*
*採用/先送りの判断: プロジェクトオーナー(2026-07-06)。*
