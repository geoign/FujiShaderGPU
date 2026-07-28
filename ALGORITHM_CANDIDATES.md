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
| G1 | Geomorphons 地形形態分類 | パターン認識/地形計測 | 検討中 | Kimi (Moonshot AI) (2026-07-18) |
| F1 | マルチスケール形態学トップハット/DMP | 数学形態学/次数統計 | 検討中 | Kimi (Moonshot AI) (2026-07-18) |
| F2 | 局所標高ランク(順位レリーフ) | 次数統計 | 検討中 | Kimi (Moonshot AI) (2026-07-18) |
| H1 | 方向性バリオグラム粗さ異方性 | 地統計/第二級統計 | 検討中 | Kimi (Moonshot AI) (2026-07-18) |
| E4 | パッチ再帰性マップ | オリジナル/非局所 | 検討中 | Kimi (Moonshot AI) (2026-07-18) |
| I1 | HAND 水文正規化レリーフ | 水文学/大域routing | 先送り | Kimi (Moonshot AI) (2026-07-18) |

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

## 検討中候補(詳細、2026-07-18 追記)

2026-07-06 採用分の実装で「方向性・位相・変分法・スケール間ダイナミクス」の空白は
概ね埋まった。残る空白の族は **次数統計/数学形態学(min-max-rank)**、
**パターン分類(カテゴリ出力)**、**第二級統計/地統計(分散・共分散)**、
**非局所パッチ法**、および大域 routing を要する**水文学系**である。
以下はこれらの族からの追加候補(Kimi (Moonshot AI) による検討)。

### G1. Geomorphons(地形形態パターン分類)

**提案: Kimi (Moonshot AI)(2026-07-18 検討)/ 判断: 検討中**

#### 概要

8方向の見通し(LOS)走査で、距離 L 以内の天頂角 ψ と天底角 ν の最大値を取り、
閾値角 t に対し各方向を三値(+1: 周囲より高い / 0: 同程度 / −1: 低い)に分類する。
8方向の三値パターンを回転・鏡映の同値類にまとめ、10地形要素
(flat / peak / ridge / shoulder / spur / slope / hollow / footslope / valley / pit)
にマッピングする。

既存の curvature(微分型の連続スカラー)とも openness(角度の方向平均)とも違い、
**パターンマッチングによる離散的な地形形態「分類」**であり、カテゴリという
新しい出力次元を持つ。連続場アルゴリズムでは混ざり合う「尾根肩」「谷肩」「沢頭」
などの地形単位が、塗り分け可能な離散ラベルとして得られる。クラス色塗り図のほか、
峰/谷の二値マスクとして既存アルゴリズムの重み・マスクにも流用できる。

#### 出典

- Jasiewicz, J. & Stepinski, T.F. (2013) "Geomorphons — a pattern recognition approach to classification and mapping of landforms." *Geomorphology* 182. — 原法。GRASS GIS r.geomorphon として広く実装済み。
- 代替の分類系: Weiss, A. (2001) "Topographic Position and Landforms Analysis." *ESRI User Conference 2001*. — 2スケール TPI + slope による10区分。
- LOS 走査の背景: Yokoyama, R. et al. (2002) openness(既存実装)と同じ8方向走査機構。

#### 実装の方向性

- 8方向走査は openness の機構(方向ごとの逐次最大角更新、距離 L で打ち切り)を
  流用できる。相違は openness が角度を平均するのに対し、こちらは天頂・天底の
  **最大角のみ**保持して三値化する点。halo = L(既定 30–60px で十分なことが多く、
  MAX_DEPTH=150 に収まる)。
- 三値パターン 3^8 = 6561 → 回転/鏡映同値類 498 → 10 クラスへのルックアップは
  事前計算して定数テーブル化し、CuPy の LUT 参照で完結させる。
- パラメータ案: `--geomorphon-t`(閾値角、既定 1°)、`--lookup-distance`(既定 40px)。
- 出力: クラス番号 0–9 の1バンド。**percentile 正規化をバイパスするカテゴリ出力の
  特例が必要**(前例がないため要対応)。連続場版としてパターンの flatness 度などを
  返すスカラーモードもオプションで用意すると既存経路に乗せやすい。
- シーム: 走査長 L ≤ halo なら厳密にタイル整合。NoData 縁は openness と同じ流儀。

---

### F1. マルチスケール形態学トップハット / 差分形態プロファイル(DMP)

**提案: Kimi (Moonshot AI)(2026-07-18 検討)/ 判断: 検討中**

#### 概要

半径 r の平坦構造要素による opening γ_r(z)(侵食→膨張)と closing φ_r(z) を計算し、
white top-hat `WTH_r = z − γ_r(z)`(r より小さい凸要素の高さ)と
black top-hat `BTH_r = φ_r(z) − z`(凹要素の深さ)を得る。r を段階的に変えた
WTH/BTH 列が「形態プロファイル」で、その差分(DMP)が最大になる r が
**その地点の地形要素の特性スケール**を示す。

ガウシアン USM(topousm)や TV 分解は線形/norm ベースの平滑との残差であり、
急崖では基準面そのものが動く。min/max 演算に基づく opening は
**基準面が「谷を埋めた」包絡面**になるため、突出部(火口丘・砂丘・畝状構造・
リッジ)の大きさと高さが分離して定量できる。次数統計(min/max)という
現行実装に皆無の数学的族であり、ハローも出ない。

#### 出典

- Serra, J. (1982) *Image Analysis and Mathematical Morphology*. Academic Press. — 原典。
- Pesaresi, M. & Benediktsson, J.A. (2001) "A new approach for the morphological segmentation of high-resolution satellite imagery." *IEEE Trans. Geosci. Remote Sensing* 39(2). — DMP(差分形態プロファイル)。
- van Herk, M. (1992) "A fast algorithm for local minimum and maximum filters on rectangular and octagonal kernels." *Pattern Recognition Letters* 13(7); Gil, J. & Werman, M. (1993). — O(1)/画素の分離可能 min-max フィルタ。
- Soille, P. (2003) *Morphological Image Analysis: Principles and Applications*. Springer. — DEM への形態学適用の総説。

#### 実装の方向性

- van Herk–Gil-Werman で行/列に分離(円盤近似は8方向走査の複合で近似可)。
  各方向1パス・画素あたり数比較で GPU 負荷は極めて軽い。halo = r_max
  (150px にクランプ)。
- `--radii` を構造要素半径系列として解釈、`--weights` は DMP 合成に使用。
- 出力モード案: `--mh-output wth|bth|both|scale`(既定 both: WTH−BTH の発散マップ)。
  `scale` は DMP 最大スケールの1バンドで、既存のどれとも異なる
  「地形の目の粗さの定量的地図」になる。
- 正規化: WTH/BTH は非負で既存の percentile 正規化に素直に載る。scale 出力は
  r_max で [0,1] に正規化。
- 検証: 既知半径・既知高さの人工突起列で scale 出力がその半径を返すことを
  ユニットテスト化。

---

### F2. 局所標高ランク(順位レリーフ / Percentile Relief)

**提案: Kimi (Moonshot AI)(2026-07-18 検討)/ 判断: 検討中**

#### 概要

窓内の標高分布に対する自画素の順位 `P(x) = #{z < z(x)} / N` を [0,1] で与える。
TPI(z − 局所平均)が「平均からの差」なのに対し、これは**分布中の順位**であり、
外れ値・長尾分布に頑健かつ振幅不変。高山の稜線も低地の微小な堤防も
「局所的に高い場所」として同じ 1.0 付近に描かれる。出力は標高への厳密な
局所ヒストグラム平坦化(CLAHE の平滑化なし版)に相当する。

openness が「見通し角度」、F1 が「包絡面からの距離」という幾何ベースの凸凹度
なのに対し、これは純粋な**順序統計量**ベースの相対高度であり、
標高値そのものの分布形状を一切仮定しない。

#### 出典

- Pizer, S.M. et al. (1987) "Adaptive histogram equalization and its variations." *Computer Vision, Graphics, and Image Processing* 39(3). — CLAHE(局所順位マッピング)。
- Gallant, J.C. & Dowling, T.I. (2003) "A multiresolution index of valley bottom flatness for mapping depositional areas." *Water Resources Research* 39(12). — 局所標高順位を谷底平野指標(MrVBF)の成分として使用(本候補はその純粋な順位場版で、流路解析・解像度ピラミッドには依存しない)。

#### 実装の方向性

- 窓 w×w の全ペア比較は O(w²)/画素だが、「シフト画像との比較を box フィルタで
  蓄積」する形で GPU に載る(w=31px・961比較でも実用的、メモリ帯域律速)。
  halo = w/2(≤75px で MAX_DEPTH に収まる)。
- 高速化の代替: 局所平均・分散からのガウス近似順位 Φ((z−μ)/σ)(安いが長尾に弱い)を
  `--rank-mode exact|gauss` で選択可能に。
- 出力: [0,1] 1バンドで正規化パイプラインは素通し可。`--radii` を窓半径系列として
  複数窓の順位の加重平均(マルチスケール順位)にも乗る。
- 検証: 単調スロープ+孤立突起の人工 DEM で、突起画素が ~1.0、周縁が ~0.5 に
  なることをユニットテスト化。

---

### H1. 方向性バリオグラム粗さ異方性(ラグ固定の地統計ファブリック)

**提案: Kimi (Moonshot AI)(2026-07-18 検討)/ 判断: 検討中**

#### 概要

ラグ d(例 4,8,16px)・方向 θ(8方向)のバリオグラム
`γ_d(θ) = ½·E[(z(x) − z(x + d·u_θ))²]` を計算し、方向ごとの粗さのローズ図から
**粗さの異方性比 A = γ_max/γ_min とその方向**を得る。

structure_tensor(A1)が「勾配の向きの揃い方」= **地形フォームの走向**を返すのに
対し、こちらは「どちらの方向に地形が粗いか」= **テクスチャの異方性**。
fractal_anomaly がスケール方向の粗さ変化(等方的)を見るのに対し、こちらは
ラグ固定で方向を見る。砂丘(走向方向に滑らか・横断方向に粗い)、侵食ガリー、
氷食地形など、フォームではなく粗さの配向を持つプロセスの指紋を拾う。
第二級統計(分散・共分散)の族は現行実装に皆無。

拡張案: 方向ごとの γ(d) を d について走らせ、シルに達するレンジや
hole effect の位置から**地形の特性波長**(砂丘間隔・ガリー間隔)の地図を作る。

#### 出典

- Matheron, G. (1963) "Principles of geostatistics." *Economic Geology* 58(8). — バリオグラム原論。
- Herzfeld, U.C. & Higginson, C.A. (1996) "Automated geostatistical seafloor classification: Parameters for zonation, classification, and pattern recognition." *Geo-Marine Letters* 16. — 海底地形の方向性粗さによる分類。
- Trevisani, S., Cavalli, M. & Marchi, L. (2012) "Surface texture analysis of a high-resolution DTM: Interpreting an alpine basin." *Geomorphology* 153–154. — バリオグラム系の地形テクスチャ解析。
- Haralick, R.M. et al. (1973) "Textural features for image classification." *IEEE Trans. SMC* 3(6). — GLCM contrast はバリオグラムのラグ固定版と等価。

#### 実装の方向性

- 計算はシフト差分二乗+窓平均のみ(畳み込み系の既存部品で組める)。
  halo = d_max + 窓半径(MAX_DEPTH に収まりやすい)。
- `--radii` をラグ d 系列として解釈。方向数は 8 固定で十分。
- 出力モード案:
  - `--vg-output anisotropy`: A−1(等方で 0)の1バンド
  - `--vg-output orientation`: 粗さ最大方向(A1 と同じ角度→[0,1) マッピング)
  - HSV 合成はマルチバンド COG 対応後(A1 と同じ判断)
- 正規化: γ を窓内分散で割れば無次元化でき、グローバル統計の percentile
  正規化にも載る。
- 検証: 一方向のみ正弦波を持つ人工縞模様 DEM で、異方性方向が縞に対し
  正しい向きを返すことをユニットテスト化。

---

### E4. パッチ再帰性マップ(非局所自己類似度)

**提案: Kimi (Moonshot AI)(2026-07-18 検討・オリジナル)/ 判断: 検討中**

#### 概要

各画素の p×p パッチを、周囲 R px の探索窓内の全パッチと SSD 比較し、
「自分自身以外にどれだけ似たパッチが存在するか」(再帰スコア =
`exp(−SSD_min/2σ²)` または上位 k 件の重み和)を可視化する。

砂丘・モレーン・耕作痕・畝状地形のように**同じ地形素形が反復出現する
プロセス領域**ではスコアが高く、噴出物堆積や崩壊地のようなカオス的地形では
低い。既存の全手法が「画素とその近傍の局所関係」しか見ないのに対し、
これは**非局所的なパッチ対応**を見る初の候補であり、「反復性」という新しい軸で
プロセスドメインを分割する。地形への適用・命名とも本提案がオリジナル
(実装時に要再調査)。

#### 出典

- (地形適用の直接の先行研究なし — オリジナル。以下は理論的基盤)
- Shechtman, E. & Irani, M. (2007) "Matching local self-similarities across images and videos." *CVPR 2007*. — 局所自己類似度ディスクリプタ。
- Buades, A., Coll, B. & Morel, J.-M. (2005) "A non-local algorithm for image denoising." *CVPR 2005*. — NL-means(パッチ重みの定式化)。
- Efros, A.A. & Leung, T.K. (1999) "Texture synthesis by non-parametric sampling." *ICCV 1999*. — テクスチャ=パッチ反復性という思想。

#### 実装の方向性

- SSD を「p×p カーネルでの box_filter((z − shift(z))²)」として各探索オフセットに
  ついて計算し、オフセット数 (2R+1)² 回の畳み込みで全探索位置をカバー
  (NL-means の標準的 GPU 実装と同型)。p=5, R=12 程度で実用的。
  halo = R + p/2(≪150px)。
- 中心オフセット(自己一致)は除外し、最小距離 r_min > p のリング状探索にすると
  パッチ重複による自明な一致を避けられる。
- 出力: 再帰スコア [0,1] の1バンド。オプションで最良一致の方向(反復の配向、
  砂丘列の配列方向検出に)。
- `--radii` は探索半径 R として解釈するが、ラグではなく探索範囲なので
  weights 合成は不自然。単一スケール運用を既定とする。
- 検証: 周期的ストライプ領域+ランダム領域の合成 DEM で、前者が高スコア・
  後者が低スコアになることをユニットテスト化。

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

### I1. HAND(最近傍排水基準の相対標高)/ 水文正規化レリーフ

**提案: Kimi (Moonshot AI)(2026-07-18)/ 判断: 先送り(C1 と同じ構造的理由)**

フロールーティングで定めた局所排水基準面(最近傍水路セルの標高を流路に沿って
参照)からの相対高度 HAND = z − z_drain。氾濫原・段丘面・扇状地を
「排水基準からの高さ」という水文的に意味のある軸で描き、絶対標高の
グラデーションを剥がした地形可視化ができる。谷底平野・テラス抽出の定番。
出典: Rennó, C.D. et al. (2008) "HAND: A new terrain descriptor using SRTM-DEM:
Mapping terra-firme rainforest environments in Amazonia." *Geomorphology* 98(3–4);
窪地埋めは Barnes, R. et al. (2014) "Priority-flood: An optimal depression-filling
and watershed-labeling algorithm for digital elevation models."
*Computers & Geosciences* 62。
流路網抽出・流量累積・窪地埋めが大域的な優先度フッド/逐次処理を要し、
タイル/GPU と相性が悪い点は C1(persistence)と同じ。採るならオーバービュー上で
グローバル routing → フル解像度へ転写するハイブリッド構成
(排水基準面のフル解像度補間方法が別途課題)。

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
*追記: 2026-07-18 — Kimi (Moonshot AI) による検討に基づき、G1 / F1 / F2 / H1 / E4(検討中)と I1(先送り)を追加。*
