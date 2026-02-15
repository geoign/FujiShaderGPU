# dask_shared.py モノリス分割リファクタリング計画

**作成日**: 2026-02-15  
**完了日**: 2026-02-15  
**ステータス**: ✅ **全Phase完了**  
**対象ファイル**: `FujiShaderGPU/algorithms/dask_shared.py`  
**元の行数**: 3,264行（約140KB）  
**最終行数**: 177行（約5.8KB） — **94.6%削減**  
**アウトライン項目数**: 135 → 再エクスポートのみ

---

## 1. 問題の詳細

`dask_shared.py` は全 Dask アルゴリズムの実装と共有ユーティリティを1ファイルに集約しており、以下の問題を抱えている：

1. **可読性の低下**: 3,264行の単一ファイルは、どの関数がどのアルゴリズムに属するかの追跡が困難。
2. **差分レビューの困難**: gitでの変更追跡時、無関係なアルゴリズムの変更が同一ファイルに混在する。
3. **循環的な名前依存**: `algorithms/dask/*.py`（例: `dask/rvi.py`）がすでに各アルゴリズム用のモジュールファイルとして存在するが、中身は1行の再エクスポートのみ：
   ```python
   # algorithms/dask/rvi.py（全5行）
   from ..dask_shared import RVIAlgorithm
   __all__ = ["RVIAlgorithm"]
   ```
4. **テスト困難**: 個別アルゴリズムの単体テストが全体ファイルのインポートを要求する。

---

## 2. 現在のファイル構造マップ

| 行範囲 | 内容 | 推定行数 | 分割先候補 |
|--------|------|---------|-----------|
| 1-16 | インポート・定数定義 | 16 | `_base.py`（共通定義） |
| 17-23 | `Constants` クラス | 7 | `_base.py` |
| 26-78 | `classify_resolution()`, `get_gradient_scale_factor()` | 53 | `_base.py` |
| 84-95 | `DaskAlgorithm` ABC | 12 | `_base.py` |
| 101-349 | **NaN処理ユーティリティ** | 249 | `_nan_utils.py` |
|  | `handle_nan_with_gaussian()` | | |
|  | `handle_nan_with_uniform()` | | |
|  | `handle_nan_for_gradient()` | | |
|  | `_normalize_spatial_radii()` | | |
|  | `_resolve_spatial_radii_weights()` | | |
|  | `_combine_multiscale_dask()` | | |
|  | `_smooth_for_radius()` | | |
|  | `_radius_to_downsample_factor()` | | |
|  | `_downsample_nan_aware()` / `_upsample_to_shape()` | | |
|  | `restore_nan()` | | |
| 354-533 | **グローバル統計ユーティリティ** | 180 | `_global_stats.py` |
|  | `determine_optimal_downsample_factor()` | | |
|  | `compute_global_stats()` | | |
|  | `apply_global_normalization()` | | |
| 541-611 | **正規化関数群**（per-algorithm） | 71 | `_normalization.py` |
|  | `rvi_stat_func()` / `rvi_norm_func()` | | |
|  | `freq_stat_func()` / `freq_norm_func()` | | |
|  | `npr_stat_func()` / `lrm_stat_func()` / `tpi_norm_func()` | | |
| 617-805 | **RVIAlgorithm** | 189 | `_impl_rvi.py` |
|  | `high_pass()`, `compute_rvi_efficient_block()` | | |
|  | `multiscale_rvi()`, `RVIAlgorithm` クラス | | |
| 811-987 | **HillshadeAlgorithm** | 177 | `_impl_hillshade.py` |
|  | `compute_hillshade_block()`, `compute_hillshade_spatial_block()` | | |
|  | `HillshadeAlgorithm` クラス | | |
| 988-1151 | **VisualSaliencyAlgorithm** | 164 | `_impl_visual_saliency.py` |
| 1157-1366 | **NPREdgesAlgorithm** | 210 | `_impl_npr_edges.py` |
| 1378-1596 | **AmbientOcclusionAlgorithm** | 219 | `_impl_ambient_occlusion.py` |
| 1602-1673 | **LRMAlgorithm** | 72 | `_impl_lrm.py` |
| 1679-1873 | **OpennessAlgorithm** | 195 | `_impl_openness.py` |
| 1875-1980 | **SlopeAlgorithm** | 106 | `_impl_slope.py` |
| 1986-2224 | **SpecularAlgorithm** | 239 | `_impl_specular.py` |
| 2230-2366 | **AtmosphericScatteringAlgorithm** | 137 | `_impl_atmospheric.py` |
| 2372-2520 | **MultiscaleDaskAlgorithm** | 149 | `_impl_multiscale_terrain.py` |
| 2526-2616 | **FrequencyEnhancementAlgorithm** | 91 | `_impl_frequency.py` |
| 2622-2748 | **CurvatureAlgorithm** | 127 | `_impl_curvature.py` |
| 2754-3060 | **FractalAnomalyAlgorithm** | 307 | `_impl_fractal_anomaly.py` |
| 3066-3107 | **ScaleSpaceSurpriseAlgorithm** | 42 | `_impl_scale_space_surprise.py` |
| 3110-3225 | **MultiLightUncertaintyAlgorithm** | 116 | `_impl_multi_light_uncertainty.py` |
| 3232-3249 | `ALGORITHMS` レジストリ | 18 | `dask_shared.py`（残置） |

---

## 3. 推奨する分割計画

### Phase 1: 共通基盤の抽出（リスク低）

新規作成するファイル：

```
algorithms/
├── dask_shared.py          ← 縮小版（再エクスポート + ALGORITHMS レジストリ）
├── _base.py                ← Constants, DaskAlgorithm ABC, classify_resolution, get_gradient_scale_factor
├── _nan_utils.py           ← NaN処理・空間モード・ダウンサンプル・リストア関数群
├── _global_stats.py        ← compute_global_stats, apply_global_normalization, downsample_factor計算
├── _normalization.py       ← rvi_stat_func, freq_stat_func, npr_stat_func, lrm_stat_func 等
```

**作業量**: 約550行を移動。共通関数群のため、他のアルゴリズム実装に変更を加えることなく分離可能。

### Phase 2: 大型アルゴリズムの個別ファイル化（リスク中）

最もサイズが大きいアルゴリズムから：

| 優先度 | アルゴリズム | 行数 | 分割先 |
|--------|------------|------|--------|
| 1 | FractalAnomaly | 307 | `_impl_fractal_anomaly.py` |
| 2 | Specular | 239 | `_impl_specular.py` |
| 3 | AmbientOcclusion | 219 | `_impl_ambient_occlusion.py` |
| 4 | NPREdges | 210 | `_impl_npr_edges.py` |
| 5 | Openness | 195 | `_impl_openness.py` |
| 6 | RVI | 189 | `_impl_rvi.py` |

### Phase 3: 残り全アルゴリズムの分離（リスク中）

残りのアルゴリズム（Hillshade, Slope, Curvature, MultiscaleTerrain, VisualSaliency, AtmosphericScattering, FrequencyEnhancement, ScaleSpaceSurprise, MultiLightUncertainty）を個別ファイルに移動。

### Phase 4: dask_shared.py を再エクスポートハブ化

最終的に `dask_shared.py` は以下のような構造になる：

```python
"""
FujiShaderGPU/algorithms/dask_shared.py
後方互換性のための再エクスポートハブ。
全アルゴリズム実装は _impl_*.py に分離済み。
"""
# 共通インフラ（外部からインポートされている公開シンボル）
from ._base import Constants, DaskAlgorithm, classify_resolution, get_gradient_scale_factor
from ._nan_utils import (
    handle_nan_with_gaussian, handle_nan_with_uniform,
    handle_nan_for_gradient, restore_nan,
    _normalize_spatial_radii, _resolve_spatial_radii_weights,
    _combine_multiscale_dask, _smooth_for_radius,
    _radius_to_downsample_factor, _downsample_nan_aware, _upsample_to_shape,
)
from ._global_stats import compute_global_stats, apply_global_normalization, determine_optimal_downsample_factor
from ._normalization import rvi_stat_func, rvi_norm_func, freq_stat_func, freq_norm_func, npr_stat_func, lrm_stat_func, tpi_norm_func

# 各アルゴリズム実装
from ._impl_rvi import RVIAlgorithm, compute_rvi_efficient_block, high_pass, multiscale_rvi
from ._impl_hillshade import HillshadeAlgorithm, compute_hillshade_block, compute_hillshade_spatial_block
# ... 以下同様 ...

# レジストリ
ALGORITHMS = {
    'rvi': RVIAlgorithm(),
    'hillshade': HillshadeAlgorithm(),
    # ...
}
```

重要: **`dask_shared.py` からの全ての公開シンボルの再エクスポートを維持する**。これにより、以下の既存のインポートパスが一切変更なしで動作し続ける：

```python
# tile/rvi.py - 変更不要
from ..dask_shared import RVIAlgorithm as _DaskRVIAlgorithm

# dask/rvi.py - 変更不要
from ..dask_shared import RVIAlgorithm

# dask_registry.py - 変更不要
from .dask_shared import RVIAlgorithm

# core/tile_processor.py - 変更不要
from ..algorithms.dask_shared import compute_rvi_efficient_block, rvi_stat_func
```

---

## 4. 影響範囲の詳細分析

### 4.1 dask_shared.py から外部にインポートされているシンボル

以下のシンボルは別ファイルから `from ..dask_shared import ...` としてインポートされており、再エクスポートが必須：

| シンボル | インポート元 | 用途 |
|---------|-----------|------|
| `RVIAlgorithm` | `dask/rvi.py`, `tile/rvi.py` | レジストリ・ブリッジ |
| `HillshadeAlgorithm` | `dask/hillshade.py`, `tile/hillshade.py` | 同上 |
| `SlopeAlgorithm` | `dask/slope.py`, `tile/slope.py` | 同上 |
| `SpecularAlgorithm` | `dask/specular.py`, `tile/specular.py` | 同上 |
| `AtmosphericScatteringAlgorithm` | `dask/atmospheric_scattering.py`, `tile/atmospheric_scattering.py` | 同上 |
| `MultiscaleDaskAlgorithm` | `dask/multiscale_terrain.py`, `tile/multiscale_terrain.py` | 同上 |
| `CurvatureAlgorithm` | `dask/curvature.py`, `tile/curvature.py` | 同上 |
| `VisualSaliencyAlgorithm` | `dask/visual_saliency.py`, `tile/visual_saliency.py` | 同上 |
| `NPREdgesAlgorithm` | `dask/npr_edges.py`, `tile/npr_edges.py` | 同上 |
| `AmbientOcclusionAlgorithm` | `dask/ambient_occlusion.py`, `tile/ambient_occlusion.py` | 同上 |
| `LRMAlgorithm` | `dask/lrm.py`, `tile/lrm.py` | 同上 |
| `OpennessAlgorithm` | `dask/openness.py`, `tile/openness.py` | 同上 |
| `FractalAnomalyAlgorithm` | `dask/fractal_anomaly.py`, `tile/fractal_anomaly.py` | 同上 |
| `ScaleSpaceSurpriseAlgorithm` | `dask/scale_space_surprise.py`, `tile/scale_space_surprise.py` | 同上 |
| `MultiLightUncertaintyAlgorithm` | `dask/multi_light_uncertainty.py`, `tile/multi_light_uncertainty.py` | 同上 |
| `compute_rvi_efficient_block` | `core/tile_processor.py` | グローバルRVI統計用 |
| `rvi_stat_func` | `core/tile_processor.py` | 同上 |
| `FractalAnomalyAlgorithm` | `core/tile_processor.py` | radii自動決定用 |
| `compute_global_stats` | `dask_processor.py` 経由 | グローバル統計 |

### 4.2 内部依存グラフ（アルゴリズム → 共通関数）

ほぼ全てのアルゴリズム実装が以下の共通関数に依存：
- `handle_nan_with_gaussian()`, `restore_nan()` ← 全アルゴリズムで使用
- `_resolve_spatial_radii_weights()`, `_combine_multiscale_dask()` ← spatial モード対応アルゴリズムで使用
- `_smooth_for_radius()` ← spatial ブロック関数で使用
- `compute_global_stats()`, `apply_global_normalization()` ← 正規化対応アルゴリズムで使用
- `Constants.DEFAULT_GAMMA` ← ガンマ補正使用アルゴリズムで使用
- `classify_resolution()` ← RVI, FractalAnomaly で使用

---

## 5. リスクと注意事項

1. **後方互換性**: `dask_shared.py` は外部から多数のシンボルがインポートされている。分割後も全て再エクスポートする必要がある。
2. **循環インポート**: `_impl_rvi.py` が `_nan_utils.py` と `_global_stats.py` をインポートし、`_global_stats.py` がアルゴリズム固有の stat 関数を引数として受け取る設計のため、循環は発生しない（stat 関数は callable として渡される）。
3. **cupyx.scipy.ndimage.gaussian_filter**: `_nan_utils.py` にインポートが集約される。現在は `from cupyx.scipy.ndimage import gaussian_filter` がファイルの先頭近くにある。
4. **テスト**: 分割後、全アルゴリズムの動作確認テストが必要。GPU環境がないとテスト不可。
5. **コメントアウトされたコード**: 3251-3259行に `AspectAlgorithm` のスケルトンがコメントアウトされている。分割時に削除するか、別途issueとして管理する。

---

## 6. 推奨実行順序

1. ✅ **Phase 1** 完了 — 共通基盤の抽出（550行移動）
2. ✅ **Phase 2** 完了 — 大型アルゴリズム6種を個別ファイル化（1,352行移動）
3. ✅ **Phase 3** 完了 — 残り全アルゴリズム9種を個別ファイル化
4. ✅ **Phase 4** 完了 — dask_shared.py を177行の再エクスポートハブ化

### 最終ファイル構成（19モジュール）

```
algorithms/
├── dask_shared.py                    # 177行 — 再エクスポートハブ + ALGORITHMSレジストリ
├── _base.py                          # Constants, DaskAlgorithm, classify_resolution
├── _nan_utils.py                     # NaN処理、空間スムージング
├── _global_stats.py                  # グローバル統計、正規化適用
├── _normalization.py                 # アルゴリズム別統計/正規化関数
├── _impl_rvi.py                      # RVI (Ridge-Valley Index)
├── _impl_hillshade.py                # Hillshade
├── _impl_slope.py                    # Slope
├── _impl_specular.py                 # Specular
├── _impl_atmospheric_scattering.py   # Atmospheric Scattering
├── _impl_multiscale_terrain.py       # Multiscale Terrain
├── _impl_curvature.py                # Curvature
├── _impl_visual_saliency.py          # Visual Saliency
├── _impl_npr_edges.py                # NPR Edges
├── _impl_ambient_occlusion.py        # Ambient Occlusion
├── _impl_lrm.py                      # LRM (Local Relief Model)
├── _impl_openness.py                 # Openness
├── _impl_fractal_anomaly.py          # Fractal Anomaly
└── _impl_experimental.py             # ScaleSpaceSurprise + MultiLightUncertainty
```

全Phaseの健全性チェック（コンパイル、インポート、関数動作）合格済み。後方互換性を完全に維持。
