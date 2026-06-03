"""
FujiShaderGPU/prepare.py

Preprocessing CLI: convert any GDAL-readable raster into a FujiShaderGPU-ready
Cloud Optimized GeoTIFF (ZSTD + internal overviews, single-band float32), with
optional NoData void filling.

Usage
-----
    python -m FujiShaderGPU.prepare input.(tif|img|vrt|...) output_cog.tif
    python -m FujiShaderGPU.prepare input.tif out.tif --fill-mode enclosed
    python -m FujiShaderGPU.prepare input.tif out.tif --fill-mode all --force

The FujiShaderGPU main pipeline assumes its input is a COG **with overviews**;
running inputs through this command first guarantees fast spatial/multi-scale
processing and removes NoData filling from the per-tile/per-chunk hot path.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .io.dem_preprocess import FILL_MODES, preprocess_dem_to_cog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m FujiShaderGPU.prepare",
        description=(
            "任意のGDALラスタを FujiShaderGPU 互換COG(overview+ZStd, float32)へ変換し、"
            "必要に応じてNoDataの穴埋めを行う前処理コマンド。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
NoData 穴埋めモード (--fill-mode):
  none      穴埋めしない（NoData を保持）
  enclosed  外縁(ラスタ境界に連結する NoData = 海/データ外)以外の内陸の穴のみ埋める【既定】
  all       外縁を含め全 NoData を埋め、NoData を完全に除去（密ラスタ。3Dモデル等向け）

例:
  python -m FujiShaderGPU.prepare input.tif output_cog.tif
  python -m FujiShaderGPU.prepare input.img output_cog.tif --fill-mode all --force
""",
    )
    parser.add_argument("input", help="入力ラスタ (GDAL が読める任意の形式)")
    parser.add_argument("output", help="出力COG (.tif)")
    parser.add_argument(
        "--fill-mode",
        choices=FILL_MODES,
        default="enclosed",
        help="NoData 穴埋めモード (default: enclosed)",
    )
    parser.add_argument(
        "--coarse-max",
        type=int,
        default=2048,
        help="穴埋め用の粗グリッド最長辺(px)。大きいほど精細だが遅い (default: 2048)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=512,
        help="COG タイル/ブロックサイズ (default: 512)",
    )
    parser.add_argument(
        "--overview-count",
        type=int,
        default=8,
        help="生成する overview レベル数 (default: 8)",
    )
    parser.add_argument(
        "--zstd-level",
        type=int,
        default=1,
        help="ZSTD 圧縮レベル (default: 1)",
    )
    parser.add_argument(
        "--num-threads",
        default="ALL_CPUS",
        help="GDAL の並列スレッド数 (default: ALL_CPUS)",
    )
    parser.add_argument(
        "--nodata",
        type=str,
        default=None,
        help="NoData値を明示指定 (例: -9999, 0, nan)。指定値はNaNに置換してから穴埋め",
    )
    parser.add_argument(
        "--no-detect-nodata",
        dest="detect_nodata",
        action="store_false",
        help="外周の固定値からの未指定NoData自動検出を無効化 (default: 有効)",
    )
    parser.set_defaults(detect_nodata=True)
    parser.add_argument(
        "--nodata-border-fraction",
        type=float,
        default=0.5,
        help="自動検出: 値が外周リングを占める最小割合 (default: 0.5)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="出力が存在する場合も上書き",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="ログレベル (default: INFO)",
    )
    return parser


def _parse_nodata(raw_value):
    """Parse a --nodata string into a float (or None). 'nan' is accepted but is a
    no-op for filling (already NaN); finite values are converted to NaN."""
    if raw_value is None:
        return None
    text = str(raw_value).strip().lower()
    if text in {"nan", "+nan", "-nan"}:
        return float("nan")
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"Invalid --nodata value: {raw_value}") from exc


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("=== DEM 前処理 (COG化 + 穴埋め: %s) ===", args.fill_mode)
    try:
        preprocess_dem_to_cog(
            args.input,
            args.output,
            fill_mode=args.fill_mode,
            coarse_max=args.coarse_max,
            block_size=args.block_size,
            overview_count=args.overview_count,
            zstd_level=args.zstd_level,
            num_threads=args.num_threads,
            overwrite=args.force,
            nodata_override=_parse_nodata(args.nodata),
            detect_nodata=args.detect_nodata,
            nodata_border_fraction=args.nodata_border_fraction,
        )
    except Exception as exc:
        logger.error("前処理に失敗しました: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
