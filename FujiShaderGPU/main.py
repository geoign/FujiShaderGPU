"""
FujiShaderGPU/main.py
"""
from .core.system_config import check_gdal_environment
from .core.tile_processor import process_dem_tiles
import os, argparse

def main():
    parser = argparse.ArgumentParser(
        description="DEM→RVI超高速処理（CuPy+マルチスケール+QGIS最適化COG出力）"
    )
    parser.add_argument("input_cog", help="入力DEM（COG形式）")
    parser.add_argument("output_cog", help="出力RVI COG")
    parser.add_argument("--tmp_dir", default="tiles_tmp", help="一時ディレクトリ")
    parser.add_argument("--tile_size", type=int, help="タイルサイズ（自動検出）")
    parser.add_argument("--padding", type=int, help="パディング（自動計算）")
    parser.add_argument("--sigma", type=float, default=10.0, help="Gaussianσ値")
    parser.add_argument("--max_workers", type=int, help="並列数（自動検出）")
    parser.add_argument("--nodata_threshold", type=float, default=1.0, help="NoDataスキップ閾値")
    parser.add_argument("--gpu_type", choices=["rtx4070", "t4", "l4", "a100", "auto"], default="auto", help="GPU種別指定（rtx4070/t4/l4/a100/auto）")
    parser.add_argument("--single_scale", action="store_true", help="シングルスケール強制")
    parser.add_argument("--pixel_size", type=float, help="ピクセルサイズ（自動検出）")
    parser.add_argument("--no_auto_scale", action="store_true", help="自動スケール分析無効")
    # ★ COG生成のみオプションを追加
    parser.add_argument("--cog-only", action="store_true", help="既存タイルからCOG生成のみ実行")
    
    args = parser.parse_args()

    # ★ COG生成のみモードの場合、入力ファイルチェックをスキップ
    if not args.cog_only:
        # ファイル存在チェック
        if not os.path.exists(args.input_cog):
            print(f"エラー: 入力ファイルが存在しません: {args.input_cog}")
            exit(1)
    else:
        # COG生成のみの場合、入力ファイルパスはダミー値を設定
        if not args.input_cog or not os.path.exists(args.input_cog):
            args.input_cog = "dummy_input.tif"  # 実際には使用されない

    # 設定決定
    multiscale_mode = not args.single_scale
    auto_scale_analysis = not args.no_auto_scale

    if args.cog_only:
        print("=== COG生成のみモード ===")
        print(f"タイルディレクトリ: {args.tmp_dir}")
        print(f"出力: {args.output_cog}")
    else:
        print(f"入力: {args.input_cog}")
        print(f"出力: {args.output_cog}")
        print(f"モード: {'マルチスケール' if multiscale_mode else 'シングルスケール'}")

    check_gdal_environment()

    try:
        process_dem_tiles(
            input_cog_path=args.input_cog,
            output_cog_path=args.output_cog,
            tmp_tile_dir=args.tmp_dir,
            tile_size=args.tile_size,
            padding=args.padding,
            sigma=args.sigma,
            max_workers=args.max_workers,
            nodata_threshold=args.nodata_threshold,
            gpu_type=args.gpu_type,
            multiscale_mode=multiscale_mode,
            pixel_size=args.pixel_size,
            auto_scale_analysis=auto_scale_analysis,
            cog_only=args.cog_only
        )
        print("✓ 処理完了")
    except Exception as e:
        print(f"✗ エラー: {e}")
        exit(1)

if __name__ == "__main__":
    main()
