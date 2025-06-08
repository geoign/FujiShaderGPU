"""
FujiShaderGPU/__main__.py
統一エントリーポイント - OS環境を自動検出して適切な処理系を選択
"""
import sys
import platform
import warnings


def main():
    """統一エントリーポイント"""
    # OS検出
    system = platform.system().lower()
    
    if system == "linux":
        # Linux環境: Dask-CUDA処理系を使用
        try:
            from .cli.linux_cli import LinuxCLI
            cli = LinuxCLI()
        except ImportError as e:
            print(f"エラー: Linux環境用の依存関係が不足しています: {e}")
            print("以下のコマンドでインストールしてください:")
            print("pip install FujiShaderGPU[linux]")
            sys.exit(1)
    
    elif system == "windows":
        # Windows環境: タイルベース処理系を使用
        from .cli.windows_cli import WindowsCLI
        cli = WindowsCLI()
    
    elif system == "darwin":
        # macOS: 現時点ではWindowsと同じ処理系を使用
        warnings.warn("macOS環境は実験的サポートです。一部機能が制限される可能性があります。")
        from .cli.windows_cli import WindowsCLI
        cli = WindowsCLI()
    
    else:
        print(f"エラー: サポートされていないOS: {system}")
        sys.exit(1)
    
    # CLIの実行
    try:
        cli.run()
    except KeyboardInterrupt:
        print("\n処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()