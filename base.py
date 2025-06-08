"""
FujiShaderGPU/cli/base.py
共通CLI基底クラス - 両プラットフォーム共通のインターフェースとパラメータ
"""
import argparse
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import logging


class BaseCLI(ABC):
    """富士シェーダーCLIの基底クラス"""
    
    def __init__(self):
        self.parser = self._create_parser()
        self.logger = logging.getLogger(__name__)
        
    def _create_parser(self) -> argparse.ArgumentParser:
        """共通のパーサーを作成"""
        parser = argparse.ArgumentParser(
            description=self.get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.get_epilog()
        )
        
        # 共通の必須引数
        parser.add_argument("input", help="入力DEM (COG形式)")
        parser.add_argument("output", help="出力ファイル (COG形式)")
        
        # 共通オプション引数
        parser.add_argument(
            "--algorithm", "--algo",
            default="rvi",
            choices=self.get_supported_algorithms(),
            help="使用するアルゴリズム (default: rvi)"
        )
        
        parser.add_argument(
            "--tmp-dir",
            default="tiles_tmp",
            help="一時ファイル用ディレクトリ (default: tiles_tmp)"
        )
        
        parser.add_argument(
            "--sigma",
            type=str,
            help="Gaussian σ値。複数の場合はカンマ区切り (例: 50,100,200)"
        )
        
        parser.add_argument(
            "--pixel-size",
            type=float,
            help="ピクセルサイズ (未指定時は自動検出)"
        )
        
        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="ログレベル (default: INFO)"
        )
        
        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="進捗表示を無効化"
        )
        
        parser.add_argument(
            "--force",
            action="store_true",
            help="出力ファイルが存在する場合も上書き"
        )
        
        # プラットフォーム固有のオプションを追加
        self._add_platform_specific_args(parser)
        
        return parser
    
    @abstractmethod
    def get_description(self) -> str:
        """CLIの説明文を返す"""
        pass
    
    @abstractmethod
    def get_epilog(self) -> str:
        """CLIのエピログ（使用例など）を返す"""
        pass
    
    @abstractmethod
    def get_supported_algorithms(self) -> List[str]:
        """サポートされているアルゴリズムのリストを返す"""
        pass
    
    @abstractmethod
    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        """プラットフォーム固有の引数を追加"""
        pass
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """引数をパース"""
        parsed_args = self.parser.parse_args(args)
        
        # ログレベルの設定
        logging.basicConfig(
            level=getattr(logging, parsed_args.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # sigmaのパース
        if parsed_args.sigma:
            try:
                parsed_args.sigma_list = [float(s.strip()) for s in parsed_args.sigma.split(",")]
            except ValueError:
                self.parser.error("無効なsigma形式です。カンマ区切りの数値を指定してください: 50,100,200")
        else:
            parsed_args.sigma_list = None
        
        # プラットフォーム固有の検証
        self._validate_platform_args(parsed_args)
        
        return parsed_args
    
    @abstractmethod
    def _validate_platform_args(self, args: argparse.Namespace):
        """プラットフォーム固有の引数検証"""
        pass
    
    @abstractmethod
    def execute(self, args: argparse.Namespace):
        """実際の処理を実行"""
        pass
    
    def run(self, args: Optional[List[str]] = None):
        """CLIのメインエントリーポイント"""
        parsed_args = self.parse_args(args)
        
        # ファイル存在チェック
        import os
        if not getattr(parsed_args, '_skip_input_check', False) and not os.path.exists(parsed_args.input):
            self.logger.error(f"入力ファイルが存在しません: {parsed_args.input}")
            raise FileNotFoundError(f"Input file not found: {parsed_args.input}")
        
        if os.path.exists(parsed_args.output) and not parsed_args.force:
            self.logger.error(f"出力ファイルが既に存在します: {parsed_args.output}")
            self.logger.error("上書きする場合は --force オプションを使用してください")
            raise FileExistsError(f"Output file already exists: {parsed_args.output}")
        
        # 実行
        self.execute(parsed_args)
    
    def get_common_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        """共通パラメータを辞書形式で取得"""
        return {
            'input_path': args.input,
            'output_path': args.output,
            'algorithm': args.algorithm,
            'tmp_dir': args.tmp_dir,
            'sigma_list': getattr(args, 'sigma_list', None),
            'pixel_size': args.pixel_size,
            'show_progress': not args.no_progress,
        }
