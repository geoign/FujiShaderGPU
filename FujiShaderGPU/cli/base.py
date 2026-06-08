"""Shared CLI base class."""
from __future__ import annotations

import argparse
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .args import SHARED_ARGS, add_arguments, parse_list_fields


class BaseCLI(ABC):
    """Base class for platform-specific CLI implementations."""

    def __init__(self):
        self.parser = self._create_parser()
        self.logger = logging.getLogger(__name__)

    def _create_parser(self) -> argparse.ArgumentParser:
        supported_algorithms = self.get_supported_algorithms()
        default_algorithm = supported_algorithms[0] if supported_algorithms else "topousm_fast"

        parser = argparse.ArgumentParser(
            description=self.get_description(),
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=self.get_epilog(),
        )

        parser.add_argument("input", help="Input DEM (COG format)")
        parser.add_argument("output", help="Output file (COG format)")

        parser.add_argument(
            "--algorithm",
            "--algo",
            default=default_algorithm,
            choices=supported_algorithms,
            help=f"Algorithm to use (default: {default_algorithm})",
        )

        parser.add_argument(
            "--tmp-dir",
            default="tiles_tmp",
            help="Directory for temporary files (default: tiles_tmp)",
        )

        parser.add_argument(
            "--pixel-size",
            "--pixel_size",
            type=float,
            help="Pixel size in meters. Auto-detected when omitted (geographic CRS uses center-latitude meter conversion)",
        )

        parser.add_argument(
            "--log-level",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            default="INFO",
            help="Log level (default: INFO)",
        )

        parser.add_argument(
            "--no-progress",
            action="store_true",
            help="Disable progress display",
        )

        parser.add_argument(
            "--force",
            action="store_true",
            help="Overwrite the output file even if it exists",
        )

        # Shared algorithm/output/spatial arguments (identical on every platform).
        add_arguments(parser, SHARED_ARGS)

        # Backend-specific options (Dask cluster knobs vs. tile/COG knobs).
        self._add_platform_specific_args(parser)
        return parser

    @abstractmethod
    def get_description(self) -> str:
        pass

    @abstractmethod
    def get_epilog(self) -> str:
        pass

    @abstractmethod
    def get_supported_algorithms(self) -> List[str]:
        pass

    @abstractmethod
    def _add_platform_specific_args(self, parser: argparse.ArgumentParser):
        pass

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        parsed_args = self.parser.parse_args(args)

        logging.basicConfig(
            level=getattr(logging, parsed_args.log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        # Parse all comma-separated list arguments uniformly (--radii, --scales, ...).
        parse_list_fields(parsed_args, self.parser)

        self._validate_platform_args(parsed_args)
        return parsed_args

    @abstractmethod
    def _validate_platform_args(self, args: argparse.Namespace):
        pass

    @abstractmethod
    def execute(self, args: argparse.Namespace):
        pass

    def run(self, args: Optional[List[str]] = None):
        parsed_args = self.parse_args(args)

        import os

        if not getattr(parsed_args, "_skip_input_check", False) and not os.path.exists(parsed_args.input):
            self.logger.error(f"Input file does not exist: {parsed_args.input}")
            raise FileNotFoundError(f"Input file not found: {parsed_args.input}")

        if os.path.exists(parsed_args.output) and not parsed_args.force:
            self.logger.error(f"Output file already exists: {parsed_args.output}")
            self.logger.error("Use the --force option to overwrite")
            raise FileExistsError(f"Output file already exists: {parsed_args.output}")

        self.execute(parsed_args)

    def get_common_params(self, args: argparse.Namespace) -> Dict[str, Any]:
        return {
            "input_path": args.input,
            "output_path": args.output,
            "algorithm": args.algorithm,
            "tmp_dir": args.tmp_dir,
            "pixel_size": args.pixel_size,
            "show_progress": not args.no_progress,
        }