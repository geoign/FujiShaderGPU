"""
FujiShaderGPU/__main__.py
Entry point that selects platform CLI implementation.
"""
import sys
import platform
import warnings


def _configure_stdio_safely() -> None:
    """Avoid crashes when stdout/stderr cannot encode some log characters."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(errors="backslashreplace", line_buffering=True, write_through=True)
            except Exception:
                pass


def main() -> None:
    _configure_stdio_safely()

    system = platform.system().lower()

    if system == "linux":
        try:
            from .cli.linux_cli import LinuxCLI
            cli = LinuxCLI()
        except ImportError as e:
            print(f"Error: Linux dependencies are not available: {e}")
            print("Install them with:")
            print("pip install FujiShaderGPU[linux]")
            sys.exit(1)

    elif system == "windows":
        from .cli.windows_cli import WindowsCLI
        cli = WindowsCLI()

    elif system == "darwin":
        warnings.warn(
            "macOS support is experimental. Windows tile pipeline is used as fallback."
        )
        from .cli.windows_cli import WindowsCLI
        cli = WindowsCLI()

    else:
        print(f"Error: Unsupported OS: {system}")
        sys.exit(1)

    try:
        cli.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
