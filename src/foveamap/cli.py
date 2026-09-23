"""Command-line entry point (README 4.4, 7).

``foveamap --mode {oracle,cached,live} --sequence 08 [--model NAME]`` starts the pipeline/server;
subcommands (``inspect``, ``render``, ``memory``) produce single artefacts. Handlers are filled in
by the phase named in their docstring.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from foveamap import __version__


def _ensure_utf8_stdio() -> None:
    """Windows consoles default to cp1252; never crash on non-ASCII output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sequence", default="08", help="SemanticKITTI sequence id (default: 08)")
    p.add_argument("--preset", default=None, help="grid preset name (default: active_preset in configs/grid.yaml)")
    p.add_argument("--config-dir", default="configs", help="directory holding the YAML configs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foveamap", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"foveamap {__version__}")
    _add_common(parser)
    parser.add_argument("--mode", choices=["oracle", "cached", "live"], help="run the pipeline/server in this mode")
    parser.add_argument("--model", default=None, help="model name for cached/live modes")
    parser.add_argument("--dry-run", type=int, default=None, metavar="N", help="process N frames headlessly and exit")

    sub = parser.add_subparsers(dest="command")

    p_inspect = sub.add_parser("inspect", help="print one scan's stats and write a bird's-eye scatter (T2.3)")
    _add_common(p_inspect)
    p_inspect.add_argument("--idx", type=int, default=0)

    p_render = sub.add_parser("render", help="render top-down 2.5D maps with the fovea overlay (T7.2)")
    _add_common(p_render)
    p_render.add_argument("--mode", choices=["oracle", "cached"], default="oracle")
    p_render.add_argument("--frames", type=int, nargs="+", default=[0])

    p_memory = sub.add_parser("memory", help="print the four-representation memory report (T7.1)")
    _add_common(p_memory)
    p_memory.add_argument("--idx", type=int, default=0)

    return parser


def _cmd_inspect(args: argparse.Namespace) -> int:
    raise NotImplementedError("`inspect` is implemented in Phase 2, T2.3 (docs/PHASES.md).")


def _cmd_render(args: argparse.Namespace) -> int:
    raise NotImplementedError("`render` is implemented in Phase 7, T7.2 (docs/PHASES.md).")


def _cmd_memory(args: argparse.Namespace) -> int:
    raise NotImplementedError("`memory` is implemented in Phase 7, T7.1 (docs/PHASES.md).")


def _cmd_run(args: argparse.Namespace) -> int:
    raise NotImplementedError("`--mode` runs are implemented in Phases 9 and 13 (docs/PHASES.md).")


COMMANDS = {"inspect": _cmd_inspect, "render": _cmd_render, "memory": _cmd_memory}


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None and args.mode is None:
        parser.print_help()
        return 0
    handler = COMMANDS.get(args.command, _cmd_run)
    try:
        return handler(args)
    except NotImplementedError as exc:
        print(f"foveamap: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
