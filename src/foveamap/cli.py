"""Command-line entry point (README 4.4, 7).

``foveamap --mode {oracle,cached,live} --sequence 08 [--model NAME]`` starts the pipeline/server;
subcommands (``inspect``, ``render``, ``memory``) produce single artefacts. Handlers are filled in
by the phase named in their docstring.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from foveamap import __version__

DEFAULT_DATA_ROOT = os.environ.get("FOVEAMAP_DATA_ROOT", "data/dataset")


def _ensure_utf8_stdio() -> None:
    """Windows consoles default to cp1252; never crash on non-ASCII output."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sequence", default="08", help="SemanticKITTI sequence id (default: 08)")
    p.add_argument(
        "--preset", default=None, help="grid preset name (default: active_preset in configs/grid.yaml)"
    )
    p.add_argument("--config-dir", default="configs", help="directory holding the YAML configs")
    p.add_argument(
        "--data-root",
        default=DEFAULT_DATA_ROOT,
        help="SemanticKITTI root (default: $FOVEAMAP_DATA_ROOT or data/dataset)",
    )
    p.add_argument("--out-dir", default="results/plots", help="where figures are written")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="foveamap", description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=f"foveamap {__version__}")
    _add_common(parser)
    parser.add_argument(
        "--mode", choices=["oracle", "cached", "live"], help="run the pipeline/server in this mode"
    )
    parser.add_argument("--model", default=None, help="model name for cached/live modes")
    parser.add_argument(
        "--dry-run", type=int, default=None, metavar="N", help="process N frames headlessly and exit"
    )

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

    p_align = sub.add_parser("align", help="frame-alignment verification of the pose maths (T4.1)")
    _add_common(p_align)
    p_align.add_argument("--sequences", nargs="+", default=["04", "08"])
    p_align.add_argument("--pairs", type=int, default=50, help="random consecutive pairs per sequence")
    p_align.add_argument("--seed", type=int, default=1337)
    p_align.add_argument("--json", default="results/frame_alignment.json")

    p_stats = sub.add_parser("stats", help="points and per-class counts per distance bucket (T4.2)")
    _add_common(p_stats)
    p_stats.add_argument("--sequences", nargs="+", default=["04", "07", "08"])
    p_stats.add_argument("--stride", type=int, default=1, help="use every k-th frame")
    p_stats.add_argument("--json", default="results/data_stats.json")
    p_stats.add_argument("--table", default="results/tables/data_stats.md")

    return parser


def _write_json(path: str | Path, payload: dict) -> Path:
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _cmd_align(args: argparse.Namespace) -> int:
    """T4.1: frame alignment; exit 0 only on PASS."""
    from foveamap.eval.alignment import alignment_check

    report = alignment_check(
        args.data_root, args.sequences, n_pairs=args.pairs, seed=args.seed, plot_dir=args.out_dir
    )
    for key in ("spec", "compact", "identity_control"):
        s = report[key]
        print(f"{key:<17} median {s['median_m']:.4f} m   p90 {s['p90_m']:.4f} m   n = {s['n']:,}")
    print(f"pairs: {report['n_pairs']}   threshold: {report['threshold_m']} m   verdict: {report['verdict']}")
    print(f"wrote {_write_json(args.json, report)}")
    return 0 if report["passed"] else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    """T2.3: point count, class histogram with names, pose, and a bird's-eye scatter PNG."""
    from foveamap.io.labels import SUPER_CLASS_NAMES, raw_name, raw_to_super, semantic_ids
    from foveamap.io.sequence import Sequence as KittiSequence
    from foveamap.viz.figures import plot_scan_bev

    seq = KittiSequence(args.data_root, args.sequence)
    scan = seq.load_frame(args.idx)
    print(f"sequence {scan.seq}  frame {scan.idx}  t = {scan.timestamp:.3f} s  ({seq.n_frames_total} frames)")
    print(f"points: {len(scan.xyz):,}")
    if scan.raw_labels is not None:
        ids, counts = np.unique(semantic_ids(scan.raw_labels), return_counts=True)
        print("raw classes:")
        for i, c in sorted(zip(ids.tolist(), counts.tolist(), strict=True), key=lambda t: -t[1]):
            print(f"  {i:>4} {raw_name(i):<22} {c:>8,}  {100 * c / len(scan.xyz):5.1f} %")
        supers, moving = raw_to_super(scan.raw_labels)
        print("super-classes:")
        for k, name in enumerate(SUPER_CLASS_NAMES):
            print(f"  {name:<22} {int((supers == k).sum()):>8,}")
        print(f"  moving points          {int(moving.sum()):>8,}")
    with np.printoptions(precision=4, suppress=True):
        print("camera-0 pose (poses.txt):")
        print(scan.pose[:3])
        offset = seq.relative_transform(0, scan.idx)[:3, 3]
        print(f"Velodyne position in frame 0: x={offset[0]:.3f} y={offset[1]:.3f} z={offset[2]:.3f} m")
    out = Path(args.out_dir) / f"inspect_{scan.seq}_{scan.idx:06d}.png"
    plot_scan_bev(scan.xyz, scan.raw_labels, out, f"sequence {scan.seq} frame {scan.idx} (raw classes)")
    print(f"wrote {out}")
    return 0


def _cmd_render(args: argparse.Namespace) -> int:
    raise NotImplementedError("`render` is implemented in Phase 7, T7.2 (docs/PHASES.md).")


def _cmd_memory(args: argparse.Namespace) -> int:
    raise NotImplementedError("`memory` is implemented in Phase 7, T7.1 (docs/PHASES.md).")


def _cmd_run(args: argparse.Namespace) -> int:
    raise NotImplementedError("`--mode` runs are implemented in Phases 9 and 13 (docs/PHASES.md).")


COMMANDS = {
    "inspect": _cmd_inspect,
    "render": _cmd_render,
    "memory": _cmd_memory,
    "align": _cmd_align,
}


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
