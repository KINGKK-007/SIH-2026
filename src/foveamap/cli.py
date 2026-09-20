"""foveamap.cli — Click entry point for the ``foveamap`` command.

CLI commands
------------
check-data
    Verify that the SemanticKITTI dataset is accessible at the configured
    root directory.  If absent, falls back to generating a synthetic scan
    and printing scan statistics (point count, XYZ bounds, super-class
    histogram).

Future commands (Phase 3+)
--------------------------
cache       Cache network predictions for an eval sequence.
run         Run the full pipeline on a sequence.
bench       Run all benchmark experiments; write results/*.json.
render      Render dashboard frames to MP4/GIF.
dashboard   Launch the Streamlit dashboard.
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np
import yaml

from foveamap import __version__


# ─────────────────────────────────────────────────────────────────────────────
# CLI root group
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version=__version__, prog_name="foveamap")
def cli() -> None:
    """FoveaMap — Foveated 2.5D LiDAR semantic mapping.

    Run ``foveamap <command> --help`` for details on each command.
    """


# ─────────────────────────────────────────────────────────────────────────────
# check-data
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("check-data")
@click.option(
    "--root",
    default=None,
    show_default=True,
    help="Path to the SemanticKITTI dataset root (overrides config).",
    type=click.Path(),
)
@click.option(
    "--config",
    "config_path",
    default=None,
    show_default=True,
    help="Path to a FoveaMap config YAML (default: configs/default.yaml relative to CWD).",
    type=click.Path(exists=False),
)
@click.option(
    "--seq",
    "sequence",
    default=None,
    show_default=True,
    help="Sequence to check (e.g. '08').  Defaults to value in config.",
)
def check_data(root: str | None, config_path: str | None, sequence: str | None) -> None:
    """Check dataset availability; fall back to a synthetic scan if absent.

    Behaviour
    ---------
    1. Load the config (default.yaml or ``--config``).
    2. Resolve the dataset root from ``--root`` or the config.
    3. If the velodyne directory for the target sequence exists:
       - Load the first scan and its labels.
       - Print point count, XYZ bounds, and super-class histogram.
    4. If the dataset is absent:
       - Print a warning and generate a synthetic scan instead.
       - Print the same statistics for the synthetic scan.
    """
    # ── Load config ──────────────────────────────────────────────────────────
    cfg = _load_config(config_path)

    dataset_root = Path(root) if root else Path(cfg.get("data", {}).get("root", "data/semkitti"))
    seq = sequence or cfg.get("data", {}).get("sequence", "08")

    click.echo(f"[foveamap check-data]  Dataset root : {dataset_root.resolve()}")
    click.echo(f"[foveamap check-data]  Sequence     : {seq}")

    velodyne_dir = dataset_root / "sequences" / seq / "velodyne"
    label_dir = dataset_root / "sequences" / seq / "labels"

    # ── Try real data first ───────────────────────────────────────────────────
    if velodyne_dir.exists():
        bin_files = sorted(velodyne_dir.glob("*.bin"))
        if bin_files:
            click.secho(f"✓ Found {len(bin_files):,} scans in {velodyne_dir}", fg="green")
            _print_real_scan_stats(bin_files[0], label_dir)
            return
        else:
            click.secho(f"⚠  Velodyne directory exists but contains no .bin files: {velodyne_dir}", fg="yellow")
    else:
        click.secho(f"✗ Dataset not found at: {velodyne_dir}", fg="red")

    # ── Fallback: synthetic scan ──────────────────────────────────────────────
    click.secho("\n↳ Falling back to synthetic scan (no real data required).\n", fg="yellow")
    _print_synthetic_scan_stats()


def _load_config(config_path: str | None) -> dict:
    """Load the FoveaMap YAML config, returning an empty dict on failure."""
    if config_path is not None:
        p = Path(config_path)
        if p.exists():
            with p.open() as fh:
                return yaml.safe_load(fh) or {}
        else:
            click.secho(f"⚠  Config not found: {config_path}; using defaults.", fg="yellow")
            return {}

    # Auto-discover default.yaml
    for candidate in [
        Path("configs/default.yaml"),
        Path(__file__).parent.parent.parent / "configs" / "default.yaml",
    ]:
        if candidate.exists():
            with candidate.open() as fh:
                return yaml.safe_load(fh) or {}

    return {}


def _print_real_scan_stats(bin_file: Path, label_dir: Path) -> None:
    """Load first scan and print statistics."""
    from foveamap.io.kitti import load_velodyne_bin, load_labels
    from foveamap.io.labels import unpack_kitti_labels, to_superclass, superclass_histogram

    points = load_velodyne_bin(bin_file)
    click.echo(f"\n  Scan file   : {bin_file.name}")
    click.echo(f"  Points      : {len(points):,}")
    _print_bounds(points)

    label_file = label_dir / bin_file.with_suffix(".label").name
    if label_file.exists():
        raw = load_labels(label_file)
        sem_ids, _ = unpack_kitti_labels(raw)
        super_ids = to_superclass(sem_ids)
        _print_histogram(superclass_histogram(super_ids))
    else:
        click.secho(f"  ⚠  No label file found at {label_file}", fg="yellow")


def _print_synthetic_scan_stats() -> None:
    """Generate synthetic scan and print statistics."""
    from foveamap.io.synthetic import generate_synthetic_scan
    from foveamap.io.labels import unpack_kitti_labels, to_superclass, superclass_histogram

    click.echo("  Generating synthetic scan (120,000 points)…")
    points, raw_labels = generate_synthetic_scan(num_points=120_000)
    sem_ids, _ = unpack_kitti_labels(raw_labels)
    super_ids = to_superclass(sem_ids)

    click.secho("  ✓ Synthetic scan generated successfully.", fg="green")
    click.echo(f"\n  Points      : {len(points):,}")
    _print_bounds(points)
    _print_histogram(superclass_histogram(super_ids))


def _print_bounds(points: np.ndarray) -> None:
    """Pretty-print XYZ bounding box."""
    xyz = points[:, :3]
    click.echo(
        f"  X range     : [{xyz[:,0].min():.2f}, {xyz[:,0].max():.2f}] m"
    )
    click.echo(
        f"  Y range     : [{xyz[:,1].min():.2f}, {xyz[:,1].max():.2f}] m"
    )
    click.echo(
        f"  Z range     : [{xyz[:,2].min():.2f}, {xyz[:,2].max():.2f}] m"
    )


def _print_histogram(hist: dict[str, int]) -> None:
    """Pretty-print super-class histogram as a bar chart."""
    click.echo("\n  Super-class point histogram:")
    total = sum(hist.values())
    bar_width = 30
    for name, count in hist.items():
        frac = count / total if total > 0 else 0
        bar = "█" * int(frac * bar_width)
        click.echo(f"    {name:<25} {count:>8,}  {bar}  ({frac*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# Future command stubs
# ─────────────────────────────────────────────────────────────────────────────

@cli.command("cache")
@click.option("--seq", default="08", help="Sequence to cache predictions for.")
@click.option("--model", default="frnet", help="Model name (frnet|cenet|salsanext).")
def cache_cmd(seq: str, model: str) -> None:
    """(Phase 3) Cache network predictions for a sequence to disk."""
    click.secho(f"[cache] Phase 3 stub — seq={seq} model={model}", fg="yellow")
    click.echo("  Implement in Phase 3 (P3: segmenter spike).")
    sys.exit(0)


@cli.command("run")
@click.option("--seq", default="08", help="SemanticKITTI sequence (e.g. '08').")
@click.option("--start", default=0, type=int, help="First frame index.")
@click.option("--n", default=1, type=int, help="Number of frames to process.")
@click.option("--labels", default="gt", type=click.Choice(["gt", "pred"]),
              help="Label source: 'gt' (oracle) or 'pred' (cached predictions).")
@click.option("--grid", default="fovea_4ring",
              type=click.Choice(["fovea_4ring", "ps_literal_2ring", "uniform_5cm", "uniform_20cm"]),
              help="Grid preset.")
@click.option("--root", default=None, type=click.Path(), help="Dataset root path.")
@click.option("--out", default="results", type=click.Path(), help="Output directory.")
@click.option("--no-render", is_flag=True, default=False, help="Skip PNG rendering.")
def run_cmd(
    seq: str, start: int, n: int, labels: str, grid: str,
    root: str | None, out: str, no_render: bool,
) -> None:
    """Run the Phase 2 oracle pipeline: bin → classify → grid → render.

    When real data is not available, falls back to a synthetic scan automatically.
    """
    import warnings
    warnings.filterwarnings("ignore")
    import os; os.environ.setdefault("MPLBACKEND", "Agg")

    from foveamap.grid.spec import load_spec_from_preset
    from foveamap.grid.clipmap import ClipmapGrid
    from foveamap.models.oracle import OracleSegmenter

    click.echo(f"[foveamap run]  seq={seq}  frames=[{start},{start+n})  grid={grid}  labels={labels}")

    # ── Load grid spec ────────────────────────────────────────────────────
    try:
        spec = load_spec_from_preset(grid)
    except Exception as e:
        click.secho(f"⚠  Could not load preset {grid}: {e}. Using fovea_4ring defaults.", fg="yellow")
        from foveamap.grid.spec import Ring, GridSpec
        spec = GridSpec(
            rings=(
                Ring(cell_m=0.05, half_extent_m=10.0),
                Ring(cell_m=0.10, half_extent_m=30.0),
                Ring(cell_m=0.20, half_extent_m=60.0),
                Ring(cell_m=0.40, half_extent_m=100.0),
            ),
            z_range_m=(-3.0, 5.0),
        )

    # ── Load data (real or synthetic) ─────────────────────────────────────
    from foveamap.io.synthetic import generate_synthetic_scan
    from foveamap.io.labels import unpack_kitti_labels

    dataset_root = Path(root) if root else Path("data/semkitti")
    velodyne_dir = dataset_root / "sequences" / seq / "velodyne"
    label_dir    = dataset_root / "sequences" / seq / "labels"
    use_real = velodyne_dir.exists() and any(velodyne_dir.glob("*.bin"))

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for frame_offset in range(n):
        frame_idx = start + frame_offset

        if use_real:
            from foveamap.io.kitti import load_velodyne_bin, load_labels
            bin_files = sorted(velodyne_dir.glob("*.bin"))
            if frame_idx >= len(bin_files):
                click.secho(f"  Frame {frame_idx} out of range ({len(bin_files)} scans).", fg="yellow")
                break
            pts = load_velodyne_bin(bin_files[frame_idx])
            label_file = label_dir / bin_files[frame_idx].with_suffix(".label").name
            raw_labels = load_labels(label_file) if label_file.exists() else np.zeros(len(pts), dtype=np.uint32)
        else:
            if frame_offset == 0:
                click.secho("  ↳ Real data not found — using synthetic scan.", fg="yellow")
            pts, raw_labels = generate_synthetic_scan(num_points=120_000, seed=frame_idx)

        # ── Classify ────────────────────────────────────────────────────
        seg = OracleSegmenter(raw_labels)
        sc, mv, vr, cf = seg.grid_inputs(pts)

        # ── Build grid ──────────────────────────────────────────────────
        grid_obj = ClipmapGrid.build(pts, sc, mv, vr, cf, spec)
        s = grid_obj.stats()

        click.echo(
            f"  frame {frame_idx:04d}  "
            f"assigned={s['n_points_assigned']:,}/{len(pts):,}  "
            f"active={s['active_cells']:,} cells  "
            f"({s['active_mb']:.1f} MB)  "
            f"build={s['build_time_ms']:.1f} ms"
        )

        # ── Render ──────────────────────────────────────────────────────
        if not no_render:
            from foveamap.viz.render import render_top_down, save_png
            fig = render_top_down(grid_obj, shade_ground=True, show_stats=True, dpi=120)
            png_path = out_dir / f"frame_{frame_idx:04d}_{grid}.png"
            save_png(fig, png_path)
            click.secho(f"  ✓ Saved {png_path}", fg="green")

    click.secho(f"\n✓ Done. Output in {out_dir.resolve()}", fg="green")



@cli.command("bench")
@click.option("--seq", default="08")
@click.option("--stride", default=5, type=int)
@click.option("--presets", default="all")
def bench_cmd(seq: str, stride: int, presets: str) -> None:
    """(Phase 6) Run all benchmark experiments; write results/*.json."""
    click.secho(f"[bench] Phase 6 stub — seq={seq} stride={stride} presets={presets}", fg="yellow")
    sys.exit(0)


@cli.command("render")
@click.option("--seq", default="08")
@click.option("--start", default=0, type=int)
@click.option("--n", default=300, type=int)
@click.option("--out", default="results/demo.mp4")
def render_cmd(seq: str, start: int, n: int, out: str) -> None:
    """(Phase 7) Render dashboard frames to MP4/GIF."""
    click.secho(f"[render] Phase 7 stub — out={out}", fg="yellow")
    sys.exit(0)


@cli.command("dashboard")
def dashboard_cmd() -> None:
    """(Phase 7) Launch the Streamlit dashboard."""
    click.secho("[dashboard] Phase 7 stub — Streamlit app not yet implemented.", fg="yellow")
    sys.exit(0)
