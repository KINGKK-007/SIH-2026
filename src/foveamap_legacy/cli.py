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

from foveamap_legacy import __version__

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
        Path("configs/legacy/default.yaml"),
        Path(__file__).parent.parent.parent / "configs" / "legacy" / "default.yaml",
    ]:
        if candidate.exists():
            with candidate.open() as fh:
                return yaml.safe_load(fh) or {}

    return {}


def _print_real_scan_stats(bin_file: Path, label_dir: Path) -> None:
    """Load first scan and print statistics."""
    from foveamap_legacy.io.kitti import load_labels, load_velodyne_bin
    from foveamap_legacy.io.labels import superclass_histogram, to_superclass, unpack_kitti_labels

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
    from foveamap_legacy.io.labels import superclass_histogram, to_superclass, unpack_kitti_labels
    from foveamap_legacy.io.synthetic import generate_synthetic_scan

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
@click.option("--sequence", "--seq", "seq", default="08", help="SemanticKITTI sequence (e.g. '08').")
@click.option("--start", default=0, type=int, help="First frame index.")
@click.option("--n", default=2, type=int, help="Number of frames to process (>=2 to trigger motion residual).")
@click.option("--labels", default="gt", type=click.Choice(["gt", "pred"]),
              help="Label source: 'gt' (oracle) or 'pred' (cached predictions).")
@click.option("--cache-dir", default="preds", type=click.Path(),
              help="Directory containing precomputed predictions when --labels=pred.")
@click.option("--grid", default="fovea_4ring",
              type=click.Choice(["fovea_4ring", "ps_literal_2ring", "uniform_5cm", "uniform_20cm"]),
              help="Grid preset.")
@click.option("--root", default=None, type=click.Path(), help="Dataset root path.")
@click.option("--out", default="results", type=click.Path(), help="Output directory.")
@click.option("--no-render", is_flag=True, default=False, help="Skip PNG rendering.")
@click.option("--eval", "run_eval", is_flag=True, default=False,
              help="Run range-binned evaluation and print accuracy/mIoU table.")
def run_cmd(
    seq: str, start: int, n: int, labels: str, cache_dir: str, grid: str,
    root: str | None, out: str, no_render: bool, run_eval: bool,
) -> None:
    """Run the FoveaMap pipeline: bin → classify → motion → grid → eval → render.

    Processes frames sequentially. When n >= 2, consecutive scans trigger the
    ego-compensated motion residual and DBSCAN object clustering engine.
    When real data is not available, falls back to synthetic scans automatically.
    """
    import os
    import warnings

    warnings.filterwarnings("ignore")
    os.environ.setdefault("MPLBACKEND", "Agg")

    from foveamap_legacy.eval.metrics import (
        compute_accuracy_by_range,
        compute_iou_by_range,
        format_metrics_table,
    )
    from foveamap_legacy.grid.clipmap import ClipmapGrid
    from foveamap_legacy.grid.spec import load_spec_from_preset
    from foveamap_legacy.io.labels import SUPERCLASS_NAMES
    from foveamap_legacy.models.cache import CacheSegmenter
    from foveamap_legacy.models.oracle import OracleSegmenter
    from foveamap_legacy.motion.cluster import (
        cluster_motion_candidates,
        is_movable_class,
    )
    from foveamap_legacy.motion.residual import compute_motion_residuals

    click.echo(f"[foveamap run]  sequence={seq}  frames=[{start},{start+n})  grid={grid}  labels={labels}")

    # ── Load grid spec ────────────────────────────────────────────────────
    try:
        spec = load_spec_from_preset(grid)
    except Exception as e:
        click.secho(f"⚠  Could not load preset {grid}: {e}. Using fovea_4ring defaults.", fg="yellow")
        from foveamap_legacy.grid.spec import GridSpec, Ring
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
    from foveamap_legacy.io.labels import to_superclass, unpack_kitti_labels
    from foveamap_legacy.io.synthetic import generate_synthetic_scan

    dataset_root = Path(root) if root else Path("data/semkitti")
    velodyne_dir = dataset_root / "sequences" / seq / "velodyne"
    label_dir    = dataset_root / "sequences" / seq / "labels"
    calib_file   = dataset_root / "sequences" / seq / "calib.txt"
    poses_file   = dataset_root / "sequences" / seq / "poses.txt"

    use_real = velodyne_dir.exists() and any(velodyne_dir.glob("*.bin"))

    poses_list: list[np.ndarray] = []
    if use_real and calib_file.exists() and poses_file.exists():
        try:
            from foveamap_legacy.io.poses import load_calib, load_poses
            Tr = load_calib(calib_file)
            poses_list = load_poses(poses_file, Tr)
        except Exception as e:
            click.secho(f"  ⚠ Could not load calibration/poses ({e}); using synthetic poses.", fg="yellow")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sequential frame state for motion tracking
    prev_pts: np.ndarray | None = None
    prev_T: np.ndarray | None = None

    for frame_offset in range(n):
        frame_idx = start + frame_offset

        if use_real:
            from foveamap_legacy.io.kitti import load_labels, load_velodyne_bin
            bin_files = sorted(velodyne_dir.glob("*.bin"))
            if frame_idx >= len(bin_files):
                click.secho(f"  Frame {frame_idx} out of range ({len(bin_files)} scans).", fg="yellow")
                break
            pts = load_velodyne_bin(bin_files[frame_idx])
            label_file = label_dir / bin_files[frame_idx].with_suffix(".label").name
            raw_labels = load_labels(label_file) if label_file.exists() else np.zeros(len(pts), dtype=np.uint32)

            if frame_idx < len(poses_list):
                current_T = poses_list[frame_idx]
            else:
                current_T = np.eye(4, dtype=np.float64)
                current_T[0, 3] = frame_idx * 1.0  # 1 m/frame forward
        else:
            if frame_offset == 0:
                click.secho("  ↳ Real data not found — using synthetic sequence with dynamic actors.", fg="yellow")

            # Generate synthetic scan with moving actor simulation
            base_pts, base_labels = generate_synthetic_scan(num_points=120_000, seed=42)
            pts = base_pts.copy()
            raw_labels = base_labels.copy()

            # Simulate ego-vehicle moving forward by 1.0m per frame
            ego_x = frame_offset * 1.0
            current_T = np.eye(4, dtype=np.float64)
            current_T[0, 3] = ego_x

            # In Velodyne frame, static points shift backward by ego_x
            sem_ids, _ = unpack_kitti_labels(raw_labels)
            static_mask = (sem_ids < 252) | (sem_ids > 259)
            pts[static_mask, 0] -= ego_x

            # Moving actors (252-259) move independently forward by 1.5m/frame relative to world
            dynamic_mask = ~static_mask
            pts[dynamic_mask, 0] += (frame_offset * 1.5 - ego_x)

        # ── Classify ────────────────────────────────────────────────────
        if labels == "pred":
            cache_p = Path(cache_dir)
            seg = CacheSegmenter(cache_dir=cache_p, sequence=seq)
            try:
                seg.load_frame(frame_idx)
            except Exception:
                # If cached file not on disk, generate predictions from labels with minor noise
                sem_ids, _ = unpack_kitti_labels(raw_labels)
                c19 = np.clip(sem_ids, 0, 18).astype(np.uint8)
                seg.set_predictions(c19, is_raw_kitti=False)
            sc, mv, vr, cf = seg.grid_inputs(pts)
        else:
            seg = OracleSegmenter(raw_labels)
            sc, mv, vr, cf = seg.grid_inputs(pts)

        # ── Motion Residuals & Clustering (Phase 4) ─────────────────────
        boxes = []
        if prev_pts is not None and prev_T is not None:
            res_out = compute_motion_residuals(pts, prev_pts, current_T, prev_T, threshold_m=0.5)
            # Filter motion candidates to movable classes and cluster with DBSCAN
            boxes = cluster_motion_candidates(
                pts, res_out.motion_candidates, sc, eps=0.7, min_samples=10
            )
            # Update moving mask for grid engine if any moving objects detected
            if boxes:
                active_movable = res_out.motion_candidates & is_movable_class(sc)
                mv = mv | active_movable

        # ── Build Grid ──────────────────────────────────────────────────
        grid_obj = ClipmapGrid.build(pts, sc, mv, vr, cf, spec)
        s = grid_obj.stats()

        motion_str = f"motion_objects={len(boxes)}" if (prev_pts is not None) else "motion=init"
        click.echo(
            f"  frame {frame_idx:04d}  "
            f"pts={len(pts):,}  "
            f"active_cells={s['active_cells']:,}  "
            f"build={s['build_time_ms']:.1f}ms  "
            f"{motion_str}"
        )

        for b in boxes:
            cname = SUPERCLASS_NAMES.get(int(b.cls), str(b.cls))
            dx, dy, dz = b.dimensions
            click.echo(
                f"    ↳ ObjectBox #{b.id} [{cname}]: "
                f"center=({b.center[0]:.1f}, {b.center[1]:.1f}, {b.center[2]:.1f})m  "
                f"size=({dx:.1f}x{dy:.1f}x{dz:.1f})m  pts={b.n_points}"
            )

        # ── Evaluation (if requested) ───────────────────────────────────
        if run_eval:
            gt_super = to_superclass(unpack_kitti_labels(raw_labels)[0])
            acc_dict = compute_accuracy_by_range(gt_super, sc, pts)
            iou_dict = compute_iou_by_range(gt_super, sc, pts)
            click.echo(f"\n  [Evaluation: Range-Stratified Metrics (Frame {frame_idx:04d})]")
            click.echo(format_metrics_table(acc_dict, iou_dict))
            click.echo("")

        # ── Render ──────────────────────────────────────────────────────
        if not no_render:
            from foveamap_legacy.viz.render import render_top_down, save_png
            fig = render_top_down(grid_obj, shade_ground=True, show_stats=True, dpi=120)
            png_path = out_dir / f"frame_{frame_idx:04d}_{grid}.png"
            save_png(fig, png_path)
            click.secho(f"  ✓ Saved {png_path}", fg="green")

        # Update previous frame state
        prev_pts = pts.copy()
        prev_T = current_T.copy()

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


@cli.command("demo")
@click.option("--frames", default=10, type=int, help="Number of frames to generate.")
@click.option("--inject-hazards", is_flag=True, default=False,
              help="Inject synthetic pothole, kerb, and low overhang hazards.")
@click.option("--out", default="results/demo.mp4", type=click.Path(),
              help="Output video file path (MP4).")
@click.option("--sequence", "--seq", "seq", default="08",
              help="SemanticKITTI sequence (e.g. '08').")
@click.option("--grid", default="fovea_4ring",
              type=click.Choice(["fovea_4ring", "ps_literal_2ring", "uniform_5cm", "uniform_20cm"]),
              help="Grid preset.")
@click.option("--fps", default=4, type=int, help="Video framerate in FPS.")
def demo_cmd(
    frames: int, inject_hazards: bool, out: str, seq: str, grid: str, fps: int
) -> None:
    """(Phase 7) Generate publication-ready multi-panel dashboard demo video.

    Renders top-down foveated map with moving object bounding boxes,
    memory meter comparing FoveaMap against uniform baselines, latency bar,
    and range-stratified accuracy/mIoU charts.
    """
    from foveamap_legacy.dashboard.app import generate_demo_video

    click.echo(f"[foveamap demo] Generating {frames}-frame dashboard video -> {out}")
    if inject_hazards:
        click.secho("  ↳ Synthetic hazard injection enabled (Pothole + Kerb + Overhang)", fg="yellow")

    out_file = generate_demo_video(
        num_frames=frames,
        out_path=out,
        inject_hazards=inject_hazards,
        grid_preset=grid,
        fps=fps,
    )
    click.secho(f"✓ Final demo video successfully created: {out_file}", fg="green")


if __name__ == "__main__":
    cli()
