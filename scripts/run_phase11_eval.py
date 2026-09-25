"""Run Phase 11 evaluations and produce artifacts for Gate 11."""

from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from foveamap.config import load_config
from foveamap.derived import (
    compute_derived_layers,
    traversability_map,
    TRAVERSABLE_CLASS,
    NON_TRAVERSABLE_CLASS,
    UNKNOWN_CLASS,
)
from foveamap.derived.hazards import inject_hazards, check_hazard_detected
from foveamap.derived.zoom import render_zoom
from foveamap.eval.hazard_eval import hazard_metrics, save_hazard_evaluation
from foveamap.grid.engine import rasterize
from foveamap.grid.layers import finalize
from foveamap.grid.presets import load_preset
from foveamap.io.labels import raw_to_super
from foveamap.io.sequence import Sequence
from foveamap.pipeline.records import ClassifiedScan


def main():
    print("Loading configs and Sequence 08...")
    cfgs = load_config("configs")
    seq = Sequence("data/dataset", "08")
    scan0 = seq.load_frame(0)

    # ── 1. Gate 11 Zoom Lens Figure ──────────────────────────────────────────
    print("1. Generating zoom lens comparison for real kerb in Sequence 08 Frame 0...")
    # Real kerb boundary along road/sidewalk at x in [3.0, 7.0], y in [-3.0, 0.0]
    world_box = (3.0, 7.0, -3.0, 0.0)
    out_zoom = Path("results/plots/zoom_kerb_fine_vs_coarse.png")
    out_zoom.parent.mkdir(parents=True, exist_ok=True)

    render_zoom(
        scan=scan0,
        world_box=world_box,
        cfg=cfgs.derived,
        out_path=out_zoom,
    )
    print(f"   Saved zoom comparison to {out_zoom}")

    # ── 2. Gate 11 Traversability Map PNG ────────────────────────────────────
    print("2. Generating traversability map PNG for Frame 0...")
    super_cls, moving = raw_to_super(scan0.raw_labels)
    conf = np.full(len(scan0.xyz), 255, dtype=np.uint8)
    preset = load_preset(cfgs.grid.active_preset, cfgs.grid)

    acc = rasterize(preset, scan0.xyz, super_cls, moving, conf)
    layers = finalize(acc, cfgs.grid, preset)
    layers = compute_derived_layers(layers, cfgs.derived)
    trav_cats = traversability_map(layers)

    # Render top-down composite traversability map
    # 0: UNKNOWN (gray), 1: TRAVERSABLE (green), 2: NON-TRAVERSABLE (red)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

    # Ring 0 traversability
    cat0 = trav_cats[0]
    color_map = np.zeros((*cat0.shape, 3), dtype=np.float32)
    color_map[cat0 == UNKNOWN_CLASS] = [0.2, 0.2, 0.25]
    color_map[cat0 == TRAVERSABLE_CLASS] = [0.18, 0.8, 0.44]  # Green
    color_map[cat0 == NON_TRAVERSABLE_CLASS] = [0.9, 0.3, 0.24]  # Red

    axes[0].imshow(color_map, origin="lower")
    axes[0].set_title(f"Ring 0 Traversability (5 cm cells, +/-10 m)")
    axes[0].axis("off")

    # Ring 1 traversability
    cat1 = trav_cats[1]
    color_map1 = np.zeros((*cat1.shape, 3), dtype=np.float32)
    color_map1[cat1 == UNKNOWN_CLASS] = [0.2, 0.2, 0.25]
    color_map1[cat1 == TRAVERSABLE_CLASS] = [0.18, 0.8, 0.44]
    color_map1[cat1 == NON_TRAVERSABLE_CLASS] = [0.9, 0.3, 0.24]

    axes[1].imshow(color_map1, origin="lower")
    axes[1].set_title(f"Ring 1 Traversability (10 cm cells, +/-25 m)")
    axes[1].axis("off")

    out_trav = Path("results/plots/traversability_08_000000.png")
    fig.suptitle("FoveaMap 3-State Traversability (Green: Passable, Red: Hazard/Obstacle, Dark: Unknown)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_trav, dpi=150)
    plt.close(fig)
    print(f"   Saved traversability map to {out_trav}")

    # ── 3. Gate 11 Synthetic Hazard Evaluation ───────────────────────────────
    print("3. Running synthetic hazard evaluation (pothole, kerb, overhang)...")
    rng = np.random.default_rng(cfgs.hazards.seed)
    results = []

    # Run over consecutive frames (0..20) with multiple seeded injections
    frames_to_eval = [0, 5, 10, 15, 20]
    for f_idx in frames_to_eval:
        scan_f = seq.load_frame(f_idx)
        s_cls, mv = raw_to_super(scan_f.raw_labels)
        cf = np.full(len(scan_f.xyz), 255, dtype=np.uint8)
        classified = ClassifiedScan(scan_f, s_cls, mv, cf)

        # Perform 50 hazard injections per frame -> 250 injections total (> 200 required)
        for _ in range(50):
            injected, hazards = inject_hazards(classified, cfgs.hazards, rng)
            if not hazards:
                continue

            acc_inj = rasterize(preset, injected.scan.xyz, injected.super_cls, injected.moving, injected.conf)
            lay_inj = finalize(acc_inj, cfgs.grid, preset)
            lay_inj = compute_derived_layers(lay_inj, cfgs.derived)

            for h in hazards:
                detected = check_hazard_detected(h, lay_inj)
                cx, cy = h.center_xy_m
                dist_m = float(np.sqrt(cx**2 + cy**2))

                # Ring index
                from foveamap.grid.engine import world_to_cell
                c_info = world_to_cell(preset, int(cx * 1000), int(cy * 1000))
                ring_idx = c_info[0] if c_info else 0

                results.append({
                    "type": h.type,
                    "ring": ring_idx,
                    "distance_m": dist_m,
                    "detected": detected,
                })

    metrics = hazard_metrics(results)
    out_json = Path("results/hazard_metrics.json")
    out_plot = Path("results/plots/hazard_detection_by_ring.png")
    save_hazard_evaluation(metrics, out_json, out_plot)
    print(f"   Total injections evaluated: {metrics['total']['n']}")
    print(f"   Overall detection rate: {metrics['total']['rate'] * 100:.1f}% (CI: {metrics['total']['ci_95']})")
    print(f"   Saved metrics to {out_json}")
    print(f"   Saved plot to {out_plot}")
    print("\nGate 11 evaluation complete!")


if __name__ == "__main__":
    main()
