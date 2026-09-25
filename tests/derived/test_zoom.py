"""Tests for zoom lens data path (README 6.7, task T11.9)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from foveamap.derived.zoom import render_zoom
from foveamap.io.labels import DRIVABLE
from foveamap.pipeline.records import Scan


def test_render_zoom_synthetic_step(tmp_path):
    # Step at x = 0 (height 12 cm)
    xs = np.linspace(-2.0, 2.0, 100)
    ys = np.linspace(-2.0, 2.0, 100)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.where(xx < 0, 0.0, 0.12)
    pts = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)

    scan = Scan(
        seq="synthetic",
        idx=0,
        xyz=pts,
        remission=np.zeros(len(pts), dtype=np.float32),
        raw_labels=np.full(len(pts), 40, dtype=np.uint32),  # road
        pose=np.eye(4),
        timestamp=0.0,
    )

    out_png = tmp_path / "zoom_test.png"
    fine_step, coarse_step, kerb_fine, kerb_coarse = render_zoom(
        scan=scan,
        world_box=(-1.5, 1.5, -1.5, 1.5),
        out_path=out_png,
    )

    assert out_png.is_file()
    assert kerb_fine is True  # 12 cm step is detected at 5 cm resolution
    assert fine_step.shape[0] > coarse_step.shape[0]
