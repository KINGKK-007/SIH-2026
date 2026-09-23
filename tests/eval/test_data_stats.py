"""T4.2: per-bucket data statistics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foveamap import cli
from foveamap.eval.data_stats import data_stats, stats_table_md
from foveamap.io.labels import raw_to_super
from foveamap.io.sequence import Sequence


def test_counts_are_consistent(synthetic_root: Path) -> None:
    stats = data_stats(synthetic_root, ["08"], stride=3)
    s = stats["sequences"][0]
    assert s["frames_used"] == 4 and s["frame_stride"] == 3
    total = sum(s["points"]) + s["points_beyond_last_bucket"]
    expected = sum(len(scan.xyz) for scan in Sequence(synthetic_root, "08", frame_stride=3))
    assert total == expected
    per_super = np.array(list(s["super_class_points"].values())).sum(axis=0)
    np.testing.assert_array_equal(per_super, s["points"])  # every in-range point has one super-class
    assert sum(s["moving_points"]) > 0 and s["safety_critical_points"][0] + s["safety_critical_points"][1] > 0


def test_object_instances(synthetic_root: Path) -> None:
    s = data_stats(synthetic_root, ["08"], stride=12)["sequences"][0]  # frame 0 only
    objects = s["object_instances"]
    assert sum(objects["vehicle_moving"]) == 1  # one moving car in the scene
    assert sum(objects["person_cyclist"]) >= 1
    assert sum(objects["vehicle_static"]) >= 3


def test_moving_points_match_labels(synthetic_root: Path) -> None:
    s = data_stats(synthetic_root, ["07"], stride=6)["sequences"][0]
    scans = list(Sequence(synthetic_root, "07", frame_stride=6))
    n_moving = sum(int(raw_to_super(sc.raw_labels)[1].sum()) for sc in scans if sc.raw_labels is not None)
    assert sum(s["moving_points"]) <= n_moving  # beyond-100 m movers are excluded


def test_table_and_cli(synthetic_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    md = stats_table_md(data_stats(synthetic_root, ["04", "08"], stride=6))
    assert md.count("## Sequence") == 2 and "| 60-100 m |" in md and "objects: vehicle_moving" in md
    js, tb = tmp_path / "s.json", tmp_path / "t.md"
    args = ["stats", "--sequences", "08", "--stride", "6", "--data-root", str(synthetic_root)]
    assert cli.main([*args, "--json", str(js), "--table", str(tb)]) == 0
    assert json.loads(js.read_text())["sequences"][0]["frames_used"] == 2
    assert tb.read_text().startswith("# Data statistics")
