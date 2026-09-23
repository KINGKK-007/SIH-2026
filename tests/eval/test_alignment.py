"""T4.1: frame-alignment verification must pass on correct maths and fail on wrong maths (D-018)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from foveamap import cli
from foveamap.eval.alignment import alignment_check
from foveamap.io.sequence import Sequence
from foveamap.io.synthetic import write_kitti_sequence


def test_correct_transform_passes(synthetic_root: Path, tmp_path: Path) -> None:
    report = alignment_check(synthetic_root, ["04", "08"], n_pairs=6, plot_dir=tmp_path, n_plots=1)
    assert report["verdict"] == "PASS", report
    assert report["compact"]["median_m"] < 0.1
    assert report["spec"]["median_m"] < 0.15
    assert report["identity_control"]["median_m"] > 0.15  # the check can see a missing transform
    assert len(list(tmp_path.glob("frame_alignment_*.png"))) == 2


@pytest.mark.parametrize("variant", ["no_conjugation", "inverse", "swapped_tr"])
def test_wrong_transforms_fail(synthetic_root: Path, monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    def wrong(self: Sequence, i: int, j: int) -> np.ndarray:
        Pi, Pj, Tr = self.poses[i], self.poses[j], self.calib.Tr
        if variant == "no_conjugation":
            return np.linalg.inv(Pi) @ Pj
        if variant == "inverse":
            return np.linalg.inv(np.linalg.inv(Tr) @ np.linalg.inv(Pi) @ Pj @ Tr)
        return Tr @ np.linalg.inv(Pi) @ Pj @ np.linalg.inv(Tr)

    monkeypatch.setattr(Sequence, "relative_transform", wrong)
    report = alignment_check(synthetic_root, ["08"], n_pairs=4)
    assert report["verdict"] == "FAIL"
    assert report["compact"]["median_m"] > 0.5


def test_stationary_vehicle_is_inconclusive(tmp_path: Path) -> None:
    write_kitti_sequence(tmp_path, "08", n_frames=4, seed=2, speed_mps=0.0)
    report = alignment_check(tmp_path, ["08"], n_pairs=3)
    assert report["verdict"] == "INCONCLUSIVE"
    assert report["identity_control"]["n"] == 0


def test_align_cli_writes_json(synthetic_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "frame_alignment.json"
    args = ["align", "--sequences", "08", "--pairs", "3", "--data-root", str(synthetic_root)]
    assert cli.main([*args, "--json", str(out), "--out-dir", str(tmp_path)]) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"


@pytest.mark.data
def test_real_alignment_gate(real_data_root: Path, real_sequences: list[str], tmp_path: Path) -> None:
    seqs = [s for s in ("04", "08") if s in real_sequences]
    report = alignment_check(real_data_root, seqs, n_pairs=10)
    assert report["verdict"] == "PASS", {k: report[k] for k in ("spec", "compact", "identity_control")}
