"""T2.2: dataset verification on synthetic KITTI-format data."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from foveamap.io.sequence import Sequence
from foveamap.io.verify import verify_dataset, verify_sequence

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_data.py"


@pytest.fixture()
def root(tmp_path: Path, synthetic_root: Path) -> Path:
    dst = tmp_path / "dataset"
    shutil.copytree(synthetic_root, dst)
    return dst


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verify_data", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_data"] = module
    spec.loader.exec_module(module)
    return module


def test_clean_dataset_passes(root: Path) -> None:
    reports = verify_dataset(root, ["04", "07", "08"], sample=5)
    assert all(r.ok for r in reports), [r.errors for r in reports]
    assert [r.n_scans for r in reports] == [12, 12, 12]


def test_poses_normalised_into_sequence_dir(root: Path) -> None:
    report = verify_sequence(root, "04")
    assert report.ok
    assert (root / "sequences" / "04" / "poses.txt").is_file()
    assert any("poses.txt" in n for n in report.notes)
    assert len(Sequence(root, "04").poses) == 12


def test_no_normalise_leaves_layout(root: Path) -> None:
    assert verify_sequence(root, "04", normalise=False).ok
    assert not (root / "sequences" / "04" / "poses.txt").exists()


def test_missing_sequence(root: Path) -> None:
    report = verify_sequence(root, "05")
    assert not report.ok and "missing directory" in report.errors[0]


def test_missing_label_file(root: Path) -> None:
    (root / "sequences" / "08" / "labels" / "000003.label").unlink()
    report = verify_sequence(root, "08")
    assert not report.ok and any("#labels" in e for e in report.errors)


def test_label_length_mismatch(root: Path) -> None:
    label = root / "sequences" / "08" / "labels" / "000002.label"
    label.write_bytes(label.read_bytes()[:-8])
    report = verify_sequence(root, "08")
    assert not report.ok and any("len(label) != N" in e for e in report.errors)


def test_truncated_scan(root: Path) -> None:
    scan = root / "sequences" / "08" / "velodyne" / "000001.bin"
    scan.write_bytes(scan.read_bytes()[:-3])
    assert not verify_sequence(root, "08").ok


def test_pose_count_mismatch(root: Path) -> None:
    poses = root / "sequences" / "07" / "poses.txt"
    poses.write_text("".join(poses.read_text().splitlines(keepends=True)[:-2]))
    report = verify_sequence(root, "07")
    assert not report.ok and any("#poses" in e for e in report.errors)


def test_bad_calib(root: Path) -> None:
    (root / "sequences" / "07" / "calib.txt").write_text("P0: 1 2 3\n")
    report = verify_sequence(root, "07")
    assert not report.ok and any("calib" in e for e in report.errors)


def test_missing_times_is_warning_only(root: Path) -> None:
    (root / "sequences" / "07" / "times.txt").unlink()
    report = verify_sequence(root, "07")
    assert report.ok and report.warnings


def test_non_finite_values_warned(root: Path) -> None:
    scan = root / "sequences" / "07" / "velodyne" / "000000.bin"
    pts = np.fromfile(scan, dtype="<f4")
    pts[0] = np.nan
    pts.tofile(scan)
    report = verify_sequence(root, "07", sample=12)
    assert report.ok and any("non-finite" in w for w in report.warnings)


def test_script_exit_codes_and_json(root: Path, script: ModuleType, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    assert script.main(["--data-root", str(root), "--sequences", "07", "08", "--json", str(out)]) == 0
    data = json.loads(out.read_text())
    assert [d["seq"] for d in data] == ["07", "08"] and all(d["ok"] for d in data)
    assert script.main(["--data-root", str(root), "--sequences", "09"]) == 1


@pytest.mark.data
def test_real_dataset_verifies(real_data_root: Path, real_sequences: list[str]) -> None:
    reports = verify_dataset(real_data_root, real_sequences, sample=10, normalise=False)
    assert all(r.ok for r in reports), [r.errors for r in reports]
