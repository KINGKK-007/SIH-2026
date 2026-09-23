"""Tests for scripts/doctor.py status rules (PHASES T1.3)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

DOCTOR_PATH = Path(__file__).resolve().parents[1] / "scripts" / "doctor.py"


@pytest.fixture(scope="module")
def doctor() -> ModuleType:
    spec = importlib.util.spec_from_file_location("doctor", DOCTOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["doctor"] = module  # @dataclass resolves annotations via sys.modules
    spec.loader.exec_module(module)
    return module


def _make_sequence(root: Path, seq: str, n_scans: int, n_labels: int, poses_in_seq: bool = True) -> None:
    seq_dir = root / "sequences" / seq
    (seq_dir / "velodyne").mkdir(parents=True)
    (seq_dir / "labels").mkdir(parents=True)
    for i in range(n_scans):
        (seq_dir / "velodyne" / f"{i:06d}.bin").write_bytes(b"")
    for i in range(n_labels):
        (seq_dir / "labels" / f"{i:06d}.label").write_bytes(b"")
    (seq_dir / "calib.txt").write_text("", encoding="utf-8")
    poses = seq_dir / "poses.txt" if poses_in_seq else root / "poses" / f"{seq}.txt"
    poses.parent.mkdir(parents=True, exist_ok=True)
    poses.write_text("", encoding="utf-8")


def test_missing_data_is_warning_until_required(doctor: ModuleType, tmp_path: Path) -> None:
    (check,) = doctor.check_data(tmp_path, ["08"], required=False)
    assert check.status == doctor.WARN
    assert "no velodyne" in check.detail
    (check,) = doctor.check_data(tmp_path, ["08"], required=True)
    assert check.status == doctor.FAIL


def test_complete_sequence_is_ok(doctor: ModuleType, tmp_path: Path) -> None:
    _make_sequence(tmp_path, "08", n_scans=3, n_labels=3)
    (check,) = doctor.check_data(tmp_path, ["08"], required=True)
    assert check.status == doctor.OK
    assert "3 scans" in check.detail


def test_poses_found_in_shipped_location(doctor: ModuleType, tmp_path: Path) -> None:
    _make_sequence(tmp_path, "04", n_scans=2, n_labels=2, poses_in_seq=False)
    (check,) = doctor.check_data(tmp_path, ["04"], required=True)
    assert check.status == doctor.OK


def test_label_count_mismatch_reported(doctor: ModuleType, tmp_path: Path) -> None:
    _make_sequence(tmp_path, "07", n_scans=3, n_labels=2)
    (check,) = doctor.check_data(tmp_path, ["07"], required=True)
    assert check.status == doctor.FAIL
    assert "2 labels vs 3 scans" in check.detail


def test_weights_warning_until_required(doctor: ModuleType) -> None:
    # configs/model.yaml ships with name/checkpoint null until Phase 8.
    assert doctor.check_weights(required=False).status == doctor.WARN
    assert doctor.check_weights(required=True).status == doctor.FAIL


def test_main_exit_code_reflects_failures(
    doctor: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = doctor.main(["--data-root", str(tmp_path), "--sequences", "08", "--require-data"])
    out = capsys.readouterr().out
    assert code == 1
    assert "data:08" in out


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("g++.exe (Rev6, Built by MSYS2 project) 13.1.0", (13, 1, 0)),
        ("cmake version 3.28.1", (3, 28, 1)),
        ("v20.20.0", (20, 20, 0)),
        ("gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0", (11, 4, 0)),
        ("Apple clang version 15.0.0 (clang-1500.3.9.4)", (15, 0, 0)),
        ("no version here", ()),
    ],
)
def test_version_tuple(doctor: ModuleType, text: str, expected: tuple[int, ...]) -> None:
    assert doctor._version_tuple(text) == expected
