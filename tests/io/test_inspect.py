"""T2.3: ``foveamap inspect`` on the synthetic sequence."""

from __future__ import annotations

from pathlib import Path

import pytest

from foveamap import cli


def test_inspect_prints_and_writes_png(
    synthetic_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["inspect", "--sequence", "08", "--idx", "3", "--data-root", str(synthetic_root)]
    assert cli.main([*args, "--out-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "points:" in out and "road" in out and "moving-car" in out
    assert "DRIVABLE" in out and "camera-0 pose" in out
    assert "x=2.4" in out  # 3 frames at 8 m/s
    png = tmp_path / "inspect_08_000003.png"
    assert png.is_file() and png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_inspect_missing_data_fails_cleanly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cli.main(["inspect", "--data-root", str(tmp_path)])
