"""Dataset verification behind ``scripts/verify_data.py`` (README 4.3, 5.1-5.2, task T2.2)."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from foveamap.io.kitti import LABEL_RECORD_BYTES, SCAN_RECORD_BYTES, load_scan_bin
from foveamap.io.poses import load_calib, load_poses


@dataclass
class SequenceReport:
    seq: str
    n_scans: int = 0
    n_labels: int = 0
    n_poses: int = 0
    n_times: int | None = None
    poses_source: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _normalise_poses(root: Path, seq: str, report: SequenceReport, normalise: bool) -> Path | None:
    """Make ``sequences/<seq>/poses.txt`` exist, linking (or copying) ``poses/<seq>.txt`` if needed."""
    in_seq = root / "sequences" / seq / "poses.txt"
    shipped = root / "poses" / f"{seq}.txt"
    if in_seq.is_file():
        report.poses_source = str(in_seq)
        return in_seq
    if not shipped.is_file():
        report.errors.append(f"no poses: neither {in_seq} nor {shipped} exists")
        return None
    report.poses_source = str(shipped)
    if not normalise:
        return shipped
    try:
        os.symlink(os.path.relpath(shipped, in_seq.parent), in_seq)
        report.notes.append(f"symlinked {in_seq} -> {shipped}")
    except OSError:  # e.g. Windows without symlink privilege
        shutil.copyfile(shipped, in_seq)
        report.notes.append(f"copied {shipped} -> {in_seq} (symlink not permitted)")
    return in_seq


def verify_sequence(root: Path, seq: str, sample: int = 50, normalise: bool = True) -> SequenceReport:
    """Check one sequence; never raises for data problems, records them in the report instead."""
    report = SequenceReport(seq)
    seq_dir = root / "sequences" / seq
    if not seq_dir.is_dir():
        report.errors.append(f"missing directory {seq_dir}")
        return report

    scans = sorted((seq_dir / "velodyne").glob("*.bin"))
    labels = sorted((seq_dir / "labels").glob("*.label"))
    report.n_scans, report.n_labels = len(scans), len(labels)
    if not scans:
        report.errors.append(f"no scans in {seq_dir / 'velodyne'}")
    if not labels:
        report.errors.append(f"no labels in {seq_dir / 'labels'}")
    scan_stems, label_stems = {p.stem for p in scans}, {p.stem for p in labels}
    if labels and scan_stems != label_stems:
        only_scan, only_label = sorted(scan_stems - label_stems), sorted(label_stems - scan_stems)
        report.errors.append(
            f"#labels ({len(labels)}) != #scans ({len(scans)}) or names differ: "
            f"{len(only_scan)} scans without label (e.g. {only_scan[:3]}), "
            f"{len(only_label)} labels without scan (e.g. {only_label[:3]})"
        )

    bad_sizes = []
    for scan in scans:
        size = scan.stat().st_size
        label = seq_dir / "labels" / (scan.stem + ".label")
        if size % SCAN_RECORD_BYTES:
            bad_sizes.append(f"{scan.name}: {size} bytes is not N x 16")
        elif label.is_file() and label.stat().st_size != (size // SCAN_RECORD_BYTES) * LABEL_RECORD_BYTES:
            n_points = size // SCAN_RECORD_BYTES
            n_label = label.stat().st_size // LABEL_RECORD_BYTES
            bad_sizes.append(f"{scan.stem}: {n_label} labels for {n_points} points")
    if bad_sizes:
        report.errors.append(f"{len(bad_sizes)} scans with len(label) != N, e.g. {bad_sizes[:3]}")

    try:
        calib = load_calib(seq_dir / "calib.txt")
        if abs(np.linalg.det(calib.Tr)) < 1e-6:
            report.errors.append("calib.txt: Tr is singular")
    except (OSError, ValueError) as exc:
        report.errors.append(f"calib.txt does not parse: {exc}")

    poses_file = _normalise_poses(root, seq, report, normalise)
    if poses_file is not None:
        try:
            poses = load_poses(poses_file)
            report.n_poses = len(poses)
            if len(poses) != len(scans):
                report.errors.append(f"#poses ({len(poses)}) != #scans ({len(scans)})")
            if not np.isfinite(poses).all():
                report.errors.append("poses contain non-finite values")
        except (OSError, ValueError) as exc:
            report.errors.append(f"poses do not parse: {exc}")

    times_file = seq_dir / "times.txt"
    if times_file.is_file():
        report.n_times = len(np.loadtxt(times_file, ndmin=1))
        if report.n_times != len(scans):
            report.errors.append(f"#times ({report.n_times}) != #scans ({len(scans)})")
    else:
        report.warnings.append("times.txt missing (Sequence falls back to 0.1 s spacing)")

    if scans and sample > 0:
        picks = np.unique(np.linspace(0, len(scans) - 1, min(sample, len(scans))).astype(int))
        for i in picks:
            try:
                xyz, rem = load_scan_bin(scans[i])
            except ValueError:
                continue  # malformed size, already reported above

            if not (np.isfinite(xyz).all() and np.isfinite(rem).all()):
                report.warnings.append(f"{scans[i].name}: non-finite values (counted as n_invalid later)")
                break
        report.notes.append(f"loaded {len(picks)} sample scans")
    return report


def verify_dataset(
    root: str | Path, sequences: list[str], sample: int = 50, normalise: bool = True
) -> list[SequenceReport]:
    root = Path(root)
    return [verify_sequence(root, seq, sample=sample, normalise=normalise) for seq in sequences]


def format_report(reports: list[SequenceReport]) -> str:
    lines = []
    for r in reports:
        status = "PASS" if r.ok else "FAIL"
        times = "-" if r.n_times is None else r.n_times
        lines.append(
            f"[{status}] {r.seq}: scans={r.n_scans} labels={r.n_labels} poses={r.n_poses} times={times}"
        )
        lines += [f"    error: {e}" for e in r.errors]
        lines += [f"    warning: {w}" for w in r.warnings]
        lines += [f"    note: {n}" for n in r.notes]
    n_fail = sum(not r.ok for r in reports)
    lines.append(f"{len(reports) - n_fail}/{len(reports)} sequences passed")
    return "\n".join(lines)
