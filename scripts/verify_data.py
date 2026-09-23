"""Verify the SemanticKITTI layout, counts and formats (README 4.3, 5.1, task T2.2).

Usage: python scripts/verify_data.py --sequences 04 07 08 [--data-root data/dataset] [--json out.json]

Checks per sequence: folder layout; #labels == #scans with matching names; len(label) == N for every
scan (from file sizes); #poses == #scans; calib.txt and poses parse; times.txt count; a sample of scans
loads with finite values. If poses exist only as poses/<seq>.txt they are linked (or copied) into
sequences/<seq>/poses.txt. Exit code 0 only if every sequence passes.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

from foveamap.io.verify import format_report, verify_dataset

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sequences", nargs="+", default=["04", "07", "08"])
    parser.add_argument(
        "--data-root", default=os.environ.get("FOVEAMAP_DATA_ROOT", str(ROOT / "data" / "dataset"))
    )
    parser.add_argument("--sample", type=int, default=50, help="scans loaded per sequence for value checks")
    parser.add_argument("--no-normalise", action="store_true", help="do not link poses/<seq>.txt")
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    reports = verify_dataset(
        args.data_root, args.sequences, sample=args.sample, normalise=not args.no_normalise
    )
    print(f"data root: {args.data_root}")
    print(format_report(reports))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        payload = [dataclasses.asdict(r) | {"ok": r.ok} for r in reports]
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if all(r.ok for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
