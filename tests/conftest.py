"""Shared pytest configuration (README 11.1).

Fixtures for the tiny presets, the seeded synthetic scan generator and the one-scan real sample are
added with the phases that need them (Phases 3 and 5).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

LEGACY_DIR = Path(__file__).parent / "legacy"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if LEGACY_DIR in Path(str(item.path)).parents:
            item.add_marker(pytest.mark.legacy)


# ── real SemanticKITTI data (tests marked @pytest.mark.data) ────────────────
REAL_DATA_ROOT = Path(
    os.environ.get("FOVEAMAP_DATA_ROOT", Path(__file__).resolve().parents[1] / "data" / "dataset")
)


@pytest.fixture(scope="session")
def real_data_root() -> Path:
    """The SemanticKITTI root (``data/dataset`` or ``$FOVEAMAP_DATA_ROOT``); skips if absent."""
    if not (REAL_DATA_ROOT / "sequences").is_dir():
        pytest.skip(f"SemanticKITTI not found at {REAL_DATA_ROOT} (download per README 4.3)")
    return REAL_DATA_ROOT


@pytest.fixture(scope="session")
def real_sequences(real_data_root: Path) -> list[str]:
    """Sequences among 04/07/08 that have scans and labels on disk."""
    found = [
        s
        for s in ("04", "07", "08")
        if any((real_data_root / "sequences" / s / "velodyne").glob("*.bin"))
        and (real_data_root / "sequences" / s / "labels").is_dir()
    ]
    if not found:
        pytest.skip("none of sequences 04/07/08 is present")
    return found
