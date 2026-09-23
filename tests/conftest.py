"""Shared pytest configuration (README 11.1).

Fixtures for the tiny presets, the seeded synthetic scan generator and the one-scan real sample are
added with the phases that need them (Phases 3 and 5).
"""

from __future__ import annotations

from pathlib import Path

import pytest

LEGACY_DIR = Path(__file__).parent / "legacy"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    for item in items:
        if LEGACY_DIR in Path(str(item.path)).parents:
            item.add_marker(pytest.mark.legacy)
