"""Lazy, indexable access to one SemanticKITTI sequence (README 9.2, task T3.4)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from foveamap.pipeline.records import Scan


class Sequence:
    """Lazy, indexable sequence of :class:`Scan` records (supports ``frame_stride``)."""

    def __init__(self, root: str | Path, seq: str, with_labels: bool = True, frame_stride: int = 1) -> None:
        raise NotImplementedError("Implemented in Phase 3, T3.4 (docs/PHASES.md).")

    def __len__(self) -> int:
        raise NotImplementedError("Implemented in Phase 3, T3.4 (docs/PHASES.md).")

    def __getitem__(self, idx: int) -> Scan:
        raise NotImplementedError("Implemented in Phase 3, T3.4 (docs/PHASES.md).")

    def __iter__(self) -> Iterator[Scan]:
        raise NotImplementedError("Implemented in Phase 3, T3.4 (docs/PHASES.md).")
