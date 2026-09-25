"""LSK3DNet adapter stub (Phase 8, T8.1).

The actual implementation lives in the LSK3DNet source files that you will upload.
This file only defines the integration interface that FoveaMap expects.

**To complete Phase 8:**
1. Upload the LSK3DNet source files to ``data/lsk3dnet_src/`` (or set ``LSK3DNet_SRC``).
2. Place your checkpoint at the path in ``configs/model.yaml`` → ``checkpoint:``.
3. Implement :class:`LSK3DNetModel` below (or replace this file with your adapter).

The class must satisfy the :class:`~foveamap.models.base.SegmentationModel` protocol::

    class SegmentationModel(Protocol):
        name: str
        def predict(self, scan: Scan) -> Prediction: ...

Where :class:`~foveamap.pipeline.records.Prediction` is::

    @dataclass
    class Prediction:
        raw_ids: np.ndarray  # (N,) uint16 — raw SemanticKITTI IDs (via learning_map_inv)
        conf:    np.ndarray  # (N,) uint8  — confidence 0–255

See ``docs/SETUP_MODEL.md`` for the full integration checklist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foveamap.config import ModelConfig
    from foveamap.pipeline.records import Prediction, Scan


class LSK3DNetModel:
    """Placeholder — implement after uploading LSK3DNet source files."""

    name: str = "lsk3dnet"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "LSK3DNetModel is not implemented yet.\n"
            "Upload the LSK3DNet source files and implement this class per "
            "docs/SETUP_MODEL.md."
        )

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> LSK3DNetModel:
        raise NotImplementedError(
            "LSK3DNetModel.from_config() is not implemented yet.\n"
            "See docs/SETUP_MODEL.md."
        )

    def predict(self, scan: Scan) -> Prediction:  # type: ignore[empty-body]
        raise NotImplementedError("predict() is not implemented yet.")
