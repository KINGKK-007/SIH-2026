"""foveamap.models.oracle — Oracle (ground-truth) segmenter.

The OracleSegmenter passes GT labels directly through the segmenter interface.
It is:

- Used in **oracle mode** for all pre-Phase-3 testing of the grid engine and
  dashboard (master plan §4.3, design idea 1).
- The accuracy ceiling for E4 (cost-of-coarsening experiment).
- Always labelled "oracle" in every result, figure and table so reviewers are
  never misled.

Usage
-----
>>> import numpy as np
>>> from foveamap_legacy.models.oracle import OracleSegmenter
>>> seg = OracleSegmenter(gt_labels=np.array([40, 40, 30], dtype=np.uint32))
>>> out = seg(np.zeros((3, 4), dtype=np.float32))
>>> list(out.label)
[40, 40, 30]
>>> all(c == 1.0 for c in out.conf)
True

Grid-engine integration
-----------------------
The segmenter also exposes ``moving_mask`` and ``vru_mask`` properties that
are computed from the raw GT labels.  The grid engine (Phase 2) uses these
directly so oracle mode does not need a separate motion module.

>>> seg = OracleSegmenter(gt_labels=np.array([40, 252, 30], dtype=np.uint32))
>>> list(seg.moving_mask)
[False, True, False]
>>> list(seg.vru_mask)
[False, False, True]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from foveamap_legacy.io.labels import (
    to_superclass,
    unpack_kitti_labels,
)
from foveamap_legacy.models.base import SegOutput

# Raw IDs 252–259 are the SemanticKITTI "moving-*" variants.
_MOVING_MIN: int = 252
_MOVING_MAX: int = 259

# VRU raw IDs: person (30), bicyclist (31), motorcyclist (32).
# These are always DYNAMIC regardless of the motion flag (H6).
_VRU_IDS: frozenset[int] = frozenset({30, 31, 32})


class OracleSegmenter:
    """Ground-truth pass-through segmenter.

    Parameters
    ----------
    gt_labels : NDArray[uint32]
        Raw ``uint32`` SemanticKITTI labels for the current scan, shape
        ``(N,)``.  Must be aligned with the point cloud passed to
        ``__call__``.

    Attributes
    ----------
    name : str
        ``"oracle"`` — used in result logs and figure titles.
    """

    name: str = "oracle"

    def __init__(self, gt_labels: NDArray[np.uint32]) -> None:
        self._gt_labels = np.asarray(gt_labels, dtype=np.uint32)
        self._update_derived()

    def _update_derived(self) -> None:
        """Recompute all derived arrays from current GT labels."""
        semantic_ids, _ = unpack_kitti_labels(self._gt_labels)
        self._semantic_ids: NDArray[np.uint16] = semantic_ids

        # moving_mask: True for raw IDs in the "moving-*" range 252–259
        # (these are the KITTI convention for annotated motion)
        self._moving_mask: NDArray[np.bool_] = (
            (semantic_ids >= _MOVING_MIN) & (semantic_ids <= _MOVING_MAX)
        )

        # vru_mask: True for persons / bicyclists / motorcyclists (30, 31, 32)
        # These are always DYNAMIC per H6 (even if standing still).
        vru_arr = np.zeros(len(semantic_ids), dtype=np.bool_)
        for _id in _VRU_IDS:
            vru_arr |= (semantic_ids == _id)
        self._vru_mask: NDArray[np.bool_] = vru_arr

        # super-class array (for direct use by grid engine)
        self._super_cls: NDArray[np.uint8] = to_superclass(semantic_ids)

    def update_labels(self, gt_labels: NDArray[np.uint32]) -> None:
        """Replace the stored GT labels (call once per new scan).

        Parameters
        ----------
        gt_labels : NDArray[uint32]
            Raw label array for the new scan.
        """
        self._gt_labels = np.asarray(gt_labels, dtype=np.uint32)
        self._update_derived()

    # ── Segmenter Protocol ────────────────────────────────────────────────

    def __call__(self, points: NDArray[np.float32]) -> SegOutput:
        """Return GT labels as :class:`~foveamap.models.base.SegOutput`.

        Parameters
        ----------
        points : NDArray[float32]
            Shape ``(N, 4)`` — present for interface compatibility; not used.

        Returns
        -------
        SegOutput
            ``label`` holds raw semantic IDs (uint16, lower 16 bits);
            ``conf`` is all-ones (oracle has perfect confidence).

        Raises
        ------
        ValueError
            If ``len(points) != len(gt_labels)``.
        """
        N = len(points)
        if N != len(self._gt_labels):
            raise ValueError(
                f"OracleSegmenter: points ({N}) and gt_labels "
                f"({len(self._gt_labels)}) must have the same length."
            )
        conf = np.ones(N, dtype=np.float16)
        return SegOutput(label=self._semantic_ids, conf=conf)

    # ── Grid-engine helpers ───────────────────────────────────────────────

    @property
    def moving_mask(self) -> NDArray[np.bool_]:
        """Bool array — True for points with raw ID ∈ 252–259 (moving-*)."""
        return self._moving_mask

    @property
    def vru_mask(self) -> NDArray[np.bool_]:
        """Bool array — True for person/bicyclist/motorcyclist (30, 31, 32)."""
        return self._vru_mask

    @property
    def super_cls(self) -> NDArray[np.uint8]:
        """Super-class IDs for the current scan (uint8, shape (N,))."""
        return self._super_cls

    def grid_inputs(
        self, points: NDArray[np.float32]
    ) -> tuple[
        NDArray[np.uint8],   # super_cls
        NDArray[np.bool_],   # moving
        NDArray[np.bool_],   # vru
        NDArray[np.float16], # conf
    ]:
        """Convenience: return all four arrays needed by ClipmapGrid.build().

        Parameters
        ----------
        points : (N, 4) float32
            Used only for length validation.

        Returns
        -------
        super_cls, moving, vru, conf
        """
        N = len(points)
        if N != len(self._gt_labels):
            raise ValueError(
                f"OracleSegmenter: points ({N}) and gt_labels "
                f"({len(self._gt_labels)}) must have the same length."
            )
        conf = np.ones(N, dtype=np.float16)
        return self._super_cls, self._moving_mask, self._vru_mask, conf

    def __repr__(self) -> str:
        return f"OracleSegmenter(n_labels={len(self._gt_labels)})"
