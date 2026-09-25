"""Ground-truth labels as a model (README 9.3, R12, task T7.2).

``OracleModel`` is the upper bound for any segmentation model: it returns the scan's ground-truth
semantic IDs with maximum confidence (``conf = 255``).  Moving IDs are preserved so that the
motion stage sees realistic moving-point flags.

The oracle uses ``raw_to_super`` to check for unknown IDs (triggers a warning if any are found)
but returns the raw IDs unchanged through ``learning_map_inv`` so downstream stages that expect
raw SemanticKITTI IDs work correctly.
"""

from __future__ import annotations

import numpy as np

from foveamap.io.labels import semantic_ids
from foveamap.pipeline.records import Prediction, Scan


class OracleModel:
    """Returns the scan's ground-truth semantic IDs with ``conf = 255``; moving IDs are preserved.

    This model satisfies the :class:`~foveamap.models.base.SegmentationModel` protocol.
    ``raw_ids`` are the *semantic* part of the raw 32-bit label (``raw & 0xFFFF``) cast to
    ``uint16``; downstream, ``io.labels.raw_to_super`` maps them to super-classes.
    """

    name: str = "oracle"

    def predict(self, scan: Scan) -> Prediction:
        """Return ground-truth IDs with perfect confidence.

        Args:
            scan: A :class:`~foveamap.pipeline.records.Scan` with ``raw_labels`` populated.

        Returns:
            :class:`~foveamap.pipeline.records.Prediction` with ``raw_ids`` as ``uint16``
            semantic-only IDs and ``conf`` all 255.

        Raises:
            ValueError: If ``scan.raw_labels`` is ``None``.
        """
        if scan.raw_labels is None:
            raise ValueError(
                f"OracleModel requires ground-truth labels (scan {scan.seq}/{scan.idx:06d} "
                "has raw_labels=None)"
            )
        # semantic_ids extracts the lower 16 bits (works for both bare semantic and full uint32)
        raw_ids = semantic_ids(scan.raw_labels)  # (N,) uint16
        conf = np.full(len(raw_ids), 255, dtype=np.uint8)
        return Prediction(raw_ids=raw_ids, conf=conf)
