"""foveamap.models — Segmenter backends.

All backends implement the :class:`Segmenter` Protocol defined in
:mod:`foveamap.models.base`.

Available backends
------------------
oracle  : Perfect ground-truth segmenter (no network; requires GT labels).
cache   : Load cached ``.label`` predictions from disk.
frnet   : FRNet range-view network (Phase 3 — stub for now).
cenet   : CENet range-view network (Phase 3 — stub for now).
"""

from foveamap.models.base import Segmenter, SegOutput
from foveamap.models.cache import CacheSegmenter
from foveamap.models.oracle import OracleSegmenter

__all__ = [
    "SegOutput",
    "Segmenter",
    "OracleSegmenter",
    "CacheSegmenter",
]
