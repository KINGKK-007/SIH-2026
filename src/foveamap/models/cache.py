"""foveamap.models.cache — Cached-prediction segmenter.

Loads per-frame predictions that were previously cached to disk in
SemanticKITTI ``.label`` format (uint32, same bit layout as GT labels but
with 19-class IDs in the low word and instance = 0 in the high word).

Caching strategy (master plan §4.3, design idea 2)
---------------------------------------------------
A GPU is required **once** to run inference on the eval subset
(``foveamap cache --seq 08 --model frnet``).  The predictions are written to
``preds/<model>/sequences/<seq>/labels/<frame>.label``.  Subsequent runs of
the grid engine, evaluation and dashboard use ``CacheSegmenter`` and require
only a CPU.

Both GT and predicted labels share one file format, so one evaluator handles
both (``foveamap bench`` switches between them via ``--labels {gt,pred}``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from foveamap.models.base import SegOutput
from foveamap.io.kitti import load_labels


class CacheSegmenter:
    """Load pre-cached ``.label`` predictions for a given frame.

    Parameters
    ----------
    cache_dir : str | Path
        Root directory for cached label files.  Expected layout::

            <cache_dir>/sequences/<sequence>/labels/<frame_id:06d>.label

    sequence : str
        SemanticKITTI sequence identifier (e.g. ``"08"``).
    model_name : str
        Human-readable model identifier used in logs (e.g. ``"frnet"``).

    Attributes
    ----------
    name : str
        ``"cache:<model_name>"`` — recorded in results.

    Examples
    --------
    >>> from foveamap.models.cache import CacheSegmenter
    >>> seg = CacheSegmenter("/path/to/preds", "08", "frnet")
    >>> seg.name
    'cache:frnet'
    """

    def __init__(
        self,
        cache_dir: str | Path,
        sequence: str,
        model_name: str = "cached",
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._seq = sequence
        self._model_name = model_name
        self._current_frame: int | None = None
        self._current_labels: NDArray[np.uint32] | None = None

    @property
    def name(self) -> str:
        return f"cache:{self._model_name}"

    def load_frame(self, frame_id: int) -> None:
        """Pre-load labels for ``frame_id`` into memory.

        Call this before ``__call__`` when iterating over frames so disk I/O
        is measured separately from inference time.

        Parameters
        ----------
        frame_id : int
            Zero-based frame index within the sequence.
        """
        label_path = (
            self._cache_dir
            / "sequences"
            / self._seq
            / "labels"
            / f"{frame_id:06d}.label"
        )
        self._current_labels = load_labels(label_path)
        self._current_frame = frame_id

    def __call__(self, points: NDArray[np.float32]) -> SegOutput:
        """Return cached prediction labels for the current frame.

        Parameters
        ----------
        points : NDArray[float32]
            Shape ``(N, 4)`` — used only to validate size consistency.

        Returns
        -------
        SegOutput
            ``label`` holds the low-16-bit semantic IDs from the cached file;
            ``conf`` is all-ones (cached files don't store per-point
            confidence; this is a known limitation — see master plan §8).

        Raises
        ------
        RuntimeError
            If :meth:`load_frame` has not been called yet.
        ValueError
            If point count doesn't match cached label count.
        """
        if self._current_labels is None:
            raise RuntimeError(
                "CacheSegmenter: call load_frame(frame_id) before __call__."
            )
        N = len(points)
        if N != len(self._current_labels):
            raise ValueError(
                f"CacheSegmenter: points ({N}) != cached labels "
                f"({len(self._current_labels)}) for frame {self._current_frame}."
            )
        # Extract semantic IDs (low 16 bits)
        semantic_ids = (self._current_labels & 0xFFFF).astype(np.uint16)
        conf = np.ones(N, dtype=np.float16)
        return SegOutput(label=semantic_ids, conf=conf)

    def __repr__(self) -> str:
        return (
            f"CacheSegmenter(cache_dir='{self._cache_dir}', "
            f"seq='{self._seq}', model='{self._model_name}')"
        )
