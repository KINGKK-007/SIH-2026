"""foveamap.models.base — Segmenter Protocol and SegOutput dataclass.

All segmenter backends must satisfy the :class:`Segmenter` structural protocol.
Using ``typing.Protocol`` (PEP 544) means no inheritance is required — any
class that exposes the right attributes and ``__call__`` signature is accepted.

This file is intentionally dependency-free (only stdlib + numpy) so it can be
imported in environments without heavy ML frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegOutput:
    """Output of a segmenter call.

    Attributes
    ----------
    label : NDArray[uint16]
        Per-point semantic label, shape ``(N,)``.
        For oracle / GT mode: raw SemanticKITTI IDs (uint16, lower 16 bits
        already extracted).
        For pretrained network mode: 19-class benchmark IDs.
        Use :func:`foveamap.io.labels.to_superclass` or
        :func:`foveamap.io.labels.cls19_to_superclass` to convert.
    conf : NDArray[float16]
        Per-point prediction confidence in ``[0, 1]``, shape ``(N,)``.
        Oracle segmenter fills this with ``1.0``; network backends use softmax
        probability of the predicted class.
    """

    label: NDArray[np.uint16]
    conf: NDArray[np.float16]


@runtime_checkable
class Segmenter(Protocol):
    """Structural protocol for all FoveaMap segmenter backends.

    A backend must expose:
    - ``name`` (``str``) — human-readable identifier used in logs and results.
    - ``__call__`` — maps an ``(N, 4)`` point array to a :class:`SegOutput`.

    Example
    -------
    >>> import numpy as np
    >>> from foveamap_legacy.models.base import Segmenter, SegOutput
    >>> class MySegmenter:
    ...     name = "my_seg"
    ...     def __call__(self, points):
    ...         N = len(points)
    ...         return SegOutput(
    ...             label=np.zeros(N, dtype=np.uint16),
    ...             conf=np.ones(N, dtype=np.float16),
    ...         )
    >>> isinstance(MySegmenter(), Segmenter)
    True
    """

    name: str

    def __call__(self, points: NDArray[np.float32]) -> SegOutput:
        """Run semantic segmentation on a point cloud.

        Parameters
        ----------
        points : NDArray[float32]
            Shape ``(N, 4)`` — ``[x, y, z, intensity]`` in the Velodyne frame.

        Returns
        -------
        SegOutput
            Per-point labels and confidences.
        """
        ...
