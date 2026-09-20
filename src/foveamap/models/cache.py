"""foveamap.models.cache — Cached semantic segmentation reader.

Implements :class:`CacheSegmenter`, conforming to the :class:`~foveamap.models.base.Segmenter`
protocol.

Purpose
-------
Decouples grid engine validation and evaluation from live GPU inference.
Precomputed predictions from range-view networks (e.g. FRNet, CENet, SalsaNext)
or test fixtures can be saved to disk as ``.npy`` (19-class IDs or raw IDs) or
``.label`` (SemanticKITTI uint32 format) and loaded on any CPU workstation.

Taxonomy mapping
----------------
- 19-class benchmark predictions are mapped to the 4 super-classes via
  :func:`~foveamap.io.labels.cls19_to_superclass`.
- Raw SemanticKITTI uint32 labels are unpacked and mapped via
  :func:`~foveamap.io.labels.to_superclass`.
- Per-point confidence scores are loaded if available (e.g. ``*_conf.npy``),
  or assigned a configurable default (e.g. 0.9).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from foveamap.io.labels import (
    cls19_to_superclass,
    to_superclass,
    unpack_kitti_labels,
)
from foveamap.models.base import SegOutput

# VRU benchmark classes in 19-class format: 8: person, 9: bicyclist, 10: motorcyclist
_VRU_19_CLASSES: frozenset[int] = frozenset({8, 9, 10})

# VRU raw SemanticKITTI IDs: 30: person, 31: bicyclist, 32: motorcyclist
_VRU_RAW_IDS: frozenset[int] = frozenset({30, 31, 32})

# Raw moving IDs: 252-259
_RAW_MOVING_MIN: int = 252
_RAW_MOVING_MAX: int = 259


class CacheSegmenter:
    """Segmenter that serves precomputed predictions from disk or memory.

    Conforms to the :class:`~foveamap.models.base.Segmenter` Protocol.

    Parameters
    ----------
    cache_dir : str | Path, optional
        Root directory where cached predictions are stored. If provided,
        lookups can search ``cache_dir / sequence`` or ``cache_dir``.
    sequence : str, default "08"
        Sequence identifier (e.g. "08").
    name : str, default "cached_pred"
        Model or source identifier for logging and benchmarks.
    default_conf : float, default 0.9
        Default confidence value in ``[0.0, 1.0]`` when per-point confidence
        is not provided.
    labels : NDArray, optional
        Initial label array (uint32 raw, uint16 raw semantic, or uint8 19-class).
    conf : NDArray, optional
        Initial per-point confidence array.
    """

    name: str

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        sequence: str = "08",
        name: str = "cached_pred",
        default_conf: float = 0.9,
        labels: NDArray | None = None,
        conf: NDArray | None = None,
    ) -> None:
        self.name = name
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.sequence = sequence
        self.default_conf = float(default_conf)

        self._labels: NDArray[np.uint16] = np.empty(0, dtype=np.uint16)
        self._conf: NDArray[np.float16] = np.empty(0, dtype=np.float16)
        self._super_cls: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
        self._moving_mask: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)
        self._vru_mask: NDArray[np.bool_] = np.empty(0, dtype=np.bool_)

        if labels is not None:
            self.set_predictions(labels, conf=conf)

    def _resolve_sequence_dir(self) -> Path | None:
        """Find the directory containing sequence predictions."""
        if self.cache_dir is None:
            return None
        # Check cache_dir / sequence / predictions, cache_dir / sequence, or cache_dir
        candidates = [
            self.cache_dir / "sequences" / self.sequence / "predictions",
            self.cache_dir / "sequences" / self.sequence,
            self.cache_dir / self.sequence,
            self.cache_dir,
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                return c
        return self.cache_dir

    def load_frame(self, frame_idx: int) -> None:
        """Load predictions for a specific frame index from cache_dir.

        Parameters
        ----------
        frame_idx : int
            Scan frame index (e.g. 0, 1, 2).

        Raises
        ------
        FileNotFoundError
            If no matching prediction file is found for ``frame_idx``.
        """
        seq_dir = self._resolve_sequence_dir()
        if seq_dir is None or not seq_dir.exists():
            raise FileNotFoundError(
                f"Cache directory does not exist or was not specified: {self.cache_dir}"
            )

        # Look for {frame:06d}.npy, {frame:06d}.label, {frame:04d}.npy, etc.
        patterns = [
            f"{frame_idx:06d}.npy",
            f"{frame_idx:06d}.label",
            f"{frame_idx:04d}.npy",
            f"{frame_idx:04d}.label",
        ]

        found_path: Path | None = None
        for pat in patterns:
            candidate = seq_dir / pat
            if candidate.exists():
                found_path = candidate
                break

        if found_path is None:
            raise FileNotFoundError(
                f"No cached prediction found for frame {frame_idx} in {seq_dir}. "
                f"Tried patterns: {patterns}"
            )

        self.load_file(found_path)

    def load_file(self, file_path: str | Path) -> None:
        """Load predictions directly from a file path.

        Parameters
        ----------
        file_path : str | Path
            Path to ``.npy`` or ``.label`` file.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Prediction file does not exist: {path}")

        conf_arr: NDArray[np.float16] | None = None
        # Look for matching confidence file
        for conf_suffix in ["_conf.npy", ".conf.npy"]:
            conf_cand = path.with_name(f"{path.stem}{conf_suffix}")
            if conf_cand.exists():
                raw_c = np.load(conf_cand)
                conf_arr = np.clip(raw_c, 0.0, 1.0).astype(np.float16)
                break

        if path.suffix == ".label":
            raw = np.fromfile(path, dtype=np.uint32)
            self.set_predictions(raw, conf=conf_arr, is_raw_kitti=True)
        elif path.suffix == ".npy":
            arr = np.load(path)
            self.set_predictions(arr, conf=conf_arr)
        else:
            raise ValueError(f"Unsupported cached prediction format: {path.suffix} (expected .npy or .label)")

    def set_predictions(
        self,
        labels: NDArray,
        conf: NDArray | None = None,
        is_raw_kitti: bool | None = None,
    ) -> None:
        """Set in-memory predictions and compute derived super-classes.

        Parameters
        ----------
        labels : NDArray
            Label array. Can be:
            - uint32 raw SemanticKITTI format (upper 16 = instance, lower 16 = semantic)
            - uint8 / int array of 19-class benchmark IDs (0..18, 255)
            - uint16 raw semantic IDs
        conf : NDArray, optional
            Per-point confidence in [0, 1].
        is_raw_kitti : bool, optional
            Explicitly flag whether labels are raw uint32 SemanticKITTI labels.
            If None, inferred automatically from dtype and max values.
        """
        labels_arr = np.asarray(labels)
        N = len(labels_arr)

        if is_raw_kitti is None:
            is_raw_kitti = (labels_arr.dtype == np.uint32) or (labels_arr.size > 0 and np.max(labels_arr) > 255)

        if is_raw_kitti:
            semantic_ids, _ = unpack_kitti_labels(labels_arr.astype(np.uint32))
            self._labels = semantic_ids.astype(np.uint16)
            self._super_cls = to_superclass(semantic_ids)

            # Moving mask: raw IDs 252-259
            self._moving_mask = (
                (semantic_ids >= _RAW_MOVING_MIN) & (semantic_ids <= _RAW_MOVING_MAX)
            )

            # VRU mask: raw IDs 30, 31, 32
            vru = np.zeros(N, dtype=np.bool_)
            for vid in _VRU_RAW_IDS:
                vru |= (semantic_ids == vid)
            self._vru_mask = vru
        else:
            # 19-class benchmark format (or directly super-class)
            c19_ids = labels_arr.astype(np.uint8)
            self._labels = c19_ids.astype(np.uint16)
            self._super_cls = cls19_to_superclass(c19_ids)

            # 19-class alone does not distinguish moving vs static vehicles
            # (motion module will set this via set_moving_mask).
            self._moving_mask = np.zeros(N, dtype=np.bool_)

            # VRU mask in 19-class space (person=8, bicyclist=9, motorcyclist=10)
            vru = np.zeros(N, dtype=np.bool_)
            for vid in _VRU_19_CLASSES:
                vru |= (c19_ids == vid)
            self._vru_mask = vru

        if conf is not None:
            c_arr = np.asarray(conf, dtype=np.float32)
            if len(c_arr) != N:
                raise ValueError(f"Confidence length ({len(c_arr)}) does not match labels ({N})")
            self._conf = np.clip(c_arr, 0.0, 1.0).astype(np.float16)
        else:
            self._conf = np.full(N, self.default_conf, dtype=np.float16)

    def set_moving_mask(self, moving_mask: NDArray[np.bool_]) -> None:
        """Update the moving mask (e.g. from the Phase 4 motion engine).

        Parameters
        ----------
        moving_mask : NDArray[bool]
            Boolean array of shape ``(N,)``.
        """
        mask = np.asarray(moving_mask, dtype=np.bool_)
        if len(mask) != len(self._labels):
            raise ValueError(
                f"moving_mask length ({len(mask)}) does not match labels ({len(self._labels)})"
            )
        self._moving_mask = mask

    # ── Segmenter Protocol ───────────────────────────────────────────────────

    def __call__(self, points: NDArray[np.float32]) -> SegOutput:
        """Return cached labels and confidences as :class:`SegOutput`.

        Parameters
        ----------
        points : NDArray[float32]
            Shape ``(N, 4)`` point cloud array in Velodyne frame.

        Returns
        -------
        SegOutput
            Per-point labels and confidences.

        Raises
        ------
        ValueError
            If the number of points does not match the cached predictions.
        """
        N = len(points)
        if len(self._labels) != N:
            raise ValueError(
                f"CacheSegmenter '{self.name}': point count ({N}) does not match "
                f"cached predictions length ({len(self._labels)})."
            )
        return SegOutput(label=self._labels, conf=self._conf)

    # ── Grid-engine helpers ──────────────────────────────────────────────────

    @property
    def super_cls(self) -> NDArray[np.uint8]:
        """Per-point super-class IDs (0=DRIVABLE, 1=TERRAIN, 2=OBSTACLE, 3=DYNAMIC, 255=UNKNOWN)."""
        return self._super_cls

    @property
    def moving_mask(self) -> NDArray[np.bool_]:
        """Bool array — True for points identified as moving."""
        return self._moving_mask

    @property
    def vru_mask(self) -> NDArray[np.bool_]:
        """Bool array — True for vulnerable road users (pedestrians, cyclists)."""
        return self._vru_mask

    @property
    def conf(self) -> NDArray[np.float16]:
        """Per-point confidence values in ``[0, 1]``."""
        return self._conf

    def grid_inputs(
        self, points: NDArray[np.float32]
    ) -> tuple[
        NDArray[np.uint8],
        NDArray[np.bool_],
        NDArray[np.bool_],
        NDArray[np.float16],
    ]:
        """Return all four arrays needed by :meth:`ClipmapGrid.build`.

        Parameters
        ----------
        points : NDArray[float32]
            Points array used for length validation.

        Returns
        -------
        super_cls : NDArray[uint8]
        moving : NDArray[bool]
        vru : NDArray[bool]
        conf : NDArray[float16]
        """
        N = len(points)
        if len(self._labels) != N:
            raise ValueError(
                f"CacheSegmenter '{self.name}': point count ({N}) does not match "
                f"cached predictions length ({len(self._labels)})."
            )
        return self._super_cls, self._moving_mask, self._vru_mask, self._conf

    def __repr__(self) -> str:
        return (
            f"CacheSegmenter(name={self.name!r}, n_points={len(self._labels)}, "
            f"default_conf={self.default_conf})"
        )
