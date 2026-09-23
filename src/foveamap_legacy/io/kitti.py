"""foveamap.io.kitti — Binary point-cloud and label-file readers.

File formats
------------
Velodyne binary (``.bin``)
    Raw little-endian float32 values stored in groups of 4:
    ``x, y, z, intensity`` (in metres and arbitrary units respectively).
    One scan from a Velodyne HDL-64E contains ≈ 120,000 points.

Label file (``.label``)
    One ``uint32`` value per point (same order as the matching ``.bin`` file).
    Bit layout: ``[instance_id(16) | semantic_id(16)]``.
    Use :func:`foveamap.io.labels.unpack_kitti_labels` to split the words.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def load_velodyne_bin(bin_path: str | Path) -> NDArray[np.float32]:
    """Load a Velodyne HDL-64E scan from a SemanticKITTI ``.bin`` file.

    Parameters
    ----------
    bin_path:
        Path to the ``.bin`` file (e.g.
        ``data/semkitti/sequences/08/velodyne/000000.bin``).

    Returns
    -------
    points : NDArray[float32]
        Array of shape ``(N, 4)`` with columns ``[x, y, z, intensity]``.
        Coordinates are in the **Velodyne frame** (x forward, y left, z up).
        The sensor is mounted ≈ 1.73 m above the ground, so flat road sits
        near ``z ≈ −1.73 m``.

    Raises
    ------
    FileNotFoundError
        If ``bin_path`` does not exist.
    ValueError
        If the file size is not a multiple of 16 bytes (4 × float32).

    Examples
    --------
    >>> import tempfile, numpy as np
    >>> data = np.array([[1.0, 2.0, -1.73, 0.5]], dtype=np.float32)
    >>> with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
    ...     _ = f.write(data.tobytes())
    ...     path = f.name
    >>> pts = load_velodyne_bin(path)
    >>> pts.shape
    (1, 4)
    >>> pts.dtype
    dtype('float32')
    """
    bin_path = Path(bin_path)
    if not bin_path.exists():
        raise FileNotFoundError(f"Velodyne bin not found: {bin_path}")

    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 4 != 0:
        raise ValueError(
            f"File size {raw.size * 4} bytes is not a multiple of 16 (4 × float32): {bin_path}"
        )

    return raw.reshape(-1, 4)


def load_labels(label_path: str | Path) -> NDArray[np.uint32]:
    """Load per-point labels from a SemanticKITTI ``.label`` file.

    Parameters
    ----------
    label_path:
        Path to the ``.label`` file (e.g.
        ``data/semkitti/sequences/08/labels/000000.label``).

    Returns
    -------
    labels : NDArray[uint32]
        Flat array of shape ``(N,)`` with raw ``uint32`` label values.
        Split into semantic / instance IDs with
        :func:`foveamap.io.labels.unpack_kitti_labels`.

    Raises
    ------
    FileNotFoundError
        If ``label_path`` does not exist.

    Examples
    --------
    >>> import tempfile, numpy as np
    >>> lab = np.array([0x00000028], dtype=np.uint32)  # semantic=40 (road)
    >>> with tempfile.NamedTemporaryFile(suffix='.label', delete=False) as f:
    ...     _ = f.write(lab.tobytes())
    ...     path = f.name
    >>> out = load_labels(path)
    >>> out.dtype, out.shape
    (dtype('uint32'), (1,))
    >>> hex(int(out[0]))
    '0x28'
    """
    label_path = Path(label_path)
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    return np.fromfile(label_path, dtype=np.uint32)
