"""FoveaMap — Foveated 2.5D LiDAR semantic mapping.

Foveated 2.5D LiDAR mapping: sharp where a mistake hurts, cheap where it doesn't.

Package layout
--------------
foveamap.io        : KITTI data readers, label maps, pose utilities, synthetic generator
foveamap.models    : Segmenter protocol, oracle and cache backends
foveamap.grid      : GridSpec and Ring dataclasses
foveamap.motion    : Scan-to-scan residual motion detection (Phase 4)
foveamap.derive    : Derived layers — slope, step, clearance, traversability (Phase 5)
foveamap.eval      : Metrics, range-bucket analysis, latency/memory accounting (Phase 6)
foveamap.viz       : Palette, top-down renderer, Streamlit dashboard (Phase 7)
foveamap.cli       : Click entry point — ``foveamap`` command
"""

__version__ = "0.1.0"
__author__ = "Kanav Kumar"
