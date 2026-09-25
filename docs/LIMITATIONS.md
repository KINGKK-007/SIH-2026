# Limitations & System Boundaries

This document provides an honest, empirical accounting of system boundaries, assumptions, and failure modes across all modules.

---

## 1. Geometric Motion & Dynamic Object Detection (Phase 10)

Per **README Section 6.4** and Locked Decision **L12**, moving object state is computed purely geometrically by `foveamap.motion` using ego-compensated range-image residuals and range-scaled Euclidean clustering, rather than relying on multi-scan neural network features.

### Strengths & Architectural Guarantees
1. **Zero Deep-Learning / GPU Dependency (R13):** The entire motion pipeline runs deterministically in pure NumPy/SciPy on CPU (~5–15 ms per frame), preserving strict decoupling from neural network inference and requiring 0 MB of GPU VRAM.
2. **Deterministic Safety Priority:** Vulnerable Road Users (VRUs: pedestrians, bicyclists, motorcyclists) are categorically elevated to `DYNAMIC` and flagged `safety_critical = True` regardless of instantaneous point-residual motion votes.
3. **Exact Bounding Box Recovery:** The ground-plane angle sweep (`oriented_box`) deterministically recovers the minimum bounding rectangle in $O(K \cdot N)$ vectorized operations without heuristic non-maximum suppression or anchor boxes.
4. **Temporal Consistency:** Multi-frame nearest-neighbour tracking with velocity gating prevents flickering and ensures persistent object identity across short occlusions.

### Known Limitations & Failure Modes
1. **Strictly Radial & Creeping Motion:** Objects moving directly along the LiDAR beam (pure radial trajectory) or creeping below $\approx 0.2\text{ m/s}$ (e.g. slow pedestrians traversing only 10–14 cm over 0.1 s) produce range residuals near or below the sensor noise floor $\tau(r) = \tau_0 + \tau_1 \cdot r$.
2. **Disocclusion Artefacts:** When a moving foreground obstacle reveals a previously occluded background surface, the sudden range jump can mimic motion without careful occlusion handling (`occlusion_rule: true`).
3. **Odometry & Pose Sensitivity:** Ego-motion compensation strictly relies on accurate SE(3) transformation between scans ($T_{\text{vel}, t \leftarrow t-\text{gap}}$). High angular velocity drift or uncompensated vehicle pitch would inject artificial residuals across the static ground.
4. **Upgrade Roadmap:** A future learned Moving Object Segmentation (MOS) network (e.g. multi-scan sparse voxel network or range-flow network) is identified in Section 16 as the targeted replacement for high-speed highway scenarios.

---
