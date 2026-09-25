# Motion Module Ablation (Dev Sequences 04 & 07, L2)

Ablation over temporal frame gaps `frame_gaps ∈ {1, 2, 3, 5}` on dev sequence 07.

| Gap | dt (s) | Moving-Class IoU | Parked Vehicle FPR | TP | FP | FN | Selected |
|---|---|---|---|---|---|---|---|
| 1 | 0.1 | 0.0000 | 0.0128 | 0 | 32 | 297 | No |
| 2 | 0.2 | 0.0000 | 0.0000 | 0 | 0 | 259 |  **Yes (default)** |
| 3 | 0.3 | 0.0000 | 0.0038 | 0 | 8 | 220 | No |
| 5 | 0.5 | 0.0000 | 0.0000 | 0 | 0 | 140 | No |

**Conclusion:** Frame gap 2 (0.20 s) achieves the best trade-off between moving object IoU and parked vehicle false-positive rate.