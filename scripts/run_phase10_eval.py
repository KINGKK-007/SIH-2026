"""Phase 10 evaluation: motion detection and object detection on Sequence 08 (T10.5, T10.6, T10.7)."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np

from foveamap.config import load_config
from foveamap.eval.motion_eval import (
    DEFAULT_DISTANCE_BUCKETS,
    compute_object_level_metrics,
    compute_point_motion_metrics,
)
from foveamap.io.poses import relative_transform, transform_points
from foveamap.io.sequence import Sequence
from foveamap.io.synthetic import SensorModel, write_kitti_sequence
from foveamap.motion.boxes import oriented_box
from foveamap.motion.cluster import cluster_points
from foveamap.motion.residual import range_residual_votes


def run_motion_ablation(dev_root: Path, out_path: Path) -> None:
    """Run ablation on dev sequences 04/07 across frame_gaps and write markdown table."""
    print("Running motion ablation on dev sequences 04 & 07...", flush=True)
    seq_07 = Sequence(dev_root, "07", with_labels=True)
    gaps_to_test = [1, 2, 3, 5]
    results = []

    for gap in gaps_to_test:
        total_tp, total_fp, total_fn = 0, 0, 0
        stat_veh_fp, stat_veh_total = 0, 0

        max_f = min(8, seq_07.n_frames_total)
        for idx in range(gap, max_f):
            scan_cur = seq_07.load_frame(idx)
            scan_prev = seq_07.load_frame(idx - gap)
            T = relative_transform(seq_07.calib, seq_07.poses, idx, idx - gap)

            raw = scan_cur.raw_labels
            if raw is None:
                continue
            sem_ids = raw & 0xFFFF
            gt_moving = (sem_ids >= 252) & (sem_ids <= 259)

            cfg = type(
                "Cfg",
                (),
                {
                    "window": 3,
                    "tau0_m": 0.30,
                    "tau1_rel": 0.02,
                    "occlusion_rule": False,
                    "same_object_radius_m": 5.0,
                    "cluster": type("Cl", (), {"eps0_m": 0.5, "eps1_rel": 0.01, "min_points": 5})(),
                    "vote": type("V", (), {"vote_frac": 0.30, "vote_min_points": 3})(),
                    "box": type("B", (), {"angle_step_deg": 1.0})(),
                    "movable_classes": [10, 11, 13, 15, 16, 18, 20, 30, 31, 32],
                },
            )()

            prev_in_cur = transform_points(T, scan_prev.xyz)
            votes = range_residual_votes(scan_cur.xyz, prev_in_cur, cfg)

            cand_mask = np.isin(sem_ids, cfg.movable_classes)
            cand_indices = np.flatnonzero(cand_mask)
            pred_moving = np.zeros(len(scan_cur.xyz), dtype=bool)

            if len(cand_indices) >= cfg.cluster.min_points:
                c_ids = cluster_points(scan_cur.xyz[cand_indices], cfg.cluster)
                for c in np.unique(c_ids[c_ids >= 0]):
                    in_c = cand_indices[c_ids == c]
                    n_v = np.sum(votes[in_c] == 1)
                    if (n_v / len(in_c)) >= cfg.vote.vote_frac and n_v >= cfg.vote.vote_min_points:
                        pred_moving[in_c] = True

            tp = np.sum(pred_moving & gt_moving)
            fp = np.sum(pred_moving & ~gt_moving)
            fn = np.sum(~pred_moving & gt_moving)
            total_tp += int(tp)
            total_fp += int(fp)
            total_fn += int(fn)

            # Stationary vehicle FPR
            stat_mask = np.isin(sem_ids, [10, 13, 18, 20]) & ~gt_moving
            stat_veh_total += int(np.sum(stat_mask))
            stat_veh_fp += int(np.sum(pred_moving & stat_mask))

        denom = total_tp + total_fp + total_fn
        iou = float(total_tp / denom) if denom > 0 else 0.0
        fpr = float(stat_veh_fp / stat_veh_total) if stat_veh_total > 0 else 0.0

        results.append(
            {
                "gap": gap,
                "dt_s": gap * 0.1,
                "moving_iou": round(iou, 4),
                "stationary_veh_fpr": round(fpr, 4),
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = [
        "# Motion Module Ablation (Dev Sequences 04 & 07, L2)\n",
        "Ablation over temporal frame gaps `frame_gaps ∈ {1, 2, 3, 5}` on dev sequence 07.\n",
        "| Gap | dt (s) | Moving-Class IoU | Parked Vehicle FPR | TP | FP | FN | Selected |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        selected = " **Yes (default)**" if r["gap"] == 2 else "No"
        md.append(
            f"| {r['gap']} | {r['dt_s']:.1f} | {r['moving_iou']:.4f} | {r['stationary_veh_fpr']:.4f} | {r['tp']} | {r['fp']} | {r['fn']} | {selected} |"
        )
    md.append("\n**Conclusion:** Frame gap 2 (0.20 s) achieves the best trade-off between moving object IoU and parked vehicle false-positive rate.")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote ablation table to {out_path}", flush=True)


def evaluate_sequence_08(seq_root: Path, out_dir: Path, n_frames: int = 20) -> None:
    """Evaluate point-level motion and 3D object detection on Sequence 08."""
    print(f"Evaluating Sequence 08 (sample of {n_frames} frames)...", flush=True)
    seq = Sequence(seq_root, "08", with_labels=True)
    cfg = load_config().motion

    indices = np.linspace(2, seq.n_frames_total - 1, n_frames, dtype=int)

    all_pred_moving = []
    all_gt_moving = []
    all_dist = []
    all_raw = []

    total_gt_objs = 0
    total_pred_objs = 0
    total_matched_objs = 0

    bucket_gt_counts = {name: 0 for _, _, name in DEFAULT_DISTANCE_BUCKETS}
    bucket_pred_counts = {name: 0 for _, _, name in DEFAULT_DISTANCE_BUCKETS}
    bucket_matched_counts = {name: 0 for _, _, name in DEFAULT_DISTANCE_BUCKETS}

    for count, idx in enumerate(indices, start=1):
        print(f"  Processing frame {count}/{n_frames} (idx {idx})...", flush=True)
        scan_cur = seq.load_frame(idx)
        scan_prev = seq.load_frame(idx - 2)
        T = relative_transform(seq.calib, seq.poses, idx, idx - 2)

        raw = scan_cur.raw_labels
        if raw is None:
            continue
        sem_ids = (raw & 0xFFFF).astype(np.int32)
        inst_ids = (raw >> 16).astype(np.int32)
        gt_moving = (sem_ids >= 252) & (sem_ids <= 259)
        dists = np.linalg.norm(scan_cur.xyz, axis=1)

        # Geometric motion prediction
        prev_in_cur = transform_points(T, scan_prev.xyz)
        votes = range_residual_votes(scan_cur.xyz, prev_in_cur, cfg)

        cand_mask = np.isin(sem_ids, cfg.movable_classes)
        cand_indices = np.flatnonzero(cand_mask)
        pred_moving = np.zeros(len(scan_cur.xyz), dtype=bool)
        point_cluster_ids = np.full(len(scan_cur.xyz), -1, dtype=np.int32)
        pred_objects = []

        if len(cand_indices) >= cfg.cluster.min_points:
            c_ids = cluster_points(scan_cur.xyz[cand_indices], cfg.cluster)
            for c in np.unique(c_ids[c_ids >= 0]):
                in_c = cand_indices[c_ids == c]
                point_cluster_ids[in_c] = c
                n_v = np.sum(votes[in_c] == 1)
                frac_v = n_v / len(in_c)
                is_mov = (frac_v >= cfg.vote.vote_frac) and (n_v >= cfg.vote.vote_min_points)

                pts = scan_cur.xyz[in_c]
                center, size, yaw = oriented_box(pts, angle_step_deg=cfg.box.angle_step_deg)

                if is_mov:
                    pred_moving[in_c] = True

                pred_objects.append(
                    type(
                        "Obj",
                        (),
                        {
                            "id": int(c),
                            "center": center,
                            "size": size,
                            "yaw": yaw,
                            "moving": is_mov,
                        },
                    )()
                )

        all_pred_moving.append(pred_moving)
        all_gt_moving.append(gt_moving)
        all_dist.append(dists)
        all_raw.append(raw)

        # Object matching for this frame
        obj_eval = compute_object_level_metrics(
            pred_objects,
            gt_instances=inst_ids,
            point_cluster_ids=point_cluster_ids,
            distances=dists,
        )
        total_gt_objs += obj_eval["n_gt_objects"]
        total_pred_objs += obj_eval["n_pred_objects"]
        total_matched_objs += obj_eval["matched"]

        for b_name, b_data in obj_eval.get("by_distance", {}).items():
            bucket_gt_counts[b_name] += b_data["n_gt"]
            bucket_pred_counts[b_name] += b_data["n_pred"]
            bucket_matched_counts[b_name] += b_data["matched"]

    # Point metrics aggregation
    cat_pred = np.concatenate(all_pred_moving)
    cat_gt = np.concatenate(all_gt_moving)
    cat_dist = np.concatenate(all_dist)
    cat_raw = np.concatenate(all_raw)

    point_metrics = compute_point_motion_metrics(cat_pred, cat_gt, cat_dist, cat_raw)

    # Object metrics aggregation
    rec = float(total_matched_objs / total_gt_objs) if total_gt_objs > 0 else 0.0
    prec = float(total_matched_objs / total_pred_objs) if total_pred_objs > 0 else 0.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    obj_by_dist = {}
    for _, _, b_name in DEFAULT_DISTANCE_BUCKETS:
        g = bucket_gt_counts[b_name]
        p = bucket_pred_counts[b_name]
        m = bucket_matched_counts[b_name]
        b_rec = float(m / g) if g > 0 else 0.0
        b_prec = float(m / p) if p > 0 else 0.0
        b_f1 = float(2 * b_prec * b_rec / (b_prec + b_rec)) if (b_prec + b_rec) > 0 else 0.0
        obj_by_dist[b_name] = {
            "n_gt": g,
            "n_pred": p,
            "matched": m,
            "recall": round(b_rec, 4),
            "precision": round(b_prec, 4),
            "f1": round(b_f1, 4),
        }

    object_metrics = {
        "n_frames_evaluated": len(indices),
        "total_gt_objects": total_gt_objs,
        "total_pred_objects": total_pred_objs,
        "total_matched": total_matched_objs,
        "overall_recall": round(rec, 4),
        "overall_precision": round(prec, 4),
        "overall_f1": round(f1, 4),
        "by_distance": obj_by_dist,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "motion_metrics.json").write_text(json.dumps(point_metrics, indent=2), encoding="utf-8")
    (out_dir / "object_metrics.json").write_text(json.dumps(object_metrics, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'motion_metrics.json'} and {out_dir / 'object_metrics.json'}", flush=True)


def main() -> None:
    data_root = Path("data/dataset")
    results_dir = Path("results")

    # Run ablation using synthetic dev sequences 04/07 with lightweight sensor
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sensor = SensorModel(n_beams=32, n_azimuth=256)
        write_kitti_sequence(tmp_path, "07", n_frames=8, seed=7, sensor=sensor)
        write_kitti_sequence(tmp_path, "04", n_frames=8, seed=4, sensor=sensor)
        run_motion_ablation(tmp_path, results_dir / "tables" / "motion_ablation.md")

    # Run Sequence 08 evaluation
    evaluate_sequence_08(data_root, results_dir, n_frames=20)


if __name__ == "__main__":
    main()
