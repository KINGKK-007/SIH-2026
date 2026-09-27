"""Unit tests for motion pipeline and PipelineRunner integration (T10.3)."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from foveamap.config import load_config
from foveamap.grid.presets import load_preset
from foveamap.io.labels import DYNAMIC, STATIC_OBSTACLE, raw_to_super
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel
from foveamap.motion.pipeline import estimate_motion
from foveamap.pipeline.records import ClassifiedScan, Prediction, Scan
from foveamap.pipeline.runner import PipelineRunner


def test_estimate_motion_oracle_mode() -> None:
    """Oracle mode extracts moving points from IDs 252-259 and forms objects from GT instances."""
    n_pts = 100
    xyz = np.random.uniform(-10, 10, size=(n_pts, 3)).astype(np.float32)

    # Moving and parked cars have instances; a building instance should stay excluded.
    sem_ids = np.full(n_pts, 40, dtype=np.uint32)  # road
    inst_ids = np.zeros(n_pts, dtype=np.uint32)

    sem_ids[:30] = 252  # moving-car
    inst_ids[:30] = 5
    sem_ids[30:60] = 10  # static car
    inst_ids[30:60] = 6
    sem_ids[60:80] = 50  # building
    inst_ids[60:80] = 7

    raw_labels = (inst_ids << 16) | sem_ids

    scan = Scan(
        seq="08",
        idx=0,
        xyz=xyz,
        remission=np.zeros(n_pts, dtype=np.float32),
        raw_labels=raw_labels,
        pose=np.eye(4),
        timestamp=0.0,
    )
    super_cls, moving = raw_to_super(sem_ids)
    cur = ClassifiedScan(
        scan=scan,
        super_cls=super_cls,
        moving=moving,
        conf=np.full(n_pts, 255, dtype=np.uint8),
        objects=[],
    )

    cfg = SimpleNamespace(
        box=SimpleNamespace(angle_step_deg=1.0),
        cluster=SimpleNamespace(min_points=5),
        enabled=True,
    )

    out = estimate_motion(cur, prev_scans=[], transforms=[], cfg=cfg, use_oracle=True)

    # Moving points and dynamic super class
    assert np.all(out.moving[:30] == True)  # noqa: E712
    assert np.all(out.super_cls[:30] == DYNAMIC)
    assert np.all(out.moving[30:] == False)  # noqa: E712

    # Object was created
    assert len(out.objects) == 2
    obj = next(obj for obj in out.objects if obj.id == 5)
    assert obj.id == 5
    assert obj.cls_name == "moving-car"
    assert obj.moving is True
    assert obj.n_points == 30
    parked = next(obj for obj in out.objects if obj.id == 6)
    assert parked.cls_name == "car"
    assert parked.moving is False
    assert parked.n_points == 30
    assert all(obj.id != 7 for obj in out.objects)


def test_pipeline_runner_with_motion(synthetic_root: Path) -> None:
    """PipelineRunner should record motion_ms and populate FrameResult.objects."""
    seq = Sequence(synthetic_root, "08", with_labels=True)
    fovea_cfg = load_config()
    preset = load_preset("tiny_test", fovea_cfg.grid)
    model = OracleModel()

    runner = PipelineRunner(mode="oracle", model=model, preset=preset, cfgs=fovea_cfg)
    result = runner.process(seq, idx=0)

    # FrameResult contains objects and motion timing
    assert "motion_ms" in result.timings_ms
    assert result.timings_ms["motion_ms"] >= 0.0
    assert isinstance(result.objects, list)
    # Check that any dynamic objects have valid bounding boxes
    for obj in result.objects:
        assert len(obj.center) == 3
        assert len(obj.size) == 3
        assert obj.size[0] >= obj.size[1]  # l >= w
        assert isinstance(obj.cls_name, str)


def test_oracle_disconnected_vehicle_clusters_get_separate_fitted_boxes() -> None:
    rng = np.random.default_rng(4)
    first = rng.normal((10.0, -5.0, 0.0), (0.6, 0.3, 0.2), (40, 3))
    second = rng.normal((10.0, 6.0, 0.0), (0.6, 0.3, 0.2), (40, 3))
    xyz = np.vstack((first, second)).astype(np.float32)
    labels = np.full(len(xyz), (7 << 16) | 252, dtype=np.uint32)
    scan = Scan("08", 0, xyz, np.zeros(len(xyz), dtype=np.float32), labels,
                np.eye(4), 0.0)
    super_cls, moving = raw_to_super(labels & 0xFFFF)
    cur = ClassifiedScan(scan, super_cls, moving, np.full(len(xyz), 255, dtype=np.uint8))
    cfg = SimpleNamespace(box=SimpleNamespace(angle_step_deg=1.0),
                          cluster=SimpleNamespace(min_points=5), enabled=True)

    out = estimate_motion(cur, [], [], cfg, use_oracle=True)

    assert len(out.objects) == 2
    assert {obj.instance_id for obj in out.objects} == {7}
    assert all(obj.size[0] < 4 and obj.size[1] < 4 for obj in out.objects)
    assert all(obj.moving for obj in out.objects)
