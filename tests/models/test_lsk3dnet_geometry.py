"""CPU-only checks for LSK3DNet native and extended sparse geometry."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from foveamap.models.lsk3dnet import resolve_inference_geometry


def _upstream_config() -> dict[str, object]:
    return {
        "model_params": {"spatial_shape": [2000, 2000, 120]},
        "dataset_params": {
            "min_volume_space": [-50.0, -50.0, -4.0],
            "max_volume_space": [50.0, 50.0, 2.0],
            "spatial_shape": [2000, 2000, 120],
        },
    }


def _settings(profile: str = "extended_100m") -> SimpleNamespace:
    return SimpleNamespace(
        coverage_profile=profile,
        extended_min_volume_space=(-100.0, -100.0, -4.0),
        extended_max_volume_space=(100.0, 100.0, 2.0),
        extended_spatial_shape=(4000, 4000, 120),
    )


def test_native_profile_is_unchanged_and_does_not_mutate_input() -> None:
    source = _upstream_config()
    resolved = resolve_inference_geometry(source, _settings("native"))
    assert resolved == source
    assert resolved is not source


def test_extended_profile_reaches_100m_and_preserves_voxel_size() -> None:
    source = _upstream_config()
    resolved = resolve_inference_geometry(source, _settings())
    assert resolved["dataset_params"]["min_volume_space"] == [-100.0, -100.0, -4.0]
    assert resolved["dataset_params"]["max_volume_space"] == [100.0, 100.0, 2.0]
    assert resolved["model_params"]["spatial_shape"] == [4000, 4000, 120]
    assert resolved["dataset_params"]["spatial_shape"] == [4000, 4000, 120]

    native_voxel = (np.array([50.0, 50.0, 2.0]) - np.array([-50.0, -50.0, -4.0])) / np.array(
        [2000, 2000, 120]
    )
    extended_voxel = (np.array([100.0, 100.0, 2.0]) - np.array([-100.0, -100.0, -4.0])) / np.array(
        [4000, 4000, 120]
    )
    np.testing.assert_allclose(extended_voxel, native_voxel, rtol=0.0, atol=1e-12)


def test_extended_profile_rejects_changed_voxel_size() -> None:
    cfg = _settings()
    cfg.extended_spatial_shape = (2000, 2000, 120)
    with pytest.raises(ValueError, match="preserve the checkpoint voxel size"):
        resolve_inference_geometry(_upstream_config(), cfg)

