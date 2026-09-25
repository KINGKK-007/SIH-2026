"""Tests for server ring texture rendering and message protocol serialization (T13.1)."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from foveamap.config import load_config
from foveamap.grid.presets import load_preset
from foveamap.io.sequence import Sequence
from foveamap.models.oracle import OracleModel
from foveamap.pipeline.runner import PipelineRunner
from foveamap.server.protocol import SCHEMA_VERSION, serialise_frame_result
from foveamap.server.render import ring_textures


def test_render_textures(synthetic_root: Path) -> None:
    cfgs = load_config()
    preset = load_preset("fovea_default", cfgs.grid)
    runner = PipelineRunner(mode="oracle", model=OracleModel(), preset=preset, cfgs=cfgs)
    seq = Sequence(synthetic_root, "08")
    result = runner.process(seq, 0)

    for layer_name in ["class", "height", "traversability", "moving", "confidence"]:
        textures = ring_textures(result.layers, layer_name)
        assert len(textures) == len(preset.rings)
        for k, tex in enumerate(textures):
            n_side = preset.rings[k].side
            assert tex.shape == (n_side, n_side, 4)
            assert tex.dtype == np.uint8


def test_protocol_serialise(synthetic_root: Path) -> None:
    cfgs = load_config()
    preset = load_preset("fovea_default", cfgs.grid)
    runner = PipelineRunner(mode="oracle", model=OracleModel(), preset=preset, cfgs=cfgs)
    seq = Sequence(synthetic_root, "08")
    result = runner.process(seq, 0)

    payload = serialise_frame_result(
        result=result,
        seq="08",
        timestamp=0.0,
        model="oracle",
        preset_name="fovea_default",
    )

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["seq"] == "08"
    assert payload["frame_idx"] == 0
    assert payload["timestamp"] == 0.0
    assert payload["model"] == "oracle"
    assert payload["preset"] == "fovea_default"

    # Counters
    for k in ["n_raw", "n_invalid", "n_in_grid", "n_out_of_grid", "n_z_saturated"]:
        assert k in payload["counters"]
        assert isinstance(payload["counters"][k], int)

    # Timings
    for k in ["io_ms", "model_ms", "label_ms", "grid_ms", "finalize_ms", "memory_ms"]:
        assert k in payload["timings_ms"]
        assert isinstance(payload["timings_ms"][k], float)

    # Memory
    for k in ["basis", "dense3d_bytes", "sparse3d_bytes", "uniform25d_bytes", "fovea_bytes", "rss_delta_bytes"]:
        assert k in payload["memory"]

    # Rings
    assert len(payload["rings"]) == len(preset.rings)
    for ring in payload["rings"]:
        for field in [
            "ring_idx", "cell_mm", "r_max_mm", "side",
            "iy", "ix", "ground_z", "top_z", "clearance",
            "cls", "moving_frac", "count", "conf", "flags",
        ]:
            assert field in ring
        # All sparse cell arrays have matching length
        n_occ = len(ring["iy"])
        assert len(ring["ix"]) == n_occ
        assert len(ring["cls"]) == n_occ
        assert len(ring["count"]) == n_occ
