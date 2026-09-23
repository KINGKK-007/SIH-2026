"""Smoke test: every module of the README Section 7 layout imports, and the CLI starts."""

from __future__ import annotations

import importlib
import pkgutil

import pytest

import foveamap
from foveamap import cli


def _all_modules() -> list[str]:
    return sorted(m.name for m in pkgutil.walk_packages(foveamap.__path__, prefix="foveamap."))


def test_layout_is_complete() -> None:
    expected = {
        "foveamap.cli",
        "foveamap.config",
        "foveamap.io.kitti",
        "foveamap.io.poses",
        "foveamap.io.labels",
        "foveamap.io.sequence",
        "foveamap.models.base",
        "foveamap.models.oracle",
        "foveamap.models.geom",
        "foveamap.models.rangeview",
        "foveamap.models.cache",
        "foveamap.motion.residual",
        "foveamap.motion.cluster",
        "foveamap.motion.boxes",
        "foveamap.motion.tracker",
        "foveamap.motion.pipeline",
        "foveamap.grid.presets",
        "foveamap.grid.engine",
        "foveamap.grid.accumulators",
        "foveamap.grid.layers",
        "foveamap.grid.baselines",
        "foveamap.grid.memory",
        "foveamap.grid.backends.numpy_backend",
        "foveamap.grid.backends.cpp_backend",
        "foveamap.derived.halo",
        "foveamap.derived.slope",
        "foveamap.derived.step",
        "foveamap.derived.clearance",
        "foveamap.derived.traversability",
        "foveamap.derived.hazards",
        "foveamap.pipeline.runner",
        "foveamap.pipeline.records",
        "foveamap.eval.buckets",
        "foveamap.eval.metrics",
        "foveamap.eval.accuracy",
        "foveamap.eval.coarsening",
        "foveamap.eval.motion_eval",
        "foveamap.eval.hazard_eval",
        "foveamap.eval.density",
        "foveamap.eval.tradeoff",
        "foveamap.eval.latency",
        "foveamap.eval.memory_eval",
        "foveamap.eval.report",
        "foveamap.server.app",
        "foveamap.server.sockets",
        "foveamap.server.protocol",
        "foveamap.server.render",
        "foveamap.viz.figures",
        "foveamap.viz.video",
    }
    missing = expected - set(_all_modules())
    assert not missing, f"missing modules: {sorted(missing)}"


@pytest.mark.parametrize("name", _all_modules())
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_cli_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    assert "foveamap" in capsys.readouterr().out


def test_cli_unimplemented_command_reports_phase(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["inspect", "--sequence", "08", "--idx", "0"]) == 2
    assert "Phase 2" in capsys.readouterr().err
