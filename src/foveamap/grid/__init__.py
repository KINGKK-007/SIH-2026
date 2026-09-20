"""foveamap.grid — Variable-resolution grid engine.

Public surface
--------------
spec     : GridSpec and Ring dataclasses (frozen, YAML-loadable).
clipmap  : ClipmapGrid — build(), layer(), stats().
layers   : Cell constants, dtypes, flag bits, aggregation rules.
aggregate: Numba scatter-reduce kernel.
"""

from foveamap.grid.clipmap import ClipmapGrid
from foveamap.grid.spec import GridSpec, Ring

__all__ = ["Ring", "GridSpec", "ClipmapGrid"]
