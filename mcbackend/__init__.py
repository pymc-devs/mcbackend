"""
A framework agnostic implementation for storage of MCMC draws.
"""

from .backends.null import NullBackend
from .backends.numpy import NumPyBackend
from .core import Backend, Chain, Run
from .meta import ChainMeta, Coordinate, DataVariable, ExtendedValue, RunMeta, Variable

# Backends
try:
    from .backends import clickhouse
    from .backends.clickhouse import ClickHouseBackend
except ModuleNotFoundError:
    pass

__version__ = "0.5.2"
__all__ = [
    "NumPyBackend",
    "NullBackend",
    "Backend",
    "Chain",
    "Run",
    "ChainMeta",
    "Coordinate",
    "DataVariable",
    "ExtendedValue",
    "RunMeta",
    "Variable",
    "clickhouse",
    "ClickHouseBackend",
]
