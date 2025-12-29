"""
A framework agnostic implementation for storage of MCMC draws.
"""

import importlib.metadata

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

__version__ = importlib.metadata.version(__package__ or __name__)
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
