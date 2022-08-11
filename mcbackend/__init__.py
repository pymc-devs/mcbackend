"""
A framework agnostic implementation for storage of MCMC draws.
"""
from .backends.numpy import NumPyBackend
from .core import Backend, Chain, Run
from .meta import ChainMeta, Coordinate, DataVariable, ExtendedValue, RunMeta, Variable

# Backends
try:
    from .backends import clickhouse
    from .backends.clickhouse import ClickHouseBackend
except ModuleNotFoundError:
    pass

# Adapters
try:
    from .adapters import pymc
    from .adapters.pymc import TraceBackend
except ModuleNotFoundError:
    pass


__version__ = "0.1.3"
