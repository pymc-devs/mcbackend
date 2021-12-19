"""
A framework agnostic implementation for storage of MCMC draws.
"""
from .core import Backend, Chain, ChainMeta, Run, RunMeta

# Backends
try:
    from backends import clickhouse
except ModuleNotFoundError:
    pass

# Adapters
try:
    from adapters import pymc
except ModuleNotFoundError:
    pass


__version__ = "0.1.0"
