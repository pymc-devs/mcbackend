"""
This backend holds draws in memory, managing them via NumPy arrays.
"""
import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy

from ..core import Backend, Chain, Run, is_rigid
from ..meta import ChainMeta, RunMeta


def grow_append(
    storage_dict: Dict[str, numpy.ndarray],
    values: Mapping[str, numpy.ndarray],
    rigid: Mapping[str, bool],
    draw_idx: int,
):
    """Writes values into storage arrays, growing them if needed."""
    for vn, v in values.items():
        target = storage_dict[vn]
        length = len(target)
        if length == draw_idx:
            # Grow the array by 10 %
            ngrow = math.ceil(0.1 * length)
            if rigid[vn]:
                extension = numpy.empty((ngrow,) + numpy.shape(v))
            else:
                extension = numpy.array([None] * ngrow)
            storage_dict[vn] = numpy.concatenate((target, extension), axis=0)
            target = storage_dict[vn]
        target[draw_idx] = v
    return


class NumPyChain(Chain):
    """Stores value draws in NumPy arrays and can pre-allocate memory."""

    def __init__(self, cmeta: ChainMeta, rmeta: RunMeta, *, preallocate: int) -> None:
        """Creates an in-memory storage for draws from a chain.

        Parameters
        ----------
        cmeta : ChainMeta
            Metadata of the chain.
        rmeta : RunMeta
            Metadata of the MCMC run.
        preallocate : int
            Influences the memory pre-allocation behavior.
            The default is to reserve memory for ``preallocate`` draws
            and grow the allocated memory by 10 % when needed.
            Exceptions are variables with non-rigid shapes (indicated by 0 in the shape tuple)
            where the correct amount of memory cannot be pre-allocated.
            In these cases, and when ``preallocate == 0`` object arrays are used.
        """
        self._var_is_rigid: Dict[str, bool] = {}
        self._samples: Dict[str, numpy.ndarray] = {}
        self._stat_is_rigid: Dict[str, bool] = {}
        self._stats: Dict[str, numpy.ndarray] = {}
        self._draw_idx = 0

        # Create storage ndarrays for each model variable and sampler stat.
        for target_dict, rigid_dict, variables in [
            (self._samples, self._var_is_rigid, rmeta.variables),
            (self._stats, self._stat_is_rigid, rmeta.sample_stats),
        ]:
            for var in variables:
                rigid = is_rigid(var.shape) and not var.undefined_ndim and var.dtype != "str"
                rigid_dict[var.name] = rigid
                if preallocate > 0 and rigid:
                    reserve = (preallocate, *var.shape)
                    target_dict[var.name] = numpy.empty(reserve, var.dtype)
                else:
                    target_dict[var.name] = numpy.array([None] * preallocate)

        super().__init__(cmeta, rmeta)

    def append(
        self, draw: Mapping[str, numpy.ndarray], stats: Optional[Mapping[str, numpy.ndarray]] = None
    ):
        grow_append(self._samples, draw, self._var_is_rigid, self._draw_idx)
        if stats:
            grow_append(self._stats, stats, self._stat_is_rigid, self._draw_idx)
        self._draw_idx += 1
        return

    def __len__(self) -> int:
        return self._draw_idx

    def get_draws(self, var_name: str, slc: slice = slice(None)) -> numpy.ndarray:
        data = self._samples[var_name][: self._draw_idx][slc]
        if self.variables[var_name].dtype == "str":
            return numpy.array(data.tolist(), dtype=str)
        return data

    def get_draws_at(self, idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        return {vn: numpy.asarray(self._samples[vn][idx]) for vn in var_names}

    def get_stats(self, stat_name: str, slc: slice = slice(None)) -> numpy.ndarray:
        data = self._stats[stat_name][: self._draw_idx][slc]
        if self.sample_stats[stat_name].dtype == "str":
            return numpy.array(data.tolist(), dtype=str)
        return data

    def get_stats_at(self, idx: int, stat_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        return {sn: numpy.asarray(self._stats[sn][idx]) for sn in stat_names}


class NumPyRun(Run):
    """An MCMC run where samples are kept in memory."""

    def __init__(self, meta: RunMeta, *, preallocate: int) -> None:
        self._settings = dict(preallocate=preallocate)
        self._chains: List[NumPyChain] = []
        super().__init__(meta)

    def init_chain(self, chain_number: int) -> NumPyChain:
        cmeta = ChainMeta(self.meta.rid, chain_number)
        chain = NumPyChain(cmeta, self.meta, **self._settings)
        self._chains.append(chain)
        return chain

    def get_chains(self) -> Tuple[NumPyChain, ...]:
        return tuple(self._chains)


class NumPyBackend(Backend):
    """An in-memory backend using NumPy."""

    def __init__(self, preallocate: int = 1_000) -> None:
        self._settings = dict(
            preallocate=preallocate,
        )
        super().__init__()

    def init_run(self, meta: RunMeta) -> NumPyRun:
        return NumPyRun(meta, **self._settings)
