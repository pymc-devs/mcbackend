"""
This backend holds draws in memory, managing them via NumPy arrays.
"""
import math
from typing import Dict, Optional, Sequence

import numpy

from ..core import Backend, Chain, Run, is_rigid
from ..meta import ChainMeta, RunMeta


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
        self._is_rigid = {}
        self._samples = {}
        self._draw_idx = 0
        # Create storage ndarrays for each variable.
        for var in rmeta.variables:
            rigid = is_rigid(var.shape)
            self._is_rigid[var.name] = rigid
            if preallocate > 0 and rigid:
                reserve = (preallocate, *var.shape)
                self._samples[var.name] = numpy.empty(reserve, var.dtype)
            else:
                self._samples[var.name] = numpy.repeat(None, preallocate)
        super().__init__(cmeta, rmeta)

    def append(
        self, draw: Dict[str, numpy.ndarray], stats: Optional[Dict[str, numpy.ndarray]] = None
    ):
        for vn, v in draw.items():
            target = self._samples[vn]
            length = len(target)
            if length == self._draw_idx:
                # Grow the array by 10 %
                ngrow = math.ceil(0.1 * length)
                if self._is_rigid[vn]:
                    extension = numpy.empty((ngrow,) + numpy.shape(v))
                else:
                    extension = numpy.repeat(None, ngrow)
                self._samples[vn] = numpy.concatenate((target, extension), axis=0)
                target = self._samples[vn]
            target[self._draw_idx] = v
        self._draw_idx += 1
        return

    def get_draws(self, var_name: str) -> numpy.ndarray:
        return self._samples[var_name][: self._draw_idx]

    def get_draws_at(self, idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        return {vn: numpy.asarray(self._samples[vn][idx]) for vn in var_names}

    def get_stats(self, stat_name: str) -> numpy.ndarray:
        raise NotImplementedError()

    def get_stats_at(self, idx: int, stat_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        raise NotImplementedError()


class NumPyRun(Run):
    """An MCMC run where samples are kept in memory."""

    def __init__(self, meta: RunMeta, *, preallocate: int) -> None:
        self._settings = dict(preallocate=preallocate)
        super().__init__(meta)

    def init_chain(self, chain_number: int) -> NumPyChain:
        cmeta = ChainMeta(self.meta.rid, chain_number)
        return NumPyChain(cmeta, self.meta, **self._settings)


class NumPyBackend(Backend):
    """An in-memory backend using NumPy."""

    def __init__(self, preallocate: int = 1_000) -> None:
        self._settings = dict(
            preallocate=preallocate,
        )
        super().__init__()

    def init_run(self, meta: RunMeta) -> NumPyRun:
        return NumPyRun(meta, **self._settings)
