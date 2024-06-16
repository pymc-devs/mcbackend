"""
This backend simply discards draws. There are not stored in memory.
This can be used in situations where we want to run an MCMC but not permanently
store its output.
"""

# Code-wise, a NullChain is essentially just a NumpyChain without the underlying data array.

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy

from ..core import Backend, Chain, Run
from ..meta import ChainMeta, RunMeta
from .numpy import grow_append, prepare_storage


class NullChain(Chain):
    """A null storage: discards values immediately and allocates no memory.

    Use cases are

    - Online computations: Draws are used and discarded immediately, allowing for much larger sample spaces.
    - Profiling: To use as a baseline, to measure compute time & memory before allocating memory for draws.
      Comparing with another backend would then show how much overhead it adds.

    Since draws are not stored, only a subset of the `Chain` interface is supported:

    - Supported: `__len__`, `append`, `get_stats`, `get_stats_at`
    - Not supported: `get_draws`, `get_draws_at`

    .. Todo:: Option to also sampling stats?
    .. Todo:: Allow retrieving the most recent draw?

    """

    def __init__(self, cmeta: ChainMeta, rmeta: RunMeta, *, preallocate: int) -> None:
        """Creates a null storage for draws from a chain: will gobble outputs without storing them

        Parameters
        ----------
        cmeta : ChainMeta
            Metadata of the chain.
        rmeta : RunMeta
            Metadata of the MCMC run.
        preallocate : int
            Influences the memory pre-allocation behavior.
            (Draws are not saved, but stats may still be.)
            The default is to reserve memory for ``preallocate`` draws
            and grow the allocated memory by 10 % when needed.
            Exceptions are variables with non-rigid shapes (indicated by 0 in the shape tuple)
            where the correct amount of memory cannot be pre-allocated.
            In these cases object arrays are used.
        """
        self._draw_idx = 0

        # Create storage ndarrays only for sampler stats.
        self._stats, self._stat_is_rigid = prepare_storage(rmeta.sample_stats, preallocate)

        super().__init__(cmeta, rmeta)

    def append(  # pylint: disable=duplicate-code
        self, draw: Mapping[str, numpy.ndarray], stats: Optional[Mapping[str, numpy.ndarray]] = None
    ):
        if stats:
            grow_append(self._stats, stats, self._stat_is_rigid, self._draw_idx)
        self._draw_idx += 1
        return

    def __len__(self) -> int:
        return self._draw_idx

    def get_draws(self, var_name: str, slc: slice = slice(None)) -> numpy.ndarray:
        raise RuntimeError("NullChain does not save draws.")

    def get_draws_at(self, idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        raise RuntimeError("NullChain does not save draws.")

    def get_stats(  # pylint: disable=duplicate-code
        self, stat_name: str, slc: slice = slice(None)
    ) -> numpy.ndarray:
        data = self._stats[stat_name][: self._draw_idx][slc]
        if self.sample_stats[stat_name].dtype == "str":
            return numpy.array(data.tolist(), dtype=str)
        return data

    def get_stats_at(self, idx: int, stat_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        return {sn: numpy.asarray(self._stats[sn][idx]) for sn in stat_names}


class NullRun(Run):
    """An MCMC run where samples are immediately discarded."""

    def __init__(self, meta: RunMeta, *, preallocate: int) -> None:
        self._settings = {"preallocate": preallocate}
        self._chains: List[NullChain] = []
        super().__init__(meta)

    def init_chain(self, chain_number: int) -> NullChain:
        cmeta = ChainMeta(self.meta.rid, chain_number)
        chain = NullChain(cmeta, self.meta, **self._settings)
        self._chains.append(chain)
        return chain

    def get_chains(self) -> Tuple[NullChain, ...]:
        return tuple(self._chains)


class NullBackend(Backend):
    """A backend which discards samples immediately."""

    def __init__(self, preallocate: int = 1_000) -> None:
        self._settings = {"preallocate": preallocate}
        super().__init__()

    def init_run(self, meta: RunMeta) -> NullRun:
        return NullRun(meta, **self._settings)
