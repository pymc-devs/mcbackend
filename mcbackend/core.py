"""
Module with metadata structures and abstract classes.
"""
from typing import Dict, Optional, Sequence, Sized

import numpy

from .meta import ChainMeta, RunMeta, Variable

Shape = Sequence[int]


def is_rigid(shape: Optional[Shape]):
    if shape is None:
        return False
    return all(s != 0 for s in shape)


def chain_id(meta: ChainMeta):
    return f"{meta.rid}_chain_{meta.chain_number}"


class Chain(Sized):
    """A handle on one Markov-chain."""

    def __init__(self, cmeta: ChainMeta, rmeta: RunMeta) -> None:
        self.cmeta = cmeta
        self.rmeta = rmeta
        self._cid = chain_id(cmeta)
        super().__init__()

    def append(
        self, draw: Dict[str, numpy.ndarray], stats: Optional[Dict[str, numpy.ndarray]] = None
    ):
        """Appends an iteration to the chain.

        Parameters
        ----------
        draw : dict of ndarray
            Values for all model variables.
        stats : dict of ndarray, optional
            Values of sampler stats in this iteration.
        """
        raise NotImplementedError()

    def get_draws(self, var_name: str) -> numpy.ndarray:
        """Retrieve all draws of a variable from an MCMC chain."""
        raise NotImplementedError()

    def get_stats(self, stat_name: str) -> numpy.ndarray:
        """Retrieve all values of a sampler statistic."""
        raise NotImplementedError()

    def get_draws_at(self, idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        """Retrieve one draw from an MCMC chain."""
        raise NotImplementedError()

    def get_stats_at(self, idx: int, stat_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        """Retrieve the sampler stats corresponding to one draw in the chain."""
        raise NotImplementedError()

    def __len__(self) -> int:
        """Determine the length of the chain.

        ⚠ The base implementation does this by fetching all values of a variable or sampler stat.
        ⚠ For higher performance, backends should consider to overwrite the base implementation.
        """
        for method, items in [
            (self.get_draws, self.rmeta.variables),
            (self.get_stats, self.rmeta.sample_stats),
        ]:
            for var in items:
                return len(method(var.name))
        raise Exception("This chain has no variables or sample stats.")

    @property
    def cid(self) -> str:
        """An identifier, unique for the combination of Run ID and chain number."""
        return self._cid

    @property
    def variables(self) -> Dict[str, Variable]:
        """Convenience dictionary to access the ``RunMeta.variables``."""
        return {var.name: var for var in self.rmeta.variables}

    @property
    def sample_stats(self) -> Dict[str, Variable]:
        """Convenience dictionary to access the ``RunMeta.sample_stats``."""
        return {var.name: var for var in self.rmeta.sample_stats}


class Run:
    """A handle on one MCMC run."""

    def __init__(self, meta: RunMeta) -> None:
        self.meta = meta
        super().__init__()

    def init_chain(self, chain_number: int) -> Chain:
        raise NotImplementedError()


class Backend:
    """Base class for all MCMC draw storage backends."""

    def init_run(self, meta: RunMeta) -> Run:
        """Register a new MCMC run with the backend."""
        raise NotImplementedError()
