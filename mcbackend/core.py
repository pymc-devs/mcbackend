"""
Module with metadata structures and abstract classes.
"""
from typing import Dict, Optional, Sequence

import numpy

from .meta import ChainMeta, RunMeta, Variable

Shape = Sequence[int]


def is_rigid(shape: Optional[Shape]):
    if shape is None:
        return False
    return all(s != 0 for s in shape)


def chain_id(meta: ChainMeta):
    return f"{meta.rid}_chain_{meta.chain_number}"


class Chain:
    """A handle on one Markov-chain."""

    def __init__(self, cmeta: ChainMeta, rmeta: RunMeta) -> None:
        self.cmeta = cmeta
        self.rmeta = rmeta
        self._cid = chain_id(cmeta)
        super().__init__()

    def add_draw(self, draw: Dict[str, numpy.ndarray]):
        raise NotImplementedError()

    def get_variable(self, var_name: str) -> numpy.ndarray:
        """Retrieve all draws of a variable from an MCMC chain."""
        raise NotImplementedError()

    def get_draw(self, draw_idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        """Retrieve one draw from an MCMC chain."""
        raise NotImplementedError()

    @property
    def cid(self) -> str:
        return self._cid

    @property
    def variables(self) -> Dict[str, Variable]:
        return {var.name: var for var in self.rmeta.variables}


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
