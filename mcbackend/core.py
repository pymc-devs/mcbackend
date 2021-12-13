"""
Module with metadata structures and abstract classes.
"""
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy


@dataclass
class RunMeta:
    """Metadata of a multi-chain MCMC run."""

    run_id: str
    var_names: Sequence[str]
    var_dtypes: Sequence[str]
    var_shapes: Sequence[Tuple[int, ...]]
    var_is_free: Sequence[bool]


@dataclass
class ChainMeta:
    """Metadata of one MCMC chain."""

    run_id: str
    chain_number: int

    @property
    def chain_id(self) -> str:
        """Unique identifier of this MCMC chain."""
        return f"{self.run_id}_chain_{self.chain_number}"


class BackendBase:
    """Base class for all MCMC draw storage backends."""

    def init_backend(self):
        """
        Initialization of the sampling backend.
        For example the creation of database tables.
        """
        raise NotImplementedError()

    def init_run(self, run_meta: RunMeta):
        """Register a new MCMC run with the backend."""
        raise NotImplementedError()

    def init_chain(self, chain_meta: ChainMeta):
        """Register a new MCMC chain with the backend."""
        raise NotImplementedError()

    def add_draw(self, chain_id: str, draw_idx: int, draw: Dict[str, numpy.ndarray]):
        """Add a draw to an MCMC chain."""
        raise NotImplementedError()

    def get_draw(
        self, chain_id: str, draw_idx: int, var_names: Sequence[str] = None
    ) -> Dict[str, numpy.ndarray]:
        """Retrieve one draw from an MCMC chain."""
        raise NotImplementedError()

    def get_variable(self, chain_id: str, var_name: str) -> Dict[str, numpy.ndarray]:
        """Retrieve all draws of a variable from an MCMC chain."""
        raise NotImplementedError()
