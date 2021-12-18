"""
Module with metadata structures and abstract classes.
"""
import datetime
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy


class RunMeta:
    """Metadata of a multi-chain MCMC run."""

    def __init__(
        self,
        run_id: str,
        var_names: Sequence[str],
        var_dtypes: Sequence[str],
        var_shapes: Sequence[Sequence[int]],
        var_is_free: Sequence[bool],
        *,
        created_at: datetime = None,
    ):
        self._created_at = created_at or datetime.datetime.now().astimezone(datetime.timezone.utc)
        self._run_id = run_id
        self._var_names = tuple(var_names)
        self._var_dtypes = tuple(var_dtypes)
        self._var_shapes = tuple(map(tuple, var_shapes))
        self._var_is_free = tuple(map(bool, var_is_free))

    @property
    def created_at(self) -> datetime.datetime:
        return self._created_at

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def var_names(self) -> Tuple[str]:
        return self._var_names

    @property
    def var_dtypes(self) -> Tuple[str]:
        return self._var_dtypes

    @property
    def var_shapes(self) -> Tuple[Tuple[int, ...]]:
        return self._var_shapes

    @property
    def var_is_free(self) -> Tuple[bool]:
        return self._var_is_free


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
        self, chain_id: str, draw_idx: int, var_names: Sequence[str]
    ) -> Dict[str, numpy.ndarray]:
        """Retrieve one draw from an MCMC chain."""
        raise NotImplementedError()

    def get_variable(self, chain_id: str, var_name: str) -> Dict[str, numpy.ndarray]:
        """Retrieve all draws of a variable from an MCMC chain."""
        raise NotImplementedError()
