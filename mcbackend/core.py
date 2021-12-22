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


class Chain:
    """A handle on one Markov-chain."""

    def __init__(self, meta: ChainMeta) -> None:
        self.meta = meta
        super().__init__()

    def add_draw(self, draw: Dict[str, numpy.ndarray]):
        raise NotImplementedError()

    def get_variable(self, var_name: str) -> numpy.ndarray:
        """Retrieve all draws of a variable from an MCMC chain."""
        raise NotImplementedError()

    def get_draw(self, draw_idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        """Retrieve one draw from an MCMC chain."""
        raise NotImplementedError()


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
