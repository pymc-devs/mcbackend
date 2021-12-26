"""
Module with metadata structures and abstract classes.
"""
import datetime
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy

Shape = Sequence[int]


def is_rigid(shape: Optional[Shape]):
    if shape is None:
        return False
    return all(s != 0 for s in shape)


class RunMeta:
    """Metadata of a multi-chain MCMC run."""

    def __init__(
        self,
        rid: str,
        var_names: Sequence[str],
        var_dtypes: Sequence[str],
        var_shapes: Sequence[Sequence[int]],
        var_is_free: Sequence[bool],
        *,
        created_at: datetime = None,
    ):
        self._created_at = created_at or datetime.datetime.now().astimezone(datetime.timezone.utc)
        self._rid = rid
        self._var_names = tuple(var_names)
        self._var_dtypes = tuple(var_dtypes)
        self._var_shapes = tuple(map(tuple, var_shapes))
        self._var_is_free = tuple(map(bool, var_is_free))

    @property
    def created_at(self) -> datetime.datetime:
        return self._created_at

    @property
    def rid(self) -> str:
        return self._rid

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

    rid: str
    chain_number: int

    @property
    def id(self) -> str:
        """Unique identifier of this MCMC chain."""
        return chain_id(self)


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
