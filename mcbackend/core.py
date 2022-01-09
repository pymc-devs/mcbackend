"""
Module with metadata structures and abstract classes.
"""
import collections
import logging
from typing import Dict, Optional, Sequence, Sized, TypeVar

import numpy

from .meta import ChainMeta, RunMeta, Variable
from .npproto.utils import ndarray_to_numpy

try:
    from arviz import InferenceData, from_dict
except ModuleNotFoundError:
    InferenceData = TypeVar("InferenceData")

Shape = Sequence[int]
_log = logging.getLogger(__file__)


def is_rigid(nshape: Optional[Shape]):
    """Determines wheather the shape is constant.

    Parameters
    ----------
    nshape : array-like, optional
        This "nullable shape" is interpreted as follows:
        - ``[]`` indicates scalar shape (rigid: True).
        - ``[2, 3]`` indicates a matrix with 2 rows and 3 columns (rigid: True).
        - ``[2, 0]`` indicates a matrix with 2 rows and dynamic number of columns (rigid: False).
        - ``None`` indicates dynamic dimensionality (rigid: False).
    """
    if nshape is None or any(s == 0 for s in nshape):
        return False
    return True


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

    def get_chains(self) -> Sequence[Chain]:
        raise NotImplementedError()

    @property
    def coords(self) -> Dict[str, numpy.ndarray]:
        return {coord.name: ndarray_to_numpy(coord.values) for coord in self.meta.coordinates}

    @property
    def dims(self) -> Dict[str, Sequence[str]]:
        dims = {}
        for var in self.meta.variables:
            if len(var.dims) == len(var.shape) and not var.undefined_ndim:
                dims[var.name] = var.dims
        for dvar in self.meta.data:
            if len(dvar.dims) > 0:
                dims[dvar.name] = dvar.dims
        return dims

    @property
    def constant_data(self) -> Dict[str, numpy.ndarray]:
        return {dv.name: ndarray_to_numpy(dv.value) for dv in self.meta.data if not dv.is_observed}

    @property
    def observed_data(self) -> Dict[str, numpy.ndarray]:
        return {dv.name: ndarray_to_numpy(dv.value) for dv in self.meta.data if dv.is_observed}

    def to_inferencedata(self, **kwargs) -> InferenceData:
        """Creates an ArviZ ``InferenceData`` object from this run.

        Parameters
        ----------
        **kwargs
            Will be forwarded to ``arviz.from_dict()``.

        Returns
        -------
        idata : arviz.InferenceData
            Samples and metadata of this inference run.
        """
        if isinstance(InferenceData, TypeVar):
            raise ModuleNotFoundError("ArviZ is not installed.")

        variables = self.meta.variables
        chains = self.get_chains()

        nonrigid_vars = {var for var in variables if var.undefined_ndim or not is_rigid(var.shape)}
        if nonrigid_vars:
            raise NotImplementedError(
                "Creating InferenceData from runs with non-rigid variables is not supported."
                f" The non-rigid variables are: {nonrigid_vars}."
            )

        chain_lengths = {c.cid: len(c) for c in chains}
        if len(set(chain_lengths.values())) != 1:
            _log.warning("Chains vary in length. Lenghts are: %s", chain_lengths)

        # Aggregate draws and stats, while splitting into warmup/posterior
        warmup_posterior = collections.defaultdict(list)
        warmup_sample_stats = collections.defaultdict(list)
        posterior = collections.defaultdict(list)
        sample_stats = collections.defaultdict(list)
        for c, chain in enumerate(chains):
            # Obtain a mask by which draws can be split into warmup/posterior
            if "tune" in chain.sample_stats:
                tune = chain.get_stats("tune").astype(bool)
            else:
                if c == 0:
                    _log.warning(
                        "No 'tune' stat found. Assuming all iterations are posterior draws."
                    )
                tune = numpy.repeat((chain_lengths[chain.cid],), False)

            # Split all variables draws into warmup/posterior
            for var in variables:
                draws = chain.get_draws(var.name)
                warmup_posterior[var.name].append(draws[tune])
                posterior[var.name].append(draws[~tune])
            # Same for sample stats
            for svar in self.meta.sample_stats:
                stats = chain.get_stats(svar.name)
                warmup_sample_stats[svar.name].append(stats[tune])
                sample_stats[svar.name].append(stats[~tune])

        kwargs.setdefault("save_warmup", True)
        idata = from_dict(
            warmup_posterior=warmup_posterior,
            warmup_sample_stats=warmup_sample_stats,
            posterior=posterior,
            sample_stats=sample_stats,
            coords=self.coords,
            dims=self.dims,
            attrs=self.meta.attributes,
            constant_data=self.constant_data,
            observed_data=self.observed_data,
            **kwargs,
        )
        return idata


class Backend:
    """Base class for all MCMC draw storage backends."""

    def init_run(self, meta: RunMeta) -> Run:
        """Register a new MCMC run with the backend."""
        raise NotImplementedError()
