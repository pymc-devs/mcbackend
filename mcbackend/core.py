"""
Module with metadata structures and abstract classes.
"""
import collections
import logging
from typing import Dict, List, Mapping, Optional, Sequence, Sized, TypeVar, Union, cast

import numpy

from .meta import ChainMeta, RunMeta, Variable
from .npproto.utils import ndarray_to_numpy
from .utils import as_array_from_ragged

try:
    from arviz import InferenceData, from_dict

    _HAS_ARVIZ = True
except ModuleNotFoundError:
    InferenceData = TypeVar("InferenceData")  # type: ignore
    _HAS_ARVIZ = False

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
        - ``[2, -1]`` indicates a matrix with 2 rows and dynamic number of columns (rigid: False).
        - ``None`` indicates dynamic dimensionality (rigid: False).
    """
    if nshape is None:
        return False
    if any(s == -1 for s in nshape):
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
        self, draw: Mapping[str, numpy.ndarray], stats: Optional[Mapping[str, numpy.ndarray]] = None
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

    def get_draws(self, var_name: str, slc: slice = slice(None)) -> numpy.ndarray:
        """Retrieve draws of a variable from an MCMC chain.

        Parameters
        ----------
        var_name : str
            Name of the variable.
        slc : slice, optional
            Optional ``slice`` object to retrieve only a subset of elements.
            Passing this can be more performant than slicing the returned value.
        """
        raise NotImplementedError()

    def get_stats(self, stat_name: str, slc: slice = slice(None)) -> numpy.ndarray:
        """Retrieve values of a sampler statistic.

        Parameters
        ----------
        stat_name : str
            Name of the stats variable.
        slc : slice, optional
            Optional ``slice`` object to retrieve only a subset of elements.
            Passing this can be more performant than slicing the returned value.
        """
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
    def dims(self) -> Dict[str, List[str]]:
        dims = {}
        for var in self.meta.variables:
            if len(var.dims) == len(var.shape) and not var.undefined_ndim:
                dims[var.name] = list(var.dims)
        for dvar in self.meta.data:
            if len(dvar.dims) > 0:
                dims[dvar.name] = list(dvar.dims)
        return dims

    @property
    def constant_data(self) -> Dict[str, numpy.ndarray]:
        return {dv.name: ndarray_to_numpy(dv.value) for dv in self.meta.data if not dv.is_observed}

    @property
    def observed_data(self) -> Dict[str, numpy.ndarray]:
        return {dv.name: ndarray_to_numpy(dv.value) for dv in self.meta.data if dv.is_observed}

    def to_inferencedata(self, *, equalize_chain_lengths: bool = True, **kwargs) -> InferenceData:
        """Creates an ArviZ ``InferenceData`` object from this run.

        Parameters
        ----------
        equalize_chain_lengths : bool
            Whether to truncate all chains to the shortest chain length (default: ``True``).
        **kwargs
            Will be forwarded to ``arviz.from_dict()``.

        Returns
        -------
        idata : arviz.InferenceData
            Samples and metadata of this inference run.
        """
        if not _HAS_ARVIZ:
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
            msg = f"Chains vary in length. Lenghts are: {chain_lengths}"
            if not equalize_chain_lengths:
                msg += (
                    "\nArviZ does not properly support uneven chain lengths (see ArviZ issue #2094)."
                    "\nWe'll try to give you an InferenceData, but best case the chain & draw dimensions"
                    " will be messed-up as {'chain': 1, 'draws': n_chains}."
                    "\nYou won't be able to save this InferenceData to a file"
                    " and you should expect many ArviZ functions to choke on it."
                    "\nSpecify `to_inferencedata(equalize_chain_lengths=True)` to get regular InferenceData."
                )
            else:
                msg += "\nTruncating to the length of the shortest chain."
            _log.warning(msg)
        min_clen = None
        if equalize_chain_lengths:
            # A minimum chain length is introduced so that all chains have equal length
            min_clen = min(chain_lengths.values())
        # Aggregate draws and stats, while splitting into warmup/posterior
        warmup_posterior = collections.defaultdict(list)
        warmup_sample_stats = collections.defaultdict(list)
        posterior = collections.defaultdict(list)
        sample_stats = collections.defaultdict(list)
        for c, chain in enumerate(chains):
            # Create a slice to use when fetching the variables
            if min_clen is None:
                # Every retrieved array is shortened to the previously determined chain length.
                # Needed for backends which may get inserts inbetween our get_draws/get_stats calls.
                slc = slice(0, chain_lengths[chain.cid])
            else:
                slc = slice(0, min_clen)

            # Obtain a mask by which draws can be split into warmup/posterior
            if "tune" in chain.sample_stats:
                tune = chain.get_stats("tune", slc).astype(bool)
            else:
                if c == 0:
                    _log.warning(
                        "No 'tune' stat found. Assuming all iterations are posterior draws."
                    )
                tune = numpy.repeat((chain_lengths[chain.cid],), False)

            # Split all variables draws into warmup/posterior
            for var in variables:
                draws = chain.get_draws(var.name, slc)
                warmup_posterior[var.name].append(draws[tune])
                posterior[var.name].append(draws[~tune])
            # Same for sample stats
            for svar in self.meta.sample_stats:
                stats = chain.get_stats(svar.name, slc)
                warmup_sample_stats[svar.name].append(stats[tune])
                sample_stats[svar.name].append(stats[~tune])

        w_pst = cast(Dict[str, Union[Sequence, numpy.ndarray]], warmup_posterior)
        w_ss = cast(Dict[str, Union[Sequence, numpy.ndarray]], warmup_sample_stats)
        pst = cast(Dict[str, Union[Sequence, numpy.ndarray]], posterior)
        ss = cast(Dict[str, Union[Sequence, numpy.ndarray]], sample_stats)
        if not equalize_chain_lengths:
            # Convert ragged arrays to object-dtyped ndarray because NumPy >=1.24.0 no longer does that automatically
            w_pst = {k: as_array_from_ragged(v) for k, v in warmup_posterior.items()}
            w_ss = {k: as_array_from_ragged(v) for k, v in warmup_sample_stats.items()}
            pst = {k: as_array_from_ragged(v) for k, v in posterior.items()}
            ss = {k: as_array_from_ragged(v) for k, v in sample_stats.items()}

        idata = from_dict(
            warmup_posterior=w_pst,
            warmup_sample_stats=w_ss,
            posterior=pst,
            sample_stats=ss,
            coords=self.coords,
            dims=self.dims,
            attrs=self.meta.attributes,
            constant_data=self.constant_data,
            observed_data=self.observed_data,
            save_warmup=True,
            **kwargs,
        )
        return idata


class Backend:
    """Base class for all MCMC draw storage backends."""

    def init_run(self, meta: RunMeta) -> Run:
        """Register a new MCMC run with the backend."""
        raise NotImplementedError()
