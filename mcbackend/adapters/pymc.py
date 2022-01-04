"""
This module implements an adapter to use any ``mcbackend.Backend`` as a PyMC ``BaseTrace``.

The only PyMC dependency is on the ``BaseTrace`` abstract base class.
"""
from typing import Dict, List, Sequence, Tuple

import hagelkorn
import numpy
from pymc.backends.base import BaseTrace

from mcbackend.meta import Coordinate, Variable

from ..core import Backend, Chain, Run, RunMeta
from ..npproto.utils import ndarray_from_numpy


class ReadOnlyTrace(BaseTrace):
    """An in-memory read-only PyMC trace.

    This class exists to avoid creating a dependency on
    the PyMC NDArray that this project is trying to replace.
    """

    supports_sampler_stats = True

    def __init__(
        self,
        *,
        from_trace: BaseTrace,
        length: int,
        draws: Dict[str, numpy.ndarray],
        stats: Sequence[Dict[str, numpy.ndarray]],
    ):
        self._length = length
        self._draws = draws
        self._stats = stats
        super().__init__(from_trace.name, from_trace.model, from_trace.vars)
        self.chain = from_trace.chain
        self.sampler_vars = from_trace.sampler_vars
        self.var_shapes = from_trace.var_shapes
        self.var_dtypes = from_trace.var_dtypes
        self.varnames = from_trace.varnames

    def __len__(self) -> int:
        return self._length

    def get_values(self, varname, burn=0, thin=1):
        return self._draws[varname][burn::thin]

    def _get_sampler_stats(self, stat_name, sampler_idx, burn, thin):
        if sampler_idx is None:
            return [sts[stat_name][burn::thin] for sts in self._stats]
        return self._stats[sampler_idx][stat_name][burn::thin]

    def point(self, idx):
        return {vn: vals[idx] for vn, vals in self._draws.items()}

    def _slice(self, idx: slice) -> BaseTrace:
        sliced = ReadOnlyTrace(
            from_trace=self,
            length=len(numpy.arange(self._length)[idx]),
            draws={k: v[idx] for k, v in self._draws.items()},
            stats=[{k: v[idx] for k, v in sts.items()} for sts in self._stats],
        )
        return sliced


class TraceBackend(BaseTrace):
    """Adapter to create a PyMC backend from any McBackend."""

    supports_sampler_stats = True

    def __init__(  # pylint: disable=W0622
        self,
        backend: Backend,
        *,
        name: str = None,
        model=None,
        vars=None,
        test_point=None,
    ):
        self.chain = None
        super().__init__(name, model, vars, test_point)
        self.run_id = hagelkorn.random(digits=6)
        print(f"Backend run id: {self.run_id}")
        self._backend: Backend = backend

        # Sessions created from the underlying backend
        self._run: Run = None
        self._chain: Chain = None
        self._stat_groups: List[List[Tuple[str, str]]] = None
        self._length: int = 0

    def __len__(self) -> int:
        return self._length

    def setup(self, draws, chain, sampler_vars=None) -> None:
        super().setup(draws, chain, sampler_vars)
        self.chain = chain

        # Initialize backend sessions
        free_rv_names = [rv.name for rv in self.model.free_RVs]
        if not self._run:
            variables = [
                Variable(
                    name,
                    str(self.var_dtypes[name]),
                    list(self.var_shapes[name]),
                    dims=list(self.model.RV_dims[name]) if name in self.model.RV_dims else None,
                    is_deterministic=(name not in free_rv_names),
                )
                for name in self.varnames
            ]

            self._stat_groups = []
            sample_stats = [
                Variable("tune", "bool"),
            ]
            if sampler_vars is not None:
                # In PyMC the sampler stats are grouped by the sampler.
                # ⚠ PyMC currently does not inform backends about shapes/dims of sampler stats.
                for s, names_dtypes in enumerate(sampler_vars):
                    self._stat_groups.append([])
                    for statname, dtype in names_dtypes.items():
                        sname = f"sampler_{s}__{statname}"
                        svar = Variable(
                            name=sname,
                            dtype=numpy.dtype(dtype).name,
                            # This 👇 is needed until PyMC provides shapes ahead of time.
                            undefined_ndim=True,
                        )
                        self._stat_groups[s].append((sname, statname))
                        sample_stats.append(svar)

            coordinates = [
                Coordinate(dname, ndarray_from_numpy(numpy.array(cvals)))
                for dname, cvals in self.model.coords.items()
                if cvals is not None
            ]
            rmeta = RunMeta(
                self.run_id,
                variables=variables,
                coordinates=coordinates,
                sample_stats=sample_stats,
            )
            self._run = self._backend.init_run(rmeta)
        self._chain = self._run.init_chain(chain_number=chain)
        return

    def record(self, point, sampler_states=None):
        draw = dict(zip(self.varnames, self.fn(point)))
        if sampler_states is None:
            stats = None
        else:
            stats = {
                "tune": False,
            }
            # Unpack the stats by sampler to uniquely named stats.
            for s, sts in enumerate(sampler_states):
                for statname, sval in sts.items():
                    sname = f"sampler_{s}__{statname}"
                    stats[sname] = sval
                    # Make not whether this is a tuning iteration.
                    if statname == "tune":
                        stats["tune"] = sval

        self._chain.append(draw, stats)
        self._length += 1
        return

    def get_values(self, varname, burn=0, thin=1) -> numpy.ndarray:
        return self._chain.get_draws(varname)[burn::thin]

    def _get_stats(self, varname, burn=0, thin=1) -> numpy.ndarray:
        return self._chain.get_stats(varname)[burn::thin]

    def _get_sampler_stats(self, stat_name, sampler_idx, burn, thin):
        return self._get_stats(f"sampler_{sampler_idx}__{stat_name}", burn, thin)

    def point(self, idx: int):
        return self._chain.get_draws_at(idx, self.var_names)

    def as_readonly(self) -> ReadOnlyTrace:
        """Creates a PyMC trace object from this chain."""
        # Re-organize draws and stats in the PyMC style
        draws = {varname: self.get_values(varname) for varname in self.varnames}
        stats = [
            {statname: self._get_stats(sname) for sname, statname in namemap}
            for namemap in self._stat_groups
        ]

        rotrace = ReadOnlyTrace(
            from_trace=self,
            length=len(self),
            draws=draws,
            stats=stats,
        )
        return rotrace

    def _slice(self, idx) -> ReadOnlyTrace:
        return self.as_readonly()[idx]
