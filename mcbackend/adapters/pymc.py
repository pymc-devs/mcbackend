"""
This module implements an adapter to use any ``mcbackend.Backend`` as a PyMC ``BaseTrace``.

The only PyMC dependency is on the ``BaseTrace`` abstract base class.
"""
import base64
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import hagelkorn
import numpy

try:
    from pytensor.graph.basic import Constant
    from pytensor.tensor.sharedvar import SharedVariable
except ModuleNotFoundError:
    from aesara.graph.basic import Constant
    from aesara.tensor.sharedvar import SharedVariable

from pymc.backends.base import BaseTrace
from pymc.model import Model

from mcbackend.meta import Coordinate, DataVariable, Variable

from ..core import Backend, Chain, Run, RunMeta
from ..npproto.utils import ndarray_from_numpy


def find_data(pmodel: Model) -> List[DataVariable]:
    """Extracts data variables from a model."""
    observed_rvs = {pmodel.rvs_to_values[rv] for rv in pmodel.observed_RVs}
    dvars = []
    # All data containers are named vars!
    for name, var in pmodel.named_vars.items():
        dv = DataVariable(name)
        if isinstance(var, Constant):
            dv.value = ndarray_from_numpy(var.data)
        elif isinstance(var, SharedVariable):
            dv.value = ndarray_from_numpy(var.get_value())
        else:
            continue
        dv.dims = list(pmodel.named_vars_to_dims.get(name, []))
        dv.is_observed = var in observed_rvs
        dvars.append(dv)
    return dvars


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
        self.chain: int = -1
        super().__init__(name, model, vars, test_point)
        self.run_id = hagelkorn.random(digits=6)
        print(f"Backend run id: {self.run_id}")
        self._backend: Backend = backend

        # Sessions created from the underlying backend
        self._run: Optional[Run] = None
        self._chain: Optional[Chain] = None
        self._stat_groups: List[List[Tuple[str, str]]] = []
        self._length: int = 0

    def __len__(self) -> int:
        return self._length

    def setup(
        self,
        draws: int,
        chain: int,
        sampler_vars: Optional[List[Dict[str, numpy.dtype]]] = None,
    ) -> None:
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
                    dims=list(self.model.named_vars_to_dims[name])
                    if name in self.model.named_vars_to_dims
                    else [],
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
                        if statname == "warning":
                            # SamplerWarnings will be pickled and stored as string!
                            svar = Variable(sname, "str")
                        else:
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
                data=find_data(self.model),
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
                    # Automatically pickle SamplerWarnings
                    if statname == "warning":
                        sval_bytes = pickle.dumps(sval)
                        sval = base64.encodebytes(sval_bytes).decode("ascii")
                    stats[sname] = numpy.asarray(sval)
                    # Make note whether this is a tuning iteration.
                    if statname == "tune":
                        stats["tune"] = sval

        self._chain.append(draw, stats)
        self._length += 1
        return

    def get_values(self, varname, burn=0, thin=1) -> numpy.ndarray:
        if self._chain is None:
            raise Exception("Trace setup was not completed. Call `.setup()` first.")
        return self._chain.get_draws(varname)[burn::thin]

    def _get_stats(self, varname, burn=0, thin=1) -> numpy.ndarray:
        if self._chain is None:
            raise Exception("Trace setup was not completed. Call `.setup()` first.")
        values = self._chain.get_stats(varname)[burn::thin]
        if "warning" in varname:
            objs = []
            for v in values:
                enc = v.encode("ascii")
                str_ = base64.decodebytes(enc)
                obj = pickle.loads(str_)
                objs.append(obj)
            values = numpy.array(objs, dtype=object)
        return values

    def _get_sampler_stats(self, stat_name, sampler_idx, burn, thin):
        if self._chain is None:
            raise Exception("Trace setup was not completed. Call `.setup()` first.")
        return self._get_stats(f"sampler_{sampler_idx}__{stat_name}", burn, thin)

    def point(self, idx: int):
        if self._chain is None:
            raise Exception("Trace setup was not completed. Call `.setup()` first.")
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
