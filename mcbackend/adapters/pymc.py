"""
This module implements an adapter to
"""
import hagelkorn
import numpy
import pymc as pm

from mcbackend.meta import Coordinate, Variable

from ..core import Backend, Chain, Run, RunMeta
from ..npproto.utils import ndarray_from_numpy


class TraceBackend(pm.backends.base.BaseTrace):
    """Adapter to create a PyMC backend from any McBackend."""

    supports_sampler_stats = False

    def __init__(  # pylint: disable=W0622
        self,
        backend: Backend,
        *,
        name: str = None,
        model: pm.Model = None,
        vars=None,
        test_point=None,
    ):
        super().__init__(name, model, vars, test_point)
        self.run_id = hagelkorn.random(digits=6)
        print(f"Backend run id: {self.run_id}")
        self._backend = backend
        self._stats = None

        # Sessions created from the underlying backend
        self._run: Run = None
        self._chain: Chain = None

        # These are needed py the PyMC BaseTrace
        self.chain: int = None
        self._draw_idx: int = 0

    def __len__(self) -> int:
        return self._draw_idx

    def setup(self, draws, chain, sampler_vars=None) -> None:
        self.chain = chain
        super().setup(draws, chain, sampler_vars)

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
            coordinates = [
                Coordinate(dname, ndarray_from_numpy(numpy.array(cvals)))
                for dname, cvals in self.model.coords.items()
                if cvals is not None
            ]
            rmeta = RunMeta(
                self.run_id,
                variables=variables,
                coordinates=coordinates,
            )
            self._run = self._backend.init_run(rmeta)
        self._chain = self._run.init_chain(chain_number=chain)
        return

    def record(self, point, sampler_stats=None) -> None:  # pylint: disable=W0613
        draw = dict(zip(self.varnames, self.fn(point)))
        self._chain.add_draw(draw)
        self._draw_idx += 1
        return

    def get_values(self, varname, burn=0, thin=1):
        return self._chain.get_variable(varname)[burn::thin]

    def point(self, idx: int):
        return self._chain.get_draw(idx, self.var_names)

    def _slice(self, idx) -> pm.backends.base.BaseTrace:
        idx = slice(*idx.indices(len(self)))

        sliced = pm.backends.NDArray(model=self.model, vars=self.vars)
        sliced.chain = self.chain
        sliced.samples = {varname: self.get_values(varname)[idx] for varname in self.varnames}
        sliced.sampler_vars = self.sampler_vars
        sliced.draw_idx = (idx.stop - idx.start) // idx.step  # pylint: disable=W0212

        if self._stats is None:
            return sliced
        sliced._stats = []  # pylint: disable=W0212
        for svars in self._stats:  # pylint: disable=E1133
            var_sliced = {}
            sliced._stats.append(var_sliced)  # pylint: disable=W0212
            for key, vals in svars.items():
                var_sliced[key] = vals[idx]

        return sliced

    def close(self):  # pylint: disable=R0201
        return
