"""
This module implements an adapter to
"""
import logging

import hagelkorn
import pymc as pm

from ..core import BackendBase, ChainMeta, RunMeta

_log = logging.getLogger(__file__)


class TraceBackend(pm.backends.base.BaseTrace):
    """Adapter to create a PyMC backend from any McBackend."""

    supports_sampler_stats = False

    def __init__(  # pylint: disable=W0622
        self,
        backend: BackendBase,
        *,
        name: str = None,
        model: pm.Model = None,
        vars=None,
        test_point=None,
    ):
        super().__init__(name, model, vars, test_point)
        self.run_id = hagelkorn.random(digits=6)
        self._backend = backend
        self._stats = None
        # The following are specific to chains
        self.chain: int = None
        self._draw_idx: int = 0
        self._chain_id = None
        _log.info("Backend run id: %s", self.run_id)

    def __len__(self) -> int:
        return self._draw_idx

    def setup(self, draws, chain, sampler_vars=None) -> None:
        self.chain = chain
        super().setup(draws, chain, sampler_vars)

        # Determine relevant meta information
        rm = RunMeta(
            self.run_id,
            self.varnames,
            tuple(map(str, self.var_dtypes.values())),
            tuple(self.var_shapes.values()),
            [(rv in self.model.free_RVs) for rv in self.vars],
        )
        cm = ChainMeta(self.run_id, self.chain)

        # Initialize backend
        self._backend.init_backend()
        self._backend.init_run(rm)
        self._backend.init_chain(cm)
        self._chain_id = cm.chain_id
        return

    def record(self, point, sampler_stats=None) -> None:  # pylint: disable=W0613
        draw = dict(zip(self.varnames, self.fn(point)))
        self._backend.add_draw(self._chain_id, self._draw_idx, draw)
        self._draw_idx += 1
        return

    def get_values(self, varname, burn=0, thin=1):
        return self._backend.get_variable(self._chain_id, varname)[burn::thin]

    def point(self, idx: int):
        return self._backend.get_draw(self._chain_id, idx, self.var_names)

    def _slice(self, idx) -> pm.backends.base.BaseTrace:
        idx = slice(*idx.indices(len(self)))

        sliced = pm.backends.NDArray(model=self.model, vars=self.vars)
        sliced.chain = self.chain
        sliced.samples = {varname: self.get_values(varname)[idx] for varname in self.varnames}
        sliced.sampler_vars = self.sampler_vars
        sliced._draw_idx = (idx.stop - idx.start) // idx.step  # pylint: disable=W0212

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
