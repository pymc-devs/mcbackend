import numpy
import numpy as np

import hopsy
from typing import Dict, List, Optional, Union
import hagelkorn

from mcbackend.meta import Coordinate, DataVariable, Variable

from ..core import Backend, Chain, Run, RunMeta


class TraceBackend(hopsy.Backend):
    """Adapter to create a hospy backend from any McBackend."""

    supports_sampler_stats = True

    def __init__(self, backend: Backend, name: str = None):
        super().__init__(name)
        self.run_id = hagelkorn.random(digits=6)
        print(f"Backend run id: {self.run_id}")
        self._backend: Backend = backend

        # Names
        self.var_names = None

        # Sessions created from the underlying backend
        self._run: Optional[Run] = None
        self._chains: Optional[List[Chain]] = None

    def setup(
            self,
            n_chains: int,
            n_samples: int,
            n_dim: int,
            meta_names: List[str],
            meta_shapes: List[List[int]],
    ) -> None:
        super().setup(n_chains, n_samples, n_dim, meta_names, meta_shapes)

        # Initialize backend sessions
        self.var_names = ["variable_{}".format(i) for i in range(n_dim)]
        if not self._run:
            variables = [
                Variable(
                    name,
                    np.dtype(float).name,
                    [],
                    [],
                    is_deterministic=False,
                )
                for name in self.var_names
            ]

            sample_stats = []
            if meta_names is not None:
                for i in range(len(meta_names)):
                    sample_stats.append(Variable(name=meta_names[i], dtype=np.dtype(float).name, shape=meta_shapes[i]))


            run_meta = RunMeta(
                self.run_id,
                variables=variables,
                sample_stats=sample_stats,
            )
            self._run = self._backend.init_run(run_meta)
        self._chains = [self._run.init_chain(chain_number=i) for i in range(n_chains)]

    def record(self, chain_idx: int, state: np.ndarray, meta: Dict[str, Union[float, np.ndarray]]) -> None:
        draw = dict(zip(self.var_names, state.tolist()))

        self._chains[chain_idx].append(draw, meta)
        self._chains[chain_idx]._commit()

    def finish(self) -> None:
        pass
