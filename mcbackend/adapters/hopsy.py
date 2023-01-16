import numpy
import numpy as np

import hopsy
from typing import Dict, List, Optional, Sequence, Tuple
import hagelkorn

from mcbackend.meta import Coordinate, DataVariable, Variable

from ..core import Backend, Chain, Run, RunMeta


class TraceBackend(hopsy.BaseTrace):
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
        self._chain: Optional[Chain] = None

    def setup(
            self,
            chain_idx: int,
            n_samples: int,
            n_dim: int,
            meta_names: List[str] = None
    ) -> None:
        super().setup(chain_idx, n_samples, n_dim)

        # Initialize backend sessions
        self.var_names = ["variable_{}".format(i) for i in range(n_dim)]
        if not self._run:
            variables = [
                Variable(
                    name,
                    np.dtype(np.float).name,
                    [],
                    [],
                    is_deterministic=False,
                )
                for name in self.var_names
            ]

            sample_stats = []
            if meta_names is not None:
                # In PyMC the sampler stats are grouped by the sampler.
                # ⚠ PyMC currently does not inform backends about shapes/dims of sampler stats.
                for name in meta_names:
                    sample_stats.append(
                        Variable(
                            name=name,
                            dtype=np.dtype(numpy.float).name if name == "acceptance_rate" else np.dtype(numpy.ndarray).name,
                            # This 👇 is needed until PyMC provides shapes ahead of time.
                            # undefined_ndim=True,
                            undefined_ndim=True,
                        )
                    )

            run_meta = RunMeta(
                self.run_id,
                variables=variables,
                sample_stats=sample_stats,
            )
            self._run = self._backend.init_run(run_meta)
        self._chain = self._run.init_chain(chain_number=chain_idx)
        return

    def record(self, point, meta):
        draw = dict(zip(self.var_names, np.tolist(point)))

        self._chain.append(draw, meta)
        return

    def finish(self):
        pass
