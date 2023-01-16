import logging

import clickhouse_driver
import hagelkorn
import numpy
import hopsy
import pytest

from mcbackend.meta import RunMeta

from .adapters.hopsy import TraceBackend
from .backends.clickhouse import ClickHouseBackend
from .test_backend_clickhouse import DB_KWARGS, HAS_REAL_DB

_log = logging.getLogger(__file__)


@pytest.fixture
def test_objects():
    A, b = [[1, 1, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], [1, 0, 0, 0]
    model = hopsy.Gaussian(mean=[0, 0, 0])
    problem = hopsy.Problem(A, b, model)

    n_chains = 4
    chains = [hopsy.MarkovChain(hopsy.GaussianProposal, problem, starting_point=[0, 0, 0])
              for i in range(n_chains)]
    for chain in chains:
        chain.proposal.stepsize = 0.2

    rngs = [hopsy.RandomNumberGenerator(seed=42, stream=i) for i in range(n_chains)]

    return chains, rngs


@pytest.mark.skipif(
    condition=not HAS_REAL_DB,
    reason="Integration tests need a ClickHouse server on localhost:9000 without authentication.",
)
class TestHopsyAdapter:
    def setup_method(self, method):
        """Initializes a fresh database just for this test method."""
        self._db = "testing_" + hagelkorn.random()
        self._client_main = clickhouse_driver.Client(**DB_KWARGS)
        self._client_main.execute(f"CREATE DATABASE {self._db};")
        self._client = clickhouse_driver.Client(**DB_KWARGS, database=self._db)
        self.backend = ClickHouseBackend(self._client)
        return

    def teardown_method(self, method):
        self._client.disconnect()
        self._client_main.execute(f"DROP DATABASE {self._db};")
        self._client_main.disconnect()
        return

    @pytest.mark.parametrize("cores", [1, 3])
    def test_cores(self, test_objects, cores):
        backend = ClickHouseBackend(self._client)

        # To extract the run meta that the adapter passes to the backend:
        args = []
        original = backend.init_run

        def wrapper(meta: RunMeta):
            args.append(meta)
            return original(meta)

        backend.init_run = wrapper

        chains, rngs = test_objects
        trace = TraceBackend(backend)
        record_meta = ['state_negative_log_likelihood', 'proposal.proposal']
        meta, samples = hopsy.sample(chains, rngs, n_samples=50, thinning=10, n_procs=cores,
                                     record_meta=record_meta, backend=trace)

        if not len(args) == 1:
            _log.warning("Run was initialized multiple times.")
        rmeta = args[0]

        assert numpy.all([var.name == "variable_{}".format(i) for i, var in enumerate(rmeta.variables)])
        assert numpy.all([var.name == record_meta[i].format(i) for i, var in enumerate(rmeta.sampler_stats)])
