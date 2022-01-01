import clickhouse_driver
import hagelkorn
import pymc as pm
import pytest

from .adapters.pymc import TraceBackend
from .backends.clickhouse import ClickHouseBackend
from .test_backend_clickhouse import HAS_REAL_DB


@pytest.fixture
def simple_model():
    with pm.Model() as pmodel:
        pm.Normal("n")
        pm.Uniform("u", size=3)
        pm.Normal("2d", size=(5, 2))
        pm.Bernoulli("b", p=0.5)
    return pmodel


@pytest.mark.skipif(
    condition=not HAS_REAL_DB,
    reason="Integration tests need a ClickHouse server on localhost:9000 without authentication.",
)
class TestPyMCAdapter:
    def setup_method(self, method):
        """Initializes a fresh database just for this test method."""
        self._db = "testing_" + hagelkorn.random()
        self._client_main = clickhouse_driver.Client("localhost")
        self._client_main.execute(f"CREATE DATABASE {self._db};")
        self._client = clickhouse_driver.Client("localhost", database=self._db)
        self.backend = ClickHouseBackend(self._client)
        return

    def teardown_method(self, method):
        self._client.disconnect()
        self._client_main.execute(f"DROP DATABASE {self._db};")
        self._client_main.disconnect()
        return

    @pytest.mark.parametrize("cores", [1, 3])
    def test_cores(self, simple_model, cores):
        backend = ClickHouseBackend(self._client)

        with simple_model:
            trace = TraceBackend(backend)
            idata = pm.sample(
                trace=trace,
                tune=3,
                draws=5,
                chains=2,
                cores=cores,
                step=pm.Metropolis(),
                discard_tuned_samples=False,
                compute_convergence_checks=False,
            )
        assert idata.posterior.dims["chain"] == 2
        assert idata.posterior.dims["draw"] == 5
        assert idata.warmup_posterior.dims["chain"] == 2
        assert idata.warmup_posterior.dims["draw"] == 3
        pass
