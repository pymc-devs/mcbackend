import clickhouse_driver
import pymc as pm
import pytest

from .adapters.pymc import TraceBackend
from .clickhouse import ClickHouseBackend


@pytest.fixture
def simple_model():
    with pm.Model() as pmodel:
        pm.Normal("n")
        pm.Uniform("u", size=3)
        pm.Normal("2d", size=(5, 2))
        pm.Bernoulli("b", p=0.5)
    return pmodel


class TestPyMCAdapter:
    @pytest.mark.parametrize("cores", [1, 3])
    def test_cores(self, simple_model, cores):
        client = clickhouse_driver.Client("localhost")
        backend = ClickHouseBackend(client)

        with simple_model:
            trace = TraceBackend(backend)
            pm.sample(trace=trace, tune=3, draws=5, chains=2, cores=cores, step=pm.Metropolis())
        pass
