import logging

import clickhouse_driver
import hagelkorn
import numpy
import pymc as pm
import pytest

from mcbackend.meta import RunMeta
from mcbackend.npproto.utils import ndarray_to_numpy

from .adapters.pymc import TraceBackend, find_data
from .backends.clickhouse import ClickHouseBackend
from .test_backend_clickhouse import HAS_REAL_DB

_log = logging.getLogger(__file__)


@pytest.fixture
def simple_model():
    seconds = numpy.linspace(0, 5)
    observations = numpy.random.normal(
        0.5 + numpy.random.uniform(size=3)[:, None] * seconds[None, :]
    )
    with pm.Model(
        coords={
            "condition": ["A", "B", "C"],
        }
    ) as pmodel:
        x = pm.Data("seconds", seconds, dims="time")
        a = pm.Normal("scalar")
        b = pm.Uniform("vector", dims="condition")
        pm.Deterministic("matrix", a + b[:, None] * x[None, :], dims=("condition", "time"))
        pm.Bernoulli("integer", p=0.5)
        obs = pm.Data("obs", observations, dims=("condition", "time"))
        pm.Normal("L", pmodel["matrix"], observed=obs, dims=("condition", "time"))
    return pmodel


def test_find_data(simple_model):
    dvars = find_data(simple_model)
    dvars = {dv.name: dv for dv in dvars}
    assert "seconds" in dvars
    assert dvars["seconds"].dims == ["time"]
    assert not dvars["seconds"].is_observed
    assert "obs" in dvars
    assert dvars["obs"].dims == ["condition", "time"]
    assert dvars["obs"].is_observed
    pass


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

        # To extract the run meta that the adapter passes to the backend:
        args = []
        original = backend.init_run

        def wrapper(meta: RunMeta):
            args.append(meta)
            return original(meta)

        backend.init_run = wrapper

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
        if not len(args) == 1:
            _log.warning("Run was initialized multiple times.")
        rmeta = args[0]

        # Chain lenghts after conversion
        assert idata.posterior.dims["chain"] == 2
        assert idata.posterior.dims["draw"] == 5
        assert idata.warmup_posterior.dims["chain"] == 2
        assert idata.warmup_posterior.dims["draw"] == 3

        # Tracking of named variable dimensions
        vars = {var.name: var for var in rmeta.variables}
        assert vars["vector"].dims == ["condition"]
        assert vars["matrix"].dims == ["condition", "time"]

        # Tracking of coordinates
        coords = {coord.name: coord for coord in rmeta.coordinates}
        assert (
            tuple(ndarray_to_numpy(coords["condition"].values)) == simple_model.coords["condition"]
        )

        # Meta-information which variables are free vs. deterministic
        assert not vars["vector"].is_deterministic
        assert vars["matrix"].is_deterministic

        # Data variables should be included in the RunMeta
        datavars = {dv.name: dv for dv in rmeta.data}
        # Constant data variables
        assert "seconds" in datavars
        seconds = datavars["seconds"]
        assert tuple(seconds.dims) == ("time",)
        assert not seconds.is_observed
        numpy.testing.assert_array_equal(
            ndarray_to_numpy(seconds.value), simple_model["seconds"].get_value()
        )
        # Observed data variables
        assert "obs" in datavars
        obs = datavars["obs"]
        assert tuple(obs.dims) == ("condition", "time")
        assert obs.is_observed
        numpy.testing.assert_array_equal(
            ndarray_to_numpy(obs.value), simple_model["obs"].get_value()
        )
        pass
