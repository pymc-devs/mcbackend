import random
from typing import Sequence

import arviz
import hagelkorn
import numpy
import pytest

from mcbackend.meta import ChainMeta, DataVariable, RunMeta, Variable
from mcbackend.npproto import utils

from .core import Backend, Chain, Run, is_rigid


def make_runmeta(*, flexibility: bool = False, **kwargs) -> RunMeta:
    defaults = dict(
        rid=hagelkorn.random(),
        variables=[
            Variable("tensor", "int8", [3, 4, 5], dims=["a", "b", "c"], is_deterministic=True),
            Variable("scalar", "float64", [], dims=[], is_deterministic=False),
        ],
        sample_stats=[
            # Compound/blocked stepping may emit stats for each sampler.
            Variable("accepted", "bool", list((3,)), dims=["sampler"]),
            # But some stats may refer to the iteration.
            Variable("logp", "float64", []),
        ],
        data=[
            DataVariable(
                "seconds", utils.ndarray_from_numpy(numpy.array([0, 0.5, 1.5])), dims=["time"]
            ),
            DataVariable(
                "coinflips",
                utils.ndarray_from_numpy(numpy.array([[0, 1, 0], [1, 1, 0]])),
                dims=["coin", "flip"],
                is_observed=True,
            ),
        ],
    )
    if flexibility:
        defaults["variables"].append(
            Variable("changeling", "uint16", [3, 0], dims=["a", "d"], is_deterministic=True)
        )
    defaults.update(kwargs)
    return RunMeta(**defaults)


def make_draw(variables: Sequence[Variable]):
    draw = {}
    for var in variables:
        dshape = tuple(
            # A pre-registered dim length of 0 means that it's random!
            s or random.randint(0, 10)
            for s in var.shape
        )
        if "float" in var.dtype:
            draw[var.name] = numpy.random.normal(size=dshape).astype(var.dtype)
        else:
            draw[var.name] = numpy.random.randint(low=0, high=100, size=dshape).astype(var.dtype)
    return draw


class BaseBackendTest:
    """Can be used to test different backends in the same way."""

    cls_backend = None
    cls_run = None
    cls_chain = None

    def setup_method(self, method):
        """Override this when the backend has no parameterless constructor."""
        self.backend: Backend = self.cls_backend()

    def teardown_method(self, method):
        pass


class CheckBehavior(BaseBackendTest):
    """Validates that a backend shows the expected behavior via the common API."""

    def test__initialization(self):
        assert isinstance(self.backend, Backend)
        assert isinstance(self.backend, self.cls_backend)
        rmeta = make_runmeta()
        run = self.backend.init_run(rmeta)
        assert isinstance(run, Run)
        assert isinstance(run, self.cls_run)
        chain = run.init_chain(7)
        assert isinstance(chain.cmeta, ChainMeta)
        assert chain.cmeta.chain_number == 7
        assert isinstance(chain, Chain)
        assert isinstance(chain, self.cls_chain)
        pass

    @pytest.mark.parametrize("with_stats", [False, True])
    def test__append_get_at(self, with_stats):
        rmeta = make_runmeta()
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(7)

        # Generate data
        draw = make_draw(rmeta.variables)
        stats = make_draw(rmeta.sample_stats) if with_stats else None

        # Append to the chain
        assert len(chain) == 0
        chain.append(draw, stats)
        assert len(chain) == 1

        # Retrieve by index
        actual = chain.get_draws_at(0, [v.name for v in rmeta.variables])
        assert isinstance(actual, dict)
        assert set(actual) == set(draw)
        for vn, act in actual.items():
            numpy.testing.assert_array_equal(act, draw[vn])

        if with_stats:
            actual = chain.get_stats_at(0, [v.name for v in rmeta.sample_stats])
            assert isinstance(actual, dict)
            assert set(actual) == set(stats)
            for vn, act in actual.items():
                numpy.testing.assert_array_equal(act, stats[vn])
        pass

    @pytest.mark.parametrize("with_stats", [False, True])
    def test__append_get_with_changelings(self, with_stats):
        rmeta = make_runmeta(flexibility=True)
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(7)

        # Generate draws and add them to the chain
        n = 10
        draws = [make_draw(rmeta.variables) for _ in range(n)]
        if with_stats:
            stats = [make_draw(rmeta.sample_stats) for _ in range(n)]
        else:
            stats = [None] * n

        for d, s in zip(draws, stats):
            chain.append(d, s)

        # Fetch each variable once to check the returned type and values
        for var in rmeta.variables:
            expected = [draw[var.name] for draw in draws]
            actual = chain.get_draws(var.name)
            assert isinstance(actual, numpy.ndarray)
            if var.name == "changeling":
                # Non-ridid variables are returned as object-arrays.
                assert actual.shape == (len(expected),)
                assert actual.dtype == object
                # Their values must be asserted elementwise to avoid shape problems.
                for act, exp in zip(actual, expected):
                    numpy.testing.assert_array_equal(act, exp)
            else:
                assert tuple(actual.shape) == tuple(numpy.shape(expected))
                assert actual.dtype == var.dtype
                numpy.testing.assert_array_equal(actual, expected)

        if with_stats:
            for var in rmeta.sample_stats:
                expected = [stat[var.name] for stat in stats]
                actual = chain.get_stats(var.name)
                assert isinstance(actual, numpy.ndarray)
                if is_rigid(var.shape):
                    assert tuple(actual.shape) == tuple(numpy.shape(expected))
                    assert actual.dtype == var.dtype
                    numpy.testing.assert_array_equal(actual, expected)
                else:
                    # Non-ridid variables are returned as object-arrays.
                    assert actual.shape == (len(expected),)
                    assert actual.dtype == object
                    # Their values must be asserted elementwise to avoid shape problems.
                    for act, exp in zip(actual, expected):
                        numpy.testing.assert_array_equal(act, exp)
        pass

    def test__get_chains(self):
        rmeta = make_runmeta()
        run = self.backend.init_run(rmeta)
        for c in range(3):
            chain = run.init_chain(c)
            assert isinstance(chain, Chain)
            # Insert 1 draw, so we can check lengths of chains returned later
            chain.append(make_draw(rmeta.variables))

        chains = run.get_chains()
        assert len(chains) == 3
        for c, chain in enumerate(chains):
            assert isinstance(chain, Chain)
            assert chain.cmeta.chain_number == c
            assert chain.rmeta == run.meta
            assert len(chain) == 1
        pass

    def test__to_inferencedata(self):
        rmeta = make_runmeta(
            flexibility=False,
            sample_stats=[
                Variable("tune", "bool"),
                Variable("sampler_0__logp", "float32"),
            ],
        )
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(1)

        # Generate draws and add them to the chain
        n = 10
        draws = [make_draw(rmeta.variables) for _ in range(n)]
        stats = [make_draw(rmeta.sample_stats) for _ in range(n)]
        for i, (d, s) in enumerate(zip(draws, stats)):
            s["tune"] = i < 4
            chain.append(d, s)

        idata = run.to_inferencedata()
        assert isinstance(idata, arviz.InferenceData)
        assert idata.warmup_posterior.dims["chain"] == 1
        assert idata.warmup_posterior.dims["draw"] == 4
        assert idata.posterior.dims["chain"] == 1
        assert idata.posterior.dims["draw"] == 6
        for var in rmeta.variables:
            assert var.name in set(idata.posterior.keys())
        for svar in rmeta.sample_stats:
            assert svar.name in set(idata.sample_stats.keys())
        # Data variables
        assert hasattr(idata, "constant_data")
        assert hasattr(idata, "observed_data")
        for dv in rmeta.data:
            if dv.is_observed:
                group = idata.observed_data
            else:
                group = idata.constant_data
            assert hasattr(group, dv.name)
            assert tuple(dv.dims) == tuple(group[dv.name].dims)
            numpy.testing.assert_array_equal(
                group[dv.name].values,
                utils.ndarray_to_numpy(dv.value),
            )
        pass


class CheckPerformance(BaseBackendTest):
    """Checks that the backend is reasonably fast via various high-load tests."""

    def test__many_draws(self):
        pass

    def test__many_variables(self):
        pass

    def test__big_variables(self):
        pass
