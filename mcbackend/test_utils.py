import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import arviz
import hagelkorn
import numpy
import pandas
import pytest

import mcbackend
from mcbackend import utils as mutils
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
            # String dtypes may be used for more complex information
            Variable("message", "str"),
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
            (random.randint(0, 10) if s == -1 else s)
            for s in var.shape
        )
        if "float" in var.dtype:
            draw[var.name] = numpy.random.normal(size=dshape).astype(var.dtype)
        elif var.dtype == "str":
            alphabet = tuple("abcdef#+*/'")
            words = [
                "".join(numpy.random.choice(alphabet, size=numpy.random.randint(3, 10)))
                for _ in range(int(numpy.prod(dshape)))
            ]
            draw[var.name] = numpy.array(words, dtype=var.dtype).reshape(dshape)
        else:
            draw[var.name] = numpy.random.randint(low=0, high=100, size=dshape).astype(var.dtype)
        assert draw[var.name].shape == dshape
    return draw


class BaseBackendTest:
    """Can be used to test different backends in the same way."""

    cls_backend: Optional[type] = None
    cls_run: Optional[type] = None
    cls_chain: Optional[type] = None

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
            if not is_rigid(var.shape) or var.dtype == "str":
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
                if var.dtype == "str":
                    assert tuple(actual.shape) == tuple(numpy.shape(expected))
                    # String dtypes have strange names
                    assert "str" in actual.dtype.name
                elif is_rigid(var.shape):
                    assert tuple(actual.shape) == tuple(numpy.shape(expected))
                    assert actual.dtype.name == var.dtype
                    numpy.testing.assert_array_equal(actual, expected)
                else:
                    # Non-ridid variables are returned as object-arrays.
                    assert actual.shape == (len(expected),)
                    assert actual.dtype == object
                    # Their values must be asserted elementwise to avoid shape problems.
                    for act, exp in zip(actual, expected):
                        numpy.testing.assert_array_equal(act, exp)
        pass

    @pytest.mark.parametrize(
        "slc",
        [
            None,
            slice(None, None, None),
            slice(2, None, None),
            slice(2, 10, None),
            slice(2, 15, 3),  # every 3rd
            slice(15, 2, -3),  # backwards every 3rd
            slice(2, 15, -3),  # empty
            slice(-8, None, None),  # the last 8
            slice(-8, -2, 2),
            slice(-50, -2, 2),
            slice(15, 10),  # empty
            slice(1, 1),  # empty
        ],
    )
    def test__get_slicing(self, slc: slice):
        # "A" are just numbers to make diagnosis easier.
        # "B" are dynamically shaped to cover the edge cases.
        rmeta = RunMeta(
            variables=[Variable("A", "uint8"), Variable("M", "str", [2, 3])],
            sample_stats=[Variable("B", "uint8", [2, -1])],
            data=[],
        )
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(0)

        # Generate draws and add them to the chain
        N = 20
        draws = [make_draw(rmeta.variables) for n in range(N)]
        stats = [make_draw(rmeta.sample_stats) for n in range(N)]
        for d, s in zip(draws, stats):
            chain.append(d, s)
        assert len(chain) == N

        # slc=None in this test means "don't pass it".
        # The implementations should default to slc=slice(None, None, None).
        kwargs = dict(slc=slc) if slc is not None else {}
        act_draws_A = chain.get_draws("A", **kwargs)
        act_draws_M = chain.get_draws("M", **kwargs)
        act_stats = chain.get_stats("B", **kwargs)
        expected_draws_A = [d["A"] for d in draws][slc or slice(None, None, None)]
        expected_draws_M = [d["M"] for d in draws][slc or slice(None, None, None)]
        expected_stats = [s["B"] for s in stats][slc or slice(None, None, None)]

        # Variable "A" has a rigid shape
        if expected_draws_A:
            numpy.testing.assert_array_equal(act_draws_A, expected_draws_A)
        else:
            assert len(act_draws_A) == 0

        # Variable "M" is a string matrix
        if expected_draws_M:
            numpy.testing.assert_array_equal(act_draws_M, expected_draws_M)
        else:
            assert len(act_draws_M) == 0

        # Stat "B" is dynamically shaped, which means we're dealing with
        # dtype=object arrays. These must be checked elementwise.
        assert len(act_stats) == len(expected_stats)
        assert act_stats.dtype == object
        for a, e in zip(act_stats, expected_stats):
            numpy.testing.assert_array_equal(a, e)
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

    @pytest.mark.parametrize("tstatname", ["tune", "sampler__tune", "nottune"])
    def test__to_inferencedata(self, tstatname, caplog):
        rmeta = make_runmeta(
            flexibility=False,
            sample_stats=[
                Variable(tstatname, "bool"),
                Variable("sampler_0__logp", "float32"),
                Variable("warning", "str"),
            ],
        )
        run = self.backend.init_run(rmeta)
        chain = run.init_chain(1)

        # Generate draws and add them to the chain
        n = 10
        draws = [make_draw(rmeta.variables) for _ in range(n)]
        stats = [make_draw(rmeta.sample_stats) for _ in range(n)]
        for i, (d, s) in enumerate(zip(draws, stats)):
            s[tstatname] = i < 4
            chain.append(d, s)

        idata = run.to_inferencedata()
        assert isinstance(idata, arviz.InferenceData)
        assert idata.warmup_posterior.dims["chain"] == 1
        assert idata.posterior.dims["chain"] == 1
        if tstatname == "nottune":
            # Splitting into warmup/posterior requires a tune stat!
            assert any("No 'tune' stat" in r.message for r in caplog.records)
            assert idata.warmup_posterior.dims["draw"] == 0
            assert idata.posterior.dims["draw"] == 10
        else:
            assert idata.warmup_posterior.dims["draw"] == 4
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


@dataclass
class AppendSpeed:
    draws_per_second: float
    bytes_per_draw: float

    @property
    def mib_per_second(self) -> float:
        return self.draws_per_second * self.bytes_per_draw / 1024 / 1024

    def __str__(self):
        return f"{self.mib_per_second:.1f} MiB/s ({self.draws_per_second:.1f} draws/s)"


def run_chain(run: Run, chain_number: int = 0, tmax: float = 10) -> AppendSpeed:
    """Append with max speed to one chain for `tmax` seconds."""
    draw = make_draw(run.meta.variables)
    bytes_per_draw = sum(v.size * v.itemsize for v in draw.values())

    chain = run.init_chain(chain_number)
    t_start = time.time()
    d = 0
    last_update = t_start
    while time.time() - t_start < tmax:
        chain.append(draw)
        d += 1
        now = time.time()
        if now - last_update > 1:
            print(f"Inserted {d} draws")
            last_update = now

    assert len(chain) == d
    t_end = time.time()
    dps = d / (t_end - t_start)
    return AppendSpeed(dps, bytes_per_draw)


class BackendBenchmark:
    """A collection of backend benchmarking methods."""

    backend: mcbackend.Backend

    def run_all_benchmarks(self) -> pandas.DataFrame:
        """Runs each benchmark method and summarizes the results in a DataFrame."""
        df = pandas.DataFrame(
            columns=["title", "bytes_per_draw", "append_speed", "description"]
        ).set_index("title")
        for attr in dir(BackendBenchmark):
            meth = getattr(self, attr, None)
            if callable(meth) and meth.__name__.startswith("measure_"):
                if hasattr(self, "setup_method"):
                    self.setup_method(meth)
                print(f"Running {meth.__name__}")
                speed = meth()
                df.loc[meth.__name__[8:], ["bytes_per_draw", "append_speed", "description"]] = (
                    speed.bytes_per_draw,
                    str(speed),
                    meth.__doc__,
                )
        return df

    def measure_many_draws(self) -> AppendSpeed:
        """One chain of (), (3,) and (5,2) float32 variables."""
        rmeta = RunMeta(
            rid=hagelkorn.random(),
            variables=[
                Variable("v1", "float32", []),
                Variable("v2", "float32", list((3,))),
                Variable("v3", "float32", [5, 2]),
            ],
        )
        return run_chain(self.backend.init_run(rmeta))

    def measure_many_variables(self) -> AppendSpeed:
        """One chain with 300 variables of shapes (), (3,) and (5,2)."""
        rmeta = RunMeta(
            rid=hagelkorn.random(),
            variables=[Variable(f"v{v}", "float32", [5, 2][: v % 2]) for v in range(300)],
        )
        return run_chain(self.backend.init_run(rmeta))

    def measure_big_variables(self) -> AppendSpeed:
        """One chain with 3 variables of shapes (100,), (1000,) and (100, 100)."""
        rmeta = RunMeta(
            rid=hagelkorn.random(),
            variables=[
                Variable("v1", "float32", list((100,))),
                Variable("v2", "float32", list((1000,))),
                Variable("v3", "float32", list((100, 100))),
            ],
        )
        return run_chain(self.backend.init_run(rmeta))


class CheckPerformance(BaseBackendTest, BackendBenchmark):
    """Checks that the backend is reasonably fast via various high-load tests."""

    def test__many_draws(self):
        speed = self.measure_many_draws()
        assert speed.draws_per_second > 5000 or speed.mib_per_second > 1
        pass

    def test__many_variables(self):
        speed = self.measure_many_variables()
        assert speed.draws_per_second > 500 or speed.mib_per_second > 5
        pass

    def test__big_variables(self):
        speed = self.measure_big_variables()
        assert speed.draws_per_second > 500 or speed.mib_per_second > 5
        pass


def test_as_array_from_ragged():
    even = mutils.as_array_from_ragged(
        [
            numpy.ones(2),
            numpy.ones(3),
        ]
    )
    assert isinstance(even, numpy.ndarray)
    assert even.dtype == numpy.dtype(object)
    pass
