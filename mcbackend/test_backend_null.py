import random

import hagelkorn
import numpy
import pytest

from mcbackend.backends.null import NullBackend, NullChain, NullRun
from mcbackend.core import RunMeta, is_rigid
from mcbackend.meta import Variable
from mcbackend.test_utils import CheckBehavior, CheckPerformance, make_runmeta, make_draw

class CheckNullBehavior(CheckBehavior):
    """
    Overrides tests which assert that data are recorded correctly
    We perform all the operations of the original test, but in the
    end we do the opposite: assert that an exception is raised
    when either `get_draws` or `get_draws_at` is called.
    Stats are still recorded, so that part of the tests is reproduced unchanged.
    """

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

        # Retrieve by index - Raises exception
        with pytest.raises(RuntimeError):
            chain.get_draws_at(0, [v.name for v in rmeta.variables])

        # NB: Stats are still recorded and can be retrieved as with other chains
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

        # Fetching variables raises exception
        for var in rmeta.variables:
            expected = [draw[var.name] for draw in draws]
            with pytest.raises(RuntimeError):
                chain.get_draws(var.name)

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
        with pytest.raises(RuntimeError):
            chain.get_draws("A", **kwargs)
        with pytest.raises(RuntimeError):
            chain.get_draws("M", **kwargs)
        act_stats = chain.get_stats("B", **kwargs)
        expected_stats = [s["B"] for s in stats][slc or slice(None, None, None)]

        # Stat "B" is dynamically shaped, which means we're dealing with
        # dtype=object arrays. These must be checked elementwise.
        assert len(act_stats) == len(expected_stats)
        assert act_stats.dtype == object
        for a, e in zip(act_stats, expected_stats):
            numpy.testing.assert_array_equal(a, e)
        pass

    def test__to_inferencedata(self):
        """
        NullBackend doesn’t support `to_inferencedata`, so there isn’t
        anything to test here.
        """
        pass

class TestNullBackend(CheckNullBehavior, CheckPerformance):
    cls_backend = NullBackend
    cls_run = NullRun
    cls_chain = NullChain

    # `test_targets` and `test_growing` are copied over from TestNumPyBackend.
    # The lines testing sample storage removed, since neither `_samples`
    # nor `_var_is_rigid` are not supported by NullBackend.
    # However if one were to add tests for `_stats` and `_stat_is_rigid`
    # to the NumPy suite, we could port those here.

    def test_targets(self):
        imb = NullBackend(preallocate=123)
        rm = RunMeta(
            rid=hagelkorn.random(),
            variables=[
                Variable("tensor", "int8", (3, 4, 5)),
                Variable("scalar", "float64", ()),
                Variable("changeling", "uint16", (3, -1)),
            ],
        )
        run = imb.init_run(rm)
        chain = run.init_chain(0)
        pass

    @pytest.mark.parametrize("preallocate", [0, 75])
    def test_growing(self, preallocate):
        imb = NullBackend(preallocate=preallocate)
        rm = RunMeta(
            rid=hagelkorn.random(),
            variables=[
                Variable(
                    "A",
                    "float32",
                    (2,),
                ),
                Variable(
                    "B",
                    "float32",
                    (-1,),
                ),
            ],
        )
        run = imb.init_run(rm)
        chain = run.init_chain(0)
        # TODO: Check dimensions of stats array ?
        for _ in range(130):
            draw = {
                "A": numpy.random.uniform(size=(2,)),
                "B": numpy.random.uniform(size=(random.randint(0, 10),)),
            }
            chain.append(draw)
        # TODO: Check dimensions of stats array ?
        pass

if __name__ == "__main__":
    tc = TestNullBackend()
    df = tc.run_all_benchmarks()
    print(df)
