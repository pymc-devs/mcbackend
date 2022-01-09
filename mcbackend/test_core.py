from datetime import datetime, timezone

import numpy
import pytest

from mcbackend.meta import ChainMeta, RunMeta, Variable
from mcbackend.test_utils import make_runmeta

from . import core


def test_is_rigid():
    assert core.is_rigid([])
    assert core.is_rigid([1, 2])
    assert not core.is_rigid(None)
    assert not core.is_rigid((0,))
    assert not core.is_rigid([1, 0, 2])
    pass


def test_meta_equals():
    kwargs = dict(
        name="bla",
        dtype="float32",
        shape=[1, 2],
        dims=["a", "b"],
        is_deterministic=True,
    )
    assert Variable(**kwargs) == Variable(**kwargs)
    assert ChainMeta("ABC", 0) == ChainMeta("ABC", 0)
    now = datetime.now().astimezone(timezone.utc)
    v1 = Variable("v1", "float32", ())
    assert RunMeta("ABC", now, [v1]) == RunMeta("ABC", now, [v1])
    pass


class TestRun:
    def test_run_properties(self):
        rmeta = make_runmeta()
        run = core.Run(rmeta)
        assert isinstance(run.constant_data, dict)
        assert isinstance(run.observed_data, dict)
        assert "seconds" in run.constant_data
        assert run.dims.get("seconds", None) == ["time"]
        assert "coinflips" in run.observed_data
        assert run.dims.get("coinflips", None) == ["coin", "flip"]
        pass


class TestChain:
    def test_chain_properties(self):
        rmeta = RunMeta(
            rid="testid",
            variables=[
                Variable("v1", "float32", []),
            ],
            sample_stats=[
                Variable("s1", "bool", []),
            ],
        )
        cmeta = ChainMeta(rmeta.rid, 7)
        chain = core.Chain(cmeta, rmeta)

        assert chain.cid == "testid_chain_7"

        assert isinstance(chain.variables, dict)
        assert chain.variables["v1"] == rmeta.variables[0]

        assert isinstance(chain.sample_stats, dict)
        assert chain.sample_stats["s1"] == rmeta.sample_stats[0]
        pass

    def test_chain_length(self):
        class _TestChain(core.Chain):
            def get_draws(self, var_name: str):
                return numpy.arange(12)

            def get_stats(self, stat_name: str):
                return numpy.arange(42)

        rmeta = RunMeta("test", variables=[Variable("v1")])
        cmeta = ChainMeta("test", 0)
        assert len(_TestChain(cmeta, rmeta)) == 12

        rmeta = RunMeta("test", sample_stats=[Variable("s1")])
        cmeta = ChainMeta("test", 0)
        assert len(_TestChain(cmeta, rmeta)) == 42
        pass
