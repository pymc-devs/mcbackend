from datetime import datetime, timezone

import pytest

from mcbackend.meta import ChainMeta, RunMeta, Variable

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
