from datetime import datetime, timezone

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
    def test_chain_id(self):
        meta = ChainMeta("testid", 7)
        chain = core.Chain(meta, RunMeta())
        assert chain.cid == "testid_chain_7"
        pass
