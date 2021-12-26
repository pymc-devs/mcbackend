from . import core


def test_is_rigid():
    assert core.is_rigid([])
    assert core.is_rigid([1, 2])
    assert not core.is_rigid(None)
    assert not core.is_rigid((0,))
    assert not core.is_rigid([1, 0, 2])
    pass


class TestRun:
    pass


class TestChain:
    def test_chain_id(self):
        meta = core.ChainMeta("testid", 7)
        chain = core.Chain(meta, core.RunMeta("test", [], [], [], []))
        assert chain.cid == "testid_chain_7"
        pass
