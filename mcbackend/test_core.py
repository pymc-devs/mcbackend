from . import core


class TestRunMeta:
    pass


class TestChainMeta:
    def test_chain_id(self):
        meta = core.ChainMeta("testid", 7)
        assert meta.chain_id == "testid_chain_7"
        pass
