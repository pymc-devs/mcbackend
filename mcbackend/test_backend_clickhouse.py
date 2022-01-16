import logging
from subprocess import call
from typing import Sequence, Tuple

import clickhouse_driver
import hagelkorn
import numpy
import pandas
import pytest

from mcbackend.backends.clickhouse import (
    ClickHouseBackend,
    ClickHouseChain,
    ClickHouseRun,
    create_chain_table,
    create_runs_table,
)
from mcbackend.core import Chain, Run, chain_id
from mcbackend.meta import ChainMeta, RunMeta, Variable
from mcbackend.test_utils import CheckBehavior, CheckPerformance, make_runmeta

try:
    client = clickhouse_driver.Client("localhost")
    client.execute("SHOW DATABASES;")
    HAS_REAL_DB = True
except:
    HAS_REAL_DB = False


def fully_initialized(
    cbackend: ClickHouseBackend, rmeta: RunMeta, *, nchains: int = 1
) -> Tuple[ClickHouseRun, Sequence[ClickHouseChain]]:
    run = cbackend.init_run(rmeta)
    chains = []
    for c in range(nchains):
        chain = run.init_chain(c)
        chains.append(chain)
    return run, chains


@pytest.mark.skipif(
    condition=not HAS_REAL_DB,
    reason="Integration tests need a ClickHouse server on localhost:9000 without authentication.",
)
class TestClickHouseBackendInitialization:
    """This is separate because ``TestClickHouseBackend.setup_method`` depends on these things."""

    def test_exceptions(self):
        with pytest.raises(ValueError, match="must be provided"):
            ClickHouseBackend()
        pass

    def test_backend_from_client_object(self):
        db = "testing_" + hagelkorn.random()
        _client_main = clickhouse_driver.Client("localhost")
        _client_main.execute(f"CREATE DATABASE {db};")

        try:
            # When created from a client object, all chains share the client
            backend = ClickHouseBackend(client=clickhouse_driver.Client("localhost", database=db))
            assert callable(backend._client_fn)
            run = backend.init_run(make_runmeta())
            c1 = run.init_chain(0)
            c2 = run.init_chain(1)
            assert c1._client is c2._client
        finally:
            _client_main.execute(f"DROP DATABASE {db};")
            _client_main.disconnect()
        pass

    def test_backend_from_client_function(self):
        db = "testing_" + hagelkorn.random()
        _client_main = clickhouse_driver.Client("localhost")
        _client_main.execute(f"CREATE DATABASE {db};")

        def client_fn():
            return clickhouse_driver.Client("localhost", database=db)

        try:
            # When created from a client function, each chain has its own client
            backend = ClickHouseBackend(client_fn=client_fn)
            assert backend._client is not None
            run = backend.init_run(make_runmeta())
            c1 = run.init_chain(0)
            c2 = run.init_chain(1)
            assert c1._client is not c2._client

            # By passing both, one may use different settings
            bclient = client_fn()
            backend = ClickHouseBackend(client=bclient, client_fn=client_fn)
            assert backend._client is bclient
        finally:
            _client_main.execute(f"DROP DATABASE {db};")
            _client_main.disconnect()
        pass


@pytest.mark.skipif(
    condition=not HAS_REAL_DB,
    reason="Integration tests need a ClickHouse server on localhost:9000 without authentication.",
)
class TestClickHouseBackend(CheckBehavior, CheckPerformance):
    cls_backend = ClickHouseBackend
    cls_run = ClickHouseRun
    cls_chain = ClickHouseChain

    def setup_method(self, method):
        """Initializes a fresh database just for this test method."""
        self._db = "testing_" + hagelkorn.random()
        self._client_main = clickhouse_driver.Client("localhost")
        self._client_main.execute(f"CREATE DATABASE {self._db};")
        self._client = clickhouse_driver.Client("localhost", database=self._db)
        self.backend = ClickHouseBackend(
            client_fn=lambda: clickhouse_driver.Client("localhost", database=self._db)
        )
        return

    def teardown_method(self, method):
        self._client.disconnect()
        self._client_main.execute(f"DROP DATABASE {self._db};")
        self._client_main.disconnect()
        return

    def test_test_database(self):
        assert self._client.execute("SHOW TABLES;") == [("runs",)]
        pass

    def test_init_run(self):
        runs = self.backend.get_runs()
        assert len(runs) == 0

        meta = make_runmeta(rid="my_first_run")
        run = self.backend.init_run(meta)
        assert isinstance(run, Run)
        assert len(self._client.execute("SELECT * FROM runs;")) == 1
        runs = self.backend.get_runs()
        assert isinstance(runs, pandas.DataFrame)
        assert runs.index.name == "rid"
        assert "my_first_run" in runs.index.values
        pass

    def test_get_run(self):
        meta = make_runmeta()
        self.backend.init_run(meta)
        run = self.backend.get_run(meta.rid)
        assert run.meta == meta
        pass

    def test_create_chain_table(self):
        rmeta = make_runmeta(
            variables=[
                Variable("scalar", "uint16", []),
                Variable("1D", "float32", list((3,))),
                Variable("3D", "float64", [2, 5, 6]),
            ],
            sample_stats=[
                Variable("accepted", "bool", []),
            ],
        )
        self.backend.init_run(rmeta)
        cmeta = ChainMeta(rmeta.rid, 1)
        create_chain_table(self._client, cmeta, rmeta)
        rows, names_and_types = self._client.execute(
            f"SELECT * FROM {chain_id(cmeta)};", with_column_types=True
        )
        assert len(rows) == 0
        assert names_and_types == [
            ("_draw_idx", "UInt64"),
            ("scalar", "UInt16"),
            ("1D", "Array(Float32)"),
            ("3D", "Array(Array(Array(Float64)))"),
            ("__stat_accepted", "UInt8"),
        ]
        pass

    def test_create_chain_table_with_undefined_ndim(self, caplog):
        rmeta = make_runmeta(variables=[Variable("v1", "uint8", undefined_ndim=True)])
        with pytest.raises(NotImplementedError, match="Dimensionality of variable 'v1'"):
            create_chain_table(self._client, ChainMeta(rmeta.rid, 0), rmeta)

        rmeta = make_runmeta(sample_stats=[Variable("s1", "bool", undefined_ndim=True)])
        with caplog.at_level(logging.WARNING):
            create_chain_table(self._client, ChainMeta(rmeta.rid, 0), rmeta)
        assert "Dimensionality of sample stat 's1'" in caplog.records[0].message
        assert "Assuming ndim=0" in caplog.records[0].message
        pass

    def test_insert_draw(self):
        run, chains = fully_initialized(
            self.backend,
            make_runmeta(
                variables=[
                    Variable("v1", "uint16", []),
                    Variable("v2", "float32", list((3,))),
                    Variable("v3", "float64", [2, 5, 6]),
                ],
            ),
        )
        draw = {
            "v1": 12,
            "v2": numpy.array([0.5, -2, 1.4], dtype="float32"),
            "v3": numpy.random.uniform(size=(2, 5, 6)).astype("float64"),
        }
        chain = chains[0]
        chain.append(draw)
        assert len(chain._insert_queue) == 1
        chain._commit()
        assert len(chain._insert_queue) == 0
        rows = self._client.execute(f"SELECT _draw_idx,v1,v2,v3 FROM {chain.cid};")
        assert len(rows) == 1
        idx, v1, v2, v3 = rows[0]
        assert idx == 0
        assert v1 == 12
        numpy.testing.assert_array_equal(v2, draw["v2"])
        numpy.testing.assert_array_equal(v3, draw["v3"])
        pass


if __name__ == "__main__":
    tc = TestClickHouseBackend()
    df = tc.run_all_benchmarks()
    print(df)
