import base64
import logging
import os
from datetime import datetime, timezone
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
    column_spec_for,
    create_chain_table,
)
from mcbackend.core import Run, chain_id
from mcbackend.meta import ChainMeta, RunMeta, Variable
from mcbackend.test_utils import CheckBehavior, CheckPerformance, make_runmeta

try:
    DB_HOST = os.environ.get("CLICKHOUSE_HOST", "localhost")
    DB_PASS = os.environ.get("CLICKHOUSE_PASS", "")
    DB_PORT = os.environ.get("CLICKHOUSE_PORT", 9000)
    DB_KWARGS = dict(host=DB_HOST, port=DB_PORT, password=DB_PASS)
    client = clickhouse_driver.Client(**DB_KWARGS)
    client.execute("SHOW DATABASES;")
    HAS_REAL_DB = True
except:
    HAS_REAL_DB = False


def test_column_spec_for():
    assert column_spec_for(Variable("A", "float32", [])) == "`A` Float32"
    assert column_spec_for(Variable("A", "float32", []), is_stat=True) == "`__stat_A` Float32"
    assert column_spec_for(Variable("A", "float32", [2])) == "`A` Array(Float32)"
    assert column_spec_for(Variable("A", "float32", [2, 3])) == "`A` Array(Array(Float32))"
    with pytest.raises(KeyError, match="float16 of 'A'"):
        column_spec_for(Variable("A", "float16", []))
    pass


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
    reason="Integration tests need a ClickHouse server.",
)
class TestClickHouseBackendInitialization:
    """This is separate because ``TestClickHouseBackend.setup_method`` depends on these things."""

    def test_exceptions(self):
        with pytest.raises(ValueError, match="must be provided"):
            ClickHouseBackend()
        pass

    def test_backend_from_client_object(self):
        db = "testing_" + hagelkorn.random()
        _client_main = clickhouse_driver.Client(**DB_KWARGS)
        _client_main.execute(f"CREATE DATABASE {db};")

        try:
            # When created from a client object, all chains share the client
            backend = ClickHouseBackend(client=clickhouse_driver.Client(**DB_KWARGS, database=db))
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
        _client_main = clickhouse_driver.Client(**DB_KWARGS)
        _client_main.execute(f"CREATE DATABASE {db};")

        def client_fn():
            return clickhouse_driver.Client(**DB_KWARGS, database=db)

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
    reason="Integration tests need a ClickHouse server.",
)
class TestClickHouseBackend(CheckBehavior, CheckPerformance):
    cls_backend = ClickHouseBackend
    cls_run = ClickHouseRun
    cls_chain = ClickHouseChain

    def setup_method(self, method):
        """Initializes a fresh database just for this test method."""
        self._db = "testing_" + hagelkorn.random()
        self._client_main = clickhouse_driver.Client(**DB_KWARGS)
        self._client_main.execute(f"CREATE DATABASE {self._db};")
        self._client = clickhouse_driver.Client(**DB_KWARGS, database=self._db)
        self.backend = ClickHouseBackend(
            client_fn=lambda: clickhouse_driver.Client(**DB_KWARGS, database=self._db)
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

        # Illegaly create a duplicate entry
        created_at = datetime.now().astimezone(timezone.utc)
        query = "INSERT INTO runs (created_at, rid, proto) VALUES"
        params = dict(
            created_at=created_at,
            rid=meta.rid,
            proto=base64.encodebytes(bytes(meta)).decode("ascii"),
        )
        self._client.execute(query, [params])
        assert len(self._client.execute("SELECT * FROM runs;")) == 2
        with pytest.raises(Exception, match="Unexpected number of 2 results"):
            self.backend.get_run("my_first_run")
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

        with pytest.raises(Exception, match="already exists"):
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

    def test_get_chains_via_query(self):
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
        newrun = ClickHouseRun(run.meta, client_fn=self.backend._client_fn)
        chains_fetched = newrun.get_chains()
        assert len(chains_fetched) > 0
        assert len(chains_fetched) == len(chains)
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

        # Get empty vector from empty chain
        nodraws = chain._get_rows("v1", [], "uint16")
        assert nodraws.shape == (0,)
        assert nodraws.dtype == numpy.uint16

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

    def test_get_row_at(self):
        run, chains = fully_initialized(
            self.backend,
            make_runmeta(
                variables=[
                    Variable("v1", "uint16", []),
                    Variable("v2", "float32", list((3,))),
                ],
            ),
        )
        chain = chains[0]
        for i in range(10):
            chain.append(dict(v1=i, v2=numpy.array([i, 2, 3])))
        assert len(chain) == 10

        row5 = chain.get_draws_at(5, ["v1", "v2"])
        assert "v1" in row5
        assert "v2" in row5
        assert row5["v1"] == 5
        assert tuple(row5["v2"]) == (5, 2, 3)

        with pytest.raises(Exception, match="No record found for draw"):
            chain._get_row_at(20, var_names=["v1"])

        # Issue #56 was caused by querying just one variable
        assert len(chain._get_row_at(5, var_names=["v1"])) == 1
        pass

    def test_exotic_var_names(self):
        run, chains = fully_initialized(
            self.backend,
            make_runmeta(
                variables=[
                    Variable("v1[a]", "uint16", []),
                ],
            ),
        )
        chain = chains[0]
        for i in range(10):
            chain.append({var.name: i for var in run.meta.variables})
        assert len(chain) == 10

        row2 = chain._get_row_at(2, var_names=["v1[a]"])
        assert "v1[a]" in row2
        pass

    def test_to_inferencedata_equalize_chain_lengths(self, caplog):
        run, chains = fully_initialized(
            self.backend,
            make_runmeta(
                variables=[
                    Variable("A", "uint16", []),
                ],
                sample_stats=[Variable("tune", "bool")],
                data=[],
            ),
            nchains=2,
        )
        # Create chains of uneven lengths:
        # - Chain 0 has 5 tune and 15 draws (length 20)
        # - Chain 1 has 5 tune and 14 draws (length 19)
        # This simulates the situation where chains aren't synchronized.
        ntune = 5

        c0 = chains[0]
        for i in range(0, 20):
            c0.append(dict(A=i), stats=dict(tune=i < ntune))

        c1 = chains[1]
        for i in range(0, 19):
            c1.append(dict(A=i), stats=dict(tune=i < ntune))

        assert len(c0) == 20
        assert len(c1) == 19

        # With equalize=True all chains should have the length of the shortest (here: 7)
        # But the first 3 are tuning, so 4 posterior draws remain.
        with caplog.at_level(logging.WARNING):
            idata_even = run.to_inferencedata(equalize_chain_lengths=True)
        assert "Chains vary in length" in caplog.records[0].message
        assert "Truncating to" in caplog.records[0].message
        assert len(idata_even.posterior.draw) == 14

        # With equalize=False the "draw" dim has the length of the longest chain (here: 8-3 = 5)
        caplog.clear()
        with caplog.at_level(logging.WARNING):
            idata_uneven = run.to_inferencedata(equalize_chain_lengths=False)
        # These are the messed-up chain and draw dimensions!
        assert idata_uneven.posterior.dims["chain"] == 1
        assert idata_uneven.posterior.dims["draw"] == 2
        # The "draws" are actually the chains, but in a weird scalar object-array?!
        # Doing .tolist() seems to be the only way to get our hands on it.
        d1 = idata_uneven.posterior.A.sel(chain=0, draw=0).values.tolist()
        d2 = idata_uneven.posterior.A.sel(chain=0, draw=1).values.tolist()
        numpy.testing.assert_array_equal(d1, list(range(ntune, 20)))
        numpy.testing.assert_array_equal(d2, list(range(ntune, 19)))
        assert "Chains vary in length" in caplog.records[0].message
        assert "see ArviZ issue #2094" in caplog.records[0].message
        pass


if __name__ == "__main__":
    tc = TestClickHouseBackend()
    df = tc.run_all_benchmarks()
    print(df)
