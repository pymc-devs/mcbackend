from typing import Sequence, Tuple

import clickhouse_driver
import hagelkorn
import numpy
import pandas
import pytest

from .backends.clickhouse import (
    ClickHouseBackend,
    create_chain_table,
    create_runs_table,
)
from .core import Chain, ChainMeta, Run, RunMeta

try:
    client = clickhouse_driver.Client("localhost")
    client.execute("SHOW DATABASES;")
    HAS_REAL_DB = True
except:
    HAS_REAL_DB = False


@pytest.fixture
def cclient():
    """Gives a ClickHouse Client connected to a temporary database."""
    # Set up a randomly named database for this test execution
    db = "testing_" + hagelkorn.random()
    main = clickhouse_driver.Client("localhost")
    main.execute(f"CREATE DATABASE {db};")
    # Give the test a client that targets the empty database
    client = clickhouse_driver.Client("localhost", database=db)
    yield client
    # Teardown
    client.disconnect()
    main.execute(f"DROP DATABASE {db};")
    main.disconnect()
    return


@pytest.fixture
def cbackend(cclient: clickhouse_driver.Client):
    """Gives a ClickHouse Client connected to a temporary database."""
    backend = ClickHouseBackend(cclient)
    yield backend
    # No teardown needed.
    return


def make_runmeta(**kwargs):
    defaults = dict(
        run_id=hagelkorn.random(),
        var_names=["A", "B"],
        var_dtypes=["int8", "float64"],
        var_shapes=[(3, 4, 5), ()],
        var_is_free=[True, False],
    )
    defaults.update(kwargs)
    return RunMeta(**defaults)


def fully_initialized(
    cbackend: ClickHouseBackend, rmeta: RunMeta, *, nchains: int = 1
) -> Tuple[Run, Sequence[Chain]]:
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
class TestClickHouseBackend:
    def test_test_database(self, cclient: clickhouse_driver.Client):
        assert cclient.execute("SHOW TABLES;") == []
        pass

    def test_create_runs_table(self, cclient: clickhouse_driver.Client):
        create_runs_table(cclient)
        tables = cclient.execute("SHOW TABLES;")
        assert set(tables) == {
            ("runs",),
        }
        pass

    def test_init_run(self, cbackend: ClickHouseBackend):
        meta = make_runmeta(run_id="my_first_run")
        run = cbackend.init_run(meta)
        assert isinstance(run, Run)
        assert len(cbackend._client.execute("SELECT * FROM runs;")) == 1
        runs = cbackend.get_runs()
        assert isinstance(runs, pandas.DataFrame)
        assert runs.index.name == "run_id"
        assert "my_first_run" in runs.index.values
        pass

    def test_get_run(self, cbackend: ClickHouseBackend):
        meta = make_runmeta()
        cbackend.init_run(meta)
        run = cbackend.get_run(meta.run_id)
        assert run.meta.__dict__ == meta.__dict__
        pass

    def test_create_chain_table(self, cbackend: ClickHouseBackend):
        rmeta = make_runmeta(
            var_names=["scalar", "1D", "3D"],
            var_shapes=[(), (3,), (2, 5, 6)],
            var_dtypes=["uint16", "float32", "float64"],
        )
        cbackend.init_run(rmeta)
        cmeta = ChainMeta(rmeta.run_id, 1)
        create_chain_table(cbackend._client, cmeta)
        rows, names_and_types = cbackend._client.execute(
            f"SELECT * FROM {cmeta.chain_id};", with_column_types=True
        )
        assert len(rows) == 0
        assert names_and_types == [
            ("_draw_idx", "UInt64"),
            ("scalar", "UInt16"),
            ("1D", "Array(Float32)"),
            ("3D", "Array(Array(Array(Float64)))"),
        ]
        pass

    def test_insert_draw(self, cbackend: ClickHouseBackend):
        run, chains = fully_initialized(
            cbackend,
            make_runmeta(
                var_names=["v1", "v2", "v3"],
                var_shapes=[(), (3,), (2, 5, 6)],
                var_dtypes=["uint16", "float32", "float64"],
            ),
        )
        draw = {
            "v1": 12,
            "v2": numpy.array([0.5, -2, 1.4], dtype="float32"),
            "v3": numpy.random.uniform(size=(2, 5, 6)).astype("float64"),
        }
        chain = chains[0]
        chain.add_draw(0, draw)
        rows = cbackend._client.execute(f"SELECT _draw_idx,v1,v2,v3 FROM {chain.meta.chain_id};")
        assert len(rows) == 1
        idx, v1, v2, v3 = rows[0]
        assert idx == 0
        assert v1 == 12
        numpy.testing.assert_array_equal(v2, draw["v2"])
        numpy.testing.assert_array_equal(v3, draw["v3"])
        pass

    def test_get_draw(self, cbackend: ClickHouseBackend):
        run, chains = fully_initialized(
            cbackend,
            make_runmeta(
                var_names=["v1", "v2", "v3"],
                var_shapes=[(), (3,), (2, 5, 6)],
                var_dtypes=["uint16", "float32", "float64"],
            ),
        )
        draw = {
            "v1": 12,
            "v2": numpy.array([0.5, -2, 1.4], dtype="float32"),
            "v3": numpy.random.uniform(size=(2, 5, 6)).astype("float64"),
        }
        chain = chains[0]
        chain.add_draw(0, draw)
        actual = chain.get_draw(0, run.meta.var_names)
        assert set(actual) == set(draw)
        for k in run.meta.var_names:
            numpy.testing.assert_array_equal(actual[k], draw[k])
        pass
