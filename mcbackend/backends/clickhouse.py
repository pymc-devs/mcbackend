"""
This module implements a backend for sample storage in ClickHouse.
"""
import datetime
import inspect
import logging
import time
from typing import Dict, Sequence

import clickhouse_driver
import numpy
import pandas

from ..core import Backend, Chain, ChainMeta, Run, RunMeta

_log = logging.getLogger(__file__)


CLICKHOUSE_DTYPES = {
    "uint8": "UInt8",
    "uint16": "UInt16",
    "uint32": "UInt32",
    "uint64": "UInt64",
    "int8": "Int8",
    "int16": "Int16",
    "int32": "Int32",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
}


def create_runs_table(client: clickhouse_driver.Client):
    query = """
        CREATE TABLE IF NOT EXISTS runs (
            created_at DateTime64(9, "UTC"),
            run_id String,
            var_names Array(String),
            var_dtypes Array(String),
            var_shapes Array(Array(UInt64)),
            var_is_free Array(UInt8)
        )
        ENGINE MergeTree()
        PRIMARY KEY (run_id)
        PARTITION BY (created_at)
        ORDER BY (run_id);
    """
    return client.execute(query)


def create_chain_table(client: clickhouse_driver.Client, meta: ChainMeta):
    # Check that it does not already exist
    if client.execute(f"SHOW TABLES LIKE '{meta.chain_id}';"):
        raise Exception("A table for {meta.chain_id} already exists.")

    # Fetch column metadata from the run
    rows = client.execute(
        f"SELECT var_names, var_dtypes, var_shapes FROM runs WHERE run_id='{meta.run_id}';"
    )
    if len(rows) != 1:
        raise Exception(f"Unexpected number of {len(rows)} rows found for run_id {meta.run_id}.")
    var_names, var_dtypes, var_shapes = rows[0]

    # Create a table with columns corresponding to the model variables
    columns = []
    for name, dtype, shape in zip(var_names, var_dtypes, var_shapes):
        cdt = CLICKHOUSE_DTYPES[dtype]
        ndim = len(shape)
        for _ in range(ndim):
            cdt = f"Array({cdt})"
        columns.append(f"`{name}` {cdt}")
    columns = ",\n        ".join(columns)

    query = f"""
        CREATE TABLE {meta.chain_id}
        (
            `_draw_idx` UInt64,
            {columns}
        )
        ENGINE TinyLog();
    """
    return client.execute(query)


class ClickHouseChain(Chain):
    """Represents an MCMC chain stored in ClickHouse."""

    def __init__(
        self,
        meta: ChainMeta,
        *,
        rmeta: RunMeta,
        client: clickhouse_driver.Client,
        insert_interval: int = 1,
        draw_idx: int = 0,
    ):
        self._draw_idx = draw_idx
        self._client = client
        self._rmeta = rmeta
        # The following attributes belong to the batched insert mechanism.
        # Inserting in batches is much faster than inserting single rows.
        self._insert_query = None
        self._insert_queue = []
        self._last_insert = time.time()
        self._insert_interval = insert_interval
        super().__init__(meta)

    def add_draw(self, draw: Dict[str, numpy.ndarray]):
        params = {"_draw_idx": self._draw_idx, **draw}
        self._draw_idx += 1
        if not self._insert_query:
            chain_id = self.meta.chain_id
            names = ", ".join(params.keys())
            self._insert_query = f"INSERT INTO {chain_id} ({names}) VALUES"
        self._insert_queue.append(params)

        if time.time() - self._last_insert > self._insert_interval:
            self._commit()
        return

    def _commit(self):
        if not self._insert_queue:
            return
        params = self._insert_queue
        self._insert_queue = []
        self._last_insert = time.time()
        self._client.execute(self._insert_query, params)
        return

    def __del__(self):
        self._commit()
        return

    def get_draw(
        self,
        draw_idx: int,
        var_names: Sequence[str],
    ) -> Dict[str, numpy.ndarray]:
        self._commit()
        chain_id = self.meta.chain_id
        names = ",".join(var_names)
        data = self._client.execute(f"SELECT ({names}) FROM {chain_id} WHERE _draw_idx={draw_idx};")
        if not data:
            raise Exception(f"No record found for draw index {draw_idx}.")
        result = dict(zip(var_names, data[0][0]))
        return result

    def get_variable(  # pylint: disable=W0221
        self,
        var_name: int,
        *,
        burn: int = 0,
    ) -> numpy.ndarray:
        self._commit()
        # What do we expect?
        v = self._rmeta.var_names.index(var_name)
        dtype = self._rmeta.var_dtypes[v]
        rigid = all(s != 0 for s in self._rmeta.var_shapes[v])

        # Now fetch it
        chain_id = self.meta.chain_id
        data = self._client.execute(
            f"SELECT (`{var_name}`) FROM {chain_id} WHERE _draw_idx>={burn};"
        )
        draws = len(data)

        # Safety checks
        if not draws:
            raise Exception(f"No draws in chain {chain_id}.")

        # The unpacking must also account for non-rigid shapes
        if rigid:
            buffer = numpy.empty((draws, *self._rmeta.var_shapes[v]), dtype)
        else:
            buffer = numpy.repeat(None, draws)
        for d, (vals,) in enumerate(data):
            buffer[d] = numpy.asarray(vals, dtype)
        return buffer


class ClickHouseRun(Run):
    """Represents an MCMC run stored in ClickHouse."""

    def __init__(self, meta: RunMeta, *, client: clickhouse_driver.Client) -> None:
        self._client = client
        super().__init__(meta)

    def init_chain(self, chain_number: int) -> ClickHouseChain:
        cmeta = ChainMeta(self.meta.run_id, chain_number)
        create_chain_table(self._client, cmeta)
        return ClickHouseChain(cmeta, rmeta=self.meta, client=self._client)

    def get_chains(self) -> Sequence[ClickHouseChain]:
        chains = []
        for (cid,) in self._client.execute(f"SHOW TABLES LIKE '{self.meta.run_id}%'"):
            cm = ChainMeta(self.meta.run_id, int(cid.split("_")[-1]))
            chains.append(ClickHouseChain(cm, rmeta=self.meta, client=self._client))
        return chains


class ClickHouseBackend(Backend):
    """A backend to store samples in a ClickHouse database."""

    def __init__(self, client: clickhouse_driver.Client) -> None:
        self._client = client
        create_runs_table(client)
        super().__init__()

    def init_run(self, meta: RunMeta) -> ClickHouseRun:
        existing = self._client.execute(f"SELECT run_id FROM runs WHERE run_id='{meta.run_id}';")
        if existing:
            _log.warning("A run with id %s is already present in the database.", meta.run_id)
        else:
            query = "INSERT INTO runs (created_at, run_id, var_names, var_dtypes, var_shapes, var_is_free) VALUES"
            params = dict(
                created_at=meta.created_at,
                run_id=meta.run_id,
                var_names=meta.var_names,
                var_dtypes=meta.var_dtypes,
                var_shapes=meta.var_shapes,
                var_is_free=list(map(int, meta.var_is_free)),
            )
            self._client.execute(query, [params])
        return ClickHouseRun(meta, client=self._client)

    def get_runs(self) -> pandas.DataFrame:
        df = self._client.query_dataframe("SELECT * FROM runs ORDER BY created_at;")
        df["created_at"] = [ca.replace(tzinfo=datetime.timezone.utc) for ca in df["created_at"]]
        df["var_is_free"] = [list(map(bool, vif)) for vif in df["var_is_free"]]
        return df.set_index("run_id")

    def get_run(self, run_id: str) -> ClickHouseRun:
        keys = tuple(inspect.signature(RunMeta.__init__).parameters)[1:]
        names = ",".join(keys)
        rows = self._client.execute(
            f"SELECT {names} FROM runs WHERE run_id=%(run_id)s;",
            {"run_id": run_id},
        )
        if len(rows) != 1:
            raise Exception(f"Unexpected number of {len(rows)} results for run_id='{run_id}'.")
        kwargs = dict(zip(keys, rows[0]))
        kwargs["created_at"] = kwargs["created_at"].replace(tzinfo=datetime.timezone.utc)
        meta = RunMeta(**kwargs)
        return ClickHouseRun(meta, client=self._client)
