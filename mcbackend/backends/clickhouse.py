"""
This module implements a backend for sample storage in ClickHouse.
"""
import logging
import time
from datetime import datetime, timezone
from typing import Dict, Sequence

import clickhouse_driver
import numpy
import pandas

from mcbackend.meta import ChainMeta, RunMeta

from ..core import Backend, Chain, Run, chain_id, is_rigid

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
            rid String,
            proto String
        )
        ENGINE MergeTree()
        PRIMARY KEY (rid)
        PARTITION BY (created_at)
        ORDER BY (rid);
    """
    return client.execute(query)


def create_chain_table(client: clickhouse_driver.Client, meta: ChainMeta, rmeta: RunMeta):
    # Check that it does not already exist
    cid = chain_id(meta)
    if client.execute(f"SHOW TABLES LIKE '{cid}';"):
        raise Exception(f"A table for {cid} already exists.")

    # Create a table with columns corresponding to the model variables
    columns = []
    for var in rmeta.variables:
        cdt = CLICKHOUSE_DTYPES[var.dtype]
        ndim = len(var.shape)
        for _ in range(ndim):
            cdt = f"Array({cdt})"
        columns.append(f"`{var.name}` {cdt}")
    columns = ",\n        ".join(columns)

    query = f"""
        CREATE TABLE {cid}
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
        cmeta: ChainMeta,
        rmeta: RunMeta,
        *,
        client: clickhouse_driver.Client,
        insert_interval: int = 1,
        draw_idx: int = 0,
    ):
        self._draw_idx = draw_idx
        self._client = client
        # The following attributes belong to the batched insert mechanism.
        # Inserting in batches is much faster than inserting single rows.
        self._insert_query = None
        self._insert_queue = []
        self._last_insert = time.time()
        self._insert_interval = insert_interval
        super().__init__(cmeta, rmeta)

    def add_draw(self, draw: Dict[str, numpy.ndarray]):
        params = {"_draw_idx": self._draw_idx, **draw}
        self._draw_idx += 1
        if not self._insert_query:
            names = ", ".join(params.keys())
            self._insert_query = f"INSERT INTO {self.cid} ({names}) VALUES"
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
        names = ",".join(var_names)
        data = self._client.execute(f"SELECT ({names}) FROM {self.cid} WHERE _draw_idx={draw_idx};")
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
        var = self.variables[var_name]
        data = self._client.execute(
            f"SELECT (`{var_name}`) FROM {self.cid} WHERE _draw_idx>={burn};"
        )
        draws = len(data)

        # Safety checks
        if not draws:
            raise Exception(f"No draws in chain {self.cid}.")

        # The unpacking must also account for non-rigid shapes
        if is_rigid(var.shape):
            buffer = numpy.empty((draws, *var.shape), var.dtype)
        else:
            buffer = numpy.repeat(None, draws)
        for d, (vals,) in enumerate(data):
            buffer[d] = numpy.asarray(vals, var.dtype)
        return buffer


class ClickHouseRun(Run):
    """Represents an MCMC run stored in ClickHouse."""

    def __init__(
        self, meta: RunMeta, *, created_at: datetime = None, client: clickhouse_driver.Client
    ) -> None:
        self._client = client
        if created_at is None:
            created_at = datetime.now().astimezone(timezone.utc)
        self.created_at = created_at
        super().__init__(meta)

    def init_chain(self, chain_number: int) -> ClickHouseChain:
        cmeta = ChainMeta(self.meta.rid, chain_number)
        create_chain_table(self._client, cmeta, self.meta)
        return ClickHouseChain(cmeta, self.meta, client=self._client)

    def get_chains(self) -> Sequence[ClickHouseChain]:
        chains = []
        for (cid,) in self._client.execute(f"SHOW TABLES LIKE '{self.meta.rid}%'"):
            cm = ChainMeta(self.meta.rid, int(cid.split("_")[-1]))
            chains.append(ClickHouseChain(cm, self.meta, client=self._client))
        return chains


class ClickHouseBackend(Backend):
    """A backend to store samples in a ClickHouse database."""

    def __init__(self, client: clickhouse_driver.Client) -> None:
        self._client = client
        create_runs_table(client)
        super().__init__()

    def init_run(self, meta: RunMeta) -> ClickHouseRun:
        existing = self._client.execute(f"SELECT rid, created_at FROM runs WHERE rid='{meta.rid}';")
        if existing:
            _log.warning("A run with id %s is already present in the database.", meta.rid)
            created_at = existing[0][1].replace(tzinfo=timezone.utc)
        else:
            created_at = datetime.now().astimezone(timezone.utc)
            query = "INSERT INTO runs (created_at, rid, proto) VALUES"
            params = dict(
                created_at=created_at,
                rid=meta.rid,
                proto=bytes(meta).decode("utf-8"),
            )
            self._client.execute(query, [params])
        return ClickHouseRun(meta, client=self._client, created_at=created_at)

    def get_runs(self) -> pandas.DataFrame:
        df = self._client.query_dataframe(
            "SELECT created_at,rid,proto FROM runs ORDER BY created_at;"
        )
        if df.empty:
            df["created_at,rid,proto".split(",")] = None
        df["created_at"] = [ca.replace(tzinfo=timezone.utc) for ca in df["created_at"]]
        df["proto"] = [RunMeta().parse(proto.encode("utf-8")) for proto in df.proto]
        return df.set_index("rid")

    def get_run(self, rid: str) -> ClickHouseRun:
        rows = self._client.execute(
            "SELECT rid,created_at,proto FROM runs WHERE rid=%(rid)s;",
            {"rid": rid},
        )
        if len(rows) != 1:
            raise Exception(f"Unexpected number of {len(rows)} results for rid='{rid}'.")
        meta = RunMeta().parse(rows[0][2].encode("utf-8"))
        return ClickHouseRun(
            meta, client=self._client, created_at=rows[0][1].replace(tzinfo=timezone.utc)
        )
