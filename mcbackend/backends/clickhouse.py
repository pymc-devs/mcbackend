"""
This module implements a backend for sample storage in ClickHouse.
"""
import base64
import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import clickhouse_driver
import numpy
import pandas

from mcbackend.meta import ChainMeta, RunMeta, Variable

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
    "bool": "UInt8",
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


def column_spec_for(var: Variable, is_stat: bool = False):
    cdt = CLICKHOUSE_DTYPES[var.dtype]
    ndim = len(var.shape)
    for _ in range(ndim):
        cdt = f"Array({cdt})"
    if not is_stat:
        return f"`{var.name}` {cdt}"
    return f"`__stat_{var.name}` {cdt}"


def create_chain_table(client: clickhouse_driver.Client, meta: ChainMeta, rmeta: RunMeta):
    # Check that it does not already exist
    cid = chain_id(meta)
    if client.execute(f"SHOW TABLES LIKE '{cid}';"):
        raise Exception(f"A table for {cid} already exists.")

    # Create a table with columns corresponding to the model variables
    columns = []
    for var in rmeta.variables:
        if var.undefined_ndim:
            raise NotImplementedError(
                f"Dimensionality of variable '{var.name}' is undefined."
                " This is not supported by the ClickHouseBackend."
            )
        columns.append(column_spec_for(var))
    for var in rmeta.sample_stats:
        if var.undefined_ndim:
            _log.warning(
                "Dimensionality of sample stat '%s' is undefined. Assuming ndim=0.", var.name
            )
        columns.append(column_spec_for(var, is_stat=True))
    assert len(set(columns)) == len(columns), columns
    cols = ",\n            ".join(columns)

    query = f"""
        CREATE TABLE {cid}
        (
            `_draw_idx` UInt64,
            {cols}
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
        insert_every: int = 500,
        draw_idx: int = 0,
    ):
        self._draw_idx = draw_idx
        self._client = client
        # The following attributes belong to the batched insert mechanism.
        # Inserting in batches is much faster than inserting single rows.
        self._insert_query: str = ""
        self._insert_queue: List[Dict[str, Any]] = []
        self._last_insert = time.time()
        self._insert_interval = insert_interval
        self._insert_every = insert_every
        super().__init__(cmeta, rmeta)

    def append(
        self, draw: Dict[str, numpy.ndarray], stats: Optional[Dict[str, numpy.ndarray]] = None
    ):
        stat = {f"__stat_{sname}": svals for sname, svals in (stats or {}).items()}
        params: Dict[str, Any] = {"_draw_idx": self._draw_idx, **draw, **stat}
        self._draw_idx += 1
        if not self._insert_query:
            names = ", ".join(params.keys())
            self._insert_query = f"INSERT INTO {self.cid} ({names}) VALUES"
        self._insert_queue.append(params)

        if (
            len(self._insert_queue) >= self._insert_every
            or time.time() - self._last_insert > self._insert_interval
        ):
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

    def __len__(self) -> int:
        self._commit()
        result = self._client.execute(f"SELECT count() FROM {self.cid}")
        return result[0][0]

    def __del__(self):
        self._commit()
        return

    def _get_row_at(
        self,
        idx: int,
        var_names: Sequence[str],
    ) -> Dict[str, numpy.ndarray]:
        self._commit()
        names = ",".join(var_names)
        data = self._client.execute(f"SELECT ({names}) FROM {self.cid} WHERE _draw_idx={idx};")
        if not data:
            raise Exception(f"No record found for draw index {idx}.")
        result = dict(zip(var_names, data[0][0]))
        return result

    def _get_rows(  # pylint: disable=W0221
        self,
        var_name: str,
        nshape: Optional[Sequence[int]],
        dtype: str,
        *,
        burn: int = 0,
    ) -> numpy.ndarray:
        self._commit()
        data = self._client.execute(
            f"SELECT (`{var_name}`) FROM {self.cid} WHERE _draw_idx>={burn};"
        )
        draws = len(data)

        # Safety checks
        if not draws:
            raise Exception(f"No draws in chain {self.cid}.")

        # The unpacking must also account for non-rigid shapes
        if is_rigid(nshape):
            assert nshape is not None
            buffer = numpy.empty((draws, *nshape), dtype)
        else:
            buffer = [None] * draws
        for d, (vals,) in enumerate(data):
            buffer[d] = numpy.asarray(vals, dtype)

        # If the values are "ragged" (have different shapes) the
        # numpy array must be dtype=object.
        if len({v.shape for v in buffer}) > 1:
            # To circumvent NumPy issue #19113
            arr = numpy.empty(draws, dtype=object)
            arr[:] = buffer
            return arr
        # Otherwise (identical shapes) we can collapse into one ndarray
        return numpy.asarray(buffer, dtype=dtype)

    def get_draws(self, var_name: str) -> numpy.ndarray:
        var = self.variables[var_name]
        nshape = var.shape if not var.undefined_ndim else None
        return self._get_rows(var_name, nshape, var.dtype)

    def get_draws_at(self, idx: int, var_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        return self._get_row_at(idx, var_names)

    def get_stats(self, stat_name: str) -> numpy.ndarray:
        var = self.sample_stats[stat_name]
        nshape = var.shape if not var.undefined_ndim else None
        return self._get_rows(f"__stat_{stat_name}", nshape, var.dtype)

    def get_stats_at(self, idx: int, stat_names: Sequence[str]) -> Dict[str, numpy.ndarray]:
        stats = self._get_row_at(idx, [f"__stat_{sname}" for sname in stat_names])
        return {sname[7:]: vals for sname, vals in stats.items()}


class ClickHouseRun(Run):
    """Represents an MCMC run stored in ClickHouse."""

    def __init__(
        self,
        meta: RunMeta,
        *,
        created_at: datetime = None,
        client_fn: Callable[[], clickhouse_driver.Client],
    ) -> None:
        self._client_fn = client_fn
        self._client = client_fn()
        if created_at is None:
            created_at = datetime.now().astimezone(timezone.utc)
        self.created_at = created_at
        # We need handles on the chains to commit their batched inserts
        # before returning them to callers of `.get_chains()`.
        self._chains: List[ClickHouseChain] = []
        super().__init__(meta)

    def init_chain(self, chain_number: int) -> ClickHouseChain:
        cmeta = ChainMeta(self.meta.rid, chain_number)
        create_chain_table(self._client, cmeta, self.meta)
        chain = ClickHouseChain(cmeta, self.meta, client=self._client_fn())
        self._chains.append(chain)
        return chain

    def get_chains(self) -> Tuple[ClickHouseChain, ...]:
        # Preferably return existing handles on chains that might have
        # uncommitted inserts pending.
        if self._chains:
            return tuple(self._chains)

        # Otherwise fetch existing chains from the DB.
        chains = []
        for (cid,) in self._client.execute(f"SHOW TABLES LIKE '{self.meta.rid}%'"):
            cm = ChainMeta(self.meta.rid, int(cid.split("_")[-1]))
            chains.append(ClickHouseChain(cm, self.meta, client=self._client_fn()))
        return tuple(chains)


class ClickHouseBackend(Backend):
    """A backend to store samples in a ClickHouse database."""

    def __init__(
        self,
        client: clickhouse_driver.Client = None,
        client_fn: Callable[[], clickhouse_driver.Client] = None,
    ):
        """Create a ClickHouse backend around a database client.

        Parameters
        ----------
        client : clickhouse_driver.Client
            One client to use for all runs and chains.
        client_fn : callable
            A function to create database clients.
            Use this in multithreading scenarios to get higher insert performance.
        """
        if client is None and client_fn is None:
            raise ValueError("Either a `client` or a `client_fn` must be provided.")

        if client_fn is None:
            client_fn = lambda: client
        if client is None:
            client = client_fn()

        self._client_fn: Callable[[], clickhouse_driver.Client] = client_fn
        self._client: clickhouse_driver.Client = client

        create_runs_table(self._client)
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
                proto=base64.encodebytes(bytes(meta)).decode("ascii"),
            )
            self._client.execute(query, [params])
        return ClickHouseRun(meta, client_fn=self._client_fn, created_at=created_at)

    def get_runs(self) -> pandas.DataFrame:
        df = self._client.query_dataframe(
            "SELECT created_at,rid,proto FROM runs ORDER BY created_at;"
        )
        if df.empty:
            df["created_at,rid,proto".split(",")] = None
        df["created_at"] = [ca.replace(tzinfo=timezone.utc) for ca in df["created_at"]]
        df["proto"] = [
            RunMeta().parse(base64.decodebytes(proto.encode("ascii"))) for proto in df.proto
        ]
        return df.set_index("rid")

    def get_run(self, rid: str) -> ClickHouseRun:
        rows = self._client.execute(
            "SELECT rid,created_at,proto FROM runs WHERE rid=%(rid)s;",
            {"rid": rid},
        )
        if len(rows) != 1:
            raise Exception(f"Unexpected number of {len(rows)} results for rid='{rid}'.")
        data = base64.decodebytes(rows[0][2].encode("ascii"))
        meta = RunMeta().parse(data)
        return ClickHouseRun(
            meta, client_fn=self._client_fn, created_at=rows[0][1].replace(tzinfo=timezone.utc)
        )
