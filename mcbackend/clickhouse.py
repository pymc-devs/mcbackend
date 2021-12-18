"""
This module implements a backend for sample storage in ClickHouse.
"""
import inspect
from typing import Dict, Sequence

import clickhouse_driver
import numpy
import pandas

from .core import BackendBase, ChainMeta, RunMeta

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


class ClickHouseBackend(BackendBase):
    """A backend to store samples in a ClickHouse database."""

    def __init__(self, client: clickhouse_driver.Client) -> None:
        self.client = client
        super().__init__()

    def init_backend(self):
        query = """
            CREATE TABLE IF NOT EXISTS runs (
                run_id String,
                var_names Array(String),
                var_dtypes Array(String),
                var_shapes Array(Array(UInt64)),
                var_is_free Array(UInt8)
            )
            ENGINE MergeTree()
            PRIMARY KEY (run_id)
            PARTITION BY (run_id)
            ORDER BY (run_id);
        """
        return self.client.execute(query)

    def init_run(self, run_meta: RunMeta):
        existing = self.client.execute(f"SELECT run_id FROM runs WHERE run_id='{run_meta.run_id}';")
        if existing:
            # See https://github.com/michaelosthege/mcbackend/issues/6
            return

        query = "INSERT INTO runs (run_id, var_names, var_dtypes, var_shapes, var_is_free) VALUES"
        params = dict(
            run_id=run_meta.run_id,
            var_names=run_meta.var_names,
            var_dtypes=run_meta.var_dtypes,
            var_shapes=run_meta.var_shapes,
            var_is_free=list(map(int, run_meta.var_is_free)),
        )
        self.client.execute(query, [params])
        return

    def init_chain(self, chain_meta: ChainMeta):
        rows = self.client.execute(
            f"SELECT var_names, var_dtypes, var_shapes FROM runs WHERE run_id='{chain_meta.run_id}';"
        )
        if len(rows) != 1:
            raise Exception(
                f"Unexpected number of {len(rows)} rows found for run_id {chain_meta.run_id}."
            )
        var_names, var_dtypes, var_shapes = rows[0]

        columns = []
        for name, dtype, shape in zip(var_names, var_dtypes, var_shapes):
            cdt = CLICKHOUSE_DTYPES[dtype]
            ndim = len(shape)
            for _ in range(ndim):
                cdt = f"Array({cdt})"
            columns.append(f"`{name}` {cdt}")
        columns = ",\n            ".join(columns)

        query = f"""
            CREATE TABLE {chain_meta.chain_id}
            (
                `_draw_idx` UInt64,
                {columns}
            )
            ENGINE TinyLog();
        """
        return self.client.execute(query)

    def add_draw(
        self,
        chain_id: str,
        draw_idx: int,
        draw: Dict[str, numpy.ndarray],
    ):
        params = {"_draw_idx": draw_idx, **draw}
        names = ", ".join(params.keys())
        query = f"INSERT INTO {chain_id} ({names}) VALUES"
        self.client.execute(query, [params])

    def get_draw(
        self,
        chain_id: str,
        draw_idx: int,
        var_names: Sequence[str],
    ) -> Dict[str, numpy.ndarray]:
        names = ",".join(var_names)
        data = self.client.execute(f"SELECT ({names}) FROM {chain_id} WHERE _draw_idx={draw_idx};")
        if not data:
            raise Exception(f"No record found for draw index {draw_idx}.")
        result = dict(zip(var_names, data[0][0]))
        return result

    def get_variable(  # pylint: disable=W0221
        self,
        chain_id: str,
        var_name: int,
        *,
        burn: int = 0,
    ) -> Dict[str, numpy.ndarray]:
        data = self.client.execute(
            f"SELECT (`{var_name}`) FROM {chain_id} WHERE _draw_idx>={burn};"
        )
        draws = len(data)
        if not draws:
            raise Exception(f"No draws in chain {chain_id}.")
        return numpy.array([vals for vals, in data])

    def get_runs(self) -> pandas.DataFrame:
        df = self.client.query_dataframe("SELECT * FROM runs;")
        df["var_is_free"] = [list(map(bool, vif)) for vif in df["var_is_free"]]
        return df.set_index("run_id")

    def get_run(self, run_id: str) -> RunMeta:
        keys = tuple(inspect.signature(RunMeta.__init__).parameters)[1:]
        names = ",".join(keys)
        rows = self.client.execute(
            f"SELECT {names} FROM runs WHERE run_id=%(run_id)s;",
            {"run_id": run_id},
        )
        if len(rows) != 1:
            raise Exception(f"Unexpected number of {len(rows)} results for run_id='{run_id}'.")
        meta = RunMeta(**dict(zip(keys, rows[0])))
        return meta
