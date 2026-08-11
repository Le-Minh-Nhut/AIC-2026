"""Parquet repository helpers with explicit optional-dependency errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable


def write_parquet_records(path: Path, records: Iterable[dict[str, Any]], schema: object) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "Parquet manifest output requires pyarrow. Install the project dependencies first."
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(records), schema=schema)
    pq.write_table(table, path, compression="zstd")


def parquet_schema(fields: list[tuple[str, str]]) -> object:
    try:
        import pyarrow as pa
    except ImportError as error:
        raise RuntimeError(
            "Parquet manifest output requires pyarrow. Install the project dependencies first."
        ) from error
    types = {
        "string": pa.string(),
        "int64": pa.int64(),
        "float64": pa.float64(),
        "bool": pa.bool_(),
    }
    return pa.schema([pa.field(name, types[type_name], nullable=True) for name, type_name in fields])

