"""Funnel: the only write path into object storage.

write_batch  - bulk loads from pipelines (upsert or replace)
write_stream - incremental live writes (upsert, logged as stream)
apply_action - validated, audited edits; merged_* views reflect them at once
write_raw    - land a source dataset as raw_<name> with lineage columns
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable

import duckdb
import polars as pl

from ontology import oms


class OntologyError(ValueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _frame(rows: pl.DataFrame | Iterable[dict[str, Any]]) -> pl.DataFrame:
    return rows if isinstance(rows, pl.DataFrame) else pl.DataFrame(list(rows), infer_schema_length=None)


def _validate(ot: oms.ObjectType, df: pl.DataFrame) -> pl.DataFrame:
    unknown = [c for c in df.columns if c not in ot.properties]
    if unknown:
        raise OntologyError(f"{ot.name}: unknown properties {unknown}")
    missing_pk = [c for c in ot.pk if c not in df.columns]
    if missing_pk:
        raise OntologyError(f"{ot.name}: missing primary key columns {missing_pk}")
    for c in ot.pk:
        if df[c].null_count():
            raise OntologyError(f"{ot.name}: null primary key in {c}")
    if df.select(ot.pk).is_duplicated().any():
        raise OntologyError(f"{ot.name}: duplicate primary keys in batch")
    for c in ot.properties:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).alias(c))
    return df.select(list(ot.properties))


def _check_locks(con, ot: oms.ObjectType, df: pl.DataFrame) -> None:
    if ot.name != "Prediction" or df.is_empty():
        return
    versions = df["model_version"].unique().to_list()
    locked = con.execute(
        "SELECT model_version FROM locks WHERE model_version IN (SELECT unnest(?))", [versions]
    ).fetchall()
    if locked:
        raise OntologyError(f"model_version locked, writes refused: {[r[0] for r in locked]}")


def _insert(con, ot: oms.ObjectType, df: pl.DataFrame, mode: str, source: str) -> int:
    df = _validate(ot, df)
    _check_locks(con, ot, df)
    cols = ", ".join(ot.properties)
    casts = ", ".join(f"CAST({c} AS {t}) AS {c}" for c, t in ot.properties.items())
    con.register("_funnel_in", df.to_arrow())
    try:
        con.execute("BEGIN")
        if mode == "replace":
            con.execute(f"DELETE FROM {ot.table}")
        else:
            con.execute(
                f"DELETE FROM {ot.table} WHERE {ot.pk_expr()} IN "
                f"(SELECT {ot.pk_expr()} FROM _funnel_in)"
            )
        con.execute(f"INSERT INTO {ot.table} ({cols}) SELECT {casts} FROM _funnel_in")
        con.execute("INSERT INTO ingest_log VALUES (?, ?, ?, ?, ?)", [ot.name, mode, df.height, source, _now()])
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("_funnel_in")
    return df.height


def write_batch(con, type_name: str, rows, mode: str = "upsert", source: str = "batch") -> int:
    if mode not in ("upsert", "replace"):
        raise OntologyError(f"bad mode {mode}")
    return _insert(con, oms.load().objects[type_name], _frame(rows), mode, source)


def write_stream(con, type_name: str, rows, source: str = "stream") -> int:
    return _insert(con, oms.load().objects[type_name], _frame(rows), "stream", source)


def _next_edit_id(con) -> int:
    return oms.one(con, "SELECT coalesce(max(edit_id), 0) + 1 FROM edits")[0]


def apply_action(con, action: str, *, pk: dict | None = None, patch: dict | None = None,
                 row: dict | list[dict] | None = None, actor: str = "system") -> None:
    """Edit actions take pk+patch (only schema-editable columns). Create actions take row(s)."""
    ont = oms.load()
    if action not in ont.actions:
        raise OntologyError(f"unknown action {action}")
    at = ont.actions[action]
    ot = ont.objects[at.object_type]
    ts = _now()
    if at.creates:
        rows = row if isinstance(row, list) else [row]
        if not rows or rows[0] is None:
            raise OntologyError(f"{action} requires row")
        _insert(con, ot, _frame(rows), "action", action)
        eid = _next_edit_id(con)
        con.executemany(
            "INSERT INTO edits VALUES (?, ?, ?, ?, '*', ?, ?, ?)",
            [[eid + i, action, ot.name, "|".join(str(r[c]) for c in ot.pk),
              json.dumps(r, default=str), actor, ts] for i, r in enumerate(rows)],
        )
        return
    if not pk or not patch:
        raise OntologyError(f"{action} requires pk and patch")
    if set(pk) != set(ot.pk):
        raise OntologyError(f"{action}: pk must be {ot.pk}")
    bad = [c for c in patch if c not in at.editable]
    if bad:
        raise OntologyError(f"{action} may not edit {bad} (editable: {at.editable})")
    key = "|".join(str(pk[c]) for c in ot.pk)
    exists = oms.one(con, f"SELECT count(*) FROM {ot.table} WHERE {ot.pk_expr()} = ?", [key])[0]
    if not exists:
        raise OntologyError(f"{ot.name} {key} does not exist")
    eid = _next_edit_id(con)
    con.executemany(
        "INSERT INTO edits VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [[eid + i, action, ot.name, key, c, None if v is None else str(v), actor, ts]
         for i, (c, v) in enumerate(patch.items())],
    )


def write_raw(con, name: str, df: pl.DataFrame, source: str, source_url: str = "") -> int:
    df = df.with_columns(
        pl.lit(source).alias("_source"),
        pl.lit(source_url).alias("_source_url"),
        pl.lit(_now()).alias("_fetched_at"),
    )
    con.register("_raw_in", df.to_arrow())
    try:
        con.execute(f"CREATE OR REPLACE TABLE raw_{name} AS SELECT * FROM _raw_in")
    finally:
        con.unregister("_raw_in")
    return df.height


def lock_version(con: duckdb.DuckDBPyConnection, model_version: str, inputs_hash: str, outputs_hash: str) -> None:
    if oms.one(con, "SELECT 1 FROM locks WHERE model_version = ?", [model_version]):
        raise OntologyError(f"{model_version} already locked")
    con.execute("INSERT INTO locks VALUES (?, ?, ?, ?)", [model_version, inputs_hash, outputs_hash, _now()])
