"""Ontology Metadata Service: loads schema.yaml, owns DDL for object/edit/lock
tables and the merged_* views (source rows overlaid with latest edits)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import duckdb
import yaml

SCHEMA_PATH = Path(__file__).with_name("schema.yaml")
DEFAULT_DB = Path(os.environ.get("TWIN_DB") or Path(__file__).resolve().parent.parent / "data" / "twin.duckdb")


@dataclass(frozen=True)
class ObjectType:
    name: str
    pk: tuple[str, ...]
    properties: dict[str, str]

    @property
    def table(self) -> str:
        return f"obj_{self.name}"

    @property
    def view(self) -> str:
        return f"merged_{self.name}"

    def pk_expr(self, alias: str = "") -> str:
        p = f"{alias}." if alias else ""
        return "concat_ws('|', " + ", ".join(f"CAST({p}{c} AS VARCHAR)" for c in self.pk) + ")"


@dataclass(frozen=True)
class LinkType:
    name: str
    src: str
    dst: str
    via: dict[str, str]
    join: str | None = None


@dataclass(frozen=True)
class ActionType:
    name: str
    object_type: str
    editable: tuple[str, ...] = ()
    creates: bool = False


@dataclass(frozen=True)
class Ontology:
    objects: dict[str, ObjectType]
    links: dict[str, LinkType]
    actions: dict[str, ActionType]

    def editable_columns(self, type_name: str) -> list[str]:
        cols: list[str] = []
        for a in self.actions.values():
            if a.object_type == type_name:
                cols += [c for c in a.editable if c not in cols]
        return cols

    def links_between(self, a: str, b: str) -> list[LinkType]:
        return [l for l in self.links.values() if {l.src, l.dst} == {a, b}]


@lru_cache(maxsize=1)
def load(path: Path = SCHEMA_PATH) -> Ontology:
    raw = yaml.safe_load(path.read_text())
    objects = {
        n: ObjectType(n, tuple(d["pk"]), dict(d["properties"]))
        for n, d in raw["object_types"].items()
    }
    links = {
        n: LinkType(n, d["from"], d["to"], dict(d["via"]), d.get("join"))
        for n, d in raw["link_types"].items()
    }
    actions = {
        n: ActionType(n, d["object_type"], tuple(d.get("editable", ())), bool(d.get("creates")))
        for n, d in raw["action_types"].items()
    }
    for l in links.values():
        for t in (l.src, l.dst) + ((l.join,) if l.join else ()):
            if t not in objects:
                raise ValueError(f"link {l.name} references unknown type {t}")
    for a in actions.values():
        ot = objects[a.object_type]
        bad = [c for c in a.editable if c not in ot.properties]
        if bad:
            raise ValueError(f"action {a.name} edits unknown columns {bad}")
    return Ontology(objects, links, actions)


def connect(path: Path | str = DEFAULT_DB, init_schema: bool = True) -> duckdb.DuckDBPyConnection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    if init_schema:
        init(con)
    return con


def init(con: duckdb.DuckDBPyConnection) -> None:
    ont = load()
    con.execute(
        """CREATE TABLE IF NOT EXISTS edits (
            edit_id BIGINT, action VARCHAR, object_type VARCHAR, pk VARCHAR,
            col VARCHAR, value VARCHAR, actor VARCHAR, ts TIMESTAMP)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS locks (
            model_version VARCHAR PRIMARY KEY, inputs_hash VARCHAR,
            outputs_hash VARCHAR, locked_at TIMESTAMP)"""
    )
    con.execute(
        """CREATE TABLE IF NOT EXISTS ingest_log (
            object_type VARCHAR, mode VARCHAR, n_rows BIGINT, source VARCHAR, ts TIMESTAMP)"""
    )
    for ot in ont.objects.values():
        cols = ", ".join(f"{c} {t}" for c, t in ot.properties.items())
        con.execute(f"CREATE TABLE IF NOT EXISTS {ot.table} ({cols})")
        _create_merged_view(con, ont, ot)


def _create_merged_view(con, ont: Ontology, ot: ObjectType) -> None:
    editable = ont.editable_columns(ot.name)
    if not editable:
        con.execute(f"CREATE OR REPLACE VIEW {ot.view} AS SELECT * FROM {ot.table}")
        return
    joins, replaces = [], []
    for c in editable:
        a = f"e_{c}"
        joins.append(
            f"LEFT JOIN (SELECT pk, arg_max(value, edit_id) AS v FROM edits "
            f"WHERE object_type = '{ot.name}' AND col = '{c}' GROUP BY pk) {a} "
            f"ON {a}.pk = {ot.pk_expr('b')}"
        )
        replaces.append(
            f"CASE WHEN {a}.pk IS NULL THEN b.{c} "
            f"ELSE TRY_CAST({a}.v AS {ot.properties[c]}) END AS {c}"
        )
    con.execute(
        f"CREATE OR REPLACE VIEW {ot.view} AS SELECT b.* REPLACE ({', '.join(replaces)}) "
        f"FROM {ot.table} b " + " ".join(joins)
    )


def one(con, sql: str, params: list | None = None):
    """First row or None. DuckDB 2.0 requires results be fully consumed before reuse."""
    rows = con.execute(sql, params or []).fetchall()
    return rows[0] if rows else None


import warnings  # noqa: E402

warnings.filterwarnings("ignore", message="Arrow output on a DB-API result")
