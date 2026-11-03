"""Object Set Service: search, filter, link traversal, aggregation over merged_* views.

    ObjectSet(con, "Race").filter(chamber="senate").links("Poll").where("end_date > '2026-10-01'").df()
"""
from __future__ import annotations

from typing import Any

import polars as pl

from ontology import oms


class ObjectSet:
    def __init__(self, con, type_name: str, _sql: str | None = None, _params: list | None = None):
        self.con = con
        self.ont = oms.load()
        if type_name not in self.ont.objects:
            raise KeyError(type_name)
        self.type = self.ont.objects[type_name]
        self._sql = _sql or f"SELECT * FROM {self.type.view}"
        self._params = list(_params or [])

    def _derive(self, sql: str, params: list | None = None, type_name: str | None = None) -> "ObjectSet":
        return ObjectSet(self.con, type_name or self.type.name, sql, self._params + list(params or []))

    def filter(self, **eq: Any) -> "ObjectSet":
        conds, params = [], []
        for c, v in eq.items():
            if c not in self.type.properties:
                raise KeyError(f"{self.type.name}.{c}")
            if isinstance(v, (list, tuple, set)):
                conds.append(f"{c} IN (SELECT unnest(?))")
                params.append(list(v))
            elif v is None:
                conds.append(f"{c} IS NULL")
            else:
                conds.append(f"{c} = ?")
                params.append(v)
        return self._derive(f"SELECT * FROM ({self._sql}) WHERE {' AND '.join(conds)}", params)

    def where(self, predicate: str, *params) -> "ObjectSet":
        return self._derive(f"SELECT * FROM ({self._sql}) WHERE {predicate}", list(params))

    def links(self, target: str, link: str | None = None) -> "ObjectSet":
        """Traverse to linked objects of type `target`."""
        cands = [self.ont.links[link]] if link else self.ont.links_between(self.type.name, target)
        if len(cands) != 1:
            raise ValueError(f"ambiguous or missing link {self.type.name}->{target}: {[c.name for c in cands]}")
        lt = cands[0]
        tgt = self.ont.objects[target]
        mine, theirs = lt.via[self.type.name], lt.via[target]
        if lt.join:
            jt = self.ont.objects[lt.join]
            sql = (
                f"SELECT * FROM {tgt.view} WHERE {theirs} IN (SELECT j.{theirs} FROM {jt.view} j "
                f"WHERE j.{mine} IN (SELECT {mine} FROM ({self._sql})))"
            )
        else:
            sql = f"SELECT * FROM {tgt.view} WHERE {theirs} IN (SELECT {mine} FROM ({self._sql}))"
        return self._derive(sql, type_name=target)

    def aggregate(self, by: list[str] | None = None, **metrics: str) -> pl.DataFrame:
        sel = ", ".join([*(by or []), *(f"{expr} AS {name}" for name, expr in metrics.items())])
        group = f" GROUP BY {', '.join(by)} ORDER BY {', '.join(by)}" if by else ""
        return pl.from_arrow(self.con.execute(f"SELECT {sel} FROM ({self._sql}){group}", self._params).to_arrow_table())

    def count(self) -> int:
        return oms.one(self.con, f"SELECT count(*) FROM ({self._sql})", self._params)[0]

    def df(self, order_by: str | None = None) -> pl.DataFrame:
        sql = self._sql + (f" ORDER BY {order_by}" if order_by else "")
        return pl.from_arrow(self.con.execute(f"SELECT * FROM ({sql})", self._params).to_arrow_table())

    def get(self, **pk: Any) -> dict | None:
        rows = self.filter(**pk).df().to_dicts()
        return rows[0] if rows else None

    def __repr__(self) -> str:
        return f"ObjectSet<{self.type.name}>"
