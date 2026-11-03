"""Post-election scoring: locked pregame forecast vs markets at lock, Brier + log loss."""
from __future__ import annotations

import numpy as np
import polars as pl

LOCKED = "pregame-v1-LOCKED"


def score(con, model_version: str = LOCKED, market_tag: str = "lock") -> pl.DataFrame:
    df = pl.from_arrow(con.execute(
        """
        WITH pred AS (
            SELECT race_id, p_dem FROM merged_Prediction WHERE model_version = ? AND scenario_id IS NULL
            QUALIFY row_number() OVER (PARTITION BY race_id ORDER BY ts DESC) = 1),
        mkt AS (
            SELECT m.race_id, avg(s.p_dem) AS p_market, count(DISTINCT m.venue) AS venues
            FROM merged_MarketSnapshot s JOIN merged_Market m USING (market_id)
            WHERE s.tag = ? GROUP BY m.race_id)
        SELECT r.race_id, r.chamber, o.winner_party, pred.p_dem AS p_model, mkt.p_market, mkt.venues
        FROM merged_Outcome o
        JOIN pred USING (race_id)
        LEFT JOIN mkt USING (race_id)
        LEFT JOIN merged_Race r USING (race_id)
        """, [model_version, market_tag]).to_arrow_table())
    if df.is_empty():
        return df
    y = (df["winner_party"] == "D").cast(pl.Float64)
    eps = 1e-6
    df = df.with_columns(y.alias("y"))
    for col in ("p_model", "p_market"):
        p = pl.col(col).clip(eps, 1 - eps)
        df = df.with_columns(
            ((pl.col(col) - pl.col("y")) ** 2).alias(f"brier_{col[2:]}"),
            (-(pl.col("y") * p.log() + (1 - pl.col("y")) * (1 - p).log())).alias(f"logloss_{col[2:]}"),
        )
    return df


def summary(df: pl.DataFrame) -> pl.DataFrame:
    both = df.filter(pl.col("p_market").is_not_null())
    groups = [("all (model)", df), ("head-to-head", both)]
    for ch in sorted(df["chamber"].drop_nulls().unique().to_list()):
        groups.append((f"{ch} head-to-head", both.filter(pl.col("chamber") == ch)))
    rows = []
    for name, g in groups:
        if g.is_empty():
            continue
        rows.append({"set": name, "n": g.height,
                     "brier_model": float(g["brier_model"].mean()),
                     "brier_market": float(g["brier_market"].mean()) if g["brier_market"].null_count() < g.height else np.nan,
                     "logloss_model": float(g["logloss_model"].mean()),
                     "logloss_market": float(g["logloss_market"].mean()) if g["logloss_market"].null_count() < g.height else np.nan})
    return pl.DataFrame(rows)
