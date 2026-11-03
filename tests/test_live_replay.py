"""Replay 2024 NC/GA with an injected uniform swing; the live model must recover it early."""
import polars as pl
import pytest

from ingest import live_ga, live_nc
from models import live
from ontology import oms

DB = oms.DEFAULT_DB


@pytest.fixture(scope="module")
def base():
    if not DB.exists():
        pytest.skip("needs built twin.duckdb (county baselines)")
    import duckdb
    con = duckdb.connect(str(DB), read_only=True)
    return live.baselines(con)


@pytest.mark.parametrize("mod,st,actual", [(live_nc, "NC", -3.22), (live_ga, "GA", -2.19)])
def test_recovers_injected_swing(base, mod, st, actual):
    county, bm = base
    shift = 0.04  # move 4% of the two-party vote D -> ~ +8 pts margin
    for i, chunk in enumerate(mod.replay_2024(6)):
        rows = (chunk.filter(pl.col("race_id") == f"PRES-{st}").sort("ts")
                .group_by(["race_id", "fips", "vote_type"]).last()
                .with_columns(((pl.col("dem") + pl.col("rep")) * shift).round().cast(pl.Int64).alias("_m"))
                .with_columns((pl.col("dem") + pl.col("_m")), (pl.col("rep") - pl.col("_m")))
                .with_columns((pl.col("dem") + pl.col("rep") + pl.col("other").fill_null(0)).alias("votes")))
        p = live.race_projection(rows, county, bm, live.LiveConfig(turnout_ratio=1.0))
        assert p["projected"] == pytest.approx(actual + 8.0, abs=0.6), (i, p)
        if i == 0:
            assert p["frac"] < 0.85
