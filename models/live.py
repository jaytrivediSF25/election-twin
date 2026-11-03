"""Live model: VoteBatches -> county swing vs 2024 (same vote type) -> race projections ->
posterior on the national swing -> rerun the correlated MC for every race.

Only NC/FL/GA have feeds, so races elsewhere move solely through the national posterior.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl

from models import pregame
from models.montecarlo import SimParams
from ontology.objects import ObjectSet

PREGAME_VERSION = "pregame-v1-LOCKED"
LIVE_VERSION = "live-v1"


@dataclass
class LiveConfig:
    turnout_ratio: float = 0.72      # 2026 midterm turnout / 2024 presidential, per county
    shrink_votes: float = 2000.0     # county swing shrinks toward race swing below this many votes
    min_votes: int = 50
    min_frac_for_national: float = 0.10
    sd_floor: float = 0.75


def latest_batches(con) -> pl.DataFrame:
    vb = ObjectSet(con, "VoteBatch").df()
    if vb.is_empty():
        return vb
    return (vb.sort("ts").group_by(["race_id", "fips", "vote_type"]).last()
            .with_columns((pl.col("dem") + pl.col("rep") + pl.col("other").fill_null(0)).alias("votes")))


def baselines(con) -> tuple[pl.DataFrame, pl.DataFrame | None]:
    county = ObjectSet(con, "County").df().select("fips", "state", "pres24_total", "pres24_margin")
    try:
        bm = pl.from_arrow(con.execute(
            "SELECT fips, vote_type, sum(dem)::DOUBLE AS dem, sum(rep)::DOUBLE AS rep, sum(total)::DOUBLE AS total "
            "FROM raw_pres24_by_method GROUP BY ALL").to_arrow_table())
        bm = bm.with_columns((100 * (pl.col("dem") - pl.col("rep")) / pl.col("total")).alias("base_margin_vt"))
    except Exception:
        bm = None
    return county, bm


def race_projection(rows: pl.DataFrame, county: pl.DataFrame, by_method: pl.DataFrame | None,
                    cfg: LiveConfig) -> dict | None:
    """rows: latest batches for one statewide race. Returns projection dict or None."""
    df = rows.filter(pl.col("votes") >= cfg.min_votes).join(county, on="fips", how="left")
    if df.is_empty():
        return None
    if by_method is not None:
        df = df.join(by_method.select("fips", "vote_type", "base_margin_vt"), on=["fips", "vote_type"], how="left")
        df = df.with_columns(pl.coalesce("base_margin_vt", "pres24_margin").alias("base"))
    else:
        df = df.with_columns(pl.col("pres24_margin").alias("base"))
    df = df.with_columns((100 * (pl.col("dem") - pl.col("rep")) / pl.col("votes")).alias("m_obs"))
    df = df.with_columns((pl.col("m_obs") - pl.col("base")).alias("swing")).filter(pl.col("swing").is_not_null())
    if df.is_empty():
        return None
    v = df["votes"].to_numpy().astype(float)
    race_swing = float((v * df["swing"].to_numpy()).sum() / v.sum())

    per_c = (df.group_by("fips").agg(
        pl.col("votes").sum().alias("counted"),
        ((pl.col("dem") - pl.col("rep")).sum()).alias("net"),
        (pl.col("swing") * pl.col("votes")).sum().alias("sw_v"),
    ))
    allc = county.filter(pl.col("state") == df["state"].drop_nulls()[0]).join(per_c, on="fips", how="left").with_columns(
        pl.col("counted").fill_null(0), pl.col("net").fill_null(0), pl.col("sw_v").fill_null(0.0))
    k = cfg.shrink_votes
    allc = allc.with_columns(
        pl.max_horizontal(pl.col("pres24_total") * cfg.turnout_ratio, pl.col("counted") * 1.02).alias("expected"),
        ((pl.col("sw_v") + k * race_swing) / (pl.col("counted") + k)).alias("swing_hat"),
    ).with_columns((pl.col("expected") - pl.col("counted")).clip(0).alias("remaining"))
    net = (allc["net"] + allc["remaining"] * (allc["pres24_margin"] + allc["swing_hat"]) / 100).sum()
    total = float(allc["expected"].sum())
    counted = float(allc["counted"].sum())
    return {"projected": float(100 * net / total), "frac": min(0.99, counted / total) if total else 0.0,
            "race_swing": race_swing, "counted": counted}


def build_live_inputs(con, cfg: LiveConfig | None = None) -> tuple[pl.DataFrame, dict, SimParams]:
    cfg = cfg or LiveConfig()
    try:
        _, inputs, meta, params = pregame.latest_run(con, PREGAME_VERSION)
    except RuntimeError:
        _, inputs, meta, params = pregame.latest_run(con, "pregame-v1")
    p = SimParams(**{k: v for k, v in params.items() if k in SimParams.__dataclass_fields__})
    batches = latest_batches(con)
    county, by_method = baselines(con)
    pre = {r["race_id"]: r for r in inputs.iter_rows(named=True)}

    obs: dict[str, dict] = {}
    if not batches.is_empty():
        statewide = batches.filter(pl.col("race_id").str.starts_with("SEN-") | pl.col("race_id").str.starts_with("GOV-"))
        for rid in statewide["race_id"].unique().to_list():
            if rid not in pre:
                continue
            proj = race_projection(statewide.filter(pl.col("race_id") == rid), county, by_method, cfg)
            if proj:
                proj["dev"] = proj["projected"] - pre[rid]["mean"]
                obs[rid] = proj

    # National posterior from per-state mean deviations.
    by_state: dict[str, list[tuple[float, float]]] = {}
    for rid, o in obs.items():
        if o["frac"] >= cfg.min_frac_for_national:
            by_state.setdefault(pre[rid]["state"], []).append((o["dev"], pre[rid]["sd_race"]))
    prec, wsum, state_dev = 1 / p.sd_national ** 2, 0.0, {}
    for st, items in by_state.items():
        d = float(np.mean([x for x, _ in items]))
        var = p.sd_state ** 2 + float(np.mean([s ** 2 for _, s in items])) / len(items)
        prec += 1 / var
        wsum += d / var
        state_dev[st] = (d, var)
    nat_mu, nat_sd = wsum / prec, math.sqrt(1 / prec)

    rows = []
    for r in inputs.iter_rows(named=True):
        r = dict(r)
        rid, st = r["race_id"], r["state"]
        if rid in obs:
            o, f = obs[rid], obs[rid]["frac"]
            r["mean"] = o["projected"]
            r["sd_race"] = max(cfg.sd_floor, r["sd_race"] * math.sqrt(1 - f))
            r["nat_load"] = r["state_load"] = 1 - f
        elif st in state_dev:
            d, var = state_dev[st]
            shrink = p.sd_state ** 2 / var
            r["mean"] = r["mean"] + nat_mu + shrink * (d - nat_mu)
            r["state_load"] = 1 - shrink
        else:
            r["mean"] = r["mean"] + nat_mu
        rows.append(r)
    live_meta = {**meta, "nat_posterior_mu": nat_mu, "nat_posterior_sd": nat_sd,
                 "observed": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in obs.items()},
                 "state_dev": {k: v[0] for k, v in state_dev.items()}}
    p.sd_national = nat_sd
    return pl.DataFrame(rows, schema=inputs.schema), live_meta, p


def run_once(con, cfg: LiveConfig | None = None) -> tuple[pl.DataFrame, dict]:
    inputs, meta, params = build_live_inputs(con, cfg)
    preds, _, _ = pregame.run(con, LIVE_VERSION, inputs, meta, params)
    return preds, meta
