"""Pregame model: uniform swing fundamentals + ratings prior + poll blend -> correlated MC.

Reads only through the Object Set Service; writes Prediction objects through the Funnel.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone

import numpy as np
import polars as pl

from models.montecarlo import SimParams, SimResult, simulate
from ontology import funnel, oms
from ontology.objects import ObjectSet

PRES24_NATIONAL = -1.48            # Trump +1.48 national popular vote margin
RATING_MARGIN = {"Safe D": 20.0, "Likely D": 9.0, "Lean D": 4.0, "Tossup": 0.0,
                 "Lean R": -4.0, "Likely R": -9.0, "Safe R": -20.0}
RATING_WEIGHT = 0.4                # pull of non-safe ratings on fundamentals
INCUMBENCY = {"house": 2.5, "senate": 2.0, "governor": 2.0}
SD_RACE = {"house": 5.0, "senate": 5.5, "governor": 5.5}   # unpolled
SD_RACE_POLLED = (3.5, 4.5)        # (>=3 polls, 1-2 polls)
POLL_HALF_LIFE_DAYS = 21.0
POLL_CUTOFF = {"GENERIC": date(2026, 8, 1), "default": date(2026, 7, 1)}
POP_WEIGHT = {"lv": 1.0, "rv": 0.8, "a": 0.6}


@dataclass
class ModelConfig:
    as_of: date = date(2026, 11, 3)
    generic_override: float | None = None   # scenario / testing hook


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def poll_average(polls: pl.DataFrame, as_of: date) -> tuple[float | None, int]:
    if polls.is_empty():
        return None, 0
    df = polls.filter(~pl.col("excluded").fill_null(False) & pl.col("margin").is_not_null())
    if df.is_empty():
        return None, 0
    age = df.select((pl.lit(as_of) - pl.col("end_date")).dt.total_days().fill_null(60)).to_series().to_numpy().astype(float)
    n = df["sample_size"].fill_null(600).to_numpy().astype(float)
    pop = np.array([POP_WEIGHT.get((p or "").lower(), 0.8) for p in df["population"].to_list()])
    w = df["weight"].fill_null(1.0).to_numpy() * np.sqrt(np.clip(n, 100, 2000) / 600) * pop
    w = w * 0.5 ** (np.clip(age, 0, None) / POLL_HALF_LIFE_DAYS)
    if w.sum() <= 0:
        return None, 0
    return float((w * df["margin"].to_numpy()).sum() / w.sum()), df.height


def race_polls(con, race_id: str) -> tuple[float | None, int]:
    cutoff = POLL_CUTOFF.get(race_id, POLL_CUTOFF["default"])
    polls = ObjectSet(con, "Poll").filter(race_id=race_id).where("end_date >= ?", cutoff).df()
    own, n = poll_average(polls, ModelConfig.as_of)
    sites = ObjectSet(con, "PollAverage").filter(race_id=race_id).df()
    site = float(sites["margin"].mean()) if sites.height and sites["margin"].null_count() < sites.height else None
    if own is None and site is None:
        return None, 0
    if own is None:
        return site, max(n, int(sites["n_polls"].fill_null(3).max() or 3))
    if site is None:
        return own, n
    return 0.5 * own + 0.5 * site, max(n, int(sites["n_polls"].fill_null(0).max() or 0))


def state_pres_margins(con) -> dict[str, float]:
    df = ObjectSet(con, "County").aggregate(by=["state"], d="sum(pres24_dem)::DOUBLE", r="sum(pres24_rep)::DOUBLE", t="sum(pres24_total)::DOUBLE")
    return {s: 100.0 * (d - r) / t for s, d, r, t in df.iter_rows() if t}


def senate_holdover_d(con) -> int:
    try:
        rows = con.execute("SELECT party, seats FROM raw_senate_holdovers").fetchall()
    except Exception:
        return 0
    return int(sum(s for p, s in rows if p in ("D", "I")))


def build_inputs(con, cfg: ModelConfig | None = None) -> tuple[pl.DataFrame, dict]:
    cfg = cfg or ModelConfig()
    generic, n_generic = (cfg.generic_override, 0) if cfg.generic_override is not None else race_polls(con, "GENERIC")
    if generic is None:
        raise RuntimeError("no generic ballot data: cannot compute national swing")
    swing = generic - PRES24_NATIONAL
    state_pres = state_pres_margins(con)
    races = ObjectSet(con, "Race").where("chamber IN ('house', 'senate', 'governor')").df(order_by="race_id")

    rows = []
    for r in races.iter_rows(named=True):
        ch = r["chamber"]
        base = r["pres24_margin"]
        if base is None:
            base = state_pres.get(r["state"])
        if base is None:
            continue
        inc = {"D": 1, "R": -1}.get(r["incumbent_party"], 0) if not r["open_seat"] else 0
        fundamental = base + swing + inc * INCUMBENCY[ch]
        rating = r["rating"]
        if rating in ("Safe D", "Safe R"):
            bound = 10.0 * (1 if rating == "Safe D" else -1)
            fundamental = max(fundamental, bound) if bound > 0 else min(fundamental, bound)
        elif rating in RATING_MARGIN:
            fundamental = (1 - RATING_WEIGHT) * fundamental + RATING_WEIGHT * RATING_MARGIN[rating]
        poll, n_polls = race_polls(con, r["race_id"])
        if poll is not None and n_polls > 0:
            w = min(0.85, n_polls / (n_polls + 1.5))
            mean = w * poll + (1 - w) * fundamental
            sd = SD_RACE_POLLED[0] if n_polls >= 3 else SD_RACE_POLLED[1]
        else:
            mean, sd = fundamental, SD_RACE[ch]
        rows.append({
            "race_id": r["race_id"], "chamber": ch, "state": r["state"], "mean": float(mean),
            "sd_race": float(sd), "nat_load": 1.0, "state_load": 1.0, "fundamental": float(fundamental),
            "poll_avg": poll, "n_polls": int(n_polls), "rating": rating,
        })
    meta = {"generic": generic, "n_generic_polls": n_generic, "swing": swing,
            "senate_holdover_d": senate_holdover_d(con), "as_of": str(cfg.as_of)}
    return pl.DataFrame(rows, infer_schema_length=None), meta


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def frame_hash(df: pl.DataFrame) -> str:
    return _hash(df.sort(df.columns[0]).to_dicts())


def _ensure_run_tables(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS model_runs (run_id VARCHAR, model_version VARCHAR, ts TIMESTAMP,
                   params JSON, meta JSON, inputs_hash VARCHAR, outputs_hash VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS model_inputs (run_id VARCHAR, race_id VARCHAR, chamber VARCHAR,
                   state VARCHAR, mean DOUBLE, sd_race DOUBLE, nat_load DOUBLE, state_load DOUBLE,
                   fundamental DOUBLE, poll_avg DOUBLE, n_polls INTEGER, rating VARCHAR)""")


def run(con, model_version: str, inputs: pl.DataFrame, meta: dict, params: SimParams | None = None,
        scenario_id: str | None = None, force: dict[str, str] | None = None) -> tuple[pl.DataFrame, SimResult, str]:
    """Simulate, write Predictions + run lineage. Returns (predictions, sims, run_id)."""
    params = params or SimParams()
    params.senate_holdover_d = int(meta.get("senate_holdover_d", 0))
    sims = simulate(inputs, params)
    mask = sims.mask_for(force) if force else None
    ts = utcnow()
    in_hash = _hash({"inputs": inputs.sort("race_id").to_dicts(), "params": asdict(params), "meta": meta,
                     "force": force})
    preds = sims.summarize(mask).with_columns(
        pl.lit(model_version).alias("model_version"), pl.lit(ts).alias("ts"),
        pl.lit(scenario_id, dtype=pl.Utf8).alias("scenario_id"), pl.lit(in_hash).alias("inputs_hash"),
    )
    out_hash = frame_hash(preds.select("race_id", "p_dem", "mean_margin", "p10", "p90", "n_sims"))
    funnel.write_batch(con, "Prediction", preds, source=f"model:{model_version}")
    _ensure_run_tables(con)
    run_id = f"{model_version}@{ts.isoformat()}"
    con.execute("INSERT INTO model_runs VALUES (?, ?, ?, ?, ?, ?, ?)",
                [run_id, model_version, ts, json.dumps(asdict(params)), json.dumps(meta, default=str), in_hash, out_hash])
    cols = ["race_id", "chamber", "state", "mean", "sd_race", "nat_load", "state_load", "fundamental",
            "poll_avg", "n_polls", "rating"]
    con.register("_mi", inputs.select([c for c in cols if c in inputs.columns]).to_arrow())
    try:
        present = [c for c in cols if c in inputs.columns]
        con.execute(f"INSERT INTO model_inputs (run_id, {', '.join(present)}) SELECT ?, {', '.join(present)} FROM _mi", [run_id])
    finally:
        con.unregister("_mi")
    return preds, sims, run_id


def latest_run(con, model_version: str) -> tuple[str, pl.DataFrame, dict, dict]:
    row = oms.one(con, "SELECT run_id, meta, params FROM model_runs WHERE model_version = ? ORDER BY ts DESC LIMIT 1",
                  [model_version])
    if not row:
        raise RuntimeError(f"no run for {model_version}")
    run_id, meta, params = row
    inputs = pl.from_arrow(con.execute("SELECT * EXCLUDE (run_id) FROM model_inputs WHERE run_id = ? ORDER BY race_id",
                                       [run_id]).to_arrow_table())
    return run_id, inputs, json.loads(meta), json.loads(params)
