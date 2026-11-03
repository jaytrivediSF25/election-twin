"""Scenarios: Scenario objects hold overrides; each one reruns the simulation.

overrides = {"national_shift": -3.0, "force": {"SEN-GA": "D"}, "base": "pregame-v1-LOCKED"}
"""
from __future__ import annotations

import hashlib
import json
import re

import polars as pl

from models import pregame
from models.montecarlo import SimParams
from ontology import funnel

STATE_NAMES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA", "colorado": "CO",
    "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", "kentucky": "KY", "louisiana": "LA",
    "maine": "ME", "maryland": "MD", "massachusetts": "MA", "michigan": "MI", "minnesota": "MN",
    "mississippi": "MS", "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN", "texas": "TX",
    "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}


def parse(text: str, favorites: dict[str, float]) -> dict:
    """Plain-English scenario -> overrides. favorites: race_id -> p_dem from the base run.

    "polls miss by R+3"  -> national_shift -3
    "flip Georgia"       -> force SEN-GA to the pregame underdog
    "D win SEN-ME"       -> force SEN-ME D
    """
    t = text.lower()
    out: dict = {}
    m = re.search(r"\b([dr])\s*\+\s*(\d+(?:\.\d+)?)", t)
    if m:
        out["national_shift"] = float(m.group(2)) * (1 if m.group(1) == "d" else -1)
    force: dict[str, str] = {}
    for m in re.finditer(r"\b(?:(d|r|dem|rep)\w*\s+(?:win|wins|take|takes|hold|holds)\s+|(flip)\s+)"
                         r"((?:sen|gov|house)-[a-z]{2}(?:-[a-z0-9]{1,2})?|[a-z]+(?: [a-z]+)?)", t):
        party, flip, target = m.group(1), m.group(2), m.group(3)
        rid = target.upper() if "-" in target else _statewide(target, favorites)
        if not rid:
            continue
        if flip:
            force[rid] = "R" if favorites.get(rid, 0.5) >= 0.5 else "D"
        else:
            force[rid] = "D" if party.startswith("d") else "R"
    if force:
        out["force"] = force
    if not out:
        raise ValueError(f"could not parse scenario: {text!r}")
    return out


def _statewide(name: str, favorites: dict[str, float]) -> str | None:
    words = name.split()
    for n in (" ".join(words[:2]), words[0]):
        st = STATE_NAMES.get(n)
        if st:
            for prefix in ("SEN", "GOV"):
                for rid in (f"{prefix}-{st}", f"{prefix}-{st}-S"):
                    if rid in favorites:
                        return rid
    return None


def run(con, description: str, overrides: dict | None = None, base: str = "pregame-v1-LOCKED",
        actor: str = "user") -> pl.DataFrame:
    try:
        _, inputs, meta, params = pregame.latest_run(con, base)
    except RuntimeError:
        base = "pregame-v1"
        _, inputs, meta, params = pregame.latest_run(con, base)
    if overrides is None:
        fav_df = pl.from_arrow(con.execute(
            "SELECT race_id, p_dem FROM merged_Prediction WHERE model_version = ? "
            "QUALIFY row_number() OVER (PARTITION BY race_id ORDER BY ts DESC) = 1", [base]).to_arrow_table())
        overrides = parse(description, dict(zip(fav_df["race_id"], fav_df["p_dem"])))
    overrides = {**overrides, "base": base}
    sid = "scn-" + hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:10]
    funnel.apply_action(con, "create_scenario", actor=actor, row={
        "scenario_id": sid, "description": description, "overrides": json.dumps(overrides),
        "created_at": pregame.utcnow()})
    p = SimParams(**{k: v for k, v in params.items() if k in SimParams.__dataclass_fields__})
    p.national_shift = float(overrides.get("national_shift", 0.0))
    preds, _, _ = pregame.run(con, f"scenario:{base}", inputs, meta, p, scenario_id=sid,
                              force=overrides.get("force"))
    return preds.with_columns(pl.lit(sid).alias("scenario_id"))
