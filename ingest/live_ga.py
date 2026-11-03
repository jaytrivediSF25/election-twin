"""Georgia SOS live results (Enhanced Voting platform; GA left Clarity in 2024).

API (public, JSON, same app as https://results.sos.ga.gov/results/public/Georgia):
  {BASE}/elections/Georgia/{electionId}/data                        -> election + ballotItems (state totals)
  {BASE}/elections/Georgia/{electionId}/data/ballot-item/{itemId}   -> ballotItemWithBreakdown.breakdownResults:
       one entry per county (locality.name), ballotOptions[].groupResults[] by vote method
electionId: 2026 general = "11032026GeneralAndSpecialElection" (live, zero votes, confirmed 2026-11-03);
            2024 general = "2024NovGen".
Vote-method groups: Election Day, Advance(d) Voting, Absentee by Mail, Provisional.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Iterator

import polars as pl

from ingest import _live_common as C

STATE = "GA"
BASE = "https://results.sos.ga.gov/results/public/api"
ELECTION_2026 = "11032026GeneralAndSpecialElection"
ELECTION_2024 = "2024NovGen"
CACHE = C.RAW / "live_ga"
SOURCE = "gasos"
_HOUSE = re.compile(r"^US House of Representatives\s*-\s*District\s*(\d+)", re.I)


def race_id(name: str) -> str | None:
    n = name.strip()
    low = n.lower()
    if low == "us senate":
        return "SEN-GA"
    if low == "governor":
        return "GOV-GA"
    if low.startswith("president of the us"):
        return "PRES-GA"
    m = _HOUSE.match(n)
    return f"HOUSE-GA-{int(m.group(1)):02d}" if m else None


def _text(xs) -> str:
    if not xs:
        return ""
    for x in xs:
        if x.get("languageId") == "en":
            return x.get("text") or ""
    return xs[0].get("text") or ""


def _side(opt: dict) -> str:
    party = (opt.get("party") or {})
    ab = (party.get("abbreviation") or _text(party.get("name"))).strip()
    if not ab:
        m = re.search(r"\((Dem|Rep)\)\s*$", _text(opt.get("name")))
        ab = m.group(1) if m else ""
    return C.side_from_party(ab)


def _url(election: str, item: str | None = None) -> str:
    u = f"{BASE}/elections/Georgia/{election}/data"
    return f"{u}/ballot-item/{item}" if item else u


def parse_ballot_item(doc: dict, include_zero: bool = False, source: str = SOURCE,
                      ts: datetime | None = None) -> pl.DataFrame:
    """Parse one /data/ballot-item/{id} response into VoteBatch rows."""
    b = doc["ballotItemWithBreakdown"]
    rid = race_id(_text(b["name"]))
    if rid is None:
        raise ValueError(f"unmapped GA ballot item {_text(b['name'])!r}")
    if ts is None:
        ts = _election_ts(doc.get("election") or {})
    rows = []
    for loc in b.get("breakdownResults") or []:
        fips = C.to_fips(STATE, _text(loc["locality"]["name"]))
        for opt in loc.get("ballotOptions") or []:
            side = _side(opt)
            groups = opt.get("groupResults") or []
            if not groups:  # feed without method split
                rows.append((rid, fips, "total", side, int(opt.get("voteCount") or 0)))
            for g in groups:
                rows.append((rid, fips, C.vote_type(_text(g["groupName"])), side, int(g.get("voteCount") or 0)))
    long = pl.DataFrame(rows, schema=["race_id", "fips", "vote_type", "side", "votes"], orient="row")
    return C.finalize(long, source, ts, include_zero)


def _election_ts(e: dict) -> datetime:
    s = e.get("lastUpdated") or e.get("asOf")
    if not s:
        return C.utcnow()
    s = re.sub(r"(\.\d{6})\d*", r"\1", s.rstrip("Z"))
    return datetime.fromisoformat(s)


def _get_json(url: str, cache_name: str, refresh: bool = True) -> dict:
    data, _ = C.fetch_bytes(url, CACHE / cache_name, refresh=refresh)
    return json.loads(data)


def _target_items(election: str, refresh: bool) -> tuple[dict, list[tuple[str, str]]]:
    d = _get_json(_url(election), f"{election}/data.json", refresh)
    items = [(b["id"], race_id(_text(b["name"]))) for b in d["ballotItems"]]
    return d, [(i, r) for i, r in items if r]


def fetch_election(election: str, include_zero: bool = False, refresh: bool = True,
                   source: str = SOURCE) -> pl.DataFrame:
    d, items = _target_items(election, refresh)
    ts = _election_ts(d.get("election") or {})

    def one(item):
        iid, _ = item
        doc = _get_json(_url(election, iid), f"{election}/item_{iid}.json", refresh)
        return parse_ballot_item(doc, include_zero, source, ts)

    with ThreadPoolExecutor(max_workers=6) as ex:
        frames = list(ex.map(one, items))
    frames = [f for f in frames if not f.is_empty()]
    return pl.concat(frames) if frames else pl.DataFrame(schema=C.VOTEBATCH_SCHEMA)


def poll(election: str = ELECTION_2026, include_zero: bool = False) -> pl.DataFrame:
    return fetch_election(election, include_zero)


def load_2024(refresh: bool = False) -> pl.DataFrame:
    return fetch_election(ELECTION_2024, refresh=refresh, source=f"{SOURCE}:2024")


def replay_2024(n_chunks: int = 10) -> Iterator[pl.DataFrame]:
    yield from C.replay(load_2024(), n_chunks, source=f"{SOURCE}:replay2024")


def county_district_2026(refresh: bool = False) -> pl.DataFrame:
    """GA's 2023 court-ordered map was used in 2024 and is unchanged for 2026, so weight =
    share of the county's 2024 US House votes cast in each district (vote-weighted proxy;
    understates districts with an uncontested 2024 race - none in GA 2024)."""
    h = load_2024(refresh).filter(pl.col("race_id").str.starts_with("HOUSE-GA-"))
    v = h.group_by("fips", "race_id").agg((pl.col("dem") + pl.col("rep") + pl.col("other")).sum().alias("v"))
    return (v.with_columns((pl.col("v") / pl.col("v").sum().over("fips")).alias("weight"))
            .select("fips", pl.col("race_id").str.replace("HOUSE-", "").alias("district_id"), "weight")
            .sort("fips", "district_id"))
