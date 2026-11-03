"""Shared helpers for live result adapters (live_nc / live_fl / live_ga) and the
2024 by-method baseline in county_2024.

VoteBatch rows: batch_id, race_id, fips, vote_type, dem, rep, other, ts, source.
batch_id = sha1(source|race_id|fips|vote_type|dem|rep|other) so unchanged counts dedupe.
"""
from __future__ import annotations

import csv
import hashlib
import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import httpx
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
REF = Path(__file__).resolve().parent / "ref"
UA = {"User-Agent": "Mozilla/5.0 (election-twin ingest)"}

VOTEBATCH_SCHEMA = {
    "batch_id": pl.Utf8, "race_id": pl.Utf8, "fips": pl.Utf8, "vote_type": pl.Utf8,
    "dem": pl.Int64, "rep": pl.Int64, "other": pl.Int64, "ts": pl.Datetime("us"), "source": pl.Utf8,
}

# feed label (lowercased) -> schema vote_type
VOTE_TYPES = {
    "election day": "election_day",
    "early voting": "early",           # NC one-stop
    "advanced voting": "early",        # GA 2024 label
    "advance voting": "early",         # GA 2026 label
    "absentee by mail": "mail",
    "provisional": "provisional",
    "total": "total",
    "total votes": "total",
}


def vote_type(label: str) -> str:
    key = label.strip().lower()
    if key not in VOTE_TYPES:
        raise ValueError(f"unknown vote type label {label!r}")  # fail loud: a new label tonight must be mapped
    return VOTE_TYPES[key]


def norm_county(name: str) -> str:
    s = name.upper().replace(" COUNTY", "").replace(" PARISH", "")
    return re.sub(r"[^A-Z]", "", s)


@lru_cache(maxsize=None)
def fips_lookup(state: str) -> dict[str, str]:
    with open(REF / "county_fips_nc_fl_ga.csv") as f:
        return {norm_county(r["name"]): r["fips"] for r in csv.DictReader(f) if r["state"] == state}


def to_fips(state: str, county_name: str) -> str:
    lk = fips_lookup(state)
    key = norm_county(county_name)
    if key not in lk:
        raise KeyError(f"{state}: no FIPS for county {county_name!r}")
    return lk[key]


def batch_id(source: str, race_id: str, fips: str, vtype: str, dem: int, rep: int, other: int) -> str:
    s = f"{source}|{race_id}|{fips}|{vtype}|{int(dem)}|{int(rep)}|{int(other)}"
    return hashlib.sha1(s.encode()).hexdigest()[:24]


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def finalize(long: pl.DataFrame, source: str, ts: datetime, include_zero: bool = False) -> pl.DataFrame:
    """long: race_id, fips, vote_type, side ('dem'|'rep'|'other'), votes -> VoteBatch frame."""
    if long.is_empty():
        return pl.DataFrame(schema=VOTEBATCH_SCHEMA)
    wide = (
        long.group_by(["race_id", "fips", "vote_type", "side"]).agg(pl.col("votes").sum())
        .pivot(on="side", index=["race_id", "fips", "vote_type"], values="votes")
    )
    for c in ("dem", "rep", "other"):
        if c not in wide.columns:
            wide = wide.with_columns(pl.lit(0).alias(c))
    wide = wide.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in ("dem", "rep", "other")])
    if not include_zero:
        wide = wide.filter((pl.col("dem") + pl.col("rep") + pl.col("other")) > 0)
    return stamp(wide, source, ts)


def stamp(wide: pl.DataFrame, source: str, ts: datetime) -> pl.DataFrame:
    ids = [batch_id(source, r, f, v, d, p, o) for r, f, v, d, p, o in
           wide.select("race_id", "fips", "vote_type", "dem", "rep", "other").iter_rows()]
    return wide.with_columns(
        pl.Series("batch_id", ids, pl.Utf8), pl.lit(ts).cast(pl.Datetime("us")).alias("ts"), pl.lit(source).alias("source"),
    ).select(list(VOTEBATCH_SCHEMA)).sort(["race_id", "fips", "vote_type"])


def fetch_bytes(url: str, cache: Path, *, method: str = "GET", data: dict | None = None,
                refresh: bool = True, timeout: float = 60) -> tuple[bytes, str | None]:
    """Download to cache (data only, never executed). Returns (content, Last-Modified header)."""
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not refresh:
        return cache.read_bytes(), None
    with httpx.Client(headers=UA, timeout=timeout, follow_redirects=True) as c:
        r = c.request(method, url, data=data)
        r.raise_for_status()
    cache.write_bytes(r.content)
    return r.content, r.headers.get("last-modified")


def parse_http_date(s: str | None) -> datetime | None:
    if not s:
        return None
    from email.utils import parsedate_to_datetime
    return parsedate_to_datetime(s).astimezone(timezone.utc).replace(tzinfo=None)


def replay(final: pl.DataFrame, n_chunks: int, source: str,
           start: datetime = datetime(2024, 11, 6, 0, 0)) -> Iterator[pl.DataFrame]:
    """Yield cumulative VoteBatch frames that build up to `final` (a 2024 VoteBatch frame).

    Order mimics election night: early + mail are released first for every county,
    election-day votes arrive county by county (deterministic hashed order, each county
    passing through a partial count), provisional in the last chunk. Total-only feeds
    (vote_type='total') reveal each county's totals gradually, scaled proportionally.
    """
    if n_chunks < 1:
        raise ValueError("n_chunks >= 1")
    base = final.select("race_id", "fips", "vote_type", "dem", "rep", "other")
    ranks = {f: (int(hashlib.md5(f.encode()).hexdigest(), 16) % 10_000) / 10_000
             for f in base["fips"].unique().to_list()}
    rank = pl.col("fips").replace_strict(ranks, return_dtype=pl.Float64)
    for k in range(1, n_chunks + 1):
        t = k / n_chunks
        if k == n_chunks:
            frac = pl.lit(1.0)
        else:
            ed = ((t - 0.15 - 0.6 * rank) / 0.25).clip(0.0, 1.0)
            tot = ((t - 0.6 * rank) / 0.4).clip(0.0, 1.0)
            frac = (pl.when(pl.col("vote_type").is_in(["early", "mail"])).then(1.0)
                    .when(pl.col("vote_type") == "election_day").then(ed)
                    .when(pl.col("vote_type") == "provisional").then(0.0)
                    .otherwise(tot))
        cur = base.with_columns(frac.alias("_f")).with_columns(
            [(pl.col(c) * pl.col("_f")).floor().cast(pl.Int64).alias(c) for c in ("dem", "rep", "other")]
        ).drop("_f").filter((pl.col("dem") + pl.col("rep") + pl.col("other")) > 0)
        yield stamp(cur, source, start + timedelta(minutes=15 * k))


def side_from_party(party: str | None) -> str:
    p = (party or "").strip().upper()
    if p in ("DEM", "DEMOCRAT", "DEMOCRATIC"):
        return "dem"
    if p in ("REP", "REPUBLICAN"):
        return "rep"
    return "other"
