"""NC State Board of Elections live results.

Source: precinct-level results zip on the SBE S3 bucket, same tab-delimited format since
at least 2016:  https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/{YYYY_MM_DD}/results_pct_{YYYYMMDD}.zip
Columns: County, Election Date, Precinct, Contest Group ID, Contest Type, Contest Name, Choice,
Choice Party, Vote For, Election Day, Early Voting, Absentee by Mail, Provisional, Total Votes,
Real Precinct.  The 2026-11-03 file is published (zero-vote template since 2026-10-05) and is
overwritten in place on election night.

poll() -> VoteBatch frame for SEN-NC and HOUSE-NC-xx, county level, split by
election_day / early / mail / provisional.
"""
from __future__ import annotations

import io
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterator

import polars as pl

from ingest import _live_common as C

STATE = "NC"
URL_2026 = "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/2026_11_03/results_pct_20261103.zip"
URL_2024 = "https://s3.amazonaws.com/dl.ncsbe.gov/ENRS/2024_11_05/results_pct_20241105.zip"
CACHE = C.RAW / "live_nc"
SOURCE = "ncsbe"

METHOD_COLS = {"Election Day": "election_day", "Early Voting": "early",
               "Absentee by Mail": "mail", "Provisional": "provisional"}
_HOUSE = re.compile(r"^US HOUSE OF REPRESENTATIVES DISTRICT (\d+)$")


def race_id(contest: str) -> str | None:
    c = contest.strip().upper()
    if c == "US SENATE":
        return "SEN-NC"
    if c == "NC GOVERNOR":
        return "GOV-NC"
    if c == "US PRESIDENT":
        return "PRES-NC"
    m = _HOUSE.match(c)
    return f"HOUSE-NC-{int(m.group(1)):02d}" if m else None


def read_txt(data: bytes) -> pl.DataFrame:
    """Raw precinct rows (all strings). Accepts the .txt or the .zip bytes."""
    if data[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".txt"))
            data = z.read(name)
    df = pl.read_csv(io.BytesIO(data), separator="\t", infer_schema=False, encoding="utf8-lossy",
                     quote_char=None, truncate_ragged_lines=True)
    df = df.rename({c: c.strip() for c in df.columns})
    keep = [c for c in df.columns if c and not c.startswith("_")]
    missing = {"County", "Contest Name", "Choice", "Choice Party", *METHOD_COLS} - set(keep)
    if missing:
        raise ValueError(f"NC results file missing columns {missing}; header={df.columns}")
    return df.select(keep)


def parse(data: bytes, ts: datetime, source: str = SOURCE, include_zero: bool = False,
          races: set[str] | None = None) -> pl.DataFrame:
    df = read_txt(data)
    df = df.with_columns(pl.col("Contest Name").map_elements(race_id, return_dtype=pl.Utf8).alias("race_id"))
    df = df.filter(pl.col("race_id").is_not_null())
    if races is not None:
        df = df.filter(pl.col("race_id").is_in(list(races)))
    counties = df["County"].unique().to_list()
    fmap = {c: C.to_fips(STATE, c) for c in counties}
    long = (
        df.with_columns(
            pl.col("County").replace_strict(fmap).alias("fips"),
            pl.col("Choice Party").map_elements(C.side_from_party, return_dtype=pl.Utf8).alias("side"),
        )
        .unpivot(index=["race_id", "fips", "side"], on=list(METHOD_COLS), variable_name="vt", value_name="votes")
        .with_columns(
            pl.col("vt").replace_strict(METHOD_COLS).alias("vote_type"),
            pl.col("votes").str.strip_chars().cast(pl.Int64, strict=False).fill_null(0),
        )
    )
    return C.finalize(long.select("race_id", "fips", "vote_type", "side", "votes"), source, ts, include_zero)


def poll(url: str = URL_2026, include_zero: bool = False) -> pl.DataFrame:
    data, lm = C.fetch_bytes(url, CACHE / Path(url).name)
    ts = C.parse_http_date(lm) or C.utcnow()
    return parse(data, ts, include_zero=include_zero)


def load_2024(refresh: bool = False) -> pl.DataFrame:
    data, _ = C.fetch_bytes(URL_2024, CACHE / Path(URL_2024).name, refresh=refresh)
    return parse(data, datetime(2024, 12, 3, 18, 32), source=f"{SOURCE}:2024")


def replay_2024(n_chunks: int = 10) -> Iterator[pl.DataFrame]:
    yield from C.replay(load_2024(), n_chunks, source=f"{SOURCE}:replay2024")


def county_district_2026(refresh: bool = False) -> pl.DataFrame:
    """fips -> HOUSE district weights for the 2026 NC map.

    Precinct->district from the 2026 results template; weight = share of the county's 2024
    presidential votes cast in those precincts (vote-weighted proxy for population share).
    Falls back to precinct-count share where 2024 precinct ids don't match."""
    d26, _ = C.fetch_bytes(URL_2026, CACHE / Path(URL_2026).name, refresh=refresh)
    d24, _ = C.fetch_bytes(URL_2024, CACHE / Path(URL_2024).name, refresh=False)
    t26 = read_txt(d26).with_columns(pl.col("Contest Name").map_elements(race_id, return_dtype=pl.Utf8).alias("race_id"))
    pd = (t26.filter(pl.col("race_id").str.starts_with("HOUSE-NC-") & (pl.col("Real Precinct") == "Y"))
          .select("County", "Precinct", pl.col("race_id").str.replace("HOUSE-", "").alias("district_id")).unique())
    v24 = (read_txt(d24).filter(pl.col("Contest Name") == "US PRESIDENT")
           .group_by("County", "Precinct").agg(pl.col("Total Votes").cast(pl.Int64, strict=False).sum().alias("v")))
    j = pd.join(v24, on=["County", "Precinct"], how="left")
    agg = j.group_by("County", "district_id").agg(pl.col("v").sum().alias("v"), pl.len().alias("n"))
    tot = agg.group_by("County").agg(pl.col("v").sum().alias("vt"), pl.col("n").sum().alias("nt"))
    out = agg.join(tot, on="County").with_columns(
        pl.when(pl.col("vt") > 0).then(pl.col("v") / pl.col("vt")).otherwise(pl.col("n") / pl.col("nt")).alias("weight"))
    fmap = {c: C.to_fips(STATE, c) for c in out["County"].unique().to_list()}
    return out.select(pl.col("County").replace_strict(fmap).alias("fips"), "district_id", "weight").sort("fips", "district_id")
