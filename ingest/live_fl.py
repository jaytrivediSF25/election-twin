"""Florida Division of Elections live results.

Source: the DOS "Election Results Data Extract Utility":
  GET  https://results.elections.myflorida.com/downloadresults.asp?ElectionDate=M/D/YYYY&DATAMODE=
       (HTML form; hidden fields ElectionDate, OfficialResults, PartyRaces, DataMode)
  POST https://results.elections.myflorida.com/ResultsExtract.Asp  -> tab-delimited file
Columns: ElectionDate PartyCode PartyName RaceCode OfficeDesc CountyCode CountyName Juris1num
Juris2num Precincts PrecinctsReporting CanNameLast CanNameFirst CanNameMiddle CanVotes
RaceCodes: PRE, USS, USR (Juris1num = district), GOV.  County level, NO vote-method split, so
every row is vote_type='total'.  DOS: "may not be available prior to the closing of polls at 7 PM".
As of 2026-11-03 10:30 PT the 2026 general isn't in the election list yet (only "8/18/2026"
primary); expected ElectionDate=11/3/2026.
2026 Senate race is the special for the Rubio seat -> SEN-FL-S.
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from typing import Iterator

import httpx
import polars as pl

from ingest import _live_common as C

STATE = "FL"
SITE = "https://results.elections.myflorida.com"
DATE_2026 = "11/3/2026"
DATE_2024 = "11/5/2024"
CACHE = C.RAW / "live_fl"
SOURCE = "fldos"
SENATE_ID = {"2026": "SEN-FL-S", "2024": "SEN-FL", "2022": "SEN-FL"}


def race_id(code: str, juris1: str | None, year: str) -> str | None:
    code = (code or "").strip().upper()
    if code == "USS":
        return SENATE_ID.get(year, "SEN-FL")
    if code == "GOV":
        return "GOV-FL"
    if code == "PRE":
        return "PRES-FL"
    if code == "USR":
        d = re.sub(r"\D", "", juris1 or "")
        return f"HOUSE-FL-{int(d):02d}" if d else None
    return None


def read_extract(data: bytes) -> pl.DataFrame:
    df = pl.read_csv(io.BytesIO(data), separator="\t", infer_schema=False, encoding="utf8-lossy",
                     quote_char=None, truncate_ragged_lines=True)
    df = df.rename({c: c.strip() for c in df.columns})
    need = {"ElectionDate", "PartyCode", "RaceCode", "CountyName", "Juris1num", "CanVotes"}
    if need - set(df.columns):
        raise ValueError(f"FL extract missing columns {need - set(df.columns)}; header={df.columns}")
    return df


def parse(data: bytes, ts: datetime, source: str = SOURCE, include_zero: bool = False) -> pl.DataFrame:
    df = read_extract(data)
    if df.is_empty():
        return pl.DataFrame(schema=C.VOTEBATCH_SCHEMA)
    year = df["ElectionDate"][0].strip()[-4:]
    df = df.with_columns(
        pl.struct("RaceCode", "Juris1num").map_elements(
            lambda s: race_id(s["RaceCode"], s["Juris1num"], year), return_dtype=pl.Utf8).alias("race_id")
    ).filter(pl.col("race_id").is_not_null())
    fmap = {c: C.to_fips(STATE, c) for c in df["CountyName"].unique().to_list()}
    long = df.select(
        "race_id",
        pl.col("CountyName").replace_strict(fmap).alias("fips"),
        pl.lit("total").alias("vote_type"),
        pl.col("PartyCode").map_elements(C.side_from_party, return_dtype=pl.Utf8).alias("side"),
        pl.col("CanVotes").str.strip_chars().str.replace_all(",", "").cast(pl.Int64, strict=False).fill_null(0).alias("votes"),
    )
    return C.finalize(long, source, ts, include_zero)


def download(election_date: str, refresh: bool = True) -> tuple[bytes, datetime]:
    cache = CACHE / f"extract_{election_date.replace('/', '-')}.txt"
    if cache.exists() and not refresh:
        return cache.read_bytes(), datetime.fromtimestamp(cache.stat().st_mtime)
    with httpx.Client(headers=C.UA, timeout=60, follow_redirects=True) as c:
        form = c.get(f"{SITE}/downloadresults.asp", params={"ElectionDate": election_date, "DATAMODE": ""})
        form.raise_for_status()
        html = form.content.decode("latin-1")
        fields = dict(re.findall(r'NAME="(\w+)"\s+VALUE="([^"]*)"', html, re.I))
        if fields.get("ElectionDate") != election_date:
            raise RuntimeError(f"FL DOS extract for {election_date} not published yet (form fields={fields})")
        fields.setdefault("FormsButton2", "Download")
        r = c.post(f"{SITE}/ResultsExtract.Asp", data=fields)
        r.raise_for_status()
    if b"ElectionDate\t" not in r.content[:200]:
        raise RuntimeError(f"FL DOS extract not a results file: {r.content[:200]!r}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(r.content)
    return r.content, C.utcnow()


def poll(election_date: str = DATE_2026, include_zero: bool = False) -> pl.DataFrame:
    data, ts = download(election_date)
    return parse(data, ts, include_zero=include_zero)


def load_2024(refresh: bool = False) -> pl.DataFrame:
    data, _ = download(DATE_2024, refresh=refresh)
    return parse(data, datetime(2024, 11, 20), source=f"{SOURCE}:2024")


def replay_2024(n_chunks: int = 10) -> Iterator[pl.DataFrame]:
    yield from C.replay(load_2024(), n_chunks, source=f"{SOURCE}:replay2024")


def county_district_from_extract(data: bytes) -> pl.DataFrame:
    """fips -> district weights from a FL extract's USR rows, weight = share of the county's
    precincts in each district (DOS 'Precincts' column). Run on the 2026 extract tonight to get
    the NEW (2026) map; uncontested districts don't appear on the ballot and are missing."""
    df = read_extract(data).filter(pl.col("RaceCode") == "USR")
    df = (df.select("CountyName", pl.col("Juris1num").str.replace_all(r"\D", "").cast(pl.Int64).alias("d"),
                    pl.col("Precincts").cast(pl.Int64, strict=False).alias("p")).unique(["CountyName", "d"]))
    fmap = {c: C.to_fips(STATE, c) for c in df["CountyName"].unique().to_list()}
    return df.select(
        pl.col("CountyName").replace_strict(fmap).alias("fips"),
        pl.format("FL-{}", pl.col("d").cast(pl.Utf8).str.zfill(2)).alias("district_id"),
        (pl.col("p") / pl.col("p").sum().over("CountyName")).alias("weight"),
    ).sort("fips", "district_id")
