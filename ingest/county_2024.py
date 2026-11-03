"""County objects (2024 presidential results) + NC/GA 2024 vote-method baseline + CountyDistrict.

County source: tonmcg/US_County_Level_Election_Results_08-24 (2024 file).
Quirks:
  - Alaska is reported by 40 state house districts with fake FIPS 02001-02040 (which collide
    with real borough FIPS) and two-party-only totals -> collapsed to ONE statewide row, fips
    '02000', name 'Alaska (statewide)'. pres24_total there = dem + rep (others not reported).
  - DC is reported by 8 wards (11001-11008) -> collapsed to '11001'.
  - Connecticut uses the 9 Census planning regions (09110-09190), kept as-is (they are the
    current Census county-equivalents; old 8-county FIPS 09001-09015 do NOT appear).

pres24_by_method (raw): fips, vote_type, dem, rep, total for NC and GA only, from the official
state files (NC SBE results_pct zip; GA SOS Enhanced Voting API). vote_type values are the
VoteBatch ones: election_day / early / mail / provisional. FL is not included: neither the DOS
statewide files nor MEDSL split by method (method splits only exist on 67 county SOE sites).

CountyDistrict: NC (2026 map from the 2026 results template, weights = 2024 pres-vote share of
precincts) and GA (unchanged map, weights = 2024 US House vote share). FL optional (2026 map from
the live Election Watch file, precinct-count share, can sum < 1 where districts are uncontested).
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from ingest import _live_common as C
from ingest import live_fl, live_ga, live_nc

NAME = "county_2024"
CACHE = C.RAW / NAME
TONMCG = ("https://raw.githubusercontent.com/tonmcg/US_County_Level_Election_Results_08-24/"
          "master/2024_US_County_Level_Presidential_Results.csv")
SOURCES = {
    "pres24_county": TONMCG,
    "pres24_by_method": f"{live_nc.URL_2024} ; {live_ga._url(live_ga.ELECTION_2024)}/ballot-item/<president>",
    "county_district": f"{live_nc.URL_2026} + {live_nc.URL_2024} ; {live_ga._url(live_ga.ELECTION_2024)}",
}
STATE_PO = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def _county_raw(refresh: bool = False) -> pl.DataFrame:
    data, _ = C.fetch_bytes(TONMCG, CACHE / "pres24_county.csv", refresh=refresh)
    return pl.read_csv(data, schema_overrides={"county_fips": pl.Utf8})


def pres24_by_method(refresh: bool = False) -> pl.DataFrame:
    frames = [live_nc.load_2024(refresh), live_ga.load_2024(refresh)]
    vb = pl.concat(frames).filter(pl.col("race_id").str.starts_with("PRES-"))
    return (vb.select("fips", "vote_type", "dem", "rep", (pl.col("dem") + pl.col("rep") + pl.col("other")).alias("total"))
            .sort("fips", "vote_type"))


def county_district(include_fl: bool = False, refresh: bool = False) -> pl.DataFrame:
    frames = [live_nc.county_district_2026(refresh), live_ga.county_district_2026(refresh)]
    if include_fl:
        frames.append(live_fl.county_district_2026(refresh))
    return pl.concat([f.select("fips", "district_id", pl.col("weight").cast(pl.Float64)) for f in frames])


def fetch(refresh: bool = False) -> dict[str, pl.DataFrame]:
    return {
        "pres24_county": _county_raw(refresh),
        "pres24_by_method": pres24_by_method(refresh),
        "county_district": county_district(refresh=refresh),
    }


def to_objects(raw: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    df = raw["pres24_county"].with_columns(
        pl.col("county_fips").str.zfill(5).alias("fips"),
        pl.col("state_name").replace_strict(STATE_PO).alias("state"),
        pl.col("county_name").alias("name"),
    )
    df = df.with_columns(
        pl.when(pl.col("state") == "AK").then(pl.lit("02000"))
        .when(pl.col("state") == "DC").then(pl.lit("11001")).otherwise(pl.col("fips")).alias("fips"),
        pl.when(pl.col("state") == "AK").then(pl.lit("Alaska (statewide)"))
        .when(pl.col("state") == "DC").then(pl.lit("District of Columbia")).otherwise(pl.col("name")).alias("name"),
    )
    county = (
        df.group_by("fips", "state", "name", maintain_order=True)
        .agg(pl.col("votes_dem").sum().alias("pres24_dem"), pl.col("votes_gop").sum().alias("pres24_rep"),
             pl.col("total_votes").sum().alias("pres24_total"))
        .with_columns(((pl.col("pres24_dem") - pl.col("pres24_rep")) / pl.col("pres24_total") * 100).alias("pres24_margin"))
        .sort("fips")
    )
    out = {"County": county}
    if "county_district" in raw and raw["county_district"] is not None:
        out["CountyDistrict"] = raw["county_district"].select("fips", "district_id", "weight")
    return out
