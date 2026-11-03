"""Fixture parsing tests for live adapters (2024 archived files, same format as 2026 feeds),
batch_id dedupe, replay, and County/CountyDistrict loads into an in-memory DB."""
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from ingest import _live_common as C
from ingest import live_fl, live_ga, live_nc
from ontology import funnel, oms

FX = Path(__file__).parent / "fixtures"
TS = datetime(2024, 11, 6, 3, 0)


def _row(df, race, fips, vt):
    r = df.filter((pl.col("race_id") == race) & (pl.col("fips") == fips) & (pl.col("vote_type") == vt))
    assert r.height == 1, (race, fips, vt, r)
    return r.row(0, named=True)


# ---------- NC ----------
@pytest.fixture(scope="module")
def nc():
    return live_nc.parse((FX / "nc_results_pct_20241105_sample.txt").read_bytes(), TS)


def test_nc_fixture(nc):
    assert set(nc["vote_type"]) == {"election_day", "early", "mail", "provisional"}
    assert {"PRES-NC", "GOV-NC", "HOUSE-NC-03", "HOUSE-NC-09"} <= set(nc["race_id"])
    assert "NC LIEUTENANT GOVERNOR" not in nc["race_id"].to_list()
    # Alamance (37001) president, summed independently from the fixture with awk
    assert (_row(nc, "PRES-NC", "37001", "election_day")["dem"], _row(nc, "PRES-NC", "37001", "election_day")["rep"]) == (8321, 11562)
    assert _row(nc, "PRES-NC", "37001", "early")["dem"] == 29553
    assert _row(nc, "PRES-NC", "37001", "mail")["rep"] == 1793
    assert _row(nc, "PRES-NC", "37001", "provisional")["dem"] == 217
    assert set(nc["fips"]) == {"37001", "37055"}
    assert nc["batch_id"].is_unique().all()


# ---------- GA ----------
@pytest.fixture(scope="module")
def ga():
    return live_ga.parse_ballot_item(json.loads((FX / "ga_2024NovGen_president_sample.json").read_text()))


def test_ga_fixture(ga):
    assert set(ga["race_id"]) == {"PRES-GA"}
    assert set(ga["fips"]) == {"13001", "13121", "13051"}  # Appling, Fulton, Chatham
    ed = _row(ga, "PRES-GA", "13001", "election_day")
    assert (ed["dem"], ed["rep"]) == (381, 1928)
    tot = ga.filter(pl.col("fips") == "13001").select(pl.col("dem").sum(), pl.col("rep").sum()).row(0)
    assert tot == (1560, 6761)
    assert ga["ts"][0] == datetime(2025, 1, 7, 16, 55, 0, 635883)


def test_ga_race_ids_and_labels():
    assert live_ga.race_id("US Senate") == "SEN-GA"
    assert live_ga.race_id("Governor") == "GOV-GA"
    assert live_ga.race_id("Lieutenant Governor") is None
    assert live_ga.race_id("US House of Representatives - District 6") == "HOUSE-GA-06"
    assert C.vote_type("Advance Voting") == C.vote_type("Advanced Voting") == "early"  # 2026 vs 2024 label
    with pytest.raises(ValueError):
        C.vote_type("Some New Bucket")


# ---------- FL ----------
@pytest.fixture(scope="module")
def fl():
    return live_fl.parse((FX / "fl_20241105_ElecResultsFL_sample.txt").read_bytes(), TS)


def test_fl_fixture(fl):
    assert set(fl["vote_type"]) == {"total"}
    assert set(fl["fips"]) == {"12001", "12086", "12109"}  # Alachua, Miami-Dade, St. Johns
    md = _row(fl, "PRES-FL", "12086", "total")
    assert (md["dem"], md["rep"], md["other"]) == (480355, 605590, 2096 + 1308 + 476 + 455 + 3813 + 12)
    assert "SEN-FL" in set(fl["race_id"])  # 2024 regular; 2026 maps to the special
    assert any(r.startswith("HOUSE-FL-") for r in fl["race_id"])


def test_fl_race_ids():
    assert live_fl.race_id("USS", "   ", "2026") == "SEN-FL-S"
    assert live_fl.race_id("USR", "007", "2026") == "HOUSE-FL-07"
    assert live_fl.race_id("GOV", " ", "2026") == "GOV-FL"
    assert live_fl.race_id("ATG", " ", "2026") is None


# ---------- batch ids / dedupe ----------
def test_batch_id_dedupe(nc):
    again = live_nc.parse((FX / "nc_results_pct_20241105_sample.txt").read_bytes(), datetime(2024, 11, 6, 4, 0))
    assert again["batch_id"].to_list() == nc["batch_id"].to_list()  # unchanged counts -> same ids, ts ignored
    b1 = C.batch_id("ncsbe", "SEN-NC", "37001", "early", 10, 20, 1)
    assert b1 == C.batch_id("ncsbe", "SEN-NC", "37001", "early", 10, 20, 1)
    assert b1 != C.batch_id("ncsbe", "SEN-NC", "37001", "early", 11, 20, 1)
    assert b1 != C.batch_id("ncsbe", "SEN-NC", "37001", "mail", 10, 20, 1)


def test_votebatch_writes_and_dedupes(nc):
    con = oms.connect(":memory:")
    funnel.write_stream(con, "VoteBatch", nc)
    funnel.write_stream(con, "VoteBatch", nc)  # re-poll, unchanged
    assert oms.one(con, "SELECT count(*) FROM obj_VoteBatch")[0] == nc.height


def test_zero_rows_dropped_by_default():
    txt = (FX / "nc_results_pct_20241105_sample.txt").read_text().splitlines(True)
    header, rows = txt[0], txt[1:]
    zeroed = []
    for line in rows:
        f = line.rstrip("\r\n").split("\t")
        f[9:14] = ["0"] * 5
        zeroed.append("\t".join(f) + "\n")
    data = (header + "".join(zeroed)).encode()
    assert live_nc.parse(data, TS).is_empty()
    assert live_nc.parse(data, TS, include_zero=True).height > 0


# ---------- replay ----------
def test_replay_cumulative(nc):
    frames = list(C.replay(nc, 6, "ncsbe:replay2024"))
    assert len(frames) == 6
    key = ["race_id", "fips", "vote_type"]
    tot = [f.select((pl.col("dem") + pl.col("rep") + pl.col("other")).sum()).item() for f in frames]
    assert tot == sorted(tot) and tot[-1] == nc.select((pl.col("dem") + pl.col("rep") + pl.col("other")).sum()).item()
    first = frames[0]
    assert set(first["vote_type"]) <= {"early", "mail", "election_day"}
    assert "provisional" not in set(first["vote_type"])
    final = frames[-1].sort(key).select(*key, "dem", "rep", "other")
    assert final.equals(nc.sort(key).select(*key, "dem", "rep", "other"))
    assert all(f["ts"].n_unique() == 1 for f in frames)
    assert frames[0]["ts"][0] < frames[-1]["ts"][0]


def test_replay_total_only(fl):
    frames = list(C.replay(fl, 4, "fldos:replay2024"))
    assert frames[-1].select(pl.col("dem").sum()).item() == fl.select(pl.col("dem").sum()).item()
    assert frames[0].select(pl.col("dem").sum()).item() < frames[-1].select(pl.col("dem").sum()).item()


# ---------- County load (uses cached data/raw/county_2024, downloads if missing) ----------
def test_county_load_in_memory():
    from ingest import county_2024 as m
    raw = {"pres24_county": m._county_raw()}
    objs = m.to_objects(raw)
    con = oms.connect(":memory:")
    n = funnel.write_batch(con, "County", objs["County"], mode="replace")
    assert n >= 3100
    by_state = dict(con.execute("SELECT state, count(*) FROM obj_County GROUP BY state").fetchall())
    assert by_state["NC"] == 100 and by_state["GA"] == 159 and by_state["FL"] == 67
    assert by_state["AK"] == 1 and by_state["DC"] == 1 and by_state["CT"] == 9
    m_ = con.execute("SELECT pres24_margin FROM obj_County WHERE fips='37119'").fetchone()[0]  # Mecklenburg
    assert 32 < m_ < 34
    assert con.execute("SELECT count(*) FROM obj_County WHERE length(fips) <> 5").fetchone()[0] == 0


def test_county_district_and_method_baseline():
    """Uses the cached NC/GA raw files (network if missing)."""
    from ingest import county_2024 as m
    raw = m.fetch()
    pbm = raw["pres24_by_method"]
    assert pbm.columns == ["fips", "vote_type", "dem", "rep", "total"]
    assert set(pbm["vote_type"]) == {"election_day", "early", "mail", "provisional"}
    assert pbm.filter(pl.col("fips").str.starts_with("37"))["fips"].n_unique() == 100
    assert pbm.filter(pl.col("fips").str.starts_with("13"))["fips"].n_unique() == 159
    nc_d = pbm.filter(pl.col("fips").str.starts_with("37"))["dem"].sum()
    assert nc_d == 2715375  # official NC 2024 Harris total
    objs = m.to_objects(raw)
    con = oms.connect(":memory:")
    funnel.write_batch(con, "County", objs["County"])
    funnel.write_batch(con, "CountyDistrict", objs["CountyDistrict"])
    sums = con.execute("SELECT fips, sum(weight) FROM obj_CountyDistrict GROUP BY fips").fetchall()
    assert len(sums) == 259 and all(abs(s - 1) < 1e-9 for _, s in sums)
    orphans = con.execute("SELECT count(*) FROM obj_CountyDistrict cd LEFT JOIN obj_County c USING (fips) WHERE c.fips IS NULL").fetchone()[0]
    assert orphans == 0
