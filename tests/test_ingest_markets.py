import os
import time
from datetime import datetime

import httpx
import polars as pl
import pytest

from ingest import markets as M
from ontology import funnel, oms


# ------------------------------------------------------------ offline mapping

@pytest.mark.parametrize("ticker,title,expected", [
    ("CONTROLH-2026", "Who will control the House?", "CONTROL-HOUSE"),
    ("CONTROLS-2026", "Who will control the Senate?", "CONTROL-SENATE"),
    ("CONTROLH-2028", "2028 House winner", None),
    ("SENATEGA-26", "Georgia Senate winner?", "SEN-GA"),
    ("SENATEOHS-26", "Ohio Senate winner?", "SEN-OH-S"),
    ("SENATEFLS-26", "Florida Senate winner?", "SEN-FL-S"),
    ("KXSENATELA-26NOV", "Louisiana Senate winner?", "SEN-LA"),
    ("SENATEWV-26", "West Virginia Senate winner?", "SEN-WV"),
    ("SENATEGA-28", "Georgia Senate winner? (2028)", None),
    ("SENATELA-26", "Kentucky Senate winner?", "SEN-KY"),      # explicit override
    ("SENATEVA-26", "West Virginia Senate winner?", None),     # title/ticker mismatch -> drop
    ("HOUSEPA7-26", "PA-07 House winner?", "HOUSE-PA-07"),
    ("KXHOUSERACE-PA07-26", "PA-07 House winner?", "HOUSE-PA-07"),
    ("KXHOUSEUT01-26", "UT-01 House winner?", "HOUSE-UT-01"),
    ("HOUSEAKAL-26", "Alaska House winner?", "HOUSE-AK-AL"),
    ("KXHOUSERACE-WYAL-26", "WY-AL House winner?", "HOUSE-WY-AL"),
    ("GOVPARTYAZ-26", "Arizona Governor winner?", "GOV-AZ"),
    ("KXGOVPARTYAK-26", "Alaska Governor winner? (Party)", "GOV-AK"),
    ("GOVPARTYKS-27", "Kansas Governor winner?", "GOV-KS"),    # override
    ("GOVPARTYKY-27", "Kentucky governor winner? (2027)", None),
    ("KXGOVCA-26", "California Governor winner? (Person)", None),
    ("KXSENATEGAD-26", "Georgia Democratic Senate nominee", None),
])
def test_kalshi_race_id(ticker, title, expected):
    assert M.kalshi_race_id(ticker, title) == expected


@pytest.mark.parametrize("ticker,expected", [
    ("SENATEGA-26-D", "D"), ("SENATEGA-26-R", "R"),
    ("KXHOUSENC11-26-DEM", "D"), ("KXHOUSENC11-26-GOP", "R"),
    ("SENATENE-26-DOSB", "D"), ("SENATENE-26-D", None),       # Osborn is the D side
    ("GOVPARTYRI-26-KBLO", None), ("HOUSEIA1-26-I", None),
])
def test_kalshi_yes_means(ticker, expected):
    assert M.kalshi_yes_means(ticker) == expected


@pytest.mark.parametrize("slug,expected", [
    ("which-party-will-win-the-senate-in-2026", "CONTROL-SENATE"),
    ("which-party-will-win-the-house-in-2026", "CONTROL-HOUSE"),
    ("georgia-senate-election-winner", "SEN-GA"),
    ("alabama-senate-election-winner-154", "SEN-AL"),
    ("ohio-senate-election-winner", "SEN-OH-S"),
    ("florida-senate-election-winner", "SEN-FL-S"),
    ("north-carolina-senate-election-winner", "SEN-NC"),
    ("west-virginia-senate-election-winner", "SEN-WV"),
    ("arizona-governor-winner-2026", "GOV-AZ"),
    ("alaska-governor-election-winner", "GOV-AK"),
    ("california-governor-election-2026", "GOV-CA"),
    ("pa-07-house-election-winner", "HOUSE-PA-07"),
    ("ak-al-house-election-winner", "HOUSE-AK-AL"),
    ("me-02-house-election-winner-20260720160226434", "HOUSE-ME-02"),
    ("which-party-will-win-the-house-race-for-the-ca-22-seat", "HOUSE-CA-22"),
    ("ca-11-house-election-winner-individual-20260710015446690", None),
    ("georgia-lieutenant-governor-election-winner-2026", None),
    ("guam-governor-election-winner-2026", None),
    ("pa-07-house-margin-of-victory-2026", None),
    ("texas-senate-election-tarrant-county-winner", None),
    ("which-party-will-win-the-texas-state-senate-in-2026", None),
])
def test_polymarket_race_id(slug, expected):
    assert M.polymarket_race_id(slug) == expected


@pytest.mark.parametrize("mid,question,group,race,expected", [
    ("1", "Will the Democrats win the Georgia Senate race in 2026?", "Jon Ossoff (D)", "SEN-GA", "D"),
    ("2", "Will the Republican Party win the PA-07 House seat?", "Ryan Mackenzie (R)", "HOUSE-PA-07", "R"),
    ("3", "Will the Democratic Party control the House after the 2026 Midterm elections?", "", "CONTROL-HOUSE", "D"),
    ("4", "Will another party win the PA-07 House seat?", "Other", "HOUSE-PA-07", None),
    ("634893", "Will an independent win the Nebraska Senate race in 2026?", "Independent", "SEN-NE", "D"),
    ("634891", "Will the Democrats win the Nebraska Senate race in 2026?", "Democrat", "SEN-NE", None),
    ("5", "Will Mary Peltola win the Alaska Senate race in 2026?", "Mary Peltola", "SEN-AK", "D"),
    ("6", "Will Mary Peltola win the 2026 Alaska governor election?", "Mary Peltola", "GOV-AK", None),
    ("7", "Will Todd Achilles win the Idaho Senate race in 2026?", "Todd Achilles (I)", "SEN-ID", None),
    ("8", "Will Jane Doe win the X race?", "Jane Doe (R)", "SEN-ZZ", "R"),
])
def test_polymarket_yes_means(mid, question, group, race, expected):
    assert M.polymarket_yes_means(mid, question, group, race) == expected


def test_price():
    assert M.price(0.40, 0.42, 0.9) == pytest.approx(0.41)
    assert M.price(None, 0.004, 0.002) == 0.002      # one-sided book -> last
    assert M.price(0.0, 1.0, 0.3) == 0.3             # empty book -> last
    assert M.price(None, None, None) is None


def _raw_fixture():
    ts = datetime(2026, 11, 3, 18, 0)
    row = dict(token_id=None, status="active", fetched_at=ts, clob_mid=None, tag="pull")
    k = pl.DataFrame([
        dict(venue="kalshi", native_id="CONTROLH-2026-D", event_id="CONTROLH-2026",
             event_title="Who will control the House?", title="House D", outcome="Democratic Party",
             bid=0.92, ask=0.94, last=0.93, **row),
        dict(venue="kalshi", native_id="SENATEMT-26-D", event_id="SENATEMT-26",
             event_title="Montana Senate winner?", title="MT D", outcome="Alani Bankhead",
             bid=0.003, ask=0.005, last=0.004, **row),
        dict(venue="kalshi", native_id="SENATEMT-26-R", event_id="SENATEMT-26",
             event_title="Montana Senate winner?", title="MT R", outcome="Kurt Alme",
             bid=0.90, ask=0.92, last=0.91, **row),
    ], schema={**M._RAW_SCHEMA, "clob_mid": pl.Float64, "tag": pl.Utf8})
    p = pl.DataFrame([
        dict(venue="polymarket", native_id="630693", event_id="georgia-senate-election-winner",
             event_title="Georgia Senate Election Winner",
             title="Will the Republicans win the Georgia Senate race in 2026?",
             outcome="Mike Collins (R)", bid=0.03, ask=0.031, last=0.03,
             **{**row, "clob_mid": 0.0305, "token_id": "tok"}),
    ], schema={**M._RAW_SCHEMA, "clob_mid": pl.Float64, "tag": pl.Utf8})
    return {"kalshi_markets": k, "polymarket_markets": p}


def test_to_objects_offline_loads_into_funnel():
    objs = M.to_objects(_raw_fixture())
    mk, sn = objs["Market"], objs["MarketSnapshot"]
    assert set(mk["market_id"]) == {"kalshi:CONTROLH-2026-D", "kalshi:SENATEMT-26-R", "polymarket:630693"}
    by = dict(zip(sn["market_id"], sn["p_dem"]))
    assert by["kalshi:CONTROLH-2026-D"] == pytest.approx(0.93)
    assert by["kalshi:SENATEMT-26-R"] == pytest.approx(0.09)        # 1 - mid(0.90, 0.92)
    assert by["polymarket:630693"] == pytest.approx(1 - 0.0305)     # CLOB mid, YES=R
    con = oms.connect(":memory:")
    assert funnel.write_batch(con, "Market", mk) == 3
    assert funnel.write_batch(con, "MarketSnapshot", sn) == 3
    um = M.unmapped(_raw_fixture())
    assert um["market_id"].to_list() == ["kalshi:SENATEMT-26-D"]


# ------------------------------------------------------------ live smoke

@pytest.mark.skipif(os.environ.get("MARKETS_OFFLINE") == "1", reason="offline")
def test_live_fetch_snapshot_into_memory_db():
    t0 = time.time()
    try:
        raw = M.fetch(tag="test", cache=True)
    except httpx.HTTPError as e:
        pytest.skip(f"network unavailable: {e!r}")
    objs = M.to_objects(raw)
    mk, sn = objs["Market"], objs["MarketSnapshot"]
    con = oms.connect(":memory:")
    funnel.write_batch(con, "Market", mk)
    funnel.write_batch(con, "MarketSnapshot", sn)
    got = dict(con.execute(
        "SELECT market_id, p_dem FROM obj_MarketSnapshot WHERE market_id IN "
        "('kalshi:CONTROLH-2026-D', 'kalshi:CONTROLS-2026-D')").fetchall())
    assert len(got) == 2 and all(0 < v < 1 for v in got.values())
    races = set(mk["race_id"])
    assert {"CONTROL-HOUSE", "CONTROL-SENATE", "SEN-GA", "SEN-OH-S"} <= races
    assert set(mk["venue"]) == {"kalshi", "polymarket"}
    assert sn["p_dem"].is_between(0, 1).all()
    assert time.time() - t0 < 300

    snap = M.snapshot("lock")
    assert snap.height >= 0.95 * mk.height and set(snap["tag"]) == {"lock"}
    funnel.write_batch(con, "MarketSnapshot", snap)
    n = con.execute("SELECT count(*) FROM obj_MarketSnapshot WHERE tag = 'lock'").fetchone()[0]
    assert n == snap.height
