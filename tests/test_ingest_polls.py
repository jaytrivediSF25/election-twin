"""Polls ingest -> in-memory object store. Uses cached downloads under data/raw/polls when present
(fetch(refresh=False)); downloads them otherwise."""
import re
from datetime import date

import polars as pl
import pytest

from ingest import polls
from ontology import funnel, oms

RACE_ID = re.compile(r"^(GENERIC|SEN-[A-Z]{2}(-S)?|GOV-[A-Z]{2}|HOUSE-[A-Z]{2}-(\d{2}|AL))$")


@pytest.fixture(scope="module")
def objs():
    raw = polls.fetch(refresh=False)
    assert "votehub_polls" in raw or "sb_generic_polls" in raw, polls.FAILED
    return polls.to_objects(raw)


@pytest.fixture(scope="module")
def con(objs):
    c = oms.connect(":memory:")
    for name in ("Pollster", "Poll", "PollAverage"):
        funnel.write_batch(c, name, objs[name], source="ingest.polls")
    return c


def test_generic_average_exists(con):
    rows = con.execute("SELECT source, margin, as_of FROM obj_PollAverage WHERE race_id = 'GENERIC'").fetchall()
    assert rows, "no GENERIC PollAverage"
    for _src, margin, as_of in rows:
        assert -40 <= margin <= 40
        assert as_of >= date(2026, 10, 1)


def test_margins_in_range_and_consistent(con):
    bad = con.execute("""
        SELECT count(*) FROM obj_Poll
        WHERE abs(margin - (dem_pct - rep_pct)) > 0.05
    """).fetchall()[0][0]
    assert bad == 0
    # every modelled poll is within +-40; blowouts are kept only as excluded rows
    out = con.execute("SELECT count(*) FROM obj_Poll WHERE NOT excluded AND abs(margin) > 40").fetchall()[0][0]
    assert out == 0
    assert con.execute("SELECT count(*) FROM obj_Poll WHERE excluded AND abs(margin) <= 40").fetchall()[0][0] == 0
    assert con.execute("SELECT max(abs(margin)) FROM obj_PollAverage").fetchall()[0][0] <= 40


def test_race_ids_follow_conventions(con):
    ids = [r[0] for r in con.execute(
        "SELECT race_id FROM obj_Poll UNION SELECT race_id FROM obj_PollAverage").fetchall()]
    assert ids
    assert [i for i in ids if not RACE_ID.match(i)] == []
    assert "SEN-OH-S" in ids and "SEN-OH" not in ids        # Ohio is the special
    assert "SEN-NE" in ids                                    # Osborn as D side


def test_poll_fields(con, objs):
    df = objs["Poll"]
    assert df.height > 100
    assert df["poll_id"].n_unique() == df.height
    assert set(df["population"].drop_nulls().unique()) <= {"lv", "rv", "a", "v"}
    sen_gov = df.filter(~pl.col("race_id").eq("GENERIC"))
    assert sen_gov["end_date"].min() >= date(2026, 7, 1)
    assert df.filter(pl.col("race_id") == "GENERIC")["end_date"].min() >= date(2025, 1, 1)
    assert df["weight"].eq(1.0).all()
    # every Poll's pollster exists
    orphans = con.execute("""
        SELECT count(*) FROM obj_Poll p LEFT JOIN obj_Pollster s USING (pollster_id)
        WHERE s.pollster_id IS NULL""").fetchall()[0][0]
    assert orphans == 0


def test_poll_id_stable():
    a = polls.poll_id("SEN-GA", "Emerson College", date(2026, 10, 1), 1000, 48.0, 46.0)
    b = polls.poll_id("SEN-GA", "Emerson  College", date(2026, 10, 1), 1000, 48, 46)
    assert a == b


def test_reingest_upserts_cleanly(con, objs):
    n = con.execute("SELECT count(*) FROM obj_Poll").fetchall()[0][0]
    funnel.write_batch(con, "Poll", objs["Poll"])
    assert con.execute("SELECT count(*) FROM obj_Poll").fetchall()[0][0] == n


def test_race_id_mapping():
    assert polls._race_id("us-senator", "2026 Ohio") == "SEN-OH-S"
    assert polls._race_id("us-senator", "2026 Georgia") == "SEN-GA"
    assert polls._race_id("us-senator", "2026 Texas Republican") is None
    assert polls._race_id("governor", "2026 Arizona") == "GOV-AZ"
    assert polls._race_id("us-representative", "2026 PA-07") == "HOUSE-PA-07"
    assert polls._race_id("us-representative", "2026 AK-01") == "HOUSE-AK-AL"
    assert polls._race_id("generic-ballot", "2026") == "GENERIC"
