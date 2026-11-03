"""Races ingest: parses cached sources under data/raw/races/ (fetch downloads them if missing)
and loads the objects into an in-memory DB through the funnel."""
import polars as pl
import pytest

from ingest import races
from ontology import funnel, oms

ALLOWED = set(races.RATINGS)


@pytest.fixture(scope="module")
def raw():
    return races.fetch()


@pytest.fixture(scope="module")
def loaded(raw):
    objs = races.to_objects(raw)
    con = oms.connect(":memory:")
    for name, df in objs.items():
        funnel.write_batch(con, name, df, mode="replace", source="races")
    funnel.write_raw(con, "senate_holdovers", raw["senate_holdovers"], "races", races.SOURCES["senate_holdovers"])
    return con, objs


def count(con, sql):
    return con.execute(sql).fetchone()[0]


def test_house_435_on_2026_lines(loaded):
    con, _ = loaded
    assert count(con, "SELECT count(*) FROM obj_Race WHERE chamber='house'") == 435
    assert count(con, "SELECT count(*) FROM obj_District") == 435
    assert count(con, "SELECT count(*) FROM obj_Race WHERE chamber='house' AND pres24_margin IS NULL") == 0
    assert count(con, "SELECT count(*) FROM obj_District WHERE pres24_margin IS NULL") == 0
    # at-large and zero-padded ids
    for rid in ("HOUSE-AK-AL", "HOUSE-WY-AL", "HOUSE-PA-07", "HOUSE-CA-52", "HOUSE-TX-38"):
        assert count(con, f"SELECT count(*) FROM obj_Race WHERE race_id='{rid}'") == 1


def test_senate_and_governor_counts(loaded):
    con, _ = loaded
    n_sen = count(con, "SELECT count(*) FROM obj_Race WHERE chamber='senate'")
    assert 33 <= n_sen <= 37
    assert count(con, "SELECT count(*) FROM obj_Race WHERE race_id IN ('SEN-OH-S','SEN-FL-S') AND special") == 2
    assert count(con, "SELECT count(*) FROM obj_Race WHERE chamber='governor'") == 36
    assert count(con, "SELECT count(*) FROM obj_Race WHERE chamber<>'house' AND pres24_margin IS NULL") == 0


def test_seats_up_plus_holdovers_is_100(loaded):
    con, _ = loaded
    up = count(con, "SELECT count(*) FROM obj_Race WHERE chamber='senate'")
    hold = count(con, "SELECT sum(seats) FROM raw_senate_holdovers")
    assert set(r[0] for r in con.execute("SELECT party FROM raw_senate_holdovers").fetchall()) <= {"D", "R"}
    assert up + hold == 100


def test_every_rating_allowed(loaded):
    con, _ = loaded
    ratings = {r[0] for r in con.execute("SELECT DISTINCT rating FROM obj_Race").fetchall()}
    assert None not in ratings
    assert ratings <= ALLOWED


def test_margin_sign_convention(loaded):
    # D minus R: Trump-won states negative, Harris-won positive.
    con, _ = loaded
    m = dict(con.execute("SELECT race_id, pres24_margin FROM obj_Race WHERE race_id IN ('SEN-WY','GOV-CA','SEN-GA')").fetchall())
    assert m["SEN-WY"] < -30 and m["GOV-CA"] > 15 and -5 < m["SEN-GA"] < 0


def test_candidates_link_to_races(loaded):
    con, objs = loaded
    assert count(con, "SELECT count(*) FROM obj_Candidate c LEFT JOIN obj_Race r USING (race_id) WHERE r.race_id IS NULL") == 0
    statewide = objs["Candidate"].filter(~pl.col("race_id").str.starts_with("HOUSE"))
    # one D-side and one R nominee per Senate/governor race
    assert statewide.group_by("race_id").len()["len"].max() == 2
    ne = statewide.filter(pl.col("race_id") == "SEN-NE")
    assert set(ne["party"]) == {"R", "I"}


def test_normalize_rating():
    assert races.normalize_rating("Solid R") == "Safe R"
    assert races.normalize_rating("Tilt D (flip)") == "Lean D"
    assert races.normalize_rating("Toss-up") == "Tossup"
    assert races.normalize_rating("TBD") is None
    assert races.derive_rating(-1.0) == "Tossup"
    assert races.derive_rating(10.0) == "Likely D"
    assert races.derive_rating(-20.0) == "Safe R"
