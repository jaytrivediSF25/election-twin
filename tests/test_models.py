from datetime import date

import numpy as np
import polars as pl
import pytest

from models import pregame
from models.montecarlo import SimParams, simulate
from ontology import funnel, oms
from ontology.funnel import OntologyError
from ontology.objects import ObjectSet


def _races(n_house=10):
    rows = [{"race_id": f"HOUSE-ZZ-{i:02d}", "chamber": "house", "state": "ZZ", "mean": float(i - 5),
             "sd_race": 5.0} for i in range(n_house)]
    rows += [{"race_id": "SEN-ZZ", "chamber": "senate", "state": "ZZ", "mean": 1.0, "sd_race": 4.0}]
    return pl.DataFrame(rows)


def test_zero_sd_is_deterministic():
    p = SimParams(n_sims=200, sd_national=0, sd_state=0, t_df=np.inf)
    sims = simulate(_races().with_columns(pl.lit(0.0).alias("sd_race")), p)
    s = sims.summarize()
    got = dict(zip(s["race_id"], s["p_dem"]))
    assert got["HOUSE-ZZ-06"] == 1.0 and got["HOUSE-ZZ-04"] == 0.0 and got["HOUSE-ZZ-05"] == 0.0


def test_seat_totals_match_race_wins():
    sims = simulate(_races(), SimParams(n_sims=500))
    house = sims.margins[:, sims.chambers == "house"] > 0
    assert (house.sum(axis=1) == sims.house_d).all()


def test_senate_counts_holdovers():
    sims = simulate(_races(), SimParams(n_sims=500, senate_holdover_d=50))
    assert set(np.unique(sims.senate_d)) <= {50, 51}
    ctrl = sims.summarize().filter(pl.col("race_id") == "CONTROL-SENATE")
    assert ctrl["p_dem"][0] == pytest.approx((sims.senate_d >= 51).mean())


def test_force_conditions_correlated_races():
    sims = simulate(_races(), SimParams(n_sims=5000))
    base = sims.summarize()
    cond = sims.summarize(sims.mask_for({"SEN-ZZ": "R"}))
    b = dict(zip(base["race_id"], base["p_dem"]))
    c = dict(zip(cond["race_id"], cond["p_dem"]))
    assert c["SEN-ZZ"] == 0.0
    assert c["HOUSE-ZZ-05"] < b["HOUSE-ZZ-05"]  # shared national/state shocks move with it


@pytest.fixture
def con():
    c = oms.connect(":memory:")
    funnel.write_batch(c, "County", [
        {"fips": "99001", "state": "ZZ", "pres24_dem": 480, "pres24_rep": 520, "pres24_total": 1000, "pres24_margin": -4.0},
    ])
    funnel.write_batch(c, "Race", [
        {"race_id": "HOUSE-ZZ-01", "chamber": "house", "state": "ZZ", "pres24_margin": 0.0, "open_seat": True, "rating": "Tossup"},
        {"race_id": "HOUSE-ZZ-02", "chamber": "house", "state": "ZZ", "pres24_margin": -30.0, "incumbent_party": "R",
         "open_seat": False, "rating": "Safe R"},
        {"race_id": "SEN-ZZ", "chamber": "senate", "state": "ZZ", "open_seat": True, "rating": "Tossup"},
    ])
    funnel.write_batch(c, "PollAverage", [{"race_id": "GENERIC", "source": "t", "margin": 3.52, "n_polls": 20}])
    return c


def test_no_polls_collapses_to_uniform_swing(con):
    inputs, meta = pregame.build_inputs(con)
    row = inputs.filter(pl.col("race_id") == "HOUSE-ZZ-01").row(0, named=True)
    assert meta["swing"] == pytest.approx(5.0)
    assert row["mean"] == pytest.approx(0.6 * 5.0)          # tossup rating pulls 40% toward 0
    sen = inputs.filter(pl.col("race_id") == "SEN-ZZ").row(0, named=True)
    assert sen["fundamental"] == pytest.approx(0.6 * (-4.0 + 5.0))  # state pres from counties


def test_polls_pull_mean(con):
    funnel.write_batch(con, "Poll", [
        {"poll_id": f"p{i}", "race_id": "SEN-ZZ", "end_date": date(2026, 10, 30), "sample_size": 800,
         "population": "lv", "margin": 8.0, "weight": 1.0, "excluded": False} for i in range(4)
    ])
    inputs, _ = pregame.build_inputs(con)
    sen = inputs.filter(pl.col("race_id") == "SEN-ZZ").row(0, named=True)
    assert sen["n_polls"] == 4 and 5.0 < sen["mean"] < 8.0 and sen["sd_race"] == 3.5


def test_run_writes_predictions_and_lock_blocks(con):
    inputs, meta = pregame.build_inputs(con)
    preds, _, run_id = pregame.run(con, "test-LOCKED", inputs, meta, SimParams(n_sims=1000))
    assert ObjectSet(con, "Prediction").filter(model_version="test-LOCKED").count() == preds.height
    rid, inputs2, meta2, _ = pregame.latest_run(con, "test-LOCKED")
    assert rid == run_id and inputs2.height == inputs.height
    funnel.lock_version(con, "test-LOCKED", "x", "y")
    with pytest.raises(OntologyError):
        pregame.run(con, "test-LOCKED", inputs, meta, SimParams(n_sims=1000))


def test_live_update_moves_national_and_observed_race(con):
    from datetime import datetime
    from models import live
    funnel.write_batch(con, "County", [
        {"fips": "99001", "state": "ZZ", "pres24_dem": 480, "pres24_rep": 520, "pres24_total": 1000, "pres24_margin": -4.0},
        {"fips": "98001", "state": "YY", "pres24_dem": 500, "pres24_rep": 500, "pres24_total": 1000, "pres24_margin": 0.0},
    ])
    funnel.write_batch(con, "Race", [{"race_id": "SEN-YY", "chamber": "senate", "state": "YY", "open_seat": True,
                                      "rating": "Tossup"}])
    inputs, meta = pregame.build_inputs(con)
    pregame.run(con, live.PREGAME_VERSION, inputs, meta, SimParams(n_sims=2000))
    pre_sen = inputs.filter(pl.col("race_id") == "SEN-ZZ")["mean"][0]
    # ZZ county 90% in, D+12 vs 2024's R+4 -> big D overperformance
    funnel.write_stream(con, "VoteBatch", [{"batch_id": "b1", "race_id": "SEN-ZZ", "fips": "99001",
                        "vote_type": "total", "dem": 360, "rep": 288, "other": 0, "ts": datetime(2026, 11, 4, 1)}])
    preds, lmeta = live.run_once(con, live.LiveConfig(min_votes=10))
    assert lmeta["nat_posterior_mu"] > 1.0
    assert lmeta["nat_posterior_sd"] < 3.0
    p = dict(zip(preds["race_id"], preds["mean_margin"]))
    assert p["SEN-ZZ"] > pre_sen + 5
    pre_yy = inputs.filter(pl.col("race_id") == "SEN-YY")["mean"][0]
    assert p["SEN-YY"] > pre_yy  # unobserved state moves via national posterior


def test_scenario_parse_and_run(con):
    from models import scenarios
    fav = {"SEN-GA": 0.6, "SEN-ME": 0.3}
    assert scenarios.parse("polls miss by R+3", fav) == {"national_shift": -3.0}
    assert scenarios.parse("flip Georgia", fav) == {"force": {"SEN-GA": "R"}}
    assert scenarios.parse("D win SEN-ME", fav) == {"force": {"SEN-ME": "D"}}
    inputs, meta = pregame.build_inputs(con)
    base, _, _ = pregame.run(con, "pregame-v1", inputs, meta, SimParams(n_sims=2000))
    out = scenarios.run(con, "polls miss by R+3", base="pregame-v1")
    b = dict(zip(base["race_id"], base["mean_margin"]))
    o = dict(zip(out["race_id"], out["mean_margin"]))
    assert o["SEN-ZZ"] == pytest.approx(b["SEN-ZZ"] - 3, abs=0.3)
    assert ObjectSet(con, "Scenario").count() == 1
