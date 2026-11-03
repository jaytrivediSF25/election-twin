from datetime import datetime

import pytest

from ontology import funnel, oms
from ontology.funnel import OntologyError
from ontology.objects import ObjectSet


@pytest.fixture
def con():
    c = oms.connect(":memory:")
    funnel.write_batch(c, "Race", [
        {"race_id": "SEN-GA", "chamber": "senate", "state": "GA", "rating": "Tossup"},
        {"race_id": "SEN-ME", "chamber": "senate", "state": "ME", "rating": "Lean R"},
    ])
    funnel.write_batch(c, "Poll", [
        {"poll_id": "p1", "race_id": "SEN-GA", "pollster_id": "x", "margin": 2.0, "weight": 1.0, "excluded": False},
        {"poll_id": "p2", "race_id": "SEN-GA", "pollster_id": "y", "margin": -1.0, "weight": 1.0, "excluded": False},
        {"poll_id": "p3", "race_id": "SEN-ME", "pollster_id": "x", "margin": 4.0, "weight": 1.0, "excluded": False},
    ])
    funnel.write_batch(c, "County", [{"fips": "13121", "state": "GA"}, {"fips": "13089", "state": "GA"}])
    funnel.write_batch(c, "District", [{"district_id": "GA-05"}, {"district_id": "GA-04"}])
    funnel.write_batch(c, "CountyDistrict", [
        {"fips": "13121", "district_id": "GA-05", "weight": 0.6},
        {"fips": "13089", "district_id": "GA-04", "weight": 1.0},
    ])
    return c


def test_rejects_unknown_property(con):
    with pytest.raises(OntologyError):
        funnel.write_batch(con, "Race", [{"race_id": "X", "bogus": 1}])


def test_rejects_null_and_duplicate_pk(con):
    with pytest.raises(OntologyError):
        funnel.write_batch(con, "Race", [{"race_id": None}])
    with pytest.raises(OntologyError):
        funnel.write_batch(con, "Race", [{"race_id": "A"}, {"race_id": "A"}])


def test_upsert_replaces_by_pk(con):
    funnel.write_batch(con, "Race", [{"race_id": "SEN-GA", "chamber": "senate", "rating": "Lean D"}])
    assert ObjectSet(con, "Race").count() == 2
    assert ObjectSet(con, "Race").get(race_id="SEN-GA")["rating"] == "Lean D"


def test_edit_visible_immediately_in_merged(con):
    funnel.apply_action(con, "override_poll", pk={"poll_id": "p1"}, patch={"excluded": True})
    assert ObjectSet(con, "Poll").get(poll_id="p1")["excluded"] is True
    # base dataset untouched
    assert oms.one(con, "SELECT excluded FROM obj_Poll WHERE poll_id='p1'")[0] is False
    # edits survive re-ingest of source rows
    funnel.write_batch(con, "Poll", [{"poll_id": "p1", "race_id": "SEN-GA", "margin": 2.0, "excluded": False}])
    assert ObjectSet(con, "Poll").get(poll_id="p1")["excluded"] is True


def test_action_rejects_non_editable_and_missing(con):
    with pytest.raises(OntologyError):
        funnel.apply_action(con, "override_poll", pk={"poll_id": "p1"}, patch={"margin": 9})
    with pytest.raises(OntologyError):
        funnel.apply_action(con, "override_poll", pk={"poll_id": "nope"}, patch={"excluded": True})


def test_link_traversal(con):
    polls = ObjectSet(con, "Race").filter(race_id="SEN-GA").links("Poll")
    assert sorted(polls.df()["poll_id"].to_list()) == ["p1", "p2"]
    back = ObjectSet(con, "Poll").filter(pollster_id="x").links("Race")
    assert sorted(back.df()["race_id"].to_list()) == ["SEN-GA", "SEN-ME"]
    dists = ObjectSet(con, "County").filter(fips="13121").links("District")
    assert dists.df()["district_id"].to_list() == ["GA-05"]


def test_aggregate(con):
    agg = ObjectSet(con, "Poll").aggregate(by=["race_id"], m="avg(margin)", n="count(*)")
    assert agg.filter(agg["race_id"] == "SEN-GA")["m"][0] == pytest.approx(0.5)


def test_locked_version_refuses_writes(con):
    row = {"race_id": "SEN-GA", "model_version": "v1-LOCKED", "ts": datetime(2026, 11, 3), "p_dem": 0.5}
    funnel.write_batch(con, "Prediction", [row])
    funnel.lock_version(con, "v1-LOCKED", "a", "b")
    with pytest.raises(OntologyError):
        funnel.write_batch(con, "Prediction", [{**row, "ts": datetime(2026, 11, 4)}])
    with pytest.raises(OntologyError):
        funnel.lock_version(con, "v1-LOCKED", "a", "b")


def test_create_action_writes_object_and_audit(con):
    funnel.apply_action(con, "create_scenario", row={"scenario_id": "s1", "description": "R+3 miss",
                                                    "overrides": '{"national_shift": -3}'})
    assert ObjectSet(con, "Scenario").count() == 1
    assert oms.one(con, "SELECT count(*) FROM edits WHERE action='create_scenario'")[0] == 1
