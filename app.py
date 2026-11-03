"""Election Twin dashboard. Reads only through the Object Set Service.

    uv run streamlit run app.py
"""
from __future__ import annotations

import time

import altair as alt
import duckdb
import polars as pl
import streamlit as st

from ontology import oms
from ontology.objects import ObjectSet

LOCKED, LIVE = "pregame-v1-LOCKED", "live-v1"
SERIES = {"CONTROL-HOUSE": "#2a78d6", "CONTROL-SENATE": "#eb6834"}  # validated categorical slots 1-2

st.set_page_config(page_title="Election Twin", layout="wide")


def connect():
    # The live loop holds the write lock only briefly per cycle, so retry.
    for _ in range(20):
        try:
            return duckdb.connect(str(oms.DEFAULT_DB), read_only=True)
        except duckdb.IOException:
            time.sleep(0.5)
    st.error("database busy; refresh in a moment")
    st.stop()


con = connect()


def q(sql: str, params: list | None = None) -> pl.DataFrame:
    return pl.from_arrow(con.execute(sql, params or []).to_arrow_table())


def latest(version: str) -> pl.DataFrame:
    return q("""SELECT * FROM merged_Prediction WHERE model_version = ? AND scenario_id IS NULL
                QUALIFY row_number() OVER (PARTITION BY race_id ORDER BY ts DESC) = 1""", [version])


locked, live = latest(LOCKED), latest(LIVE)
current = live if live.height else locked if locked.height else latest("pregame-v1")
markets = q("""SELECT m.race_id, avg(s.p_dem) AS p_market FROM merged_MarketSnapshot s
               JOIN merged_Market m USING (market_id)
               WHERE s.ts = (SELECT max(ts) FROM merged_MarketSnapshot s2 WHERE s2.market_id = s.market_id)
               GROUP BY m.race_id""")

st.title("Election Twin: 2026 midterms")
st.caption(("Live: NC, FL and GA county feeds update every race through the inferred national swing. "
            "Other states have no direct feed." if live.height else "Pregame forecast.")
           + f"  Locked forecast: {'yes' if locked.height else 'not yet'}.")

cols = st.columns(4)
for i, rid in enumerate(["CONTROL-HOUSE", "CONTROL-SENATE"]):
    cur = current.filter(pl.col("race_id") == rid)
    lk = locked.filter(pl.col("race_id") == rid)
    mk = markets.filter(pl.col("race_id") == rid)
    if cur.height:
        r = cur.row(0, named=True)
        delta = f"{100 * (r['p_dem'] - lk['p_dem'][0]):+.1f} pts vs lock" if lk.height and live.height else None
        cols[2 * i].metric(f"P(Dem {rid.split('-')[1].title()})", f"{100 * r['p_dem']:.1f}%", delta)
        cols[2 * i + 1].metric(f"Dem seats, 80% interval", f"{r['mean_margin']:.0f}  [{r['p10']:.0f}–{r['p90']:.0f}]",
                               f"market {100 * mk['p_market'][0]:.0f}%" if mk.height else None, delta_color="off")

hist = q("""SELECT race_id, ts, p_dem FROM merged_Prediction
            WHERE race_id LIKE 'CONTROL-%' AND model_version IN (?, ?) AND scenario_id IS NULL ORDER BY ts""",
         [LOCKED, LIVE])
if hist.height > 1:
    hover = alt.selection_point(fields=["ts"], nearest=True, on="pointerover", empty=False)
    base = alt.Chart(hist.to_pandas()).encode(
        x=alt.X("ts:T", title=None),
        y=alt.Y("p_dem:Q", title="P(Dem control)", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("race_id:N", title=None, scale=alt.Scale(domain=list(SERIES), range=list(SERIES.values()))),
    )
    lines = base.mark_line(strokeWidth=2)
    points = base.mark_circle(size=64).encode(opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                                              tooltip=["race_id", alt.Tooltip("ts:T", format="%H:%M"),
                                                       alt.Tooltip("p_dem:Q", format=".1%")]).add_params(hover)
    rule = alt.Chart(pl.DataFrame({"y": [0.5]}).to_pandas()).mark_rule(color="#a3a29b", strokeDash=[4, 4]).encode(y="y")
    st.altair_chart((lines + points + rule).properties(height=260), width="stretch")

st.subheader("Races")
chamber = st.segmented_control("Chamber", ["senate", "house", "governor"], default="senate")
races = ObjectSet(con, "Race").filter(chamber=chamber).df().select("race_id", "rating", "incumbent_party", "pres24_margin")
table = (races.join(current.select("race_id", "p_dem", "mean_margin", "p10", "p90"), on="race_id", how="left")
         .join(locked.select("race_id", pl.col("p_dem").alias("p_locked")), on="race_id", how="left")
         .join(markets, on="race_id", how="left")
         .with_columns((pl.col("p_dem") - 0.5).abs().alias("_c")).sort("_c").drop("_c"))
st.dataframe(table, width="stretch", height=360, column_config={
    c: st.column_config.NumberColumn(format="percent") for c in ("p_dem", "p_locked", "p_market")})

st.subheader("Object drill-down")
rid = st.selectbox("Race", table["race_id"].to_list())
if rid:
    race = ObjectSet(con, "Race").filter(race_id=rid)
    a, b = st.columns(2)
    a.write("**Race**")
    a.json(race.get(race_id=rid), expanded=False)
    a.write("**Model inputs (locked run lineage)**")
    a.dataframe(q("""SELECT i.* EXCLUDE (run_id), r.inputs_hash FROM model_inputs i JOIN model_runs r USING (run_id)
                     WHERE i.race_id = ? AND r.model_version = ? ORDER BY r.ts DESC LIMIT 1""", [rid, LOCKED]))
    b.write("**Linked Polls**")
    b.dataframe(race.links("Poll").df(order_by="end_date DESC"), height=200)
    b.write("**Linked Markets**")
    b.dataframe(race.links("Market").df())
    st.write("**Prediction history**")
    st.dataframe(race.links("Prediction").df(order_by="ts DESC").head(50), height=200)
    vb = race.links("VoteBatch").df(order_by="ts DESC")
    if vb.height:
        st.write("**Vote batches**")
        st.dataframe(vb.head(200), height=200)

st.subheader("Scenarios")
st.caption('Run with `uv run python cli.py scenario "polls miss by R+3"` or `"flip Georgia"`.')
scn = q("""SELECT s.scenario_id, s.description, p.race_id, p.p_dem, p.mean_margin, p.p10, p.p90
           FROM merged_Scenario s JOIN merged_Prediction p USING (scenario_id)
           WHERE p.race_id LIKE 'CONTROL-%' ORDER BY s.created_at DESC""")
st.dataframe(scn, width="stretch")
