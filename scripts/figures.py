"""README figures from the locked forecast and the 2024 replays.

    uv run python scripts/figures.py        # -> docs/img/*-{light,dark}.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ingest import live_ga, live_nc  # noqa: E402
from models import live, pregame  # noqa: E402
from models.montecarlo import SimParams, simulate  # noqa: E402
from ontology import oms  # noqa: E402

OUT = ROOT / "docs" / "img"
LOCKED = "pregame-v1-LOCKED"
THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e4e3de",
                  dem="#2a78d6", rep="#e34948", s1="#2a78d6", s2="#eb6834", s3="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", grid="#383835",
                 dem="#3987e5", rep="#e66767", s1="#3987e5", s2="#d95926", s3="#199e70"),
}


def style(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 11,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 12,
        "legend.frameon": False, "xtick.major.size": 0, "ytick.major.size": 0,
    })


def save(fig, name: str, mode: str) -> None:
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=180, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def seats(sims, preds: pl.DataFrame, t: dict, mode: str) -> None:
    p = dict(zip(preds["race_id"], preds["p_dem"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, draws, need, name, key in [(axes[0], sims.house_d, 218, "House", "CONTROL-HOUSE"),
                                       (axes[1], sims.senate_d, 51, "Senate", "CONTROL-SENATE")]:
        vals, counts = np.unique(draws, return_counts=True)
        share = counts / counts.sum()
        ax.bar(vals, share * 100, width=0.82, color=[t["dem"] if v >= need else t["rep"] for v in vals],
               linewidth=0)
        ax.axvline(need - 0.5, color=t["ink2"], linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_ylim(0, share.max() * 100 * 1.22)
        ax.text(need - 0.5, ax.get_ylim()[1], f"{need} for majority ", color=t["ink2"], va="top", ha="right",
                fontsize=10)
        lo, hi = np.percentile(draws, [10, 90])
        x0, x1 = np.percentile(draws, [0.05, 99.95])
        ax.set_xlim(min(x0, need) - 3, max(x1, need) + 3)
        ax.set_title(f"{name}: Dem control {100 * p[key]:.0f}%", color=t["ink"])
        ax.set_xlabel(f"Democratic seats  (median {np.median(draws):.0f}, 80% interval {lo:.0f}–{hi:.0f})")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:g}%")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share of simulations")
    fig.tight_layout(w_pad=3)
    save(fig, "seats", mode)


def senate_vs_market(rows: pl.DataFrame, t: dict, mode: str) -> None:
    rows = rows.sort("p_model")
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * rows.height + 1.4))
    y = np.arange(rows.height)
    for i, r in enumerate(rows.iter_rows(named=True)):
        ax.plot([r["p_model"] * 100, r["p_market"] * 100], [i, i], color=t["grid"], linewidth=2.5, zorder=1,
                solid_capstyle="round")
    ax.scatter(rows["p_market"] * 100, y, s=70, facecolor=t["surface"], edgecolor=t["s2"], linewidth=2,
               zorder=2, label="Prediction markets (Kalshi/Polymarket, pre-close)")
    ax.scatter(rows["p_model"] * 100, y, s=70, color=t["s1"], edgecolor=t["surface"], linewidth=2, zorder=3,
               label="Locked model")
    ax.axvline(50, color=t["ink2"], linestyle=(0, (4, 3)), linewidth=1)
    ax.set_yticks(y, [r.replace("SEN-", "").replace("-S", " (special)") for r in rows["race_id"]])
    ax.set_xlim(0, 100)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_xlabel("P(Democrat wins)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Senate: locked model vs markets", color=t["ink"])
    ax.legend(loc="lower right", fontsize=9.5, labelcolor=t["ink2"])
    fig.tight_layout()
    save(fig, "senate-vs-market", mode)


def model_vs_market(rows: pl.DataFrame, t: dict, mode: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 6))
    ax.plot([0, 100], [0, 100], color=t["ink2"], linestyle=(0, (4, 3)), linewidth=1)
    for ch, color in [("house", t["s1"]), ("senate", t["s2"]), ("governor", t["s3"])]:
        g = rows.filter(pl.col("chamber") == ch)
        mad = (g["p_model"] - g["p_market"]).abs().mean() * 100
        ax.scatter(g["p_market"] * 100, g["p_model"] * 100, s=44, color=color, edgecolor=t["surface"],
                   linewidth=1.5, alpha=0.9, label=f"{ch.title()} ({g.height} races, mean gap {mad:.1f} pts)")
    ax.set_xlim(-2, 102), ax.set_ylim(-2, 102)
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
    ax.set_xlabel("Market P(Democrat)")
    ax.set_ylabel("Model P(Democrat)")
    ax.set_title("Every race with a market", color=t["ink"])
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=t["ink2"])
    fig.tight_layout()
    save(fig, "model-vs-market", mode)


def replay_curves(county, bm) -> dict[str, dict[str, list[tuple[float, float]]]]:
    out = {}
    cfg = live.LiveConfig(turnout_ratio=1.0)
    for mod, st, actual in [(live_nc, "NC", -3.22), (live_ga, "GA", -2.19)]:
        out[st] = {"matched": [], "totals": []}
        for chunk in mod.replay_2024(14):
            rows = (chunk.filter(pl.col("race_id") == f"PRES-{st}").sort("ts")
                    .group_by(["race_id", "fips", "vote_type"]).last()
                    .with_columns((pl.col("dem") + pl.col("rep") + pl.col("other").fill_null(0)).alias("votes")))
            m = live.race_projection(rows, county, bm, cfg)
            tot = (rows.group_by("race_id", "fips").agg(pl.col("dem", "rep", "other", "votes").sum())
                   .with_columns(pl.lit("total").alias("vote_type")))
            n = live.race_projection(tot, county, None, cfg)
            out[st]["matched"].append((100 * m["frac"], m["projected"] - actual))
            out[st]["totals"].append((100 * n["frac"], n["projected"] - actual))
    return out


def replay(curves: dict, t: dict, mode: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (st, c) in zip(axes, curves.items()):
        ax.axhline(0, color=t["ink2"], linewidth=1)
        for key, color, label in [("totals", t["s2"], "County totals vs 2024 totals"),
                                  ("matched", t["s1"], "Matched by vote type (this model)")]:
            x, y = zip(*c[key])
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5, label=label)
        ax.set_title(f"{st} 2024 replay", color=t["ink"])
        ax.set_xlabel("share of expected vote counted")
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0f}%")
        ax.grid(axis="x", visible=False)
        ax.legend(loc="upper right", fontsize=9.5, labelcolor=t["ink2"])
    axes[0].set_ylabel("projected − final margin (pts)")
    fig.tight_layout(w_pad=3)
    save(fig, "replay", mode)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(oms.DEFAULT_DB), read_only=True)
    _, inputs, meta, params = pregame.latest_run(con, LOCKED)
    p = SimParams(**{k: v for k, v in params.items() if k in SimParams.__dataclass_fields__})
    sims = simulate(inputs, p)
    preds = sims.summarize()
    markets = pl.from_arrow(con.execute("""
        WITH pred AS (SELECT race_id, p_dem FROM merged_Prediction WHERE model_version = ? AND scenario_id IS NULL)
        SELECT r.race_id, r.chamber, pred.p_dem AS p_model, avg(s.p_dem) AS p_market
        FROM merged_MarketSnapshot s JOIN merged_Market m USING (market_id)
        JOIN pred ON pred.race_id = m.race_id JOIN merged_Race r ON r.race_id = m.race_id
        WHERE s.tag = 'lock' AND s.p_dem IS NOT NULL GROUP BY ALL""", [LOCKED]).to_arrow_table())
    sen = markets.filter((pl.col("chamber") == "senate")
                         & ((pl.col("p_model").is_between(0.03, 0.97)) | pl.col("p_market").is_between(0.03, 0.97)))
    curves = replay_curves(*live.baselines(con))
    for mode, t in THEMES.items():
        style(t)
        seats(sims, preds, t, mode)
        senate_vs_market(sen, t, mode)
        model_vs_market(markets, t, mode)
        replay(curves, t, mode)
    print(f"wrote {len(list(OUT.glob('*.png')))} images to {OUT}")


if __name__ == "__main__":
    main()
