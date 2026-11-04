"""twin: CLI for the election ontology."""
from __future__ import annotations

import importlib
import json
import subprocess
import time
from pathlib import Path

import polars as pl
import typer

from models import live, pregame, scenarios
from ontology import funnel, oms
from ontology.objects import ObjectSet

app = typer.Typer(no_args_is_help=True)
ROOT = Path(__file__).resolve().parent
LOCK_DIR = ROOT / "locked"
LOCKED = "pregame-v1-LOCKED"
pl.Config.set_tbl_rows(60)


def _con():
    return oms.connect()


def _show(preds: pl.DataFrame, top: int = 25) -> None:
    ctrl = preds.filter(pl.col("race_id").str.starts_with("CONTROL-"))
    for r in ctrl.iter_rows(named=True):
        typer.echo(f"{r['race_id']:>15}: P(D)={r['p_dem']:.3f}  D seats mean={r['mean_margin']:.1f} "
                   f"[{r['p10']:.0f}-{r['p90']:.0f}]")
    races = (preds.filter(~pl.col("race_id").str.starts_with("CONTROL-"))
             .with_columns((pl.col("p_dem") - 0.5).abs().alias("_c")).sort("_c").head(top).drop("_c"))
    typer.echo(races.select("race_id", "p_dem", "mean_margin", "p10", "p90"))


@app.command()
def build(modules: str = typer.Option("", help="comma-separated ingest modules (default all)")):
    """Fetch sources -> raw datasets -> objects."""
    con = _con()
    from pipelines.build_objects import build as run_build
    status = run_build(con, [m for m in modules.split(",") if m] or None)
    for k, v in status.items():
        typer.echo(f"{k:>12}: {v}")


@app.command()
def forecast(version: str = "pregame-v1", n_sims: int = 40_000):
    """Run the pregame model (unlocked)."""
    con = _con()
    inputs, meta = pregame.build_inputs(con)
    typer.echo(f"generic={meta['generic']:+.2f} swing={meta['swing']:+.2f} holdover_D={meta['senate_holdover_d']} "
               f"races={inputs.height}")
    preds, _, run_id = pregame.run(con, version, inputs, meta, pregame.SimParams(n_sims=n_sims))
    typer.echo(f"run {run_id}")
    _show(preds)


@app.command()
def lock(n_sims: int = 40_000, push: bool = True,
         from_run: bool = typer.Option(False, help="re-run the latest stored pregame-v1 run (same inputs, params, "
                                                   "seed) instead of fresh inputs; reuse its pre-run market snapshot")):
    """Final pregame run -> LOCKED predictions, market snapshot, hash, git commit+push."""
    con = _con()
    source = None
    if from_run:
        src_id, inputs, meta, params = pregame.latest_run(con, "pregame-v1")
        src_ts, src_out = oms.one(con, "SELECT ts, outputs_hash FROM model_runs WHERE run_id = ?", [src_id])
        sim = pregame.SimParams(**{k: v for k, v in params.items() if k in pregame.SimParams.__dataclass_fields__})
        source = {"run_id": src_id, "ts": src_ts, "outputs_hash": src_out}
    else:
        inputs, meta = pregame.build_inputs(con)
        sim = pregame.SimParams(n_sims=n_sims)
    preds, _, run_id = pregame.run(con, LOCKED, inputs, meta, sim)
    if from_run:
        # Benchmark = the last market pull taken before the source run, not prices that already see returns.
        snap = pl.from_arrow(con.execute(
            "SELECT * FROM merged_MarketSnapshot WHERE ts = (SELECT max(ts) FROM merged_MarketSnapshot WHERE ts <= ?)",
            [source["ts"]]).to_arrow_table())
        funnel.write_batch(con, "MarketSnapshot", snap.with_columns(pl.lit("lock").alias("tag")), source="lock")
        typer.echo(f"market snapshot: {snap.height} rows from {snap['ts'].max()}")
    else:
        try:
            snap = importlib.import_module("ingest.markets").snapshot("lock")
            funnel.write_batch(con, "MarketSnapshot", snap, source="lock")
            typer.echo(f"market snapshot: {snap.height} rows")
        except Exception as e:
            typer.echo(f"WARNING market snapshot failed: {e}")
    in_hash, out_hash = oms.one(con, "SELECT inputs_hash, outputs_hash FROM model_runs WHERE run_id = ?", [run_id])
    if source and source["outputs_hash"] != out_hash:
        raise typer.Exit(f"rerun outputs {out_hash} != source run {source['outputs_hash']}; not locking")
    funnel.lock_version(con, LOCKED, in_hash, out_hash)

    LOCK_DIR.mkdir(exist_ok=True)
    preds.sort("race_id").write_csv(LOCK_DIR / f"{LOCKED}.csv")
    inputs.sort("race_id").write_csv(LOCK_DIR / f"{LOCKED}.inputs.csv")
    (LOCK_DIR / "LOCK.json").write_text(json.dumps({
        "model_version": LOCKED, "run_id": run_id, "inputs_hash": in_hash, "outputs_hash": out_hash,
        "locked_at": pregame.utcnow(), "rerun_of": source, "meta": meta}, indent=2, default=str))
    _show(preds)
    typer.echo(f"LOCKED {run_id}\n inputs_hash={in_hash}\n outputs_hash={out_hash}")
    subprocess.run(["git", "add", "-A"], cwd=ROOT, check=True)
    subprocess.run(["git", "commit", "-q", "-m", f"LOCK {LOCKED} outputs={out_hash[:16]}\n\n"
                    "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"], cwd=ROOT, check=True)
    if push:
        subprocess.run(["git", "push", "-q"], cwd=ROOT, check=False)


@app.command()
def verify_lock():
    """Recompute the locked outputs hash from stored Prediction rows."""
    con = _con()
    stored = oms.one(con, "SELECT outputs_hash FROM locks WHERE model_version = ?", [LOCKED])
    preds = ObjectSet(con, "Prediction").filter(model_version=LOCKED).df()
    h = pregame.frame_hash(preds.select("race_id", "p_dem", "mean_margin", "p10", "p90", "n_sims"))
    typer.echo(f"stored={stored and stored[0]}\nrecomputed={h}\n{'OK' if stored and stored[0] == h else 'MISMATCH'}")


@app.command("live")
def live_cmd(interval: int = 90, once: bool = False, states: str = "nc,fl,ga"):
    """Poll NC/FL/GA feeds -> VoteBatches -> live predictions."""
    while True:
        con = _con()  # reopened each cycle so the dashboard can read between cycles
        for st in states.split(","):
            try:
                df = importlib.import_module(f"ingest.live_{st}").poll()
                n = funnel.write_stream(con, "VoteBatch", df, source=f"live_{st}") if df.height else 0
                typer.echo(f"{st}: {n} batches")
            except Exception as e:
                typer.echo(f"{st}: poll failed: {type(e).__name__}: {e}")
        try:
            preds, meta = live.run_once(con)
            typer.echo(f"national posterior {meta['nat_posterior_mu']:+.2f} ± {meta['nat_posterior_sd']:.2f}; "
                       f"observed races: {list(meta['observed'])}")
            _show(preds, top=15)
        except Exception as e:
            typer.echo(f"live model failed: {type(e).__name__}: {e}")
        con.close()
        if once:
            return
        time.sleep(interval)


@app.command()
def scenario(text: str, base: str = LOCKED):
    """e.g. twin scenario "polls miss by R+3"   |   twin scenario "flip Georgia" """
    con = _con()
    preds = scenarios.run(con, text, base=base)
    typer.echo(f"scenario {preds['scenario_id'][0]}")
    _show(preds)


@app.command()
def certify(race_id: str, winner: str, margin: float = typer.Option(None)):
    """Record a race outcome (certify_outcome action)."""
    con = _con()
    funnel.apply_action(con, "certify_outcome", actor="user", row={
        "race_id": race_id, "winner_party": winner.upper(), "margin": margin, "certified_at": pregame.utcnow()})


@app.command("score")
def score_cmd():
    import score as sc
    con = _con()
    df = sc.score(con)
    if df.is_empty():
        typer.echo("no outcomes certified yet")
        return
    typer.echo(sc.summary(df))


@app.command()
def show(type_name: str, where: str = "", limit: int = 20):
    """Inspect any object type through the Object Set Service."""
    os_ = ObjectSet(_con(), type_name)
    if where:
        os_ = os_.where(where)
    typer.echo(f"{os_.count()} objects")
    typer.echo(os_.df().head(limit))


if __name__ == "__main__":
    app()
