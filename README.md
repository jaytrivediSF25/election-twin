# Election Twin

A live ontology ("digital twin") of the 2026 midterms, modeled on Palantir Foundry's Ontology.
Polls, markets, past results and live vote counts all resolve to the same typed objects. A
correlated Monte Carlo turns them into race and chamber odds. The pregame forecast is locked
and hashed before polls close, and NC/FL/GA county results update the odds on election night.

The ontology makes the system traceable and easy to update. Accuracy comes from the data and models.

## Foundry mapping

| Foundry | Here |
|---|---|
| Ontology Metadata Service | `ontology/schema.yaml` + `ontology/oms.py` (object, link and action types; DDL; merged views) |
| Datasets | `raw_*` tables with `_source`, `_source_url`, `_fetched_at` lineage |
| Object Storage | `obj_<Type>` tables |
| Edits / merged dataset | append-only `edits` table; `merged_<Type>` = source rows overlaid with the latest edit |
| Funnel | `ontology/funnel.py`: `write_batch`, `write_stream`, `apply_action` (validated, audited) |
| Object Set Service | `ontology/objects.py`: `ObjectSet(con, "Race").filter(...).links("Poll").aggregate(...)` |
| Functions / closed loop | `models/*` read object sets and write `Prediction` objects back; `score.py` scores them against `Outcome`s |

## Usage

```bash
uv run python cli.py build                       # fetch all sources -> raw -> objects
uv run python cli.py forecast                    # pregame-v1 (unlocked)
uv run python cli.py lock                        # pregame-v1-LOCKED + market snapshot + hash + git commit/push
uv run python cli.py verify-lock
uv run python cli.py live --interval 90          # poll NC/FL/GA -> VoteBatch -> live-v1 predictions
uv run python cli.py scenario "polls miss by R+3"
uv run python cli.py scenario "flip Georgia"
uv run python cli.py certify SEN-GA D --margin 4.2
uv run python cli.py score                       # Brier/log loss: locked model vs markets at lock
uv run python cli.py show Poll --where "race_id = 'SEN-ME'"
uv run streamlit run app.py                      # dashboard
```

`TWIN_DB=/path/to.duckdb` points everything at a different database.

## Model

- **Pregame:** national swing = generic-ballot average − 2024 presidential margin (R+1.48).
  - **Fundamentals:** 2024 presidential margin on the 2026 lines + swing + incumbency. The result is clamped into the band for the race's Sabato rating, then pulled 40% toward that rating's center.
  - **Polled races:** blended with a recency-, sample-size- and population-weighted poll average (and site averages) in proportion to how many polls there are.
  - **Simulation:** 40k correlated simulations with Student-t (df=5) shocks: national σ=3, state σ=2, race σ=3.5–5.5. The Senate count includes holdovers, and 50–50 goes R (VP Vance breaks ties).
- **Live:** each county's margin is compared to its 2024 margin **for the same vote type**, because early and mail votes are reported first and lean D.
  - **Remaining vote:** projected per county and vote type, so uncounted election-day vote is projected at its own 2024 margin.
  - **National swing:** per-state deviations from the locked forecast update a posterior on the national swing, and every race is resimulated with it.
  - **Limit:** only NC, FL and GA have feeds, so races elsewhere move only through the inferred national swing. FL's 2024 baseline is totals only.
- **Markets** (Kalshi, Polymarket) are a benchmark only and are never model inputs.

## Known limits

- Ratings are Sabato's, taken from the Wikipedia ratings tables (the Crystal Ball site blocked scraping).
- No XGBoost model for House seats with few polls (cut for time). Those seats use uniform swing plus ratings.
- Turnout is assumed to be 72% of 2024's presidential vote in each county until counts exceed that.
