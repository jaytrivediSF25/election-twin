"""Correlated Monte Carlo over race margins.

margin[s, r] = mean[r] + nat_load[r]*N[s] + state_load[r]*S[s, state(r)] + E[s, r]
with Student-t shocks scaled to the given standard deviations.
Chamber totals are counted per simulation, so correlation flows into seat odds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

HOUSE_MAJORITY = 218
SENATE_MAJORITY_D = 51  # 50-50 goes to R: VP Vance breaks ties


@dataclass
class SimParams:
    n_sims: int = 40_000
    sd_national: float = 3.0
    sd_state: float = 2.0
    t_df: float = 5.0
    seed: int = 20261103
    national_shift: float = 0.0       # scenario: add to every race's national component
    senate_holdover_d: int = 0


@dataclass
class SimResult:
    race_ids: list[str]
    chambers: np.ndarray
    margins: np.ndarray                # (n_sims, n_races) float32
    national: np.ndarray               # (n_sims,)
    params: SimParams
    house_d: np.ndarray = field(init=False)
    senate_d: np.ndarray = field(init=False)

    def __post_init__(self):
        self.house_d = (self.margins[:, self.chambers == "house"] > 0).sum(axis=1)
        self.senate_d = (self.margins[:, self.chambers == "senate"] > 0).sum(axis=1) + self.params.senate_holdover_d

    def mask_for(self, force: dict[str, str]) -> np.ndarray:
        """Boolean mask of sims consistent with forced outcomes {race_id: 'D'|'R'}."""
        m = np.ones(self.margins.shape[0], dtype=bool)
        idx = {r: i for i, r in enumerate(self.race_ids)}
        for rid, party in force.items():
            if rid not in idx:
                raise KeyError(f"unknown race {rid}")
            col = self.margins[:, idx[rid]]
            m &= (col > 0) if party == "D" else (col <= 0)
        return m

    def summarize(self, mask: np.ndarray | None = None) -> pl.DataFrame:
        mg = self.margins if mask is None else self.margins[mask]
        if mg.shape[0] == 0:
            raise ValueError("no simulations satisfy the scenario conditions")
        q10, q90 = np.percentile(mg, [10, 90], axis=0)
        races = pl.DataFrame({
            "race_id": self.race_ids,
            "p_dem": (mg > 0).mean(axis=0).astype(float),
            "mean_margin": mg.mean(axis=0).astype(float),
            "p10": q10.astype(float),
            "p90": q90.astype(float),
        })
        hd = self.house_d if mask is None else self.house_d[mask]
        sd = self.senate_d if mask is None else self.senate_d[mask]
        rows = []
        if (self.chambers == "house").any():
            rows.append(_control_row("CONTROL-HOUSE", hd, HOUSE_MAJORITY))
        if (self.chambers == "senate").any():
            rows.append(_control_row("CONTROL-SENATE", sd, SENATE_MAJORITY_D))
        out = pl.concat([races, pl.DataFrame(rows, schema=races.schema)]) if rows else races
        return out.with_columns(pl.lit(int(mg.shape[0])).alias("n_sims"))


def _control_row(rid: str, seats: np.ndarray, majority: int) -> dict:
    # For control rows mean_margin/p10/p90 are D seat counts, p_dem is P(D majority).
    return {
        "race_id": rid,
        "p_dem": float((seats >= majority).mean()),
        "mean_margin": float(seats.mean()),
        "p10": float(np.percentile(seats, 10)),
        "p90": float(np.percentile(seats, 90)),
    }


def _t(rng: np.random.Generator, df: float, size, sd) -> np.ndarray:
    if df <= 2 or np.isinf(df):
        return rng.standard_normal(size) * sd
    return rng.standard_t(df, size) * (np.asarray(sd) * np.sqrt((df - 2) / df))


def simulate(races: pl.DataFrame, params: SimParams | None = None) -> SimResult:
    """races columns: race_id, chamber, state, mean, sd_race, [nat_load], [state_load]."""
    p = params or SimParams()
    rng = np.random.default_rng(p.seed)
    n, r = p.n_sims, races.height
    mean = races["mean"].to_numpy().astype(np.float32)
    sd_race = races["sd_race"].to_numpy().astype(np.float32)
    nat_load = races["nat_load"].to_numpy() if "nat_load" in races.columns else np.ones(r)
    st_load = races["state_load"].to_numpy() if "state_load" in races.columns else np.ones(r)

    states, state_idx = np.unique(races["state"].to_numpy().astype(str), return_inverse=True)
    national = _t(rng, p.t_df, n, p.sd_national).astype(np.float32) + np.float32(p.national_shift)
    state_shock = _t(rng, p.t_df, (n, len(states)), p.sd_state).astype(np.float32)

    margins = (
        mean[None, :]
        + national[:, None] * nat_load[None, :].astype(np.float32)
        + state_shock[:, state_idx] * st_load[None, :].astype(np.float32)
        + _t(rng, p.t_df, (n, r), sd_race[None, :]).astype(np.float32)
    )
    return SimResult(races["race_id"].to_list(), races["chamber"].to_numpy().astype(str), margins, national, p)
