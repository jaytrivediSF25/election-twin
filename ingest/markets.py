"""Prediction-market prices (Kalshi + Polymarket) for 2026 midterm races.

BENCHMARK ONLY: these rows are used to score the model (Brier) and are never
model inputs.

    fetch(tag="pull")   -> raw market rows with prices (also cached to data/raw/markets/)
    to_objects(raw)     -> {"Market": ..., "MarketSnapshot": ...}
    snapshot(tag)       -> MarketSnapshot rows, re-pricing only already-known markets
    unmapped(raw)       -> race-winner markets we could not map to (race_id, party)

race_id mapping is deterministic from Kalshi event tickers / Polymarket event
slugs; oddities live in the explicit override dicts below.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import polars as pl

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
SOURCES = {
    "kalshi_markets": f"{KALSHI}/events?series_ticker=<series>&status=open&with_nested_markets=true",
    "polymarket_markets": f"{GAMMA}/events?tag_slug=midterms&closed=false",
}
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "markets"

STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", "new hampshire": "NH",
    "new jersey": "NJ", "new mexico": "NM", "new york": "NY", "north carolina": "NC",
    "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", "oregon": "OR", "pennsylvania": "PA",
    "rhode island": "RI", "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
    "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA", "washington": "WA",
    "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
}
POSTAL = set(STATES.values())

# 2026 Senate specials (Vance / Rubio seats). Polymarket slugs don't flag them.
SPECIAL_SENATE = {"OH", "FL"}

# Market-level overrides: market_id -> yes_means ("D" | "R") or None to drop.
# README: an independent the Dems back is the "D" side (NE: Osborn).
MARKET_OVERRIDES: dict[str, str | None] = {
    "kalshi:SENATENE-26-DOSB": "D",     # Dan Osborn (I), D side
    "kalshi:SENATENE-26-D": None,       # Cindy Burbank (D) is not the D side
    "polymarket:634893": "D",           # NE: "Will an independent win" == Osborn
    "polymarket:634891": None,          # NE: "Will the Democrats win" (not the D side)
}
# Races where the non-R side is split / ambiguous: keep only YES=R markets so
# p_dem == P(not R). MT: Bodnar (I) vs Bankhead (D).
R_ONLY_RACES = {"SEN-MT"}
# Per-candidate markets with no party in the question (Polymarket): name -> party.
CANDIDATE_PARTY = {
    "dan sullivan": "R", "mary peltola": "D",                  # AK Senate
    "jonathan kreiss-tomkins": "D", "bernadette wilson": "R",  # AK Governor
    "xavier becerra": "D", "steve hilton": "R",                # CA Governor
}

_K_RACE = [
    (re.compile(r"^CONTROLH-2026$"), lambda m: "CONTROL-HOUSE"),
    (re.compile(r"^CONTROLS-2026$"), lambda m: "CONTROL-SENATE"),
    (re.compile(r"^(?:KX)?SENATE([A-Z]{2})(S?)-26(?:NOV)?$"),
     lambda m: _sen(m[1], bool(m[2]))),
    (re.compile(r"^(?:KX)?HOUSE(?:RACE-)?([A-Z]{2})(\d{1,2}|AL)-26$"),
     lambda m: _house(m[1], m[2])),
    (re.compile(r"^(?:KX)?GOVPARTY([A-Z]{2})-26$"), lambda m: _gov(m[1])),
]
_K_SERIES = re.compile(
    r"^(SENATE[A-Z]{2}S?|KXSENATE[A-Z]{2}|HOUSE[A-Z]{2}(\d+|AL)|KXHOUSE[A-Z]{2}\d+|KXHOUSERACE"
    r"|GOVPARTY[A-Z]{2}|KXGOVPARTY[A-Z]{2}|CONTROL[HS])$"
)
_K_PARTY_SUFFIX = {"D": "D", "DEM": "D", "R": "R", "GOP": "R"}

_P_RACE = [
    (re.compile(r"^which-party-will-win-the-senate-in-2026$"), lambda m: "CONTROL-SENATE"),
    (re.compile(r"^which-party-will-win-the-house-in-2026$"), lambda m: "CONTROL-HOUSE"),
    (re.compile(r"^([a-z-]+?)-senate-election-winner(?:-\d+)?$"),
     lambda m: _sen(STATES.get(m[1].replace("-", " ")))),
    (re.compile(r"^([a-z-]+?)-governor-(?:election-)?(?:winner|election)(?:-2026)?$"),
     lambda m: _gov(STATES.get(m[1].replace("-", " ")))),
    (re.compile(r"^([a-z]{2})-(\d{1,2}|al)-house-election-winner(?:-\d{8,})?$"),
     lambda m: _house(m[1].upper(), m[2].upper())),
    (re.compile(r"^which-party-will-win-the-house-race-for-the-([a-z]{2})-(\d{1,2}|al)-seat$"),
     lambda m: _house(m[1].upper(), m[2].upper())),
]
# Polymarket events that look like race winners but are not party outcomes.
_P_SKIP = re.compile(r"individual|lieutenant|county|loser|first-round|margin|turnout")
# Down-ballot / non-race events never reported as unmapped.
_P_NOT_RACE = (r"lieutenant|county|loser|first-round|margin|turnout|how-many|hold-more|young|older"
               r"|latino|white|black|attorney-general|secretary-of-state|state-senate|railroad|supreme-court")
_P_WINNER = re.compile(r"(election-winner|governor-winner|governor-election|which-party-will-win-the-)")


def _sen(st: str | None, special: bool = False) -> str | None:
    if st not in POSTAL:
        return None
    return f"SEN-{st}-S" if (special or st in SPECIAL_SENATE) else f"SEN-{st}"


def _house(st: str, dist: str) -> str | None:
    if st not in POSTAL:
        return None
    return f"HOUSE-{st}-AL" if dist == "AL" else f"HOUSE-{st}-{int(dist):02d}"


def _gov(st: str | None) -> str | None:
    return f"GOV-{st}" if st in POSTAL else None


def kalshi_race_id(event_ticker: str) -> str | None:
    for rx, fn in _K_RACE:
        m = rx.match(event_ticker)
        if m:
            return fn(m)
    return None


def kalshi_yes_means(ticker: str) -> str | None:
    if f"kalshi:{ticker}" in MARKET_OVERRIDES:
        return MARKET_OVERRIDES[f"kalshi:{ticker}"]
    return _K_PARTY_SUFFIX.get(ticker.rsplit("-", 1)[-1])


def polymarket_race_id(slug: str) -> str | None:
    if _P_SKIP.search(slug):
        return None
    for rx, fn in _P_RACE:
        m = rx.match(slug)
        if m:
            return fn(m)
    return None


def polymarket_yes_means(market_id: str, question: str, group_title: str = "") -> str | None:
    if f"polymarket:{market_id}" in MARKET_OVERRIDES:
        return MARKET_OVERRIDES[f"polymarket:{market_id}"]
    q = question.lower()
    has_d, has_r = bool(re.search(r"\bdemocrat", q)), bool(re.search(r"\brepublican", q))
    if has_d != has_r:
        return "D" if has_d else "R"
    for name, party in CANDIDATE_PARTY.items():
        if re.search(rf"\bwill {re.escape(name)} win\b", q):
            return party
    g = re.search(r"\(([DR])\)\s*$", group_title.strip())  # "Katie Hobbs (D)"
    return g[1] if g else None


# ---------------------------------------------------------------- pricing

def _f(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def price(bid: float | None, ask: float | None, last: float | None) -> float | None:
    """Mid of best bid/ask when there is a real two-sided book, else last price."""
    if bid is not None and ask is not None and 0 < bid <= ask < 1:
        return round((bid + ask) / 2, 6)
    if last is not None and 0 <= last <= 1:
        return last
    return None


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)


def _client() -> httpx.Client:
    return httpx.Client(timeout=30, headers={"User-Agent": "election-twin/0.1 (read-only)"})


def _get(c: httpx.Client, url: str, params: dict | None = None, tries: int = 4) -> Any:
    for i in range(tries):
        r = c.get(url, params=params)
        if r.status_code == 429 or r.status_code >= 500:
            import time
            time.sleep(1.5 * (i + 1))
            continue
        r.raise_for_status()
        return r.json()
    r.raise_for_status()


# ---------------------------------------------------------------- Kalshi

def _kalshi_rows(m: dict, event: dict, ts: datetime) -> dict:
    return {
        "venue": "kalshi", "native_id": m["ticker"], "event_id": event["event_ticker"],
        "event_title": event.get("title"), "title": m.get("title"),
        "outcome": m.get("yes_sub_title"), "token_id": None,
        "bid": _f(m.get("yes_bid_dollars")), "ask": _f(m.get("yes_ask_dollars")),
        "last": _f(m.get("last_price_dollars")), "status": m.get("status"), "fetched_at": ts,
    }


def _fetch_kalshi(c: httpx.Client) -> pl.DataFrame:
    series = [s["ticker"] for s in _get(c, f"{KALSHI}/series", {"category": "Elections"})["series"]
              if _K_SERIES.match(s["ticker"])]
    ts = _now()

    def one(st: str) -> list[dict]:
        rows, cur = [], None
        with _client() as cc:
            while True:
                p = {"series_ticker": st, "status": "open", "with_nested_markets": "true", "limit": 200}
                if cur:
                    p["cursor"] = cur
                d = _get(cc, f"{KALSHI}/events", p)
                for e in d.get("events", []):
                    if kalshi_race_id(e["event_ticker"]) or re.search(r"-(26|2026)", e["event_ticker"]):
                        rows += [_kalshi_rows(m, e, ts) for m in e.get("markets", [])]
                cur = d.get("cursor")
                if not cur or not d.get("events"):
                    return rows

    with ThreadPoolExecutor(8) as ex:
        rows = [r for rs in ex.map(one, series) for r in rs]
    return pl.DataFrame(rows, schema=_RAW_SCHEMA)


def _kalshi_prices(c: httpx.Client, tickers: list[str]) -> dict[str, tuple]:
    out = {}
    for i in range(0, len(tickers), 100):
        d = _get(c, f"{KALSHI}/markets", {"tickers": ",".join(tickers[i:i + 100]), "limit": 1000})
        for m in d.get("markets", []):
            out[m["ticker"]] = (_f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars")),
                                _f(m.get("last_price_dollars")))
    return out


# ---------------------------------------------------------------- Polymarket

def _yes_token(m: dict) -> str | None:
    try:
        outs, toks = json.loads(m.get("outcomes") or "[]"), json.loads(m.get("clobTokenIds") or "[]")
        return toks[outs.index("Yes")]
    except (ValueError, IndexError, TypeError):
        return None


def _fetch_polymarket(c: httpx.Client) -> pl.DataFrame:
    events, off = [], 0
    while True:
        d = _get(c, f"{GAMMA}/events", {"tag_slug": "midterms", "closed": "false", "limit": 100, "offset": off})
        events += d
        if len(d) < 100:
            break
        off += 100
    ts, rows = _now(), []
    for e in events:
        for m in e.get("markets", []):
            if not m.get("active") or m.get("closed") or not m.get("outcomePrices"):
                continue  # unlisted placeholder slots ("Person A", "Option B", ...)
            rows.append({
                "venue": "polymarket", "native_id": str(m["id"]), "event_id": e["slug"],
                "event_title": e.get("title"), "title": m.get("question"),
                "outcome": m.get("groupItemTitle"), "token_id": _yes_token(m),
                "bid": _f(m.get("bestBid")), "ask": _f(m.get("bestAsk")),
                "last": _f(m.get("lastTradePrice")), "status": "active", "fetched_at": ts,
            })
    df = pl.DataFrame(rows, schema=_RAW_SCHEMA)
    # CLOB midpoint for party markets only (cheaper; others never become objects).
    mapped = _map(df).filter(pl.col("race_id").is_not_null() & pl.col("yes_means").is_not_null())
    mids = _clob_mids(c, [t for t in mapped["token_id"].to_list() if t])
    return df.with_columns(pl.col("token_id").replace_strict(mids, default=None, return_dtype=pl.Float64).alias("clob_mid"))


def _clob_mids(c: httpx.Client, tokens: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(0, len(tokens), 400):
        try:
            r = c.post(f"{CLOB}/midpoints", json=[{"token_id": t} for t in tokens[i:i + 400]])
            r.raise_for_status()
            out.update({k: float(v) for k, v in r.json().items() if v is not None})
        except (httpx.HTTPError, ValueError):
            pass  # fall back to Gamma bestBid/bestAsk
    return out


def _gamma_prices(c: httpx.Client, ids: list[str]) -> dict[str, tuple]:
    out = {}
    for i in range(0, len(ids), 50):
        for m in _get(c, f"{GAMMA}/markets", {"id": ids[i:i + 50], "limit": 50}):
            out[str(m["id"])] = (_f(m.get("bestBid")), _f(m.get("bestAsk")), _f(m.get("lastTradePrice")))
    return out


# ---------------------------------------------------------------- contract

_RAW_SCHEMA = {
    "venue": pl.Utf8, "native_id": pl.Utf8, "event_id": pl.Utf8, "event_title": pl.Utf8,
    "title": pl.Utf8, "outcome": pl.Utf8, "token_id": pl.Utf8, "bid": pl.Float64,
    "ask": pl.Float64, "last": pl.Float64, "status": pl.Utf8, "fetched_at": pl.Datetime,
}


def fetch(tag: str = "pull", cache: bool = True) -> dict[str, pl.DataFrame]:
    with _client() as c:
        k, p = _fetch_kalshi(c), _fetch_polymarket(c)
    k = k.with_columns(pl.lit(None, pl.Float64).alias("clob_mid"), pl.lit(tag).alias("tag"))
    p = p.with_columns(pl.lit(tag).alias("tag"))
    raw = {"kalshi_markets": k, "polymarket_markets": p}
    if cache:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        for name, df in raw.items():
            df.write_parquet(RAW_DIR / f"{name}.parquet")
    return raw


def _map(df: pl.DataFrame) -> pl.DataFrame:
    """Attach market_id, race_id, yes_means to raw rows (either venue)."""
    race, yes = [], []
    for r in df.iter_rows(named=True):
        if r["venue"] == "kalshi":
            rid, ym = kalshi_race_id(r["event_id"]), kalshi_yes_means(r["native_id"])
        else:
            rid = polymarket_race_id(r["event_id"])
            ym = polymarket_yes_means(r["native_id"], r["title"] or "", r["outcome"] or "")
        if rid in R_ONLY_RACES and ym != "R":
            ym = None
        race.append(rid)
        yes.append(ym if rid else None)
    return df.with_columns(
        (pl.col("venue") + ":" + pl.col("native_id")).alias("market_id"),
        pl.Series("race_id", race, pl.Utf8), pl.Series("yes_means", yes, pl.Utf8),
    )


def _raw_all(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    frames = [df.select(list(_RAW_SCHEMA) + ["clob_mid", "tag"]) for df in raw.values() if df.height]
    return pl.concat(frames, how="vertical_relaxed") if frames else pl.DataFrame()


def _snapshot_rows(mapped: pl.DataFrame, ts: datetime, tag: str) -> pl.DataFrame:
    rows = []
    for r in mapped.iter_rows(named=True):
        p = price(r["bid"], r["ask"], r["last"])
        mid = r.get("clob_mid")
        two_sided = r["bid"] is not None and r["ask"] is not None and 0 < r["bid"] <= r["ask"] < 1
        if two_sided and mid is not None and 0 < mid < 1:
            p = mid  # live CLOB midpoint beats Gamma's cached top-of-book
        if p is None:
            continue
        rows.append({"market_id": r["market_id"], "ts": ts, "p_yes": p,
                     "p_dem": p if r["yes_means"] == "D" else round(1 - p, 6), "tag": tag})
    return pl.DataFrame(rows, schema={"market_id": pl.Utf8, "ts": pl.Datetime, "p_yes": pl.Float64,
                                      "p_dem": pl.Float64, "tag": pl.Utf8})


def to_objects(raw: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    all_ = _raw_all(raw)
    if all_.is_empty():
        return {"Market": pl.DataFrame(), "MarketSnapshot": pl.DataFrame()}
    m = _map(all_).filter(pl.col("race_id").is_not_null() & pl.col("yes_means").is_not_null())
    market = m.select("market_id", "venue", pl.col("title").fill_null(pl.col("outcome")).alias("title"),
                      "race_id", "yes_means").unique("market_id", keep="first")
    tag = m["tag"][0] if m.height else "pull"
    ts = m["fetched_at"].max() if m.height else _now()
    return {"Market": market, "MarketSnapshot": _snapshot_rows(m.unique("market_id", keep="first"), ts, tag)}


def unmapped(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Race-winner-looking markets dropped for lack of a race_id or party."""
    m = _map(_raw_all(raw))
    looks = (
        ((pl.col("venue") == "kalshi") & pl.col("event_id").str.contains(r"-(26|2026|26NOV)$"))
        | ((pl.col("venue") == "polymarket") & pl.col("event_id").str.contains(_P_WINNER.pattern)
           & ~pl.col("event_id").str.contains(_P_NOT_RACE))
    )
    return m.filter(looks & (pl.col("race_id").is_null() | pl.col("yes_means").is_null())).select(
        "market_id", "event_id", "title", "outcome", "race_id", "bid", "ask", "last")


def load_known(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    raw = {p.stem: pl.read_parquet(p) for p in raw_dir.glob("*_markets.parquet")}
    if not raw:
        raise FileNotFoundError(f"no cached markets under {raw_dir}; run fetch() first")
    return _map(_raw_all(raw)).filter(pl.col("race_id").is_not_null() & pl.col("yes_means").is_not_null())


def snapshot(tag: str = "pull", known: pl.DataFrame | None = None) -> pl.DataFrame:
    """Re-price already-known markets only (no discovery). Returns MarketSnapshot rows.

    `known` defaults to the mapped markets from the last fetch() cache. Rows need
    market_id, venue, native_id, token_id, yes_means (fetch/_map output has them).
    """
    known = load_known() if known is None else known
    ts = _now()
    with _client() as c:
        k_ids = known.filter(pl.col("venue") == "kalshi")["native_id"].to_list()
        p_known = known.filter(pl.col("venue") == "polymarket")
        kp = _kalshi_prices(c, k_ids) if k_ids else {}
        gp = _gamma_prices(c, p_known["native_id"].to_list()) if p_known.height else {}
        mids = _clob_mids(c, [t for t in p_known["token_id"].to_list() if t]) if p_known.height else {}
    rows = []
    for r in known.iter_rows(named=True):
        bid = ask = last = None
        mid = None
        if r["venue"] == "kalshi":
            bid, ask, last = kp.get(r["native_id"], (None, None, None))
        else:
            bid, ask, last = gp.get(r["native_id"], (None, None, None))
            mid = mids.get(r["token_id"])
        rows.append({"market_id": r["market_id"], "yes_means": r["yes_means"], "bid": bid,
                     "ask": ask, "last": last, "clob_mid": mid})
    return _snapshot_rows(pl.DataFrame(rows, schema={"market_id": pl.Utf8, "yes_means": pl.Utf8,
                                                     "bid": pl.Float64, "ask": pl.Float64,
                                                     "last": pl.Float64, "clob_mid": pl.Float64}),
                          ts, tag)
