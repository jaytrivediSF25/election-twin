"""Polls ingest: generic ballot, Senate, governor and House polls + published averages.

Sources (machine-readable only; HTML-only / bot-walled sites are recorded as failures):
  votehub_polls       api.votehub.com/polls               individual polls, all offices (JSON)
  fpo_averages        fiftyplusone.news/averages           race averages w/ candidate party (SvelteKit data)
  fpo_generic         fiftyplusone.news generic page       generic-ballot average series
  sb_generic_avg      Silver Bulletin Datawrapper CSV      generic-ballot average series
  sb_generic_polls    Silver Bulletin Datawrapper CSV      generic-ballot poll list (with population)
  wiki_*_candidates   Wikipedia 2026 Senate/Gov/House pages candidate lists -> party map

Not ingested (checked 2026-11-03): FiftyPlusOne CSV API needs an API key; RealClearPolling and
votehub.com pages return 403 (bot wall); RacetotheWH charts are Infogram embeds with no data
payload; PollCheck domain is parked.

Margins are D minus R. Where the main non-Republican is an independent (NE Osborn, MT Bodnar,
SD Bengs, ID Achilles) that candidate is the D side.
"""
from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from datetime import date, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "polls"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) election-twin/0.1"}
# Wikimedia rejects browser-like UAs from scripts; it wants a descriptive bot UA.
WIKI_UA = {"User-Agent": "election-twin/0.1 (research; https://github.com/jaytrivediSF25) httpx"}
RACES_RAW_DIR = RAW_DIR.parent / "races"

SB_AVG_CHART = "rfiFi"
SB_POLLS_CHART = "sEHv2"
WIKI_PAGES = {
    "wiki_senate_candidates": "2026_United_States_Senate_elections",
    "wiki_governor_candidates": "2026_United_States_gubernatorial_elections",
    "wiki_house_candidates": "2026_United_States_House_of_Representatives_elections",
}

SOURCES: dict[str, str] = {
    "votehub_polls": "https://api.votehub.com/polls",
    "fpo_averages": "https://fiftyplusone.news/averages/__data.json",
    "fpo_generic": "https://fiftyplusone.news/polls/generic-ballot/generic-ballot/__data.json",
    "sb_generic_avg": f"https://datawrapper.dwcdn.net/{SB_AVG_CHART}/",
    "sb_generic_polls": f"https://datawrapper.dwcdn.net/{SB_POLLS_CHART}/",
    **{k: f"https://en.wikipedia.org/w/index.php?title={v}&action=raw" for k, v in WIKI_PAGES.items()},
}

SENATE_MIN_DATE = date(2026, 7, 1)
GENERIC_MIN_DATE = date(2025, 1, 1)

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA",
    "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
    "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
    "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}
SENATE_SPECIALS = {"FL", "OH"}           # 2026 specials (Rubio and Vance seats)
AT_LARGE = {"AK", "DE", "ND", "SD", "VT", "WY"}
# Independent treated as the D side (README: main non-R independent backed by / caucusing w/ Dems)
IND_D_SIDE = {"SEN-NE": "osborn", "SEN-MT": "bodnar", "SEN-SD": "bengs", "SEN-ID": "achilles"}
POP_RANK = {"lv": 0, "rv": 1, "v": 2, "a": 3}


# ----------------------------------------------------------------------------- helpers

def _get(url: str, *, timeout: float = 60.0) -> httpx.Response:
    headers = WIKI_UA if "wikipedia.org" in url else UA
    r = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r


def _cached(name: str, url: str, refresh: bool, suffix: str) -> bytes:
    """Download url to data/raw/polls/<name><suffix>; on failure (or refresh=False) use cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}{suffix}"
    if refresh or not path.exists():
        try:
            path.write_bytes(_get(url).content)
        except Exception:
            if not path.exists():
                raise
    return path.read_bytes()


def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", " ", s)
    s = re.sub(r"\b[a-z]\.", " ", s)           # middle initials
    return " ".join(re.sub(r"[^a-z ]+", " ", s).split())


def _last(s: str) -> str:
    n = _norm_name(s)
    return n.split()[-1] if n else ""


def poll_id(race_id, pollster, end_date, sample_size, dem_pct, rep_pct) -> str:
    key = "|".join(str(x) for x in (race_id, slugify(pollster), end_date, sample_size,
                                     f"{float(dem_pct):.1f}", f"{float(rep_pct):.1f}"))
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _devalue(arr: list) -> Any:
    """Decode SvelteKit's flattened (devalue) payload."""
    cache: dict[int, Any] = {}

    def h(i: int):
        if i < 0:
            return None
        if i in cache:
            return cache[i]
        v = arr[i]
        if isinstance(v, dict):
            o: dict = {}
            cache[i] = o
            o.update({k: h(j) for k, j in v.items()})
            return o
        if isinstance(v, list):
            if v and isinstance(v[0], str) and v[0] in ("Date", "Set", "Map", "BigInt", "RegExp"):
                return v[1] if v[0] == "Date" else None
            o2: list = []
            cache[i] = o2
            o2.extend(h(j) for j in v)
            return o2
        return v

    return h(0)


def _svelte_nodes(blob: bytes) -> list:
    d = json.loads(blob)
    return [_devalue(n["data"]) if n and n.get("type") == "data" else None for n in d["nodes"]]


def _dw_latest(chart: str) -> str:
    """Datawrapper's bare chart URL redirects (meta refresh) to the newest published version."""
    t = _get(f"https://datawrapper.dwcdn.net/{chart}/").text
    m = re.findall(rf"datawrapper\.dwcdn\.net/{chart}/(\d+)", t)
    if not m:
        raise RuntimeError(f"no datawrapper version for {chart}")
    return f"https://datawrapper.dwcdn.net/{chart}/{max(int(x) for x in m)}/dataset.csv"


# ----------------------------------------------------------------------------- fetch

def _fetch_votehub(refresh: bool) -> pl.DataFrame:
    polls = json.loads(_cached("votehub_polls", SOURCES["votehub_polls"], refresh, ".json"))
    rows = []
    for p in polls:
        for a in p.get("answers") or []:
            rows.append({
                "id": p.get("id"), "poll_type": p.get("poll_type"), "subject": p.get("subject"),
                "seat_name": p.get("seat_name"), "pollster": p.get("pollster"),
                "sponsors": "; ".join(p.get("sponsors") or []), "start_date": p.get("start_date"),
                "end_date": p.get("end_date"), "sample_size": p.get("sample_size"),
                "population": p.get("population"), "internal": p.get("internal"),
                "partisan": p.get("partisan"), "url": p.get("url"),
                "choice": a.get("choice"), "pct": a.get("pct"),
            })
    return pl.DataFrame(rows, infer_schema_length=None)


def _fetch_fpo_averages(refresh: bool) -> pl.DataFrame:
    nodes = _svelte_nodes(_cached("fpo_averages", SOURCES["fpo_averages"], refresh, ".data.json"))
    series = next(n for n in nodes if isinstance(n, dict) and "chartData" in n)["chartData"]
    rows = []
    for s in series:
        for r in s["data"]:
            rows.append({"label": s["label"], "path": s["path"], "fpo_race_id": r.get("race_id"),
                         "office": r.get("office"), "end_date": r.get("end_date"),
                         "candidate_name": r.get("candidate_name"), "party": r.get("party"),
                         "estimate": r.get("estimate"), "lo": r.get("lo"), "hi": r.get("hi")})
    return pl.DataFrame(rows, infer_schema_length=None)


def _fetch_fpo_generic(refresh: bool) -> pl.DataFrame:
    nodes = _svelte_nodes(_cached("fpo_generic", SOURCES["fpo_generic"], refresh, ".data.json"))
    cd = next(n for n in nodes if isinstance(n, dict) and "chartData" in n)["chartData"]
    keep = ("end_date", "subgroup", "dem_estimate", "dem_lo", "dem_hi", "rep_estimate", "rep_lo", "rep_hi")
    return pl.DataFrame([{k: r.get(k) for k in keep} for r in cd], infer_schema_length=None)


def _fetch_sb(name: str, chart: str, refresh: bool) -> pl.DataFrame:
    path = RAW_DIR / f"{name}.csv"
    if refresh or not path.exists():
        try:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            path.write_bytes(_get(_dw_latest(chart)).content)
        except Exception:
            if not path.exists():
                raise
    return pl.read_csv(path, infer_schema_length=0)   # all Utf8, parsed in to_objects


_STRIPE = re.compile(r"^\*\s*\{\{Party stripe\|([^}|]*)[^}]*\}\}(.*)$", re.M)


def _fetch_wiki_candidates(name: str, refresh: bool) -> pl.DataFrame:
    try:
        blob = _cached(name, SOURCES[name], refresh, ".wikitext")
    except Exception:
        # fall back to the copy the races ingest already landed
        blob = (RACES_RAW_DIR / f"{WIKI_PAGES[name]}.wikitext").read_bytes()
    text = blob.decode("utf-8", "replace")
    rows = []
    for m in _STRIPE.finditer(text):
        party_raw, rest = m.group(1), m.group(2)
        rest = re.sub(r"\{\{(Aye|Nay)\}\}|'''", "", rest)
        rest = re.sub(r"<ref.*", "", rest)
        link = re.match(r"\s*\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", rest)
        cand = link.group(1) if link else re.split(r"\(", rest)[0]
        cand = cand.strip()
        if not cand:
            continue
        p = party_raw.lower()
        party = ("DEM" if "democratic" in p else "REP" if "republican" in p
                 else "IND" if "independent" in p else "OTH")
        rows.append({"candidate_name": cand, "party": party, "party_raw": party_raw})
    return pl.DataFrame(rows, schema={"candidate_name": pl.Utf8, "party": pl.Utf8, "party_raw": pl.Utf8})


FAILED: dict[str, str] = {}


def fetch(refresh: bool = True) -> dict[str, pl.DataFrame]:
    """Download (or, if refresh=False, reuse cached) sources. Individual failures are recorded in
    FAILED and skipped so a partial result is still usable."""
    FAILED.clear()
    jobs = {
        "votehub_polls": lambda: _fetch_votehub(refresh),
        "fpo_averages": lambda: _fetch_fpo_averages(refresh),
        "fpo_generic": lambda: _fetch_fpo_generic(refresh),
        "sb_generic_avg": lambda: _fetch_sb("sb_generic_avg", SB_AVG_CHART, refresh),
        "sb_generic_polls": lambda: _fetch_sb("sb_generic_polls", SB_POLLS_CHART, refresh),
        **{k: (lambda k=k: _fetch_wiki_candidates(k, refresh)) for k in WIKI_PAGES},
    }
    out: dict[str, pl.DataFrame] = {}
    for name, job in jobs.items():
        try:
            out[name] = job()
        except Exception as e:  # noqa: BLE001 - one dead source must not sink the batch
            FAILED[name] = f"{type(e).__name__}: {e}"[:300]
    return out


# ----------------------------------------------------------------------------- to_objects

def _race_id(poll_type: str, subject: str | None, seat_name: str | None = None) -> str | None:
    if poll_type == "generic-ballot":
        return "GENERIC" if subject == "2026" else None
    if not subject or not subject.startswith("2026 "):
        return None
    rest = subject[5:]
    if rest.endswith((" Democratic", " Republican")):      # primaries
        return None
    if poll_type == "us-representative":
        m = re.fullmatch(r"([A-Z]{2})-(\d{1,2}|AL)", rest)
        if not m:
            return None
        st, d = m.groups()
        return f"HOUSE-{st}-AL" if st in AT_LARGE or d == "AL" else f"HOUSE-{st}-{int(d):02d}"
    st = STATES.get(rest)
    if not st:
        return None
    if poll_type == "us-senator":
        special = st in SENATE_SPECIALS or (seat_name or "").lower().startswith("special")
        return f"SEN-{st}-S" if special else f"SEN-{st}"
    if poll_type == "governor":
        return f"GOV-{st}"
    return None


def _fpo_race_id(label: str) -> str | None:
    m = re.fullmatch(r"(.+) (Senate|Governor)", label)
    if not m or m.group(1) not in STATES:
        return None
    st = STATES[m.group(1)]
    if m.group(2) == "Governor":
        return f"GOV-{st}"
    return f"SEN-{st}-S" if st in SENATE_SPECIALS else f"SEN-{st}"


class _Parties:
    """name -> party lookup: FiftyPlusOne (per race) first, then Wikipedia candidate lists."""

    def __init__(self, raw: dict[str, pl.DataFrame]):
        self.race_last: dict[str, dict[str, str]] = {}
        self.full: dict[str, str] = {}
        self.last: dict[str, set[str]] = {}
        fpo = raw.get("fpo_averages")
        if fpo is not None and fpo.height:
            for r in fpo.select("label", "candidate_name", "party").unique().iter_rows(named=True):
                rid = _fpo_race_id(r["label"])
                if rid:
                    self.race_last.setdefault(rid, {})[_last(r["candidate_name"])] = r["party"]
                self.full.setdefault(_norm_name(r["candidate_name"]), r["party"])
        for k in WIKI_PAGES:
            df = raw.get(k)
            if df is None:
                continue
            for r in df.iter_rows(named=True):
                self.full.setdefault(_norm_name(r["candidate_name"]), r["party"])
                self.last.setdefault(_last(r["candidate_name"]), set()).add(r["party"])

    def party(self, race_id: str, name: str) -> str | None:
        ln = _last(name)
        if ln in self.race_last.get(race_id, {}):
            return self.race_last[race_id][ln]
        if _norm_name(name) in self.full:
            return self.full[_norm_name(name)]
        ps = self.last.get(ln)
        return next(iter(ps)) if ps and len(ps) == 1 else None


def _votehub_polls(raw: dict[str, pl.DataFrame], parties: _Parties) -> list[dict]:
    vh = raw.get("votehub_polls")
    if vh is None or vh.is_empty():
        return []
    vh = vh.filter(pl.col("poll_type").is_in(["generic-ballot", "us-senator", "governor", "us-representative"]))
    # 1) per poll: candidate list with parties
    polls: dict[str, dict] = {}
    for r in vh.iter_rows(named=True):
        rid = _race_id(r["poll_type"], r["subject"], r["seat_name"])
        if rid is None or r["pct"] is None or not r["end_date"]:
            continue
        end = date.fromisoformat(r["end_date"][:10])
        if end < (GENERIC_MIN_DATE if rid == "GENERIC" else SENATE_MIN_DATE):
            continue
        p = polls.setdefault(r["id"], {**r, "race_id": rid, "end": end, "cands": []})
        if rid == "GENERIC":
            party = {"dem": "DEM", "rep": "REP"}.get(str(r["choice"]).lower())
        else:
            party = parties.party(rid, r["choice"])
        p["cands"].append((r["choice"], party, float(r["pct"])))

    # 2) nominee pair per race = most common (D-side, R) last-name pair in polls since Sep 1
    def sides(p):
        rid, cands = p["race_id"], p["cands"]
        reps = sorted([c for c in cands if c[1] == "REP"], key=lambda c: -c[2])
        if rid in IND_D_SIDE:
            ds = [c for c in cands if _last(c[0]) == IND_D_SIDE[rid]]
        else:
            ds = sorted([c for c in cands if c[1] == "DEM"], key=lambda c: -c[2])
        return (ds[0] if ds else None), (reps[0] if reps else None)

    pair_counts: dict[str, dict[tuple, int]] = {}
    for p in polls.values():
        d, r = sides(p)
        if d and r and p["race_id"] != "GENERIC":
            w = 3 if p["end"] >= date(2026, 9, 1) else 1
            key = (_last(d[0]), _last(r[0]))
            pair_counts.setdefault(p["race_id"], {}).setdefault(key, 0)
            pair_counts[p["race_id"]][key] += w
    nominees = {rid: max(c, key=c.get) for rid, c in pair_counts.items()}

    out = []
    for p in polls.values():
        d, r = sides(p)
        if not (d and r):
            continue
        if p["race_id"] != "GENERIC" and (_last(d[0]), _last(r[0])) != nominees.get(p["race_id"]):
            continue   # hypothetical / non-nominee matchup
        out.append({
            "race_id": p["race_id"], "pollster": p["pollster"] or "Unknown", "end_date": p["end"],
            "sample_size": p["sample_size"], "population": (p["population"] or None),
            "dem_pct": d[2], "rep_pct": r[2], "source": "votehub",
        })
    return out


_SB_DATE = re.compile(r"(\d{1,2})/(\d{1,2})(?:,\s*(\d{4}))?\s*-\s*(\d{1,2})/(\d{1,2}),\s*(\d{4})")


def _sb_polls(raw: dict[str, pl.DataFrame]) -> list[dict]:
    df = raw.get("sb_generic_polls")
    if df is None or df.is_empty():
        return []
    out = []
    for r in df.iter_rows(named=True):
        dates = (r.get("Dates") or "").split("@@")[0]
        m = _SB_DATE.search(dates)
        if m:
            end = date(int(m.group(6)), int(m.group(4)), int(m.group(5)))
        else:
            m2 = re.search(r"(\d{1,2})/(\d{1,2}),\s*(\d{4})", dates)
            if not m2:
                continue
            end = date(int(m2.group(3)), int(m2.group(1)), int(m2.group(2)))
        if end < GENERIC_MIN_DATE:
            continue
        name = re.sub(r"<[^>]+>", "", (r.get("Pollster") or "").split("^")[0]).strip()
        samp = (r.get("Sample") or "")
        sm = re.match(r"\s*([\d,]+)\s*([A-Za-z]+)?", samp.split("@@")[0])
        size = int(sm.group(1).replace(",", "")) if sm and sm.group(1).strip(",") else None
        pop = sm.group(2).lower() if sm and sm.group(2) else None
        try:
            dem, rep = float(r["Democrats"]), float(r["Republicans"])
        except (TypeError, ValueError):
            continue
        out.append({"race_id": "GENERIC", "pollster": name or "Unknown", "end_date": end,
                    "sample_size": size, "population": pop if pop in POP_RANK else None,
                    "dem_pct": dem, "rep_pct": rep, "source": "silver_bulletin"})
    return out


def _tokens(name: str) -> set[str]:
    stop = {"the", "of", "and", "college", "university", "poll", "polling", "research", "group",
            "insights", "news", "survey", "strategies", "associates", "inc", "llc", "co"}
    return {t for t in slugify(name).split("-") if t and t not in stop}


def _dedupe(polls: list[dict]) -> list[dict]:
    """Same poll in several sources: same race, end date within a day, same toplines (+-0.5) and
    a shared pollster token or identical sample size. Keep the copy with population info,
    then votehub over silver_bulletin. Within one source+pollster+date keep the best population
    (lv > rv > v > a) so LV/RV versions of one survey are not double counted."""
    src_rank = {"votehub": 0, "silver_bulletin": 1}

    def score(p):
        return (p["population"] is None, src_rank.get(p["source"], 9),
                POP_RANK.get(p["population"] or "", 9))

    polls = sorted(polls, key=score)
    kept: list[dict] = []
    by_race: dict[str, list[dict]] = {}
    for p in polls:
        dup = False
        for q in by_race.get(p["race_id"], []):
            if abs((p["end_date"] - q["end_date"]).days) > 1:
                continue
            same_org = bool(_tokens(p["pollster"]) & _tokens(q["pollster"])) or (
                p["sample_size"] and p["sample_size"] == q["sample_size"])
            if not same_org:
                continue
            same_numbers = abs(p["dem_pct"] - q["dem_pct"]) <= 0.5 and abs(p["rep_pct"] - q["rep_pct"]) <= 0.5
            same_survey_other_pop = (p["source"] == q["source"] and p["end_date"] == q["end_date"]
                                     and slugify(p["pollster"]) == slugify(q["pollster"])
                                     and p["population"] != q["population"])
            if same_numbers or same_survey_other_pop:
                dup = True
                break
        if not dup:
            kept.append(p)
            by_race.setdefault(p["race_id"], []).append(p)
    return kept


def _averages(raw: dict[str, pl.DataFrame], polls: pl.DataFrame) -> list[dict]:
    out = []
    # FiftyPlusOne race averages: latest date per series
    fpo = raw.get("fpo_averages")
    if fpo is not None and fpo.height:
        fpo = fpo.with_columns(pl.col("end_date").str.slice(0, 10).str.to_date())
        for label, g in fpo.group_by("label"):
            rid = _fpo_race_id(label[0])
            if not rid:
                continue
            last = g.filter(pl.col("end_date") == g["end_date"].max())
            reps = last.filter(pl.col("party") == "REP").sort("estimate", descending=True)
            if rid in IND_D_SIDE:
                ds = last.filter(pl.col("candidate_name").map_elements(_last, return_dtype=pl.Utf8)
                                 == IND_D_SIDE[rid])
            else:
                ds = last.filter(pl.col("party") == "DEM").sort("estimate", descending=True)
            if ds.is_empty() or reps.is_empty():
                continue
            out.append({"race_id": rid, "source": "fiftyplusone",
                        "margin": round(float(ds["estimate"][0]) - float(reps["estimate"][0]), 2),
                        "n_polls": None, "as_of": last["end_date"][0]})
    g = raw.get("fpo_generic")
    if g is not None and g.height:
        g = g.filter(pl.col("subgroup") == "All polls") if "All polls" in g["subgroup"].to_list() else g
        g = g.sort("end_date")
        r = g.row(-1, named=True)
        out.append({"race_id": "GENERIC", "source": "fiftyplusone",
                    "margin": round(float(r["dem_estimate"]) - float(r["rep_estimate"]), 2),
                    "n_polls": None, "as_of": date.fromisoformat(str(r["end_date"])[:10])})
    sb = raw.get("sb_generic_avg")
    if sb is not None and sb.height:
        sb = sb.with_columns(pl.col("modeldate").str.to_date("%m/%d/%Y"),
                             pl.col("dem").cast(pl.Float64), pl.col("rep").cast(pl.Float64)).sort("modeldate")
        r = sb.row(-1, named=True)
        out.append({"race_id": "GENERIC", "source": "silver_bulletin",
                    "margin": round(r["dem"] - r["rep"], 2), "n_polls": None, "as_of": r["modeldate"]})
    # n_polls: polls this ingest holds for the race (all sources) ending within 60 days of as_of
    if polls.height:
        for a in out:
            sub = polls.filter((pl.col("race_id") == a["race_id"])
                               & (pl.col("end_date") > pl.lit(a["as_of"]) - pl.duration(days=60))
                               & (pl.col("end_date") <= pl.lit(a["as_of"])))
            a["n_polls"] = sub.height
    return out


POLL_COLS = ["poll_id", "race_id", "pollster_id", "end_date", "sample_size", "population",
             "dem_pct", "rep_pct", "margin", "weight", "excluded", "source"]


def to_objects(raw: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    parties = _Parties(raw)
    polls = _dedupe(_votehub_polls(raw, parties) + _sb_polls(raw))
    rows, seen = [], set()
    pollsters: dict[str, str] = {}
    for p in polls:
        pid = poll_id(p["race_id"], p["pollster"], p["end_date"], p["sample_size"], p["dem_pct"], p["rep_pct"])
        if pid in seen:
            continue
        seen.add(pid)
        slug = slugify(p["pollster"]) or "unknown"
        pollsters.setdefault(slug, p["pollster"])
        margin = round(p["dem_pct"] - p["rep_pct"], 2)
        rows.append({"poll_id": pid, "race_id": p["race_id"], "pollster_id": slug,
                     "end_date": p["end_date"],
                     "sample_size": int(p["sample_size"]) if p["sample_size"] else None,
                     "population": p["population"], "dem_pct": p["dem_pct"], "rep_pct": p["rep_pct"],
                     "margin": margin, "weight": 1.0,
                     # blowouts beyond +-40 are kept for lineage but excluded from modelling
                     "excluded": abs(margin) > 40, "source": p["source"]})
    poll_schema = {"poll_id": pl.Utf8, "race_id": pl.Utf8, "pollster_id": pl.Utf8, "end_date": pl.Date,
                   "sample_size": pl.Int64, "population": pl.Utf8, "dem_pct": pl.Float64,
                   "rep_pct": pl.Float64, "margin": pl.Float64, "weight": pl.Float64,
                   "excluded": pl.Boolean, "source": pl.Utf8}
    poll_df = pl.DataFrame(rows, schema=poll_schema).sort("race_id", "end_date")
    pollster_df = pl.DataFrame(
        [{"pollster_id": k, "name": v, "house_effect": None} for k, v in sorted(pollsters.items())],
        schema={"pollster_id": pl.Utf8, "name": pl.Utf8, "house_effect": pl.Float64})
    avg_df = pl.DataFrame(_averages(raw, poll_df),
                          schema={"race_id": pl.Utf8, "source": pl.Utf8, "margin": pl.Float64,
                                  "n_polls": pl.Int64, "as_of": pl.Date})
    return {"Pollster": pollster_df, "Poll": poll_df, "PollAverage": avg_df}


if __name__ == "__main__":
    import sys
    raw = fetch(refresh="--cached" not in sys.argv)
    objs = to_objects(raw)
    for k, v in objs.items():
        print(k, v.height)
    print("failed:", FAILED or "none")
    with pl.Config(tbl_rows=100):
        print(objs["PollAverage"].sort("race_id", "source"))
