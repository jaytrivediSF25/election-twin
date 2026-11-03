"""Races ingest: 2026 Senate / House / governor Race objects, House District objects,
Senate/governor (and House) Candidate objects, plus the raw `senate_holdovers` frame.

Sources (cached under data/raw/races/, parsed as data only):
- Wikipedia 2026 Senate / House / gubernatorial election pages and the House election
  ratings page: incumbents, status (open seat), specials, nominees, and the per-rater
  ratings tables (we take Sabato's Crystal Ball, falling back to Cook, then other raters).
- The Downballot "2024 presidential results by congressional district" sheet on the lines
  used in 2026 (vote totals tab). House/District pres24_margin comes from it; statewide
  pres24_margin is the sum of district vote totals per state.

Margins are D minus R in points (Trump +10 = -10.0).
"""
from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import httpx
import polars as pl
from lxml import html as lxml_html

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "races"
UA = {"User-Agent": "election-twin/0.1 (election forecasting research)"}

WIKI = "https://en.wikipedia.org/wiki/"
DOWNBALLOT_SHEET = "1eZfaFI-c-PFOoKx1-zZA2MP0_dxRq_LVK0re3BOQqy0"
DOWNBALLOT_TOTALS_GID = "1491069057"  # exact vote totals per district, 2026 lines

_FILES = {
    "senate": ("2026_United_States_Senate_elections.html", WIKI + "2026_United_States_Senate_elections"),
    "house": ("2026_United_States_House_of_Representatives_elections.html",
              WIKI + "2026_United_States_House_of_Representatives_elections"),
    "house_ratings": ("2026_United_States_House_of_Representatives_election_ratings.html",
                      WIKI + "2026_United_States_House_of_Representatives_election_ratings"),
    "governor": ("2026_United_States_gubernatorial_elections.html", WIKI + "2026_United_States_gubernatorial_elections"),
    "downballot": ("downballot_pres24_cd26_1491069057.csv",
                   f"https://docs.google.com/spreadsheets/d/{DOWNBALLOT_SHEET}/export?format=csv&gid={DOWNBALLOT_TOTALS_GID}"),
}

SOURCES: dict[str, str] = {
    "senate_ratings": _FILES["senate"][1] + "#Predictions",
    "senate_incumbents": _FILES["senate"][1] + "#Race_summary",
    "senate_holdovers": _FILES["senate"][1] + "#Summary_results",
    "house_members": _FILES["house"][1],
    "house_ratings": _FILES["house_ratings"][1],
    "governor_ratings": _FILES["governor"][1] + "#Predictions",
    "governor_incumbents": _FILES["governor"][1] + "#States",
    "rating_asof": "wikipedia ratings table headers",
    "downballot_pres24_cd": f"https://docs.google.com/spreadsheets/d/{DOWNBALLOT_SHEET}/edit?gid={DOWNBALLOT_TOTALS_GID}",
}

RATINGS = ("Safe D", "Likely D", "Lean D", "Tossup", "Lean R", "Likely R", "Safe R")
RATER_PRIORITY = ("Sabato", "Cook", "IE", "DDHQ", "FPO", "ST", "Econ", "Econ.", "NBC", "Fox", "RCP", "Silver", "MS NOW")

STATES = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
    "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY",
}
PARTY = {"Democratic": "D", "Republican": "R", "Independent": "I", "Democratic–NPL": "D", "DFL": "D",
         "Democratic-Farmer-Labor": "D", "Democratic–Farmer–Labor": "D"}
_MONTHS = "Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec"

# Thresholds used only when no rater covers a race (abs D-R pres24 margin, points).
DERIVE_THRESHOLDS = ((3.0, "Tossup"), (8.0, "Lean"), (15.0, "Likely"))


# ---------------------------------------------------------------- download / parse helpers
def _download(key: str, refresh: bool = False) -> Path:
    name, url = _FILES[key]
    path = RAW_DIR / name
    if path.exists() and path.stat().st_size > 0 and not refresh:
        return path
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    r = httpx.get(url, headers=UA, follow_redirects=True, timeout=60)
    r.raise_for_status()
    path.write_bytes(r.content)
    return path


def _clean(s: str) -> str:
    s = re.sub(r"\[[^\]]*\]", "", s or "")
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


def _cell_text(el) -> str:
    for bad in el.xpath(".//style|.//sup[contains(@class,'reference')]"):
        bad.drop_tree()
    for br in el.xpath(".//br"):
        br.tail = "\n" + (br.tail or "")
    for li in el.xpath(".//li"):
        li.tail = "\n" + (li.tail or "")
    return el.text_content()


def _grid(table) -> list[list[str]]:
    """Expand a wikitable into a rectangular grid, honouring rowspan/colspan."""
    grid: list[list[str]] = []
    pending: dict[tuple[int, int], str] = {}
    for ri, tr in enumerate(table.xpath("./tr|./tbody/tr|./thead/tr")):
        row: list[str] = []
        ci = 0
        cells = tr.xpath("./th|./td")
        k = 0
        while k < len(cells) or (ri, ci) in pending:
            if (ri, ci) in pending:
                row.append(pending.pop((ri, ci)))
                ci += 1
                continue
            c = cells[k]
            k += 1
            txt = _cell_text(c)
            rs = int(re.sub(r"\D", "", c.get("rowspan", "1")) or 1)
            cs = int(re.sub(r"\D", "", c.get("colspan", "1")) or 1)
            for dc in range(cs):
                row.append(txt)
                for dr in range(1, rs):
                    pending[(ri + dr, ci)] = txt
                ci += 1
        grid.append(row)
    return grid


def _wikitables(path: Path):
    doc = lxml_html.parse(str(path)).getroot()
    out = []
    for t in doc.xpath('//table[contains(@class,"wikitable")]'):
        h = t.xpath("preceding::*[self::h2 or self::h3][1]")
        out.append((_clean(h[0].text_content()) if h else "", t))
    return out


def _rater_header(h: str) -> tuple[str, str] | None:
    h = _clean(h)
    m = re.match(rf"^(.*?)\s*((?:{_MONTHS})\.?\s*\d{{1,2}},\s*\d{{4}})", h)
    if not m:
        return None
    return m.group(1).strip(), re.sub(r"\s+", " ", m.group(2).replace(",", ", ")).replace(" ,", ",")


def _ratings_table(table, chamber: str) -> tuple[pl.DataFrame, list[dict]]:
    g = _grid(table)
    header = g[1]
    raters = {i: _rater_header(h) for i, h in enumerate(header)}
    raters = {i: v for i, v in raters.items() if v}
    rows = []
    for r in g[2:]:
        label = _clean(r[0])
        if not label or label.startswith("Overall") or len(r) < len(header):
            continue
        rec = {"label": label, "pvi": _clean(r[1]), "incumbent": _clean(r[2])}
        for i, (name, _) in raters.items():
            rec[name] = _clean(r[i])
        rows.append(rec)
    asof = [{"chamber": chamber, "rater": n, "as_of": d} for n, d in raters.values()]
    return pl.DataFrame(rows, infer_schema_length=None), asof


# ---------------------------------------------------------------- fetch
def fetch(refresh: bool = False) -> dict[str, pl.DataFrame]:
    paths = {k: _download(k, refresh) for k in _FILES}
    raw: dict[str, pl.DataFrame] = {}
    asof: list[dict] = []

    # Senate
    sen = _wikitables(paths["senate"])
    pred = next(t for h, t in sen if h == "Predictions")
    raw["senate_ratings"], a = _ratings_table(pred, "senate")
    asof += a
    inc_rows = []
    for h, t in sen:
        if h.startswith("Special elections during") or h.startswith("Elections leading to"):
            for r in _grid(t)[2:]:
                if len(r) >= 7 and _clean(r[0]):
                    inc_rows.append({"label": _clean(r[0]), "senator": _clean(r[2]), "party": _clean(r[3]),
                                     "status": _clean(r[5]), "candidates": r[6].replace("\xa0", " "),
                                     "special": h.startswith("Special")})
    raw["senate_incumbents"] = pl.DataFrame(inc_rows)
    seats = next(t for h, t in sen if h == "Seats")
    hold = {}
    for r in _grid(seats):
        lab = _clean(r[0]) if r else ""
        if lab in ("Before these elections", "Not up") and len(r) >= 5:
            nums = [int(x) if x.isdigit() else 0 for x in (_clean(v) for v in r[-4:-1])]
            hold[lab] = {"D": nums[0], "I": nums[1], "R": nums[2]}
    raw["senate_composition"] = pl.DataFrame(
        [{"row": k, "party": p, "seats": v} for k, d in hold.items() for p, v in d.items()])

    # House members / candidates (one row per member; rowspans expanded)
    rows = []
    for h, t in _wikitables(paths["house"]):
        if h not in STATES:
            continue
        for r in _grid(t)[2:]:
            if len(r) < 7 or not _clean(r[0]).startswith(h):
                continue
            rows.append({"label": _clean(r[0]), "pvi": _clean(r[1]), "member": _clean(r[2]),
                         "party": _clean(r[3]), "first_elected": _clean(r[4]), "status": _clean(r[5]),
                         "candidates": r[6].replace("\xa0", " ")})
    raw["house_members"] = pl.DataFrame(rows)

    hr = _wikitables(paths["house_ratings"])
    raw["house_ratings"], a = _ratings_table(hr[0][1], "house")
    asof += a

    gov = _wikitables(paths["governor"])
    raw["governor_ratings"], a = _ratings_table(next(t for h, t in gov if h == "Predictions"), "governor")
    asof += a
    rows = []
    for r in _grid(next(t for h, t in gov if h == "States"))[1:]:
        if len(r) >= 7 and _clean(r[0]) in STATES:
            rows.append({"state": _clean(r[0]), "governor": _clean(r[1]), "party": _clean(r[2]),
                         "status": _clean(r[5]), "candidates": r[6].replace("\xa0", " ")})
    raw["governor_incumbents"] = pl.DataFrame(rows)
    raw["rating_asof"] = pl.DataFrame(asof)

    # Downballot: 2024 pres by CD on 2026 lines (vote totals tab)
    text = paths["downballot"].read_text(encoding="utf-8")
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if not r or not re.match(r"^[A-Z]{2}-(\d{2}|AL)$", r[0].strip()):
            continue
        num = lambda s: int(s.replace(",", "")) if s and s.replace(",", "").isdigit() else None  # noqa: E731
        rows.append({"district": r[0].strip(), "incumbent": r[1].strip(), "party": r[2].strip("() "),
                     "harris": num(r[4]), "trump": num(r[5]), "total": num(r[6]),
                     "margin_pct": r[9].strip(),
                     "has_2020_same_lines": bool(r[11].strip()) if len(r) > 11 else False})
    raw["downballot_pres24_cd"] = pl.DataFrame(rows)

    raw["senate_holdovers"] = senate_holdovers(raw)
    return raw


# ---------------------------------------------------------------- normalisation
def normalize_rating(s: str | None) -> str | None:
    if not s:
        return None
    s = _clean(s).replace("(flip)", "").strip()
    s = re.sub(r"(?i)toss-?up", "Tossup", s)
    m = re.match(r"(?i)^(safe|solid|likely|lean|tilt)\s+([DR])\b", s)
    if m:
        lvl = {"safe": "Safe", "solid": "Safe", "likely": "Likely", "lean": "Lean", "tilt": "Lean"}[m.group(1).lower()]
        return f"{lvl} {m.group(2).upper()}"
    if s.lower().startswith("tossup"):
        return "Tossup"
    return None


def derive_rating(margin: float) -> str:
    side = "D" if margin > 0 else "R"
    for thr, lvl in DERIVE_THRESHOLDS:
        if abs(margin) < thr:
            return "Tossup" if lvl == "Tossup" else f"{lvl} {side}"
    return f"Safe {side}"


def _pick_rating(rec: dict) -> tuple[str | None, str | None]:
    for rater in RATER_PRIORITY:
        v = normalize_rating(rec.get(rater))
        if v:
            return v, rater
    return None, None


def _district_id(label: str) -> str | None:
    label = _clean(label)
    for name, ab in sorted(STATES.items(), key=lambda kv: -len(kv[0])):
        if label.startswith(name):
            rest = label[len(name):].strip().lower()
            if rest.startswith("at-large") or rest == "at large":
                return f"{ab}-AL"
            m = re.match(r"^(\d+)", rest)
            return f"{ab}-{int(m.group(1)):02d}" if m else None
    return None


def _senate_race_id(label: str) -> str:
    label = _clean(label)
    special = bool(re.search(r"\((special|Class 3)\)", label))
    name = re.sub(r"\(.*?\)", "", label).strip()
    return f"SEN-{STATES[name]}" + ("-S" if special else "")


_RUNNING = re.compile(r"(?i)^(incumbent (renominated|running|advanced|re-?nominated|seeking)|interim appointee nominated)")


def _is_open(status: str) -> bool:
    return not _RUNNING.match(status or "")


def _party(p: str) -> str | None:
    p = _clean(p)
    if p in PARTY:
        return PARTY[p]
    if p.startswith("Democratic"):
        return "D"
    if p.startswith("Republican"):
        return "R"
    if p.startswith("Independent"):
        return "I"
    return None


_CAND = re.compile(r"([^\n▌()]+?)\s*\(([^()\n]+)\)")


def _parse_candidates(cell: str) -> list[tuple[str, str]]:
    out = []
    for line in re.split(r"[\n▌]+", cell or ""):
        line = _clean(line)
        m = _CAND.search(line)
        if m:
            out.append((m.group(1).strip(" .,"), m.group(2).strip()))
    return out


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _same_person(a: str, b: str) -> bool:
    la, lb = _clean(a).lower().split(), _clean(b).lower().split()
    sfx = {"jr.", "jr", "sr.", "iii", "ii", "iv"}
    la = [x for x in la if x not in sfx]
    lb = [x for x in lb if x not in sfx]
    return bool(la and lb) and la[-1] == lb[-1] and la[0][0] == lb[0][0]


def senate_holdovers(raw: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """D/R seats not up in 2026 = composition before the election minus incumbent party of seats up.
    Independents caucusing with Democrats count as D. Cross-checked against the page's 'Not up' row."""
    comp = raw["senate_composition"]
    before = {r["party"]: r["seats"] for r in comp.filter(pl.col("row") == "Before these elections").iter_rows(named=True)}
    up = {"D": 0, "R": 0}
    for r in raw["senate_incumbents"].iter_rows(named=True):
        up[_party(r["party"]) if _party(r["party"]) in ("D", "R") else "D"] += 1
    d = before.get("D", 0) + before.get("I", 0) - up["D"]
    rr = before.get("R", 0) - up["R"]
    notup = {r["party"]: r["seats"] for r in comp.filter(pl.col("row") == "Not up").iter_rows(named=True)}
    if notup and (notup.get("D", 0) + notup.get("I", 0), notup.get("R", 0)) != (d, rr):
        raise ValueError(f"holdover mismatch: computed D{d}/R{rr}, page says {notup}")
    return pl.DataFrame({"party": ["D", "R"], "seats": [d, rr]})


# ---------------------------------------------------------------- objects
def _state_margins(db: pl.DataFrame) -> dict[str, float]:
    s = (db.with_columns(pl.col("district").str.slice(0, 2).alias("st"))
           .group_by("st").agg(pl.col("harris").sum(), pl.col("trump").sum(), pl.col("total").sum()))
    return {r["st"]: round(100.0 * (r["harris"] - r["trump"]) / r["total"], 2) for r in s.iter_rows(named=True)}


def to_objects(raw: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    db = raw["downballot_pres24_cd"]
    cd_margin = {r["district"]: round(100.0 * (r["harris"] - r["trump"]) / r["total"], 2)
                 for r in db.iter_rows(named=True) if r["total"]}
    st_margin = _state_margins(db)
    races, cands = [], []
    meta = {"rating_source": {}, "derived": []}

    def add_cands(race_id: str, cell: str, incumbent_name: str | None, keep_independent_d_side: bool):
        parsed = [(n, _party(p) or p) for n, p in _parse_candidates(cell)]
        majors = [(n, p) for n, p in parsed if p in ("D", "R")]
        if keep_independent_d_side and not any(p == "D" for _, p in majors):
            ind = [(n, p) for n, p in parsed if p == "I"]
            majors += ind[:1]
        seen = set()
        for n, p in majors:
            cid = f"{race_id}:{_slug(n)}"
            if cid in seen:
                continue
            seen.add(cid)
            cands.append({"candidate_id": cid, "race_id": race_id, "name": n, "party": p,
                          "incumbent": bool(incumbent_name) and _same_person(n, incumbent_name)})

    # ---- Senate
    inc = {_senate_race_id(r["label"]): r for r in raw["senate_incumbents"].iter_rows(named=True)}
    for rec in raw["senate_ratings"].iter_rows(named=True):
        rid = _senate_race_id(rec["label"])
        st = rid.split("-")[1]
        i = inc.get(rid, {})
        ip = _party(i.get("party", ""))
        rating, src = _pick_rating(rec)
        m = st_margin.get(st)
        if rating is None:
            rating, src = derive_rating(m), "derived"
            meta["derived"].append(rid)
        meta["rating_source"][rid] = src
        races.append({"race_id": rid, "chamber": "senate", "state": st, "district": None,
                      "incumbent_party": "D" if ip == "I" else ip, "open_seat": _is_open(i.get("status", "")),
                      "pres24_margin": m, "rating": rating, "special": rid.endswith("-S")})
        add_cands(rid, i.get("candidates", ""), i.get("senator"), True)

    # ---- Governor
    ginc = {STATES[r["state"]]: r for r in raw["governor_incumbents"].iter_rows(named=True)}
    for rec in raw["governor_ratings"].iter_rows(named=True):
        st = STATES[re.sub(r"\(.*?\)", "", rec["label"]).strip()]
        rid = f"GOV-{st}"
        i = ginc.get(st, {})
        ip = _party(i.get("party", ""))
        rating, src = _pick_rating(rec)
        m = st_margin.get(st)
        if rating is None:
            rating, src = derive_rating(m), "derived"
            meta["derived"].append(rid)
        meta["rating_source"][rid] = src
        races.append({"race_id": rid, "chamber": "governor", "state": st, "district": None,
                      "incumbent_party": ip, "open_seat": _is_open(i.get("status", "")),
                      "pres24_margin": m, "rating": rating, "special": False})
        add_cands(rid, i.get("candidates", ""), i.get("governor"), True)

    # ---- House
    hr = {_district_id(r["label"]): r for r in raw["house_ratings"].iter_rows(named=True)}
    members: dict[str, list[dict]] = {}
    for r in raw["house_members"].iter_rows(named=True):
        did = _district_id(r["label"])
        if did:
            members.setdefault(did, []).append(r)
    districts = []
    for did in sorted(cd_margin):
        st, num = did.split("-")
        rid = f"HOUSE-{did}"
        ms = members.get(did, [])
        running = [x for x in ms if not _is_open(x["status"]) and _party(x["party"])]
        holders = running or [x for x in ms if _party(x["party"])]
        ip = _party(holders[0]["party"]) if holders else None
        if ip is None and ms:  # vacant seat: "Rep. X (R) resigned ..."
            mm = re.search(r"\(([DR])\)", ms[0]["party"] + " " + ms[0]["status"])
            ip = mm.group(1) if mm else None
        m = cd_margin[did]
        if did in hr:
            rating, src = _pick_rating(hr[did])
        else:
            # not listed as competitive by any rater on the ratings page => Safe for the favoured side
            rating, src = (f"Safe {'D' if m > 0 else 'R'}", "unlisted-safe")
        if rating is None:
            rating, src = derive_rating(m), "derived"
            meta["derived"].append(rid)
        meta["rating_source"][rid] = src
        races.append({"race_id": rid, "chamber": "house", "state": st, "district": num,
                      "incumbent_party": "D" if ip == "I" else ip, "open_seat": not running,
                      "pres24_margin": m, "rating": rating, "special": False})
        districts.append({"district_id": did, "state": st, "pres24_margin": m})
        if ms:
            add_cands(rid, ms[0]["candidates"], running[0]["member"] if running else None, False)

    race_df = pl.DataFrame(races, schema={
        "race_id": pl.Utf8, "chamber": pl.Utf8, "state": pl.Utf8, "district": pl.Utf8,
        "incumbent_party": pl.Utf8, "open_seat": pl.Boolean, "pres24_margin": pl.Float64,
        "rating": pl.Utf8, "special": pl.Boolean})
    cand_df = pl.DataFrame(cands, schema={"candidate_id": pl.Utf8, "race_id": pl.Utf8, "name": pl.Utf8,
                                          "party": pl.Utf8, "incumbent": pl.Boolean})
    cand_df = cand_df.unique("candidate_id", keep="first", maintain_order=True)
    to_objects.meta = meta  # type: ignore[attr-defined]  # diagnostics for reporting
    return {"Race": race_df, "District": pl.DataFrame(districts), "Candidate": cand_df}
