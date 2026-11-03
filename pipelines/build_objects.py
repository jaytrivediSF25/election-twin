"""Batch pipeline: for each ingest module, fetch -> raw_* datasets -> objects (via Funnel)."""
from __future__ import annotations

import importlib
import traceback

from ontology import funnel

MODULES = ["races", "county_2024", "polls", "markets"]


def build(con, modules: list[str] | None = None) -> dict[str, str]:
    status: dict[str, str] = {}
    for name in modules or MODULES:
        try:
            mod = importlib.import_module(f"ingest.{name}")
            raw = mod.fetch()
            sources = getattr(mod, "SOURCES", {})
            for rname, df in raw.items():
                funnel.write_raw(con, rname, df, source=f"ingest.{name}", source_url=sources.get(rname, ""))
            objs = mod.to_objects(raw)
            counts = {}
            for type_name, df in objs.items():
                counts[type_name] = funnel.write_batch(con, type_name, df, mode="upsert", source=f"ingest.{name}")
            status[name] = "ok " + ", ".join(f"{k}={v}" for k, v in counts.items())
        except ModuleNotFoundError as e:
            status[name] = f"missing ({e.name})"
        except Exception as e:
            traceback.print_exc()
            status[name] = f"FAILED: {type(e).__name__}: {e}"
    return status
