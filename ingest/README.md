# Ingest contract

Each module `ingest/<name>.py` exposes:

    def fetch() -> dict[str, pl.DataFrame]          # raw datasets, as close to source as possible
    def to_objects(raw: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]   # ObjectType name -> rows
    SOURCES: dict[str, str]                          # raw name -> source URL (lineage)

Downloads are cached under data/raw/<name>/ (untrusted data, never executed).
`pipelines/build_objects.py` calls fetch -> funnel.write_raw -> to_objects -> funnel.write_batch.
Columns in to_objects output must be properties from ontology/schema.yaml.

## IDs (shared across all sources)
- Senate: `SEN-GA`; special elections: `SEN-OH-S`
- House: `HOUSE-PA-07` (two-digit), at-large: `HOUSE-AK-AL`
- Governor: `GOV-AZ`
- Chamber control: `CONTROL-HOUSE`, `CONTROL-SENATE`; generic ballot: `GENERIC`
- County: 5-digit FIPS string
- Margins are always D minus R in points (R+3 = -3.0). Where the main non-Republican is an
  independent the D side caucuses with or is backed by Dems (e.g. NE Senate), treat them as the "D" side.
