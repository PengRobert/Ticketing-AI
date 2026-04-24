"""
data_processor.py
-----------------
Aggregates multi-row historical ticket CSV(s) into one-row-per-ticket format
and outputs a clean processed CSV ready for ChromaDB indexing.

Input columns (actual schema from source system):
    APPLICATION_NAME, RESOLUTION_CODE, RESOLUTION, TICKET_NO, OPEN_DATE,
    END_STATE_DATE_NEXT_BUSINESS_DATE, CLOSED_DATE, PRIORITY, RESPOND_SLA,
    ASSIGNEE_NAME, BRIEF_DESCRIPTION, DURATION, WORKNOTE, PRODUCT_TYPE, TYPE

Priority values normalised:
    "1 - Critical" → P1   "2 - High"     → P2
    "3 - Medium"   → P3   "3 - Moderate" → P3   "4 - Low" → P4

Usage:
    # Single file
    python data_processor.py --input data/raw/tickets.csv

    # Multiple files (merged before processing)
    python data_processor.py --input data/raw/file1.csv data/raw/file2.csv

    # Auto-discover every CSV in data/raw/
    python data_processor.py --auto
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column constants
# ---------------------------------------------------------------------------
COL_TICKET     = "TICKET_NO"
COL_APP        = "APPLICATION_NAME"
COL_RES_CODE   = "RESOLUTION_CODE"
COL_RESOLUTION = "RESOLUTION"
COL_OPEN       = "OPEN_DATE"
COL_CLOSED     = "CLOSED_DATE"
COL_PRIORITY   = "PRIORITY"
COL_SLA        = "RESPOND_SLA"
COL_ASSIGNEE   = "ASSIGNEE_NAME"
COL_BRIEF      = "BRIEF_DESCRIPTION"
COL_DURATION   = "DURATION"
COL_WORKNOTE   = "WORKNOTE"
COL_PRODUCT    = "PRODUCT_TYPE"
COL_TYPE       = "TYPE"

# Alias so aggregate_tickets() can reference it before it's used
COL_RESOLUTION_CODE = COL_RES_CODE

REQUIRED_COLUMNS = {
    COL_TICKET, COL_APP, COL_RES_CODE, COL_RESOLUTION,
    COL_OPEN, COL_PRIORITY, COL_SLA, COL_ASSIGNEE,
    COL_BRIEF, COL_WORKNOTE, COL_PRODUCT, COL_TYPE,
}

# CLOSED_DATE is present but sometimes empty; END_STATE_DATE_NEXT_BUSINESS_DATE
# is a bonus column we carry through but don't actively use in aggregation.
OPTIONAL_COLUMNS = {"CLOSED_DATE", "END_STATE_DATE_NEXT_BUSINESS_DATE", COL_DURATION}

# ---------------------------------------------------------------------------
# Priority normalisation
# ---------------------------------------------------------------------------
_PRIORITY_MAP = {
    # Numeric prefixes
    "1": "P1", "1 - critical": "P1", "1 - critical/emergency": "P1",
    "2": "P2", "2 - high": "P2", "2 - urgent": "P2",
    "3": "P3", "3 - medium": "P3", "3 - moderate": "P3",
                "3 - average": "P3", "3 - normal": "P3",
    "4": "P4", "4 - low": "P4", "4 - minor": "P4",
    # Plain labels
    "critical": "P1", "emergency": "P1",
    "high": "P2", "urgent": "P2",
    "medium": "P3", "moderate": "P3", "average": "P3",
    "normal": "P3", "standard": "P3",
    "low": "P4", "minor": "P4",
    # Already normalised
    "p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4",
}


def _normalize_priority(val) -> str:
    if pd.isna(val):
        return "P3"          # sensible default
    key = str(val).strip().lower()
    return _PRIORITY_MAP.get(key, str(val).strip())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _last_non_null(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[-1] if not valid.empty else None


def _first_non_null(series: pd.Series):
    valid = series.dropna()
    return valid.iloc[0] if not valid.empty else None


def _concat_worknotes(series: pd.Series) -> str:
    notes = series.dropna().astype(str).str.strip()
    notes = notes[notes != ""]
    return "\n---\n".join(notes)


def _parse_resolution_fields(resolution_text: str) -> dict:
    """
    Parse structured RESOLUTION field into Problem, RCA, and Solution.

    Expected format:
        Problem: <text>
        RCA: <text>
        Solution: <text>
    """
    result = {"PROBLEM": "", "RCA": "", "SOLUTION": ""}
    if not isinstance(resolution_text, str) or not resolution_text.strip():
        return result

    current_key = None
    buffer: list[str] = []

    for line in resolution_text.splitlines():
        stripped = line.strip()
        lower = stripped.lower()
        if lower.startswith("problem:"):
            if current_key:
                result[current_key] = " ".join(buffer).strip()
            current_key = "PROBLEM"
            buffer = [stripped[len("problem:"):].strip()]
        elif lower.startswith("rca:"):
            if current_key:
                result[current_key] = " ".join(buffer).strip()
            current_key = "RCA"
            buffer = [stripped[len("rca:"):].strip()]
        elif lower.startswith("solution:"):
            if current_key:
                result[current_key] = " ".join(buffer).strip()
            current_key = "SOLUTION"
            buffer = [stripped[len("solution:"):].strip()]
        else:
            if current_key:
                buffer.append(stripped)

    if current_key:
        result[current_key] = " ".join(buffer).strip()

    return result


# ---------------------------------------------------------------------------
# Multi-file loader
# ---------------------------------------------------------------------------

def load_and_merge(input_paths: list[str]) -> pd.DataFrame:
    """Read one or more CSV files and concatenate them into a single DataFrame."""
    frames = []
    for p in input_paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        log.info("Reading %s …", path.name)
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df.replace("", pd.NA, inplace=True)
        frames.append(df)
        log.info("  → %d rows, %d cols", len(df), len(df.columns))

    merged = pd.concat(frames, ignore_index=True)
    log.info("Merged total: %d rows from %d file(s).", len(merged), len(frames))
    return merged


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {sorted(missing)}\n"
            f"Found columns: {sorted(df.columns.tolist())}"
        )


def validate_output(df: pd.DataFrame) -> None:
    blank_resolution = df[COL_RESOLUTION].isna().sum()
    blank_worknote = (df[COL_WORKNOTE] == "").sum()
    if blank_resolution > 0:
        log.warning("%d tickets have no RESOLUTION text.", blank_resolution)
    if blank_worknote > 0:
        log.warning("%d tickets have no WORKNOTE entries.", blank_worknote)


# ---------------------------------------------------------------------------
# Core aggregation
# ---------------------------------------------------------------------------

def aggregate_tickets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group by TICKET_NO and collapse all transaction rows into one summary row.

    Aggregation strategy:
        TICKET_NO          → group key
        APPLICATION_NAME   → first (stable metadata)
        RESOLUTION_CODE    → last non-null (final classification)
        RESOLUTION         → last non-null (final resolution text)
        OPEN_DATE          → min (earliest)
        CLOSED_DATE        → max (latest)
        PRIORITY           → first non-null, normalised to P1–P4
        RESPOND_SLA        → last non-null (final SLA outcome)
        ASSIGNEE_NAME      → last non-null (who closed it)
        BRIEF_DESCRIPTION  → first (original description)
        DURATION           → sum
        WORKNOTE           → all entries concatenated
        PRODUCT_TYPE       → first
        TYPE               → first
    Derived:
        PROBLEM / RCA / SOLUTION → parsed from RESOLUTION text
        CYCLE_TIME_HOURS         → CLOSED_DATE − OPEN_DATE
    """
    log.info(
        "Aggregating %d rows across %d unique tickets …",
        len(df), df[COL_TICKET].nunique(),
    )

    # Coerce date columns
    for col in [COL_OPEN, COL_CLOSED]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Coerce duration to numeric
    if COL_DURATION in df.columns:
        df[COL_DURATION] = pd.to_numeric(df[COL_DURATION], errors="coerce")

    grouped = df.sort_values(COL_OPEN).groupby(COL_TICKET, sort=False)

    records = []
    for ticket_no, group in grouped:
        rec: dict = {COL_TICKET: ticket_no}

        rec[COL_APP]      = _first_non_null(group[COL_APP])
        rec[COL_PRODUCT]  = _first_non_null(group[COL_PRODUCT])
        rec[COL_TYPE]     = _first_non_null(group[COL_TYPE])
        rec[COL_BRIEF]    = _first_non_null(group[COL_BRIEF])
        rec[COL_ASSIGNEE] = _last_non_null(group[COL_ASSIGNEE])

        # Priority: take first non-null and normalise
        raw_pri = _first_non_null(group[COL_PRIORITY])
        rec[COL_PRIORITY] = _normalize_priority(raw_pri)

        rec[COL_OPEN]   = group[COL_OPEN].min() if COL_OPEN in group.columns else None
        rec[COL_CLOSED] = group[COL_CLOSED].max() if COL_CLOSED in group.columns else None

        rec[COL_RESOLUTION_CODE] = _last_non_null(group[COL_RES_CODE])
        rec[COL_RESOLUTION]      = _last_non_null(group[COL_RESOLUTION])
        rec[COL_SLA]             = _last_non_null(group[COL_SLA])

        rec[COL_DURATION] = (
            group[COL_DURATION].sum(min_count=1)
            if COL_DURATION in group.columns else None
        )
        rec[COL_WORKNOTE] = _concat_worknotes(group[COL_WORKNOTE])

        # Parse structured resolution into separate columns
        parsed = _parse_resolution_fields(rec[COL_RESOLUTION] or "")
        rec["PROBLEM"]  = parsed["PROBLEM"]
        rec["RCA"]      = parsed["RCA"]
        rec["SOLUTION"] = parsed["SOLUTION"]

        # Derived: cycle time in hours
        if pd.notna(rec.get(COL_OPEN)) and pd.notna(rec.get(COL_CLOSED)):
            delta = rec[COL_CLOSED] - rec[COL_OPEN]
            rec["CYCLE_TIME_HOURS"] = round(delta.total_seconds() / 3600, 2)
        else:
            rec["CYCLE_TIME_HOURS"] = None

        records.append(rec)

    result = pd.DataFrame(records)
    log.info("Aggregation complete: %d unique tickets.", len(result))
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"  Total unique tickets : {len(df)}")

    if COL_RES_CODE in df.columns:
        print(f"\n  Resolution Code distribution:")
        for code, cnt in df[COL_RES_CODE].value_counts().items():
            print(f"    {str(code):<45} {cnt}")

    if COL_PRIORITY in df.columns:
        print(f"\n  Priority distribution (normalised):")
        for pri, cnt in df[COL_PRIORITY].value_counts().sort_index().items():
            print(f"    {str(pri):<10} {cnt}")

    if COL_SLA in df.columns:
        print(f"\n  SLA distribution:")
        for sla, cnt in df[COL_SLA].value_counts().items():
            print(f"    {str(sla):<10} {cnt}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Public process() function
# ---------------------------------------------------------------------------

def process(input_paths: list[str], output_path: str) -> pd.DataFrame:
    output_p = Path(output_path)
    output_p.parent.mkdir(parents=True, exist_ok=True)

    df_raw = load_and_merge(input_paths)
    validate_columns(df_raw)
    log.info("Combined raw shape: %s", df_raw.shape)

    df_processed = aggregate_tickets(df_raw)
    validate_output(df_processed)

    df_processed.to_csv(output_p, index=False)
    log.info("Saved: %s  (%d rows)", output_p, len(df_processed))

    _print_summary(df_processed)
    return df_processed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate multi-row ticket CSV(s) into one-row-per-ticket."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--input",
        nargs="+",
        metavar="FILE",
        help="One or more raw CSV files to merge and process.",
    )
    group.add_argument(
        "--auto",
        action="store_true",
        help="Auto-discover and merge all CSVs in data/raw/.",
    )
    parser.add_argument(
        "--output",
        default="data/processed/tickets_processed.csv",
        help="Output path (default: data/processed/tickets_processed.csv)",
    )
    args = parser.parse_args()

    if args.auto:
        raw_dir = Path("data/raw")
        input_paths = sorted(str(p) for p in raw_dir.glob("*.csv"))
        if not input_paths:
            log.error("No CSV files found in data/raw/")
            sys.exit(1)
        log.info("Auto-discovered %d file(s): %s", len(input_paths),
                 [Path(p).name for p in input_paths])
    elif args.input:
        input_paths = args.input
    else:
        # Default: process all CSVs in data/raw/
        raw_dir = Path("data/raw")
        input_paths = sorted(str(p) for p in raw_dir.glob("*.csv"))
        if not input_paths:
            parser.error(
                "No files specified and no CSVs found in data/raw/. "
                "Use --input or --auto."
            )

    process(input_paths, args.output)


if __name__ == "__main__":
    main()
