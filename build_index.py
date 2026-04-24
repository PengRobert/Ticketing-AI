"""
build_index.py
--------------
One-command script to build the ChromaDB vector index from the processed CSV.

Run this once after data_processor.py — and again whenever you update the data.

Usage:
    python build_index.py                    # standard build
    python build_index.py --force            # delete and rebuild from scratch
    python build_index.py --csv path/to.csv  # custom processed CSV path
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

from src.rag.indexer import index_tickets


def main():
    parser = argparse.ArgumentParser(
        description="Build ChromaDB index from processed ticket CSV."
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to processed CSV (default: data/processed/tickets_processed.csv)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete existing index and rebuild from scratch",
    )
    args = parser.parse_args()

    count = index_tickets(csv_path=args.csv, force_reindex=args.force)
    print(f"\n✓ ChromaDB index ready — {count} documents indexed.\n")


if __name__ == "__main__":
    main()
