#!/usr/bin/env python3
"""Usage: python scripts/index.py [--force-full-rebuild]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.indexer import build_indexes  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Build dense + sparse indexes from ingested documents.")
    parser.add_argument("--force-full-rebuild", action="store_true",
                         help="Re-ingest and re-embed everything, ignoring the incremental cache.")
    args = parser.parse_args()

    stats = build_indexes(force_full_rebuild=args.force_full_rebuild)
    print("Index build complete:")
    print(f"  Documents indexed: {stats['num_documents']}")
    print(f"  Chunks indexed:    {stats['num_chunks']}")
    print(f"  Changed documents this run: {stats['changed_documents']}")


if __name__ == "__main__":
    main()
