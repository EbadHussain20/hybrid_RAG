#!/usr/bin/env python3
"""Usage: python scripts/ingest.py [--force]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.indexer import run_ingestion  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Ingest documents from the docs/ directory.")
    parser.add_argument("--force", action="store_true", help="Re-ingest even unchanged files.")
    args = parser.parse_args()

    results = run_ingestion(force=args.force)
    ok = [r for r in results if not r.error]
    errors = [r for r in results if r.error]
    skipped = [r for r in results if r.skipped]

    print(f"Processed {len(results)} files:")
    print(f"  - {len(ok) - len(skipped)} newly parsed/chunked")
    print(f"  - {len(skipped)} unchanged (skipped)")
    print(f"  - {len(errors)} errors")
    for r in errors:
        print(f"    ERROR [{r.document_id}]: {r.error}")

    total_chunks = sum(len(r.chunks) for r in results)
    print(f"Total new chunks produced this run: {total_chunks}")


if __name__ == "__main__":
    main()
