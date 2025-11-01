#!/usr/bin/env python3
"""Aggregate crisis profile categories by age/decade."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Tuple


def load_profiles(profile_dir: Path) -> Iterable[Tuple[str, Dict[str, Dict[str, object]]]]:
    for path in sorted(profile_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        yield path.name, data


def decade_of(age_str: str) -> str:
    age = int(age_str)
    decade = (age // 10) * 10
    return f"{decade}s"


def analyze(profile_dir: Path) -> Tuple[Counter[str], Dict[str, Counter[str]]]:
    overall: Counter[str] = Counter()
    by_decade: Dict[str, Counter[str]] = defaultdict(Counter)
    for _, profile in load_profiles(profile_dir):
        for age_str, entry in profile.items():
            category = str(entry.get("category", "unknown")).lower()
            overall[category] += 1
            by_decade[decade_of(age_str)][category] += 1
    return overall, by_decade


def write_overall_csv(counter: Counter[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("category,count\n")
        for category, count in counter.most_common():
            handle.write(f"{category},{count}\n")


def write_decade_csv(by_decade: Dict[str, Counter[str]], path: Path) -> None:
    categories = sorted({cat for counter in by_decade.values() for cat in counter})
    decades = sorted(by_decade.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("decade," + ",".join(categories) + "\n")
        for decade in decades:
            counts = [str(by_decade[decade].get(cat, 0)) for cat in categories]
            handle.write(f"{decade}," + ",".join(counts) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise crisis profile categories by age/decade."
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("Artifacts/crisis_profiles"),
        help="Directory containing crisis profile JSON files.",
    )
    parser.add_argument(
        "--overall-csv",
        type=Path,
        default=Path("Artifacts/crisis_profile_categories_overall.csv"),
    )
    parser.add_argument(
        "--decade-csv",
        type=Path,
        default=Path("Artifacts/crisis_profile_categories_by_decade.csv"),
    )
    args = parser.parse_args()

    overall, by_decade = analyze(args.profile_dir)
    write_overall_csv(overall, args.overall_csv)
    write_decade_csv(by_decade, args.decade_csv)
    print(f"Wrote overall counts to {args.overall_csv}")
    print(f"Wrote decade breakdown to {args.decade_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
