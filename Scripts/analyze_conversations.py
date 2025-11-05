#!/usr/bin/env python3
"""Aggregate conversation crisis contexts by category and age."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Tuple, Dict


def load_conversations(conversation_dir: Path) -> Iterable[Tuple[str, dict]]:
    for path in sorted(conversation_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        yield path.name, data


def decade_of(age: int) -> str:
    decade = (age // 10) * 10
    return f"{decade}s"


def analyze(conversation_dir: Path) -> Tuple[Counter, Counter, Dict[str, Counter]]:
    category_counts: Counter[str] = Counter()
    age_counts: Counter[int] = Counter()
    decade_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    for name, data in load_conversations(conversation_dir):
        context = data.get("crisis_context") or {}
        category = str(context.get("category", "") or "other").strip().lower()
        age_value = context.get("age")

        category_counts[category] += 1

        try:
            age = int(age_value)
        except (TypeError, ValueError):
            continue

        age_counts[age] += 1
        decade_counts[decade_of(age)][category] += 1

    return category_counts, age_counts, decade_counts


def write_counter_csv(counter: Counter, path: Path, header: Tuple[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{header[0]},{header[1]}\n")
        for key, value in sorted(counter.items()):
            handle.write(f"{key},{value}\n")


def write_decade_csv(decade_counts: Dict[str, Counter[str]], path: Path) -> None:
    categories = sorted({cat for counter in decade_counts.values() for cat in counter} | {"other"})
    decades = sorted(decade_counts.keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("decade," + ",".join(categories) + "\n")
        for decade in decades:
            counts = [str(decade_counts[decade].get(cat, 0)) for cat in categories]
            handle.write(f"{decade}," + ",".join(counts) + "\n")


def try_plot(counter: Counter, path: Path, *, title: str, xlabel: str, ylabel: str) -> None:
    if not counter:
        print(f"No data to plot for {path.name}; skipping chart.")
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping chart output.")
        return

    items = sorted(counter.items())
    labels = [str(item[0]) for item in items]
    values = [item[1] for item in items]

    plt.figure(figsize=(max(8, len(labels) * 0.4), 5))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=220)
    plt.close()
    print(f"Saved chart to {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyse conversation crisis contexts.")
    parser.add_argument(
        "--conversation-dir",
        type=Path,
        default=Path("artifacts/conversations"),
        help="Directory containing conversation JSON files.",
    )
    parser.add_argument(
        "--category-csv",
        type=Path,
        default=Path("artifacts/conversation_category_distribution.csv"),
        help="Output CSV for crisis categories.",
    )
    parser.add_argument(
        "--category-chart",
        type=Path,
        default=Path("artifacts/conversation_category_distribution.png"),
        help="Output bar chart for crisis categories.",
    )
    parser.add_argument(
        "--age-csv",
        type=Path,
        default=Path("artifacts/conversation_age_distribution.csv"),
        help="Output CSV for age counts.",
    )
    parser.add_argument(
        "--age-chart",
        type=Path,
        default=Path("artifacts/conversation_age_distribution.png"),
        help="Output bar chart for age counts.",
    )
    parser.add_argument(
        "--decade-csv",
        type=Path,
        default=Path("artifacts/conversation_category_by_decade.csv"),
        help="Output CSV for category counts by decade.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    category_counts, age_counts, decade_counts = analyze(args.conversation_dir)
    if "other" not in category_counts:
        category_counts["other"] = 0

    write_counter_csv(category_counts, args.category_csv, ("category", "count"))
    write_counter_csv(age_counts, args.age_csv, ("age", "count"))
    write_decade_csv(decade_counts, args.decade_csv)

    try_plot(
        category_counts,
        args.category_chart,
        title="Conversation crisis categories",
        xlabel="Category",
        ylabel="Conversations",
    )
    try_plot(
        age_counts,
        args.age_chart,
        title="Conversation age distribution",
        xlabel="Age",
        ylabel="Conversations",
    )
    print(f"Wrote category counts to {args.category_csv}")
    print(f"Wrote age counts to {args.age_csv}")
    print(f"Wrote decade breakdown to {args.decade_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
