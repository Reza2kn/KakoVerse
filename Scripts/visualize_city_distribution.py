#!/usr/bin/env python3
"""Tabulate and optionally chart persona city usage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


def iter_personas(persona_dir: Path) -> Iterable[Path]:
    for path in sorted(persona_dir.glob("*.json")):
        yield path


def load_included_keys(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    meta = data.get("meta", {})
    included = meta.get("included_cities", [])
    if isinstance(included, list):
        return [str(item) for item in included]
    return []


def write_csv(counter: Counter[str], output_path: Path) -> None:
    lines = ["city|country|decade,count"]
    for key, count in counter.most_common():
        lines.append(f"{key},{count}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def try_plot(counter: Counter[str], output_path: Path, top: int | None) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping chart output.")
        return

    items = counter.most_common(top)
    if not items:
        print("No data to plot.")
        return
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    plt.figure(figsize=(14, max(4, len(items) * 0.3)))
    positions = list(range(len(labels)))[::-1]
    plt.barh(positions, values)
    plt.yticks(positions, labels)
    plt.xlabel("Occurrences")
    plt.title("Persona included_cities usage")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved chart to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize persona city usage and produce a CSV/PNG report."
    )
    parser.add_argument(
        "--persona-dir",
        type=Path,
        default=Path("Artifacts/personas"),
        help="Directory containing persona JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("Artifacts/city_distribution.csv"),
        help="Where to write the usage counts CSV.",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("Artifacts/city_distribution.png"),
        help="Where to save the bar chart (requires matplotlib).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Optional limit for number of bars in the chart (default: all).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    counter: Counter[str] = Counter()
    persona_paths = list(iter_personas(args.persona_dir))
    if not persona_paths:
        print(f"No persona JSON files found in {args.persona_dir}")
        return 1
    for path in persona_paths:
        counter.update(load_included_keys(path))

    write_csv(counter, args.output_csv)
    print(f"Wrote counts to {args.output_csv}")
    try_plot(counter, args.chart, args.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
