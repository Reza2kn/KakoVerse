#!/usr/bin/env python3
"""Summarise presenting_problem categories across persona cards."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

DEFAULT_KEYWORD_MAP: Dict[str, Tuple[str, ...]] = {
    "immigration": (
        r"\bimmigrat",
        r"\basylum",
        r"\bvisa",
        r"relocation",
        r"displacement",
    ),
    "grief_bereavement": (
        r"\bgrief",
        r"\bgriev",
        r"loss of",
        r"bereave",
        r"widow",
    ),
    "relationship_breakup": (
        r"breakup",
        r"divorce",
        r"separation",
        r"estrang",
    ),
    "family_conflict": (
        r"family conflict",
        r"parent",
        r"sibling",
        r"custody",
        r"family dispute",
    ),
    "loneliness_isolation": (
        r"lonely",
        r"isolation",
        r"social withdrawal",
        r"disconnected",
    ),
    "chronic_health_pain": (
        r"chronic pain",
        r"autoimmune",
        r"cancer",
        r"long[- ]?covid",
        r"illness",
        r"diagnos",
    ),
    "mental_health_crisis": (
        r"depress",
        r"anxiet",
        r"panic",
        r"burnout",
        r"trauma",
    ),
    "financial_employment": (
        r"job loss",
        r"unemploy",
        r"layoff",
        r"financial",
        r"debt",
        r"bankrupt",
    ),
    "caregiver_burden": (
        r"caregiver",
        r"caring for",
        r"caretaker",
        r"aging parent",
    ),
    "violence_abuse": (
        r"abuse",
        r"violence",
        r"assault",
        r"coerc",
    ),
    "substance_use": (
        r"addict",
        r"substance",
        r"alcohol",
        r"relapse",
    ),
    "legal_housing": (
        r"evict",
        r"legal",
        r"lawsuit",
        r"housing",
        r"landlord",
    ),
    "academic_stress": (
        r"exam",
        r"school",
        r"university",
        r"thesis",
    ),
    "parenting": (
        r"parenting",
        r"child custody",
        r"pregnan",
        r"fertility",
    ),
}


def load_persona_files(persona_dir: Path) -> Iterable[Path]:
    yield from sorted(persona_dir.glob("*.json"))


def extract_presenting_problem(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    presenting = data.get("presenting_problem") or {}
    summary = presenting.get("summary")
    return summary or ""


def classify_problem(summary: str, keyword_map: Dict[str, Tuple[str, ...]]) -> List[str]:
    summary_lower = summary.lower()
    categories: List[str] = []
    for category, patterns in keyword_map.items():
        if any(re.search(pattern, summary_lower) for pattern in patterns):
            categories.append(category)
    if not categories:
        categories.append("other")
    return categories


def analyze(persona_dir: Path, keyword_map: Dict[str, Tuple[str, ...]]) -> Tuple[Counter[str], Dict[str, List[str]]]:
    counter: Counter[str] = Counter()
    details: Dict[str, List[str]] = defaultdict(list)
    for persona_path in load_persona_files(persona_dir):
        summary = extract_presenting_problem(persona_path)
        categories = classify_problem(summary, keyword_map)
        for category in categories:
            counter[category] += 1
            details[category].append(summary)
    return counter, details


def try_plot(counter: Counter[str], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        print("matplotlib not installed; skipping chart output.")
        return

    items = counter.most_common()
    if not items:
        print("No presenting problems found.")
        return
    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    plt.figure(figsize=(10, max(4, len(labels) * 0.3)))
    positions = list(range(len(labels)))[::-1]
    plt.barh(positions, values)
    plt.yticks(positions, labels)
    plt.xlabel("Occurrences")
    plt.title("Presenting problem categories")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved bar chart to {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Categorise persona presenting problems.")
    parser.add_argument(
        "--persona-dir",
        type=Path,
        default=Path("Artifacts/personas"),
        help="Directory containing persona JSON files.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("Artifacts/presenting_problem_distribution.csv"),
        help="Where to write the category counts.",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=Path("Artifacts/presenting_problem_distribution.png"),
        help="Where to save the bar chart (requires matplotlib).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    keyword_map = DEFAULT_KEYWORD_MAP
    counter, _ = analyze(args.persona_dir, keyword_map)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8") as handle:
        handle.write("category,count\n")
        for category, count in counter.most_common():
            handle.write(f"{category},{count}\n")
    print(f"Wrote counts to {args.output_csv}")

    try_plot(counter, args.chart)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
