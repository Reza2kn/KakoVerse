#!/usr/bin/env python3
"""Create a deterministic age→category plan for each persona."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence


DEFAULT_CATEGORIES: List[str] = [
    "mental_health_crisis",
    "immigration",
    "grief_bereavement",
    "relationship_breakup",
    "loneliness_isolation",
    "chronic_health_pain",
    "financial_employment",
    "caregiver_burden",
    "violence_abuse",
    "legal_housing",
    "academic_stress",
]


def load_persona_ids(persona_dir: Path) -> List[str]:
    ids = []
    for path in sorted(persona_dir.glob("seeker_auto_*.json")):
        ids.append(path.stem)
    if not ids:
        raise FileNotFoundError(f"No persona JSON files found in {persona_dir}")
    return ids


def build_category_cycle(persona_index: int, categories: Sequence[str]) -> List[str]:
    rotation = persona_index % len(categories)
    return list(categories[rotation:] + categories[:rotation])


def assign_categories(
    persona_ids: Sequence[str],
    categories: Sequence[str],
    ages: Sequence[int],
) -> Dict[str, Dict[str, str]]:
    plan: Dict[str, Dict[str, str]] = {}
    for idx, persona_id in enumerate(persona_ids):
        cycle = build_category_cycle(idx, categories)
        assignments: Dict[str, str] = {}
        for step, age in enumerate(ages):
            category = cycle[step % len(categories)]
            assignments[str(age)] = category
        plan[persona_id] = assignments
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a balanced crisis category plan for personas."
    )
    parser.add_argument(
        "--persona-dir",
        type=Path,
        default=Path("Artifacts/personas"),
        help="Directory containing persona JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Artifacts/crisis_category_plan.json"),
        help="Path to write the plan JSON.",
    )
    parser.add_argument(
        "--age-start",
        type=int,
        default=20,
        help="Starting age (inclusive).",
    )
    parser.add_argument(
        "--age-end",
        type=int,
        default=100,
        help="Ending age (inclusive).",
    )
    parser.add_argument(
        "--categories",
        type=str,
        nargs="*",
        default=DEFAULT_CATEGORIES,
        help="Category labels to cycle through (default: 11 crisis categories).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.age_start >= args.age_end:
        parser.error("age-start must be less than age-end")
        return 2
    if len(args.categories) < 2:
        parser.error("Provide at least two categories.")
        return 2

    persona_ids = load_persona_ids(args.persona_dir)
    ages = list(range(args.age_start, args.age_end + 1))
    plan = assign_categories(persona_ids, args.categories, ages)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "meta": {
                    "persona_count": len(persona_ids),
                    "age_range": [args.age_start, args.age_end],
                    "categories": args.categories,
                },
                "plan": plan,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    print(
        f"Wrote plan for {len(persona_ids)} personas covering ages {args.age_start}-{args.age_end} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
