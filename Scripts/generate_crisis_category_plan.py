#!/usr/bin/env python3
"""Create a life-stage-aware age→category plan for each persona."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


DEFAULT_CATEGORIES: List[str] = [
    "Anger Management Issues",
    "Anxiety Disorders",
    "Bipolar Disorder",
    "Death of a Loved One",
    "Emotional Fluctuations",
    "Grief and Loss",
    "Identity Crises",
    "Obsessive-Compulsive Disorder (OCD)",
    "Ongoing Depression",
    "Post-Traumatic Stress Disorder (PTSD)",
    "Schizophrenia",
    "Self-Esteem Issues",
    "Spirituality and Faith",
    "Sexual Orientation",
    "Sexual Assault or Domestic Violence Recovery",
    "Academic Pressure",
    "Burnout",
    "Chronic Stress",
    "Financial Problems",
    "Health Problems",
    "Job Crisis",
    "Life Transitions (e.g., Retirement, Relocation)",
    "Workplace Stress",
    "Breakups or Divorce",
    "Conflicts or Communication Problems",
    "Issues with Children",
    "Issues with Parents",
    "Marital Problems",
    "Problems with Friends",
    "School Bullying",
    "Culture Shock",
    "Appearance Anxiety",
    "Career Development Issues",
    "Goal Setting Issues",
    "Motivation Problems",
    "Personal Growth Challenges",
    "Procrastination",
    "Sleep Problems",
    "Addictive Behaviors (e.g., Drug Use, Gambling)",
    "Alcohol Abuse",
    "Compulsive Behaviors",
    "Eating Disorders",
    "Internet Addiction",
    "Self-Harm Behaviors",
    "Debt Problems",
]

CATEGORY_RULES: Dict[str, Tuple[int, int]] = {
    "Anger Management Issues": (20, 90),
    "Anxiety Disorders": (20, 90),
    "Bipolar Disorder": (20, 80),
    "Death of a Loved One": (20, 100),
    "Emotional Fluctuations": (20, 70),
    "Grief and Loss": (20, 100),
    "Identity Crises": (20, 55),
    "Obsessive-Compulsive Disorder (OCD)": (20, 80),
    "Ongoing Depression": (20, 90),
    "Post-Traumatic Stress Disorder (PTSD)": (20, 90),
    "Schizophrenia": (20, 75),
    "Self-Esteem Issues": (20, 65),
    "Spirituality and Faith": (25, 100),
    "Sexual Orientation": (20, 55),
    "Sexual Assault or Domestic Violence Recovery": (20, 90),
    "Academic Pressure": (20, 45),
    "Burnout": (25, 70),
    "Chronic Stress": (20, 90),
    "Financial Problems": (20, 80),
    "Health Problems": (35, 100),
    "Job Crisis": (25, 65),
    "Life Transitions (e.g., Retirement, Relocation)": (45, 100),
    "Workplace Stress": (25, 65),
    "Breakups or Divorce": (22, 80),
    "Conflicts or Communication Problems": (20, 90),
    "Issues with Children": (28, 85),
    "Issues with Parents": (20, 70),
    "Marital Problems": (25, 80),
    "Problems with Friends": (20, 70),
    "School Bullying": (20, 30),
    "Culture Shock": (20, 85),
    "Appearance Anxiety": (20, 65),
    "Career Development Issues": (24, 60),
    "Goal Setting Issues": (20, 60),
    "Motivation Problems": (20, 65),
    "Personal Growth Challenges": (20, 90),
    "Procrastination": (20, 60),
    "Sleep Problems": (20, 90),
    "Addictive Behaviors (e.g., Drug Use, Gambling)": (20, 75),
    "Alcohol Abuse": (20, 75),
    "Compulsive Behaviors": (20, 75),
    "Eating Disorders": (20, 55),
    "Internet Addiction": (20, 55),
    "Self-Harm Behaviors": (20, 55),
    "Debt Problems": (20, 80),
}

MIN_CATEGORIES_REQUIRED = 2


def load_persona_ids(persona_dir: Path) -> List[str]:
    ids = [path.stem for path in sorted(persona_dir.glob("seeker_auto_*.json"))]
    if not ids:
        raise FileNotFoundError(f"No persona JSON files found in {persona_dir}")
    return ids


def category_age_pool(
    categories: Sequence[str],
    ages: Sequence[int],
) -> Dict[str, List[int]]:
    pool: Dict[str, List[int]] = {}
    for category in categories:
        min_age, max_age = CATEGORY_RULES.get(category, (ages[0], ages[-1]))
        allowed = [age for age in ages if min_age <= age <= max_age]
        if not allowed:
            raise ValueError(
                f"Category '{category}' has no allowable ages ({min_age}-{max_age})."
            )
        pool[category] = allowed
    return pool


def compute_target_counts(
    categories: Sequence[str],
    ages: Sequence[int],
    pool: Dict[str, List[int]],
    rng: random.Random,
) -> Dict[str, int]:
    remaining_slots = len(ages)
    counts = {category: 1 for category in categories}
    remaining_slots -= len(categories)
    if remaining_slots < 0:
        raise ValueError("More categories than available ages.")

    while remaining_slots > 0:
        candidates = [
            category
            for category in categories
            if counts[category] < len(pool[category])
        ]
        if not candidates:
            break
        candidates.sort(key=lambda cat: counts[cat] / len(pool[cat]))
        lowest_ratio = counts[candidates[0]] / len(pool[candidates[0]])
        top = [
            cat for cat in candidates if counts[cat] / len(pool[cat]) == lowest_ratio
        ]
        chosen = rng.choice(top)
        counts[chosen] += 1
        remaining_slots -= 1
    return counts


def assign_categories(
    persona_ids: Sequence[str],
    categories: Sequence[str],
    ages: Sequence[int],
    seed: int | None,
) -> Dict[str, Dict[str, str]]:
    plan: Dict[str, Dict[str, str]] = {}
    rng = random.Random(seed)
    pool = category_age_pool(categories, ages)
    allowed_by_age: Dict[int, List[str]] = {
        age: [category for category in categories if age in pool[category]]
        for age in ages
    }

    total_slots = len(persona_ids) * len(ages)
    base = total_slots // len(categories)
    remainder = total_slots % len(categories)
    global_remaining: Dict[str, int] = {}
    for idx, category in enumerate(categories):
        count = base + (1 if idx < remainder else 0)
        capacity = len(pool[category]) * len(persona_ids)
        if count > capacity:
            count = capacity
        global_remaining[category] = count

    age_order_template = sorted(ages, key=lambda age: len(allowed_by_age[age]))

    for persona_id in persona_ids:
        assignments: Dict[str, str] = {}
        for age in age_order_template:
            options = [
                category
                for category in allowed_by_age[age]
                if global_remaining.get(category, 0) > 0
            ]
            if not options:
                options = allowed_by_age[age][:]
            if not options:
                raise ValueError(f"No categories available for age {age}")
            options.sort(
                key=lambda cat: (
                    global_remaining.get(cat, 0),
                    len(pool[cat]),
                    rng.random(),
                ),
                reverse=True,
            )
            chosen = options[0]
            assignments[str(age)] = chosen
            global_remaining[chosen] = global_remaining.get(chosen, 0) - 1
        plan[persona_id] = assignments

    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a stage-aware crisis category plan for personas."
    )
    parser.add_argument(
        "--persona-dir",
        type=Path,
        default=Path("artifacts/personas"),
        help="Directory containing persona JSON files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/crisis_category_plan.json"),
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
        help="Category labels to use (default: detailed 45-category set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic planning.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.age_start >= args.age_end:
        parser.error("age-start must be less than age-end")
        return 2
    if len(args.categories) < MIN_CATEGORIES_REQUIRED:
        parser.error(f"Provide at least {MIN_CATEGORIES_REQUIRED} categories.")
        return 2

    persona_ids = load_persona_ids(args.persona_dir)
    ages = list(range(args.age_start, args.age_end + 1))
    plan = assign_categories(persona_ids, args.categories, ages, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "meta": {
                    "persona_count": len(persona_ids),
                    "age_range": [args.age_start, args.age_end],
                    "categories": args.categories,
                    "seed": args.seed,
                },
                "plan": plan,
            },
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")
    print(
        f"Wrote plan for {len(persona_ids)} personas covering ages "
        f"{args.age_start}-{args.age_end} to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
