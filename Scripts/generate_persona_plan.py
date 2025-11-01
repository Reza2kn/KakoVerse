#!/usr/bin/env python3
"""Create a balanced persona itinerary plan covering every city/decade evenly."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class CityRecord:
    city: str
    country: str
    decade: str
    metadata: str

    @property
    def key(self) -> str:
        return f"{self.city}|{self.country}|{self.decade}"

    @property
    def decade_value(self) -> int:
        return int(self.decade[:4])


def load_city_records(csv_path: Path) -> List[CityRecord]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        grouped: Dict[Tuple[str, str], List[CityRecord]] = {}
        for row in reader:
            record = CityRecord(
                city=row["city"].strip(),
                country=row["country"].strip(),
                decade=row["decade"].strip(),
                metadata=row["metadata"].strip(),
            )
            grouped.setdefault((record.city, record.country), []).append(record)

    if len(grouped) != 69:
        raise ValueError(
            f"Expected 69 unique cities in {csv_path}, found {len(grouped)}."
        )

    selected: List[CityRecord] = []
    for records in grouped.values():
        records.sort(key=lambda rec: rec.decade_value)
        if not records:
            continue
        indices = [0, len(records) // 2, len(records) - 1]
        seen = set()
        for idx in indices:
            rec = records[idx]
            if rec.key in seen:
                continue
            selected.append(rec)
            seen.add(rec.key)
        while len(seen) < 3 and records:
            rec = records[len(seen) % len(records)]
            if rec.key not in seen:
                selected.append(rec)
                seen.add(rec.key)

    if len(selected) != 69 * 3:
        raise ValueError(
            f"Expected 207 selected city records, found {len(selected)}."
        )
    return selected


def persona_id_list(count: int) -> List[str]:
    return [f"seeker_auto_{i:03d}" for i in range(1, count + 1)]


class PlanBuilder:
    def __init__(
        self,
        *,
        records: Sequence[CityRecord],
        usage_target: int,
        lengths: Sequence[int],
        persona_ids: Sequence[str],
        seed: int,
    ) -> None:
        if usage_target * len(records) != sum(lengths):
            raise ValueError("City usage target does not match persona lengths total.")
        if len(lengths) != len(persona_ids):
            raise ValueError("Lengths list must match number of personas.")
        self.records = list(records)
        self.record_by_key: Dict[str, CityRecord] = {rec.key: rec for rec in records}
        self.remaining = {rec.key: usage_target for rec in records}
        self.lengths = list(lengths)
        self.persona_ids = list(persona_ids)
        self.plan: Dict[str, List[str]] = {}
        self.transitions: Dict[Tuple[str, str], int] = {}
        self.rng = random.Random(seed)

    def build(self) -> Dict[str, List[str]]:
        if not self._assign(0):
            raise RuntimeError("Failed to construct a balanced plan.")
        return self.plan

    def _assign(self, idx: int) -> bool:
        if idx >= len(self.persona_ids):
            return True
        persona_id = self.persona_ids[idx]
        length = self.lengths[idx]
        sequence = self._build_sequence(length)
        if sequence is None:
            return False
        self.plan[persona_id] = [rec.key for rec in sequence]
        if self._assign(idx + 1):
            return True
        # backtrack
        self._rollback_sequence(sequence)
        del self.plan[persona_id]
        return False

    def _build_sequence(self, length: int) -> List[CityRecord] | None:
        sequence: List[CityRecord] = []
        used: set[str] = set()

        def dfs(position: int, prev: CityRecord | None, relaxed: bool) -> bool:
            if position == length:
                return True
            candidates = self._candidate_records(prev, used, relaxed)
            if not candidates and not relaxed:
                return dfs(position, prev, True)
            for record in candidates:
                self._use_record(record, prev)
                used.add(record.key)
                sequence.append(record)
                if dfs(position + 1, record, False):
                    return True
                sequence.pop()
                used.remove(record.key)
                self._unuse_record(record, prev)
            return False

        if dfs(0, None, False):
            return sequence
        # cleanup on failure (dfs already undone) simply return None
        return None

    def _candidate_records(
        self,
        prev: CityRecord | None,
        used: set[str],
        relaxed: bool,
    ) -> List[CityRecord]:
        candidates: List[CityRecord] = []
        for record in self.records:
            if self.remaining[record.key] <= 0:
                continue
            if record.key in used:
                continue
            if not relaxed and prev is not None and record.decade_value < prev.decade_value:
                continue
            candidates.append(record)
        candidates.sort(key=lambda rec: self._score(prev, rec))
        return candidates

    def _score(self, prev: CityRecord | None, record: CityRecord) -> float:
        # lower scores preferred
        city_need = self.remaining[record.key]
        transition_need = 0
        imbalance = 0
        if prev is not None:
            forward = (prev.key, record.key)
            reverse = (record.key, prev.key)
            transition_need = self.transitions.get(forward, 0)
            imbalance = abs(
                self.transitions.get(forward, 0) - self.transitions.get(reverse, 0)
            )
        jitter = self.rng.random() * 0.01
        return city_need * 5 + transition_need * 7 + imbalance * 9 + jitter

    def _use_record(self, record: CityRecord, prev: CityRecord | None) -> None:
        self.remaining[record.key] -= 1
        if self.remaining[record.key] < 0:
            raise RuntimeError("Remaining count became negative")
        if prev is not None:
            key = (prev.key, record.key)
            self.transitions[key] = self.transitions.get(key, 0) + 1

    def _unuse_record(self, record: CityRecord, prev: CityRecord | None) -> None:
        self.remaining[record.key] += 1
        if prev is not None:
            key = (prev.key, record.key)
            self.transitions[key] -= 1
            if self.transitions[key] == 0:
                del self.transitions[key]

    def _rollback_sequence(self, sequence: Sequence[CityRecord]) -> None:
        prev: CityRecord | None = None
        for record in sequence:
            self.remaining[record.key] += 1
            if prev is not None:
                key = (prev.key, record.key)
                self.transitions[key] -= 1
                if self.transitions[key] == 0:
                    del self.transitions[key]
            prev = record


def build_lengths(count: int) -> List[int]:
    if count % 3 != 0:
        raise ValueError("Persona count must be divisible by 3 for the default length mix.")
    third = count // 3
    lengths = [2] * third + [3] * third + [4] * third
    return lengths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a deterministic persona plan.")
    parser.add_argument(
        "--city-csv",
        type=Path,
        default=Path("Prompts/Kakoverse_Global_Cities_1920s_2020s.csv"),
        help="CSV file containing the 69 city/decade records.",
    )
    parser.add_argument(
        "--persona-count",
        type=int,
        default=69,
        help="Number of personas to plan for.",
    )
    parser.add_argument(
        "--usage",
        type=int,
        default=1,
        help="Number of times each city record should appear across personas.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for tie-breaking.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Artifacts/persona_plan.json"),
        help="Where to write the generated plan JSON.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = load_city_records(args.city_csv)
    persona_ids = persona_id_list(args.persona_count)
    lengths = build_lengths(args.persona_count)
    random.Random(args.seed).shuffle(lengths)
    builder = PlanBuilder(
        records=records,
        usage_target=args.usage,
        lengths=lengths,
        persona_ids=persona_ids,
        seed=args.seed,
    )
    plan = builder.build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "meta": {
                    "persona_count": args.persona_count,
                    "city_usage_target": args.usage,
                    "length_distribution": lengths,
                    "seed": args.seed,
                },
                "plan": plan,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote plan for {args.persona_count} personas to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
