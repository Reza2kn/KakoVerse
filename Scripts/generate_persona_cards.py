#!/usr/bin/env python3
"""
Persona card generator that orchestrates one OpenRouter chat completion per persona.

The script pulls a system prompt and reference persona card, samples timeline
contexts from Kakoverse metadata, and asks the alibaba/tongyi-deepresearch-30b-a3b:free
model (via OpenRouter) to produce personas consistent with the sampled environments.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import requests


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "alibaba/tongyi-deepresearch-30b-a3b:free"
DEFAULT_COUNT = 68
DEFAULT_SLEEP = 5.0
DEFAULT_JITTER = 1.5
MAX_MODEL_ATTEMPTS = 3


class PersonaGenerationError(RuntimeError):
    """Raised when the LLM output cannot be parsed or validated."""


@dataclass(frozen=True)
class CityMetadata:
    city: str
    country: str
    decade: str
    metadata: str

    @property
    def decade_value(self) -> int:
        try:
            return int(self.decade[:4])
        except ValueError as exc:
            raise ValueError(f"Unexpected decade value: {self.decade}") from exc

    @property
    def city_key(self) -> str:
        return f"{self.city}|{self.country}|{self.decade}"

    def as_prompt_block(self) -> str:
        return (
            f"- {self.city}, {self.country} ({self.decade}): "
            f"{self.metadata}"
        )


def decade_sort_key(decade: str) -> int:
    try:
        return int(decade[:4])
    except ValueError as exc:
        raise ValueError(f"Cannot interpret decade string: {decade}") from exc


def load_env_value(key: str, env_path: Path | None) -> str:
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    if env_path is None or not env_path.exists():
        raise FileNotFoundError(
            f"Environment variable {key} is not set and no .env file was found."
        )
    value = None
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            striped = line.strip()
            if not striped or striped.startswith("#"):
                continue
            if "=" not in striped:
                continue
            env_key, env_value = striped.split("=", 1)
            if env_key.strip() == key:
                value = env_value.strip().strip("\"'")
    if not value:
        raise KeyError(
            f"Unable to locate {key} in environment or {env_path}"
        )
    return value


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def load_city_metadata(path: Path) -> Tuple[List[str], Dict[str, List[CityMetadata]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        mapping: Dict[str, List[CityMetadata]] = {}
        for row in reader:
            entry = CityMetadata(
                city=row["city"].strip(),
                country=row["country"].strip(),
                decade=row["decade"].strip(),
                metadata=row["metadata"].strip(),
            )
            mapping.setdefault(entry.decade, []).append(entry)
    if not mapping:
        raise ValueError(f"No city metadata found in {path}")
    sorted_decades = sorted(mapping.keys(), key=decade_sort_key)
    return sorted_decades, mapping


def sequence_to_timeline(sequence: Sequence[CityMetadata]) -> Dict[str, Sequence[CityMetadata]]:
    if not sequence:
        raise ValueError("Timeline sequence may not be empty")
    birth = sequence[0]
    current = sequence[-1]
    formative = tuple(sequence[1:-1])
    return {
        "birth": (birth,),
        "formative": formative,
        "current": (current,),
    }


def load_plan(plan_path: Path) -> Dict[str, List[str]]:
    if not plan_path.exists():
        raise FileNotFoundError(
            f"Persona plan not found at {plan_path}. Run Scripts/generate_persona_plan.py first."
        )
    with plan_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    plan = data.get("plan")
    if not isinstance(plan, dict):
        raise ValueError("Invalid plan format: missing 'plan' mapping")
    normalized = {str(k): [str(item) for item in v] for k, v in plan.items()}
    return normalized


def build_system_prompt(
    seeker_system_prompt: str,
    seeker_persona_card: str,
) -> str:
    header = (
        "You are a data generator producing Seeker persona cards for crisis support "
        "training. Follow the schema, tone, and field semantics shown below. Always "
        "return strictly valid JSON with double-quoted keys/strings and numeric values "
        "where expected. Do not include Markdown fences."
    )
    return (
        f"{header}\n\n"
        "=== SEEKER SYSTEM PROMPT JSON ===\n"
        f"{seeker_system_prompt.strip()}\n"
        "=== SEEKER PERSONA CARD EXAMPLE ===\n"
        f"{seeker_persona_card.strip()}\n"
        "=== END OF REFERENCES ==="
    )


def timeline_prompt_block(
    timeline: Dict[str, Sequence[CityMetadata]],
) -> str:
    birth = timeline["birth"][0]
    formative = timeline["formative"]
    current = timeline["current"][0]
    block_lines = [
        "Birth context:",
        birth.as_prompt_block(),
        "",
        "Formative contexts (in order):",
    ]
    block_lines.extend(item.as_prompt_block() for item in formative)
    block_lines.extend(
        [
            "",
            "Current socio-demographic context:",
            current.as_prompt_block(),
        ]
    )
    return "\n".join(block_lines)


def build_user_message(
    persona_id: str,
    timeline: Dict[str, Sequence[CityMetadata]],
    included_cities: Sequence[str],
    target_year: int,
    estimated_age: int,
) -> str:
    birth = timeline["birth"][0]
    formative = timeline["formative"]
    current = timeline["current"][0]
    formative_summary = [
        {
            "decade": item.decade,
            "city": item.city,
            "country": item.country,
            "metadata": item.metadata,
        }
        for item in formative
    ]
    prompt_block = timeline_prompt_block(timeline)
    formative_json = json.dumps(formative_summary, ensure_ascii=False, indent=2)
    included_cities_json = json.dumps(list(included_cities), ensure_ascii=False)
    instructions = [
        f"Generate one unique Seeker persona card with id '{persona_id}'.",
        "Use the schema and field semantics from the reference JSON exactly.",
        "Return valid JSON only (no Markdown fences, prefixes, or natural language).",
        "All `metadata`, `decade`, `city`, and `country` fields for birth, formative, "
        "and current contexts must match the provided records verbatim and remain in "
        "the same chronological order.",
        "Ensure the persona's history, skills, interests, and presenting problem are "
        "plausible given the sampled contexts.",
        "Environmental influences should clearly shape natural_tendencies and life "
        "details (e.g., formative years in Iran can yield Persian cooking expertise).",
        "Align timelines: birth decade precedes formative decades, which precede the "
        "current context. Keep the narrative coherent with that progression.",
        "Set `current_socio_demographics.current_location` to the current context "
        "city and country (e.g., 'City, Country').",
        f"Set `current_socio_demographics.year` to {target_year} and choose an age "
        f"consistent with that year and the birth decade (around {estimated_age}).",
        "Populate `meta.created_from` with 'prompts/Kakoverse_Global_Cities_1920s_2020s.csv', "
        f"set `meta.included_cities` exactly to {included_cities_json}, and provide "
        "a semantic `meta.version` string (e.g., '0.2').",
        "Keep numeric scales (Schwartz values, HEXACO, CSI) within 0-100 and use "
        "nuanced combinations, not all extremes.",
        "Ensure arrays like `natural_tendencies`, `symptoms`, `triggers` have varied, "
        "contextually grounded entries.",
    ]
    instructions_text = "\n- ".join([""] + instructions)
    return (
        f"{instructions_text}\n\n"
        "Timeline data to honor:\n"
        f"{prompt_block}\n\n"
        "Formative contexts as JSON for reference (do not echo unless matching schema):\n"
        f"{formative_json}\n"
        "Birth context metadata: "
        f"{birth.metadata}\n"
        "Current context metadata: "
        f"{current.metadata}\n"
        "Remember: respond with JSON only."
    )


def estimate_year_and_age(
    timeline: Dict[str, Sequence[CityMetadata]],
    rng: random.Random,
) -> Tuple[int, int]:
    birth_decade = timeline["birth"][0].decade_value
    current_decade = timeline["current"][0].decade_value
    birth_year = birth_decade + rng.randint(0, 9)
    current_year = current_decade + rng.randint(0, 9)
    if current_year <= birth_year + 17:
        current_year = birth_year + rng.randint(18, 30)
    age = max(18, min(85, current_year - birth_year))
    return current_year, age


def openrouter_generate_text(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    user_agent: str = "vf-CrisisSupport Persona Generator",
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Title": user_agent,
    }
    payload = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_tokens,
    }
    backoff = 2.0
    for attempt in range(5):
        try:
            response = requests.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120,
            )
        except requests.RequestException as exc:
            if attempt == 4:
                raise RuntimeError(f"OpenRouter request failed: {exc}") from exc
            time.sleep(backoff)
            backoff *= 1.8
            continue

        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait_time = float(retry_after) if retry_after else backoff
            time.sleep(wait_time)
            backoff *= 1.8
            continue
        if response.status_code >= 500:
            if attempt == 4:
                raise RuntimeError(
                    f"OpenRouter server error {response.status_code}: {response.text}"
                )
            time.sleep(backoff)
            backoff *= 1.8
            continue
        if response.status_code != 200:
            raise RuntimeError(
                f"OpenRouter error {response.status_code}: {response.text}"
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise RuntimeError("OpenRouter returned non-JSON response") from exc

        choices = data.get("choices")
        if not choices:
            raise RuntimeError("OpenRouter returned no choices.")
        content = choices[0]["message"].get("content", "")
        cleaned = strip_code_fences(content).strip()
        if not cleaned:
            raise RuntimeError("OpenRouter returned empty content.")
        return cleaned

    raise RuntimeError("Exceeded retry attempts contacting OpenRouter.")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


def parse_persona(content: str) -> Dict[str, object]:
    payload = strip_code_fences(content)
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise PersonaGenerationError(f"Failed to parse persona JSON: {exc}") from exc


def validate_persona(
    persona: Dict[str, object],
    persona_id: str,
    timeline: Dict[str, Sequence[CityMetadata]],
    included_cities: Sequence[str],
) -> None:
    missing_keys = [
        key
        for key in (
            "agent",
            "id",
            "birth_context",
            "formative_context",
            "current_socio_demographics",
            "meta",
        )
        if key not in persona
    ]
    if missing_keys:
        raise PersonaGenerationError(f"Missing keys: {', '.join(missing_keys)}")
    if persona["agent"] != "Seeker":
        raise PersonaGenerationError("agent must be 'Seeker'")
    if persona["id"] != persona_id:
        raise PersonaGenerationError(f"id must be '{persona_id}'")

    birth_expected = timeline["birth"][0]
    birth_ctx = persona["birth_context"]
    if not isinstance(birth_ctx, dict):
        raise PersonaGenerationError("birth_context must be an object")
    for field, expected in (
        ("decade", birth_expected.decade),
        ("city", birth_expected.city),
        ("country", birth_expected.country),
        ("metadata", birth_expected.metadata),
    ):
        if birth_ctx.get(field) != expected:
            raise PersonaGenerationError(
                f"birth_context.{field} mismatch (expected {expected!r})"
            )

    formative_expected = timeline["formative"]
    formative_ctx = persona["formative_context"]
    if not isinstance(formative_ctx, list):
        raise PersonaGenerationError("formative_context must be an array")
    if len(formative_ctx) != len(formative_expected):
        raise PersonaGenerationError(
            f"formative_context length mismatch (expected {len(formative_expected)})"
        )
    for idx, (actual, expected) in enumerate(
        zip(formative_ctx, formative_expected)
    ):
        if not isinstance(actual, dict):
            raise PersonaGenerationError(
                f"formative_context[{idx}] must be an object"
            )
        for field, expected_value in (
            ("decade", expected.decade),
            ("city", expected.city),
            ("country", expected.country),
            ("metadata", expected.metadata),
        ):
            if actual.get(field) != expected_value:
                raise PersonaGenerationError(
                    f"formative_context[{idx}].{field} mismatch "
                    f"(expected {expected_value!r})"
                )

    current_expected = timeline["current"][0]
    current_ctx = persona["current_socio_demographics"]
    if not isinstance(current_ctx, dict):
        raise PersonaGenerationError(
            "current_socio_demographics must be an object"
        )
    location = current_ctx.get("current_location")
    if not isinstance(location, str):
        raise PersonaGenerationError(
            "current_socio_demographics.current_location must be a string"
        )
    location_lower = location.lower()
    if current_expected.city.lower() not in location_lower or current_expected.country.lower() not in location_lower:
        raise PersonaGenerationError(
            "current location must mention the provided city and country"
        )

    meta = persona.get("meta")
    if not isinstance(meta, dict):
        raise PersonaGenerationError("meta must be an object")
    if meta.get("created_from") != "prompts/Kakoverse_Global_Cities_1920s_2020s.csv":
        raise PersonaGenerationError(
            "meta.created_from must be 'prompts/Kakoverse_Global_Cities_1920s_2020s.csv'"
        )
    if meta.get("included_cities") != list(included_cities):
        raise PersonaGenerationError(
            "meta.included_cities must match the provided list"
        )
    if "version" not in meta:
        raise PersonaGenerationError("meta.version is required")


def postprocess_persona(
    persona: Dict[str, object],
    timeline: Dict[str, Sequence[CityMetadata]],
    included_cities: Sequence[str],
    target_year: int,
    estimated_age: int,
) -> Dict[str, object]:
    birth_record = timeline["birth"][0]
    birth_ctx = dict(persona.get("birth_context") or {})
    birth_ctx.update(
        {
            "decade": birth_record.decade,
            "city": birth_record.city,
            "country": birth_record.country,
            "metadata": birth_record.metadata,
        }
    )
    persona["birth_context"] = birth_ctx

    formative_records = timeline["formative"]
    existing_formative = persona.get("formative_context") or []
    formative_entries: List[Dict[str, object]] = []
    for idx, record in enumerate(formative_records):
        existing_entry = existing_formative[idx] if idx < len(existing_formative) else {}
        entry = dict(existing_entry)
        entry.update(
            {
                "decade": record.decade,
                "city": record.city,
                "country": record.country,
                "metadata": record.metadata,
            }
        )
        entry.setdefault("age_group_at_time", "teen")
        entry.setdefault("citizenship_status", "citizen")
        formative_entries.append(entry)
    persona["formative_context"] = formative_entries

    current_record = timeline["current"][0]
    current_ctx = dict(persona.get("current_socio_demographics") or {})
    current_ctx.update(
        {
            "current_location": f"{current_record.city}, {current_record.country}",
            "location_current": f"{current_record.city}, {current_record.country}",
            "year": target_year,
        }
    )
    current_ctx.setdefault("age", estimated_age)
    persona["current_socio_demographics"] = current_ctx

    meta = dict(persona.get("meta") or {})
    meta["created_from"] = "prompts/Kakoverse_Global_Cities_1920s_2020s.csv"
    meta["included_cities"] = list(included_cities)
    meta.setdefault("version", "balanced-0.2")
    persona["meta"] = meta
    return persona


def save_persona(persona: Dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(persona, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def generate_persona_card(
    *,
    api_key: str,
    model_name: str,
    system_prompt: str,
    persona_id: str,
    request_index: int,
    timeline: Dict[str, Sequence[CityMetadata]],
    rng: random.Random,
) -> Dict[str, object]:
    included_cities = [
        timeline["birth"][0].city_key,
        *[item.city_key for item in timeline["formative"]],
        timeline["current"][0].city_key,
    ]
    target_year, estimated_age = estimate_year_and_age(timeline, rng)
    base_message = build_user_message(
        persona_id,
        timeline,
        included_cities,
        target_year,
        estimated_age,
    )
    sleep_between_attempts = 1.5

    for attempt in range(1, MAX_MODEL_ATTEMPTS + 1):
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": base_message},
        ]
        if attempt > 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Previous output failed schema validation. Re-read the "
                        "instructions carefully and regenerate JSON that fully "
                        "matches the schema requirements."
                    ),
                }
            )
        try:
            content = openrouter_generate_text(
                api_key=api_key,
                model=model_name,
                messages=messages,
                temperature=0.95,
                top_p=0.95,
                max_tokens=1800,
            )
        except RuntimeError as err:
            if attempt == MAX_MODEL_ATTEMPTS:
                raise PersonaGenerationError(
                    f"OpenRouter call failed after {attempt} attempts: {err}"
                ) from err
            time.sleep(sleep_between_attempts)
            sleep_between_attempts *= 1.5
            continue
        persona = parse_persona(content)
        persona = postprocess_persona(
            persona,
            timeline,
            included_cities,
            target_year,
            estimated_age,
        )
        try:
            validate_persona(persona, persona_id, timeline, included_cities)
            return persona
        except PersonaGenerationError as err:
            if attempt == MAX_MODEL_ATTEMPTS:
                raise PersonaGenerationError(
                    f"Validation failed after {attempt} attempts: {err}"
                ) from err
            time.sleep(sleep_between_attempts)
            sleep_between_attempts *= 1.5
    raise PersonaGenerationError("Unreachable state in persona generation loop.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Seeker persona cards via OpenRouter."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of persona cards to generate (default: {DEFAULT_COUNT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/personas"),
        help="Directory for generated persona JSON files.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file containing OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenRouter model identifier (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("artifacts/persona_plan.json"),
        help="Path to the precomputed persona plan JSON.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=f"Base sleep (seconds) between successful calls (default: {DEFAULT_SLEEP}).",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=DEFAULT_JITTER,
        help=f"Random jitter added to sleep (default: {DEFAULT_JITTER}).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="(Deprecated) Existing personas are automatically detected and skipped.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    rng = random.Random(args.seed)

    project_root = Path(__file__).resolve().parent.parent
    prompts_dir = project_root / "prompts"
    seeker_system_prompt_path = prompts_dir / "Seeker_System_Prompt.json"
    seeker_persona_card_path = prompts_dir / "Seeker_Persona_Card.json"
    cities_csv_path = prompts_dir / "Kakoverse_Global_Cities_1920s_2020s.csv"

    try:
        api_key = load_env_value("OPENROUTER_API_KEY", args.env_file)
    except (KeyError, FileNotFoundError) as err:
        parser.error(str(err))
        return 2

    seeker_system_prompt = load_text(seeker_system_prompt_path)
    seeker_persona_card = load_text(seeker_persona_card_path)
    sorted_decades, cities_by_decade = load_city_metadata(cities_csv_path)
    city_lookup: Dict[str, CityMetadata] = {}
    for records in cities_by_decade.values():
        for record in records:
            city_lookup[record.city_key] = record
    system_prompt = build_system_prompt(seeker_system_prompt, seeker_persona_card)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    plan = load_plan(args.plan)

    pattern = re.compile(r"seeker_auto_(\d{3})\.json$")
    existing_indices = {
        int(match.group(1))
        for path in output_dir.glob("seeker_auto_*.json")
        if (match := pattern.match(path.name))
    }
    target_total = max(0, args.count)
    needed_indices = [
        idx for idx in range(1, target_total + 1) if idx not in existing_indices
    ]

    if not needed_indices:
        print(
            f"All {target_total} persona cards already exist in {output_dir}. "
            "Nothing to do."
        )
        return 0

    success_count = 0
    total_to_generate = len(needed_indices)
    for position, index in enumerate(needed_indices, start=1):
        persona_id = f"seeker_auto_{index:03d}"
        output_path = output_dir / f"{persona_id}.json"
        print(
            f"[{position:02d}/{total_to_generate}] Generating persona {persona_id} "
            f"(target index {index:03d})..."
        )
        sequence_keys = plan.get(persona_id)
        if not sequence_keys:
            print(f"[skip] {persona_id} missing from plan; skipping.")
            continue
        try:
            sequence_records = [city_lookup[key] for key in sequence_keys]
        except KeyError as err:
            raise ValueError(f"Plan references unknown city key: {err}") from err
        timeline = sequence_to_timeline(sequence_records)
        try:
            persona = generate_persona_card(
                api_key=api_key,
                model_name=args.model,
                system_prompt=system_prompt,
                persona_id=persona_id,
                request_index=index,
                timeline=timeline,
                rng=rng,
            )
        except PersonaGenerationError as err:
            print(f"[error] Persona {persona_id} failed validation: {err}")
            continue
        except RuntimeError as err:
            print(f"[error] Persona {persona_id} failed due to runtime issue: {err}")
            continue
        save_persona(persona, output_path)
        success_count += 1
        sleep_time = max(0.0, args.sleep + rng.uniform(0, args.jitter))
        time.sleep(sleep_time)

    print(
        f"Completed {success_count} new persona cards "
        f"(target total {target_total}, existing {len(existing_indices)})."
    )
    if success_count < total_to_generate:
        print("Some personas failed to generate. Review errors above.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
