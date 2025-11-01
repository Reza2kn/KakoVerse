#!/usr/bin/env python3
"""Generate age-indexed crisis profiles for each persona using OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Sequence

import requests


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "alibaba/tongyi-deepresearch-30b-a3b:free"
DEFAULT_AGE_RANGE = range(20, 101)
MAX_RETRIES = 5


def load_env_value(key: str, env_path: Path) -> str:
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            striped = line.strip()
            if not striped or striped.startswith("#") or "=" not in striped:
                continue
            lhs, rhs = striped.split("=", 1)
            if lhs.strip() == key:
                return rhs.strip().strip("\"'")
    raise KeyError(f"Missing {key} in environment or {env_path}")


def call_openrouter(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    user_agent: str,
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
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                OPENROUTER_ENDPOINT, headers=headers, json=payload, timeout=120
            )
        except requests.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"OpenRouter request failed: {exc}")
            time.sleep(backoff)
            backoff *= 1.5
            continue
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else backoff
            time.sleep(wait)
            backoff *= 1.5
            continue
        if response.status_code >= 500:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"OpenRouter server error {response.status_code}: {response.text}"
                )
            time.sleep(backoff)
            backoff *= 1.5
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
            raise RuntimeError("OpenRouter returned empty choices")
        return choices[0]["message"].get("content", "")
    raise RuntimeError("Exceeded OpenRouter retry attempts")


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


def build_prompt(persona_json: Dict[str, object], age_range: Sequence[int]) -> List[Dict[str, str]]:
    persona_str = json.dumps(persona_json, indent=2, ensure_ascii=False)
    instructions = (
        "You receive a detailed persona profile. For each age listed below, "
        "propose a plausible primary crisis or presenting problem they might "
        "face at that age, grounded in their background, values, and life events. "
        "Ensure crises vary across the lifespan (e.g., academic stress in youth, "
        "career burnout in midlife, loneliness or chronic pain later). Return JSON "
        "with keys as ages (numeric) and each value an object containing: \n"
        "- 'summary': short crisis description (<= 20 words).\n"
        "- 'category': concise tag (e.g., grief, chronic_pain, financial_stress).\n"
        "- 'confidence': 0-1 float for likelihood.\n"
        "No markdown."
    )
    ages_list = ", ".join(str(age) for age in age_range)
    user_prompt = (
        f"Persona Profile:\n{persona_str}\n\n"
        f"Target Ages: [{ages_list}]\n"
        "Return JSON mapping ages to crisis info as described."
    )
    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]


def generate_crisis_profile(
    persona_path: Path,
    api_key: str,
    model: str,
    age_range: Sequence[int],
    output_path: Path,
    sleep: float,
) -> None:
    with persona_path.open("r", encoding="utf-8") as handle:
        persona_json = json.load(handle)
    messages = build_prompt(persona_json, age_range)
    content = call_openrouter(
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=0.3,
        top_p=0.9,
        max_tokens=1800,
        user_agent="vf-CrisisSupport Crisis Profile Generator",
    )
    content = strip_code_fences(content)
    try:
        parsed = json.loads(content)
    except ValueError:
        import re

        match = re.search(r"\{.*\}$", content, re.DOTALL)
        if not match:
            raise RuntimeError(
                f"Failed to parse crisis profile for {persona_path.name}: {content}"
            )
        parsed = json.loads(match.group(0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(parsed, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    time.sleep(max(0.0, sleep))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate age-indexed crisis profiles for personas."
    )
    parser.add_argument(
        "--persona-dir",
        type=Path,
        default=Path("Artifacts/personas"),
        help="Directory containing persona JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts/crisis_profiles"),
        help="Where to write crisis profile JSON files.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env containing OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenRouter model identifier (default: alibaba/tongyi-deepresearch-30b-a3b:free).",
    )
    parser.add_argument(
        "--ages",
        type=int,
        nargs=2,
        default=[20, 100],
        metavar=("START", "END"),
        help="Age range inclusive (default: 20 100).",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Sleep seconds between API calls (default: 1.5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Reserved for future use (no effect currently).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip personas whose crisis profile JSON already exists.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.ages[0] >= args.ages[1]:
        parser.error("Age START must be less than END.")
    age_range = range(args.ages[0], args.ages[1] + 1)

    try:
        api_key = load_env_value("OPENROUTER_API_KEY", args.env_file)
    except KeyError as err:
        parser.error(str(err))
        return 2

    persona_paths = sorted(args.persona_dir.glob("*.json"))
    if not persona_paths:
        parser.error(f"No persona JSON files found in {args.persona_dir}")
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for persona_path in persona_paths:
        output_path = output_dir / persona_path.name
        if args.skip_existing and output_path.exists():
            print(f"[skip] {persona_path.name} already has a crisis profile.")
            continue
        print(f"Generating crisis profile for {persona_path.name}...")
        try:
            generate_crisis_profile(
                persona_path=persona_path,
                api_key=api_key,
                model=args.model,
                age_range=age_range,
                output_path=output_path,
                sleep=args.sleep,
            )
        except RuntimeError as err:
            print(f"[error] {persona_path.name}: {err}")
            continue
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
