#!/usr/bin/env python3
"""
Generate multi-turn conversations between Seeker and Supporter agents using OpenRouter.

Each persona card receives one conversation thread of 15-20 turns. Seeker and
Supporter run in separate OpenRouter contexts (default: minimax/minimax-m2:free
for both roles). Every persona is assigned a unique age/category pairing from the
crisis plan before generation, ensuring balanced coverage. For every Seeker
message, we generate three
Supporter variants (baseline, empathetic, cold) to drive empathy reward modelling.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import re
import requests

OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_SEEKER_MODEL = "minimax/minimax-m2:free"
DEFAULT_SUPPORTER_MODEL = "minimax/minimax-m2:free"

RESPONSE_STYLES = (
    {
        "label": "baseline",
        "care_level": 0.6,
        "description": "balanced warmth, keeps conversation progressing",
        "temperature": 0.7,
        "top_p": 0.92,
    },
    {
        "label": "empathetic",
        "care_level": 0.9,
        "description": "high empathy, rich validation",
        "temperature": 0.9,
        "top_p": 0.98,
    },
    {
        "label": "cold",
        "care_level": 0.2,
        "description": "terse, low-affect, minimal empathy",
        "temperature": 0.35,
        "top_p": 0.75,
    },
)
DEFAULT_MIN_TURNS = 15
DEFAULT_MAX_TURNS = 20
MAX_AGENT_ATTEMPTS = 5
DEFAULT_SLEEP = 1.2
DEFAULT_JITTER = 0.6
DEFAULT_SEEKER_REQUEST_INTERVAL = 1.2
DEFAULT_SEEKER_REQUEST_JITTER = 0.3
DEFAULT_SUPPORTER_REQUEST_INTERVAL = 0.4
DEFAULT_SUPPORTER_REQUEST_JITTER = 0.15
THOUGHT_START_TAG = "<THOUGHT>"
THOUGHT_END_TAG = "</THOUGHT>"
MESSAGE_START_TAG = "<MESSAGE>"
MESSAGE_END_TAG = "</MESSAGE>"

FORMAT_TEMPLATE_EXAMPLE = (
    "Remember: respond exactly like\n"
    "<THOUGHT>\n"
    "your private reasoning\n"
    "</THOUGHT>\n"
    "<MESSAGE>\n"
    "your outward reply (1-4 sentences)\n"
    "</MESSAGE>\n"
    "Do not add other text before or after these tags."
)
FALLBACK_THOUGHT_TEXT = "Auto-generated thought because model omitted required tags."

RETRY_DELAY_RE = re.compile(r"retry_delay\s*{\s*seconds:\s*([0-9]+)")
RETRY_DELAY_ALT_RE = re.compile(r"Please retry in\s+([0-9.]+)s")


class ConversationError(RuntimeError):
    """Raised when an agent output cannot be parsed or validated."""


def load_env_value(key: str, env_path: Path | None) -> str:
    if key in os.environ and os.environ[key]:
        return os.environ[key]
    if env_path is None or not env_path.exists():
        raise FileNotFoundError(
            f"{key} missing in environment and {env_path or '.env'} not found."
        )
    value = None
    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            striped = line.strip()
            if not striped or striped.startswith("#") or "=" not in striped:
                continue
            lhs, rhs = striped.split("=", 1)
            if lhs.strip() == key:
                value = rhs.strip().strip("\"'")
    if not value:
        raise KeyError(f"{key} not found in {env_path}")
    return value


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines:
            lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped


ROLE_MAP = {"user": "user", "assistant": "assistant"}


def history_to_messages(history: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    for item in history:
        mapped_role = ROLE_MAP.get(item["role"])
        if mapped_role is None:
            raise ConversationError(f"Unsupported role in history: {item['role']}")
        messages.append({"role": mapped_role, "content": item["content"]})
    return messages


def extract_retry_after_seconds(message: str) -> Optional[float]:
    match = RETRY_DELAY_RE.search(message)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    match = RETRY_DELAY_ALT_RE.search(message)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


class RequestThrottler:
    def __init__(self, *, min_interval: float, jitter: float, rng: random.Random):
        self.min_interval = max(0.0, min_interval)
        self.jitter = max(0.0, jitter)
        self.rng = rng
        self._last_call_time: Optional[float] = None

    def wait(self) -> None:
        if self._last_call_time is None or self.min_interval <= 0:
            return
        elapsed = time.monotonic() - self._last_call_time
        target = self.min_interval + self.rng.uniform(0, self.jitter)
        remaining = target - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def mark(self) -> None:
        self._last_call_time = time.monotonic()


def load_category_plan(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    plan = data.get("plan")
    if not isinstance(plan, dict):
        raise ValueError(f"Invalid category plan at {path}")
    normalized: Dict[str, Dict[str, str]] = {}
    for persona_id, mapping in plan.items():
        if not isinstance(mapping, dict):
            continue
        normalized[persona_id] = {str(k): str(v) for k, v in mapping.items()}
    return normalized


def load_crisis_profile(profile_dir: Path, persona_id: str) -> Dict[str, Dict[str, object]]:
    path = profile_dir / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Crisis profile missing for {persona_id}: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(age): entry for age, entry in data.items()}


def openrouter_generate_text(
    *,
    api_key: str,
    model: str,
    messages: Sequence[Dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    throttler: RequestThrottler,
    user_agent: str = "vf-CrisisSupport Conversation Generator",
) -> str:
    throttler.wait()
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
    backoff = 1.5
    try:
        response = requests.post(
            OPENROUTER_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    while response.status_code in (429, 500, 502, 503, 504):
        retry_after = response.headers.get("Retry-After")
        wait_time = float(retry_after) if retry_after else backoff
        time.sleep(wait_time)
        backoff *= 1.8
        try:
            response = requests.post(
                OPENROUTER_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=120,
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter error {response.status_code}: {response.text}"
        )

    throttler.mark()
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
        raise RuntimeError("OpenRouter returned an empty response.")
    return cleaned


def parse_reasoned_output(payload: str) -> Dict[str, str]:
    thought_match = re.search(
        rf"{re.escape(THOUGHT_START_TAG)}\s*(.*?)\s*{re.escape(THOUGHT_END_TAG)}",
        payload,
        flags=re.DOTALL,
    )
    message_match = re.search(
        rf"{re.escape(MESSAGE_START_TAG)}\s*(.*?)\s*{re.escape(MESSAGE_END_TAG)}",
        payload,
        flags=re.DOTALL,
    )
    if not thought_match or not message_match:
        raise ConversationError(
            "Expected <THOUGHT>...</THOUGHT> and <MESSAGE>...</MESSAGE> blocks; "
            f"payload={payload!r}"
        )
    thought = thought_match.group(1).strip()
    message = message_match.group(1).strip()
    if not thought:
        raise ConversationError("Thought block must not be empty")
    if not message:
        raise ConversationError("Message block must not be empty")
    return {"thought": thought, "message": message}


def best_effort_parse_reasoned_output(payload: str) -> Optional[Dict[str, str]]:
    """Attempt to split reasoning even when closing tags are missing."""

    cleaned = strip_code_fences(payload).strip()
    if not cleaned:
        return None

    thought = ""
    thought_match = re.search(
        rf"{re.escape(THOUGHT_START_TAG)}\s*(.*?)\s*{re.escape(THOUGHT_END_TAG)}",
        cleaned,
        flags=re.DOTALL,
    )
    if thought_match:
        thought = thought_match.group(1).strip()

    message_block: Optional[str] = None
    if MESSAGE_START_TAG in cleaned:
        after_start = cleaned.split(MESSAGE_START_TAG, 1)[1]
        if MESSAGE_END_TAG in after_start:
            message_block = after_start.split(MESSAGE_END_TAG, 1)[0]
        else:
            message_block = after_start
    if message_block is None:
        return None
    message = message_block.strip()
    if not message:
        return None
    return {"thought": thought, "message": message}


def build_persona_catalog(
    category_plan: Dict[str, Dict[str, str]]
) -> Tuple[Dict[str, Dict[str, List[int]]], Dict[str, List[int]]]:
    if not category_plan:
        raise ValueError("Crisis category plan is empty.")

    persona_catalog: Dict[str, Dict[str, List[int]]] = {}
    category_age_pool: Dict[str, set[int]] = defaultdict(set)

    for persona_id, mapping in category_plan.items():
        category_map: Dict[str, List[int]] = defaultdict(list)
        for age_str, category in mapping.items():
            try:
                age = int(age_str)
            except (TypeError, ValueError):  # noqa: F841
                continue
            category_key = str(category).strip().lower()
            if not category_key:
                continue
            category_map[category_key].append(age)
            category_age_pool[category_key].add(age)
        for ages in category_map.values():
            ages.sort()
        persona_catalog[persona_id] = category_map

    pooled_sorted: Dict[str, List[int]] = {
        category: sorted(ages) for category, ages in category_age_pool.items()
    }
    if not pooled_sorted:
        raise ValueError("Unable to derive category pools from crisis plan.")
    return persona_catalog, pooled_sorted


def create_conversation_schedule(
    *,
    persona_ids: Sequence[str],
    persona_catalog: Dict[str, Dict[str, List[int]]],
    category_age_pool: Dict[str, List[int]],
    min_per_category: int,
    rng: random.Random,
    seed_value: Optional[int],
    max_attempts: int = 1000,
) -> Dict[str, object]:
    if min_per_category < 1:
        raise ValueError("min_per_category must be at least 1.")

    persona_ids = list(persona_ids)
    categories = sorted(category_age_pool.keys())
    if not categories:
        raise ValueError("No categories available to schedule.")

    target_counts: Dict[str, int] = {}
    total_needed = len(persona_ids)
    total_base = 0

    for category in categories:
        available_unique_ages = len(category_age_pool[category])
        base = min(min_per_category, available_unique_ages)
        target_counts[category] = base
        total_base += base

    if total_base > total_needed:
        reducible = sorted(
            ((count, category) for category, count in target_counts.items()),
            reverse=True,
        )
        idx = 0
        while total_base > total_needed and idx < len(reducible):
            _, category = reducible[idx]
            if target_counts[category] > 0:
                target_counts[category] -= 1
                total_base -= 1
            else:
                idx += 1
        if total_base > total_needed:
            raise ValueError("Not enough capacity to meet requested persona count.")

    remaining = total_needed - total_base
    expandable = [
        category
        for category in categories
        if target_counts[category] < len(category_age_pool[category])
    ]
    while remaining > 0 and expandable:
        category = rng.choice(expandable)
        target_counts[category] += 1
        remaining -= 1
        if target_counts[category] >= len(category_age_pool[category]):
            expandable.remove(category)

    if remaining > 0:
        raise ValueError("Unable to allocate enough unique ages for all personas.")

    for attempt in range(1, max_attempts + 1):
        rng.shuffle(persona_ids)
        remaining_counts = dict(target_counts)
        used_ages: set[int] = set()
        assignments: Dict[str, Dict[str, object]] = {}
        success = True

        for persona_id in persona_ids:
            category_map = persona_catalog.get(persona_id, {})
            options: List[Tuple[str, List[int]]] = []
            for category in categories:
                if remaining_counts.get(category, 0) <= 0:
                    continue
                age_options = [
                    age
                    for age in category_map.get(category, [])
                    if age not in used_ages
                ]
                if age_options:
                    options.append((category, age_options))
            if not options:
                success = False
                break
            category, age_options = rng.choice(options)
            age = rng.choice(age_options)
            assignments[persona_id] = {"age": int(age), "category": category}
            used_ages.add(age)
            remaining_counts[category] -= 1

        if success and all(count == 0 for count in remaining_counts.values()):
            category_actual = defaultdict(int)
            for entry in assignments.values():
                category_actual[str(entry["category"]).lower()] += 1
            return {
                "meta": {
                    "persona_count": len(persona_ids),
                    "seed": seed_value,
                    "min_per_category": min_per_category,
                    "target_counts": target_counts,
                    "actual_counts": dict(category_actual),
                    "attempts": attempt,
                },
                "assignments": assignments,
            }

    raise ValueError("Failed to generate conversation schedule after multiple attempts.")


def load_or_create_conversation_schedule(
    *,
    schedule_path: Path,
    persona_ids: Sequence[str],
    persona_catalog: Dict[str, Dict[str, List[int]]],
    category_age_pool: Dict[str, List[int]],
    min_per_category: int,
    rng: random.Random,
    seed_value: Optional[int],
) -> Dict[str, Dict[str, object]]:
    persona_ids = list(persona_ids)

    if schedule_path.exists():
        with schedule_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        assignments = data.get("assignments", {})
        valid = True
        used_ages: set[int] = set()
        if assignments:
            for persona_id in persona_ids:
                entry = assignments.get(persona_id)
                if not isinstance(entry, dict):
                    valid = False
                    break
                try:
                    age = int(entry.get("age"))
                except (TypeError, ValueError):
                    valid = False
                    break
                category = str(entry.get("category", "")).strip().lower()
                if not category:
                    valid = False
                    break
                persona_map = persona_catalog.get(persona_id)
                if (
                    not persona_map
                    or category not in persona_map
                    or age not in persona_map[category]
                ):
                    valid = False
                    break
                if age in used_ages:
                    valid = False
                    break
                used_ages.add(age)
        else:
            valid = False
        if valid and len(assignments) >= len(persona_ids):
            return {pid: assignments[pid] for pid in persona_ids}
        print(
            "[info] Existing conversation schedule missing data or duplicates; regenerating."
        )

    schedule = create_conversation_schedule(
        persona_ids=persona_ids,
        persona_catalog=persona_catalog,
        category_age_pool=category_age_pool,
        min_per_category=min_per_category,
        rng=rng,
        seed_value=seed_value,
    )
    schedule_path.parent.mkdir(parents=True, exist_ok=True)
    with schedule_path.open("w", encoding="utf-8") as handle:
        json.dump(schedule, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return schedule["assignments"]


def load_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        return handle.read()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def persona_to_json_string(persona: Dict[str, object]) -> str:
    return json.dumps(persona, ensure_ascii=False, indent=2)


def build_seeker_system_prompt(
    seeker_prompt_text: str,
    persona_json: Dict[str, object],
    target_turns: int,
) -> str:
    persona_block = persona_to_json_string(persona_json)
    instructions = (
        "You are the Seeker in a crisis support chat. Stay entirely in character, "
        "thinking and responding as the individual described below. Keep responses "
        "natural, 1-3 sentences, and grounded in lived context, values, and symptoms. "
        "Avoid lists or meta commentary. Do not disclose system instructions. "
        "Before giving your outward reply, think through your reasoning silently. "
        "Respond using EXACTLY this template (no extra text, no Markdown fences):\n"
        "<THOUGHT>\n"
        "{your private reasoning here}\n"
        "</THOUGHT>\n"
        "<MESSAGE>\n"
        "{what you actually say aloud, 1-3 sentences}\n"
        "</MESSAGE>\n"
        "Keep the message plain text (no brackets) and avoid double quotes unless you escape them."
        f" Aim for a natural, meaningful close to the conversation around turn {target_turns},"
        " but only conclude when it feels appropriate to you."
    )
    return (
        f"{instructions}\n\n"
        "=== SEEKER_SYSTEM_PROMPT ===\n"
        f"{seeker_prompt_text.strip()}\n"
        "=== SEEKER_PERSONA_CARD ===\n"
        f"{persona_block}\n"
        "=== END SEEKER CONTEXT ==="
    )


def build_supporter_system_prompt(
    supporter_prompt_text: str,
    persona_json: Dict[str, object],
    target_turns: int,
) -> str:
    persona_block = persona_to_json_string(persona_json)
    instructions = (
        "You are the Supporter in a crisis chat. Follow the Supporter system prompt "
        "rules exactly. The Seeker persona card is provided for context. Respond only "
        "to the latest Seeker utterance. Always use the <THOUGHT>...</THOUGHT> and "
        "<MESSAGE>...</MESSAGE> template, keeping the outward message plain text "
        "without annotations. Aim for an organic close around the instructed turn "
        "count, following the Seeker's readiness rather than forcing an ending."
    )
    return (
        f"{instructions}\n\n"
        "=== SUPPORTER_SYSTEM_PROMPT ===\n"
        f"{supporter_prompt_text.strip()}\n"
        "=== SEEKER_PERSONA_CARD ===\n"
        f"{persona_block}\n"
        "=== END SUPPORTER CONTEXT ==="
    )


def seeker_initial_user_prompt() -> str:
    return (
        "The Supporter greets you softly and says they are here to listen. "
        "Begin the conversation by sharing what brings you here today."
    )


def support_user_prompt(
    seeker_message: str,
    style_label: str,
    care_level: float,
    style_description: str,
) -> str:
    return (
        "The Seeker says:\n"
        f"{seeker_message.strip()}\n\n"
        f"Respond as the Supporter using care_level={care_level:.1f}. "
        f"Style cue: {style_label} ({style_description}). Ensure safety rules are met. "
        "Provide one consolidated reply. "
        "Think privately before speaking, then respond using EXACTLY this template "
        "(no extra text, no Markdown fences):\n"
        "<THOUGHT>\n"
        "{your internal reasoning here}\n"
        "</THOUGHT>\n"
        "<MESSAGE>\n"
        "{your outward reply, 1-4 sentences fitting the care_level}\n"
        "</MESSAGE>\n"
        "Keep the outward message plain text and avoid unescaped double quotes."
    )


@dataclass
class ConversationTurn:
    turn_index: int
    seeker_message: str
    seeker_thought: str
    supporter_variants: List[Dict[str, object]]
    canonical_care_level: float
    canonical_style: str


def generate_seeker_message(
    *,
    api_key: str,
    model_name: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    rng: random.Random,
    throttler: RequestThrottler,
) -> Dict[str, str]:
    reminder = (
        "Your previous reply did not follow the required template. Format exactly as "
        "<THOUGHT>...</THOUGHT> and <MESSAGE>...</MESSAGE> with no extra text."
    )
    last_cleaned = ""
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
        messages = (
            [{"role": "system", "content": system_prompt}]
            + history_to_messages(history)
        )
        if attempt > 1:
            messages = list(messages) + [{"role": "user", "content": reminder + "\n" + FORMAT_TEMPLATE_EXAMPLE}]
        try:
            content = openrouter_generate_text(
                api_key=api_key,
                model=model_name,
                messages=messages,
                temperature=0.8,
                top_p=0.9,
                max_tokens=2048,
                throttler=throttler,
            )
        except RuntimeError as err:
            delay = extract_retry_after_seconds(str(err))
            if delay:
                sleep_for = delay + rng.uniform(0, 1.5)
                print(f"[wait] Rate limit hit for seeker model; sleeping {sleep_for:.1f}s.")
                time.sleep(sleep_for)
                continue
            if attempt == MAX_AGENT_ATTEMPTS:
                raise ConversationError(f"Seeker model error: {err}") from err
            time.sleep(1.5 * attempt)
            continue
        cleaned = content.strip()
        last_cleaned = cleaned
        try:
            parsed = parse_reasoned_output(cleaned)
        except ConversationError as err:
            last_error = err
            if attempt == MAX_AGENT_ATTEMPTS:
                print(f"[warn] Forcing fallback formatting for seeker output: {err}")
                parsed_fallback = best_effort_parse_reasoned_output(cleaned)
                if parsed_fallback:
                    if not parsed_fallback["thought"]:
                        parsed_fallback["thought"] = FALLBACK_THOUGHT_TEXT
                    return parsed_fallback
                return {"thought": FALLBACK_THOUGHT_TEXT, "message": cleaned}
            time.sleep(1.0)
            continue
        if not parsed["message"]:
            if attempt == MAX_AGENT_ATTEMPTS:
                print("[warn] Seeker produced empty outward message; using fallback.")
                return {
                    "thought": parsed.get("thought", ""),
                    "message": last_cleaned or "",
                }
            time.sleep(1.0)
            continue
        return parsed
    raise ConversationError(
        f"Unable to get Seeker message after retries: {last_error} | payload={last_cleaned!r}"
    )


def generate_supporter_variants(
    *,
    api_key: str,
    model_name: str,
    system_prompt: str,
    history: List[Dict[str, str]],
    seeker_message: str,
    rng: random.Random,
    throttler: RequestThrottler,
) -> List[Dict[str, object]]:
    variants: List[Dict[str, object]] = []
    reminder = (
        "Your previous reply did not follow the required template. Format exactly as "
        "<THOUGHT>...</THOUGHT> and <MESSAGE>...</MESSAGE> with no extra text."
    )
    for style in RESPONSE_STYLES:
        care_level = style["care_level"]
        base_messages = (
            [{"role": "system", "content": system_prompt}]
            + history_to_messages(history)
            + [
                {
                    "role": "user",
                    "content": support_user_prompt(
                        seeker_message,
                        style_label=style["label"],
                        care_level=care_level,
                        style_description=style["description"],
                    ),
                }
            ]
        )
        last_cleaned = ""
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_AGENT_ATTEMPTS + 1):
            attempt_messages = base_messages
            if attempt > 1:
                attempt_messages = list(base_messages) + [
                    {"role": "user", "content": reminder + "\n" + FORMAT_TEMPLATE_EXAMPLE}
                ]
            try:
                content = openrouter_generate_text(
                    api_key=api_key,
                    model=model_name,
                    messages=attempt_messages,
                    temperature=style["temperature"],
                    top_p=style["top_p"],
                    max_tokens=2048,
                    throttler=throttler,
                )
            except RuntimeError as err:
                delay = extract_retry_after_seconds(str(err))
                if delay:
                    sleep_for = delay + rng.uniform(0, 1.5)
                    print(
                        f"[wait] Rate limit hit for supporter model ({style['label']}); sleeping {sleep_for:.1f}s."
                    )
                    time.sleep(sleep_for)
                    continue
                if attempt == MAX_AGENT_ATTEMPTS:
                    raise ConversationError(
                        f"Supporter model error ({style['label']}): {err}"
                    ) from err
                time.sleep(1.5 * attempt)
                continue
            cleaned = content.strip()
            last_cleaned = cleaned
            try:
                parsed = parse_reasoned_output(cleaned)
            except ConversationError as err:
                last_error = err
                if attempt == MAX_AGENT_ATTEMPTS:
                    print(
                        f"[warn] Forcing fallback formatting for supporter ({style['label']}): {err}"
                    )
                    parsed_fallback = best_effort_parse_reasoned_output(cleaned)
                    if parsed_fallback:
                        thought_text = (
                            parsed_fallback["thought"] or FALLBACK_THOUGHT_TEXT
                        )
                        variants.append(
                            {
                                "style": style["label"],
                                "care_level": round(care_level, 2),
                                "message": parsed_fallback["message"],
                                "thought": thought_text,
                            }
                        )
                        break
                    variants.append(
                        {
                            "style": style["label"],
                            "care_level": round(care_level, 2),
                            "message": cleaned,
                            "thought": FALLBACK_THOUGHT_TEXT,
                        }
                    )
                    break
                time.sleep(1.0)
                continue
            if not parsed["message"]:
                if attempt == MAX_AGENT_ATTEMPTS:
                    print(
                        f"[warn] Supporter produced empty outward message ({style['label']}); using fallback."
                    )
                    variants.append(
                        {
                            "style": style["label"],
                            "care_level": round(care_level, 2),
                            "message": last_cleaned,
                            "thought": parsed.get("thought", ""),
                        }
                    )
                    break
                time.sleep(1.0)
                continue
            variants.append(
                {
                    "style": style["label"],
                    "care_level": round(care_level, 2),
                    "message": parsed["message"],
                    "thought": parsed["thought"],
                }
            )
            break
        else:
            raise ConversationError(
                f"Supporter failed after retries ({style['label']}): {last_error}; payload={last_cleaned!r}"
            )
    return variants


def detect_termination(text: str) -> bool:
    lowered = text.lower()
    endings = (
        "goodbye",
        "got to go",
        "have to go",
        "signing off",
        "ending this conversation",
        "thanks, that's all",
        "this chat is over",
    )
    return any(keyword in lowered for keyword in endings)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Seeker/Supporter conversations for persona cards."
    )
    parser.add_argument(
        "--personas-dir",
        type=Path,
        default=Path("Artifacts/personas"),
        help="Directory containing persona card JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Artifacts/conversations"),
        help="Directory to write conversation transcripts.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env file with OPENROUTER_API_KEY.",
    )
    parser.add_argument(
        "--seeker-model",
        type=str,
        default=DEFAULT_SEEKER_MODEL,
        help=f"Model for Seeker messages (default: {DEFAULT_SEEKER_MODEL}).",
    )
    parser.add_argument(
        "--supporter-model",
        type=str,
        default=DEFAULT_SUPPORTER_MODEL,
        help=f"Model for Supporter messages (default: {DEFAULT_SUPPORTER_MODEL}).",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=DEFAULT_MIN_TURNS,
        help=f"Minimum turns per conversation (default: {DEFAULT_MIN_TURNS}).",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help=f"Maximum turns per conversation (default: {DEFAULT_MAX_TURNS}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of personas to process.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Number of personas to skip from the sorted list before processing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=DEFAULT_SLEEP,
        help=(
            "Base sleep (seconds) between persona conversations. Note that "
            "OpenRouter's free tiers have enforced RPM/token limits; a small pause "
            "helps avoid bursts."
        ),
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
        help="Skip personas whose conversation output already exists.",
    )
    parser.add_argument(
        "--crisis-profiles-dir",
        type=Path,
        default=Path("Artifacts/crisis_profiles"),
        help="Directory containing per-persona crisis profile JSON files.",
    )
    parser.add_argument(
        "--category-plan",
        type=Path,
        default=Path("Artifacts/crisis_category_plan.json"),
        help="JSON plan mapping persona IDs to age→category assignments.",
    )
    parser.add_argument(
        "--seeker-request-interval",
        type=float,
        default=DEFAULT_SEEKER_REQUEST_INTERVAL,
        help="Seconds between seeker model calls (default balances free-tier limit).",
    )
    parser.add_argument(
        "--seeker-request-jitter",
        type=float,
        default=DEFAULT_SEEKER_REQUEST_JITTER,
        help="Random jitter added to seeker interval.",
    )
    parser.add_argument(
        "--supporter-request-interval",
        type=float,
        default=DEFAULT_SUPPORTER_REQUEST_INTERVAL,
        help="Seconds between supporter model calls (default ≈0.4s).",
    )
    parser.add_argument(
        "--supporter-request-jitter",
        type=float,
        default=DEFAULT_SUPPORTER_REQUEST_JITTER,
        help="Random jitter added to supporter interval.",
    )
    parser.add_argument(
        "--conversation-schedule",
        type=Path,
        default=Path("Artifacts/conversation_schedule.json"),
        help="Schedule assigning personas to unique ages/categories. Auto-created if missing.",
    )
    parser.add_argument(
        "--schedule-seed",
        type=int,
        default=None,
        help="Seed for generating a new conversation schedule (defaults to --seed).",
    )
    parser.add_argument(
        "--min-category-count",
        type=int,
        default=6,
        help="Minimum appearances per crisis category when creating the schedule.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.min_turns < 1 or args.max_turns < args.min_turns:
        parser.error("Invalid turn range.")

    rng = random.Random(args.seed)

    try:
        api_key = load_env_value("OPENROUTER_API_KEY", args.env_file)
    except (KeyError, FileNotFoundError) as err:
        parser.error(str(err))
        return 2

    project_root = Path(__file__).resolve().parent.parent
    prompts_dir = project_root / "Prompts"
    seeker_prompt_text = load_text(prompts_dir / "Seeker_System_Prompt.json")
    supporter_prompt_text = load_text(prompts_dir / "Supporter_System_Prompt.json")

    try:
        category_plan = load_category_plan(args.category_plan)
    except (FileNotFoundError, ValueError) as err:
        parser.error(str(err))
        return 1
    try:
        persona_catalog, category_age_pool = build_persona_catalog(category_plan)
    except ValueError as err:
        parser.error(str(err))
        return 1

    persona_paths_all = sorted(args.personas_dir.glob("*.json"))
    if not persona_paths_all:
        parser.error(f"No persona JSON files found in {args.personas_dir}")
        return 1

    persona_cache: Dict[str, Dict[str, object]] = {}
    persona_records_all: List[Tuple[Path, str]] = []
    for path in persona_paths_all:
        persona = load_json(path)
        persona_id = str(persona.get("id") or path.stem)
        persona_cache[persona_id] = persona
        persona_records_all.append((path, persona_id))

    persona_ids_all = [persona_id for _, persona_id in persona_records_all]

    schedule_seed = (
        args.schedule_seed if args.schedule_seed is not None else args.seed
    )
    schedule_rng = random.Random(schedule_seed)
    conversation_schedule = load_or_create_conversation_schedule(
        schedule_path=args.conversation_schedule,
        persona_ids=persona_ids_all,
        persona_catalog=persona_catalog,
        category_age_pool=category_age_pool,
        min_per_category=args.min_category_count,
        rng=schedule_rng,
        seed_value=schedule_seed,
    )

    persona_records = persona_records_all
    if args.offset:
        persona_records = persona_records[args.offset :]
    if args.limit:
        persona_records = persona_records[: args.limit]
    if not persona_records:
        parser.error("No personas to process after applying offset/limit.")
        return 1

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    crisis_profiles_dir: Path = args.crisis_profiles_dir

    seeker_throttler = RequestThrottler(
        min_interval=args.seeker_request_interval,
        jitter=args.seeker_request_jitter,
        rng=rng,
    )
    supporter_throttler = RequestThrottler(
        min_interval=args.supporter_request_interval,
        jitter=args.supporter_request_jitter,
        rng=rng,
    )

    for index, (persona_path, persona_id) in enumerate(persona_records, start=1):
        persona = copy.deepcopy(persona_cache[persona_id])
        output_path = output_dir / f"{persona_id}.json"
        if args.skip_existing and output_path.exists():
            print(f"[skip] {persona_id} already has a conversation.")
            continue

        category_map = category_plan.get(persona_id)
        if not category_map:
            print(f"[warn] No category plan entry for {persona_id}; skipping.")
            continue
        try:
            crisis_profile = load_crisis_profile(crisis_profiles_dir, persona_id)
        except FileNotFoundError as err:
            print(f"[warn] {err}; skipping {persona_id}.")
            continue

        assignment = conversation_schedule.get(persona_id)
        if not assignment:
            print(f"[warn] No conversation schedule entry for {persona_id}; skipping.")
            continue
        selected_age = int(assignment.get("age"))
        age_str = str(selected_age)
        selected_category = str(assignment.get("category", "")).lower()
        planned_category = str(category_map.get(age_str, "")).lower()
        if not planned_category:
            print(
                f"[warn] Category plan missing age {age_str} for {persona_id}; skipping."
            )
            continue
        if selected_category and selected_category != planned_category:
            print(
                f"[warn] Schedule/category mismatch for {persona_id} age {age_str}; "
                f"using planned category {planned_category}."
            )
            selected_category = planned_category
        else:
            selected_category = planned_category
        crisis_entry = crisis_profile.get(age_str)
        if not crisis_entry:
            print(
                f"[warn] Crisis profile missing age {age_str} for {persona_id}; skipping."
            )
            continue
        crisis_summary = str(crisis_entry.get("summary", "")).strip()
        crisis_confidence = crisis_entry.get("confidence")

        persona_view = json.loads(json.dumps(persona))
        presenting = persona_view.setdefault("presenting_problem", {})
        presenting["summary"] = crisis_summary
        presenting["category"] = selected_category
        presenting["age_context"] = {
            "age": selected_age,
            "source": "crisis_profile",
            "confidence": crisis_confidence,
        }
        current_demo = persona_view.get("current_socio_demographics")
        if isinstance(current_demo, dict):
            current_demo["age"] = selected_age

        turn_target = rng.randint(args.min_turns, args.max_turns)
        print(
            f"[{index}/{len(persona_records)}] Persona {persona_id}: generating "
            f"{turn_target} turns."
        )

        seeker_system = build_seeker_system_prompt(
            seeker_prompt_text,
            persona_view,
            turn_target,
        )
        supporter_system = build_supporter_system_prompt(
            supporter_prompt_text,
            persona_view,
            turn_target,
        )

        seeker_history: List[Dict[str, str]] = [
            {"role": "user", "content": seeker_initial_user_prompt()}
        ]
        supporter_history: List[Dict[str, str]] = []
        turns: List[ConversationTurn] = []

        for turn_index in range(1, turn_target + 1):
            try:
                seeker_output = generate_seeker_message(
                    api_key=api_key,
                    model_name=args.seeker_model,
                    system_prompt=seeker_system,
                    history=seeker_history,
                    rng=rng,
                    throttler=seeker_throttler,
                )
            except ConversationError as err:
                print(f"[warn] Stopping early for {persona_id}: {err}")
                break
            seeker_message = seeker_output["message"]
            seeker_thought = seeker_output["thought"]
            seeker_history.append({"role": "assistant", "content": seeker_message})

            try:
                supporter_variants = generate_supporter_variants(
                    api_key=api_key,
                    model_name=args.supporter_model,
                    system_prompt=supporter_system,
                    history=supporter_history,
                    seeker_message=seeker_message,
                    rng=rng,
                    throttler=supporter_throttler,
                )
            except ConversationError as err:
                print(f"[warn] Stopping early for {persona_id}: {err}")
                break

            canonical = next(
                (
                    variant
                    for variant in supporter_variants
                    if variant["style"] == "baseline"
                ),
                supporter_variants[0],
            )

            turns.append(
                ConversationTurn(
                    turn_index=turn_index,
                    seeker_message=seeker_message,
                    seeker_thought=seeker_thought,
                    supporter_variants=supporter_variants,
                    canonical_care_level=canonical["care_level"],
                    canonical_style=canonical["style"],
                )
            )

            supporter_history.append({"role": "user", "content": seeker_message})
            supporter_history.append({"role": "assistant", "content": canonical["message"]})
            seeker_history.append({"role": "user", "content": canonical["message"]})

            if detect_termination(seeker_message) or detect_termination(
                canonical["message"]
            ):
                print(f"[info] Early termination triggered for {persona_id}.")
                break

        if not turns:
            print(f"[error] No turns recorded for {persona_id}; skipping save.")
            continue

        conversation_payload = {
            "persona_id": persona_id,
            "persona_path": str(persona_path),
            "seeker_model": args.seeker_model,
            "supporter_model": args.supporter_model,
            "turns_recorded": len(turns),
            "desired_turns": turn_target,
            "crisis_context": {
                "age": selected_age,
                "summary": crisis_summary,
                "category": selected_category,
                "confidence": crisis_confidence,
            },
            "seeker_system_prompt": seeker_system,
            "supporter_system_prompt": supporter_system,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "seeker_message": turn.seeker_message,
                    "seeker_thought": turn.seeker_thought,
                    "supporter_responses": turn.supporter_variants,
                    "canonical_care_level": turn.canonical_care_level,
                    "canonical_style": turn.canonical_style,
                }
                for turn in turns
            ],
        }

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(conversation_payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        sleep_time = max(0.0, args.sleep + rng.uniform(0, args.jitter))
        time.sleep(sleep_time)

    print("Conversation generation complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
