#!/usr/bin/env python3
"""
Best-effort cleanup for generated conversations.

When the LLM omits closing tags we may store the raw `<THOUGHT>...</THOUGHT><MESSAGE>...`
payload as the outward message with a placeholder thought. This utility re-parses those
payloads and restores dedicated `thought`/`message` fields so downstream pipelines do not
need to strip tags manually.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

THOUGHT_START_TAG = "<THOUGHT>"
THOUGHT_END_TAG = "</THOUGHT>"
MESSAGE_START_TAG = "<MESSAGE>"
MESSAGE_END_TAG = "</MESSAGE>"


def best_effort_parse(payload: str) -> Optional[Dict[str, str]]:
    if not isinstance(payload, str):
        return None
    cleaned = payload.strip()
    if not cleaned:
        return None

    thought = ""
    if THOUGHT_START_TAG in cleaned and THOUGHT_END_TAG in cleaned:
        fragment = cleaned.split(THOUGHT_START_TAG, 1)[1]
        fragment = fragment.split(THOUGHT_END_TAG, 1)[0]
        thought = fragment.strip()

    if MESSAGE_START_TAG not in cleaned:
        return None
    message_section = cleaned.split(MESSAGE_START_TAG, 1)[1]
    if MESSAGE_END_TAG in message_section:
        message_section = message_section.split(MESSAGE_END_TAG, 1)[0]
    message = message_section.strip()
    if not message:
        return None
    return {"thought": thought, "message": message}


def clean_turn(turn: Dict[str, object]) -> bool:
    changed = False

    seeker_message = turn.get("seeker_message")
    if isinstance(seeker_message, str):
        needs_clean = (
            THOUGHT_START_TAG in seeker_message
            or str(turn.get("seeker_thought", "")).startswith("Auto-generated thought")
        )
        if needs_clean:
            parsed = best_effort_parse(seeker_message)
            if parsed:
                if (
                    not turn.get("seeker_thought")
                    or str(turn.get("seeker_thought")).startswith("Auto-generated thought")
                ):
                    turn["seeker_thought"] = parsed["thought"]
                turn["seeker_message"] = parsed["message"]
                changed = True

    for variant in turn.get("supporter_responses", []):
        if not isinstance(variant, dict):
            continue
        message = variant.get("message")
        if not isinstance(message, str):
            continue
        needs_clean = THOUGHT_START_TAG in message or str(
            variant.get("thought", "")
        ).startswith("Auto-generated thought")
        if not needs_clean:
            continue
        parsed = best_effort_parse(message)
        if not parsed:
            continue
        if (
            not variant.get("thought")
            or str(variant.get("thought")).startswith("Auto-generated thought")
        ):
            variant["thought"] = parsed["thought"]
        variant["message"] = parsed["message"]
        changed = True

    return changed


def process_file(path: Path, inplace: bool = True) -> bool:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    changed_any = False
    for turn in data.get("turns", []):
        if isinstance(turn, dict):
            if clean_turn(turn):
                changed_any = True

    if changed_any and inplace:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    return changed_any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Post-process conversation JSON files to normalise thought/message fields."
    )
    parser.add_argument(
        "--conversation-dir",
        type=Path,
        default=Path("artifacts/conversations"),
        help="Directory containing conversation JSON transcripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report files that would change without modifying them.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    conversation_dir: Path = args.conversation_dir
    if not conversation_dir.exists():
        parser.error(f"{conversation_dir} does not exist.")
        return 2

    paths = sorted(conversation_dir.glob("*.json"))
    if not paths:
        print("No conversation files found; nothing to do.")
        return 0

    changed = 0
    for path in paths:
        updated = process_file(path, inplace=not args.dry_run)
        if updated:
            changed += 1
            status = "would change" if args.dry_run else "fixed"
            print(f"[{status}] {path}")

    if changed == 0:
        print("No fixes required.")
    else:
        print(f"Updated {changed} conversation files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
