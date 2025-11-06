#!/usr/bin/env python3
"""Publish personas, crisis profiles, and conversations to the Hugging Face Hub.

The script expects the full synthetic datasets to be available under `artifacts/`.
We keep only tiny samples in `examples/`, so this utility is the canonical way to
materialise the full corpora on the Hub for downstream use.

It creates/updates three dataset repos (configurable via CLI), using the chunked
storage backend that is automatically enabled for dataset repositories on the Hub.

Usage example:

```bash
uv run python Scripts/publish_to_hf.py \\
  --personas-repo reza2kn/kakoverse-personas-v0 \\
  --crisis-repo reza2kn/kakoverse-crisis-profiles-v0 \\
  --conversations-repo reza2kn/kakoverse-conversations-v0
```

This script reads the HF token from the `HF_TOKEN` environment variable. If not
set, it will fall back to your `.env` file (kept out of git).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Dataset
from dotenv import dotenv_values
from huggingface_hub import HfApi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def load_token(env_file: Path) -> str:
    """Fetch HF token from ENV or .env."""

    token = os.getenv("HF_TOKEN")
    if token:
        return token

    if env_file.exists():
        values = dotenv_values(env_file)
        token = values.get("HF_TOKEN")
        if token:
            return token

    raise RuntimeError(
        "HF token not set. Export HF_TOKEN or add it to your .env file."
    )


def sorted_persona_files() -> List[Path]:
    personas_dir = ARTIFACTS_DIR / "personas"
    if not personas_dir.exists():
        raise FileNotFoundError(
            f"Persona directory {personas_dir} is missing. "
            "Generate personas first (see Makefile)."
        )
    return sorted(personas_dir.glob("*.json"))


def sorted_profile_files() -> List[Path]:
    profiles_dir = ARTIFACTS_DIR / "crisis_profiles"
    if not profiles_dir.exists():
        raise FileNotFoundError(
            f"Crisis profiles directory {profiles_dir} not found. "
            "Run generate_crisis_profiles.py beforehand."
        )
    return sorted(profiles_dir.glob("*.json"))


def sorted_conversation_files() -> List[Path]:
    conversations_dir = ARTIFACTS_DIR / "conversations"
    if not conversations_dir.exists():
        raise FileNotFoundError(
            f"Conversation directory {conversations_dir} not found. "
            "Run generate_conversations.py beforehand."
        )
    return sorted(conversations_dir.glob("*.json"))


def build_persona_records() -> List[Dict]:
    records: List[Dict] = []
    for path in sorted_persona_files():
        with path.open("r", encoding="utf-8") as handle:
            persona = json.load(handle)
        persona_id = persona.get("id", path.stem)
        presenting = persona.get("presenting_problem") or {}
        record = {
            "id": persona_id,
            "persona": persona,
            "presenting_problem_summary": presenting.get("summary"),
            "presenting_problem_category": presenting.get("category"),
        }
        records.append(record)
    return records


def build_crisis_records() -> List[Dict]:
    records: List[Dict] = []
    for path in sorted_profile_files():
        with path.open("r", encoding="utf-8") as handle:
            profile_map: Dict[str, Dict[str, object]] = json.load(handle)
        entries = []
        for age_str, payload in sorted(profile_map.items(), key=lambda kv: int(kv[0])):
            entry = {
                "age": int(age_str),
                "summary": payload.get("summary"),
                "category": payload.get("category"),
                "confidence": payload.get("confidence"),
            }
            entries.append(entry)
        records.append(
            {
                "id": path.stem,
                "entries": entries,
            }
        )
    return records


def build_conversation_records() -> List[Dict]:
    """Flatten conversation payloads."""

    records: List[Dict] = []
    for path in sorted_conversation_files():
        with path.open("r", encoding="utf-8") as handle:
            convo = json.load(handle)
        context = convo.get("crisis_context") or {}
        record = {
            "persona_id": convo.get("persona_id", path.stem),
            "desired_turns": convo.get("desired_turns"),
            "turns_recorded": convo.get("turns_recorded"),
            "crisis_summary": context.get("summary"),
            "crisis_category": context.get("category"),
            "seeker_model": convo.get("seeker_model"),
            "supporter_model": convo.get("supporter_model"),
            "turns": convo.get("turns"),
        }
        records.append(record)
    return records


def push_dataset(
    records: List[Dict],
    repo_id: str,
    token: str,
    card_markdown: str,
    *,
    exist_ok: bool = True,
) -> None:
    """Create/update a dataset repo with the provided records."""

    if not records:
        raise ValueError(f"No records found for {repo_id}.")

    dataset = Dataset.from_list(records)
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=exist_ok,
        token=token,
    )

    dataset.push_to_hub(
        repo_id,
        token=token,
        private=False,
        commit_message="Update dataset",
    )

    api.upload_file(
        path_or_fileobj=card_markdown.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


def build_cards() -> Dict[str, str]:
    """Dataset cards for the three corpora."""

    base_card = """---
annotations_creators:
- synthetic
language:
- en
license:
- apache-2.0
pretty_name: {title}
task_categories:
- text-generation
size_categories:
- 10K<n<100K
---

# {title}

{description}

## Usage

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}")
print(ds["train"][0])
```

## Provenance

- Generator version: {generator_version}
- Contact: {contact}
"""

    return {
        "personas": base_card.format(
            title="KakoVerse Personas v0",
            description="69 richly structured persona cards spanning global cities and decades.",
            repo_id="{repo_id}",
            generator_version="Scripts/publish_to_hf.py",
            contact="https://huggingface.co/Reza2kn",
        ),
        "crisis": base_card.format(
            title="KakoVerse Crisis Profiles v0",
            description="Age-indexed crisis summaries (ages 20-100) aligned with the persona grid.",
            repo_id="{repo_id}",
            generator_version="Scripts/publish_to_hf.py",
            contact="https://huggingface.co/Reza2kn",
        ),
        "conversations": base_card.format(
            title="KakoVerse Conversations v0",
            description="69 multi-turn empathy dialogues with baseline/empathetic/cold Supporter variants.",
            repo_id="{repo_id}",
            generator_version="Scripts/publish_to_hf.py",
            contact="https://huggingface.co/Reza2kn",
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish KakoVerse datasets to HF Hub.")
    parser.add_argument(
        "--personas-repo",
        required=True,
        help="Destination dataset repo for persona cards (e.g., reza2kn/kakoverse-personas-v0).",
    )
    parser.add_argument(
        "--crisis-repo",
        required=True,
        help="Destination dataset repo for crisis profiles (e.g., reza2kn/kakoverse-crisis-profiles-v0).",
    )
    parser.add_argument(
        "--conversations-repo",
        required=True,
        help="Destination dataset repo for conversations (e.g., reza2kn/kakoverse-conversations-v0).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Path to .env containing HF_TOKEN fallback.",
    )
    args = parser.parse_args()

    token = load_token(args.env_file)
    cards = build_cards()

    persona_records = build_persona_records()
    crisis_records = build_crisis_records()
    conversation_records = build_conversation_records()

    print(f"[personas] Uploading {len(persona_records)} rows to {args.personas_repo} ...")
    push_dataset(
        persona_records,
        repo_id=args.personas_repo,
        token=token,
        card_markdown=cards["personas"].format(repo_id=args.personas_repo),
    )
    print("[personas] Done.")

    print(f"[crisis] Uploading {len(crisis_records)} rows to {args.crisis_repo} ...")
    push_dataset(
        crisis_records,
        repo_id=args.crisis_repo,
        token=token,
        card_markdown=cards["crisis"].format(repo_id=args.crisis_repo),
    )
    print("[crisis] Done.")

    print(f"[conversations] Uploading {len(conversation_records)} rows to {args.conversations_repo} ...")
    push_dataset(
        conversation_records,
        repo_id=args.conversations_repo,
        token=token,
        card_markdown=cards["conversations"].format(repo_id=args.conversations_repo),
    )
    print("[conversations] Done.")


if __name__ == "__main__":
    main()
