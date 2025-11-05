#!/usr/bin/env python3
"""Fire small bursts at OpenRouter models to observe rate limits.

Usage:
  .venv/bin/python Scripts/probe_openrouter_rate_limits.py \
      --models alibaba/tongyi-deepresearch-30b-a3b:free minimax/minimax-m2:free \
      --requests 50 --interval 0.4

This sends simple chat completions at a fixed interval and prints the
status/headers so you can see when 429s or Retry-After headers appear.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Sequence

import requests


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def load_api_key(env_file: Path | None) -> str:
    if "OPENROUTER_API_KEY" in os.environ and os.environ["OPENROUTER_API_KEY"]:
        return os.environ["OPENROUTER_API_KEY"]
    if env_file and env_file.exists():
        with env_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                striped = line.strip()
                if not striped or striped.startswith("#") or "=" not in striped:
                    continue
                lhs, rhs = striped.split("=", 1)
                if lhs.strip() == "OPENROUTER_API_KEY":
                    key = rhs.strip().strip("\"'")
                    if key:
                        return key
    raise RuntimeError("OPENROUTER_API_KEY not found in environment or env file")


def make_payload(model: str) -> Dict[str, object]:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Return the word 'ack'."},
        ],
        "temperature": 0.1,
        "max_output_tokens": 5,
    }


def fire_request(api_key: str, model: str, timeout: float) -> requests.Response:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-Title": "vf-CrisisSupport rate-limit probe",
    }
    payload = make_payload(model)
    resp = requests.post(
        OPENROUTER_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    return resp


def probe_model(
    *,
    api_key: str,
    model: str,
    requests_per_model: int,
    interval: float,
    timeout: float,
) -> None:
    print(f"\n🔍 Probing model: {model}")
    print(f"Requests: {requests_per_model}, interval: {interval:.3f}s")
    successes = 0
    for idx in range(1, requests_per_model + 1):
        start = time.perf_counter()
        try:
            resp = fire_request(api_key, model, timeout)
        except requests.RequestException as err:
            print(f"[{idx}] ERROR: {err}")
            break

        latency = time.perf_counter() - start
        status = resp.status_code
        retry_after = resp.headers.get("Retry-After")
        remaining = resp.headers.get("X-RateLimit-Remaining")
        usage = resp.headers.get("OpenRouter-Processing-Ms")

        log = (
            f"[{idx}] status={status} latency={latency*1000:.1f}ms"
            f" retry_after={retry_after}" if retry_after else f"[{idx}] status={status} latency={latency*1000:.1f}ms"
        )
        print(log, end="")
        if remaining:
            print(f" remaining={remaining}", end="")
        if usage:
            print(f" process_ms={usage}", end="")
        print()

        if status == 200:
            successes += 1
        elif status == 429:
            print("⚠️  Hit rate limit (429). Consider increasing interval.")
            break
        else:
            print(f"Received non-200 response: {resp.text[:200]}")

        # Respect Retry-After if provided
        if retry_after:
            try:
                sleep_for = float(retry_after)
            except ValueError:
                sleep_for = interval
            print(f"Sleeping {sleep_for:.2f}s (Retry-After)")
            time.sleep(sleep_for)
        else:
            time.sleep(interval)

    print(f"✅ Successful responses: {successes}/{requests_per_model}")


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe OpenRouter rate limits by sending bursts of chat completions."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "alibaba/tongyi-deepresearch-30b-a3b:free",
            "minimax/minimax-m2:free",
        ],
        help="Model identifiers to probe (space separated).",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Number of requests per model (default: 50).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds to sleep between requests (default: 0.5s).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP request timeout in seconds (default: 20s).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Path to .env containing OPENROUTER_API_KEY.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        api_key = load_api_key(args.env_file)
    except RuntimeError as err:
        print(f"ERROR: {err}")
        return 1

    for model in args.models:
        probe_model(
            api_key=api_key,
            model=model,
            requests_per_model=args.requests,
            interval=args.interval,
            timeout=args.timeout,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
