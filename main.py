#!/usr/bin/env python3
"""
Persona Tester — Entry point.

Usage:
    python main.py --persona personas/example.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an LLM persona test session against a Flutter web app.",
    )
    parser.add_argument(
        "--persona",
        required=True,
        help="Path to a persona JSON file (e.g. personas/example.json).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import here so logging is configured first
    from agent import run_session  # noqa: E402

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║          Persona Tester — LLM Vision Agent             ║")
    print("╚══════════════════════════════════════════════════════════╝\n")
    print(f"  Persona file : {args.persona}")

    session = await run_session(args.persona)

    # ---- Summary ----
    outcome = session.get("outcome", "unknown")
    total_steps = len(session.get("steps", []))
    persona_name = session.get("persona", {}).get("name", "?")

    print("\n┌──────────────────────────────────────────────────────────┐")
    print(f"│  Session complete                                        │")
    print(f"│  Persona : {persona_name:<47}│")
    print(f"│  Steps   : {total_steps:<47}│")
    print(f"│  Outcome : {outcome:<47}│")
    print(f"│  Log     : {session.get('session_id', '?'):<47}│")
    print("└──────────────────────────────────────────────────────────┘\n")

    if outcome == "stuck":
        sys.exit(2)
    elif outcome == "max_steps":
        sys.exit(1)
    # outcome == "completed" → exit 0


if __name__ == "__main__":
    asyncio.run(_main())
