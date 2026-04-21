"""
Ollama vision-LLM client.

Sends a base64 screenshot + persona context to a local Qwen-VL model
and parses the structured JSON action response.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx

from config import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TIMEOUT


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are EXACTLY this person:
- **Name:** {name}
- **Description:** {description}
- **Traits:** {traits}
- **Your Goal:** {goal}

CRITICAL INSTRUCTION: You are NOT an AI assistant or a testing agent. You are {name}. You must evaluate the screen, make decisions, and write your reasoning strictly from the perspective, knowledge level, vocabulary, and emotional state of {name}. 
- If your description says you are not tech-savvy, you must struggle with complex UIs, ignore subtle icons, and rely on explicit text. 
- If you are impatient, you should act hastily or get frustrated.
- Never use technical jargon (e.g., "DOM", "navigation bar", "UI element") unless your specific persona would know those words.

## Rules
1. You can ONLY see the screenshot provided. You have NO access to the DOM, \
accessibility tree, or any other metadata. Decide your next action based \
solely on the pixels you see.
2. Identify what you want to interact with and output its location \
as a **normalised bounding box** on a **1000×1000 grid**, formatted as \
`[x1, y1, x2, y2]` where `(x1, y1)` is the top-left corner and \
`(x2, y2)` is the bottom-right corner of the element. \
`(0, 0)` is the top-left of the screen and `(1000, 1000)` is the bottom-right.
3. Respond with **only** a single JSON object — no markdown fences, no \
commentary outside the JSON.

## Available actions
| action             | description |
|--------------------|-------------|
| `click`            | Click the center of the bounding box. Use for buttons, links, checkboxes, menu items, video thumbnails, etc. NEVER use this to focus a text field. Do NOT include a `value` field. |
| `type`             | Automatically clicks the center of the bounding box to focus it, CLEARS any existing text, then types the text in `value`. Use for ALL text input. You MUST include the `value` field. |
| `press_key`        | Press a keyboard key. Set `value` to the key name: `Enter`, `Escape`, `Tab`, `Backspace`, `ArrowDown`, `ArrowUp`, etc. The bounding_box is ignored for this action. |
| `scroll_down`      | Move cursor to center of bounding box, then scroll DOWN to see lower content. |
| `scroll_up`        | Move cursor to center of bounding box, then scroll UP to see higher content. |
| `scroll_right`     | Move cursor to center of bounding box, then scroll RIGHT. |
| `scroll_left`      | Move cursor to center of bounding box, then scroll LEFT. |
| `done`             | You have achieved your goal. |
| `stuck`            | You are confused, frustrated, or cannot figure out how to proceed based on your technological skill level. |
| `wait`             | Do nothing and wait for the page to load if the screen is blank or showing a loading spinner. |

**CRITICAL RULES**:
- **Text Input:** NEVER use `click` to select a text box or search bar. The `type` action automatically handles clicking, focusing, and clearing the field. ALWAYS use `type` directly in one single step.
- After typing into a search bar, use `press_key` with `value: "Enter"` to \
submit the search. Do NOT type again if the text is already visible.
- NEVER repeat the exact same action twice in a row. If your previous action \
did not work, try a DIFFERENT approach that makes sense for your persona.

## Response schema (strict)
```
{{
  "reasoning": "<Write strictly in the first person ('I'). 1. State your immediate human reaction to the screen based on your Traits. 2. Explain your next action, heavily influenced by your Description.>",
  "action": "click" | "type" | "press_key" | "scroll_down" | "scroll_up" | "scroll_right" | "scroll_left" | "wait" | "done" | "stuck",
  "bounding_box": [x1, y1, x2, y2],
  "value": "<text to type (for 'type') or key name (for 'press_key')>",
  "confidence": <float 0.0–1.0>
}}
```

IMPORTANT: The bounding_box MUST be an array of exactly 4 integers in the \
range [0, 1000], ordered as [x1, y1, x2, y2]. For "done", "stuck", "wait", \
and "press_key" actions, use [0, 0, 1000, 1000].
"""


# ---------------------------------------------------------------------------
# History formatting
# ---------------------------------------------------------------------------

def _format_history(history: list[dict]) -> str:
    """Return a compact text summary of recent actions.

    Includes action type, value/key, and success status so the model
    knows what already happened and avoids repeating itself.
    """
    if not history:
        return "No previous actions."
    lines: list[str] = []
    for entry in history:
        step = entry.get("step", "?")
        action = entry.get("action", {})
        act = action.get("action", "")
        value = action.get("value", "")
        success = entry.get("success", True)
        status = "OK" if success else "FAILED"

        detail = ""
        if act == "type" and value:
            detail = f" → typed '{value}'"
        elif act == "press_key" and value:
            detail = f" → pressed {value}"
        elif act == "click":
            detail = " → clicked"

        reasoning = action.get("reasoning", "")
        lines.append(f"Step {step}: [{act}]{detail} [{status}] {reasoning}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ask_llm(
    screenshot_base64: str,
    persona: dict[str, Any],
    history: list[dict],
) -> dict[str, Any]:
    """Send a screenshot to the Ollama Qwen-VL model and return the parsed action dict.

    Raises ``RuntimeError`` on communication or parse failures so the caller
    can treat them as retriable errors.
    """

    system_prompt = _SYSTEM_PROMPT.format(
        name=persona["name"],
        description=persona["description"],
        traits=", ".join(persona.get("traits", [])),
        goal=persona["goal"],
    )

    history_text = _format_history(history)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"## Action history\n{history_text}\n\n"
                "## Current screenshot\nAnalyze the screenshot below and "
                "respond with a single JSON action object."
            ),
            "images": [screenshot_base64],
        },
    ]

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
        },
    }

    async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
        try:
            resp = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

    body = resp.json()
    raw_content: str = body.get("message", {}).get("content", "")
    return _parse_action(raw_content)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_action(raw: str) -> dict[str, Any]:
    """Extract the first JSON object from the model response."""

    # Try direct parse first
    try:
        return _validate(json.loads(raw))
    except (json.JSONDecodeError, ValueError):
        pass

    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
    try:
        return _validate(json.loads(cleaned))
    except (json.JSONDecodeError, ValueError):
        pass

    # Extract the first balanced JSON object (handles models that output
    # multiple JSON blocks or trailing commentary).
    first_obj = _extract_first_json_object(raw)
    if first_obj is not None:
        try:
            return _validate(json.loads(first_obj))
        except (json.JSONDecodeError, ValueError):
            pass

    raise RuntimeError(f"Failed to parse LLM response as JSON:\n{raw}")


def _extract_first_json_object(text: str) -> str | None:
    """Find and return the first balanced ``{ … }`` substring in *text*.

    Uses a simple brace-depth counter so it stops at the matching close
    brace instead of greedily eating everything up to the last ``}``.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\":
            if in_string:
                escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _validate(data: dict) -> dict:
    """Minimal schema validation with auto-promotion."""
    if "action" not in data:
        raise ValueError("Missing 'action' key in LLM response")
    
    # Updated allowed actions
    if data["action"] not in (
        "click", "type", "press_key", "scroll_down", "scroll_up", 
        "scroll_right", "scroll_left", "wait", "done", "stuck",
    ):
        raise ValueError(f"Unknown action: {data['action']}")

    # ... (keep the rest of the validation logic the same)

    # Auto-promote: if the model says "click" but provides a non-empty
    # "value" field, it clearly intends to type into a text field.
    if data["action"] == "click" and data.get("value"):
        data["action"] = "type"

    # Coerce bounding_box to list[int]
    if "bounding_box" in data:
        data["bounding_box"] = [int(v) for v in data["bounding_box"]]
    else:
        data["bounding_box"] = [0, 0, 1000, 1000]
    # Defaults
    data.setdefault("reasoning", "")
    data.setdefault("confidence", 0.5)
    return data
