"""
Core agent loop.

Orchestrates the screenshot → LLM → action cycle, logging every step
into a JSON session file.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from browser import BrowserController
from llm import ask_llm
from config import MAX_STEPS, SCREENSHOT_DIR, VIEWPORT_WIDTH, VIEWPORT_HEIGHT

log = logging.getLogger(__name__)


async def run_session(persona_path: str) -> dict[str, Any]:
    """Run a full persona testing session and return the session log dict."""

    # ---- Load persona ----
    persona = _load_persona(persona_path)
    persona_name: str = persona.get("name", "unnamed")
    log.info("Loaded persona: %s", persona_name)

    # ---- Prepare session directory ----
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_id = f"{persona_name}_{timestamp}"
    session_dir = Path(SCREENSHOT_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session_log: dict[str, Any] = {
        "persona": persona,
        "session_id": session_id,
        "started_at": timestamp,
        "viewport": {"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        "steps": [],
        "outcome": "max_steps",
    }

    history: list[dict] = []
    browser = BrowserController()

    try:
        # ---- Launch browser ----
        await browser.launch()
        # Give Flutter's canvas a moment to render
        await asyncio.sleep(2)

        for step_num in range(1, MAX_STEPS + 1):
            log.info("──── Step %d / %d ────", step_num, MAX_STEPS)

            # Wrap the entire step so a closed browser doesn't crash us
            try:
                # 1. Screenshot
                screenshot_file = str(session_dir / f"step_{step_num:03d}.png")
                await browser.save_screenshot(screenshot_file)
                screenshot_b64 = await browser.take_screenshot()
            except Exception as exc:
                log.error("Browser unavailable at step %d: %s", step_num, exc)
                session_log["outcome"] = "error"
                break

            # 2. Ask LLM
            try:
                action = await ask_llm(screenshot_b64, persona, history)
            except RuntimeError as exc:
                log.error("LLM error at step %d: %s", step_num, exc)
                step_entry = _make_step_entry(
                    step_num, screenshot_file,
                    reasoning=f"LLM error: {exc}",
                    action={"action": "error"},
                    center=None,
                    success=False,
                )
                session_log["steps"].append(step_entry)
                history.append(step_entry)
                await asyncio.sleep(1)
                continue

            # 2b. Repeated-action warning (informational only).
            #     The model's own history context + 15-step max provide
            #     sufficient guard against infinite loops.
            if (
                action.get("action") in ("click", "type")
                and _is_action_cycle(history, action)
            ):
                log.warning(
                    "Repeated %s '%s' detected (step %d)",
                    action["action"], action.get("value", ""), step_num,
                )

            # 3. Compute center for logging
            bbox = action.get("bounding_box", [0, 0, 1000, 1000])
            center = _compute_center(bbox)

            # 3b. Save annotated debug screenshot
            annotated_file = str(session_dir / f"step_{step_num:03d}_debug.png")
            _save_debug_screenshot(screenshot_file, annotated_file, bbox, center, action)

            # 4. Pretty-print step
            _print_step(step_num, action, center)

            # 5. Terminal actions
            if action["action"] in ("done", "stuck"):
                step_entry = _make_step_entry(
                    step_num, screenshot_file,
                    reasoning=action.get("reasoning", ""),
                    action=action, center=center, success=True,
                )
                session_log["steps"].append(step_entry)
                session_log["outcome"] = "completed" if action["action"] == "done" else "stuck"
                break

            # 6. Execute action
            try:
                keep_going = await browser.execute_action(action)
            except Exception as exc:
                log.error("Action execution error: %s", exc)
                # If the browser is gone, stop the loop
                if "closed" in str(exc).lower():
                    session_log["outcome"] = "error"
                    break
                keep_going = True  # retriable — count as a step

            step_entry = _make_step_entry(
                step_num, screenshot_file,
                reasoning=action.get("reasoning", ""),
                action=action, center=center, success=keep_going,
            )
            session_log["steps"].append(step_entry)
            history.append(step_entry)

            if not keep_going:
                session_log["outcome"] = "completed"
                break

    finally:
        try:
            await browser.close()
        except Exception:
            pass  # already closed or interrupted

    # ---- Persist session log ----
    log_path = session_dir / "session.json"
    log_path.write_text(json.dumps(session_log, indent=2, default=str))
    log.info("Session log saved to %s", log_path)

    return session_log


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_persona(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _bboxes_overlap(a: list[int], b: list[int], threshold: float = 0.5) -> bool:
    """Return True if two [x1, y1, x2, y2] boxes have IoU above *threshold*.

    Used by loop detection so that clicks on *different* UI elements are
    not mistakenly treated as repeated actions.
    """
    if len(a) != 4 or len(b) != 4:
        return False
    # Intersection
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return False
    # Union
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    if union == 0:
        return False
    return (inter / union) >= threshold


def _action_sig(action: dict) -> str:
    """Return a compact signature for an action dict.

    The signature includes the action type and a coarse spatial bucket
    (bbox center quantised to a 4×4 grid) so that clicks on the *same*
    region are considered identical even if the pixel coords jitter.
    """
    act = action.get("action", "?")
    val = action.get("value", "")
    bbox = action.get("bounding_box", [500, 500, 500, 500])
    if len(bbox) == 4:
        cx = (bbox[0] + bbox[2]) // 2
        cy = (bbox[1] + bbox[3]) // 2
        # Quantise to 4×4 grid (each cell = 250 units on 1000 scale)
        qx, qy = cx // 250, cy // 250
    else:
        qx, qy = 0, 0
    return f"{act}|{val}|{qx},{qy}"


def _is_action_cycle(history: list[dict], next_action: dict) -> bool:
    """Detect repeating action cycles of length 1 or 2.

    Thresholds are intentionally set high so the model has a chance to
    self-correct before we intervene:
    - A, A, A        → 3 consecutive identical actions  (cycle length 1)
    - A, B, A, B     → 4-step alternating cycle         (cycle length 2)
    """
    if not history:
        return False

    # Build signatures, excluding any auto-injected actions so they
    # don't pollute the pattern.
    sigs: list[str] = []
    for h in history:
        act = h.get("action", {})
        if act.get("reasoning", "").startswith("AUTO:"):
            continue
        sigs.append(_action_sig(act))

    next_sig = _action_sig(next_action)

    # --- Cycle length 1: A, A, A (3 identical in a row) ---
    if len(sigs) >= 2 and sigs[-1] == next_sig and sigs[-2] == next_sig:
        return True

    # --- Cycle length 2: A, B, A, B (full 4-step alternating cycle) ---
    # History ends with [..., A, B, A] and next = B  →  pattern is A-B-A-B
    # OR: history ends with [..., B, A, B] and next = A →  same cycle
    if len(sigs) >= 3:
        s1, s2, s3 = sigs[-3], sigs[-2], sigs[-1]
        if s1 == s3 and s2 == next_sig and s1 != s2:
            return True

    return False


def _compute_center(bbox: list[int]) -> dict[str, float]:
    """Compute pixel center from Qwen-VL [xmin, ymin, xmax, ymax] normalised box."""
    # 1. Unpack as X, Y, X, Y
    xmin, ymin, xmax, ymax = bbox
    
    # 2. Scale independently
    scale_x = VIEWPORT_WIDTH / 1000.0
    scale_y = VIEWPORT_HEIGHT / 1000.0
    
    return {
        "x": round(((xmin + xmax) / 2) * scale_x, 1),
        "y": round(((ymin + ymax) / 2) * scale_y, 1),
    }


def _make_step_entry(
    step: int,
    screenshot: str,
    reasoning: str,
    action: dict,
    center: dict | None,
    success: bool,
) -> dict:
    return {
        "step": step,
        "screenshot": screenshot,
        "reasoning": reasoning,
        "action": action,
        "calculated_center": center,
        "success": success,
    }


def _print_step(step: int, action: dict, center: dict) -> None:
    act = action.get("action", "?")
    reasoning = action.get("reasoning", "")
    bbox = action.get("bounding_box", [])
    confidence = action.get("confidence", "?")
    value = action.get("value", "")

    print(f"\n{'='*60}")
    print(f"  Step {step}")
    print(f"  Action      : {act}")
    print(f"  BBox        : {bbox}")
    print(f"  Center (px) : ({center['x']}, {center['y']})")
    if value:
        print(f"  Value       : {value}")
    print(f"  Confidence  : {confidence}")
    print(f"  Reasoning   : {reasoning}")
    print(f"{'='*60}")


def _save_debug_screenshot(
    source_path: str,
    output_path: str,
    bbox: list[int],
    center: dict[str, float],
    action: dict,
) -> None:
    """Draw a debug overlay on the screenshot showing the bounding box and click target."""
    try:
        img = Image.open(source_path).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        w, h = img.size
        
        # Unpack as X, Y, X, Y
        xmin, ymin, xmax, ymax = bbox

        # Independent scaling
        scale_x = w / 1000.0
        scale_y = h / 1000.0
        
        px_left = xmin * scale_x
        px_top = ymin * scale_y
        px_right = xmax * scale_x
        px_bottom = ymax * scale_y

        cx = center["x"]
        cy = center["y"]

        # ---- Bounding box rectangle (red, semi-transparent fill) ----
        draw.rectangle(
            [px_left, px_top, px_right, px_bottom],
            outline=(255, 0, 0, 220),
            fill=(255, 0, 0, 40),
            width=3,
        )

        # ---- Crosshair at center (bright green) ----
        cross_len = 20
        line_w = 3
        green = (0, 255, 0, 255)
        # Horizontal line
        draw.line([(cx - cross_len, cy), (cx + cross_len, cy)], fill=green, width=line_w)
        # Vertical line
        draw.line([(cx, cy - cross_len), (cx, cy + cross_len)], fill=green, width=line_w)
        # Circle around center
        r = 12
        draw.ellipse(
            [cx - r, cy - r, cx + r, cy + r],
            outline=green,
            width=line_w,
        )

        # ---- Text label ----
        act = action.get("action", "?")
        label = f"{act} @ ({cx:.0f}, {cy:.0f})"
        # Position label above the bounding box, or at top if no room
        label_y = max(4, px_top - 22)
        label_x = max(4, px_left)

        # Draw text with a dark shadow for readability
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]:
            color = (0, 0, 0, 200) if (dx or dy) else (255, 255, 255, 255)
            draw.text((label_x + dx, label_y + dy), label, fill=color)

        # Composite and save
        result = Image.alpha_composite(img, overlay)
        result.convert("RGB").save(output_path, "PNG")
        log.info("Debug screenshot saved: %s", output_path)

    except Exception as exc:
        log.warning("Failed to save debug screenshot: %s", exc)
