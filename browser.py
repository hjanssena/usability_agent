"""
Playwright browser controller.

Manages a Chromium instance, captures screenshots as base64 PNG,
and translates normalised-bounding-box actions into physical
mouse / keyboard events.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

from playwright.async_api import async_playwright, Browser, BrowserContext, Page

from config import (
    FLUTTER_URL,
    HEADLESS,
    VIEWPORT_WIDTH,
    VIEWPORT_HEIGHT,
    ACTION_DELAY_MS,
)

log = logging.getLogger(__name__)


class BrowserController:
    """Async wrapper around a single Playwright Chromium page."""

    def __init__(self) -> None:
        self._pw = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def launch(self) -> None:
        """Start the browser and navigate to the Flutter app."""
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(headless=HEADLESS)
        self._context = await self._browser.new_context(
            viewport={"width": VIEWPORT_WIDTH, "height": VIEWPORT_HEIGHT},
        )
        self._page = await self._context.new_page()
        log.info("Navigating to %s", FLUTTER_URL)
        await self._page.goto(FLUTTER_URL, wait_until="networkidle")

    async def close(self) -> None:
        """Tear down browser resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()

    # ------------------------------------------------------------------
    # Screenshot
    # ------------------------------------------------------------------

    async def take_screenshot(self) -> str:
        """Capture the current viewport and return a base64-encoded PNG string."""
        assert self._page is not None, "Browser not launched"
        raw_bytes: bytes = await self._page.screenshot(type="png")
        return base64.b64encode(raw_bytes).decode("ascii")

    async def save_screenshot(self, path: str) -> None:
        """Save the current viewport to a file on disk."""
        assert self._page is not None
        await self._page.screenshot(path=path, type="png")

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    async def execute_action(self, action: dict[str, Any]) -> bool:
        """Translate a normalised-bounding-box action into Playwright calls.

        Returns ``True`` if the loop should continue, ``False`` if the agent
        signalled completion or is stuck.
        """
        assert self._page is not None, "Browser not launched"
        act = action.get("action", "")

        if act in ("done", "stuck"):
            return False

        # ---- Coordinate math ----
        bbox = action.get("bounding_box", [0, 0, 1000, 1000])
        center_x, center_y = self._bbox_to_pixel(bbox)

        try:
            if act == "click":
                log.info("click @ (%.0f, %.0f)", center_x, center_y)
                await self._page.mouse.click(center_x, center_y)

            elif act == "type":
                text = action.get("value", "")
                log.info("type '%s' @ (%.0f, %.0f)", text, center_x, center_y)
                # Click to focus, select-all to replace existing text, then type
                await self._page.mouse.click(center_x, center_y)
                await self._page.keyboard.press("Control+a")
                await self._page.keyboard.type(text)

            elif act == "press_key":
                key = action.get("value", "Enter")
                log.info("press_key '%s'", key)
                await self._page.keyboard.press(key)

            elif act == "scroll_down":
                log.info("scroll_down 500 @ (%.0f, %.0f)", center_x, center_y)
                await self._page.mouse.move(center_x, center_y)
                await self._page.mouse.wheel(0, 500)

            elif act == "scroll_up":
                log.info("scroll_up 500 @ (%.0f, %.0f)", center_x, center_y)
                await self._page.mouse.move(center_x, center_y)
                await self._page.mouse.wheel(0, -500)

            elif act == "scroll_right":
                log.info("scroll_right 500 @ (%.0f, %.0f)", center_x, center_y)
                await self._page.mouse.move(center_x, center_y)
                await self._page.mouse.wheel(500, 0)

            elif act == "scroll_left":
                log.info("scroll_left 500 @ (%.0f, %.0f)", center_x, center_y)
                await self._page.mouse.move(center_x, center_y)
                await self._page.mouse.wheel(-500, 0)

            elif act == "wait":
                log.info("wait (letting page load)")
                # A 3 to 4-second delay is usually enough for heavy pages like Amazon
                await self._page.wait_for_timeout(4000)

            else:
                log.warning("Unknown action '%s' — skipping", act)

        except Exception as exc:
            log.error("Action '%s' failed: %s", act, exc)
            # Treat as retriable — caller will count it as a step
            return True

        # Wait for UI to settle
        await self._page.wait_for_timeout(ACTION_DELAY_MS)
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bbox_to_pixel(self, bbox: list[int]) -> tuple[float, float]:
        """Convert a normalised [xmin, ymin, xmax, ymax] box to pixel-space center.
        
        Qwen-VL scales X and Y independently from 0 to 1000.
        """
        xmin, ymin, xmax, ymax = bbox

        width = VIEWPORT_WIDTH
        height = VIEWPORT_HEIGHT
        
        # Scale independently!
        scale_x = width / 1000.0
        scale_y = height / 1000.0

        center_x = ((xmin + xmax) / 2) * scale_x
        center_y = ((ymin + ymax) / 2) * scale_y

        # Clamp
        center_x = max(0, min(center_x, width - 1))
        center_y = max(0, min(center_y, height - 1))

        return center_x, center_y
