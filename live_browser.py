from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


@dataclass
class LiveBrowserConfig:
    screenshots_dir: str = "screenshots_live"
    headless: bool = False  # for demos
    slow_mo_ms: int = 250   # makes the run visible
    viewport: Tuple[int, int] = (1280, 720)
    navigation_timeout_ms: int = 60000
    action_timeout_ms: int = 20000


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _screenshot_path(dir_path: str, label: str) -> str:
    ts = int(time.time() * 1000)
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in label.strip().lower())
    safe = "-".join([p for p in safe.split("-") if p]) or "step"
    return os.path.join(dir_path, f"{ts}_{safe}.png")


class LiveBrowserSession:
    """
    Keeps a single Playwright browser/page alive across multiple tasks.
    Call .close() when the user is done (so the window closes).
    """

    def __init__(self, config: Optional[LiveBrowserConfig] = None):
        self.config = config or LiveBrowserConfig()
        _ensure_dir(self.config.screenshots_dir)

        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(headless=self.config.headless, slow_mo=self.config.slow_mo_ms)
        self._context = self._browser.new_context(
            viewport={"width": self.config.viewport[0], "height": self.config.viewport[1]}
        )
        self.page = self._context.new_page()
        self.page.set_default_navigation_timeout(self.config.navigation_timeout_ms)
        self.page.set_default_timeout(self.config.action_timeout_ms)

    def screenshot(self, label: str, *, full_page: bool = True) -> str:
        path = _screenshot_path(self.config.screenshots_dir, label)
        self.page.screenshot(path=path, full_page=full_page)
        return path

    def close(self) -> None:
        try:
            self._context.close()
        finally:
            try:
                self._browser.close()
            finally:
                self._pw.stop()


def run_live_plan(
    *,
    url: str,
    plan: List[Dict[str, Any]],
    config: Optional[LiveBrowserConfig] = None,
    session: Optional[LiveBrowserSession] = None,
) -> Dict[str, Any]:
    """
    Execute a constrained action plan on a website and capture screenshots.

    Supported actions:
      - {"type":"goto","url":"https://..."}
      - {"type":"click","text":"Button text"}  (uses best-effort text locator)
      - {"type":"fill","label":"Search","value":"foo"}  (tries label/placeholder/name)
      - {"type":"press","key":"Enter"}
      - {"type":"wait","ms":1500}
      - {"type":"screenshot","label":"after-search"}
    """
    config = config or LiveBrowserConfig()
    _ensure_dir(config.screenshots_dir)

    screenshots: List[str] = []
    steps: List[Dict[str, Any]] = []
    error: Optional[str] = None

    owned_session = False
    if session is None:
        session = LiveBrowserSession(config=config)
        owned_session = True
    page = session.page

    def snap(label: str) -> None:
        screenshots.append(session.screenshot(label))

    try:
        page.goto(url, wait_until="domcontentloaded")
        snap("start")

        for i, action in enumerate(plan):
            a_type = str(action.get("type") or "").strip().lower()
            step_rec: Dict[str, Any] = {"index": i, "action": action}

            try:
                if a_type == "goto":
                    tgt = str(action.get("url") or url)
                    page.goto(tgt, wait_until="domcontentloaded")
                elif a_type == "click":
                    text = str(action.get("text") or "").strip()
                    if not text:
                        raise ValueError("click.text is required")
                    # Try role-based first, then plain text.
                    loc = page.get_by_role("button", name=text)
                    if loc.count() == 0:
                        loc = page.get_by_text(text, exact=False)
                    loc.first.click()
                elif a_type == "fill":
                    label = str(action.get("label") or "").strip()
                    value = str(action.get("value") or "")
                    if not label:
                        raise ValueError("fill.label is required")
                    # Try label, placeholder, then name attribute.
                    loc = page.get_by_label(label)
                    if loc.count() == 0:
                        loc = page.get_by_placeholder(label)
                    if loc.count() == 0:
                        loc = page.locator(f'input[name="{label}"], textarea[name="{label}"]')
                    loc.first.fill(value)
                elif a_type == "press":
                    key = str(action.get("key") or "Enter")
                    page.keyboard.press(key)
                elif a_type == "wait":
                    ms = int(action.get("ms") or 1000)
                    page.wait_for_timeout(ms)
                elif a_type == "screenshot":
                    label = str(action.get("label") or f"step-{i}")
                    snap(label)
                else:
                    raise ValueError(f"Unsupported action type: {a_type}")

                # Default “proof” screenshot after each step, unless it was already a screenshot step.
                if a_type != "screenshot":
                    snap(f"after-{a_type}-{i}")

                step_rec["ok"] = True
            except (PlaywrightTimeoutError, Exception) as exc:
                step_rec["ok"] = False
                step_rec["error"] = f"{type(exc).__name__}: {exc}"
                steps.append(step_rec)
                raise

            steps.append(step_rec)

    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if owned_session and session is not None:
            session.close()

    return {
        "url": url,
        "error": error,
        "screenshots": screenshots,
        "steps": steps,
    }

