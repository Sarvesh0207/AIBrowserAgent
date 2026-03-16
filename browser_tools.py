"""
browser_tools.py
~~~~~~~~~~~~~~~~
Playwright browser controller with Set-of-Mark (SoM) screenshot capability.

SoM = every interactive element gets a numbered bounding box drawn on the
screenshot (inspired by WebVoyager / SoM paper). The agent sees element IDs
and can act on them by number instead of fragile CSS selectors.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Browser, BrowserContext, Page, Playwright, sync_playwright

# ── constants ──────────────────────────────────────────────────────────────────

SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

# Colours for different element types (RGB)
ELEMENT_COLOURS: dict[str, tuple[int, int, int]] = {
    "link":    (59,  130, 246),   # blue
    "button":  (16,  185, 129),   # green
    "input":   (245, 158,  11),   # amber
    "select":  (168,  85, 247),   # purple  ← dropdowns
    "textarea":(239,  68,  68),   # red
    "other":   (107, 114, 128),   # grey
}

# CSS selector to pull all interactive elements
INTERACTIVE_SELECTOR = (
    "a[href], button, input:not([type='hidden']), select, textarea, "
    "[role='button'], [role='link'], [role='menuitem'], [role='tab'], "
    "[role='checkbox'], [role='radio'], [role='combobox'], [tabindex]:not([tabindex='-1'])"
)


# ── data classes ───────────────────────────────────────────────────────────────

@dataclass
class ElementInfo:
    index: int
    tag: str
    element_type: str        # link | button | input | select | textarea | other
    text: str
    placeholder: str
    name: str
    href: str
    value: str
    bbox: dict[str, float]   # x, y, width, height  (page coordinates)
    is_dropdown: bool = False
    dropdown_options: list[str] = field(default_factory=list)


@dataclass
class PageState:
    url: str
    title: str
    screenshot_clean: Path     # no bounding boxes
    screenshot_som: Path       # with SoM bounding boxes
    elements: list[ElementInfo]
    page_text: str             # visible text (trimmed)


# ── helpers ────────────────────────────────────────────────────────────────────

def _element_type(tag: str, role: str, input_type: str) -> str:
    tag = tag.lower()
    if tag == "a":
        return "link"
    if tag in ("button",) or role in ("button",):
        return "button"
    if tag == "select" or role in ("combobox", "listbox"):
        return "select"
    if tag == "textarea":
        return "textarea"
    if tag == "input":
        if input_type in ("submit", "button", "reset", "image"):
            return "button"
        return "input"
    if role in ("link",):
        return "link"
    if role in ("menuitem", "tab", "checkbox", "radio"):
        return "button"
    return "other"


def _stamp(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}"


def _load_font(size: int = 12) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Try to load a real TTF font; fall back to PIL default."""
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                pass
    return ImageFont.load_default()


# ── core class ─────────────────────────────────────────────────────────────────

class BrowserController:
    """
    Manages a single Playwright browser session.
    Call `open()` before use and `close()` when done.
    """

    def __init__(self, headless: bool = False):
        self.headless = headless
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self.page: Page | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def open(self) -> None:
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            args=["--start-maximized"],
        )
        self._context = self._browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        )
        self.page = self._context.new_page()

    def close(self) -> None:
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    # ── navigation ────────────────────────────────────────────────────────────

    def navigate(self, url: str) -> PageState:
        """Go to URL and return a full PageState with SoM screenshot."""
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self.page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        self.page.wait_for_timeout(1500)   # let JS settle
        return self.capture_state()

    # ── page state ────────────────────────────────────────────────────────────

    def capture_state(self) -> PageState:
        """Screenshot + element extraction + SoM overlay."""
        stamp = _stamp("page")

        # 1. Clean screenshot
        clean_path = SCREENSHOT_DIR / f"{stamp}_clean.png"
        self.page.screenshot(path=str(clean_path), full_page=False)

        # 2. Extract interactive elements
        elements = self._extract_elements()

        # 3. SoM overlay
        som_path = SCREENSHOT_DIR / f"{stamp}_som.png"
        self._draw_som(clean_path, som_path, elements)

        # 4. Visible text
        page_text = self.page.evaluate(
            "() => document.body ? document.body.innerText.slice(0, 4000) : ''"
        )

        return PageState(
            url=self.page.url,
            title=self.page.title(),
            screenshot_clean=clean_path,
            screenshot_som=som_path,
            elements=elements,
            page_text=page_text.strip(),
        )

    # ── element extraction ────────────────────────────────────────────────────

    def _extract_elements(self) -> list[ElementInfo]:
        raw: list[dict[str, Any]] = self.page.evaluate(
            """() => {
                const sel = `a[href], button, input:not([type='hidden']), select,
                    textarea, [role='button'], [role='link'], [role='menuitem'],
                    [role='tab'], [role='checkbox'], [role='radio'],
                    [role='combobox'], [tabindex]:not([tabindex='-1'])`;
                const nodes = Array.from(document.querySelectorAll(sel));
                const vw = window.innerWidth, vh = window.innerHeight;
                const results = [];
                nodes.forEach((el, i) => {
                    const r = el.getBoundingClientRect();
                    if (r.width < 2 || r.height < 2) return;
                    if (r.bottom < 0 || r.top > vh) return;  // off-screen
                    if (r.right < 0 || r.left > vw) return;
                    const style = window.getComputedStyle(el);
                    if (style.visibility === 'hidden' || style.display === 'none') return;
                    const opts = el.tagName === 'SELECT'
                        ? Array.from(el.options).map(o => o.text.trim())
                        : [];
                    results.push({
                        tag: el.tagName.toLowerCase(),
                        role: el.getAttribute('role') || '',
                        inputType: el.type || '',
                        text: (el.innerText || el.value || el.getAttribute('aria-label') || '').trim().slice(0, 120),
                        placeholder: el.placeholder || '',
                        name: el.name || el.id || '',
                        href: el.href || '',
                        value: el.value || '',
                        bbox: { x: r.left, y: r.top, w: r.width, h: r.height },
                        dropdownOptions: opts,
                    });
                });
                return results;
            }"""
        )

        elements: list[ElementInfo] = []
        for idx, r in enumerate(raw, start=1):
            etype = _element_type(r["tag"], r["role"], r["inputType"])
            elements.append(ElementInfo(
                index=idx,
                tag=r["tag"],
                element_type=etype,
                text=r["text"],
                placeholder=r["placeholder"],
                name=r["name"],
                href=r["href"],
                value=r["value"],
                bbox=r["bbox"],
                is_dropdown=(r["tag"] == "select" or r["role"] in ("combobox", "listbox")),
                dropdown_options=r["dropdownOptions"],
            ))
        return elements

    # ── SoM drawing ───────────────────────────────────────────────────────────

    def _draw_som(
        self,
        source: Path,
        dest: Path,
        elements: list[ElementInfo],
    ) -> None:
        img = Image.open(source).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        label_font = _load_font(12)
        device_pixel_ratio: float = self.page.evaluate("() => window.devicePixelRatio || 1")
        # PIL coords are in CSS pixels (1:1 with viewport for non-retina)
        # For retina displays the PNG is 2× so we scale.
        img_w, img_h = img.size
        vp_w: int = self.page.evaluate("() => window.innerWidth")
        scale = img_w / vp_w

        for el in elements:
            x = el.bbox["x"] * scale
            y = el.bbox["y"] * scale
            w = el.bbox["w"] * scale
            h = el.bbox["h"] * scale
            colour = ELEMENT_COLOURS.get(el.element_type, ELEMENT_COLOURS["other"])
            r, g, b = colour

            # Semi-transparent fill
            draw.rectangle([x, y, x + w, y + h], fill=(r, g, b, 40))
            # Border
            draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 220), width=2)

            # Label badge (top-left corner)
            label = str(el.index)
            # Estimate text size
            try:
                bbox_txt = draw.textbbox((0, 0), label, font=label_font)
                tw = bbox_txt[2] - bbox_txt[0]
                th = bbox_txt[3] - bbox_txt[1]
            except AttributeError:
                tw, th = draw.textsize(label, font=label_font)  # type: ignore[attr-defined]

            pad = 3
            lx, ly = x, max(0, y - th - pad * 2)
            draw.rectangle(
                [lx, ly, lx + tw + pad * 2, ly + th + pad * 2],
                fill=(r, g, b, 230),
            )
            draw.text((lx + pad, ly + pad), label, fill=(255, 255, 255, 255), font=label_font)

        composite = Image.alpha_composite(img, overlay).convert("RGB")
        composite.save(dest)

    # ── actions ───────────────────────────────────────────────────────────────

    def click_element(self, index: int, elements: list[ElementInfo]) -> str:
        el = self._find(index, elements)
        if el is None:
            return f"❌ No element #{index}"
        cx = el.bbox["x"] + el.bbox["w"] / 2
        cy = el.bbox["y"] + el.bbox["h"] / 2
        self.page.mouse.click(cx, cy)
        self.page.wait_for_timeout(1500)
        return f"✅ Clicked element #{index} ({el.element_type}: {el.text[:60]})"

    def type_into(self, index: int, text: str, elements: list[ElementInfo]) -> str:
        el = self._find(index, elements)
        if el is None:
            return f"❌ No element #{index}"
        cx = el.bbox["x"] + el.bbox["w"] / 2
        cy = el.bbox["y"] + el.bbox["h"] / 2
        self.page.mouse.click(cx, cy)
        self.page.wait_for_timeout(300)
        self.page.keyboard.press("Control+a")
        self.page.keyboard.type(text)
        return f"✅ Typed into element #{index} ({el.text[:40] or el.placeholder[:40]})"

    def press_key(self, key: str) -> str:
        self.page.keyboard.press(key)
        self.page.wait_for_timeout(1000)
        return f"✅ Pressed key: {key}"

    def select_option(self, index: int, option_text: str, elements: list[ElementInfo]) -> str:
        el = self._find(index, elements)
        if el is None:
            return f"❌ No element #{index}"
        cx = el.bbox["x"] + el.bbox["w"] / 2
        cy = el.bbox["y"] + el.bbox["h"] / 2
        # Use Playwright's select_option for <select> elements
        handle = self.page.evaluate_handle(
            """([x, y]) => {
                const el = document.elementFromPoint(x, y);
                return el;
            }""",
            [cx, cy],
        )
        try:
            self.page.locator("select").nth(
                self._get_select_index(el, elements)
            ).select_option(label=option_text)
        except Exception:
            # Fallback: click the element then click matching option
            self.page.mouse.click(cx, cy)
            self.page.wait_for_timeout(400)
            try:
                self.page.get_by_text(option_text, exact=False).first.click()
            except Exception as exc:
                return f"❌ Could not select '{option_text}': {exc}"
        self.page.wait_for_timeout(800)
        return f"✅ Selected '{option_text}' in dropdown #{index}"

    def scroll(self, direction: str = "down", amount: int = 400) -> str:
        delta = amount if direction == "down" else -amount
        self.page.mouse.wheel(0, delta)
        self.page.wait_for_timeout(600)
        return f"✅ Scrolled {direction} {amount}px"

    def go_back(self) -> str:
        self.page.go_back(wait_until="domcontentloaded", timeout=15_000)
        self.page.wait_for_timeout(1000)
        return "✅ Navigated back"

    def get_screenshot_b64(self, path: Path) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # ── utils ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _find(index: int, elements: list[ElementInfo]) -> ElementInfo | None:
        for el in elements:
            if el.index == index:
                return el
        return None

    @staticmethod
    def _get_select_index(target: ElementInfo, elements: list[ElementInfo]) -> int:
        """Return the nth-of-type index of a <select> element."""
        selects = [e for e in elements if e.tag == "select"]
        for i, s in enumerate(selects):
            if s.index == target.index:
                return i
        return 0
