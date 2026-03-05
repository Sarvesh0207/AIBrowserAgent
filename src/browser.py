import re
from dataclasses import dataclass
from typing import Any, Optional

from playwright.async_api import Page, async_playwright

from .config import HEADLESS, BROWSER_TIMEOUT_MS, SCREENSHOT_DIR, ensure_dirs
from .extractors import extract_title_and_description
from .logger import utc_ts


@dataclass
class PageResult:
    url: str
    final_url: str
    title: str
    description: str
    screenshot_path: str
    annotated_screenshot_path: Optional[str] = None  # screenshot with bounding boxes drawn


@dataclass
class ActionConfirmation:
    """Result of a single 'click' or 'fill' action in headless mode."""
    action_type: str  # "click" or "fill"
    selector: str
    value: Optional[str] = None  # for fill only
    success: bool = False
    message: str = ""
    submitted: bool = False  # True if we pressed Enter after fill (e.g. to submit search)


async def get_interactive_elements(page: Page) -> list[dict[str, Any]]:
    """
    Find visible interactive elements (input, button, a, textarea, [role=button])
    and return a list of {selector, index, bbox, label, tag} for each.
    bbox is {x, y, width, height}; use selector + index with page.locator(selector).nth(index).
    """
    elements: list[dict[str, Any]] = []
    # Selectors that commonly represent clickable or fillable elements
    for selector in ("input:not([type='hidden'])", "button", "textarea", "a", "[role='button']", "[role='link']"):
        try:
            locs = await page.locator(selector).all()
            for i, loc in enumerate(locs):
                try:
                    if not await loc.is_visible():
                        continue
                    box = await loc.bounding_box()
                    if not box:
                        continue
                    label = (
                        await loc.get_attribute("placeholder")
                        or await loc.get_attribute("aria-label")
                        or await loc.get_attribute("title")
                    )
                    if not label:
                        label = (await loc.evaluate("el => (el.innerText || '').slice(0, 80)") or "").strip()
                    if not label:
                        label = (await loc.evaluate("el => (el.querySelector('img[alt]')?.alt || '').trim()") or "").strip()
                    if not label:
                        label = selector.split("[")[0] or "element"
                    label = (label or "element")[:80]
                    elements.append({
                        "selector": selector,
                        "index": i,
                        "bbox": box,
                        "label": label,
                        "tag": selector.split("[")[0].split(":")[0],
                    })
                except Exception:
                    continue
        except Exception:
            continue
    return elements


def draw_bboxes_on_screenshot(
    screenshot_path: str,
    elements: list[dict[str, Any]],
    output_path: str,
) -> None:
    """Draw bounding boxes and optional labels on a screenshot; save to output_path."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return
    try:
        img = Image.open(screenshot_path).convert("RGB")
    except Exception:
        return
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
    except Exception:
        font = ImageFont.load_default()
    for idx, el in enumerate(elements):
        bbox = el.get("bbox")
        if not bbox:
            continue
        x, y, w, h = bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)
        if w <= 0 or h <= 0:
            continue
        draw.rectangle([x, y, x + w, y + h], outline="red", width=2)
        label = f"{idx}: {el.get('label', '')}"[:40]
        draw.text((x, max(0, y - 18)), label, fill="red", font=font)
    img.save(output_path)


_HIGHLIGHT_OVERLAY_ID = "agent-highlight-overlay"


async def inject_highlight_on_selector(page: Page, selector: str) -> None:
    """Draw a red box overlay on the element matching selector (visible in headful)."""
    await page.evaluate(
        """(selector) => {
            const el = document.querySelector(selector);
            if (!el) return;
            const r = el.getBoundingClientRect();
            const d = document.createElement('div');
            d.id = 'agent-highlight-overlay';
            d.style.cssText = `position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;border:3px solid red;pointer-events:none;z-index:2147483647;box-sizing:border-box;`;
            document.body.appendChild(d);
        }""",
        selector,
    )


async def remove_highlight(page: Page) -> None:
    """Remove the red box overlay from the page."""
    await page.evaluate(
        """() => {
            const d = document.getElementById('agent-highlight-overlay');
            if (d) d.remove();
        }"""
    )


async def inject_highlight_on_nth(page: Page, selector: str, index: int) -> None:
    """Draw a red box on the nth element matching selector (0-based). Used for click feedback."""
    await page.evaluate(
        """([selector, index]) => {
            const el = document.querySelectorAll(selector)[index];
            if (!el) return;
            const r = el.getBoundingClientRect();
            const d = document.createElement('div');
            d.id = 'agent-highlight-overlay';
            d.style.cssText = `position:fixed;left:${r.left}px;top:${r.top}px;width:${r.width}px;height:${r.height}px;border:3px solid red;pointer-events:none;z-index:2147483647;box-sizing:border-box;`;
            document.body.appendChild(d);
        }""",
        [selector, index],
    )


async def inject_highlight_at_box(page: Page, box: dict) -> None:
    """Draw a red box at the given bounding box {x,y,width,height}. For any element (e.g. from locator.bounding_box())."""
    x, y = box.get("x", 0), box.get("y", 0)
    w, h = box.get("width", 0), box.get("height", 0)
    if w <= 0 or h <= 0:
        return
    await page.evaluate(
        """([x, y, w, h]) => {
            const d = document.createElement('div');
            d.id = 'agent-highlight-overlay';
            d.style.cssText = `position:fixed;left:${x}px;top:${y}px;width:${w}px;height:${h}px;border:3px solid red;pointer-events:none;z-index:2147483647;box-sizing:border-box;`;
            document.body.appendChild(d);
        }""",
        [x, y, w, h],
    )


def _safe_slug(url: str) -> str:
    # simple slug for filenames
    return "".join([c if c.isalnum() else "-" for c in url.lower()])[:80].strip("-") or "site"


async def fetch_page(url: str) -> PageResult:
    ensure_dirs()
    slug = _safe_slug(url)
    screenshot_path = str(SCREENSHOT_DIR / f"{slug}_{utc_ts()}.png")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        page.set_default_timeout(BROWSER_TIMEOUT_MS)

        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(800)

            title, desc = await extract_title_and_description(page)
            final_url = page.url

            await page.screenshot(path=screenshot_path, full_page=True)

            return PageResult(
                url=url,
                final_url=final_url,
                title=title,
                description=desc,
                screenshot_path=screenshot_path,
            )
        finally:
            await page.close()
            await context.close()
            await browser.close()


async def fetch_page_with_action(
    url: str,
    click_selector: Optional[str] = None,
    fill_selector: Optional[str] = None,
    fill_value: Optional[str] = None,
    submit_after_fill: bool = False,
    open_search_selector: Optional[str] = None,
    headless: Optional[bool] = None,
) -> tuple[PageResult, Optional[ActionConfirmation]]:
    """
    Load a page, then perform one action (click or fill) and confirm.
    If open_search_selector is set, click it first to reveal the search bar, then fill.
    If submit_after_fill is True, after filling we press Enter to submit (e.g. search).
    headless: if None, use config HEADLESS; if False, run with visible browser (headful).
    Returns (PageResult, ActionConfirmation). PageResult.annotated_screenshot_path has bbox overlay when available.
    """
    ensure_dirs()
    slug = _safe_slug(url)
    action_suffix = "click" if click_selector else "fill_submit" if (fill_selector and submit_after_fill) else "fill" if fill_selector else "none"
    if open_search_selector:
        action_suffix = "open_search_" + action_suffix
    screenshot_path = str(SCREENSHOT_DIR / f"{slug}_action_{action_suffix}_{utc_ts()}.png")
    annotated_path = screenshot_path.replace(".png", "_annotated.png")

    confirmation: Optional[ActionConfirmation] = None
    use_headless = HEADLESS if headless is None else headless

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=use_headless)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        page.set_default_timeout(BROWSER_TIMEOUT_MS)

        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(800)

            if open_search_selector:
                try:
                    await page.click(open_search_selector, timeout=5000)
                    await page.wait_for_timeout(600)
                except Exception:
                    pass

            if click_selector:
                try:
                    await page.click(click_selector, timeout=5000)
                    await page.wait_for_timeout(500)
                    confirmation = ActionConfirmation(
                        action_type="click",
                        selector=click_selector,
                        success=True,
                        message=f"Clicked: {click_selector}",
                    )
                except Exception as e:
                    confirmation = ActionConfirmation(
                        action_type="click",
                        selector=click_selector,
                        success=False,
                        message=f"Click failed: {e!r}",
                    )
            elif fill_selector and fill_value is not None:
                try:
                    await page.fill(fill_selector, fill_value, timeout=5000)
                    await page.wait_for_timeout(300)
                    if submit_after_fill:
                        try:
                            async with page.expect_navigation(wait_until="domcontentloaded", timeout=10000):
                                await page.keyboard.press("Enter")
                        except Exception:
                            pass
                        await page.wait_for_timeout(1500)
                        try:
                            await page.wait_for_load_state("domcontentloaded", timeout=5000)
                        except Exception:
                            pass
                    else:
                        await page.wait_for_timeout(200)
                    confirmation = ActionConfirmation(
                        action_type="fill",
                        selector=fill_selector,
                        value=fill_value,
                        success=True,
                        message=f"Filled {fill_selector!r} with {fill_value!r}" + (" and pressed Enter" if submit_after_fill else ""),
                        submitted=submit_after_fill,
                    )
                except Exception as e:
                    confirmation = ActionConfirmation(
                        action_type="fill",
                        selector=fill_selector,
                        value=fill_value,
                        success=False,
                        message=f"Fill failed: {e!r}",
                        submitted=False,
                    )

            try:
                title, desc = await extract_title_and_description(page)
            except Exception:
                title, desc = "", ""
            final_url = page.url
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
            except Exception:
                pass

            annotated_screenshot_path: Optional[str] = None
            try:
                elements = await get_interactive_elements(page)
                if elements:
                    draw_bboxes_on_screenshot(screenshot_path, elements, annotated_path)
                    annotated_screenshot_path = annotated_path
            except Exception:
                pass

            return (
                PageResult(
                    url=url,
                    final_url=final_url,
                    title=title,
                    description=desc,
                    screenshot_path=screenshot_path,
                    annotated_screenshot_path=annotated_screenshot_path,
                ),
                confirmation,
            )
        finally:
            await page.close()
            await context.close()
            await browser.close()


async def run_nl_search(
    url: str,
    fill_selector: str,
    fill_value: str,
    submit: bool = True,
) -> PageResult:
    """
    Headful only: open url, show red box on fill_selector, wait, type fill_value, submit (Enter), remove box, screenshot.
    """
    ensure_dirs()
    slug = _safe_slug(url)
    screenshot_path = str(SCREENSHOT_DIR / f"{slug}_nl_{utc_ts()}.png")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        page.set_default_timeout(BROWSER_TIMEOUT_MS)

        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(1200)

            try:
                await inject_highlight_on_selector(page, fill_selector)
                await page.wait_for_timeout(2000)
            except Exception:
                pass

            await page.fill(fill_selector, fill_value, timeout=8000)
            await page.wait_for_timeout(300)
            if submit:
                try:
                    async with page.expect_navigation(wait_until="domcontentloaded", timeout=10000):
                        await page.keyboard.press("Enter")
                except Exception:
                    pass
                await page.wait_for_timeout(1500)

            await remove_highlight(page)
            await page.wait_for_timeout(500)

            try:
                title, desc = await extract_title_and_description(page)
            except Exception:
                title, desc = "", ""
            final_url = page.url
            try:
                await page.screenshot(path=screenshot_path, full_page=True)
            except Exception:
                pass

            return PageResult(
                url=url,
                final_url=final_url,
                title=title,
                description=desc,
                screenshot_path=screenshot_path,
                annotated_screenshot_path=None,
            )
        finally:
            await page.close()
            await context.close()
            await browser.close()


# ---- Chat loop helpers: work on an existing page ----

# Common selectors for "search box" on a page (try in order)
SEARCH_INPUT_SELECTORS = [
    "textarea[name=q]",
    "input[name=q]",
    "input[name=query]",
    "input[type=search]",
    "input[placeholder*='Search' i]",
    "input[placeholder*='cccis' i]",
    "input[placeholder*='Search' i]",
    "input[aria-label*='Search' i]",
    "input[aria-label*='Search' i]",
]


async def find_search_selector(page: Page) -> Optional[str]:
    """Return the first visible search-like input selector that exists on the page."""
    for sel in SEARCH_INPUT_SELECTORS:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                return sel
        except Exception:
            continue
    return None


async def take_screenshot(page: Page, path: str) -> None:
    ensure_dirs()
    try:
        await page.screenshot(path=path, full_page=True)
    except Exception:
        pass


async def chat_navigate(page: Page, url: str, step_prefix: str) -> str:
    """Goto url, wait, take screenshot. Returns path to screenshot."""
    await page.goto(url, wait_until="domcontentloaded")
    await page.wait_for_timeout(1200)
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path)
    return path


async def chat_go_back(page: Page, step_prefix: str) -> str:
    """Go back to previous page. First dismiss any overlay (e.g. search box) with Escape, then browser back."""
    for _ in range(2):
        try:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(250)
        except Exception:
            pass
    try:
        await page.go_back(wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
    except Exception:
        await page.wait_for_timeout(500)
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path)
    return path


async def chat_close_search(page: Page, step_prefix: str) -> str:
    """Close/dismiss the search box overlay (Escape, then screenshot). Use when user says close search."""
    for _ in range(2):
        try:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(300)
        except Exception:
            pass
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path)
    return path


async def _click_search_trigger(page: Page) -> bool:
    """If search box is hidden, click search icon/button to reveal it. Returns True if we clicked something."""
    # Try many patterns (CCC and other sites use different markup for the search icon)
    triggers = [
        page.get_by_role("button", name=re.compile(r"search", re.I)),
        page.get_by_role("link", name=re.compile(r"search", re.I)),
        page.locator("[aria-label*='Search']").first,
        page.locator("[aria-label*='search']").first,
        page.locator("[title*='Search' i]").first,
        page.locator("button[type='submit']").filter(has=page.locator("svg")).first,
        page.locator("[class*='search-icon']").first,
        page.locator("[class*='searchIcon']").first,
        page.locator("[class*='search_icon']").first,
        page.locator("header [class*='search']").first,
        page.locator("header [class*='Search']").first,
        page.locator("nav [class*='search']").first,
        page.locator("header button").first,
        page.locator("header a[href*='search']").first,
        page.locator("nav button").first,
        page.locator("[data-testid*='search']").first,
        page.locator("[class*='search'] button").first,
        page.locator("[class*='Search'] button").first,
    ]
    for trigger in triggers:
        try:
            if await trigger.count() > 0 and await trigger.is_visible():
                await trigger.scroll_into_view_if_needed(timeout=2000)
                await trigger.click(timeout=3000)
                await page.wait_for_timeout(1500)
                return True
        except Exception:
            continue
    # Fallback: click first interactive element whose label looks like search (e.g. "Q" or icon-only)
    try:
        elements = await get_interactive_elements(page)
        search_labels = ("search", "q")
        for el in elements:
            label = (el.get("label") or "").strip().lower()
            if not label or len(label) > 20:
                continue
            if any(s in label for s in search_labels) or label == "q":
                sel, idx = el.get("selector"), el.get("index", 0)
                if sel is None:
                    continue
                loc = page.locator(sel).nth(idx)
                if await loc.count() > 0 and await loc.is_visible():
                    await loc.scroll_into_view_if_needed(timeout=2000)
                    await loc.click(timeout=3000)
                    await page.wait_for_timeout(1500)
                    return True
    except Exception:
        pass
    # Last resort (CCC etc.): header/nav link or button with aria-label or title containing "search"
    try:
        for selector in ("header a", "header button", "nav a", "nav button", "[role='banner'] a", "[role='banner'] button"):
            locs = await page.locator(selector).all()
            for loc in locs:
                try:
                    if not await loc.is_visible():
                        continue
                    aria = (await loc.get_attribute("aria-label") or "").strip().lower()
                    title = (await loc.get_attribute("title") or "").strip().lower()
                    if ("search" in aria or "search" in title):
                        await loc.scroll_into_view_if_needed(timeout=2000)
                        await loc.click(timeout=3000)
                        await page.wait_for_timeout(1500)
                        return True
                except Exception:
                    continue
    except Exception:
        pass
    # JS fallback: click first header/nav a or button whose class/id/aria contains "search"
    try:
        clicked = await page.evaluate("""() => {
            const root = document.querySelector('header') || document.querySelector('nav') || document.querySelector('[role="banner"]');
            if (!root) return false;
            const nodes = root.querySelectorAll('a, button, [role="button"]');
            for (const el of nodes) {
                if (!el.offsetParent) continue;
                const c = ((el.className || '') + ' ' + (el.id || '')).toLowerCase();
                const aria = (el.getAttribute('aria-label') || '').toLowerCase();
                const title = (el.getAttribute('title') || '').toLowerCase();
                if (/search/.test(c + ' ' + aria + ' ' + title)) {
                    el.click();
                    return true;
                }
            }
            for (const el of nodes) {
                if (!el.offsetParent) continue;
                const text = (el.innerText || '').trim();
                if (text.length <= 2 && el.querySelector('svg')) {
                    el.click();
                    return true;
                }
            }
            return false;
        }""")
        if clicked:
            await page.wait_for_timeout(1500)
            return True
    except Exception:
        pass
    try:
        header_links = await page.locator("header a, nav a, [role='banner'] a").all()
        for loc in reversed(header_links):
            try:
                if not await loc.is_visible():
                    continue
                text = (await loc.inner_text() or "").strip()
                if len(text) <= 2:
                    await loc.scroll_into_view_if_needed(timeout=2000)
                    await loc.click(timeout=3000)
                    await page.wait_for_timeout(1500)
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


async def chat_search_on_page(
    page: Page,
    query: str,
    step_prefix: str,
) -> tuple[Optional[str], str]:
    """
    Find search box; if none visible, click search icon first. Then show red box, screenshot, fill, submit.
    Returns (redbox_screenshot_path, done_screenshot_path).
    """
    ensure_dirs()
    sel = await find_search_selector(page)
    if not sel:
        # No visible search input: click search icon to open the field, then wait for input
        search_input_selector = (
            "input[type=search], input[name=query], input[type=text][placeholder*='Search' i], "
            "input[placeholder*='Search' i], input[placeholder*='cccis'], input[placeholder*='search' i], "
            "textarea[placeholder*='Search' i], input[aria-label*='Search' i]"
        )
        for attempt in range(3):
            if await _click_search_trigger(page):
                await page.wait_for_timeout(1800)
                for _ in range(6):
                    try:
                        await page.wait_for_selector(search_input_selector, timeout=2500)
                        break
                    except Exception:
                        await page.wait_for_timeout(500)
                await page.wait_for_timeout(800)
                sel = await find_search_selector(page)
                if not sel:
                    for _ in range(3):
                        await page.wait_for_timeout(400)
                        sel = await find_search_selector(page)
                        if sel:
                            break
                if sel:
                    break
            await page.wait_for_timeout(600)
    if not sel:
        path_done = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
        await take_screenshot(page, path_done)
        return None, path_done

    path_redbox = str(SCREENSHOT_DIR / f"{step_prefix}_redbox_{utc_ts()}.png")
    try:
        await inject_highlight_on_selector(page, sel)
        await page.wait_for_timeout(800)
        await take_screenshot(page, path_redbox)
        await page.wait_for_timeout(1200)
    except Exception:
        path_redbox = None
    await remove_highlight(page)

    loc = page.locator(sel).first
    try:
        await loc.click(timeout=3000)
        await page.wait_for_timeout(200)
        await loc.click(click_count=3, timeout=2000)
        await page.wait_for_timeout(100)
        await page.keyboard.type(query, delay=30)
    except Exception:
        try:
            await loc.clear(timeout=3000)
            await page.wait_for_timeout(150)
            await loc.fill(query, timeout=8000)
        except Exception:
            try:
                await loc.focus()
                await page.keyboard.press("Control+a")
                await page.wait_for_timeout(100)
                await page.keyboard.type(query, delay=50)
            except Exception:
                await loc.fill(query, timeout=8000)
    await page.wait_for_timeout(200)
    try:
        async with page.expect_navigation(wait_until="domcontentloaded", timeout=10000):
            await page.keyboard.press("Enter")
    except Exception:
        pass
    await page.wait_for_timeout(1500)

    path_done = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path_done)
    return path_redbox, path_done


# Common selectors for modal/popup close buttons (try in order)
POPUP_CLOSE_SELECTORS = [
    'button[aria-label="Close"]',
    'button[aria-label="close"]',
    '[aria-label="Close"]',
    '[aria-label="close"]',
    'button[class*="close"]',
    '[class*="close-icon"]',
    '[class*="closeButton"]',
    '[class*="modal"] button[class*="close"]',
    '[class*="popup"] button[class*="close"]',
    '[class*="drawer"] [class*="close"]',
    '[data-dismiss="modal"]',
    '.modal-close',
    'div[role="dialog"] button',
    '.newsletter-close',
    '[class*="newsletter"] [class*="close"]',
    # Stanley / Shopify-style: often a circular X button
    '[class*="Popup"] button',
    '[class*="popup"] [class*="Close"]',
    'button[type="button"]',  # last resort: first button in modal (often close)
]


async def _try_close_one_popup(page: Page) -> bool:
    """Try to close one popup; return True if we clicked something."""
    try:
        close_btn = page.get_by_role("button", name=re.compile(r"close|×|✕", re.I))
        if await close_btn.count() > 0:
            await close_btn.first.click(timeout=3000)
            return True
    except Exception:
        pass
    for sel in POPUP_CLOSE_SELECTORS:
        try:
            loc = page.locator(sel).first
            if await loc.count() > 0 and await loc.is_visible():
                await loc.click(timeout=3000)
                return True
        except Exception:
            continue
    return False


async def chat_close_popup(page: Page, step_prefix: str) -> str:
    """Try to close the topmost modal/popup (and optionally a second one, e.g. cookie banner). Returns path to screenshot."""
    if await _try_close_one_popup(page):
        await page.wait_for_timeout(600)
        await _try_close_one_popup(page)  # try again for second overlay
    await page.wait_for_timeout(400)
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path)
    return path


# Cookie consent: prefer these for "accept" (order matters); avoid opening preference modal
COOKIE_ACCEPT_TEXTS = ("ok", "accept", "accept all", "accept all cookies", "confirm my choices", "confirm", "agree", "allow all")
# Buttons that open preference/settings - do NOT click these when user said "accept cookies"
COOKIE_ACCEPT_SKIP_TEXTS = ("privacy choices", "manage cookies", "cookie settings", "preference", "settings", "manage")
COOKIE_DECLINE_TEXTS = ("decline", "reject", "decline non-essential", "only essential")


async def chat_cookie_consent(page: Page, accept: bool, step_prefix: str) -> str:
    """Click Accept or Decline; for accept, avoid clicking 'Your privacy choices' (opens preference modal)."""
    if accept:
        # First try buttons inside cookie/consent modal (IKEA etc.)
        for container in ("[class*='cookie']", "[id*='cookie']", "[class*='consent']", "[id*='consent']", "[role='dialog']"):
            try:
                btns = await page.locator(f"{container} button").all()
                for btn in btns:
                    try:
                        if not await btn.is_visible():
                            continue
                        text = (await btn.inner_text() or "").strip().lower()
                        if not text or any(s in text for s in COOKIE_ACCEPT_SKIP_TEXTS):
                            continue
                        if any(p in text for p in COOKIE_ACCEPT_TEXTS):
                            await btn.click(timeout=3000)
                            await page.wait_for_timeout(800)
                            path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
                            await take_screenshot(page, path)
                            return path
                    except Exception:
                        continue
            except Exception:
                continue
        try:
            loc = page.get_by_role("button", name=re.compile(r"accept|agree|ok", re.I))
            if await loc.count() > 0:
                await loc.first.click(timeout=3000)
                await page.wait_for_timeout(800)
                path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
                await take_screenshot(page, path)
                return path
        except Exception:
            pass
        # Get all buttons, pick first that matches accept text and does NOT match skip text
        try:
            buttons = await page.locator("button").all()
            for btn in buttons:
                try:
                    if not await btn.is_visible():
                        continue
                    text = (await btn.inner_text() or "").strip().lower()
                    if not text:
                        continue
                    skip = any(s in text for s in COOKIE_ACCEPT_SKIP_TEXTS)
                    if skip:
                        continue
                    if any(part in text for part in COOKIE_ACCEPT_TEXTS):
                        await btn.click(timeout=3000)
                        await page.wait_for_timeout(800)
                        break
                except Exception:
                    continue
        except Exception:
            for part in COOKIE_ACCEPT_TEXTS:
                try:
                    loc = page.get_by_role("button", name=re.compile(re.escape(part), re.I))
                    if await loc.count() > 0:
                        await loc.first.click(timeout=3000)
                        await page.wait_for_timeout(800)
                        break
                except Exception:
                    continue
    else:
        for part in COOKIE_DECLINE_TEXTS:
            try:
                loc = page.get_by_role("button", name=re.compile(re.escape(part), re.I))
                if await loc.count() > 0:
                    await loc.first.click(timeout=3000)
                    await page.wait_for_timeout(800)
                    break
            except Exception:
                continue
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path)
    return path


def _normalize_for_match(text: str) -> str:
    """Normalize for label/description comparison: lower, replace + with space, collapse spaces. So 'Data+AI' matches 'Data + AI'."""
    if not text:
        return ""
    t = (text or "").strip().lower()
    t = re.sub(r"\s*\+\s*", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _label_is_search_only(label: str) -> bool:
    """True if label is clearly a search control (so we don't match it when user said e.g. Estimating, Parts)."""
    t = (label or "").strip().lower()
    if not t:
        return False
    if t in ("search", "q"):
        return True
    if "search" in t:
        return True
    if len(t) <= 2 and "q" in t:
        return True
    return False


def _description_matches_label(desc_normalized: str, label: str) -> bool:
    """True if user description matches element label (with +/space normalization)."""
    label_norm = _normalize_for_match(label)
    if not label_norm:
        return False
    return desc_normalized in label_norm or label_norm in desc_normalized


async def chat_click_by_description(page: Page, description: str, step_prefix: str) -> tuple[Optional[str], str]:
    """
    Find a clickable element matching description, show red box, screenshot, then click.
    Prefers top-of-page (nav) and exact label match. Uses element bbox for red box after scroll.
    Skips search-only elements when user didn't ask for search (e.g. Data+AI -> not Search).
    """
    ensure_dirs()
    path_done = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    path_redbox: Optional[str] = None
    desc_clean = _normalize_for_match(description) or (description or "").strip().lower()
    if not desc_clean:
        await take_screenshot(page, path_done)
        return None, path_done
    desc_wants_search = any(s in desc_clean for s in ("search", "q"))
    elements = await get_interactive_elements(page)
    candidates: list[dict[str, Any]] = []
    for el in elements:
        label = (el.get("label") or "").strip()
        if not label:
            continue
        if _label_is_search_only(label) and not desc_wants_search:
            continue
        if not _description_matches_label(desc_clean, label):
            continue
        # For product-like names (multiple words), skip huge clickable areas (whole card) so we pick title link
        b = el.get("bbox") or {}
        area = (b.get("width") or 0) * (b.get("height") or 0)
        if len(desc_clean.split()) >= 2 and area > 40000:
            continue
        # When description looks like a nav dropdown item (e.g. Data + AI, Ecosystem), prefer only elements in dropdown zone (y < 450)
        NAV_DROPDOWN_ZONE_Y = 450
        nav_sub_keywords = ("data", "ai", "ecosystem", "ix cloud", "intelligent experiences", "platform")
        if any(k in desc_clean for k in nav_sub_keywords) and (b.get("y") or 0) >= NAV_DROPDOWN_ZONE_Y:
            continue
        candidates.append(el)
    # Prefer exact match; for short text (e.g. "alo") prefer logo: top 120px + earlier in DOM (first links = logo)
    # For longer text (e.g. "ALO Balance") prefer smaller bbox = title link, not whole product card
    TOP_PAGE_Y = 700
    LOGO_ZONE_Y = 120
    def _sort_key(e: dict) -> tuple:
        label_norm = _normalize_for_match(e.get("label") or "")
        exact = 0 if label_norm == desc_clean else 1
        b = e.get("bbox") or {}
        y_pos = b.get("y", 9999.0)
        in_top = 0 if y_pos < TOP_PAGE_Y else 1
        in_logo_zone = 0 if y_pos < LOGO_ZONE_Y else 1
        area_val = (b.get("width") or 0) * (b.get("height") or 0)
        # Earlier in DOM (smaller index) = usually logo for "a"; use negative so smaller index wins
        link_order = -(e.get("index", 0)) if (e.get("selector") == "a") else 0
        if len(desc_clean) <= 6:
            return (in_logo_zone, exact, in_top, link_order, area_val, y_pos)
        return (exact, in_top, area_val, y_pos)
    candidates.sort(key=_sort_key)
    clicked = False
    for el in candidates:
        try:
            sel = el.get("selector")
            idx = el.get("index", 0)
            if sel is None:
                continue
            loc = page.locator(sel).nth(idx)
            if await loc.count() == 0:
                continue
            # Re-verify current label before red box/click (DOM order can change; never click search)
            try:
                current_label = (
                    await loc.get_attribute("aria-label")
                    or await loc.get_attribute("title")
                    or await loc.evaluate("el => (el.innerText || '').slice(0, 80)")
                    or ""
                )
                current_label = (current_label or "").strip()
                if not current_label or current_label.lower() == "element":
                    continue
                if _label_is_search_only(current_label) and not desc_wants_search:
                    continue
                if not _description_matches_label(desc_clean, current_label):
                    continue
            except Exception:
                continue
            # Avoid scrolling when target is in top area (e.g. dropdown); scrolling can close the menu
            box_before = (el.get("bbox") or {}).get("y", 0)
            if box_before > 400:
                await loc.scroll_into_view_if_needed(timeout=3000)
            await page.wait_for_timeout(200)
            box = await loc.bounding_box()
            if box:
                try:
                    await inject_highlight_at_box(page, box)
                    await page.wait_for_timeout(600)
                    path_redbox = str(SCREENSHOT_DIR / f"{step_prefix}_redbox_{utc_ts()}.png")
                    await take_screenshot(page, path_redbox)
                    await page.wait_for_timeout(800)
                    await remove_highlight(page)
                except Exception:
                    pass
            await loc.click(timeout=5000)
            clicked = True
            break
        except Exception:
            continue
    if not clicked:
        # Fallback: try by role or has_text; allow "Data+AI" to match "Data + AI". Never click search.
        pattern = re.escape(description[:60]).replace(r"\+", r"\s*\+\s*")
        if not desc_wants_search:
            try:
                locs = await page.get_by_role("link", name=re.compile(pattern, re.I)).all()
                for loc in locs:
                    try:
                        name = (await loc.get_attribute("aria-label") or await loc.inner_text() or "").strip().lower()
                        if name and "search" in name:
                            continue
                        if await loc.is_visible():
                            await loc.scroll_into_view_if_needed(timeout=3000)
                            await loc.click(timeout=5000)
                            clicked = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if not clicked:
            try:
                locs = await page.locator("a").filter(has_text=re.compile(pattern, re.I)).all()
                for loc in locs:
                    try:
                        name = (await loc.get_attribute("aria-label") or await loc.inner_text() or "").strip().lower()
                        if not desc_wants_search and name and "search" in name:
                            continue
                        if await loc.is_visible():
                            await loc.scroll_into_view_if_needed(timeout=3000)
                            await loc.click(timeout=5000)
                            clicked = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if not clicked:
            try:
                locs = await page.get_by_role("button", name=re.compile(pattern, re.I)).all()
                for loc in locs:
                    try:
                        if await loc.is_visible():
                            await loc.scroll_into_view_if_needed(timeout=3000)
                            await loc.click(timeout=5000)
                            clicked = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if not clicked:
            try:
                locs = await page.locator("button, [role='button']").filter(has_text=re.compile(pattern, re.I)).all()
                for loc in locs:
                    try:
                        if await loc.is_visible():
                            await loc.scroll_into_view_if_needed(timeout=3000)
                            await loc.click(timeout=5000)
                            clicked = True
                            break
                    except Exception:
                        continue
            except Exception:
                pass
        if not clicked:
            try:
                text_clean = description.strip()[:50]
                loc = page.get_by_text(text_clean, exact=True).first
                if await loc.count() > 0 and await loc.is_visible():
                    await loc.scroll_into_view_if_needed(timeout=3000)
                    await loc.click(timeout=5000)
                    clicked = True
            except Exception:
                pass
        if not clicked:
            try:
                loc = page.get_by_text(description[:50], exact=False).first
                if await loc.count() > 0 and await loc.is_visible():
                    await loc.scroll_into_view_if_needed(timeout=3000)
                    await loc.click(timeout=5000)
                    clicked = True
            except Exception:
                pass
        if not clicked:
            try:
                clicked_js = await page.evaluate("""(text) => {
                    const t = (text || '').toLowerCase().trim().slice(0, 40);
                    const nodes = document.querySelectorAll('a, button, [role="button"], [role="link"]');
                    for (const el of nodes) {
                        if (!el.offsetParent) continue;
                        const s = (el.innerText || el.textContent || '').toLowerCase().trim();
                        if (s.includes(t) || t.split(/\\s+/).every(w => s.includes(w))) {
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }""", description[:40])
                if clicked_js:
                    clicked = True
            except Exception:
                pass
    # Wait for dropdown/menu to open (e.g. Solutions)
    await page.wait_for_timeout(1200)
    await take_screenshot(page, path_done)
    return path_redbox, path_done


async def chat_hover_by_description(page: Page, description: str, step_prefix: str) -> tuple[Optional[str], str]:
    """
    Find an element matching description and HOVER over it (no click). Use for menus that open on hover (e.g. Solutions).
    Prefers top-of-page (nav). Red box from element bbox after scroll; wait ~2s so dropdown clearly appears.
    """
    ensure_dirs()
    path_done = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    path_redbox: Optional[str] = None
    desc_clean = _normalize_for_match(description) or (description or "").strip().lower()
    if not desc_clean:
        await take_screenshot(page, path_done)
        return None, path_done
    desc_wants_search = any(s in desc_clean for s in ("search", "q"))
    elements = await get_interactive_elements(page)
    candidates = []
    for el in elements:
        label = (el.get("label") or "").strip()
        if not label:
            continue
        if _label_is_search_only(label) and not desc_wants_search:
            continue
        if not _description_matches_label(desc_clean, label):
            continue
        candidates.append(el)
    TOP_PAGE_Y = 700
    def _sort_key(e: dict) -> tuple[int, int, float]:
        label_norm = _normalize_for_match(e.get("label") or "")
        exact = 0 if label_norm == desc_clean else 1
        y = (e.get("bbox") or {}).get("y", 9999.0)
        in_top = 0 if y < TOP_PAGE_Y else 1
        return (exact, in_top, y)
    candidates.sort(key=_sort_key)
    hovered = False
    for el in candidates:
        try:
            sel = el.get("selector")
            idx = el.get("index", 0)
            if sel is None:
                continue
            loc = page.locator(sel).nth(idx)
            if await loc.count() == 0:
                continue
            await loc.scroll_into_view_if_needed(timeout=3000)
            await page.wait_for_timeout(300)
            box = await loc.bounding_box()
            if box:
                try:
                    await inject_highlight_at_box(page, box)
                    await page.wait_for_timeout(600)
                    path_redbox = str(SCREENSHOT_DIR / f"{step_prefix}_redbox_{utc_ts()}.png")
                    await take_screenshot(page, path_redbox)
                    await page.wait_for_timeout(400)
                    await remove_highlight(page)
                except Exception:
                    pass
            await loc.hover(timeout=5000)
            hovered = True
            break
        except Exception:
            continue
    if not hovered:
        pattern = re.escape(description[:60]).replace(r"\+", r"\s*\+\s*")
        try:
            loc = page.get_by_role("link", name=re.compile(pattern, re.I))
            if await loc.count() > 0:
                await loc.first.scroll_into_view_if_needed(timeout=3000)
                await loc.first.hover(timeout=5000)
                hovered = True
        except Exception:
            try:
                loc = page.locator("a").filter(has_text=re.compile(pattern, re.I)).first
                if await loc.count() > 0:
                    await loc.scroll_into_view_if_needed(timeout=3000)
                    await loc.hover(timeout=5000)
                    hovered = True
            except Exception:
                pass
    # Wait so dropdown is visible; no extra done screenshot (user wants only redbox for hover)
    await page.wait_for_timeout(2000)
    if path_redbox:
        return path_redbox, path_redbox
    await take_screenshot(page, path_done)
    return None, path_done


async def chat_click(
    page: Page,
    selector: Optional[str],
    nth_link: Optional[int],
    nth_result: Optional[int],
    link_text: Optional[str],
    description: Optional[str],
    step_prefix: str,
) -> tuple[Optional[str], str]:
    """Click by selector, nth link, nth result, link text, or description. Returns (redbox_path, done_path)."""
    if description:
        return await chat_click_by_description(page, description, step_prefix)
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    redbox_path: Optional[str] = None

    if link_text:
        loc = None
        try:
            loc = page.get_by_role("link", name=re.compile(re.escape(link_text), re.I)).first
            if await loc.count() == 0:
                loc = None
        except Exception:
            loc = None
        if loc is None:
            try:
                loc = page.locator("a").filter(has_text=link_text[:80]).first
            except Exception:
                pass
        if loc and await loc.count() > 0:
            try:
                await loc.scroll_into_view_if_needed(timeout=3000)
                box = await loc.bounding_box()
                if box:
                    await inject_highlight_at_box(page, box)
                    await page.wait_for_timeout(600)
                    redbox_path = str(SCREENSHOT_DIR / f"{step_prefix}_redbox_{utc_ts()}.png")
                    await take_screenshot(page, redbox_path)
                    await page.wait_for_timeout(700)
                    await remove_highlight(page)
                await loc.click(timeout=5000)
            except Exception:
                try:
                    await loc.click(timeout=5000)
                except Exception:
                    pass
    elif nth_result is not None:
        # Search/product result: get candidate links, sort by visual order (top-to-bottom, left-to-right), then click nth
        async def _get_links_sorted_by_position(locator) -> list:
            locs = await locator.all()
            with_pos = []
            for loc in locs:
                try:
                    if not await loc.is_visible():
                        continue
                    box = await loc.bounding_box()
                    if not box:
                        continue
                    with_pos.append((box.get("y", 0), box.get("x", 0), loc))
                except Exception:
                    continue
            with_pos.sort(key=lambda t: (round(t[0] / 20) * 20, t[1]))  # group by rough row (y), then x
            return [loc for _, _, loc in with_pos]

        tried = False
        for selector in (
            "a[href*='/products/'], a[href*='/product/'], a[href*='/p/'], a[href*='/item']",
            "[class*='product-grid'] a[href], [class*='search-results'] a[href], [class*='results-list'] a[href]",
            "[class*='product-card'] a[href], [class*='ProductCard'] a[href]",
            "main a[href], [role='main'] a[href]",
        ):
            try:
                result_links = page.locator(selector)
                n = await result_links.count()
                if n > nth_result:
                    ordered = await _get_links_sorted_by_position(result_links)
                    if len(ordered) > nth_result:
                        loc = ordered[nth_result]
                        await loc.scroll_into_view_if_needed(timeout=3000)
                        try:
                            box = await loc.bounding_box()
                            if box:
                                await inject_highlight_at_box(page, box)
                                await page.wait_for_timeout(600)
                                redbox_path = str(SCREENSHOT_DIR / f"{step_prefix}_redbox_{utc_ts()}.png")
                                await take_screenshot(page, redbox_path)
                                await page.wait_for_timeout(700)
                                await remove_highlight(page)
                        except Exception:
                            pass
                        await loc.click(timeout=5000)
                    else:
                        await result_links.nth(nth_result).click(timeout=5000)
                    tried = True
                    break
            except Exception:
                continue
        if not tried:
            try:
                all_links = await page.locator("a[href]:not([href^='#']):not([href^='javascript'])").all()
                content_links = []
                for loc in all_links:
                    try:
                        in_header_nav = await loc.evaluate(
                            "el => !!(el.closest('header') || el.closest('nav') || el.closest('[role=\"banner\"]'))"
                        )
                        if not in_header_nav and await loc.is_visible():
                            content_links.append(loc)
                    except Exception:
                        continue
                if len(content_links) > nth_result:
                    with_pos = []
                    for loc in content_links:
                        try:
                            box = await loc.bounding_box()
                            if box:
                                with_pos.append((box.get("y", 0), box.get("x", 0), loc))
                        except Exception:
                            continue
                    with_pos.sort(key=lambda t: (round(t[0] / 20) * 20, t[1]))
                    ordered = [loc for _, _, loc in with_pos]
                    if len(ordered) > nth_result:
                        await ordered[nth_result].scroll_into_view_if_needed(timeout=3000)
                        await ordered[nth_result].click(timeout=5000)
                    else:
                        await content_links[nth_result].click(timeout=5000)
            except Exception:
                pass
    elif nth_link is not None:
        try:
            loc = page.locator("a[href]").nth(nth_link)
            if await loc.count() > 0:
                await loc.click(timeout=5000)
        except Exception:
            pass
    elif selector:
        try:
            await page.click(selector, timeout=5000)
        except Exception:
            pass
    await page.wait_for_timeout(1200)
    path = str(SCREENSHOT_DIR / f"{step_prefix}_done_{utc_ts()}.png")
    await take_screenshot(page, path)
    return redbox_path, path
