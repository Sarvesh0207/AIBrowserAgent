from dataclasses import dataclass
from typing import Optional

from playwright.async_api import async_playwright

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


@dataclass
class ActionConfirmation:
    """Result of a single 'click' or 'fill' action in headless mode."""
    action_type: str  # "click" or "fill"
    selector: str
    value: Optional[str] = None  # for fill only
    success: bool = False
    message: str = ""
    submitted: bool = False  # True if we pressed Enter after fill (e.g. to submit search)


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
) -> tuple[PageResult, Optional[ActionConfirmation]]:
    """
    Load a page, then perform one action (click or fill) and confirm.
    If open_search_selector is set, click it first to reveal the search bar, then fill.
    If submit_after_fill is True, after filling we press Enter to submit (e.g. search).
    Returns (PageResult, ActionConfirmation) so the caller can log and display the result.
    """
    ensure_dirs()
    slug = _safe_slug(url)
    action_suffix = "click" if click_selector else "fill_submit" if (fill_selector and submit_after_fill) else "fill" if fill_selector else "none"
    if open_search_selector:
        action_suffix = "open_search_" + action_suffix
    screenshot_path = str(SCREENSHOT_DIR / f"{slug}_action_{action_suffix}_{utc_ts()}.png")

    confirmation: Optional[ActionConfirmation] = None

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
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

            return (
                PageResult(
                    url=url,
                    final_url=final_url,
                    title=title,
                    description=desc,
                    screenshot_path=screenshot_path,
                ),
                confirmation,
            )
        finally:
            await page.close()
            await context.close()
            await browser.close()
