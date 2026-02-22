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
