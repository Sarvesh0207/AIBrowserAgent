from typing import Tuple
from playwright.async_api import Page


async def extract_title_and_description(page: Page) -> Tuple[str, str]:
    title = (await page.title() or "").strip()

    meta = page.locator("meta[name='description']").first
    meta_desc = ""
    if await meta.count():
        meta_desc = (await meta.get_attribute("content") or "").strip()

    if meta_desc:
        return title, meta_desc

    # Fallback: first chunk of body text
    desc = ""
    if await page.locator("body").count():
        body_text = await page.inner_text("body")
        body_text = " ".join((body_text or "").split())
        desc = body_text[:600].strip()

    return title, desc
