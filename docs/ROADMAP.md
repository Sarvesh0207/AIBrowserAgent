# This Week and Next: Implementation Order

## This Week (Agreed with Instructor/Team)

| Order | Item | Notes |
|-------|------|------|
| 1 | **GitHub public repo** | Create repo and email the link to Amirreza. |
| 2 | **Headful browsing** | Browser with visible window; support wait, click, screenshot, and drawing bounding boxes on screenshots. |
| 3 | **Date range search research** | Research “date range” search options and document in `docs/DATE_RANGE_SEARCH.md`. |

## What’s Needed for “Tell the agent what to search and which button to click”

Target behavior:

- After the agent opens a page, **list interactive elements** (e.g. search box, Directions button).
- User gives **natural language** instructions: “Search for 1536 w 18th st in the search box”, “Click the Directions button”.
- Agent turns that into **actions** (fill a selector / click a selector), runs them, then takes a screenshot and asks for the next step.

This can be split into three parts:

1. **Page parsing (element list + bounding box)**  
   After loading the page, use Playwright to find clickable and fillable elements and get for each:
   - CSS selector (for click/fill)
   - Bounding box (x, y, width, height)
   - Short label (placeholder, aria-label, button text, etc.)  
   Draw each element’s box on the screenshot and save an “annotated” screenshot for humans and the LLM.

2. **Wait behavior**  
   After navigation or click, wait for the page to settle before taking a screenshot or reading elements (e.g. `wait_for_load_state`, short `wait_for_timeout`) so we don’t act before the page is ready.

3. **Conversation loop + LLM for natural language**  
   - Input: current screenshot (or annotated) + element list (index, label, selector).
   - User types one instruction (e.g. “Search XXX in the search box”, “Click the 2nd button”).
   - LLM outputs: action type (click / fill), target (selector or index), and value for fill if needed.
   - Run the action with the existing `browser` click/fill → take screenshot, refresh element list → repeat.

## Suggested order (this week)

1. **Finish the three items above**  
   - GitHub, headful + wait + screenshot + bounding box, date-range doc.  
   - That gives you “visible screenshots with boxes” and “stable loading”.

2. **Add “element list + draw bbox” in the codebase**  
   - In `src/browser.py` (or a new module):  
     - Get interactive elements (input, button, a, [role=button], etc.) with selector and bbox.  
     - Use Pillow to draw rectangles (and optional labels) on the screenshot and save an “annotated” image.  
   - Then the “conversation agent” can use this element list and annotated screenshot as LLM input.

3. **Add “conversation mode”**  
   - New command (e.g. `run-chat`) or extend `run-hitl`:  
     - Open headful page → get element list + annotated screenshot → send “screenshot + element list” to Claude Sonnet.  
     - User types natural language → LLM returns click(selector) or fill(selector, value) → run with Playwright → take screenshot, refresh elements, repeat.  
   - That achieves “tell the agent what to search and which button to click”.

**Summary:** This week do GitHub + headful + wait + screenshot + bounding box + date-range doc; at the same time add “interactive element list + draw bbox” in the browser layer. Later add the “conversation loop + LLM natural language” to match the target behavior.
