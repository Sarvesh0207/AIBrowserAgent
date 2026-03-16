# 🌐 WebAgent — AI Browser with SoM + Human-in-the-Loop

A terminal-based AI web browsing agent powered by **LangGraph**, **Claude Sonnet**, and **Playwright**. Inspired by the WebVoyager paper's **Set-of-Mark (SoM)** approach for grounded, accurate web interaction.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Terminal UI)                    │
│   Rich-powered REPL  ·  interrupt handler  ·  screenshot paths  │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HumanMessage
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph StateGraph                         │
│                                                                 │
│   ┌──────────┐  tool_calls   ┌──────────┐                      │
│   │  agent   │──────────────▶│  tools   │                      │
│   │  (LLM)   │◀──────────────│  node    │                      │
│   └──────────┘  ToolMessages └────┬─────┘                      │
│        │                          │                            │
│        │                    interrupt() ──▶ Human input        │
│        ▼                                                        │
│      END                                                        │
│                                                                 │
│  MemorySaver checkpointer → full multi-turn history            │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     agent.py (Tools)                            │
│                                                                 │
│  navigate_to_url      → go to URL, return PageState            │
│  capture_current_page → screenshot + SoM                       │
│  click_element(n)     → click element by SoM index             │
│  type_text(n, text)   → type into input #n                     │
│  select_dropdown(n)   → choose dropdown option                 │
│  press_keyboard_key   → Enter, Tab, Escape, etc.               │
│  scroll_page          → up/down                                │
│  go_back              → browser back                           │
│  ask_human(question)  → LangGraph interrupt() → user input     │
└──────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  browser_tools.py (Playwright)                  │
│                                                                 │
│  BrowserController                                              │
│   ├── navigate()          → page.goto()                        │
│   ├── capture_state()     → screenshot + element extraction    │
│   ├── _extract_elements() → JS querySelectorAll → ElementInfo  │
│   └── _draw_som()         → PIL bounding box overlay           │
│                                                                 │
│  SoM Color Legend                                               │
│   🔵 Blue   = Links         🟢 Green  = Buttons               │
│   🟡 Amber  = Input fields  🟣 Purple = Dropdowns             │
│   🔴 Red    = Text areas    ⚫ Grey   = Other interactive      │
└──────────────────────────────────────────────────────────────────┘
```

## File Structure

```
webagent/
├── main.py            # Terminal entry point (Rich REPL)
├── agent.py           # LangGraph graph + tools + system prompt
├── browser_tools.py   # Playwright controller + SoM screenshot engine
├── pyproject.toml     # Python package + dependencies
├── setup.sh           # One-command bootstrap
├── .env.example       # Environment variable template
├── .env               # Your API keys (git-ignored)
└── screenshots/       # Auto-created, stores all screenshots
```

## Setup

```bash
# Clone / enter the directory
cd webagent

# One-command setup (creates venv, installs deps, installs Chromium)
chmod +x setup.sh && ./setup.sh

# Add your API key to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env

# Run
source .venv/bin/activate
python main.py
```

## Usage Examples

```bash
# Start with a URL
python main.py --url https://google.com

# Headless (no browser window)
python main.py --headless

# Interactive (browser window opens)
python main.py
```

### Natural Language Instructions

The agent understands natural language, just like ChatGPT or Claude:

```
You: go to wikipedia and search for the James Webb telescope
You: click the first search result
You: scroll down and find the launch date section
You: go to amazon.com and search for mechanical keyboards under $100
You: I want to book a flight from Chicago to Tokyo in December
```

### Human-in-the-Loop Scenarios

The agent will **pause and ask you** in these situations:

| Scenario | Example question |
|---|---|
| Dropdown with multiple options | "I see a dropdown #5 with: [Economy, Business, First Class]. Which would you like?" |
| Vague instruction | "Could you clarify — what exactly should I do on this page?" |
| Destructive action | "I'm about to submit this form. Confirm? (yes/no)" |
| Credentials needed | "This page requires a login. Please provide your username." |

## Set-of-Mark (SoM) Screenshots

Every page capture produces **two screenshots**:

1. **`_clean.png`** — Raw screenshot, no overlays
2. **`_som.png`** — Annotated with numbered bounding boxes

The SoM approach (from the WebVoyager paper) assigns each interactive element a visible number. The agent references these numbers when deciding what to click, instead of fragile CSS selectors. This makes the agent much more robust to DOM structure changes.

```
Element #7 [button]: "Add to Cart"    ← agent clicks by number
Element #12 [input]: placeholder="Search"
Element #23 [select]: options=["USD", "EUR", "GBP"]  ← triggers ask_human
```

## Key Design Decisions

**Why `interrupt()` instead of a custom tool loop?**  
LangGraph's `interrupt()` fully suspends graph execution and preserves state in the checkpointer. When the human answers, the graph resumes exactly where it left off — no state is lost, the full message history is intact.

**Why PIL for SoM instead of canvas/JS?**  
PIL operates on the final PNG, meaning the overlays are always pixel-perfect regardless of CSS transforms, z-index stacking, or scroll position. JavaScript canvas overlays can be clipped or occluded by the page's own elements.

**Why headed mode by default?**  
Many modern sites detect headless browsers and block them (Cloudflare, etc.). Headed mode with a realistic user-agent string gets much better site compatibility.
