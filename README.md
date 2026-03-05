# AIBrowserAgent
The objective of this project is to design and implement a structured AI agent that can interpret natural language instructions, launch a browser session, and execute step-by-step actions while explaining its reasoning to accomplish a given tasks.

## Requirements

- Python 3.11+ (3.11 or 3.12 recommended)
- pip and a virtual environment (e.g. `venv` or `conda`)

## Natural language entry (--prompt / run-hitl)

You can give a **sentence** instead of a raw URL; the agent uses an LLM to resolve it to a URL, then browses and summarizes.

**Command line (--prompt):**
```bash
python main.py run --prompt "Visit example.com and summarize"
python main.py run --prompt "Go to Wikipedia and summarize"
```

**Interactive (run-hitl):** When asked "What would you like me to do?", you can type either a URL or natural language (e.g. "visit Stanley's website").
- If the input looks like a URL, it is used as-is.
- Otherwise, the LLM extracts a URL from your sentence and the agent visits that URL.

## Headless click / fill (run-action)

In headless mode, you can run **one** click or one form fill and confirm the result (screenshot + log):

```bash
# Click the first link on the page
python main.py run-action --url https://example.com --click "a"

# Fill an input (e.g. search box) and confirm
python main.py run-action --url https://example.com --fill "input[name=q]" --fill-value "hello"

# Fill search box and actually submit (press Enter) to go to search results
python main.py run-action --url https://www.stanley1913.com --fill "input[placeholder*='looking for']" --fill-value "water bottle" --submit
```

- **--click** and **--fill** / **--fill-value** are mutually exclusive; use exactly one.
- **--submit**: use only with **--fill**; after filling, press Enter to submit (e.g. search). Screenshot is taken after the page navigates.
- Selectors are CSS selectors (e.g. `a`, `button.submit`, `input#search`).
- Result: screenshot under `outputs/screenshots/`, action log under `outputs/logs/action_*.jsonl`, and a printed confirmation (success/failure, and whether search was submitted).

## Sharing the project (avoid leaking API keys)

- **Do not** zip or share the folder if it contains a `.env` file — that file holds your API keys and would be leaked.
- **Safe options:**  
  1. **Git:** Push only the repo (`.env` is in `.gitignore`, so it is not pushed). Teammates clone and add their own `.env` from `.env.example`.  
  2. **Zip:** Before zipping, delete or move `.env` out of the project folder (or exclude it when creating the archive). Share `.env.example` only; each person copies it to `.env` and fills in their own keys.
- Teammates: copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY` (and any other keys). Never commit or share `.env`.
<!-- refresh contributors -->