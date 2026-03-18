# SerpAPI Agents

This folder contains two small agents built on top of [SerpAPI](https://serpapi.com/) (Google engine):

- `agent.py` — **URL → metadata**: given a website URL, fetch the first organic result and display basic metrics.
- `nl_agent.py` — **natural language → search results**: accepts a natural language instruction, parses time range constraints with an LLM, and runs a SerpAPI search (similar to the Brave Search agent).
- `unified_agent.py` — **interactive CLI**: LLM parsing + SerpAPI for URL metadata and search.

## 1. URL → Metadata (`agent.py`)

- **Input:** A website URL (e.g. `https://example.com`)
- **Process:** Queries SerpAPI with that URL as the query, takes the top organic result.
- **Output:**
  - **Title** and **Description** (snippet)
  - **Title Fetched** / **Desc Fetched** (Yes/No)
  - **Response Time** (seconds)
  - **Rate Limit Hit** (Yes/No)
  - Each run saves a text report under `results/<url>.txt` (e.g. `results/example_com.txt`)

## 2. Natural language → Search with time range (`nl_agent.py`)

This agent mirrors the Brave Search natural‑language agent, but uses SerpAPI/Google under the hood.

- **Input:** a natural language instruction, e.g.
  - `"latest AI research papers from last month"`
  - `"Python tutorials published in 2024 top 10"`
  - `"news about LangGraph between Jan and Mar 2025"`
- **LLM parsing:** a small Claude model turns the instruction into:
  - `search_query` — clean query string
  - `date_from` / `date_to` — explicit date range (YYYY‑MM‑DD) when the user mentions one
  - `freshness` — relative time window: `"pd"` (past day), `"pw"` (week), `"pm"` (month), `"py"` (year)
  - `result_count` — 1–20
- **Time range constraint:**
  - For explicit ranges, the agent appends `after:YYYY-MM-DD` / `before:YYYY-MM-DD` into the query.
  - For relative ranges, it uses Google’s `tbs=qdr:*` parameter (e.g. day/week/month/year) when possible.
- **Output:** a small list of results (title + URL), plus the parsed time range and response time.

## Setup

1. **Clone or copy** this folder.

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set your API keys**  
   Create a `.env` file in the project root:
   ```bash
   SERPAPI_API_KEY=your_serpapi_key_here
   ANTHROPIC_API_KEY=your_claude_key_here   # required for nl_agent.py
   ```
   Get keys at [`serpapi.com`](https://serpapi.com/) and [`anthropic.com`](https://www.anthropic.com/).

## Usage

### 1) Single URL → metadata

```bash
python agent.py https://www.example.com
```

Or run without arguments to be prompted for a URL:

```bash
python agent.py
```
The report is printed in the terminal and saved to `results/<sanitized_url>.txt` (e.g. `results/example_com.txt`).

### 2) Natural language → search with time range

```bash
python nl_agent.py "latest AI research papers from last month"
python nl_agent.py "Python tutorials published in 2024 top 10"
```

Or run without arguments to be prompted for an instruction:

```bash
python nl_agent.py
```

### 3) Batch (multiple URLs)

1. Put one URL per line in `urls.txt` at the project root (you can omit `https://`).
2. Run:
   ```bash
   python batch_run.py
   ```
3. Outputs:
   - **CSV:** `results/results.csv` — columns: `url`, `title`, `description`, `title_fetched`, `description_fetched`, `response_time_s`
   - **Text:** One `.txt` per URL in `results/` (same format as single runs)

You can add your own “Result Accuracy” or other score columns in the CSV when filling your table.

## Project layout

```
SerpAPI/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── agent.py            # Single-URL agent (URL -> metadata)
├── nl_agent.py         # Natural-language agent (instruction -> search results)
├── unified_agent.py    # Interactive CLI (LLM parsing + SerpAPI)
├── batch_run.py        # Batch runner (reads urls.txt, writes CSV + txt)
├── urls.txt            # Optional: one URL per line for batch runs
├── .env                # API keys (SERPAPI_API_KEY, ANTHROPIC_API_KEY)
└── results/            # Output folder
    ├── results.csv    # Batch summary (from batch_run.py)
    └── *.txt          # One report per URL (e.g. example_com.txt)
```

## Requirements

- Python 3.10+
- `requests`, `python-dotenv`, `langgraph`
- For natural-language mode: `langchain`, `langchain-core`, `langchain-anthropic`
  (all pinned in `requirements.txt`)

## License

Use as you like. SerpAPI has its own [terms and pricing](https://serpapi.com/pricing).
