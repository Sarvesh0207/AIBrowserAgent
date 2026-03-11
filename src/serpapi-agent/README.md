# SerpAPI Metadata Fetch Agent

Fetch the first organic search result for a given URL using [SerpAPI](https://serpapi.com/) (Google engine) and display metadata. No automatic scoring — you can rate results yourself in a spreadsheet.

## What it does

- **Input:** A website URL (e.g. `https://example.com`)
- **Process:** Queries SerpAPI with that URL, takes the top organic result
- **Output:**
  - **Title** and **Description** (snippet)
  - **Title Fetched** / **Desc Fetched** (Yes/No)
  - **Response Time** (seconds)
  - **Rate Limit Hit** (Yes/No)
  - Each run saves a text report under `results/<url>.txt` (e.g. `results/example_com.txt`)

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

4. **Set your API key**  
   Create a `.env` file in the project root:
   ```
   SERPAPI_API_KEY=your_serpapi_key_here
   ```
   Get a key at [serpapi.com](https://serpapi.com/).

## Usage

### Single URL

```bash
python agent.py https://www.example.com
```

Or run without arguments to be prompted for a URL:

```bash
python agent.py
```

The report is printed in the terminal and saved to `results/<sanitized_url>.txt` (e.g. `results/example_com.txt`).

### Batch (multiple URLs)

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
├── agent.py            # Single-URL agent (SerpAPI + LangGraph)
├── batch_run.py        # Batch runner (reads urls.txt, writes CSV + txt)
├── urls.txt            # Optional: one URL per line for batch runs
├── .env                # Your SERPAPI_API_KEY (create this yourself)
└── results/            # Output folder
    ├── results.csv    # Batch summary (from batch_run.py)
    └── *.txt          # One report per URL (e.g. example_com.txt)
```

## Requirements

- Python 3.10+
- `requests`, `python-dotenv`, `langgraph` (see `requirements.txt`)

## License

Use as you like. SerpAPI has its own [terms and pricing](https://serpapi.com/pricing).
