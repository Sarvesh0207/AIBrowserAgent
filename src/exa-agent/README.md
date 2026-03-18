# Exa Search Agent

## Overview

This project builds a Python agent using the Exa API to retrieve website metadata and perform natural language-based web search with date filtering.

## Part 1 — Website Metadata Benchmark

A basic agent was implemented to fetch the title and description of benchmark websites and measure response time.

## Websites Tested

17 websites were evaluated, including:

my.uic.edu
united.com
learning.postman.com
amazon.com
zillow.com
news.ycombinator.com
twitter.com/explore
airbnb.com
cnn.com
python.org
wikipedia.org
stackoverflow.com
nasa.gov
arxiv.org
medium.com
cloudflare.com
ebay.com

## Method

For each website:

- Query Exa using "<website name> official site"
- Restrict results to the expected domain
- Extract:
  - Title
  - Description
  - URL
- Record response time

Results are saved to:

results.csv

## Evaluation

Accuracy was scored manually:

Score    Meaning
5        Perfect match
4        Mostly correct
3        Partially relevant
1        Incorrect

## Results

- Websites tested: 17
- Successful results: 16 / 17
- Average response time: ~1.1 seconds
- Accuracy: ~94%

Screenshots of terminal outputs are included in the screenshots folder.

------------------------------------------------------------

# Part 2 — Natural Language Search Agent

An extended agent (nl_agent.py) was implemented to accept natural language instructions and perform web searches with time range filtering.

## Features

- Accepts natural language input
- Extracts search query and date constraints
- Supports:
  - "last month"
  - "last week"
  - "last year"
  - "between Jan and Mar 2025"
  - "in 2024"
- Uses LangGraph to structure the workflow:
  parse → search → results
- Returns:
  - Title
  - URL
  - Description
  - Response time

## Example Usage

python3 nl_agent.py "latest AI research papers from last month"

python3 nl_agent.py "Python tutorials published in 2024"

python3 nl_agent.py "news about LangGraph between Jan and Mar 2025"

## Notes

- Natural language parsing is implemented using rule-based logic (no external LLM required)
- This keeps the agent lightweight and avoids dependency on additional APIs
