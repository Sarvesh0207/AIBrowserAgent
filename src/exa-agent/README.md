# Exa Search Benchmark

## Overview

This project builds a simple Python agent using the Exa API to retrieve the **title** and **description** of a set of benchmark websites and measure response time.

## Websites Tested

17 websites were evaluated, including:

* my.uic.edu
* united.com
* learning.postman.com
* amazon.com
* zillow.com
* news.ycombinator.com
* twitter.com/explore
* airbnb.com
* cnn.com
* python.org
* wikipedia.org
* stackoverflow.com
* nasa.gov
* arxiv.org
* medium.com
* cloudflare.com
* ebay.com

## Method

For each website:

1. Query Exa using `"<website name> official site"`.
2. Restrict results to the expected domain.
3. Extract:

   * Title
   * Description
   * URL
4. Record response time.

Results are saved to:

```text
results.csv
```

## Evaluation

Accuracy was scored manually:

| Score | Meaning            |
| ----- | ------------------ |
| 5     | Perfect match      |
| 4     | Mostly correct     |
| 3     | Partially relevant |
| 1     | Incorrect          |

## Results

* Websites tested: **17**
* Successful results: **16 / 17**
* Average response time: **~1.1 seconds**
* Accuracy: **~94%**

Screenshots of terminal outputs for all tested websites are included in the screenshots folder.

