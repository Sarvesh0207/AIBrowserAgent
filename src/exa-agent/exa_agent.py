import os
import time
import csv
from dotenv import load_dotenv
from exa_py import Exa

load_dotenv()

api_key = os.getenv("EXA_API_KEY")

if not api_key:
    raise ValueError("EXA_API_KEY not found in .env file")

exa = Exa(api_key)

websites = [
    {"name": "UIC Student Portal", "domain": "my.uic.edu"},
    {"name": "United Airlines", "domain": "united.com"},
    {"name": "Postman Documentation", "domain": "learning.postman.com"},
    {"name": "Amazon", "domain": "amazon.com"},
    {"name": "Zillow", "domain": "zillow.com"},
    {"name": "Hacker News", "domain": "news.ycombinator.com"},
    {"name": "X (formerly Twitter)", "domain": "twitter.com"},
    {"name": "Airbnb", "domain": "airbnb.com"},
    {"name": "CNN", "domain": "cnn.com"},
    {"name": "Python", "domain": "python.org"},
    {"name": "Wikipedia", "domain": "wikipedia.org"},
    {"name": "Stackoverflow", "domain": "stackoverflow.com"},
    {"name": "NASA", "domain": "nasa.gov"},
    {"name": "arxiv", "domain": "arxiv.org"},
    {"name": "Medium", "domain": "medium.com"},
    {"name": "Cloudflare", "domain": "cloudflare.com"},
    {"name": "eBay", "domain": "ebay.com"}
]

results = []

for site in websites:
    website_name = site["name"]
    domain = site["domain"]

    print(f"\nSearching for: {website_name} ({domain})")
    start = time.time()

    try:
        response = exa.search(
            query=f"{website_name} official site",
            num_results=1,
            include_domains=[domain]
        )

        end = time.time()
        latency = round(end - start, 3)

        if response.results:
            result = response.results[0]
            title = result.title if result.title else "N/A"
            description = result.text[:200].replace("\n", " ") if result.text else "N/A"
            url = result.url if result.url else "N/A"
        else:
            title = "Not Found"
            description = "Not Found"
            url = "N/A"

    except Exception as e:
        end = time.time()
        latency = round(end - start, 3)
        title = "Error"
        description = str(e)
        url = "N/A"

    results.append([website_name, domain, title, description, url, latency])

    print("============================")
    print("Website Name:", website_name)
    print("Domain:", domain)
    print("Title:", title)
    print("Description:", description)
    print("URL:", url)
    print("Response Time:", latency, "seconds")

with open("results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Website_Name",
        "Domain",
        "Title",
        "Description",
        "URL",
        "Response_Time_Seconds"
    ])
    writer.writerows(results)

print("\nDone. Results saved to results.csv")
