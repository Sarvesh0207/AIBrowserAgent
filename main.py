import argparse
import asyncio
from src.agent_graph import build_graph
from src.evaluate_headless import run_headless_evaluation

def parse_args():
    parser = argparse.ArgumentParser("AIBrowserAgent")

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--url", required=True)

    eval_parser = subparsers.add_parser("benchmark")
    eval_parser.add_argument("--csv", required=True)

    return parser.parse_args()

async def run_single(url):
    app = build_graph()
    result = await app.ainvoke({"url": url})

    print("\n=== RESULT ===")
    print("URL:", result.get("url"))
    print("Title:", result.get("title"))
    print("Summary:", result.get("summary"))
    print("================\n")

async def run_benchmark(csv_path):
    await run_headless_evaluation(csv_path)

def main():
    args = parse_args()

    if args.command == "run":
        asyncio.run(run_single(args.url))

    elif args.command == "benchmark":
        asyncio.run(run_benchmark(args.csv))

if __name__ == "__main__":
    main()
