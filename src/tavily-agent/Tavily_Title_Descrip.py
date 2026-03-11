from agent import run

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        target_url = sys.argv[1]
    else:
        target_url = input("\nEnter website URL: ").strip()

    if not target_url.startswith("http"):
        target_url = "https://" + target_url

    run(target_url)

