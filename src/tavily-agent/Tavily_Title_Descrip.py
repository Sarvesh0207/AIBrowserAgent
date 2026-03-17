from agent import run

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        instruction = " ".join(sys.argv[1:]).strip()
    else:
        instruction = input("\nEnter search instruction: ").strip()

    run(instruction)

