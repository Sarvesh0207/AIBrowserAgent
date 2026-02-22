try:
    from langgraph.types import interrupt
except Exception:
    interrupt = None


def ask_user(prompt: str) -> str:
    """
    Local terminal fallback now.
    Later: replace with LangGraph interrupt / UI / queue.
    """
    if interrupt is not None:
        return interrupt(prompt)
    return input(prompt + "\n> ").strip()
