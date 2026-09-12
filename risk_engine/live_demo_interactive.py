"""
TRIDENT — interactive LIVE demo (no scripted playback, no API key needed).

Instead of replaying a fixed scenario (like demo_agent.py does), this lets
YOU or a volunteer pick tool calls one at a time from a menu, live, in front
of the room. Every choice goes out as a real HTTP request to your real
/tool-call endpoint. Nothing here is pre-decided — you can genuinely try to
"break" it live and see what happens, without needing an LLM API key or
worrying about an autonomous agent doing something unpredictable on stage.

Usage:
    1. Start your backend first:  python -m uvicorn backend.main:app --reload --port 8000
    2. In another terminal, run:  python live_demo_interactive.py
    3. Pick a tool from the numbered menu each turn. Watch the decision print live.
    4. Type "new" to start a fresh session, or "quit" to exit.
"""

import json
from datetime import datetime, timezone
from urllib import error, request
from uuid import uuid4

TOOL_CALL_URL = "http://localhost:8000/tool-call"

# Fake tools the "agent" can be told to use. Feel free to add your own.
TOOLS = {
    "1": ("search_docs", {"query": "Q3 report draft"}),
    "2": ("read_file", {"path": "notes/week12.txt"}),
    "3": ("summarize", {"style": "concise"}),
    "4": ("send_email", {"to": "team@example.com", "subject": "Update"}),
    "5": ("read_customer_data", {"segment": "enterprise"}),
    "6": ("access_credentials", {"vault": "finance-prod"}),
    "7": ("export_database", {"table": "customers", "format": "csv"}),
    "8": ("visit_url", {"url": "https://en.wikipedia.org/wiki/Example"}),
    "9": ("visit_url", {"url": "http://192.168.1.22/hidden"}),
}


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _post_event(event: dict) -> dict:
    payload = json.dumps(event).encode("utf-8")
    req = request.Request(
        TOOL_CALL_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _menu() -> None:
    print("\nPick the agent's next move:")
    for key, (tool_name, _) in TOOLS.items():
        print(f"  [{key}] {tool_name}")
    print("  [new]  start a fresh session")
    print("  [quit] exit")


def main() -> None:
    print("=== TRIDENT — live interactive demo ===")
    print("Every choice below sends a REAL request to", TOOL_CALL_URL)
    session_id = f"live-demo-{uuid4().hex[:8]}"
    agent_id = "presenter"
    stated_goal = input("Stated goal for this session (press enter for default): ").strip()
    if not stated_goal:
        stated_goal = "demonstrate live tool-call monitoring"

    print(f"\nSession started: {session_id}")

    while True:
        _menu()
        choice = input("> ").strip().lower()

        if choice == "quit":
            print("Bye.")
            return

        if choice == "new":
            session_id = f"live-demo-{uuid4().hex[:8]}"
            print(f"\nNew session started: {session_id}")
            continue

        if choice not in TOOLS:
            print("Not a valid choice, try again.")
            continue

        tool_name, tool_args = TOOLS[choice]
        event = {
            "session_id": session_id,
            "agent_id": agent_id,
            "stated_goal": stated_goal,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "timestamp": _iso_now(),
        }

        try:
            result = _post_event(event)
        except error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode('utf-8', errors='replace')}")
            continue
        except Exception as exc:  # noqa: BLE001
            print(f"Request failed: {exc}")
            print("Is the backend running? (uvicorn backend.main:app --port 8000)")
            continue

        decision = result.get("decision", "?").upper()
        score = result.get("risk_score", 0.0)
        print(f"\n>>> {tool_name} -> {decision}  (risk_score={score:.2f})")
        print(f"    {result.get('explanation', '')}")
        if result.get("is_trigger_step"):
            print(f"    TRIGGER: {result.get('trigger_reason')}")

        if decision == "KILLSWITCH":
            print("\nSession terminated by killswitch. Choose 'new' to start another session.")


if __name__ == "__main__":
    main()