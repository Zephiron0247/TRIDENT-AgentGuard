import json
from datetime import datetime, timezone
from urllib import error, request
from uuid import uuid4

TOOL_CALL_URL = "http://localhost:8000/tool-call"

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


def execute_tool(tool_name: str, tool_args: dict) -> None:
    print(f"    EXECUTING: {tool_name}")

    if tool_name == "search_docs":
        print(f"    Result: searched documents for '{tool_args['query']}'")

    elif tool_name == "read_file":
        print(f"    Result: read '{tool_args['path']}'")

    elif tool_name == "summarize":
        print("    Result: summary generated")

    elif tool_name == "send_email":
        print(f"    Result: email prepared for {tool_args['to']}")

    elif tool_name == "read_customer_data":
        print(f"    Result: customer data read ({tool_args['segment']})")

    elif tool_name == "access_credentials":
        print(f"    Result: credentials accessed ({tool_args['vault']})")

    elif tool_name == "export_database":
        print(f"    Result: database export completed ({tool_args['table']})")

    elif tool_name == "visit_url":
        print(f"    Result: visited {tool_args['url']}")

    else:
        print("    Result: tool executed")


def main() -> None:
    print("=== TRIDENT — live interactive demo ===")
    print("Every choice sends a REAL request to", TOOL_CALL_URL)

    session_id = f"live-demo-{uuid4().hex[:8]}"
    agent_id = "presenter"

    stated_goal = input(
        "Stated goal for this session (press enter for default): "
    ).strip()

    if not stated_goal:
        stated_goal = "demonstrate live tool-call monitoring"

    print(f"\nSession started: {session_id}")

    while True:
        print("\nPick the agent's next move:")

        for key, (tool_name, _) in TOOLS.items():
            print(f"  [{key}] {tool_name}")

        print("  [new]  start a fresh session")
        print("  [quit] exit")

        choice = input("> ").strip().lower()

        if choice == "quit":
            print("Bye.")
            return

        if choice == "new":
            session_id = f"live-demo-{uuid4().hex[:8]}"
            print(f"\nNew session started: {session_id}")
            continue

        if choice not in TOOLS:
            print("Not a valid choice.")
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
            print(
                f"HTTP {exc.code}: "
                f"{exc.read().decode('utf-8', errors='replace')}"
            )
            continue

        except Exception as exc:
            print(f"Request failed: {exc}")
            continue

        decision = result.get("decision", "?").upper()
        score = result.get("risk_score", 0.0)

        print(
            f"\n>>> {tool_name} -> {decision} "
            f"(risk_score={score:.2f})"
        )

        print(f"    {result.get('explanation', '')}")

        if result.get("is_trigger_step"):
            print(
                f"    TRIGGER: "
                f"{result.get('trigger_reason')}"
            )

        # REAL enforcement demonstration:
        if decision == "ALLOW" or decision == "ALLOW_FLAGGED":
            execute_tool(tool_name, tool_args)

        elif decision == "BLOCK":
            print("    BLOCKED: tool execution prevented.")

        elif decision == "KILLSWITCH":
            print("    KILLSWITCH: session terminated.")
            print(
                "    Tool was NOT executed. "
                "Choose 'new' to start another session."
            )


if __name__ == "__main__":
    main()