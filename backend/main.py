from datetime import datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.exasol_client import (
    get_connection,
    get_recent_events,
    get_session_calls,
    insert_session,
    insert_tool_call,
    update_session_status,
)
from risk_engine.scorer import score_trajectory


class ToolCallRequest(BaseModel):
    session_id: str
    agent_id: str
    stated_goal: str
    tool_name: str
    tool_args: Any
    timestamp: str


app = FastAPI(title="AgentGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _parse_timestamp(ts: Any) -> datetime:
    if isinstance(ts, datetime):
        return ts

    if not isinstance(ts, str):
        raise ValueError("timestamp must be an ISO8601 string")

    normalized = ts.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized)


def _get_session_status(session_id: str) -> str | None:
    c = get_connection()
    stmt = c.execute(
        """
        SELECT status
        FROM sessions
        WHERE session_id = {session_id}
        """,
        {"session_id": session_id},
    )
    row = stmt.fetchone()
    if not row:
        return None
    return row[0]


def _decision_from_score(score: float) -> str:
    if score < 0.4:
        return "allow"
    if score < 0.7:
        return "allow_flagged"
    if score <= 0.9:
        return "block"
    return "killswitch"


@app.post("/tool-call")
def tool_call(payload: ToolCallRequest) -> dict[str, Any]:
    status = _get_session_status(payload.session_id)

    if status and status.lower() == "killed":
        return {
            "decision": "killswitch",
            "risk_score": 1.0,
            "reason": "session already terminated",
        }

    if status is None:
        insert_session(payload.session_id, payload.agent_id, payload.stated_goal)

    prior_calls = get_session_calls(payload.session_id)

    events: list[dict[str, Any]] = []
    for call in prior_calls:
        events.append(
            {
                "tool_name": call.get("tool_name"),
                "timestamp": _parse_timestamp(call.get("called_at")),
            }
        )

    events.append(
        {
            "tool_name": payload.tool_name,
            "timestamp": _parse_timestamp(payload.timestamp),
        }
    )

    risk_score = score_trajectory(events)
    decision = _decision_from_score(risk_score)

    if decision == "killswitch":
        update_session_status(payload.session_id, "killed")

    insert_tool_call(
        call_id=str(uuid4()),
        session_id=payload.session_id,
        tool_name=payload.tool_name,
        tool_args=payload.tool_args,
        risk_score=risk_score,
        decision=decision,
    )

    return {"decision": decision, "risk_score": risk_score}


@app.get("/events")
def events() -> list[dict[str, Any]]:
    return get_recent_events(50)
