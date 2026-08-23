from datetime import datetime
import json
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
from risk_engine.scorer_hybrid import score_trajectory


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
            "explanation": "The session was already terminated by the killswitch.",
        }

    if status is None:
        insert_session(payload.session_id, payload.agent_id, payload.stated_goal)

    prior_calls = get_session_calls(payload.session_id)

    stated_goal = payload.stated_goal
    events: list[dict[str, Any]] = []
    for call in prior_calls:
        tool_args = call.get("tool_args")
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except (ValueError, TypeError):
                tool_args = None
        events.append(
            {
                "tool_name": call.get("tool_name"),
                "tool_args": tool_args,
                "stated_goal": stated_goal,
                "timestamp": _parse_timestamp(call.get("called_at")),
            }
        )

    events.append(
        {
            "tool_name": payload.tool_name,
            "tool_args": payload.tool_args,
            "stated_goal": stated_goal,
            "timestamp": _parse_timestamp(payload.timestamp),
        }
    )

    result = score_trajectory(events, stated_goal=stated_goal)

    # Enforcement is authoritative: only the trusted signals (rule / ML /
    # anomaly) may trigger a killswitch. The corroborating sequence model is
    # deliberately excluded from the enforcement decision.
    risk_score = result["final_score"]
    trusted_score = result["trusted_score"]
    explanation = result["explanation"]
    decision = _decision_from_score(trusted_score)

    prior_events = events[:-1]
    previous_result = score_trajectory(prior_events, stated_goal=stated_goal) if prior_events else None
    previous_score = previous_result["final_score"] if previous_result else 0.0

    is_trigger_step = False
    trigger_reason = None
    score_jump = risk_score - previous_score
    crossed_threshold = previous_score < 0.7 and risk_score >= 0.7
    became_killswitch = decision == "killswitch"

    if score_jump > 0.2 or crossed_threshold or became_killswitch:
        is_trigger_step = True
        if became_killswitch:
            trigger_reason = f"Trajectory triggered killswitch at {payload.tool_name}"
        elif crossed_threshold:
            trigger_reason = f"Trajectory crossed the critical threshold at {payload.tool_name}"
        else:
            trigger_reason = (
                f"Risk increased from {previous_score:.2f} to {risk_score:.2f} "
                f"after {payload.tool_name}"
            )

    if decision == "killswitch":
        update_session_status(payload.session_id, "killed")

    insert_tool_call(
        call_id=str(uuid4()),
        session_id=payload.session_id,
        tool_name=payload.tool_name,
        tool_args=payload.tool_args,
        risk_score=risk_score,
        decision=decision,
        explanation=explanation,
        is_trigger_step=is_trigger_step,
        trigger_reason=trigger_reason,
    )

    return {
        "decision": decision,
        "risk_score": risk_score,
        "trusted_score": trusted_score,
        "dominant_signal": result["dominant_signal"],
        "rule_score": result["rule_score"],
        "ml_score": result["ml_score"],
        "anomaly_score": result["anomaly_score"],
        "sequence_score": result["sequence_score"],
        "url_score": result["url_score"],
        "explanation": explanation,
        "is_trigger_step": is_trigger_step,
        "trigger_reason": trigger_reason,
    }


@app.get("/events")
def events() -> list[dict[str, Any]]:
    return get_recent_events(50)
