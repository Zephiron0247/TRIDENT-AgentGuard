import json
import os
from typing import Any

import pyexasol

_DSN = "127.0.0.1:8563"
_USER = "sys"
_SCHEMA = "AGENTGUARD"
_connection = None


def get_connection():
    """Return a single reusable pyexasol connection for this process."""
    global _connection

    if _connection is None:
        _connection = pyexasol.connect(
            dsn=_DSN,
            user=_USER,
            password=os.environ["EXAPW"],
            encryption=True,
            websocket_sslopt={"cert_reqs": 0},
            schema=_SCHEMA,
        )

    return _connection


def insert_session(session_id: str, agent_id: str, stated_goal: str) -> None:
    """Insert a new active session. Skip if session_id already exists."""
    c = get_connection()

    c.execute(
        """
        MERGE INTO sessions t
        USING (
            SELECT
                {session_id} AS session_id,
                {agent_id} AS agent_id,
                {stated_goal} AS stated_goal
        ) s
        ON t.session_id = s.session_id
        WHEN NOT MATCHED THEN
            INSERT (session_id, agent_id, stated_goal, status, started_at)
            VALUES (s.session_id, s.agent_id, s.stated_goal, 'active', CURRENT_TIMESTAMP)
        """,
        {
            "session_id": session_id,
            "agent_id": agent_id,
            "stated_goal": stated_goal,
        },
    )


def insert_tool_call(
    call_id: str,
    session_id: str,
    tool_name: str,
    tool_args: Any,
    risk_score: float,
    decision: str,
) -> None:
    """Insert one tool call row; tool_args is persisted as a JSON string."""
    c = get_connection()
    tool_args_json = json.dumps(tool_args, ensure_ascii=True, default=str)

    c.execute(
        """
        INSERT INTO tool_calls (
            call_id,
            session_id,
            tool_name,
            tool_args,
            risk_score,
            decision,
            called_at
        )
        VALUES ({call_id}, {session_id}, {tool_name}, {tool_args}, {risk_score}, {decision}, CURRENT_TIMESTAMP)
        """,
        {
            "call_id": call_id,
            "session_id": session_id,
            "tool_name": tool_name,
            "tool_args": tool_args_json,
            "risk_score": risk_score,
            "decision": decision,
        },
    )


def update_session_status(session_id: str, status: str) -> None:
    """Update session status and stamp ended_at when transitioning to killed."""
    c = get_connection()

    c.execute(
        """
        UPDATE sessions
        SET
            status = {status},
            ended_at = CASE
                WHEN LOWER({status_lower}) = 'killed' THEN CURRENT_TIMESTAMP
                ELSE ended_at
            END
        WHERE session_id = {session_id}
        """,
        {
            "status": status,
            "status_lower": status,
            "session_id": session_id,
        },
    )


def get_session_calls(session_id: str) -> list[dict[str, Any]]:
    """Return all tool calls for a session ordered by called_at."""
    c = get_connection()
    stmt = c.execute(
        """
        SELECT
            call_id,
            session_id,
            tool_name,
            tool_args,
            risk_score,
            decision,
            entailment_flag,
            called_at
        FROM tool_calls
        WHERE session_id = {session_id}
        ORDER BY called_at
        """,
        {"session_id": session_id},
    )

    rows = stmt.fetchall()
    keys = [
        "call_id",
        "session_id",
        "tool_name",
        "tool_args",
        "risk_score",
        "decision",
        "entailment_flag",
        "called_at",
    ]
    return [dict(zip(keys, row)) for row in rows]


def get_recent_events(limit: int = 50) -> list[dict[str, Any]]:
    """Return newest tool calls joined with session metadata for dashboard polling."""
    c = get_connection()
    stmt = c.execute(
        """
        SELECT
            t.call_id,
            t.session_id,
            s.agent_id,
            s.stated_goal,
            t.tool_name,
            t.tool_args,
            t.risk_score,
            t.decision,
            t.entailment_flag,
            t.called_at
        FROM tool_calls t
        JOIN sessions s
            ON s.session_id = t.session_id
        ORDER BY t.called_at DESC
        LIMIT {limit}
        """,
        {"limit": limit},
    )

    rows = stmt.fetchall()
    keys = [
        "call_id",
        "session_id",
        "agent_id",
        "stated_goal",
        "tool_name",
        "tool_args",
        "risk_score",
        "decision",
        "entailment_flag",
        "called_at",
    ]
    return [dict(zip(keys, row)) for row in rows]
