import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor

DATABASE_URL = os.environ.get("DATABASE_URL")

def get_connection():
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is missing.")
    return psycopg2.connect(DATABASE_URL)

def insert_session(session_id: str, agent_id: str, stated_goal: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions (session_id, agent_id, stated_goal, status)
                VALUES (%s, %s, %s, 'active')
                ON CONFLICT (session_id) DO NOTHING;
                """,
                (session_id, agent_id, stated_goal),
            )
        conn.commit()

def update_session_status(session_id: str, status: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE sessions
                SET status = %s
                WHERE session_id = %s;
                """,
                (status, session_id),
            )
        conn.commit()

def get_session_status(session_id: str) -> str | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM sessions WHERE session_id = %s;",
                (session_id,),
            )
            row = cur.fetchone()
            return row[0] if row else None

def get_session_calls(session_id: str) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT tool_name, tool_args, risk_score, decision, explanation, is_trigger_step, trigger_reason
                FROM tool_calls
                WHERE session_id = %s
                ORDER BY created_at ASC;
                """,
                (session_id,),
            )
            return cur.fetchall()

def insert_tool_call(
    call_id: str,
    session_id: str,
    tool_name: str,
    tool_args: any,
    risk_score: float,
    decision: str,
    explanation: str,
    is_trigger_step: bool,
    trigger_reason: str,
):
    if not isinstance(tool_args, str):
        tool_args_str = json.dumps(tool_args)
    else:
        tool_args_str = tool_args

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO tool_calls (
                    call_id, session_id, tool_name, tool_args, risk_score,
                    decision, explanation, is_trigger_step, trigger_reason
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                """,
                (
                    call_id,
                    session_id,
                    tool_name,
                    tool_args_str,
                    risk_score,
                    decision,
                    explanation,
                    is_trigger_step,
                    trigger_reason,
                ),
            )
        conn.commit()

def get_recent_events(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT 
                    t.session_id,
                    s.agent_id,
                    t.tool_name,
                    t.decision,
                    t.risk_score,
                    t.explanation,
                    t.is_trigger_step,
                    t.trigger_reason,
                    t.created_at
                FROM tool_calls t
                LEFT JOIN sessions s ON t.session_id = s.session_id
                ORDER BY t.created_at DESC
                LIMIT %s;
                """,
                (limit,),
            )
            return cur.fetchall()
