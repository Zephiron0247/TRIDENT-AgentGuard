from datetime import datetime, timezone


def _as_utc(timestamp: str | datetime) -> datetime:
    """Return ISO and database timestamps as timezone-aware UTC datetimes."""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def score_trajectory(events: list[dict]) -> float:
    """events = all tool calls so far in this session, oldest first."""
    score = 0.0
    sensitive_tools = {"export_database", "access_credentials", "send_email"}
    high_risk_tools = {"export_database", "access_credentials"}

    latest = events[-1]
    if latest["tool_name"] in sensitive_tools:
        score += 0.3
    if latest["tool_name"] in high_risk_tools:
        score += 0.4

    # escalation: did this session touch 2+ sensitive tools?
    sensitive_hits = sum(1 for e in events if e["tool_name"] in sensitive_tools)
    if sensitive_hits >= 2:
        score += 0.5

    # speed: many calls in a short window looks automated/malicious
    if len(events) >= 4 and sensitive_hits >= 2:
        timestamps = [_as_utc(event["timestamp"]) for event in events]
        span_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        if span_seconds < 10 and len(events) >= 4:
            score += 0.2

    return min(score, 1.0)
