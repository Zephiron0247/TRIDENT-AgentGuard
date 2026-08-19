def score_trajectory(events: list[dict]) -> float:
    """events = all tool calls so far in this session, oldest first."""
    score = 0.0
    sensitive_tools = {"export_database", "access_credentials", "send_email"}

    latest = events[-1]
    if latest["tool_name"] in sensitive_tools:
        score += 0.3

    # escalation: did this session touch 2+ sensitive tools?
    sensitive_hits = sum(1 for e in events if e["tool_name"] in sensitive_tools)
    if sensitive_hits >= 2:
        score += 0.3

    # speed: many calls in a short window looks automated/malicious
    if len(events) >= 4:
        span_seconds = (events[-1]["timestamp"] - events[0]["timestamp"]).total_seconds()
        if span_seconds < 10 and len(events) >= 4:
            score += 0.2

    return min(score, 1.0)