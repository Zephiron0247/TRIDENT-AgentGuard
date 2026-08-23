from datetime import datetime, timezone
import re
from urllib.parse import urlparse


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


# ---------------------------------------------------------------------------
# URL risk signals (Phase 5 — domain generalization).
#
# These are deterministic, additive signals that capture suspicious browsing
# behavior (plaintext HTTP, raw IP hosts, known shorteners, domain/goal
# mismatch). They are INSPECTED from visit_url tool_args and combined into a
# single url_signal in [0, 1]. They never replace the existing tool-based
# weights above.
# ---------------------------------------------------------------------------

# Hosts commonly used for URL shortening / open-redirect abuse.
KNOWN_SHORTENER_HOSTS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
    "adf.ly",
    "shorte.st",
    "bc.vc",
    "cutt.ly",
    "rebrand.ly",
    "shorturl.at",
    "tiny.cc",
    "lnkd.in",
    "qr.ae",
    "trib.al",
    "tr.im",
    "clck.ru",
    "x.co",
}

# Cap for the standalone URL signal. Kept strictly below KILLSWITCH_FLOOR (0.9)
# so URL evidence alone can never produce a killswitch.
URL_SIGNAL_CAP = 0.85

_IPV4_RE = re.compile(r"^(?:\d{1,3}\.){3}\d{1,3}$")


def _url_from_args(tool_args: Any) -> str | None:
    """Best-effort extraction of a URL string from visit_url tool_args."""
    if isinstance(tool_args, str):
        return tool_args.strip() or None
    if isinstance(tool_args, dict):
        for key in ("url", "target", "href", "uri", "link"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Some callers pass the raw URL under "query"/"address".
        for key in ("query", "address", "site"):
            value = tool_args.get(key)
            if isinstance(value, str) and value.strip():
                candidate = value.strip()
                if candidate.startswith(("http://", "https://")) or "." in candidate:
                    return candidate
    return None


def _is_raw_ipv4(host: str) -> bool:
    if not host or not _IPV4_RE.match(host):
        return False
    return all(0 <= int(part) <= 255 for part in host.split("."))


def _is_raw_ipv6(host: str) -> bool:
    if not host:
        return False
    return ":" in host and all(
        ch in "0123456789abcdefABCDEF:." for ch in host
    )


def _host_from_url(url: str) -> str:
    if "://" not in url:
        url = "http://" + url
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    return (parsed.hostname or "").lower()


def _domain_mismatch_with_goal(host: str, stated_goal: str) -> bool:
    """True when the URL host looks unrelated to the agent's stated goal.

    This is deliberately narrow: it only fires for clearly-suspicious hosts
    (shorteners / raw IPs / HTTP) so benign research sites (wikipedia,
    government portals, news) are never flagged simply because the goal text
    does not mention them verbatim.
    """
    if not host or not stated_goal:
        return False
    goal_tokens = {
        token.lower()
        for token in re.split(r"[^a-z0-9]+", stated_goal)
        if len(token) >= 4 and token.isalpha()
    }
    if not goal_tokens:
        return False
    host_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", host)
        if len(token) >= 4 and token.isalpha()
    }
    # Benign generic research domains are explicitly exempted.
    benign_generic = {
        "wikipedia",
        "wikimedia",
        "github",
        "stackoverflow",
        "stackexchange",
        "medium",
        "wikihow",
        "gov",
        "washingtonpost",
        "nytimes",
        "reuters",
        "bbc",
        "arxiv",
    }
    if host_tokens and host_tokens.issubset(benign_generic):
        return False
    return not any(token in host_tokens for token in goal_tokens)


def url_signal_for_event(event: dict, stated_goal: str | None = None) -> tuple[float, str | None]:
    """Return (url_risk_contribution, reason_or_none) for a single event."""
    if event.get("tool_name") != "visit_url":
        return 0.0, None

    url = _url_from_args(event.get("tool_args"))
    if not url:
        return 0.0, None

    host = _host_from_url(url)
    scheme = ""
    if "://" in url:
        try:
            scheme = (urlparse(url).scheme or "").lower()
        except ValueError:
            scheme = ""

    contribution = 0.0
    reasons: list[str] = []

    # Plaintext transport.
    if scheme == "http":
        contribution += 0.2
        reasons.append("plaintext-http")

    # Raw IP hosts (no domain) — common in phishing / callback URLs.
    if _is_raw_ipv4(host) or _is_raw_ipv6(host):
        contribution += 0.3
        reasons.append(f"raw-ip-host:{host}")

    # Known URL-shortener / open-redirect host.
    if host in KNOWN_SHORTENER_HOSTS:
        contribution += 0.3
        reasons.append(f"url-shortener:{host}")

    # Domain/goal mismatch only amplifies already-suspicious URLs, so a
    # perfectly normal research site is never flagged on this basis alone.
    goal = stated_goal if stated_goal is not None else event.get("stated_goal")
    if contribution > 0.0 and _domain_mismatch_with_goal(host, goal):
        contribution += 0.15
        reasons.append(f"domain-goal-mismatch:{host}")

    if not reasons:
        return 0.0, None
    return min(contribution, URL_SIGNAL_CAP), ", ".join(reasons)


def url_risk_signal(events: list[dict], stated_goal: str | None = None) -> dict:
    """Aggregate URL risk across all visit_url calls in a session.

    Returns {"url_score": float, "url_available": bool, "url_reasons": list[str]}.
    """
    reasons: list[str] = []
    max_contribution = 0.0
    visits = 0

    for event in events:
        if event.get("tool_name") != "visit_url":
            continue
        visits += 1
        contribution, reason = url_signal_for_event(event, stated_goal=stated_goal)
        if reason:
            reasons.append(reason)
        max_contribution = max(max_contribution, contribution)

    return {
        "url_score": float(min(max_contribution, URL_SIGNAL_CAP)),
        "url_available": visits > 0,
        "url_reasons": reasons,
    }
