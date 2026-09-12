"""Generate session-level training data for AgentGuard risk experiments."""

import csv
import json
import random
from pathlib import Path


TOOLS = [
    "search_docs",
    "read_file",
    "summarize",
    "send_email",
    "export_database",
    "access_credentials",
    "read_customer_data",
    "delete_records",
    "modify_permissions",
    "visit_url",
]
SENSITIVE_TOOLS = {
    "send_email",
    "export_database",
    "access_credentials",
    "read_customer_data",
    "delete_records",
    "modify_permissions",
}
RISKY_TOOLS = SENSITIVE_TOOLS - {"send_email"}
RANDOM_SEED = 42
ATTACK_PATTERNS = [
    ["search_docs", "read_customer_data", "access_credentials", "export_database"],
    ["read_customer_data", "access_credentials", "delete_records", "modify_permissions"],
    ["search_docs", "access_credentials", "export_database", "delete_records"],
]
FEATURE_NAMES = [
    "sensitive_tool_count",
    "max_time_gap",
    "min_time_gap",
    "avg_time_gap",
    "session_length",
    "unique_tool_count",
    "label",
]


def generate_training_sessions(n: int = 4000) -> list[dict[str, float | int]]:
    """Generate session features and save them to training_data.csv."""
    rng = random.Random(RANDOM_SEED)
    rows: list[dict[str, float | int]] = []
    sequences: list[dict[str, list[str] | int]] = []
    safe_tools = [tool for tool in TOOLS if tool not in SENSITIVE_TOOLS]
    noisy_sensitive_tools = [tool for tool in TOOLS if tool in SENSITIVE_TOOLS]
    session_types = (
        ["risky"] * round(n * 0.30)
        + ["noisy"] * round(n * 0.15)
        + ["decoy"] * round(n * 0.20)
    )
    session_types += ["benign"] * (n - len(session_types))
    rng.shuffle(session_types)
    seen_sequences: set[tuple[str, ...]] = set()

    for session_type in session_types:
        while True:
            tools = _sample_tools(
                session_type, rng, safe_tools, noisy_sensitive_tools
            )
            sequence = tuple(tools)
            if sequence not in seen_sequences:
                seen_sequences.add(sequence)
                break

        gaps = [rng.uniform(0.1, 6.0) for _ in range(len(tools) - 1)]
        avg_gap = sum(gaps) / len(gaps)
        label = int(any(_contains_pattern(tools, pattern) for pattern in ATTACK_PATTERNS))

        rows.append(
            {
                "sensitive_tool_count": sum(tool in SENSITIVE_TOOLS for tool in tools),
                "max_time_gap": max(gaps),
                "min_time_gap": min(gaps),
                "avg_time_gap": avg_gap,
                "session_length": len(tools),
                "unique_tool_count": len(set(tools)),
                "label": label,
            }
        )
        sequences.append({"tool_sequence": tools, "label": label})

    output_path = Path(__file__).resolve().parents[1] / "training_data.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FEATURE_NAMES)
        writer.writeheader()
        writer.writerows(rows)

    sequence_path = Path(__file__).resolve().parent / "training_sequences.json"
    with sequence_path.open("w", encoding="utf-8") as sequence_file:
        json.dump(sequences, sequence_file)

    return rows


def _contains_pattern(tools: list[str], pattern: list[str]) -> bool:
    """Return whether pattern appears as an ordered contiguous trajectory."""
    width = len(pattern)
    return any(tools[index : index + width] == pattern for index in range(len(tools) - width + 1))


def _sample_tools(
    session_type: str,
    rng: random.Random,
    safe_tools: list[str],
    noisy_sensitive_tools: list[str],
) -> list[str]:
    """Create one trajectory whose label is defined by ordered tool behavior."""
    if session_type == "risky":
        pattern = rng.choice(ATTACK_PATTERNS)
        extra_calls = rng.randint(0, 8 - len(pattern))
        prefix_length = rng.randint(0, extra_calls)
        return (
            [rng.choice(safe_tools) for _ in range(prefix_length)]
            + pattern
            + [rng.choice(safe_tools) for _ in range(extra_calls - prefix_length)]
        )
    if session_type == "noisy":
        session_length = rng.randint(2, 8)
        tools = [rng.choice(noisy_sensitive_tools)] + [
            rng.choice(safe_tools) for _ in range(session_length - 1)
        ]
        rng.shuffle(tools)
        return tools
    if session_type == "decoy":
        pattern = rng.choice(ATTACK_PATTERNS).copy()
        while pattern in ATTACK_PATTERNS:
            rng.shuffle(pattern)
        extra_calls = rng.randint(0, 8 - len(pattern))
        return pattern + [rng.choice(safe_tools) for _ in range(extra_calls)]

    session_length = rng.randint(2, 8)
    tools = [rng.choice(TOOLS) for _ in range(session_length)]
    while any(_contains_pattern(tools, pattern) for pattern in ATTACK_PATTERNS):
        tools = [rng.choice(safe_tools) for _ in range(session_length)]
    return tools


if __name__ == "__main__":
    generate_training_sessions()