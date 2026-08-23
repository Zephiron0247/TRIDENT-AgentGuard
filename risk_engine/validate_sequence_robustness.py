"""Validate GRU trajectory generalization on unseen, non-template sequences."""

import json
import random
from pathlib import Path

import torch
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

try:
    from risk_engine.sequence_model_torch import (
        MODEL_PATH,
        THRESHOLD_PATH,
        VOCAB_PATH,
        SequenceRiskModel,
        _encode_tools,
    )
except ModuleNotFoundError:
    from sequence_model_torch import (
        MODEL_PATH,
        THRESHOLD_PATH,
        VOCAB_PATH,
        SequenceRiskModel,
        _encode_tools,
    )


MODULE_DIR = Path(__file__).resolve().parent
TRAINING_SEQUENCES_PATH = MODULE_DIR / "training_sequences.json"
RANDOM_SEED = 2026
SAFE_TOOLS = ["search_docs", "read_file", "summarize"]
SENSITIVE_TOOLS = [
    "send_email",
    "export_database",
    "access_credentials",
    "read_customer_data",
    "delete_records",
    "modify_permissions",
]
RISKY_TOOLS = [
    "export_database",
    "access_credentials",
    "read_customer_data",
    "delete_records",
    "modify_permissions",
]
TRAINING_ATTACK_TEMPLATES = [
    ["search_docs", "read_customer_data", "access_credentials", "export_database"],
    ["read_customer_data", "access_credentials", "delete_records", "modify_permissions"],
    ["search_docs", "access_credentials", "export_database", "delete_records"],
]


def _contains_template(sequence: list[str]) -> bool:
    for template in TRAINING_ATTACK_TEMPLATES:
        width = len(template)
        if any(sequence[index : index + width] == template for index in range(len(sequence) - width + 1)):
            return True
    return False


def _load_training_sequences() -> set[tuple[str, ...]]:
    records = json.loads(TRAINING_SEQUENCES_PATH.read_text(encoding="utf-8"))
    return {tuple(record["tool_sequence"]) for record in records}


def _append_unique(
    records: list[dict],
    seen: set[tuple[str, ...]],
    training_sequences: set[tuple[str, ...]],
    category: str,
    label: int,
    builder,
    count: int,
    rng: random.Random,
) -> None:
    added = 0
    while added < count:
        sequence = builder(rng)
        sequence_key = tuple(sequence)
        if (
            sequence_key in seen
            or sequence_key in training_sequences
            or _contains_template(sequence)
        ):
            continue
        seen.add(sequence_key)
        records.append({"sequence": sequence, "label": label, "category": category})
        added += 1


def _safe_sequence(rng: random.Random, length: int | None = None) -> list[str]:
    sequence_length = rng.randint(2, 8) if length is None else length
    return [rng.choice(SAFE_TOOLS) for _ in range(sequence_length)]


def _single_sensitive_sequence(rng: random.Random) -> list[str]:
    length = rng.randint(2, 8)
    sequence = [rng.choice(SENSITIVE_TOOLS)] + _safe_sequence(rng, length - 1)
    rng.shuffle(sequence)
    return sequence


def _reordered_decoy(rng: random.Random) -> list[str]:
    sequence = ["search_docs", "read_customer_data", "access_credentials", "export_database"]
    while sequence == TRAINING_ATTACK_TEMPLATES[0]:
        rng.shuffle(sequence)
    return sequence + _safe_sequence(rng, rng.randint(0, 4))


def _mixed_noisy_sequence(rng: random.Random) -> list[str]:
    sequence = [rng.choice(SAFE_TOOLS), "send_email", rng.choice(RISKY_TOOLS), rng.choice(SAFE_TOOLS)]
    rng.shuffle(sequence)
    return sequence


def _malicious_variation(rng: random.Random) -> list[str]:
    variations = [
        ["search_docs", "read_file", "read_customer_data", "access_credentials", "export_database"],
        ["summarize", "read_customer_data", "access_credentials", "export_database"],
        ["search_docs", "read_customer_data", "modify_permissions", "access_credentials", "export_database"],
        ["read_file", "access_credentials", "read_customer_data", "export_database"],
        ["summarize", "read_customer_data", "delete_records", "modify_permissions"],
    ]
    core = rng.choice(variations).copy()
    extra_calls = rng.randint(0, 8 - len(core))
    prefix_length = rng.randint(0, extra_calls)
    return (
        _safe_sequence(rng, prefix_length)
        + core
        + _safe_sequence(rng, extra_calls - prefix_length)
    )


def _malicious_prefixed_sequence(rng: random.Random) -> list[str]:
    prefixes = [["read_file"], ["summarize"], ["read_file", "summarize"]]
    suffix = _safe_sequence(rng, rng.randint(0, 2))
    return rng.choice(prefixes) + ["read_customer_data", "access_credentials", "export_database"] + suffix


def _short_malicious_sequence(rng: random.Random) -> list[str]:
    sequence = rng.sample(RISKY_TOOLS, 3)
    return sequence


def _long_malicious_sequence(rng: random.Random) -> list[str]:
    sequence = _malicious_variation(rng)
    return sequence + _safe_sequence(rng, 8 - len(sequence))


def build_validation_set() -> list[dict]:
    """Create unseen trajectories covering benign, noisy, and ordered-risk cases."""
    rng = random.Random(RANDOM_SEED)
    records: list[dict] = []
    training_sequences = _load_training_sequences()
    seen: set[tuple[str, ...]] = set()

    _append_unique(records, seen, training_sequences, "benign", 0, _safe_sequence, 30, rng)
    _append_unique(records, seen, training_sequences, "single_sensitive", 0, _single_sensitive_sequence, 30, rng)
    _append_unique(records, seen, training_sequences, "reordered_decoy", 0, _reordered_decoy, 30, rng)
    _append_unique(records, seen, training_sequences, "mixed_noisy", 0, _mixed_noisy_sequence, 30, rng)
    _append_unique(records, seen, training_sequences, "malicious_variation", 1, _malicious_variation, 30, rng)
    _append_unique(records, seen, training_sequences, "malicious_prefix", 1, _malicious_prefixed_sequence, 30, rng)
    _append_unique(records, seen, training_sequences, "short_benign", 0, lambda value: _safe_sequence(value, value.randint(2, 3)), 10, rng)
    _append_unique(records, seen, training_sequences, "short_malicious", 1, _short_malicious_sequence, 10, rng)
    _append_unique(records, seen, training_sequences, "long_benign", 0, lambda value: _safe_sequence(value, 8), 10, rng)
    _append_unique(records, seen, training_sequences, "long_malicious", 1, _long_malicious_sequence, 10, rng)
    assert all(2 <= len(record["sequence"]) <= 8 for record in records)
    return records


def _load_model() -> tuple[SequenceRiskModel, dict[str, int], float]:
    vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    threshold = float(json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["threshold"])
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = SequenceRiskModel(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab, threshold


def validate_robustness() -> dict[str, float | list[list[int]]]:
    """Evaluate the fixed trained model on a separate robustness suite."""
    torch.set_num_threads(1)
    records = build_validation_set()
    model, vocab, threshold = _load_model()
    inputs = torch.tensor(
        [_encode_tools(record["sequence"], vocab) for record in records], dtype=torch.long
    )
    expected = [record["label"] for record in records]
    with torch.no_grad():
        probabilities = torch.sigmoid(model(inputs)).squeeze(1).tolist()
    predicted = [int(probability >= threshold) for probability in probabilities]
    matrix = confusion_matrix(expected, predicted, labels=[0, 1])
    true_negative, false_positive = matrix[0]
    false_negative, true_positive = matrix[1]
    report = {
        "validation_sequences": len(records),
        "confusion_matrix": matrix.tolist(),
        "precision": precision_score(expected, predicted, zero_division=0),
        "recall": recall_score(expected, predicted, zero_division=0),
        "f1": f1_score(expected, predicted, zero_division=0),
        "false_positive_rate": false_positive / (false_positive + true_negative),
        "false_negative_rate": false_negative / (false_negative + true_positive),
    }

    print(f"Validation sequences: {report['validation_sequences']}")
    print(f"Threshold: {threshold:.4f}")
    print(f"Confusion matrix: {report['confusion_matrix']}")
    print(
        f"Precision: {report['precision']:.4f}, Recall: {report['recall']:.4f}, "
        f"F1: {report['f1']:.4f}"
    )
    print(
        f"False-positive rate: {report['false_positive_rate']:.4f}, "
        f"False-negative rate: {report['false_negative_rate']:.4f}"
    )
    print("Representative trajectories:")
    representatives = []
    represented_categories: set[str] = set()
    for record, probability, label in zip(records, probabilities, predicted):
        if record["category"] in represented_categories:
            continue
        representatives.append((record, probability, label))
        represented_categories.add(record["category"])
        if len(representatives) == 10:
            break
    for record, probability, label in representatives:
        print(
            f"{record['category']}: {' -> '.join(record['sequence'])} "
            f"-> {probability:.4f} -> predicted={label} -> expected={record['label']}"
        )

    benign_scores = [
        probability for record, probability in zip(records, probabilities) if record["label"] == 0
    ]
    malicious_scores = [
        probability for record, probability in zip(records, probabilities) if record["label"] == 1
    ]
    separates = min(malicious_scores) > max(benign_scores)
    print(f"Ordered malicious trajectories remain separated: {separates}")
    return report


if __name__ == "__main__":
    validate_robustness()
