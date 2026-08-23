"""Train and serve a compact GRU model for AgentGuard tool trajectories."""

import json
import random
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from risk_engine.scorer import _as_utc
except ModuleNotFoundError:
    from scorer import _as_utc


SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
EPOCHS = 30
RANDOM_STATE = 42
MODULE_DIR = Path(__file__).resolve().parent
SEQUENCES_PATH = MODULE_DIR / "training_sequences.json"
MODEL_PATH = MODULE_DIR / "sequence_model.pt"
VOCAB_PATH = MODULE_DIR / "vocab.json"
THRESHOLD_PATH = MODULE_DIR / "sequence_threshold.json"


class SequenceRiskModel(nn.Module):
    """Embedding and GRU classifier for padded tool-name sequences."""

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
        self.gru = nn.GRU(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True)
        self.output = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(sequences)
        _, hidden = self.gru(embedded)
        return self.output(hidden[-1])


def _set_reproducible_seed() -> None:
    random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _load_records() -> list[dict]:
    with SEQUENCES_PATH.open(encoding="utf-8") as sequence_file:
        return json.load(sequence_file)


def _build_vocab(records: list[dict]) -> dict[str, int]:
    tool_names = sorted(
        {tool for record in records for tool in record["tool_sequence"]}
    )
    return {"PAD": 0, "UNK": 1, **{tool: index + 2 for index, tool in enumerate(tool_names)}}


def _encode_tools(tools: list[str], vocab: dict[str, int]) -> list[int]:
    encoded = [vocab.get(tool, vocab["UNK"]) for tool in tools[:SEQUENCE_LENGTH]]
    return encoded + [vocab["PAD"]] * (SEQUENCE_LENGTH - len(encoded))


def _metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict[str, float]:
    predictions = [int(probability >= threshold) for probability in probabilities]
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def _select_validation_threshold(labels: list[int], probabilities: list[float]) -> float:
    candidates = sorted(set(probabilities))
    return max(
        candidates,
        key=lambda threshold: (
            _metrics(labels, probabilities, threshold)["f1"],
            -abs(threshold - 0.5),
        ),
    )


def _print_metrics(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name} - accuracy: {metrics['accuracy']:.4f}, "
        f"precision: {metrics['precision']:.4f}, "
        f"recall: {metrics['recall']:.4f}, f1: {metrics['f1']:.4f}"
    )


def train_sequence_model() -> dict[str, dict[str, float] | float]:
    """Train the CPU-friendly GRU and save its vocabulary and checkpoint."""
    _set_reproducible_seed()
    records = _load_records()
    if VOCAB_PATH.exists():
        vocab = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    else:
        vocab = _build_vocab(records)
        VOCAB_PATH.write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    sequences = [_encode_tools(record["tool_sequence"], vocab) for record in records]
    labels = [int(record["label"]) for record in records]
    train_sequences, remaining_sequences, train_labels, remaining_labels = train_test_split(
        sequences,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    validation_sequences, test_sequences, validation_labels, test_labels = train_test_split(
        remaining_sequences,
        remaining_labels,
        test_size=0.5,
        stratify=remaining_labels,
        random_state=RANDOM_STATE,
    )

    train_inputs = torch.tensor(train_sequences, dtype=torch.long)
    train_targets = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    validation_inputs = torch.tensor(validation_sequences, dtype=torch.long)
    test_inputs = torch.tensor(test_sequences, dtype=torch.long)
    model = SequenceRiskModel(len(vocab))
    positive_count = sum(train_labels)
    negative_count = len(train_labels) - positive_count
    positive_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.Adam(model.parameters())
    train_loader = DataLoader(
        TensorDataset(train_inputs, train_targets),
        batch_size=64,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS} - loss: {total_loss / len(train_inputs):.4f}")

    model.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(model(validation_inputs)).squeeze(1).tolist()
        test_probabilities = torch.sigmoid(model(test_inputs)).squeeze(1).tolist()
    selected_threshold = _select_validation_threshold(
        validation_labels, validation_probabilities
    )
    default_metrics = _metrics(test_labels, test_probabilities, 0.5)
    tuned_metrics = _metrics(test_labels, test_probabilities, selected_threshold)
    _print_metrics("Untouched test metrics (threshold 0.50)", default_metrics)
    _print_metrics(
        f"Untouched test metrics (validation threshold {selected_threshold:.4f})",
        tuned_metrics,
    )
    tuned_predictions = [int(probability >= selected_threshold) for probability in test_probabilities]
    tuned_confusion = confusion_matrix(test_labels, tuned_predictions, labels=[0, 1]).tolist()
    print(f"Untouched test confusion matrix (tuned threshold): {tuned_confusion}")
    torch.save({"state_dict": model.state_dict(), "vocab_size": len(vocab)}, MODEL_PATH)
    THRESHOLD_PATH.write_text(
        json.dumps({"threshold": selected_threshold}, indent=2), encoding="utf-8"
    )
    _load_inference_assets.cache_clear()
    return {
        "default_threshold": default_metrics,
        "tuned_threshold": tuned_metrics,
        "selected_threshold": selected_threshold,
        "tuned_confusion_matrix": tuned_confusion,
    }


@lru_cache(maxsize=1)
def _load_inference_assets() -> tuple[SequenceRiskModel, dict[str, int]]:
    with VOCAB_PATH.open(encoding="utf-8") as vocab_file:
        vocab = json.load(vocab_file)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model = SequenceRiskModel(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, vocab


def sequence_risk(events: list[dict]) -> float:
    """Return the trained GRU's risk probability for chronological tool events."""
    model, vocab = _load_inference_assets()
    ordered_events = sorted(events, key=lambda event: _as_utc(event["timestamp"]))
    tools = [event["tool_name"] for event in ordered_events]
    encoded = torch.tensor([_encode_tools(tools, vocab)], dtype=torch.long)
    with torch.no_grad():
        score = float(torch.sigmoid(model(encoded)).item())
    return max(0.0, min(1.0, score))


def run_sanity_checks() -> None:
    """Print probabilities for representative offline trajectory checks."""
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def events(tools: list[str]) -> list[dict]:
        return [
            {"tool_name": tool, "timestamp": base + timedelta(seconds=index)}
            for index, tool in enumerate(tools)
        ]

    checks = {
        "clearly benign": ["search_docs", "read_file", "summarize"],
        "legitimate single-sensitive": ["search_docs", "send_email", "summarize"],
        "known malicious ordered": [
            "search_docs",
            "read_customer_data",
            "access_credentials",
            "export_database",
        ],
    }
    for name, tools in checks.items():
        print(f"Sanity check ({name}): {sequence_risk(events(tools)):.4f}")


if __name__ == "__main__":
    train_sequence_model()
    run_sanity_checks()
