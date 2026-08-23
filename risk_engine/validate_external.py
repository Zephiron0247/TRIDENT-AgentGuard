"""OFFLINE external validation of AgentGuard's sequence-modeling methodology.

This script evaluates whether the same GRU sequence-classification approach used
by AgentGuard (Embedding -> GRU -> final-hidden -> Linear -> sigmoid) can
distinguish malicious from benign *ordered action sequences* on an INDEPENDENT
security dataset: the Oliveira malware-analysis API-call-sequence dataset.

IMPORTANT
---------
* This is EXTERNAL VALIDATION only. The Oliveira dataset is NOT AgentGuard agent
  telemetry and uses a completely separate vocabulary.
* Performance here does NOT prove direct generalization to AI-agent tool calls.
  It only checks whether the modeling approach carries signal on an unrelated
  security sequence task.
* Nothing in the live AgentGuard pipeline (scorer.py, scorer_hybrid.py, main.py,
  sequence_model.pt, vocab.json, demo behavior, Exasol, trigger-step logic) is
  modified by this script.

Usage:
    python -m risk_engine.validate_external <path_to_oliveira.csv>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Headless backend: no display server required.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

FPR_CONSTRAINT = 0.10
VAL_SPLIT = 0.2
ANALYSIS_POINTS = 500


MODULE_DIR = Path(__file__).resolve().parent

# Architecturally identical to risk_engine/sequence_model_torch.py so the
# comparison is methodological, not incidental.
SEQUENCE_LENGTH = 10
EMBEDDING_DIM = 8
HIDDEN_DIM = 16
EPOCHS = 25
BATCH_SIZE = 64
RANDOM_STATE = 42
LEARNING_RATE = 1e-3

METRICS_PATH = MODULE_DIR / "external_validation_metrics.json"
CONFUSION_PATH = MODULE_DIR / "external_confusion_matrix.png"
MODEL_PATH = MODULE_DIR / "external_validation_model.pt"
VOCAB_PATH = MODULE_DIR / "external_validation_vocab.json"

EVAL_BATCH_SIZE = 512


class SequenceClassifier(nn.Module):
    """Embedding -> GRU -> final hidden state -> Linear -> sigmoid.

    Mirrors AgentGuard's SequenceRiskModel so the external experiment is
    methodologically comparable.
    """

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
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# CSV inspection and column detection.
# ---------------------------------------------------------------------------

_LABEL_COLUMN_ALIASES = {
    "label",
    "labels",
    "class",
    "classes",
    "target",
    "category",
    "malware",
    "malicious",
    "family",
    "type",
    "y",
    "result",
    "verdict",
    "is_malware",
    "is_malicious",
}


def _detect_label_column(columns: list[str]) -> str | None:
    lowered = {column.lower().strip(): column for column in columns}
    for alias in _LABEL_COLUMN_ALIASES:
        if alias in lowered:
            return lowered[alias]
    return None


def _is_api_call_column(series: pd.Series) -> bool:
    """Heuristic: API-call sequence columns are numeric (token ids) or short
    strings, and most values are non-null."""
    if series.isna().mean() > 0.5:
        return False
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_string_dtype(series)


def inspect_csv(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    """Load the CSV, detect the label column and ordered API-call columns.

    Returns (dataframe, label_column, api_call_columns).
    """
    print(f"Loading {path}")
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read CSV: {exc}") from exc

    print(f"Detected shape: {df.shape}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)[:20]}")
    if len(df.columns) > 20:
        print(f"  ... and {len(df.columns) - 20} more")

    label_column = _detect_label_column(list(df.columns))
    if label_column is None:
        print(
            "No label column matched known names; defaulting to the first column "
            "as the label (common convention for Oliveira-style datasets)."
        )
        label_column = df.columns[0]
    print(f"Detected label column: {label_column!r}")

    api_columns = [
        column for column in df.columns if column != label_column
    ]
    # Drop clearly non-sequence columns (ids, hashes, hashes-as-text).
    api_columns = [
        column
        for column in api_columns
        if _is_api_call_column(df[column])
    ]
    print(f"Detected {len(api_columns)} ordered API-call columns (sequence length).")

    return df, label_column, api_columns


# ---------------------------------------------------------------------------
# Label mapping: reduce the Oliveira label set to binary (benign / malicious).
# ---------------------------------------------------------------------------

_BENIGN_ALIASES = {
    "benign",
    "ben",
    "clean",
    "normal",
    "legitimate",
    "legit",
    "safe",
    "whitelist",
    "white-list",
    "good",
    "0",
    "false",
    "no",
}


def _detect_benign_value(labels: pd.Series) -> str | None:
    unique = labels.dropna().unique()
    lowered = {str(value).lower().strip(): value for value in unique}
    for alias in _BENIGN_ALIASES:
        if alias in lowered:
            return lowered[alias]
    return None


def map_to_binary(labels: pd.Series) -> tuple[pd.Series, object | None]:
    """Map the dataset labels to 0 (benign) / 1 (malicious).

    Returns (binary_series, benign_original_value_or_None).
    """
    unique = labels.dropna().unique()
    if len(unique) == 2:
        benign = _detect_benign_value(labels)
        if benign is None:
            # Fall back: treat the lexicographically/smaller value as benign.
            sorted_values = sorted(unique, key=str)
            benign = sorted_values[0]
            print(
                f"Binary labels detected ({list(unique)}); no benign alias found. "
                f"Treating {benign!r} as benign (0)."
            )
        binary = (labels != benign).astype(int)
        return binary, benign

    print(f"Multiclass labels detected ({len(unique)} unique): {list(unique)[:20]}")
    benign = _detect_benign_value(labels)
    if benign is None:
        benign = sorted(unique, key=str)[0]
        print(
            f"No benign alias detected; treating smallest value {benign!r} as benign (0)."
        )
    binary = (labels != benign).astype(int)
    return binary, benign


# ---------------------------------------------------------------------------
# Vocabulary and encoding (fully separate from AgentGuard's vocab).
# ---------------------------------------------------------------------------

def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    tokens = sorted({token for sequence in sequences for token in sequence})
    return {"PAD": 0, "UNK": 1, **{token: index + 2 for index, token in enumerate(tokens)}}


def encode_sequences(
    sequences: list[list[str]], vocab: dict[str, int], sequence_length: int
) -> np.ndarray:
    encoded = np.full((len(sequences), sequence_length), vocab["PAD"], dtype=np.int64)
    for row, sequence in enumerate(sequences):
        truncated = sequence[:sequence_length]
        encoded_row = [vocab.get(token, vocab["UNK"]) for token in truncated]
        encoded[row, : len(encoded_row)] = encoded_row
    return encoded


# ---------------------------------------------------------------------------
# Training / evaluation.
# ---------------------------------------------------------------------------

def _to_tensor(array: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(array, dtype=dtype)


def train_model(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    vocab_size: int,
) -> tuple[SequenceClassifier, float]:
    model = SequenceClassifier(vocab_size)
    positive_count = int(train_targets.sum())
    negative_count = len(train_targets) - positive_count
    if positive_count == 0:
        raise ValueError("Training set contains no positive (malicious) samples.")
    pos_weight = torch.tensor(
        [negative_count / positive_count], dtype=torch.float32
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataset = torch.utils.data.TensorDataset(
        _to_tensor(train_inputs, torch.long),
        _to_tensor(train_targets, torch.float32).unsqueeze(1),
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_STATE),
    )

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for batch_inputs, batch_targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs), batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_inputs)
        if epoch == 1 or epoch % 5 == 0 or epoch == EPOCHS:
            print(f"Epoch {epoch}/{EPOCHS} - loss: {total_loss / len(train_inputs):.4f}")

    model.eval()
    with torch.no_grad():
        probabilities = (
            torch.sigmoid(model(_to_tensor(train_inputs, torch.long)))
            .squeeze(1)
            .tolist()
        )
    threshold = _select_threshold(train_targets.tolist(), probabilities)
    return model, threshold


def _select_threshold(labels: list[int], probabilities: list[float]) -> float:
    """Pick the probability threshold that maximizes F1 (tie-break: closest to
    0.5).

    This is an O(n log n) sweep that is mathematically equivalent to scoring
    every distinct probability value, but avoids the O(n^2) cost that the
    naive "unique thresholds x sklearn per candidate" loop incurs on large
    train splits (the source of the 10+ minute hang on the Oliveira data).
    """
    n = len(labels)
    if n == 0:
        return 0.5
    total_positive = sum(labels)
    # Sort (probability, label) descending so we can sweep the decision
    # boundary from the most-confident predictions downward.
    entries = sorted(
        zip(probabilities, labels), key=lambda pair: pair[0], reverse=True
    )
    true_positives = 0
    false_positives = 0
    best_threshold = entries[-1][0]
    best_key = None
    index = 0
    while index < n:
        probability = entries[index][0]
        # Absorb every sample at this probability before evaluating the key:
        # predicting positive when prob >= probability.
        while index < n and entries[index][0] == probability:
            if entries[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        predicted_positive = true_positives + false_positives
        precision = true_positives / predicted_positive if predicted_positive else 0.0
        recall = true_positives / total_positive if total_positive else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0.0
            else 0.0
        )
        key = (f1, -abs(probability - 0.5))
        if best_key is None or key > best_key:
            best_key = key
            best_threshold = probability
    return float(best_threshold)


def _predict_probabilities(
    model: SequenceClassifier, inputs: np.ndarray
) -> list[float]:
    """Batched inference returning sigmoid probabilities for every input."""
    model.eval()
    probabilities: list[float] = []
    total = len(inputs)
    with torch.no_grad():
        for start in range(0, total, EVAL_BATCH_SIZE):
            batch = inputs[start : start + EVAL_BATCH_SIZE]
            batch_probabilities = (
                torch.sigmoid(model(_to_tensor(batch, torch.long)))
                .squeeze(1)
                .tolist()
            )
            probabilities.extend(batch_probabilities)
    return probabilities


def _threshold_row(
    threshold: float,
    true_positives: int,
    false_positives: int,
    true_negatives: int,
    false_negatives: int,
) -> dict:
    predicted_positive = true_positives + false_positives
    actual_positive = true_positives + false_negatives
    actual_negative = true_negatives + false_positives
    precision = true_positives / predicted_positive if predicted_positive else 0.0
    recall = true_positives / actual_positive if actual_positive else 0.0
    f1 = (
        (2 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(false_positives / actual_negative if actual_negative else 0.0),
        "fnr": float(false_negatives / actual_positive if actual_positive else 0.0),
        "specificity": float(
            true_negatives / actual_negative if actual_negative else 0.0
        ),
        "balanced_accuracy": float(
            0.5
            * (
                (recall if actual_positive else 0.0)
                + (true_negatives / actual_negative if actual_negative else 0.0)
            )
        ),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives),
    }


def sweep_thresholds(
    labels: list[int], probabilities: list[float], points: int = ANALYSIS_POINTS
) -> list[dict]:
    """Dense threshold analysis over a sorted probability sweep.

    Evaluates every distinct probability as a candidate threshold (the exact
    operating points of the classifier), then subsamples to ``points`` rows for
    a compact report while preserving the full extrema.
    """
    n = len(labels)
    if n == 0:
        return []
    entries = sorted(
        zip(probabilities, labels), key=lambda pair: pair[0], reverse=True
    )
    total_positive = sum(labels)
    total_negative = n - total_positive

    true_positives = 0
    false_positives = 0
    rows: list[dict] = []
    index = 0
    while index < n:
        probability = entries[index][0]
        while index < n and entries[index][0] == probability:
            if entries[index][1] == 1:
                true_positives += 1
            else:
                false_positives += 1
            index += 1
        false_negatives = total_positive - true_positives
        true_negatives = total_negative - false_positives
        rows.append(
            _threshold_row(
                probability,
                true_positives,
                false_positives,
                true_negatives,
                false_negatives,
            )
        )

    if len(rows) <= points:
        return rows
    # Subsample evenly across the sorted rows, always keeping first and last.
    step = len(rows) / points
    indices = {0, len(rows) - 1}
    for i in range(1, points - 1):
        indices.add(int(round(i * step)))
    return [rows[i] for i in sorted(indices) if i < len(rows)]


def select_threshold_fpr_constrained(
    labels: list[int], probabilities: list[float], fpr_max: float = FPR_CONSTRAINT
) -> tuple[float, bool, dict]:
    """Choose an operating threshold under a false-positive constraint.

    Policy:
    1. Restrict to thresholds whose FPR <= fpr_max.
    2. Among those, pick the threshold with the best F1 (tie-break: highest
       recall, then threshold closest to 0.5).
    3. If no threshold satisfies the constraint, fall back to the threshold with
       the lowest FPR and report that the constraint is unattainable.

    Returns (threshold, constraint_satisfied, chosen_row).
    """
    rows = sweep_thresholds(labels, probabilities)
    if not rows:
        return 0.5, False, _threshold_row(0.5, 0, 0, 0, 0)

    feasible = [row for row in rows if row["fpr"] <= fpr_max]
    if feasible:
        best = max(
            feasible,
            key=lambda row: (row["f1"], row["recall"], -abs(row["threshold"] - 0.5)),
        )
        return best["threshold"], True, best

    best = min(rows, key=lambda row: (row["fpr"], -row["recall"]))
    return best["threshold"], False, best


def _metrics(labels: list[int], probabilities: list[float], threshold: float) -> dict:
    predictions = [int(probability >= threshold) for probability in probabilities]
    actual_positive = sum(labels)
    actual_negative = len(labels) - actual_positive
    predicted_positive = sum(predictions)
    true_positives = sum(
        1 for label, prediction in zip(labels, predictions) if label == 1 and prediction == 1
    )
    false_positives = predicted_positive - true_positives
    false_negatives = actual_positive - true_positives
    true_negatives = actual_negative - false_positives
    precision = true_positives / predicted_positive if predicted_positive else 0.0
    recall = true_positives / actual_positive if actual_positive else 0.0
    specificity = true_negatives / actual_negative if actual_negative else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "specificity": float(specificity),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "fpr": float(false_positives / actual_negative if actual_negative else 0.0),
        "fnr": float(false_negatives / actual_positive if actual_positive else 0.0),
        "roc_auc": (
            float(roc_auc_score(labels, probabilities)) if len(set(labels)) == 2 else None
        ),
        "pr_auc": (
            float(average_precision_score(labels, probabilities))
            if len(set(labels)) == 2
            else None
        ),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "true_negatives": int(true_negatives),
        "false_negatives": int(false_negatives),
    }


def evaluate_model(
    model: SequenceClassifier,
    inputs: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    label_name: str,
) -> dict:
    """Batched, chunked inference with progress output and timing."""
    model.eval()
    total = len(inputs)
    probabilities = []
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, total, EVAL_BATCH_SIZE):
            batch = inputs[start : start + EVAL_BATCH_SIZE]
            batch_probabilities = (
                torch.sigmoid(model(_to_tensor(batch, torch.long)))
                .squeeze(1)
                .tolist()
            )
            probabilities.extend(batch_probabilities)
            end = min(start + EVAL_BATCH_SIZE, total)
            elapsed = time.perf_counter() - started
            print(
                f"  [eval] {end}/{total} sequences scored ({elapsed:.1f}s)",
                end="\r",
                flush=True,
            )
    inference_seconds = time.perf_counter() - started
    print(
        f"\n  inference complete: {total} sequences in {inference_seconds:.2f}s"
    )

    predictions = [int(probability >= threshold) for probability in probabilities]
    metrics = _metrics(targets.tolist(), probabilities, threshold)
    matrix = np.array(
        [
            [metrics["true_negatives"], metrics["false_positives"]],
            [metrics["false_negatives"], metrics["true_positives"]],
        ]
    )

    print(f"\n{label_name} metrics (threshold {threshold:.4f}):")
    print(f"  accuracy:          {metrics['accuracy']:.4f}")
    print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  precision:         {metrics['precision']:.4f}")
    print(f"  recall:            {metrics['recall']:.4f}")
    print(f"  f1:                {metrics['f1']:.4f}")
    print(f"  specificity:       {metrics['specificity']:.4f}")
    print(f"  fpr:               {metrics['fpr']:.4f}")
    print(f"  fnr:               {metrics['fnr']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  roc_auc:           {metrics['roc_auc']:.4f}")
    if metrics["pr_auc"] is not None:
        print(f"  pr_auc:            {metrics['pr_auc']:.4f}")
    print(f"  confusion matrix (benign=0, malicious=1):")
    print(
        f"    [[TN={metrics['true_negatives']}, FP={metrics['false_positives']}],"
    )
    print(
        f"     [FN={metrics['false_negatives']}, TP={metrics['true_positives']}]]"
    )
    metrics["label"] = label_name
    metrics["inference_seconds"] = float(inference_seconds)
    return metrics, matrix


def save_confusion_matrix_figure(matrix: np.ndarray) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.set_title("External validation confusion matrix\n(benign=0, malicious=1)")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["benign", "malicious"])
    axis.set_yticklabels(["benign", "malicious"])
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > matrix.max() / 2 else "black",
            )
    figure.tight_layout()
    figure.savefig(CONFUSION_PATH, dpi=120)
    plt.close(figure)
    print(f"Saved confusion matrix image to {CONFUSION_PATH}")


def _print_threshold_analysis_table(rows: list[dict], chosen: dict) -> None:
    print("\nThreshold analysis (validation set, FPR descending):")
    print(
        f"  {'threshold':>9} {'prec':>7} {'recall':>7} {'f1':>7} "
        f"{'spec':>7} {'fpr':>7} {'fnr':>7}"
    )
    chosen_threshold = chosen["threshold"]
    printed: set[float] = set()
    for index, row in enumerate(rows):
        is_extreme = index == 0 or index == len(rows) - 1
        is_feasible = row["fpr"] <= FPR_CONSTRAINT
        is_chosen = row["threshold"] == chosen_threshold
        if not (is_chosen or is_feasible or is_extreme):
            continue
        marker = " <-- chosen" if is_chosen else ""
        key = round(row["threshold"], 8)
        if key in printed:
            continue
        printed.add(key)
        print(
            f"  {row['threshold']:9.4f} {row['precision']:7.4f} "
            f"{row['recall']:7.4f} {row['f1']:7.4f} "
            f"{row['specificity']:7.4f} {row['fpr']:7.4f} "
            f"{row['fnr']:7.4f}{marker}"
        )


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    print("=" * 72)
    print("EXTERNAL VALIDATION (offline experiment)")
    print("Dataset : Oliveira malware-analysis API-call sequences")
    print("This dataset is NOT AgentGuard agent telemetry.")
    print("Results do NOT prove direct generalization to AI-agent tool calls.")
    print("=" * 72)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path",
        help="Local path to the downloaded Oliveira API-call-sequence CSV.",
    )
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_path).resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}", file=sys.stderr)
        return 1

    _set_reproducible_seed()

    df, label_column, api_columns = inspect_csv(csv_path)

    if not api_columns:
        print("ERROR: no ordered API-call columns detected.", file=sys.stderr)
        return 1

    original_row_count = len(df)
    df = df.dropna(subset=[label_column]).copy()
    rows_discarded = original_row_count - len(df)
    print(f"Rows discarded (missing label): {rows_discarded} of {original_row_count}")

    labels_raw = df[label_column].astype(str).str.strip()
    binary_labels, benign_value = map_to_binary(labels_raw)
    print(f"Mapped benign value: {benign_value!r} -> 0 (benign), else -> 1 (malicious)")
    print(
        f"Class distribution: 0={(binary_labels == 0).sum()}, "
        f"1={(binary_labels == 1).sum()}"
    )

    # Build ordered token sequences from the API-call columns (column order).
    def _row_to_sequence(row: pd.Series) -> list[str]:
        return [str(value) for value in row[api_columns].tolist() if pd.notna(value)]

    sequences = df.apply(_row_to_sequence, axis=1).tolist()
    lengths = [len(sequence) for sequence in sequences]
    print(
        f"Sequence length stats: min={min(lengths)}, max={max(lengths)}, "
        f"mean={np.mean(lengths):.1f}; model SEQUENCE_LENGTH={SEQUENCE_LENGTH}"
    )

    # Drop rows that produced an empty sequence after cleaning.
    mask = [len(sequence) > 0 for sequence in sequences]
    if not all(mask):
        dropped = len(mask) - sum(mask)
        sequences = [sequence for sequence, keep in zip(sequences, mask) if keep]
        binary_labels = binary_labels[mask].reset_index(drop=True)
        rows_discarded += dropped
        print(f"Additional rows dropped (empty sequence): {dropped}")
    print(f"Usable rows: {len(sequences)} (total discarded: {rows_discarded})")

    vocab = build_vocab(sequences)
    with VOCAB_PATH.open("w", encoding="utf-8") as vocab_file:
        json.dump(vocab, vocab_file, indent=2)
    print(f"Built external vocabulary: {len(vocab) - 2} tokens (+PAD, +UNK)")

    encoded = encode_sequences(sequences, vocab, SEQUENCE_LENGTH)
    targets = binary_labels.to_numpy().astype(np.float32)

    # Hold out a test set that is NEVER touched until the final report.
    (
        train_inputs,
        test_inputs,
        train_targets,
        test_targets,
    ) = train_test_split(
        encoded,
        targets,
        test_size=0.2,
        stratify=targets,
        random_state=RANDOM_STATE,
    )
    # Further split the training portion into subtrain (model fitting) and a
    # validation set used ONLY for threshold selection. A distinct seed keeps
    # the splits deterministic but independent of the test split.
    (
        subtrain_inputs,
        val_inputs,
        subtrain_targets,
        val_targets,
    ) = train_test_split(
        train_inputs,
        train_targets,
        test_size=VAL_SPLIT,
        stratify=train_targets,
        random_state=RANDOM_STATE + 1,
    )
    print(
        f"Stratified split -> subtrain={len(subtrain_inputs)}, "
        f"val={len(val_inputs)}, test={len(test_inputs)} (seed {RANDOM_STATE})"
    )

    model, _ = train_model(subtrain_inputs, subtrain_targets, len(vocab))
    torch.save(
        {"state_dict": model.state_dict(), "vocab_size": len(vocab)},
        MODEL_PATH,
    )
    print(f"Saved external validation model to {MODEL_PATH}")

    print("\nScoring the validation set (threshold selection)...")
    val_probabilities = _predict_probabilities(model, val_inputs)
    threshold, constraint_satisfied, chosen_row = select_threshold_fpr_constrained(
        val_targets.tolist(), val_probabilities, fpr_max=FPR_CONSTRAINT
    )
    val_analysis = sweep_thresholds(val_targets.tolist(), val_probabilities)
    _print_threshold_analysis_table(val_analysis, chosen_row)
    if constraint_satisfied:
        print(
            f"\nFPR<={FPR_CONSTRAINT:.0%} constraint satisfied on validation set "
            f"(chosen threshold={threshold:.4f})."
        )
    else:
        print(
            f"\nWARNING: no validation threshold achieves FPR<={FPR_CONSTRAINT:.0%}. "
            f"Falling back to lowest achievable FPR={chosen_row['fpr']:.4f} "
            f"(threshold={threshold:.4f})."
        )

    print("\nEvaluating ONCE on the untouched test set...")
    test_metrics, matrix = evaluate_model(
        model, test_inputs, test_targets, threshold, "Untouched test set"
    )
    save_confusion_matrix_figure(matrix)

    metrics_payload = {
        "experiment": "external_validation_oliveira_api_call_sequences",
        "disclaimer": (
            "External validation only. This dataset is not AgentGuard agent "
            "telemetry and does not prove direct generalization to AI-agent tool calls."
        ),
        "dataset_source": "ang3loliveira/malware-analysis-datasets-api-call-sequences",
        "csv_path": str(csv_path),
        "original_row_count": original_row_count,
        "rows_discarded": rows_discarded,
        "usable_rows": len(sequences),
        "sequence_length": SEQUENCE_LENGTH,
        "vocab_size": len(vocab),
        "benign_label_value": str(benign_value) if benign_value is not None else None,
        "subtrain_size": len(subtrain_inputs),
        "val_size": len(val_inputs),
        "test_size": len(test_inputs),
        "epochs": EPOCHS,
        "threshold_policy": {
            "fpr_constraint": FPR_CONSTRAINT,
            "constraint_satisfied": constraint_satisfied,
            "selection_method": "best F1 among thresholds with FPR<=constraint",
        },
        "chosen_threshold": float(threshold),
        "threshold_analysis_validation": val_analysis,
        "metrics": test_metrics,
        "truncation_note": (
            "Only the first SEQUENCE_LENGTH=10 tokens of each 101-call sequence "
            "were used. Using all 101 calls is likely to be materially useful: "
            "malware behavioral signatures often span long ordered call sequences, "
            "so the GRU currently sees ~10% of each trace and must discard "
            "mid/late-stage behavior. A future run should increase SEQUENCE_LENGTH "
            "to the full 101 (which also raises the longest RNN unroll and may "
            "benefit from a larger HIDDEN_DIM) before drawing strong conclusions."
        ),
    }
    with METRICS_PATH.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics_payload, metrics_file, indent=2)
    print(f"Saved metrics to {METRICS_PATH}")

    print("\n" + "=" * 72)
    print("Done. Remember: this is offline external validation only.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
