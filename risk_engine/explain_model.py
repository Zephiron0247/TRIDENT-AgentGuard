"""Offline SHAP explainability for the trained AgentGuard RandomForest model.

Loads the existing risk_model.pkl and produces:
  - risk_engine/shap_summary.png  (global SHAP summary / bar + beeswarm)
  - risk_engine/shap_examples.json (per-feature SHAP breakdowns for one benign
    and one malicious example trajectory)

This script does NOT modify any AgentGuard scoring behavior. It is a read-only
diagnostic over the already-trained artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

try:
    import shap
except ImportError:
    raise SystemExit(
        "The 'shap' package is required. Install it with: pip install shap"
    )

from risk_engine.scorer_hybrid import _extract_features, SENSITIVE_TOOLS


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "risk_model.pkl"
SUMMARY_PATH = Path(__file__).resolve().parent / "shap_summary.png"
EXAMPLES_PATH = Path(__file__).resolve().parent / "shap_examples.json"

FEATURE_COLUMNS = [
    "sensitive_tool_count",
    "max_time_gap",
    "min_time_gap",
    "avg_time_gap",
    "session_length",
    "unique_tool_count",
]

matplotlib.use("Agg")


def _feature_row(events: list[dict]) -> np.ndarray:
    """Build the exact six-feature row the RandomForest was trained on."""
    return np.array([_extract_features(events)], dtype=float)


def _example_events(name: str) -> list[dict]:
    """Hand-built example trajectories that mirror the demo sessions."""
    if name == "malicious":
        tools = [
            "search_docs",
            "read_customer_data",
            "access_credentials",
            "export_database",
        ]
    else:
        tools = ["search_docs", "read_file", "summarize", "send_email"]
    return [
        {
            "tool_name": tool,
            "timestamp": f"2026-01-01T00:00:{sec:02d}.000Z",
        }
        for sec, tool in enumerate(tools)
    ]


def _shap_to_example(explainer, feature_row: np.ndarray) -> dict:
    values = explainer.shap_values(feature_row)
    raw = explainer.shap_values(feature_row)
    array = np.asarray(raw)
    if array.ndim == 3:
        # shape (samples, features, classes): take malicious class (index 1).
        contributions = array[0, :, 1]
        base_expected = float(np.asarray(explainer.expected_value)[1])
    elif isinstance(raw, list):
        contributions = np.asarray(raw[1][0])
        base_expected = float(np.asarray(explainer.expected_value)[1])
    else:
        contributions = np.asarray(raw[0])
        base_expected = float(np.asarray(explainer.expected_value))
    prediction = float(base_expected + contributions.sum())
    rows = []
    for feature, value, contribution in zip(
        FEATURE_COLUMNS, feature_row[0], contributions
    ):
        rows.append(
            {
                "feature": feature,
                "value": float(value),
                "shap_contribution": float(contribution),
                "direction": "increases risk"
                if contribution > 0
                else ("decreases risk" if contribution < 0 else "neutral"),
            }
        )
    return {
        "feature_contributions": rows,
        "expected_value_base": float(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        "shap_log_odds_total": float(contributions.sum()),
        "risk_score_estimate_from_shap": float(prediction),
    }


def _save_summary_figure(
    explainer, X_background: np.ndarray
) -> list[tuple[str, float]]:
    shap_values = explainer.shap_values(X_background)
    # TreeExplainer may return a list [class_0, class_1] of 2D arrays, or a
    # single 3D array of shape (samples, features, classes). Normalize to the
    # malicious-class (index 1) 2D SHAP matrix.
    array = np.asarray(shap_values)
    if array.ndim == 3:
        shap_matrix = array[:, :, 1]
    elif isinstance(shap_values, list):
        shap_matrix = np.asarray(shap_values[1])
    else:
        shap_matrix = array

    plt.figure(figsize=(8, 5))
    mean_abs = np.abs(shap_matrix).mean(axis=0)
    feature_names = np.array(FEATURE_COLUMNS)
    order = np.argsort(mean_abs)
    plt.barh(
        [str(name) for name in feature_names[order]],
        mean_abs[order].tolist(),
        color="#4c72b0",
    )
    plt.xlabel("mean |SHAP value| (impact on malicious-log-odds)")
    plt.title("AgentGuard RandomForest - SHAP global feature impact")
    plt.tight_layout()
    plt.savefig(SUMMARY_PATH, dpi=120)
    plt.close("all")

    ranked = sorted(
        zip(FEATURE_COLUMNS, mean_abs.tolist()),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}", file=__import__("sys").stderr)
        return 1

    print(f"Loading model from {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)

    rng = np.random.default_rng(42)
    X_background = rng.random((200, len(FEATURE_COLUMNS)))
    # Put the background on a plausible scale for each feature.
    X_background[:, 0] *= 6  # sensitive_tool_count
    X_background[:, 1] *= 6  # max_time_gap
    X_background[:, 2] *= 6  # min_time_gap
    X_background[:, 3] *= 6  # avg_time_gap
    X_background[:, 4] *= 8  # session_length
    X_background[:, 5] *= 6  # unique_tool_count

    print("Building SHAP TreeExplainer")
    try:
        explainer = shap.TreeExplainer(model)
    except Exception as exc:
        print(f"TreeExplainer failed ({exc}); falling back to Explainer(wrapper)")
        proba = lambda X: model.predict_proba(X)[:, 1]
        explainer = shap.Explainer(proba, X_background)

    ranked = _save_summary_figure(explainer, X_background)
    print(f"Saved summary figure to {SUMMARY_PATH}")

    examples = {}
    top3 = []
    for name in ("benign", "malicious"):
        events = _example_events(name)
        row = _feature_row(events)
        example = _shap_to_example(explainer, row)
        example["trajectory"] = [event["tool_name"] for event in events]
        examples[name] = example
        print(f"\n{name.upper()} trajectory: {example['trajectory']}")
        for contribution in example["feature_contributions"]:
            print(
                f"  {contribution['feature']:>22} = {contribution['value']:6.2f}  "
                f"shap={contribution['shap_contribution']:+.3f}  "
                f"({contribution['direction']})"
            )
        if name == "malicious":
            top3 = ranked[:3]

    payload = {
        "disclaimer": (
            "SHAP explanation of the existing risk_model.pkl RandomForest. "
            "Does not change live AgentGuard scoring."
        ),
        "model_path": str(MODEL_PATH),
        "top_features_by_mean_abs_shap": [
            {"feature": name, "mean_abs_shap": float(value)}
            for name, value in ranked
        ],
        "examples": examples,
    }
    with EXAMPLES_PATH.open("w", encoding="utf-8") as examples_file:
        json.dump(payload, examples_file, indent=2)
    print(f"\nSaved example explanations to {EXAMPLES_PATH}")

    print("\nTop 3 most influential features (mean |SHAP|):")
    for rank, (name, value) in enumerate(top3, start=1):
        print(f"  {rank}. {name}  (mean |SHAP| = {value:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
