"""Train AgentGuard risk models from the generated session dataset."""

import csv
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "training_data.csv"
FEATURE_COLUMNS = [
    "sensitive_tool_count",
    "max_time_gap",
    "min_time_gap",
    "avg_time_gap",
    "session_length",
    "unique_tool_count",
]

with DATA_PATH.open(newline="", encoding="utf-8") as data_file:
    records = list(csv.DictReader(data_file))

X = [[float(record[column]) for column in FEATURE_COLUMNS] for record in records]
y = [int(record["label"]) for record in records]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
    "random_forest": RandomForestClassifier(random_state=42),
}
metrics_report: dict[str, dict[str, float]] = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
    }
    metrics_report[name] = metrics
    print(
        f"{name}: accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )

isolation_forest = IsolationForest(random_state=42)
isolation_forest.fit(X_train)
joblib.dump(isolation_forest, ROOT_DIR / "anomaly_model.pkl")
print(f"Saved anomaly model to {ROOT_DIR / 'anomaly_model.pkl'}")

with (ROOT_DIR / "metrics_report.json").open("w", encoding="utf-8") as report_file:
    json.dump(metrics_report, report_file, indent=2)

winning_name = max(metrics_report, key=lambda name: metrics_report[name]["f1"])
winning_model = models[winning_name]
joblib.dump(winning_model, ROOT_DIR / "risk_model.pkl")

random_forest = models["random_forest"]
sorted_features = sorted(
    zip(FEATURE_COLUMNS, random_forest.feature_importances_),
    key=lambda feature: feature[1],
)
plt.figure(figsize=(8, 5))
plt.barh(
    [feature[0] for feature in sorted_features],
    [feature[1] for feature in sorted_features],
)
plt.xlabel("Feature importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(ROOT_DIR / "feature_importance.png")
plt.close()

winning_predictions = winning_model.predict(X_test)
matrix = confusion_matrix(y_test, winning_predictions, labels=[0, 1])
plt.figure(figsize=(5, 4))
plt.imshow(matrix, interpolation="nearest", cmap="Blues")
plt.title(f"Confusion Matrix: {winning_name}")
plt.colorbar()
plt.xticks([0, 1], ["Benign", "Risky"])
plt.yticks([0, 1], ["Benign", "Risky"])
plt.xlabel("Predicted label")
plt.ylabel("True label")
for row in range(matrix.shape[0]):
    for column in range(matrix.shape[1]):
        plt.text(column, row, matrix[row, column], ha="center", va="center")
plt.tight_layout()
plt.savefig(ROOT_DIR / "confusion_matrix.png")
plt.close()