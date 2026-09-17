markdown

# 🔱 TRIDENT — AgentGuard

**Runtime security for autonomous AI agents.** TRIDENT sits between an agent and the tools it calls, watches the _trajectory_ of tool calls (not just single calls in isolation), scores risk continuously, explains its decisions, pinpoints the exact step where risk escalated, and enforces policy in real time.

🏆 **5th place / 120 teams** — Exasol AI Build Challenge 2026, Autonomous Agents Track (VIT Chennai × Exasol)
✅ **Status:** Complete, tested, and demonstrated end-to-end

**Team VNT — VIT Chennai**

| Material      | Link                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| 🎥 Demo Video | [Google Drive](https://drive.google.com/drive/folders/1Q4CtNy--9riIL0AgT6_fWlCK6XDT2q2j?usp=sharing) |
| 📊 Pitch Deck | [Google Drive](https://drive.google.com/drive/folders/1Q4CtNy--9riIL0AgT6_fWlCK6XDT2q2j?usp=sharing) |

---

## The Problem

Agents chain tool calls autonomously: `search_docs → read_customer_data → access_credentials → export_database`. No single call here looks obviously malicious — the danger is in the _sequence_. A per-call filter can't tell this apart from a harmless trajectory using the same tools in a different order.

## The Solution

TRIDENT tracks the whole trajectory, scores it continuously with multiple independent signals, explains _why_ risk rose, localizes the _exact step_ that caused it, and kills the session when policy thresholds are crossed.

Observe trajectory → score continuously → explain the change → localize the trigger step → enforce policy

---

## Architecture

Agent → FastAPI /tool-call → Trajectory Engine
├─ Deterministic Rules
├─ Random Forest
├─ GRU (sequence, advisory)
└─ URL Risk Heuristics
→ Hybrid Risk Score + Explanation
→ Policy Engine (ALLOW/FLAG/BLOCK/KILLSWITCH)
→ Trigger Localization + Exasol persistence
→ Live Dashboard

---

## Detection Layers

| Layer                          | Role                                                                                                                                                                                                            |
| ------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Deterministic Rules**        | Trusted safety floor — flags sensitive-tool combos, escalation, timing patterns. Fast, explainable, always on.                                                                                                  |
| **Random Forest**              | 6 trajectory features: `sensitive_tool_count`, `max_time_gap`, `min_time_gap`, `avg_time_gap`, `session_length`, `unique_tool_count`                                                                            |
| **GRU (sequence model)**       | `Embedding(vocab,8) → GRU(8,16) → Linear(16,1) → Sigmoid`. Order-aware — tells a dangerous sequence apart from the same tools in a harmless order. **Advisory only**, can't independently trigger a killswitch. |
| **URL Heuristics**             | For browsing agents — HTTPS, raw-IP hosts, goal/domain mismatch. Capped signal, can't independently trigger a killswitch.                                                                                       |
| **Anomaly (Isolation Forest)** | Exists in the codebase, computed for analysis — **excluded from trusted enforcement** because it saturated near 1.0 even on benign trajectories. Kept as diagnostic only.                                       |

**Trusted score (enforcement):**

```python
trusted_score = max(rule_score, ml_score, url_score)   # anomaly excluded
```

GRU only adds `+0.15 × sequence_score` when `sequence_score ≥ 0.5` **and** `trusted_score ≥ 0.4`. If `trusted_score < 0.90`, the final score is capped below the killswitch floor — so GRU, URL, or anomaly alone can never trigger a killswitch. Only the trusted floor can.

**Current model performance (synthetic test data):**

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 79.37% |
| Precision | 64.15% |
| Recall    | 70.83% |
| F1        | 67.33% |

**GRU live behavior:** benign ≈ `0.0004` · malicious ≈ `0.9991`

---

## Risk Policy

| Score         | Decision         | Meaning            |
| ------------- | ---------------- | ------------------ |
| `< 0.40`      | ✅ ALLOW         | Normal             |
| `0.40 – 0.70` | 🟡 ALLOW_FLAGGED | Allowed, logged    |
| `0.70 – 0.90` | 🟠 BLOCK         | This call stopped  |
| `> 0.90`      | 🔴 KILLSWITCH    | Session terminated |

---

## Trigger-Step Localization

TRIDENT identifies exactly which call caused an escalation, not just that a session is malicious:

search_docs → ALLOW 0.00
read_customer_data → ALLOW 0.03
access_credentials → BLOCK 0.85 ⚠ TRIGGER
export_database → KILLSWITCH 1.00 ⚠ TRIGGER

Exposed per-event as `is_trigger_step` / `trigger_reason`, shown live in the dashboard's Trigger Localization panel.

---

## Explainability

Every decision reports `rule_score`, `ml_score`, `url_score`, `sequence_score` (with corroboration state), `dominant_signal`, and a plain-language explanation.

SHAP (offline, on the Random Forest) ranks global risk drivers: `avg_time_gap` → `sensitive_tool_count` → `max_time_gap`. This is offline/global analysis — not a live per-event SHAP API.

---

## External Validation

The GRU's sequence-modeling _approach_ was independently validated on the **Oliveira malware API-call benchmark** (43,876 sequences) — separate from AgentGuard's own data:

| Metric    | Value  |
| --------- | ------ |
| ROC-AUC   | 0.7725 |
| PR-AUC    | 0.9914 |
| Precision | 0.9932 |
| Recall    | 0.3953 |
| F1        | 0.5656 |
| FPR       | 10.65% |

This supports the sequence-modeling methodology on real-world data; it does **not** by itself prove generalization to AI-agent trajectories.

---

## Database (Exasol Personal)

Schema `AGENTGUARD`, connected at `127.0.0.1:8563`:

| Table        | Purpose                                                             |
| ------------ | ------------------------------------------------------------------- |
| `SESSIONS`   | One row per agent session — goal, status, lifecycle                 |
| `TOOL_CALLS` | Every intercepted call — score, decision, explanation, trigger info |
| `INCIDENTS`  | High-risk / terminated sessions for review                          |

An Exasol-native rule UDF (`AGENTGUARD.RULE_RISK_SCORE`) is implemented and validated inside Exasol, but live enforcement currently runs through the application-side rule engine.

---

## Tech Stack

`Python` · `FastAPI` · `Uvicorn` · `PyExasol` · `Exasol Personal (Docker)` · `scikit-learn` · `PyTorch` · `SHAP` · `HTML/CSS/JS`

---

## Project Structure

agentguard/
├── backend/
│ ├── main.py # FastAPI: /tool-call, /events
│ ├── exasol_client.py
│ ├── run_schema.py
│ └── schema.sql
├── risk_engine/
│ ├── scorer.py # deterministic rules
│ ├── scorer_hybrid.py # combines all signals, trusted score
│ ├── sequence_model_torch.py # GRU
│ ├── generate_dataset.py
│ ├── train_models.py
│ ├── validate_external.py # Oliveira benchmark
│ ├── explain_model.py # SHAP
│ └── (risk_model.pkl, sequence_model.pt, vocab.json, anomaly_model.pkl)
├── dashboard/
│ └── index.html # final judge-facing UI
└── frontend/
└── index.html # earlier/basic version, not used in final demo

---

## Quick Start

**Terminal 1 — Exasol**

```powershell
docker start exasol-nano
```

**Terminal 2 — Backend**

```powershell
cd agentguard
.\.venv\Scripts\Activate.ps1
$env:EXAPW = Get-Content "$env:USERPROFILE\.exasol-starter-kit\credentials\nano_sys_password"
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 3 — Dashboard**

```powershell
cd agentguard\dashboard
python -m http.server 3001
```

Open **http://localhost:3001**

_(First-time setup: `pip install -r requirements.txt`, run `python backend/run_schema.py` once to initialize the Exasol schema.)_

---

## Demo Scenarios

| Scenario         | Trajectory                                                                | Result                                     |
| ---------------- | ------------------------------------------------------------------------- | ------------------------------------------ |
| Benign           | `search_docs → read_customer_data`                                        | ALLOW → ALLOW                              |
| Malicious        | `search_docs → read_customer_data → access_credentials → export_database` | ALLOW → ALLOW → **BLOCK** → **KILLSWITCH** |
| Web — normal     | `visit_url` → `en.wikipedia.org/...`                                      | ALLOW                                      |
| Web — suspicious | `visit_url` → raw internal IP                                             | **ALLOW_FLAGGED** (~0.65)                  |

A killswitch terminates the session; further calls on it are rejected until a new session is started.

---

## Research References

- **ToolSafe** (arXiv:2601.10156) — step-level tool guardrails → basis for the deterministic rule layer
- **Trajectory Guard** (arXiv:2601.00516) — order-aware trajectory modeling → basis for the GRU layer
- **TrajAD** (arXiv:2602.06443) — runtime trajectory anomaly detection & localization → basis for trigger-step localization
- **Oliveira Malware API-Call Dataset** — external sequence-model validation

These inspired the architecture at a scale appropriate for the hackathon — we don't claim to reproduce their full results.

---

## Honest Limitations

- **Anomaly detection** exists but is excluded from trusted enforcement — its calibration produced false high scores on benign trajectories.
- **SHAP** is offline/global, not a live per-request signal.
- **Training data** for the agent-trajectory models is synthetic.
- **External validation** (Oliveira) is malware data, not native agent telemetry.
- **Exasol UDF** is implemented and validated inside Exasol, but the runtime path still scores in application code.
- Local, demonstration-scale system — not a production-hardened deployment.

## Future Work

- Wire the Exasol-native UDF into the live enforcement path
- Train a properly calibrated anomaly detector on real telemetry
- Larger, purpose-built agent-trajectory dataset
- Live per-event SHAP in the API
- Multi-agent trajectory correlation

---

<p align="center">Built for the Exasol AI Build Challenge 2026 · Autonomous Agents Track</p>
