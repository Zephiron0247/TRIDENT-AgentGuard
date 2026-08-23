# 🔱 TRIDENT — AgentGuard

### Autonomous Agent Runtime Security & Trajectory Monitoring

TRIDENT is an AI-agent runtime security platform that continuously evaluates agent tool-use **trajectories**, detects suspicious behavior, identifies the exact step responsible for an escalation, and enforces runtime policy through **allow → flag → block → killswitch** decisions.

Built for the **Exasol AI Build Challenge 2026** (Autonomous Agents track) — Exasol Personal is the primary data platform for all session, trajectory, and incident data.

---

## 🎬 Submission Materials

**Team VNT — VIT Chennai**

| Material      | Link                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| 🎥 Demo Video | [Watch on Google Drive](https://drive.google.com/drive/folders/1Q4CtNy--9riIL0AgT6_fWlCK6XDT2q2j?usp=sharing) |
| 📊 Pitch Deck | [View on Google Drive](https://drive.google.com/drive/folders/1Q4CtNy--9riIL0AgT6_fWlCK6XDT2q2j?usp=sharing)  |

> Both files live in the same shared Google Drive folder — GitHub can't preview Drive content inline, so please open the link directly in your browser (and make sure you're signed in / the folder is set to "Anyone with the link") to view the video and deck.

## 📖 Table of Contents

- [The Problem](#-the-problem)
- [The Solution](#-the-solution)
- [Architecture](#-architecture)
- [Detection Layers](#-detection-layers)
- [Risk Policy](#-risk-policy)
- [Trigger-Step Localization](#-trigger-step-localization)
- [Explainability](#-explainability)
- [External Validation](#-external-validation)
- [Database Structure](#-database-structure-exasol)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [🚀 Quick Start](#-quick-start)
- [Demo Scenarios](#-demo-scenarios)
- [Research References](#-research-references)
- [Limitations](#-limitations--honest-disclosures)
- [Future Work](#-future-work)

---

## 🎯 The Problem

Modern AI agents call tools autonomously — `search`, `read_file`, `access_credentials`, `export_database`, `visit_url` — one after another, often with no human checking each step. The danger isn't usually a single action. It's the **sequence**.

```
search_documents → read_customer_data → access_credentials → export_database
```

looks radically different from

```
search_docs → read_file → summarize → send_email
```

— but a system that only inspects one tool call at a time can't tell them apart. By the time anyone notices, the agent has already finished.

## 💡 The Solution

TRIDENT watches the **whole trajectory**, not isolated calls. It scores risk continuously as an agent acts, combines multiple independent detection signals, explains _why_ risk rose, pinpoints the _exact step_ that caused it, and — when policy thresholds are crossed — kills the session before further damage.

> Observe the agent's behavior → score the trajectory continuously → explain why risk increased → localize the triggering action → enforce policy.

---

## 🏗 Architecture

```
                    ┌──────────────────────┐
                    │      AI AGENT         │
                    │   tool invocation     │
                    └──────────┬────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FastAPI Gateway     │
                    │      /tool-call       │
                    └──────────┬────────────┘
                               │
                               ▼
                  ┌──────────────────────────┐
                  │    TRAJECTORY ENGINE      │
                  │ ┌────────────────────┐   │
                  │ │ Deterministic Rules│   │
                  │ ├────────────────────┤   │
                  │ │ RandomForest ML    │   │
                  │ ├────────────────────┤   │
                  │ │ GRU Sequence Model │   │
                  │ ├────────────────────┤   │
                  │ │ URL Risk Heuristics│   │
                  │ └────────────────────┘   │
                  └────────────┬──────────────┘
                               ▼
                     ┌────────────────────┐
                     │  Hybrid Risk Score  │
                     │   + Explanation     │
                     └─────────┬───────────┘
                               ▼
              ┌──────────────────────────────────┐
              │   Policy / Enforcement Engine     │
              │  <0.4 → ALLOW · 0.4–0.7 → FLAGGED│
              │  0.7–0.9 → BLOCK · >0.9 → KILL   │
              └──────────────┬────────────────────┘
                              │
              ┌───────────────┴────────────────┐
              ▼                                 ▼
     Trigger Localization                  Exasol Personal
     (exact risky step)                    (persistence layer)
              │                                 │
              └───────────────┬─────────────────┘
                               ▼
                  ┌────────────────────────┐
                  │   TRIDENT Dashboard    │
                  │    live monitoring     │
                  └────────────────────────┘
```

---

## 🛡 Detection Layers

### 1. Deterministic Rules — _"step-level guardrails"_

Captures known-risky tool combinations and trajectory patterns (sensitive-tool escalation, rapid-fire timing). Fast, explainable, always runs, and forms part of the trusted enforcement floor.

### 2. 🌲 RandomForest Classifier

Trained on six trajectory-level features:
| Feature | Description |
|---|---|
| `sensitive_tool_count` | How many sensitive tools were touched |
| `max_time_gap` / `min_time_gap` | Timing spread between calls |
| `avg_time_gap` | Average pace of execution |
| `session_length` | Number of calls so far |
| `unique_tool_count` | Diversity of tools used |

**Test performance:** Accuracy `0.9788` · Precision `0.8772` · Recall `0.8333` · F1 `0.8547`

### 3. 🔁 GRU Sequence Model — order-aware detection

```
Embedding(vocab_size, 8) → GRU(8, 16) → final hidden state → Linear(16, 1) → Sigmoid
```

Trained over a canonical 9-tool vocabulary. Unlike aggregate-stat features, the GRU is **order-aware** — it distinguishes a genuinely dangerous sequence from a reordered decoy containing the same tools in a harmless order.

Live behavior: malicious trajectories score `≈ 0.989`; benign/edge trajectories stay `≈ 0.001`.

> ⚠️ **Safety property:** the GRU is **advisory only**. It contributes to the hybrid score but cannot independently trigger a killswitch — enforcement authority stays with the deterministic/ML floor. A black-box model with unilateral kill power is a governance risk we deliberately designed against.

### 4. 🌐 URL Risk Heuristics — beyond email/documents

Extends detection into web-browsing agent behavior: non-HTTPS URLs, raw-IP hosts, known shorteners, and stated-goal/domain mismatch. Capped signal — cannot independently force a killswitch, only escalate to `ALLOW_FLAGGED`.

### 5. 🎯 Hybrid Scoring

```python
trusted_score = max(rule_score, ml_score, anomaly_score, url_score)
final_score   = combines trusted_score with GRU corroboration
```

`dominant_signal` and a plain-language `explanation` are returned with every decision.

---

## 📊 Risk Policy

| Risk Score  | Decision             | Meaning                                               |
| ----------- | -------------------- | ----------------------------------------------------- |
| `< 0.4`     | ✅ **ALLOW**         | Normal behavior                                       |
| `0.4 – 0.7` | 🟡 **ALLOW_FLAGGED** | Allowed, logged for review                            |
| `0.7 – 0.9` | 🟠 **BLOCK**         | This specific call is stopped                         |
| `> 0.9`     | 🔴 **KILLSWITCH**    | Entire session terminated — no further calls accepted |

---

## 🔍 Trigger-Step Localization

TRIDENT doesn't just say _"this session is malicious"_ — it identifies **which exact call** caused the escalation, inspired by TrajAD's approach to runtime anomaly localization.

```
search_docs           → ALLOW        0.16
read_customer_data    → ALLOW        0.13
access_credentials    → BLOCK  0.72  ⚠ TRIGGER
export_database       → KILLSWITCH 1.00  ⚠ TRIGGER
```

Exposed via `is_trigger_step` and `trigger_reason` on every event, surfaced live in the dashboard.

---

## 🧠 Explainability

SHAP analysis on the RandomForest identifies the strongest global risk drivers:

1. `avg_time_gap`
2. `sensitive_tool_count`
3. `max_time_gap`

> _"Fast execution timing and multiple sensitive tool invocations increased the model's risk estimate."_

Live per-event SHAP values are not exposed through the API — the dashboard correctly labels SHAP output as **offline model analysis**, not a real-time signal.

---

## 🧪 External Validation

The sequence-modeling _methodology_ was independently stress-tested against the **Oliveira malware API-call sequence benchmark** (43,876 real-world sequences: 1,079 benign / 42,797 malicious), completely separate from AgentGuard's own telemetry.

**Untouched test-set results, at a low-FPR operating point:**

| Metric    | Value  |
| --------- | ------ |
| ROC-AUC   | 0.7725 |
| PR-AUC    | 0.9914 |
| Precision | 0.9932 |
| Recall    | 0.3953 |
| F1        | 0.5656 |
| FPR       | 10.65% |

The near-perfect PR-AUC shows the model's underlying ranking of risky vs. benign sequences is strong; the threshold was deliberately set to favor **precision over recall** (0.99 precision at the cost of recall) — a conservative, security-appropriate tradeoff, not a weak model.

> **Honesty note:** this validates the _sequence-modeling approach_ on real-world malware data. It is **not** AgentGuard telemetry and does **not** by itself establish generalization to AI-agent tool-call data — it's independent evidence the underlying technique works, presented as exactly that.

---

## 🗄 Database Structure (Exasol)

Schema: `AGENTGUARD` — Exasol Personal (local Starter Kit) as the primary data platform.

| Table        | Purpose                                                                                         |
| ------------ | ----------------------------------------------------------------------------------------------- |
| `sessions`   | One row per agent session — status, stated goal, start/end                                      |
| `tool_calls` | Every intercepted call — risk score, decision, explanation, `is_trigger_step`, `trigger_reason` |
| `incidents`  | Terminated/high-risk sessions for review                                                        |

---

## 🧰 Technology Stack

`Python` · `FastAPI` · `Uvicorn` · `PyExasol` · `Exasol Personal` · `scikit-learn` · `PyTorch` · `SHAP` · `HTML/CSS/JavaScript`

---

## 📁 Project Structure

```
agentguard/
├── backend/
│   ├── main.py                # FastAPI gateway, /tool-call, /events
│   ├── exasol_client.py       # Exasol connection + queries
│   ├── run_schema.py
│   └── schema.sql
├── risk_engine/
│   ├── scorer.py               # Deterministic rules
│   ├── scorer_hybrid.py        # Combines all signals
│   ├── demo_agent.py           # Demo trajectory generator
│   ├── generate_dataset.py     # Synthetic training data
│   ├── train_models.py         # RandomForest / anomaly training
│   ├── sequence_model_torch.py # GRU sequence model
│   ├── validate_external.py    # Oliveira benchmark validation
│   ├── explain_model.py        # SHAP analysis
│   └── (trained artifacts: risk_model.pkl, sequence_model.pt, vocab.json, ...)
├── dashboard/
│   ├── index.html
│   └── assets/trident-logo.png
└── .venv/
```

---

## 🚀 Quick Start

### 1. Clone & set up the environment

```powershell
git clone <your-repo-url>
cd agentguard
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start Exasol Personal (local Starter Kit)

```powershell
irm https://www.exasol.com/install/starter-kit.ps1 | iex
exakit info    # confirm it's running, note the DSN/port
```

### 3. Set your Exasol password as an environment variable

```powershell
$env:EXAPW = Get-Content ~\.exasol-starter-kit\credentials\nano_sys_password
```

### 4. Initialize the schema

```powershell
python backend/run_schema.py
```

### 5. Start the backend

```powershell
python -m uvicorn backend.main:app --reload --port 8000
```

### 6. Start the dashboard

```powershell
python -m http.server 3000
```

Open **http://localhost:3000/dashboard/**

### 7. Run the live demo

```powershell
python risk_engine/demo_agent.py
```

Watch the dashboard update live as each scenario runs.

---

## 🎬 Demo Scenarios

| Scenario             | Trajectory                                                                | Expected Result                                                                                  |
| -------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **A — Benign**       | `search_docs → read_file → summarize`                                     | ALLOW throughout                                                                                 |
| **B — Edge case**    | `search_docs → send_email → summarize`                                    | ALLOW throughout (single sensitive call, no escalation)                                          |
| **C — Malicious**    | `search_docs → read_customer_data → access_credentials → export_database` | ALLOW → ALLOW → **BLOCK** → **KILLSWITCH**, triggers on `access_credentials` & `export_database` |
| **D — Web research** | `search_docs → visit_url`                                                 | Normal URL → ALLOW · Suspicious URL → **ALLOW_FLAGGED** (URL is dominant signal)                 |

---

## 📚 Research References

TRIDENT's architecture is grounded in current (2026) agent-security literature:

- **ToolSafe** — arXiv:2601.10156 — proactive step-level guardrails for tool invocation (the basis for our deterministic policy engine)
- **Trajectory Guard** — arXiv:2601.00516 — order-aware sequence modeling for agent trajectories; finding that aggregate/pooled statistics miss anomalies that sequence-aware models catch (the basis for our GRU layer)
- **TrajAD** — arXiv:2602.06443 — runtime trajectory anomaly detection with precise error localization (the basis for our trigger-step localization)
- **Oliveira Malware API-Call Sequence Dataset** — external validation benchmark, IEEE DataPort / Kaggle

We've tried to be precise about what each reference supports: these papers inspired the _architecture_, at a scale and scope appropriate for a same-day build — we are not claiming to have reproduced their full results.

---

## ⚠️ Limitations — Honest Disclosures

We'd rather state these plainly than have them discovered:

- **Exasol-native Python UDF** was implemented (`AGENTGUARD.RULE_RISK_SCORE`) but the local Starter Kit's script-language container wasn't available at runtime (`No usable script language container is installed`, despite `PYTHON3=builtin_python3` being configured). Rather than risk destabilizing a working system this close to submission, the UDF layer was parked — the code exists and is documented, but the live product currently computes the rule signal in application code, not inside Exasol.
- **Anomaly detector slot** exists in the hybrid scorer's design but no trained anomaly artifact is currently wired in; it reports `unavailable` rather than a fabricated number.
- **External validation** (Oliveira benchmark) validates the sequence-modeling _methodology_ on real malware data — it does not directly prove generalization to AI-agent tool-call trajectories, which is a different data distribution.
- **GRU is advisory-only** by design, not a limitation to fix — see Detection Layers above for the reasoning.

## 🔮 Future Work

- Complete the Exasol-native UDF scoring layer once a compatible script-language container is available
- Train a proper anomaly-detection artifact (e.g. Isolation Forest) on production telemetry once available
- Explore a larger, purpose-built agent-trajectory dataset to replace/extend the Oliveira cross-domain validation
- Live (not just offline) per-event SHAP explanations in the API response
- Multi-agent trajectory correlation (detecting coordinated risk across multiple agents in one workflow)

---

## 📦 Submission Package

This repository contains the complete TRIDENT source code, trained model artifacts, dashboard, documentation, run instructions, and supporting research material.

The final pitch deck and demonstration video are available in the shared submission folder below (GitHub cannot embed or preview Google Drive content, so click through to view them):

| Material      | Link                                                                                                          |
| ------------- | ------------------------------------------------------------------------------------------------------------- |
| 🎥 Demo Video | [Watch on Google Drive](https://drive.google.com/drive/folders/1Q4CtNy--9riIL0AgT6_fWlCK6XDT2q2j?usp=sharing) |
| 📊 Pitch Deck | [View on Google Drive](https://drive.google.com/drive/folders/1Q4CtNy--9riIL0AgT6_fWlCK6XDT2q2j?usp=sharing)  |

<p align="center">Built for the Exasol AI Build Challenge 2026 · Autonomous Agents Track</p>
