# AGENTS.md — Who does what (human or AI agent)

Any AI coding agent (Claude Code, Codex, Cursor, etc.) working here follows `CLAUDE.md`
and `LOOP.md`. Use sub-agents only when work is genuinely parallel; otherwise one agent.

| Role | Owns | Reads | Writes |
|---|---|---|---|
| **Data agent** | Download, clean, validate data; feature engineering | `configs/features.yaml`, `docs/DATASET_DECISION.md` | `src/data/`, `src/features/` |
| **Modelling agent** | Train/compare models, calibration, thresholds | `configs/models.yaml` | `src/models/`, `mlruns/`, `reports/` |
| **Explainability agent** | SHAP / EBM reason codes, fairness report | trained models | `src/explain/`, `docs/MODEL_CARD.md` |
| **Serving agent** | FastAPI service, Docker, demo UI | `src/` | `service/`, `ui/`, `Dockerfile` |
| **Docs agent** | Plain-English explanations for Marvin | everything | `docs/`, `README.md` |

## Hand-off contract
Each agent finishes by appending to `STATE.md`: what it produced, where it is,
how it was verified, and anything the next agent must know.

## Owner
Joshua (AI/ML engineer) approves plans and model promotions.
Marvin (product/BAaaS lead) is the business stakeholder.
