# Adaptive Tutorial Agent — Technical Documentation

## Overview
Two-agent RL system for personalized learning: Q-Learning (difficulty) + Policy Gradient (content strategy). React frontend + FastAPI backend with WebSocket; trained models can be loaded at startup.

## Architecture
- Backend: FastAPI server; `TutorialEnvironment` generates questions and rewards; agents decide actions each step.
- Frontend: WebSocket client, tutorial UI, analytics dashboard.
- Models: `ModelManager` loads trained Q-table/PG weights.

## Reproducibility
- Backend: `cd backend && pip install -r requirements.txt && python main.py`
- Frontend: `cd frontend && npm install && npm start`
- Evaluation (CSV + plots): `python -m backend.scripts.evaluate --sessions 5 --steps 10 --out results`

## Learning Performance (include in report)
- Session accuracy trend across N sessions (use evaluation script CSV/plot).
- Q-Learning: min/max/argmax Q-stats or action distribution across sessions.
- Policy Gradient: episode return trend (if collected) and strategy frequencies.
- Stability: run ≥3 seeds; report mean ± std of final accuracy.
- Varied environments: compare per-topic accuracy using balanced pools.

## Analysis Depth (cover in write‑up)
- Learning dynamics: Q-learning update and REINFORCE gradient; role of reward shaping.
- Strengths: real-time adaptation, two-agent synergy, resilient WS stack.
- Limitations: small Q-values in loaded table; simple rewards; synthetic items.
- Theory links: Bellman equation; policy gradient with baseline.
- Insights: practice_problem early; topic transitions on streaks; difficulty ratcheting.

## Presentation Aids
- Demo flow: start session → submit answers → show analytics + agent actions → show evaluation plot.
- Visuals: session accuracy plot; action distribution; per-topic accuracy bar chart; system diagram.

## Code Pointers
- `backend/agents/q_learning_tutor.py`
- `backend/agents/policy_gradient_tutor.py`
- `backend/environment/tutorial_environment.py`
- `backend/main.py`