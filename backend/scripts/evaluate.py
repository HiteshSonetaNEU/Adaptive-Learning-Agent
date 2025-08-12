"""
Evaluation runner for the Adaptive Tutorial Agent (baseline vs trained, stats, and figures).

This script can:
1) Run baseline (untrained) vs trained agent sessions
2) Compute statistical tests (t-test + Wilcoxon) on accuracy distributions
3) Export CSV/JSON summaries to docs/results/
4) Save learning and behavior plots to docs/figures/

Examples:
  # Quick evaluation with small runs
  python -m backend.scripts.evaluate --sessions 20 --steps 15 --out docs

  # Larger evaluation for stronger stats
  python -m backend.scripts.evaluate --sessions 100 --steps 20 --out docs --seed 42

This script does NOT require the FastAPI server; it instantiates the agents
directly in-process using the same classes the server uses.
"""
from __future__ import annotations

import argparse
import os
import sys
import csv
from datetime import datetime
from typing import List, Dict
import json
import random

# Add the backend directory to Python path so we can import our modules
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from agents.q_learning_tutor import TutorQLearning
from agents.policy_gradient_tutor import TutorPolicyGradient
from environment.tutorial_environment import TutorialEnvironment
from models.model_manager import ModelManager


def run_session(env: TutorialEnvironment,
                q_agent: TutorQLearning,
                pg_agent: TutorPolicyGradient,
                steps: int = 10) -> Dict:
    env.reset()
    # Simple session loop with random/free-form answers; correctness is
    # determined by the environment logic and guardrails.
    session_stats = {
        'answers': 0,
        'correct': 0,
        'topic_hist': [],
        'difficulty_hist': [],
        'q_actions': [],
        'pg_actions': [],
    }

    for _ in range(steps):
        # Generate question
        env.generate_simple_question()

        # Provide an answer with some variability to induce accuracy variance
        q = env.current_question or {}
        if q.get('type') == 'multiple_choice':
            options = q.get('options') or []
            correct = str(q.get('correct_answer', '')).strip()
            if options:
                # 40% choose correct, else random option
                if random.random() < 0.4 and correct in options:
                    ans = correct
                else:
                    ans = random.choice(options)
            else:
                ans = correct or 'A'
        else:
            ans = 'answer'

        is_correct, reward, feedback = env.process_answer(student_answer=ans, question_id=q.get('id', ''))

        state = env.get_state()
        q_action = int(q_agent.get_action(state, training=False))
        pg_action = int(pg_agent.get_action(state, training=False))

        env.apply_teaching_actions(q_action, pg_action)

        session_stats['answers'] += 1
        session_stats['correct'] += 1 if is_correct else 0
        session_stats['topic_hist'].append(env.current_topic)
        session_stats['difficulty_hist'].append(env.current_difficulty)
        session_stats['q_actions'].append(q_action)
        session_stats['pg_actions'].append(pg_action)

    return session_stats


def run_experiments(sessions: int, steps: int, seed: int | None = None) -> Dict:
    if seed is not None:
        random.seed(seed)
        import numpy as _np
        _np.random.seed(seed)

    # Shared environment config
    topics = ["Math", "Science", "History", "Literature"]
    difficulty_levels = ["easy", "medium", "hard"]
    question_types = ["multiple_choice", "short_answer", "essay"]

    # Baseline (untrained)
    env_b = TutorialEnvironment(topics, difficulty_levels, question_types)
    q_b = TutorQLearning(state_size=env_b.get_state_size(), action_size=7)
    pg_b = TutorPolicyGradient(state_size=env_b.get_state_size(), action_size=5)

    # Trained (load if available)
    env_t = TutorialEnvironment(topics, difficulty_levels, question_types)
    q_t = TutorQLearning(state_size=env_t.get_state_size(), action_size=7)
    pg_t = TutorPolicyGradient(state_size=env_t.get_state_size(), action_size=5)
    ModelManager().load_all_models(q_t, pg_t)

    baseline_rows: List[Dict] = []
    trained_rows: List[Dict] = []

    for s in range(sessions):
        b_stats = run_session(env_b, q_b, pg_b, steps=steps)
        t_stats = run_session(env_t, q_t, pg_t, steps=steps)
        baseline_rows.append({
            'session': s + 1,
            'answers': b_stats['answers'],
            'correct': b_stats['correct'],
            'accuracy': b_stats['correct'] / max(1, b_stats['answers'])
        })
        trained_rows.append({
            'session': s + 1,
            'answers': t_stats['answers'],
            'correct': t_stats['correct'],
            'accuracy': t_stats['correct'] / max(1, t_stats['answers'])
        })

    return {
        'baseline': baseline_rows,
        'trained': trained_rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--out', type=str, default='docs')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    results_dir = os.path.join(args.out, 'results')
    figures_dir = os.path.join(args.out, 'figures')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    # Run experiments
    exp = run_experiments(args.sessions, args.steps, seed=args.seed)
    baseline_rows = exp['baseline']
    trained_rows = exp['trained']

    # Write CSV
    # Write CSVs
    b_csv = os.path.join(results_dir, f'baseline_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    t_csv = os.path.join(results_dir, f'trained_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(b_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
        w.writeheader()
        w.writerows(baseline_rows)
    with open(t_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(trained_rows[0].keys()))
        w.writeheader()
        w.writerows(trained_rows)
    print(f"Saved summaries: {b_csv} and {t_csv}")

    # Stats tests
    try:
        import numpy as np
        from scipy import stats
        b_acc = np.array([r['accuracy'] for r in baseline_rows], dtype=float)
        t_acc = np.array([r['accuracy'] for r in trained_rows], dtype=float)

        # Paired t-test and Wilcoxon signed-rank (non-parametric)
        t_stat, t_p = stats.ttest_rel(t_acc, b_acc)
        w_stat, w_p = stats.wilcoxon(t_acc, b_acc, zero_method='wilcox', alternative='greater') if len(b_acc) == len(t_acc) else (None, None)

        # 95% CI for mean improvement
        diff = t_acc - b_acc
        mean_diff = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se

        stats_json = {
            'sessions': int(args.sessions),
            'steps': int(args.steps),
            'seed': args.seed,
            'baseline_mean_acc': float(np.mean(b_acc)),
            'trained_mean_acc': float(np.mean(t_acc)),
            'mean_improvement': mean_diff,
            'ci_95': [ci_low, ci_high],
            't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
            'wilcoxon': {'w_stat': None if w_stat is None else float(w_stat), 'p_value': None if w_p is None else float(w_p)}
        }

        stats_path = os.path.join(results_dir, 'statistical_summary.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_json, f, indent=2)
        print(f"Saved statistical summary: {stats_path}")
    except Exception as e:
        print(f"Stats computation skipped: {e}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore
        import seaborn as sns  # type: ignore
        import numpy as np

        def safe_kde_or_hist(values, label, color=None):
            values = np.asarray(values, dtype=float)
            unique_vals = np.unique(values)
            if len(values) < 2 or len(unique_vals) < 2 or np.std(values) < 1e-6:
                sns.histplot(values, bins=10, stat='density', label=label, color=color, element='step', fill=True, alpha=0.35)
            else:
                sns.kdeplot(values, label=label, fill=True, color=color)

        # Accuracy distributions
        plt.figure(figsize=(7, 4))
        b_vals = [r['accuracy'] for r in baseline_rows]
        t_vals = [r['accuracy'] for r in trained_rows]
        safe_kde_or_hist(b_vals, 'Baseline')
        safe_kde_or_hist(t_vals, 'Trained')
        plt.xlim(0, 1)
        plt.xlabel('Accuracy')
        plt.title('Accuracy Distribution: Baseline vs Trained')
        plt.legend()
        fig_path = os.path.join(figures_dir, 'accuracy_distribution.png')
        plt.tight_layout(); plt.savefig(fig_path); plt.close()
        print(f"Saved plot: {fig_path}")

        # Per-session comparison
        plt.figure(figsize=(8, 4))
        plt.plot([r['session'] for r in baseline_rows], [r['accuracy'] for r in baseline_rows], marker='o', label='Baseline')
        plt.plot([r['session'] for r in trained_rows], [r['accuracy'] for r in trained_rows], marker='o', label='Trained')
        plt.ylim(0, 1)
        plt.xlabel('Session')
        plt.ylabel('Accuracy')
        plt.title('Per-Session Accuracy: Baseline vs Trained')
        plt.legend()
        fig_path = os.path.join(figures_dir, 'accuracy_over_sessions.png')
        plt.tight_layout(); plt.savefig(fig_path); plt.close()
        print(f"Saved plot: {fig_path}")
    except Exception as e:
        print(f"Plots skipped: {e}")


if __name__ == '__main__':
    main()

