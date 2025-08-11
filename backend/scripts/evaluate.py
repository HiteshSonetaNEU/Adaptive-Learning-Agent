"""
Lightweight evaluation runner for the Adaptive Tutorial Agent.

Runs a few headless sessions against the live agent stack (environment +
Q-learning + policy gradient) and writes simple CSV summaries and optional
PNG plots if matplotlib is available.

Usage:
  python -m backend.scripts.evaluate --sessions 5 --steps 10 --out results/

This script does NOT require the FastAPI server; it instantiates the agents
directly in-process using the same classes the server uses.
"""
from __future__ import annotations

import argparse
import os
import csv
from datetime import datetime
from typing import List, Dict

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

        # Provide a neutral answer string; correctness is determined by env
        # (exact match if ground-truth exists, otherwise simulated).
        is_correct, reward, feedback = env.process_answer(student_answer="", question_id=env.current_question.get('id', ''))

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessions', type=int, default=5)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--out', type=str, default='results')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Build system components as in the server
    topics = ["Math", "Science", "History", "Literature"]
    difficulty_levels = ["easy", "medium", "hard"]
    question_types = ["multiple_choice", "short_answer", "essay"]
    env = TutorialEnvironment(topics, difficulty_levels, question_types)
    q_agent = TutorQLearning(state_size=env.get_state_size(), action_size=7)
    pg_agent = TutorPolicyGradient(state_size=env.get_state_size(), action_size=5)

    # Load trained models if available
    ModelManager().load_all_models(q_agent, pg_agent)

    rows: List[Dict] = []
    for s in range(args.sessions):
        stats = run_session(env, q_agent, pg_agent, steps=args.steps)
        accuracy = stats['correct'] / max(1, stats['answers'])
        rows.append({
            'session': s + 1,
            'answers': stats['answers'],
            'correct': stats['correct'],
            'accuracy': accuracy,
            'final_topic': stats['topic_hist'][-1] if stats['topic_hist'] else '',
            'final_difficulty': stats['difficulty_hist'][-1] if stats['difficulty_hist'] else '',
        })

    # Write CSV
    csv_path = os.path.join(args.out, f'eval_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Saved evaluation summary: {csv_path}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(6, 4))
        plt.plot([r['session'] for r in rows], [r['accuracy'] for r in rows], marker='o')
        plt.ylim(0, 1)
        plt.xlabel('Session')
        plt.ylabel('Accuracy')
        plt.title('Adaptive Tutorial Agent: Session Accuracy')
        png_path = os.path.join(args.out, 'accuracy.png')
        plt.tight_layout()
        plt.savefig(png_path)
        print(f"Saved plot: {png_path}")
    except Exception as e:
        print(f"Plot skipped (matplotlib not available): {e}")


if __name__ == '__main__':
    main()

