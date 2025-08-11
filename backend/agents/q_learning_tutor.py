import numpy as np
import pickle
import os
import random
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

class TutorQLearning:
    """
    Q-Learning Agent for Adaptive Tutorial System
    
    This agent learns optimal difficulty adaptation strategies based on student performance.
    It makes decisions about:
    - When to increase/decrease difficulty
    - When to change topics
    - When to provide hints or review material
    
    State space: [current_topic, difficulty_level, recent_accuracy, mastery_scores, question_count]
    Action space: [easier, same_difficulty, harder, change_topic, provide_hint, review_mode, practice_mode]
    """
    
    def __init__(self, 
                 state_size: int = 10,
                 action_size: int = 7,
                 learning_rate: float = 0.1,
                 discount_factor: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-table as nested dictionaries
        # Structure: q_table[state_hash][action] = q_value
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Experience storage for learning
        self.experiences = []
        self.max_experiences = 1000
        
        # Performance tracking
        self.episode_rewards = []
        self.learning_progress = []
        self.decision_history = []
        
        # Action mapping for interpretability
        self.action_names = [
            'make_easier',      # 0: Reduce difficulty
            'keep_same',        # 1: Maintain current difficulty
            'make_harder',      # 2: Increase difficulty
            'change_topic',     # 3: Switch to different topic
            'provide_hint',     # 4: Give student a hint
            'review_mode',      # 5: Enter review/practice mode
            'adaptive_pace'     # 6: Adapt pacing based on performance
        ]
        
        print(f"🤖 Q-Learning Tutor initialized with {action_size} actions: {self.action_names}")
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Convert state array to hashable string for Q-table indexing (matches training format)"""
        # Discretize continuous values and hash to bytes, then string-ify the bytes object
        # This mirrors the original training-time behavior so loaded Q-table keys match.
        discrete_state = np.round(state * 100).astype(int)
        return str(discrete_state.tobytes())

    def _generate_hash_variants(self, state: np.ndarray) -> List[str]:
        """Generate alternative hash encodings to maximize compatibility with trained Q-tables."""
        variants: List[str] = []
        for scale in [100, 10, 1000]:
            ds = np.round(state * scale).astype(int)
            # Original bytes-string format
            variants.append(str(ds.tobytes()))
            # Readable CSV format that might have been used in other runs
            variants.append(','.join(map(str, ds.tolist())))
        return variants

    def _get_q_values_for_state(self, state: np.ndarray) -> np.ndarray:
        """Attempt to fetch Q-values using multiple hash variants before falling back to default."""
        # Try variants without creating default entries
        for key in self._generate_hash_variants(state):
            if key in self.q_table:
                values = self.q_table[key]
                try:
                    import numpy as np  # local import to avoid global side effects
                    return np.array(values)
                except Exception:
                    pass
        # Fall back to canonical key (this will create an entry if missing)
        canonical_key = self._hash_state(state)
        return self.q_table[canonical_key]

    def debug_hash_match(self, state: np.ndarray) -> dict:
        """Return info about which hash variant matches the loaded Q-table (for logging)."""
        matches = []
        for key in self._generate_hash_variants(state):
            if key in self.q_table:
                matches.append(key[:24] + '...' if len(key) > 24 else key)
        return {
            'canonical': self._hash_state(state)[:24] + '...',
            'matches': matches[:3]
        }
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state observation [topic_id, difficulty, accuracy, mastery, etc.]
            training: Whether in training mode (affects exploration)
        
        Returns:
            Selected action (0-6)
        """
        # Exploration vs Exploitation
        if training and random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            decision_type = "🎲 EXPLORE"
            q_values = self._get_q_values_for_state(state)
        else:
            # Choose action with highest Q-value (using compatibility lookups)
            q_values = self._get_q_values_for_state(state)
            action = int(np.argmax(q_values))
            decision_type = "🎯 EXPLOIT"
        
        # Store decision for analysis
        self.decision_history.append({
            'state_hash': self._hash_state(state),
            'action': action,
            'action_name': self.action_names[action],
            'decision_type': decision_type,
            'q_values': q_values.copy(),
            'epsilon': self.epsilon
        })
        
        return action
    
    def store_experience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool):
        """Store experience for learning"""
        experience = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done
        }
        
        self.experiences.append(experience)
        
        # Keep only recent experiences
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
    
    def learn(self):
        """Update Q-values using recent experiences"""
        if not self.experiences:
            return
        
        # Learn from recent experience
        experience = self.experiences[-1]
        state = experience['state']
        action = experience['action']
        reward = experience['reward']
        next_state = experience['next_state']
        done = experience['done']
        
        # Convert states to hash keys
        # Use compatibility lookups when reading, but write back to canonical key
        q_values = self._get_q_values_for_state(state)
        next_q_values = self._get_q_values_for_state(next_state)
        state_hash = self._hash_state(state)
        next_state_hash = self._hash_state(next_state)
        
        # Q-learning update rule
        current_q = float(q_values[action])
        
        if done:
            target_q = reward
        else:
            next_max_q = float(np.max(next_q_values))
            target_q = reward + self.discount_factor * next_max_q
        
        # Update Q-value
        self.q_table[state_hash][action] += self.learning_rate * (target_q - current_q)
        
        # Decay epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Track learning progress
        self.learning_progress.append({
            'episode': len(self.learning_progress),
            'reward': reward,
            'epsilon': self.epsilon,
            'q_value_update': abs(target_q - current_q)
        })
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for analysis"""
        if not self.decision_history:
            return {'status': 'no_data'}
        
        recent_decisions = self.decision_history[-50:]  # Last 50 decisions
        
        # Calculate action distribution
        action_counts = defaultdict(int)
        for decision in recent_decisions:
            action_counts[decision['action_name']] += 1
        
        # Calculate average Q-values
        total_q_values = np.zeros(self.action_size)
        for decision in recent_decisions:
            total_q_values += decision['q_values']
        avg_q_values = total_q_values / len(recent_decisions) if recent_decisions else total_q_values
        
        return {
            'total_decisions': len(self.decision_history),
            'recent_decisions': len(recent_decisions),
            'action_distribution': dict(action_counts),
            'current_epsilon': self.epsilon,
            'avg_q_values': avg_q_values.tolist(),
            'q_table_size': len(self.q_table),
            'learning_episodes': len(self.learning_progress)
        }
    
    def save_model(self, filepath: str):
        """Save Q-table and parameters"""
        model_data = {
            'q_table': dict(self.q_table),  # Convert defaultdict to regular dict
            'epsilon': self.epsilon,
            'learning_progress': self.learning_progress,
            'decision_history': self.decision_history[-100:],  # Save recent history
            'hyperparameters': {
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Q-Learning model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load Q-table and parameters"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore Q-table
            self.q_table = defaultdict(lambda: np.zeros(self.action_size))
            for state_hash, q_values in model_data['q_table'].items():
                self.q_table[state_hash] = np.array(q_values)
            
            # Restore parameters
            self.epsilon = model_data.get('epsilon', self.epsilon)
            self.learning_progress = model_data.get('learning_progress', [])
            self.decision_history = model_data.get('decision_history', [])
            
            print(f"✅ Q-Learning model loaded from {filepath}")
            print(f"   Q-table size: {len(self.q_table)} states")
            print(f"   Current epsilon: {self.epsilon:.3f}")
            print(f"   Learning episodes: {len(self.learning_progress)}")
            
        except Exception as e:
            print(f"❌ Failed to load Q-Learning model: {e}")
    
    def reset_for_new_session(self):
        """Reset episode-specific data for new tutorial session"""
        # Keep Q-table and learning progress, reset temporary data
        self.experiences = []
        
        # Start new episode in decision history
        if self.decision_history:
            self.decision_history.append({'type': 'new_session', 'episode': len(self.learning_progress)})
    
    def get_state_action_summary(self, state: np.ndarray) -> Dict:
        """Get detailed summary of Q-values for current state"""
        state_hash = self._hash_state(state)
        q_values = self.q_table[state_hash]
        
        action_summary = []
        for i, (action_name, q_val) in enumerate(zip(self.action_names, q_values)):
            action_summary.append({
                'action': action_name,
                'q_value': float(q_val),
                'is_best': i == np.argmax(q_values)
            })
        
        return {
            'state_hash': state_hash,
            'actions': action_summary,
            'best_action': self.action_names[np.argmax(q_values)],
            'exploration_rate': self.epsilon
        }
