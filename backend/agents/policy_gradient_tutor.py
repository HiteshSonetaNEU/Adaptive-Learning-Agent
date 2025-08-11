import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import os
from typing import Dict, List, Tuple, Optional

class PolicyNetwork(nn.Module):
    """Neural network for policy gradient agent"""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)

class ValueNetwork(nn.Module):
    """Value network for baseline (variance reduction)"""
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TutorPolicyGradient:
    """
    Policy Gradient (REINFORCE) Agent for Content Selection in Tutorial System
    
    This agent learns optimal content selection strategies:
    - Which question types to use
    - How to sequence topics
    - When to introduce new concepts
    - How to adapt to student learning patterns
    
    Uses REINFORCE with baseline for variance reduction.
    """
    
    def __init__(self,
                 state_size: int = 10,
                 action_size: int = 7,
                 learning_rate: float = 0.001,
                 discount_factor: float = 0.99,
                 use_baseline: bool = True,
                 device: str = None):
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.use_baseline = use_baseline
        
        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.policy_net = PolicyNetwork(state_size, action_size).to(self.device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        if use_baseline:
            self.value_net = ValueNetwork(state_size).to(self.device)
            self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=learning_rate)
        
        # Episode storage
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
        
        # Performance tracking
        self.episode_returns = []
        self.policy_losses = []
        self.value_losses = []
        
        # Action mapping for interpretability
        self.action_names = [
            'conceptual_question',    # 0: Focus on understanding concepts
            'practice_problem',       # 1: Give practice problems
            'application_example',    # 2: Real-world application
            'topic_transition',       # 3: Smoothly transition between topics
            'prerequisite_check',     # 4: Check prerequisite knowledge
            'advanced_challenge',     # 5: Provide challenging problems
            'interactive_exercise'    # 6: Interactive/visual exercises
        ]
        
        print(f"🧠 Policy Gradient Tutor initialized with actions: {self.action_names}")
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using policy network
        
        Args:
            state: Current state observation
            training: Whether in training mode
        
        Returns:
            Selected action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            action_probs = self.policy_net(state_tensor)
            
        if training:
            # Sample from probability distribution
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            # Store for learning
            self.episode_states.append(state)
            self.episode_actions.append(action.item())
            self.episode_log_probs.append(log_prob)
            
            return action.item()
        else:
            # Take most likely action during evaluation
            return action_probs.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float):
        """Store reward for current episode"""
        self.episode_rewards.append(reward)
    
    def learn(self):
        """Update policy using REINFORCE algorithm"""
        if not self.episode_rewards:
            return
        
        # Calculate returns (discounted rewards)
        returns = []
        discounted_sum = 0
        
        for reward in reversed(self.episode_rewards):
            discounted_sum = reward + self.discount_factor * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Convert episode data to tensors
        states = torch.FloatTensor(self.episode_states).to(self.device)
        log_probs = torch.stack(self.episode_log_probs).to(self.device)
        
        # Calculate baseline (if using)
        if self.use_baseline and hasattr(self, 'value_net'):
            state_values = self.value_net(states).squeeze()
            advantages = returns - state_values.detach()
            
            # Update value network
            value_loss = F.mse_loss(state_values, returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
            self.value_losses.append(value_loss.item())
        else:
            advantages = returns
        
        # Update policy network
        policy_loss = -(log_probs * advantages).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Store metrics
        self.policy_losses.append(policy_loss.item())
        self.episode_returns.append(returns.mean().item())
        
        # Clear episode data
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics for analysis"""
        if not self.episode_returns:
            return {'status': 'no_data'}
        
        recent_episodes = min(50, len(self.episode_returns))
        recent_returns = self.episode_returns[-recent_episodes:]
        recent_policy_losses = self.policy_losses[-recent_episodes:] if self.policy_losses else []
        recent_value_losses = self.value_losses[-recent_episodes:] if self.value_losses else []
        
        return {
            'total_episodes': len(self.episode_returns),
            'recent_episodes': recent_episodes,
            'avg_return': np.mean(recent_returns),
            'return_std': np.std(recent_returns),
            'avg_policy_loss': np.mean(recent_policy_losses) if recent_policy_losses else 0,
            'avg_value_loss': np.mean(recent_value_losses) if recent_value_losses else 0,
            'current_episode_length': len(self.episode_rewards),
            'using_baseline': self.use_baseline
        }
    
    def save_models(self, filepath_prefix: str):
        """Save policy and value networks"""
        os.makedirs(os.path.dirname(filepath_prefix), exist_ok=True)
        
        # Save policy network
        torch.save({
            'model_state_dict': self.policy_net.state_dict(),
            'optimizer_state_dict': self.policy_optimizer.state_dict(),
            'hyperparameters': {
                'state_size': self.state_size,
                'action_size': self.action_size,
                'learning_rate': self.learning_rate,
                'discount_factor': self.discount_factor
            },
            'performance_metrics': {
                'episode_returns': self.episode_returns,
                'policy_losses': self.policy_losses
            }
        }, f"{filepath_prefix}_policy.pth")
        
        # Save value network if using baseline
        if self.use_baseline and hasattr(self, 'value_net'):
            torch.save({
                'model_state_dict': self.value_net.state_dict(),
                'optimizer_state_dict': self.value_optimizer.state_dict(),
                'performance_metrics': {
                    'value_losses': self.value_losses
                }
            }, f"{filepath_prefix}_value.pth")
        
        print(f"💾 Policy Gradient models saved to {filepath_prefix}_*.pth")
    
    def load_models(self, filepath_prefix: str):
        """Load policy and value networks"""
        try:
            # Load policy network
            policy_path = f"{filepath_prefix}_policy.pth"
            if os.path.exists(policy_path):
                checkpoint = torch.load(policy_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.policy_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                # Restore performance metrics
                self.episode_returns = checkpoint.get('performance_metrics', {}).get('episode_returns', [])
                self.policy_losses = checkpoint.get('performance_metrics', {}).get('policy_losses', [])
                
                print(f"✅ Policy network loaded from {policy_path}")
            
            # Load value network if exists
            value_path = f"{filepath_prefix}_value.pth"
            if self.use_baseline and os.path.exists(value_path):
                checkpoint = torch.load(value_path, map_location=self.device)
                self.value_net.load_state_dict(checkpoint['model_state_dict'])
                self.value_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.value_losses = checkpoint.get('performance_metrics', {}).get('value_losses', [])
                
                print(f"✅ Value network loaded from {value_path}")
                
            print(f"   Episodes trained: {len(self.episode_returns)}")
            
        except Exception as e:
            print(f"❌ Failed to load Policy Gradient models: {e}")
    
    def get_action_probabilities(self, state: np.ndarray) -> Dict:
        """Get action probabilities for current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor).squeeze()
        
        action_summary = []
        for i, (action_name, prob) in enumerate(zip(self.action_names, action_probs)):
            action_summary.append({
                'action': action_name,
                'probability': float(prob),
                'is_most_likely': i == action_probs.argmax().item()
            })
        
        return {
            'actions': action_summary,
            'most_likely_action': self.action_names[action_probs.argmax().item()],
            'entropy': float(-(action_probs * torch.log(action_probs + 1e-8)).sum())
        }
    
    def reset_episode(self):
        """Reset episode data"""
        self.episode_states = []
        self.episode_actions = []
        self.episode_rewards = []
        self.episode_log_probs = []
