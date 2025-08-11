"""
Model Manager for Adaptive Tutorial System

Handles loading and managing trained RL models (Q-Learning and Policy Gradient)
"""

import os
import pickle
import torch
import numpy as np
from typing import Optional, Dict, Any
import logging

class ModelManager:
    """
    Manages trained RL models for the Adaptive Tutorial System
    
    Automatically detects and loads available trained models:
    - Q-Learning models (.pkl files)
    - Policy Gradient models (.pth files)
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.q_learning_model = None
        self.policy_gradient_model = None
        self.models_loaded = False
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Automatically load models on initialization
        self.load_available_models()
    
    def load_available_models(self):
        """Load any available trained models from the models directory"""
        self.logger.info(f"🔍 Scanning for trained models in {self.models_dir}/")
        
        if not os.path.exists(self.models_dir):
            self.logger.warning(f"Models directory {self.models_dir} not found")
            return
        
        # Look for trained models
        model_files = os.listdir(self.models_dir)
        
        # Priority order: final > highest checkpoint number > any available
        q_learning_files = [f for f in model_files if f.endswith('.pkl')]
        policy_gradient_files = [f for f in model_files if f.endswith('.pth')]
        
        # Load Q-Learning model
        if q_learning_files:
            # Prefer final trained model
            final_q_file = next((f for f in q_learning_files if 'final_trained_models' in f), None)
            if final_q_file:
                self.load_q_learning_model(os.path.join(self.models_dir, final_q_file))
            else:
                # Use the first available
                self.load_q_learning_model(os.path.join(self.models_dir, q_learning_files[0]))
        
        # Load Policy Gradient model
        if policy_gradient_files:
            # Prefer final trained model
            final_pg_file = next((f for f in policy_gradient_files if 'final_trained_models' in f), None)
            if final_pg_file:
                self.load_policy_gradient_model(os.path.join(self.models_dir, final_pg_file))
            else:
                # Use the first available
                self.load_policy_gradient_model(os.path.join(self.models_dir, policy_gradient_files[0]))
        
        if self.q_learning_model or self.policy_gradient_model:
            self.models_loaded = True
            self.logger.info("✅ Trained models loaded successfully!")
        else:
            self.logger.info("ℹ️  No trained models found, will use untrained agents")
    
    def load_q_learning_model(self, filepath: str) -> bool:
        """Load Q-Learning model from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.q_learning_model = model_data
            self.logger.info(f"✅ Q-Learning model loaded from {os.path.basename(filepath)}")
            
            # Log model stats
            if 'q_table' in model_data:
                q_table_size = len(model_data['q_table'])
                epsilon = model_data.get('epsilon', 'unknown')
                self.logger.info(f"   Q-table size: {q_table_size} states, epsilon: {epsilon}")
                # Sample key logging removed for production cleanliness
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Q-Learning model from {filepath}: {str(e)}")
            return False
    
    def load_policy_gradient_model(self, filepath: str) -> bool:
        """Load Policy Gradient model from PyTorch file"""
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            
            self.policy_gradient_model = checkpoint
            self.logger.info(f"✅ Policy Gradient model loaded from {os.path.basename(filepath)}")
            
            # Log model stats
            if 'episode_returns' in checkpoint:
                episodes = len(checkpoint['episode_returns'])
                avg_return = np.mean(checkpoint['episode_returns'][-100:]) if episodes >= 100 else np.mean(checkpoint['episode_returns'])
                self.logger.info(f"   Trained episodes: {episodes}, recent avg return: {avg_return:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load Policy Gradient model from {filepath}: {str(e)}")
            return False
    
    def get_q_learning_config(self) -> Optional[Dict[str, Any]]:
        """Get Q-Learning model configuration for agent initialization"""
        if not self.q_learning_model:
            return None
        
        hyperparams = self.q_learning_model.get('hyperparameters', {})
        return {
            'q_table': self.q_learning_model.get('q_table', {}),
            'epsilon': self.q_learning_model.get('epsilon', 0.1),
            'state_size': hyperparams.get('state_size', 10),
            'action_size': hyperparams.get('action_size', 7),
            'learning_rate': hyperparams.get('learning_rate', 0.1),
            'discount_factor': hyperparams.get('discount_factor', 0.95),
            'episode_rewards': self.q_learning_model.get('episode_rewards', []),
            'training_step': self.q_learning_model.get('training_step', 0)
        }
    
    def get_policy_gradient_config(self) -> Optional[Dict[str, Any]]:
        """Get Policy Gradient model configuration for agent initialization"""
        if not self.policy_gradient_model:
            return None
        
        hyperparams = self.policy_gradient_model.get('hyperparameters', {})
        return {
            'policy_net_state_dict': self.policy_gradient_model.get('policy_net_state_dict'),
            'value_net_state_dict': self.policy_gradient_model.get('value_net_state_dict'),
            'state_size': hyperparams.get('state_size', 10),
            'action_size': hyperparams.get('action_size', 7),
            'learning_rate': hyperparams.get('learning_rate', 0.001),
            'discount_factor': hyperparams.get('discount_factor', 0.99),
            'use_baseline': hyperparams.get('use_baseline', True),
            'episode_returns': self.policy_gradient_model.get('episode_returns', []),
            'policy_losses': self.policy_gradient_model.get('policy_losses', []),
            'value_losses': self.policy_gradient_model.get('value_losses', [])
        }
    
    def load_all_models(self, q_learning_agent, policy_gradient_agent) -> Dict[str, Any]:
        """Load trained models into the agents"""
        results = {
            'q_learning_loaded': False,
            'policy_gradient_loaded': False,
            'models_available': self.models_loaded
        }
        
        # Load Q-Learning model if available
        if self.q_learning_model:
            try:
                q_config = self.get_q_learning_config()
                if q_config:
                    # Reconstruct Q-table as defaultdict for safe access.
                    # Keep keys exactly as stored (stringified bytes) to match the agent's hashing.
                    from collections import defaultdict
                    import numpy as np
                    loaded_q_table = q_config.get('q_table', {})
                    safe_q_table = defaultdict(lambda: np.zeros(q_learning_agent.action_size))
                    for key, value in loaded_q_table.items():
                        normalized_key = key if isinstance(key, str) else str(key)
                        safe_q_table[normalized_key] = np.array(value)

                    q_learning_agent.q_table = safe_q_table
                    q_learning_agent.epsilon = q_config['epsilon']
                    q_learning_agent.episode_rewards = q_config['episode_rewards']
                    q_learning_agent.training_step = q_config['training_step']
                    results['q_learning_loaded'] = True
                    self.logger.info("✅ Q-Learning model loaded into agent")
            except Exception as e:
                self.logger.error(f"❌ Failed to load Q-Learning model into agent: {str(e)}")
        
        # Load Policy Gradient model if available
        if self.policy_gradient_model:
            try:
                pg_config = self.get_policy_gradient_config()
                if pg_config and pg_config['policy_net_state_dict']:
                    # Load the trained networks
                    policy_gradient_agent.policy_net.load_state_dict(pg_config['policy_net_state_dict'])
                    if pg_config['value_net_state_dict'] and hasattr(policy_gradient_agent, 'value_net'):
                        policy_gradient_agent.value_net.load_state_dict(pg_config['value_net_state_dict'])
                    
                    # Load training history
                    policy_gradient_agent.episode_returns = pg_config['episode_returns']
                    policy_gradient_agent.policy_losses = pg_config['policy_losses']
                    if pg_config['value_losses']:
                        policy_gradient_agent.value_losses = pg_config['value_losses']
                    
                    results['policy_gradient_loaded'] = True
                    self.logger.info("✅ Policy Gradient model loaded into agent")
            except Exception as e:
                self.logger.error(f"❌ Failed to load Policy Gradient model into agent: {str(e)}")
        
        return results
    
    def has_trained_models(self) -> bool:
        """Check if any trained models are available"""
        return self.models_loaded
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get summary of loaded models"""
        summary = {
            'models_loaded': self.models_loaded,
            'q_learning_available': self.q_learning_model is not None,
            'policy_gradient_available': self.policy_gradient_model is not None
        }
        
        if self.q_learning_model:
            q_config = self.get_q_learning_config()
            summary['q_learning_info'] = {
                'states': len(q_config['q_table']),
                'epsilon': q_config['epsilon'],
                'episodes_trained': len(q_config['episode_rewards'])
            }
        
        if self.policy_gradient_model:
            pg_config = self.get_policy_gradient_config()
            summary['policy_gradient_info'] = {
                'episodes_trained': len(pg_config['episode_returns']),
                'avg_return': np.mean(pg_config['episode_returns'][-100:]) if pg_config['episode_returns'] else 0
            }
        
        return summary

# Global model manager instance
model_manager = ModelManager()