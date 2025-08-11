import numpy as np
import time
from typing import Dict, List, Tuple
from environment.tutorial_environment import TutorialEnvironment
from agents.q_learning_tutor import TutorQLearning
from agents.policy_gradient_tutor import TutorPolicyGradient

class TutorTrainingManager:
    """
    Training Manager for Adaptive Tutorial System
    
    Coordinates training of Q-Learning and Policy Gradient agents
    in the tutorial environment. Handles:
    - Multi-agent training coordination
    - Performance tracking
    - Model saving and evaluation
    """
    
    def __init__(self, 
                 environment: TutorialEnvironment,
                 q_agent: TutorQLearning,
                 pg_agent: TutorPolicyGradient):
        
        self.env = environment
        self.q_agent = q_agent
        self.pg_agent = pg_agent
        
        # Training metrics
        self.training_episodes = 0
        self.episode_rewards = []
        self.episode_accuracies = []
        self.episode_lengths = []
        
        print("🎓 Tutor Training Manager initialized")
    
    def train_episode(self, max_questions: int = 20) -> Dict:
        """Train agents for one episode (tutorial session)"""
        episode_start = time.time()
        
        # Reset environment
        state = self.env.reset()
        episode_reward = 0
        questions_in_episode = 0
        
        # Reset agents for new episode
        self.q_agent.reset_for_new_session()
        self.pg_agent.reset_episode()
        
        while questions_in_episode < max_questions:
            # Get actions from both agents
            q_action = self.q_agent.get_action(state, training=True)
            pg_action = self.pg_agent.get_action(state, training=True)
            
            # Apply teaching actions to environment
            next_state, teaching_reward = self.env.apply_teaching_actions(q_action, pg_action)
            
            # Generate and simulate answering a question
            question = self.env.generate_question_from_state(state)
            
            # Simulate student answer
            is_correct, answer_reward, feedback = self.env.process_answer("simulated", question['id'])
            
            # Total reward
            total_reward = answer_reward + teaching_reward
            episode_reward += total_reward
            
            # Store experiences
            self.q_agent.store_experience(state, q_action, total_reward, next_state, False)
            self.pg_agent.store_experience(state, pg_action, total_reward)
            
            # Learn from experience
            self.q_agent.learn()
            
            # Update state
            state = next_state
            questions_in_episode += 1
            
            # Break if student performance indicates session should end
            if self.env.get_accuracy() < 0.2 and questions_in_episode > 5:
                break  # Student struggling too much
            if self.env.get_accuracy() > 0.9 and questions_in_episode > 10:
                break  # Student mastered the material
        
        # End of episode - train policy gradient
        self.pg_agent.learn()
        
        # Record episode metrics
        episode_time = time.time() - episode_start
        episode_accuracy = self.env.get_accuracy()
        
        self.training_episodes += 1
        self.episode_rewards.append(episode_reward)
        self.episode_accuracies.append(episode_accuracy)
        self.episode_lengths.append(questions_in_episode)
        
        episode_info = {
            'episode': self.training_episodes,
            'total_reward': episode_reward,
            'accuracy': episode_accuracy,
            'questions': questions_in_episode,
            'duration': episode_time,
            'final_mastery': self.env.student_profile['overall_mastery']
        }
        
        return episode_info
    
    def train_multiple_episodes(self, num_episodes: int, save_interval: int = 50) -> List[Dict]:
        """Train for multiple episodes"""
        print(f"🚀 Starting training for {num_episodes} episodes...")
        
        training_results = []
        
        for episode in range(num_episodes):
            episode_info = self.train_episode()
            training_results.append(episode_info)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                recent_rewards = [r['total_reward'] for r in training_results[-10:]]
                recent_accuracy = [r['accuracy'] for r in training_results[-10:]]
                
                print(f"Episode {episode + 1}/{num_episodes}:")
                print(f"  Avg Reward: {np.mean(recent_rewards):.3f}")
                print(f"  Avg Accuracy: {np.mean(recent_accuracy):.3f}")
                print(f"  Q-Learning ε: {self.q_agent.epsilon:.3f}")
            
            # Save models periodically
            if (episode + 1) % save_interval == 0:
                self.save_models(f"training_checkpoint_{episode + 1}")
                print(f"💾 Models saved at episode {episode + 1}")
        
        print(f"✅ Training completed! {num_episodes} episodes finished.")
        return training_results
    
    def evaluate_agents(self, num_episodes: int = 10) -> Dict:
        """Evaluate trained agents without learning"""
        print(f"📊 Evaluating agents over {num_episodes} episodes...")
        
        evaluation_results = {
            'episodes': [],
            'avg_reward': 0,
            'avg_accuracy': 0,
            'avg_questions': 0,
            'student_satisfaction': 0
        }
        
        # Temporarily disable learning
        original_epsilon = self.q_agent.epsilon
        self.q_agent.epsilon = 0  # No exploration during evaluation
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            questions_answered = 0
            
            while questions_answered < 20:
                # Get actions (no training)
                q_action = self.q_agent.get_action(state, training=False)
                pg_action = self.pg_agent.get_action(state, training=False)
                
                # Apply actions
                next_state, teaching_reward = self.env.apply_teaching_actions(q_action, pg_action)
                
                # Generate question and simulate answer
                question = self.env.generate_question_from_state(state)
                is_correct, answer_reward, feedback = self.env.process_answer("simulated", question['id'])
                
                episode_reward += answer_reward + teaching_reward
                questions_answered += 1
                state = next_state
                
                # Early stopping conditions
                if self.env.get_accuracy() < 0.15 and questions_answered > 5:
                    break
                if self.env.get_accuracy() > 0.95 and questions_answered > 8:
                    break
            
            # Calculate student satisfaction (based on difficulty appropriateness)
            satisfaction = self._calculate_student_satisfaction()
            
            episode_result = {
                'episode': episode + 1,
                'reward': episode_reward,
                'accuracy': self.env.get_accuracy(),
                'questions': questions_answered,
                'satisfaction': satisfaction,
                'final_mastery': self.env.student_profile['overall_mastery']
            }
            
            evaluation_results['episodes'].append(episode_result)
        
        # Restore original epsilon
        self.q_agent.epsilon = original_epsilon
        
        # Calculate averages
        evaluation_results['avg_reward'] = np.mean([ep['reward'] for ep in evaluation_results['episodes']])
        evaluation_results['avg_accuracy'] = np.mean([ep['accuracy'] for ep in evaluation_results['episodes']])
        evaluation_results['avg_questions'] = np.mean([ep['questions'] for ep in evaluation_results['episodes']])
        evaluation_results['student_satisfaction'] = np.mean([ep['satisfaction'] for ep in evaluation_results['episodes']])
        
        print(f"📈 Evaluation Results:")
        print(f"  Average Reward: {evaluation_results['avg_reward']:.3f}")
        print(f"  Average Accuracy: {evaluation_results['avg_accuracy']:.3f}")
        print(f"  Average Questions: {evaluation_results['avg_questions']:.1f}")
        print(f"  Student Satisfaction: {evaluation_results['student_satisfaction']:.3f}")
        
        return evaluation_results
    
    def _calculate_student_satisfaction(self) -> float:
        """Calculate student satisfaction based on learning experience"""
        # Factors: appropriate difficulty, progress made, engagement
        
        # Difficulty appropriateness
        current_mastery = self.env.student_profile['overall_mastery']
        difficulty_scores = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}
        target_difficulty = difficulty_scores[self.env.current_difficulty]
        difficulty_appropriateness = 1.0 - abs(target_difficulty - (current_mastery + 0.1))
        difficulty_appropriateness = max(0.0, difficulty_appropriateness)
        
        # Learning progress (improvement during session)
        if len(self.env.answer_history) >= 5:
            early_performance = np.mean(self.env.answer_history[:5])
            late_performance = np.mean(self.env.answer_history[-5:])
            progress = max(0, late_performance - early_performance)
        else:
            progress = 0.5
        
        # Engagement (not too easy, not too hard)
        accuracy = self.env.get_accuracy()
        engagement = 1.0 - abs(accuracy - 0.7)  # Ideal accuracy around 70%
        engagement = max(0.0, engagement)
        
        # Weighted combination
        satisfaction = (0.4 * difficulty_appropriateness + 
                       0.3 * progress + 
                       0.3 * engagement)
        
        return satisfaction
    
    def save_models(self, checkpoint_name: str = "final"):
        """Save both agent models"""
        self.q_agent.save_model(f"models/{checkpoint_name}_q_learning.pkl")
        self.pg_agent.save_models(f"models/{checkpoint_name}_policy_gradient")
        
        # Save training metadata
        import json
        metadata = {
            'training_episodes': self.training_episodes,
            'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
            'episode_accuracies': self.episode_accuracies[-100:],
            'episode_lengths': self.episode_lengths[-100:],
            'config': {
                'topics': self.env.topics,
                'difficulty_levels': self.env.difficulty_levels,
                'question_types': self.env.question_types
            },
            'timestamp': time.time()
        }
        
        with open(f"models/{checkpoint_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Complete model checkpoint saved: {checkpoint_name}")
    
    def get_training_summary(self) -> Dict:
        """Get comprehensive training summary"""
        if not self.episode_rewards:
            return {'status': 'no_training_data'}
        
        recent_episodes = min(50, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-recent_episodes:]
        recent_accuracies = self.episode_accuracies[-recent_episodes:]
        recent_lengths = self.episode_lengths[-recent_episodes:]
        
        return {
            'total_episodes': self.training_episodes,
            'recent_performance': {
                'avg_reward': np.mean(recent_rewards),
                'reward_std': np.std(recent_rewards),
                'avg_accuracy': np.mean(recent_accuracies),
                'avg_episode_length': np.mean(recent_lengths)
            },
            'overall_performance': {
                'best_reward': max(self.episode_rewards),
                'best_accuracy': max(self.episode_accuracies),
                'reward_trend': 'improving' if len(recent_rewards) > 10 and 
                              np.mean(recent_rewards[-10:]) > np.mean(recent_rewards[:10]) else 'stable'
            },
            'agent_status': {
                'q_learning_epsilon': self.q_agent.epsilon,
                'q_learning_performance': self.q_agent.get_performance_metrics(),
                'policy_gradient_performance': self.pg_agent.get_performance_metrics()
            }
        }
