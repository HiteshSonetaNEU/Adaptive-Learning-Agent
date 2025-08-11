"""
Adaptive Tutorial Agent - Main Backend Server
Reinforcement Learning system for personalized education
"""

import asyncio
import json
import logging
import uvicorn
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import our RL agents and environment
from agents.q_learning_tutor import TutorQLearning
from agents.policy_gradient_tutor import TutorPolicyGradient
from environment.tutorial_environment import TutorialEnvironment
from agents.training_manager import TutorTrainingManager
from models.model_manager import ModelManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="🤖 Adaptive Tutorial Agent API",
    description="Reinforcement Learning-powered personalized education system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system components
tutorial_system = None
connected_clients: List[WebSocket] = []
MAX_QUESTIONS_PER_SESSION = 10

# Pydantic models for API
class StudentAnswer(BaseModel):
    answer: str
    question_id: str
    response_time: Optional[float] = None

class SystemStatus(BaseModel):
    status: str
    agents: Dict[str, bool]
    environment: Dict[str, str]
    config: Dict[str, List[str]]

class SessionStats(BaseModel):
    total_sessions: int
    active_sessions: int
    average_performance: float
    learning_efficiency: float

# Tutorial System Manager
class AdaptiveTutorialSystem:
    def __init__(self):
        """Initialize the adaptive tutorial system with RL agents"""
        # Define configuration
        topics = ["Math", "Science", "History", "Literature"]
        # Align difficulty names with environment expectations
        difficulty_levels = ["easy", "medium", "hard"]
        question_types = ["multiple_choice", "short_answer", "essay"]
        
        self.environment = TutorialEnvironment(topics, difficulty_levels, question_types)
        self.q_learning_agent = TutorQLearning(
            state_size=self.environment.get_state_size(),
            action_size=7,  # [easier, harder, change_topic, hint, review, continue, assess]
            learning_rate=0.1,
            epsilon=0.1,
            discount_factor=0.95
        )
        self.policy_gradient_agent = TutorPolicyGradient(
            state_size=self.environment.get_state_size(),
            action_size=5,  # [concept_intro, practice, example, assessment, review]
            learning_rate=0.001,
            discount_factor=0.99
        )
        self.training_manager = TutorTrainingManager(
            self.environment,
            self.q_learning_agent,
            self.policy_gradient_agent
        )
        
        # Initialize model manager and load pre-trained models
        self.model_manager = ModelManager()
        model_results = self.model_manager.load_all_models(
            self.q_learning_agent, 
            self.policy_gradient_agent
        )
        
        # Session management
        self.active_sessions = {}
        self.session_counter = 0
        
        logger.info("✅ Adaptive Tutorial System initialized successfully")
    
    def start_session(self, session_id: str) -> Dict:
        """Start a new tutorial session"""
        try:
            # Reset environment for new session
            initial_state = self.environment.reset()
            # Optionally augment math pool to ensure 50+ unique items
            try:
                self.environment.augment_math_questions(min_per_difficulty=30)
                self.environment.balance_topic_pools(target_per_difficulty=30)
            except Exception:
                pass
            
            # Set a consistent starting context and generate first question
            self.environment.current_topic = "Math"
            self.environment.current_difficulty = "medium"
            question = self.environment.generate_simple_question()
            # Add helpful placeholders/examples for non-MC questions
            if question.get('type') != 'multiple_choice':
                question.setdefault('placeholder', 'Briefly explain the concept or provide a short answer...')
                question.setdefault('example', 'e.g., Define the term or give a one-sentence explanation.')
            
            # Store session state
            self.active_sessions[session_id] = {
                'start_time': datetime.now(),
                'state': initial_state,
                'questions_answered': 0,
                'correct_answers': 0,
                'current_topic': "Math",
                'current_difficulty': "easy",
                'session_history': []
            }
            
            logger.info(f"🚀 Started tutorial session: {session_id}")
            
            return {
                'type': 'session_start',
                'session_id': session_id,
                'question': question,
                'message': 'Welcome to your adaptive learning session!'
            }
            
        except Exception as e:
            logger.error(f"❌ Error starting session {session_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start session: {str(e)}")
    
    def process_answer(self, session_id: str, answer: str, question_id: str) -> Dict:
        """Process student answer and adapt using RL agents"""
        try:
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.active_sessions[session_id]
            
            # Process answer in environment (env returns tuple)
            is_correct, answer_reward, feedback = self.environment.process_answer(
                student_answer=answer,
                question_id=question_id
            )
            result = {
                'correct': is_correct,
                'explanation': feedback,
                'reward': answer_reward
            }
            
            # Update session statistics
            session['questions_answered'] += 1
            if result['correct']:
                session['correct_answers'] += 1
            
            # Get current state for RL agents
            current_state = self.environment.get_state()
            
            # Q-Learning agent decides on difficulty adaptation (ensure native int)
            q_action = int(self.q_learning_agent.get_action(current_state))
            # Policy Gradient agent decides on content strategy (ensure native int)
            pg_action = int(self.policy_gradient_agent.get_action(current_state))

            # Apply agent decisions to environment (returns new_state, teaching_reward)
            new_state, teaching_reward = self.environment.apply_teaching_actions(
                q_action, pg_action
            )
            
            # Calculate rewards for agents
            # Build a minimal adaptation summary for reward helpers
            adaptation_info = {
                'difficulty_optimal': self.environment.current_difficulty == 'medium',
                'difficulty_suboptimal': self.environment.current_difficulty not in ['easy', 'medium', 'hard'],
                'engagement_increased': teaching_reward > 0,
                'learning_efficient': is_correct,
                'strategy_effective': (answer_reward + teaching_reward) > 0
            }
            q_reward = self._calculate_q_reward(result, adaptation_info)
            pg_reward = self._calculate_pg_reward(result, adaptation_info)
            
            # Train agents with experience (use pre-action state for Q-update)
            prev_state = session['state']
            self.q_learning_agent.store_experience(
                prev_state, q_action, q_reward, new_state, False
            )
            self.q_learning_agent.learn()
            
            self.policy_gradient_agent.store_experience(session['state'], pg_action, pg_reward)
            
            # Update session state
            session['state'] = new_state
            session['current_difficulty'] = self.environment.current_difficulty
            session['current_topic'] = self.environment.current_topic
            
            # Generate next question
            next_question = self.environment.generate_simple_question()
            if next_question.get('type') != 'multiple_choice':
                next_question.setdefault('placeholder', 'Briefly explain the concept or provide a short answer...')
                next_question.setdefault('example', 'e.g., Define the term or give a one-sentence explanation.')
            
            # Store interaction in session history
            session['session_history'].append({
                'question_id': question_id,
                'answer': answer,
                'correct': result['correct'],
                'q_action': q_action,
                'pg_strategy': pg_action,
                'timestamp': datetime.now().isoformat()
            })
            
            # Optionally learn PG every few steps for live learning
            if session['questions_answered'] % 5 == 0:
                try:
                    self.policy_gradient_agent.learn()
                except Exception:
                    pass

            # Prepare response
            # Compute Q-value and policy probability (report Q(prev_state, action))
            try:
                state_hash = self.q_learning_agent._hash_state(prev_state)
                q_values_for_state = self.q_learning_agent.q_table[state_hash]
                # Ensure it's a numpy array of correct size
                import numpy as np
                if not isinstance(q_values_for_state, np.ndarray) or len(q_values_for_state) != self.q_learning_agent.action_size:
                    q_values_for_state = np.zeros(self.q_learning_agent.action_size)
                q_value = float(q_values_for_state[q_action])
                # Log Q-value stats for diagnostics
                try:
                    q_arr = np.array(q_values_for_state, dtype=float)
                    q_min = float(np.min(q_arr))
                    q_max = float(np.max(q_arr))
                    q_argmax = int(np.argmax(q_arr))
                    logger.info(
                        f"📈 Q-stats: min={q_min:.3f} max={q_max:.3f} argmax={q_argmax} chosen={q_action} val={q_value:.3f} eps={self.q_learning_agent.epsilon:.3f}"
                    )
                except Exception:
                    pass
            except Exception:
                q_value = 0.0

            try:
                action_probs_summary = self.policy_gradient_agent.get_action_probabilities(prev_state)
                # The list is in order; pick probability for taken action
                pg_prob = action_probs_summary['actions'][pg_action]['probability']
            except Exception:
                pg_prob = 0.0

            # Debug logging removed for production cleanliness

            # Ensure JSON-serializable primitives
            topic_perf = {k: float(v) for k, v in self._get_topic_performance(session).items()}
            recent_perf = [float(x) for x in self._get_recent_performance(session)]

            response = {
                'type': 'answer_processed',
                'correct': bool(result['correct']),
                'explanation': result.get('explanation', ''),
                'next_question': next_question,
                'student_progress': {
                    'current_score': float(session['correct_answers'] / session['questions_answered']),
                    'questions_answered': int(session['questions_answered']),
                    'correct_answers': int(session['correct_answers']),
                    'current_difficulty': session['current_difficulty'],
                    'current_topic': session['current_topic'],
                    'learning_rate': float(self.environment.student_profile.get('learning_rate', 0.0)),
                    'session_time': float((datetime.now() - session['start_time']).total_seconds()),
                    'topic_performance': topic_perf
                },
                'agent_decisions': {
                    'q_learning': {
                        'action': int(q_action),
                        'q_value': float(q_value),
                        'epsilon': float(self.q_learning_agent.epsilon),
                        'reward': float(q_reward)
                    },
                    'policy_gradient': {
                        'strategy': int(pg_action),
                        'probability': float(pg_prob),
                        'baseline': float(0.0),  # Would need value function for proper baseline
                        'advantage': float(pg_reward)  # Simplified advantage
                    }
                },
                'learning_stats': {
                    'avg_response_time': float(self._calculate_avg_response_time(session)),
                    'learning_efficiency': float(self._calculate_learning_efficiency(session)),
                    'improvement_rate': float(self._calculate_improvement_rate(session)),
                    'engagement_level': float(self._calculate_engagement_level(session)),
                    'recent_performance': recent_perf
                }
            }
            
            logger.info(f"📝 Processed answer for session {session_id}: {result['correct']}")
            return response
            
        except Exception as e:
            logger.exception(f"❌ Error processing answer for session {session_id}")
            raise HTTPException(status_code=500, detail=f"Failed to process answer: {str(e)}")
    
    def provide_hint(self, session_id: str, question_id: str) -> Dict:
        """Provide a hint for the current question"""
        try:
            if session_id not in self.active_sessions:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = self.active_sessions[session_id]
            
            # Generate hint based on current context
            hint = self.environment.generate_hint(
                question_id=question_id
            )
            
            return {
                'type': 'hint_provided',
                'question_id': question_id,
                'hint': hint
            }
            
        except Exception as e:
            logger.exception(f"❌ Error providing hint for session {session_id}")
            raise HTTPException(status_code=500, detail=f"Failed to provide hint: {str(e)}")
    
    def _calculate_q_reward(self, result: Dict, adaptation: Dict) -> float:
        """Calculate reward for Q-Learning agent"""
        reward = 0.0
        
        # Reward for correct answers
        if result['correct']:
            reward += 1.0
        
        # Reward for maintaining optimal difficulty
        if adaptation.get('difficulty_optimal', False):
            reward += 0.2
        
        # Penalty for making it too easy/hard
        if adaptation.get('difficulty_suboptimal', False):
            reward -= 0.1
        
        return reward
    
    def _calculate_pg_reward(self, result: Dict, adaptation: Dict) -> float:
        """Calculate reward for Policy Gradient agent"""
        reward = 0.0
        
        # Reward for engagement
        if adaptation.get('engagement_increased', False):
            reward += 0.3
        
        # Reward for learning efficiency
        if adaptation.get('learning_efficient', False):
            reward += 0.5
        
        # Reward for correct strategy selection
        if result['correct'] and adaptation.get('strategy_effective', False):
            reward += 0.4
        
        return reward
    
    def _get_topic_performance(self, session: Dict) -> Dict[str, float]:
        """Calculate performance by topic"""
        topic_stats = {}
        for interaction in session['session_history']:
            topic = session['current_topic']  # Simplified
            if topic not in topic_stats:
                topic_stats[topic] = {'correct': 0, 'total': 0}
            topic_stats[topic]['total'] += 1
            if interaction['correct']:
                topic_stats[topic]['correct'] += 1
        
        return {topic: stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0 
                for topic, stats in topic_stats.items()}
    
    def _calculate_avg_response_time(self, session: Dict) -> float:
        """Calculate average response time"""
        return 15.0  # Placeholder - would need actual timing data
    
    def _calculate_learning_efficiency(self, session: Dict) -> float:
        """Calculate learning efficiency metric"""
        if session['questions_answered'] == 0:
            return 0.0
        return session['correct_answers'] / session['questions_answered']
    
    def _calculate_improvement_rate(self, session: Dict) -> float:
        """Calculate rate of improvement"""
        if len(session['session_history']) < 2:
            return 0.0
        
        recent_correct = sum(1 for h in session['session_history'][-5:] if h['correct'])
        early_correct = sum(1 for h in session['session_history'][:5] if h['correct'])
        
        recent_rate = recent_correct / min(5, len(session['session_history'][-5:]))
        early_rate = early_correct / min(5, len(session['session_history'][:5]))
        
        return max(0.0, recent_rate - early_rate)
    
    def _calculate_engagement_level(self, session: Dict) -> float:
        """Calculate engagement level"""
        # Simplified engagement metric based on session length and performance
        if session['questions_answered'] == 0:
            return 0.5
        
        engagement = min(1.0, session['questions_answered'] / 10) * 0.7
        engagement += session['correct_answers'] / session['questions_answered'] * 0.3
        
        return engagement
    
    def _get_recent_performance(self, session: Dict) -> List[float]:
        """Get recent performance trend"""
        recent_history = session['session_history'][-10:]  # Last 10 questions
        return [1.0 if h['correct'] else 0.0 for h in recent_history]
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Build a summary for the completed session"""
        session = self.active_sessions.get(session_id)
        if not session:
            return {
                'type': 'session_end',
                'message': 'Session not found'
            }
        total = session['questions_answered']
        correct = session['correct_answers']
        accuracy = correct / max(1, total)
        summary = {
            'type': 'session_end',
            'session_id': session_id,
            'total_questions': int(total),
            'correct_answers': int(correct),
            'accuracy': float(accuracy),
            'current_topic': session['current_topic'],
            'current_difficulty': session['current_difficulty'],
            'duration_seconds': float((datetime.now() - session['start_time']).total_seconds()),
            'recent_performance': self._get_recent_performance(session),
            'agent_notes': {
                'last_decision': getattr(self.environment, 'last_decision_reasoning', '')
            }
        }
        return summary
    
    def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        return SystemStatus(
            status='operational',
            agents={
                'q_learning': True,
                'policy_gradient': True
            },
            environment={
                'type': 'tutorial',
                'state_size': str(self.environment.get_state_size()),
                'initialized': 'True'
            },
            config={
                'topics': self.environment.topics,
                'difficulty_levels': self.environment.difficulty_levels,
                'question_types': self.environment.question_types
            }
        )

# Initialize system on startup
def initialize_tutorial_system():
    """Initialize the tutorial system"""
    global tutorial_system
    try:
        logger.info("🚀 Initializing Adaptive Tutorial System...")
        tutorial_system = AdaptiveTutorialSystem()
        logger.info("✅ Tutorial system initialized successfully!")
        return tutorial_system
    except Exception as e:
        logger.error(f"❌ Failed to initialize tutorial system: {str(e)}")
        raise

# API Routes
@app.get("/")
async def root():
    return {
        "message": "🤖 Adaptive Tutorial Agent API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/api/status", response_model=SystemStatus)
async def get_system_status():
    """Get current system status"""
    if not tutorial_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    return tutorial_system.get_system_status()

@app.post("/api/reset")
async def reset_system():
    """Reset the tutorial system"""
    try:
        global tutorial_system
        tutorial_system = AdaptiveTutorialSystem()
        return {"message": "System reset successfully", "status": "operational"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset system: {str(e)}")

@app.get("/api/stats")
async def get_session_stats():
    """Get learning statistics"""
    if not tutorial_system:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    active_sessions = len(tutorial_system.active_sessions)
    total_questions = sum(s['questions_answered'] for s in tutorial_system.active_sessions.values())
    total_correct = sum(s['correct_answers'] for s in tutorial_system.active_sessions.values())
    
    return {
        "total_sessions": tutorial_system.session_counter,
        "active_sessions": active_sessions,
        "average_performance": total_correct / max(1, total_questions),
        "learning_efficiency": 0.75,  # Placeholder metric
        "total_questions_answered": total_questions
    }

# WebSocket for real-time communication
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    session_id = f"session_{len(connected_clients)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"🔗 New WebSocket connection: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            logger.info(f"📨 Received message: {message.get('type', 'unknown')}")
            
            # Route message based on type
            if message['type'] == 'start_session':
                response = tutorial_system.start_session(session_id)
                await websocket.send_text(json.dumps(response))
                
            elif message['type'] == 'answer':
                # Ensure a session exists; if not, start one and send back session_start
                if session_id not in tutorial_system.active_sessions:
                    logger.warning(f"⚠️ Answer received but session missing; auto-starting session for {session_id}")
                    init_resp = tutorial_system.start_session(session_id)
                    await websocket.send_text(json.dumps(init_resp))
                    continue
                response = tutorial_system.process_answer(
                    session_id=session_id,
                    answer=message['answer'],
                    question_id=message['question_id']
                )
                await websocket.send_text(json.dumps(response))
                # If session reached max questions, send session_end and clean up
                session = tutorial_system.active_sessions.get(session_id)
                if session and session.get('questions_answered', 0) >= MAX_QUESTIONS_PER_SESSION:
                    summary = tutorial_system.get_session_summary(session_id)
                    await websocket.send_text(json.dumps(summary))
                    # Do not immediately close the websocket; allow client to decide
                
            elif message['type'] == 'request_hint':
                # Ensure session exists
                if session_id not in tutorial_system.active_sessions:
                    logger.warning(f"⚠️ Hint requested but session missing; auto-starting session for {session_id}")
                    init_resp = tutorial_system.start_session(session_id)
                    await websocket.send_text(json.dumps(init_resp))
                    continue
                response = tutorial_system.provide_hint(
                    session_id=session_id,
                    question_id=message['question_id']
                )
                await websocket.send_text(json.dumps(response))
                
            else:
                await websocket.send_text(json.dumps({
                    'type': 'error',
                    'message': f"Unknown message type: {message['type']}"
                }))
                
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
        connected_clients.remove(websocket)
        
        # Clean up session
        if session_id in tutorial_system.active_sessions:
            del tutorial_system.active_sessions[session_id]
            
    except Exception as e:
        logger.exception(f"❌ WebSocket error for {session_id}")
        await websocket.send_text(json.dumps({
            'type': 'error',
            'message': f"Server error: {str(e)}"
        }))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_tutorial_system()
    logger.info("🎓 Adaptive Tutorial Agent is ready!")

# Main execution
if __name__ == "__main__":
    logger.info("🚀 Starting Adaptive Tutorial Agent Server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
