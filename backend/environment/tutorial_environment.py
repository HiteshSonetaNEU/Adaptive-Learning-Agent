import random
import numpy as np
import time
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

class TutorialEnvironment:
    """
    Tutorial Environment for Adaptive Learning System
    
    This environment simulates a student taking a quiz and provides:
    - Question generation based on topic and difficulty
    - Student performance simulation
    - Reward calculation for RL agents
    - State representation for learning
    """
    
    def __init__(self, topics: List[str], difficulty_levels: List[str], question_types: List[str]):
        self.topics = topics
        self.difficulty_levels = difficulty_levels
        self.question_types = question_types
        
        # Student profile simulation
        self.student_profile = {
            'topic_mastery': {topic: random.uniform(0.3, 0.7) for topic in topics},
            'overall_mastery': 0.5,
            'learning_rate': random.uniform(0.01, 0.05),
            'difficulty_preference': 'medium',
            'attention_span': random.randint(5, 15)
        }
        
        # Session state
        self.current_topic = random.choice(topics)
        self.current_difficulty = 'medium'
        self.current_question_type = 'multiple_choice'
        self.questions_answered = 0
        self.correct_answers = 0
        self.session_start_time = time.time()
        
        # Question database
        self.question_db = self._load_or_generate_question_database()
        
        # Current question
        self.current_question = None
        self.last_decision_reasoning = ""
        self.asked_question_ids = set()
        self.asked_question_texts = set()
        
        # Performance tracking
        self.answer_history = []
        self.reward_history = []
        self.state_history = []
        self.correct_streak = 0
        
        print(f"🎓 Tutorial Environment initialized")
        print(f"   Topics: {topics}")
        print(f"   Student mastery: {self.student_profile['topic_mastery']}")
    
    def _load_or_generate_question_database(self) -> Dict:
        """Load curated questions from JSON if available, else generate placeholders"""
        content_path = os.path.join(os.path.dirname(__file__), "..", "content", "questions.json")
        content_path = os.path.normpath(content_path)
        if os.path.exists(content_path):
            try:
                with open(content_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Normalize structure to ensure topics/difficulties exist
                for topic in self.topics:
                    data.setdefault(topic, {})
                    for diff in self.difficulty_levels:
                        data[topic].setdefault(diff, [])
                return data
            except Exception:
                pass

        # Fallback: generate simple placeholders
        questions = {}
        for topic in self.topics:
            questions[topic] = {}
            for difficulty in self.difficulty_levels:
                questions[topic][difficulty] = []
                for i in range(5):
                    q_type = random.choice(self.question_types)
                    questions[topic][difficulty].append({
                        'id': f"{topic}_{difficulty}_{i}",
                        'text': f"Sample {topic} question at {difficulty} level #{i+1}",
                        'type': q_type,
                        'topic': topic,
                        'difficulty': difficulty,
                        'correct_answer': 'A',
                        'options': ['A', 'B', 'C', 'D'] if q_type == 'multiple_choice' else None,
                        'hint': f"Think about the basic principles of {topic}",
                        'explanation': f"This {difficulty} {topic} question tests understanding of core concepts."
                    })
        return questions
    
    def reset(self) -> np.ndarray:
        """Reset environment for new session"""
        self.questions_answered = 0
        self.correct_answers = 0
        self.session_start_time = time.time()
        self.current_topic = random.choice(self.topics)
        self.current_difficulty = 'medium'
        self.answer_history = []
        self.reward_history = []
        self.state_history = []
        self.asked_question_ids = set()
        self.asked_question_texts = set()
        self.correct_streak = 0
        
        # Reset student profile slightly (simulate different student each session)
        for topic in self.topics:
            self.student_profile['topic_mastery'][topic] += random.uniform(-0.1, 0.1)
            self.student_profile['topic_mastery'][topic] = max(0.0, min(1.0, self.student_profile['topic_mastery'][topic]))
        
        self.student_profile['overall_mastery'] = np.mean(list(self.student_profile['topic_mastery'].values()))
        
        print(f"🔄 Tutorial session reset. Student overall mastery: {self.student_profile['overall_mastery']:.3f}")
        
        return self.get_current_state()
    
    def get_current_state(self) -> np.ndarray:
        """Get current state representation for RL agents"""
        # State: [topic_id, difficulty_id, recent_accuracy, mastery_scores, question_count, time_factor]
        topic_id = self.topics.index(self.current_topic)
        difficulty_id = self.difficulty_levels.index(self.current_difficulty)
        
        # Recent accuracy (last 5 questions)
        recent_answers = self.answer_history[-5:] if len(self.answer_history) >= 5 else self.answer_history
        recent_accuracy = sum(recent_answers) / len(recent_answers) if recent_answers else 0.5
        
        # Current topic mastery
        current_topic_mastery = self.student_profile['topic_mastery'][self.current_topic]
        
        # Overall progress
        progress_ratio = self.questions_answered / 20  # Assuming max 20 questions per session
        
        # Time factor
        session_time = time.time() - self.session_start_time
        time_factor = min(session_time / 300, 1.0)  # Normalize to 5 minutes max
        
        state = np.array([
            topic_id / len(self.topics),           # 0-1 normalized topic
            difficulty_id / len(self.difficulty_levels),  # 0-1 normalized difficulty
            recent_accuracy,                        # 0-1 recent performance
            current_topic_mastery,                  # 0-1 topic mastery
            self.student_profile['overall_mastery'], # 0-1 overall mastery
            progress_ratio,                         # 0-1 session progress
            time_factor,                           # 0-1 time pressure
            len(self.answer_history) / 10,         # 0-1 normalized question count
            self.correct_answers / max(1, self.questions_answered),  # Overall accuracy this session
            random.uniform(0, 1)                   # Random noise factor
        ])
        
        return state
    
    def get_state_size(self) -> int:
        """Get the size of the state vector"""
        return 10  # As defined in get_current_state()
    
    def get_state(self) -> np.ndarray:
        """Alias for get_current_state() for compatibility"""
        return self.get_current_state()
    
    def generate_question_from_state(self, state: np.ndarray) -> Dict:
        """Generate question based on current state, avoiding duplicates; try to honor suggested type."""
        # Select question pool for current topic/difficulty
        pool = list(self.question_db[self.current_topic][self.current_difficulty])
        # If a suggested type exists from PG, prefer it when available
        if self.current_question_type in ['multiple_choice', 'short_answer']:
            typed_pool = [q for q in pool if q.get('type') == self.current_question_type]
            if typed_pool:
                pool = typed_pool
        # Filter out previously asked by id and normalized text
        pool = [
            q for q in pool
            if q.get('id') not in self.asked_question_ids and
               (q.get('text') or '').strip().lower() not in self.asked_question_texts
        ]

        if pool:
            question = random.choice(pool)
        else:
            # Fallback question
            question = {
                'id': f"fallback_{self.questions_answered}",
                'text': f"What is a key concept in {self.current_topic}?",
                'type': 'multiple_choice',
                'topic': self.current_topic,
                'difficulty': self.current_difficulty,
                'correct_answer': 'A',
                'options': ['Correct answer', 'Wrong answer 1', 'Wrong answer 2', 'Wrong answer 3'],
                'hint': "Think about the fundamentals",
                'explanation': "This tests basic understanding."
            }
        # Track asked questions to avoid repeats
        self.current_question = question
        if question:
            if question.get('id'):
                self.asked_question_ids.add(question['id'])
            if question.get('text'):
                self.asked_question_texts.add(question['text'].strip().lower())
        return question
    
    def generate_simple_question(self) -> Dict:
        """Generate a simple fallback question"""
        return self.generate_question_from_state(self.get_current_state())
    
    def process_answer(self, student_answer: str, question_id: str) -> Tuple[bool, float, str]:
        """Process student answer and return (is_correct, reward, feedback)"""
        if not self.current_question:
            return False, -0.5, "No active question"
        
        # Simulate student performance based on mastery
        topic_mastery = self.student_profile['topic_mastery'][self.current_topic]
        difficulty_factor = {'easy': 0.8, 'medium': 0.6, 'hard': 0.4}[self.current_difficulty]
        
        # Probability of correct answer
        correct_probability = topic_mastery * difficulty_factor
        
        # Determine if answer is correct: prefer exact/keyword match if available
        is_correct = False
        if self.current_question:
            q = self.current_question
            student_answer_norm = (student_answer or "").strip().lower()
            # Exact match for MC and known answers
            if q.get('type') == 'multiple_choice' and q.get('correct_answer') is not None:
                is_correct = student_answer_norm == str(q['correct_answer']).strip().lower()
            elif q.get('acceptable_answers'):
                is_correct = any(student_answer_norm == a.strip().lower() for a in q['acceptable_answers'])
            elif q.get('answer_keywords'):
                # Check that all key keywords appear (lenient grading)
                keywords = [k.strip().lower() for k in q['answer_keywords'] if isinstance(k, str)]
                is_correct = all(k in student_answer_norm for k in keywords) and len(student_answer_norm) > 0
            else:
                # Fallback to simulated probability if no ground truth
                is_correct = random.random() < correct_probability
        else:
            is_correct = random.random() < correct_probability
        
        # Update tracking
        self.questions_answered += 1
        if is_correct:
            self.correct_answers += 1
            self.correct_streak = self.correct_streak + 1
        else:
            self.correct_streak = 0
        
        self.answer_history.append(1 if is_correct else 0)
        
        # Calculate reward
        reward = self._calculate_reward(is_correct)
        self.reward_history.append(reward)
        
        # Update student mastery (simulate learning)
        if is_correct:
            # Correct answers slightly improve mastery
            improvement = self.student_profile['learning_rate'] * (1 - topic_mastery)
            self.student_profile['topic_mastery'][self.current_topic] += improvement
        else:
            # Wrong answers might slightly decrease confidence
            decrease = self.student_profile['learning_rate'] * 0.1
            self.student_profile['topic_mastery'][self.current_topic] -= decrease
        
        # Keep mastery in bounds
        self.student_profile['topic_mastery'][self.current_topic] = max(0.0, min(1.0, 
            self.student_profile['topic_mastery'][self.current_topic]))
        
        # Update overall mastery
        self.student_profile['overall_mastery'] = np.mean(list(self.student_profile['topic_mastery'].values()))
        
        # Generate feedback
        feedback = self._generate_feedback(is_correct)
        
        return is_correct, reward, feedback
    
    def _calculate_reward(self, is_correct: bool) -> float:
        """Calculate reward for RL agents based on student performance"""
        base_reward = 1.0 if is_correct else -0.5
        
        # Bonus for appropriate difficulty
        topic_mastery = self.student_profile['topic_mastery'][self.current_topic]
        difficulty_appropriateness = self._calculate_difficulty_appropriateness(topic_mastery)
        
        # Bonus for engagement (not too easy, not too hard)
        engagement_bonus = 0.2 if 0.3 < difficulty_appropriateness < 0.8 else -0.1
        
        # Bonus for learning progress
        if len(self.answer_history) >= 3:
            recent_improvement = sum(self.answer_history[-3:]) - sum(self.answer_history[-6:-3] if len(self.answer_history) >= 6 else [])
            progress_bonus = 0.1 * recent_improvement
        else:
            progress_bonus = 0
        
        total_reward = base_reward + engagement_bonus + progress_bonus
        
        return total_reward
    
    def _calculate_difficulty_appropriateness(self, mastery: float) -> float:
        """Calculate how appropriate current difficulty is for student mastery"""
        difficulty_scores = {'easy': 0.3, 'medium': 0.6, 'hard': 0.9}
        target_difficulty = difficulty_scores[self.current_difficulty]
        
        # Ideal difficulty should be slightly above current mastery
        ideal_target = mastery + 0.1
        
        # Return appropriateness score (1.0 = perfect, 0.0 = very inappropriate)
        appropriateness = 1.0 - abs(target_difficulty - ideal_target)
        return max(0.0, appropriateness)
    
    def apply_teaching_actions(self, q_action: int, pg_action: int) -> Tuple[np.ndarray, float]:
        """Apply RL agents' teaching actions to environment"""
        teaching_reward = 0.0
        
        # Guardrail: if student is on a correct streak, nudge harder or transition topic
        if self.correct_streak >= 2:
            if self.current_difficulty in ['easy', 'medium']:
                q_action = 2  # make_harder
            elif self.current_difficulty == 'hard':
                q_action = 3  # change_topic

        # Q-Learning action (difficulty adaptation)
        if q_action == 0:  # make_easier
            if self.current_difficulty == 'hard':
                self.current_difficulty = 'medium'
                teaching_reward += 0.1
            elif self.current_difficulty == 'medium':
                self.current_difficulty = 'easy'
                teaching_reward += 0.1
            self.last_decision_reasoning = "🔻 Reduced difficulty to help student"
            
        elif q_action == 2:  # make_harder
            if self.current_difficulty == 'easy':
                self.current_difficulty = 'medium'
                teaching_reward += 0.1
            elif self.current_difficulty == 'medium':
                self.current_difficulty = 'hard'
                teaching_reward += 0.1
            self.last_decision_reasoning = "🔺 Increased difficulty to challenge student"
            
        elif q_action == 3:  # change_topic
            old_topic = self.current_topic
            available_topics = [t for t in self.topics if t != self.current_topic]
            if available_topics:
                self.current_topic = random.choice(available_topics)
                teaching_reward += 0.05
                self.last_decision_reasoning = f"🔄 Changed topic from {old_topic} to {self.current_topic}"
        
        # Policy Gradient action (content adaptation)
        content_actions = {
            0: ("📚 Using conceptual questions", 'short_answer'),
            1: ("💪 Providing practice problems", 'multiple_choice'), 
            2: ("🌍 Showing real-world applications", 'short_answer'),
            3: ("🔄 Topic transition", None),
            4: ("🔍 Checking prerequisites", 'multiple_choice'),
            5: ("🚀 Offering advanced challenges", 'short_answer')
        }
        
        if pg_action in content_actions:
            note, suggested_type = content_actions[pg_action]
            self.last_decision_reasoning += f" | {note}"
            teaching_reward += 0.02
            if suggested_type in ['multiple_choice', 'short_answer']:
                self.current_question_type = suggested_type
            if pg_action == 3:
                old_topic = self.current_topic
                available_topics = [t for t in self.topics if t != self.current_topic]
                if available_topics:
                    self.current_topic = random.choice(available_topics)
                    teaching_reward += 0.05
                    self.last_decision_reasoning += f" | 🔄 PG transitioned topic to {self.current_topic}"
        
        return self.get_current_state(), teaching_reward

    # --------- Procedural question augmentation (optional) ---------
    def augment_math_questions(self, min_per_difficulty: int = 20):
        """Ensure there are at least min_per_difficulty math questions per difficulty by generating arithmetic items"""
        topic = 'Math'
        if topic not in self.question_db:
            return
        for difficulty in self.difficulty_levels:
            lst = self.question_db[topic].setdefault(difficulty, [])
            needed = max(0, min_per_difficulty - len(lst))
            if needed <= 0:
                continue
            generated = []
            for i in range(needed):
                if difficulty == 'easy':
                    a, b = random.randint(1, 9), random.randint(1, 9)
                    text = f"What is {a} + {b}?"
                    correct = str(a + b)
                    options = [str(a + b), str(a + b - 1), str(a + b + 1), str(abs(a - b))]
                elif difficulty == 'medium':
                    a, b = random.randint(10, 99), random.randint(2, 12)
                    text = f"What is {a} × {b}?"
                    correct = str(a * b)
                    options = [correct, str(a*b + 10), str(a*b - 10), str(a*(b-1))]
                else:  # hard
                    a = random.randint(2, 5)
                    text = f"What is the derivative of x^{a}?"
                    correct = f"{a}x^{a-1}" if a-1 != 1 else f"{a}x"
                    options = [correct, f"x^{a}", f"{a+1}x^{a}", f"{a-1}x^{a-2}"]
                q = {
                    'id': f"gen_{topic}_{difficulty}_{i}_{random.randint(0, 1_000_000)}",
                    'type': 'multiple_choice',
                    'text': text,
                    'options': options,
                    'correct_answer': correct,
                    'topic': topic,
                    'difficulty': difficulty,
                    'hint': "Consider the basic rule involved.",
                    'explanation': "Basic arithmetic/calculus rule."
                }
                generated.append(q)
            lst.extend(generated)

    def balance_topic_pools(self, target_per_difficulty: int = 20):
        """Ensure each topic has at least target_per_difficulty items per difficulty by cloning/adapting existing ones."""
        for topic in self.topics:
            for difficulty in self.difficulty_levels:
                pool = self.question_db.setdefault(topic, {}).setdefault(difficulty, [])
                if len(pool) >= target_per_difficulty:
                    continue
                if not pool:
                    # Seed with a simple generic MCQ
                    pool.append({
                        'id': f'seed_{topic}_{difficulty}_{random.randint(0,1_000_000)}',
                        'type': 'multiple_choice',
                        'text': f"Which is related to {topic}?",
                        'options': ['Concept A', 'Concept B', 'Concept C', 'Concept D'],
                        'correct_answer': 'Concept A',
                        'topic': topic,
                        'difficulty': difficulty,
                        'hint': f"Recall basics of {topic}.",
                        'explanation': f"Basic {topic} concept."
                    })
                # Clone variations
                base = list(pool)
                need = target_per_difficulty - len(pool)
                for i in range(need):
                    src = random.choice(base)
                    clone = dict(src)
                    clone['id'] = f"clone_{topic}_{difficulty}_{i}_{random.randint(0,1_000_000)}"
                    # Slight text variation
                    clone['text'] = src.get('text', '') + " (variant)"
                    pool.append(clone)
    
    def _generate_feedback(self, is_correct: bool) -> str:
        """Generate feedback for student"""
        if is_correct:
            feedback_options = [
                "🎉 Excellent! You're mastering this concept.",
                "✅ Correct! Great understanding.",
                "👏 Well done! Keep up the good work.",
                "🌟 Perfect! You're making great progress."
            ]
        else:
            feedback_options = [
                "💭 Not quite right. Let's think about this differently.",
                "🤔 Close, but let's review the key concepts.",
                "📖 Let's break this down step by step.",
                "💡 Don't worry! Learning takes practice."
            ]
        
        return random.choice(feedback_options)
    
    def generate_hint(self, question_id: str) -> str:
        """Generate hint for current question"""
        if self.current_question:
            return self.current_question.get('hint', 'Think about the basic principles.')
        return "Break down the problem into smaller parts."
    
    def get_student_progress(self) -> Dict:
        """Get current student progress"""
        return {
            'questions_answered': self.questions_answered,
            'correct_answers': self.correct_answers,
            'accuracy': self.get_accuracy(),
            'current_topic': self.current_topic,
            'current_difficulty': self.current_difficulty,
            'topic_mastery': self.student_profile['topic_mastery'],
            'overall_mastery': self.student_profile['overall_mastery']
        }
    
    def get_accuracy(self) -> float:
        """Get current session accuracy"""
        if self.questions_answered == 0:
            return 0.0
        return self.correct_answers / self.questions_answered
    
    def get_session_time(self) -> float:
        """Get session time in minutes"""
        return (time.time() - self.session_start_time) / 60
    
    def get_last_decision_reasoning(self) -> str:
        """Get explanation of last RL decision"""
        return self.last_decision_reasoning
