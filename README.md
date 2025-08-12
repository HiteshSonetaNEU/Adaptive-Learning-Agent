# 🤖 Adaptive Tutorial Agent

An AI-powered educational system that uses Reinforcement Learning to personalize the learning experience for each student.

Repository: [Adaptive-Learning-Agent](https://github.com/HiteshSonetaNEU/Adaptive-Learning-Agent)
Demo Video: [YouTube Demo](https://www.youtube.com/watch?v=FjvA1jOP2cg&ab_channel=HiteshSoneta)

## Project Overview

This system implements two RL agents that work together to create an adaptive tutorial experience:
- Q-Learning Agent: adapts question difficulty
- Policy Gradient Agent (REINFORCE): selects content and teaching strategies

## 🏗️ Architecture

### Backend (FastAPI + RL Agents)
- `main.py` - FastAPI server with WebSocket support
- `agents/q_learning_tutor.py` - Q-Learning agent for difficulty adaptation
- `agents/policy_gradient_tutor.py` - Policy Gradient agent for content selection
- `environment/tutorial_environment.py` - Tutorial environment simulation
- `agents/training_manager.py` - Multi-agent training coordination

### Frontend (React)
- Interactive tutorial interface
- Real-time learning analytics dashboard
- AI agent decision visualization

## 🚀 Quick Start

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
python main.py
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

### Access the Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## 📊 Features

### 🧠 AI-Powered Adaptation
- **Dynamic Difficulty**: Q-Learning agent adjusts question difficulty in real-time
- **Content Selection**: Policy Gradient agent chooses optimal learning materials
- **Personalized Hints**: Context-aware hint generation
- **Performance Tracking**: Continuous learning analytics

### 🎓 Educational Features
- Multiple subjects (Math, Science, History, Literature)
- Various question types (Multiple Choice, Short Answer, Essay)
- Adaptive difficulty levels (Beginner, Intermediate, Advanced)
- Real-time feedback and hints

### 📈 Analytics Dashboard
- Student performance metrics
- AI agent decision insights
- Learning progress visualization
- Topic-specific performance tracking

## 🧪 Training the Agents

### Using Google Colab (Recommended)
- Open Colab and use the notebook: `colab/Complete_Tutorial_Agent_All_In_One.ipynb`.
- Run all cells to train and export models.

### Local Training (example)
```bash
cd backend
python -c "from agents.training_manager import TutorTrainingManager; from environment.tutorial_environment import TutorialEnvironment; from agents.q_learning_tutor import TutorQLearning; from agents.policy_gradient_tutor import TutorPolicyGradient; env=TutorialEnvironment(['Math','Science','History','Literature'], ['easy','medium','hard'], ['multiple_choice','short_answer','essay']); mgr=TutorTrainingManager(env, TutorQLearning(env.get_state_size(),7), TutorPolicyGradient(env.get_state_size(),5)); [mgr.train_episode() for _ in range(100)]; mgr.save_models('final')"
```

## 🔧 Configuration

### Environment Settings
```python
# In tutorial_environment.py
TOPICS = ["Math", "Science", "History", "Literature"]
DIFFICULTY_LEVELS = ["beginner", "intermediate", "advanced"]
QUESTION_TYPES = ["multiple_choice", "short_answer", "essay"]
```

### Agent Parameters
```python
# Q-Learning Agent
learning_rate = 0.1
epsilon = 0.1
gamma = 0.95

# Policy Gradient Agent
learning_rate = 0.001
gamma = 0.99
```

## 📋 API Endpoints

### WebSocket Connection
- `ws://localhost:8000/ws` - Real-time tutorial session

### REST API
- `GET /api/status` - System status and configuration
- `POST /api/reset` - Reset agent states
- `GET /api/stats` - Learning statistics

## 🎯 Key RL Components

### State Representation
- Student performance metrics
- Question difficulty history
- Topic mastery levels
- Response time patterns

### Action Spaces
- **Q-Learning**: `[easier, harder, change_topic, hint, review, continue, assess]`
- **Policy Gradient**: `[concept_intro, practice, example, assessment, review]`

### Reward Functions
- Correct answers: +1.0
- Improved performance: +0.5
- Engagement maintenance: +0.3
- Optimal difficulty: +0.2

## 🛠️ Technology Stack

- **Backend**: FastAPI, PyTorch, NumPy, WebSockets
- **Frontend**: React, CSS3, WebSocket API
- **ML**: Q-Learning, REINFORCE Policy Gradient
- **Training**: Google Colab compatible

## 📚 Educational Impact

This system demonstrates how RL can revolutionize education by:
- Personalizing learning paths for individual students
- Adapting in real-time to student needs
- Optimizing engagement and learning efficiency
- Providing data-driven insights for educators

## 🔮 Future Enhancements

- Multi-modal learning (text, images, videos)
- Advanced NLP for automated question generation
- Collaborative learning environments
- Integration with existing LMS platforms

## 📊 Reproducible Evaluation and Figures

We provide a script to produce learning curves, comparative analyses, and statistical validation, and to export artifacts to `docs/`.

### Generate results and figures
```bash
cd backend
python -m backend.scripts.evaluate --sessions 50 --steps 15 --out ../docs --seed 42
```

This will create:
- `docs/results/` with CSV summaries and `statistical_summary.json` (t-test, Wilcoxon, 95% CI)
- `docs/figures/` with accuracy distribution and per-session accuracy plots

### Expected artifacts to commit
- `docs/figures/accuracy_distribution.png`
- `docs/figures/accuracy_over_sessions.png`
- `docs/results/baseline_summary_*.csv`
- `docs/results/trained_summary_*.csv`
- `docs/results/statistical_summary.json`

## 🧭 Project Structure
```
backend/
  ├─ agents/
  ├─ environment/
  ├─ models/
  └─ scripts/
frontend/
colab/
docs/
```

## Reports and Demo
- PDFs: `docs/Experimental Design and Results - Adaptive Learning Agent.pdf`, `docs/Technical Report - Reinforcement Learning for Adaptive Educational Agents.pdf`
- Demo video: [YouTube Demo](https://www.youtube.com/watch?v=FjvA1jOP2cg&ab_channel=HiteshSoneta)

## Requirements Met
- Two RL Approaches (Q-Learning + Policy Gradient)
- Real-time Learning Visualization (Frontend)
- Evaluation with statistical validation and figures (via evaluation script)

## License
Educational use only. Add a `LICENSE` file if needed.
