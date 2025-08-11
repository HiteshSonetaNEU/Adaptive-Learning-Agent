# 🤖 Adaptive Tutorial Agent

An AI-powered educational system that uses **Reinforcement Learning** to personalize the learning experience for each student.

## � Project Overview

This system implements two RL agents that work together to create an adaptive tutorial experience:
- **Q-Learning Agent**: Adapts question difficulty based on student performance
- **Policy Gradient Agent**: Selects optimal content and teaching strategies

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
```python
# Upload training_manager.py and agent files to Colab
from training_manager import TrainingManager

manager = TrainingManager()
manager.train_agents(episodes=1000)
manager.save_models("trained_models/")
```

### Local Training
```bash
cd backend
python -c "from agents.training_manager import TrainingManager; TrainingManager().train_agents(1000)"
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

## 📄 License

This project is for educational purposes and demonstrates the application of Reinforcement Learning in adaptive education systems.
python demo.py
```

### 🧠 Step 2: Train Models in Google Colab (SIMPLIFIED!)
1. **Open Colab**: Go to [Google Colab](https://colab.research.google.com/)
2. **Upload Notebook**: Upload `colab/warehouse_rl_training.ipynb` directly to Colab
3. **Run Training**: Click "Runtime" → "Run All" (takes ~30-60 minutes)
4. **Download Models**: After training, Colab will automatically create a zip file for download

**No need to upload entire folder to Google Drive!** The notebook is self-contained.

### 🚀 Step 3: Run with Trained Models
```bash
# Backend with trained models
python backend/main.py
# Server runs on http://localhost:8000
```

### 🎨 Step 4: Frontend Visualization
```bash
cd frontend
npm install
npm start
# React app runs on http://localhost:3000
```

## Training Instructions

### Required Colab File
- **File**: `colab/warehouse_rl_training_new.ipynb` (OPTIMIZED VERSION ✅)
- **Platform**: Google Colab (recommended for GPU acceleration)
- **Duration**: 30-60 minutes
- **Output**: Trained models + performance metrics

### Model Download & Setup
**✅ MODELS ALREADY DEPLOYED!** The trained models from the optimized Colab session are now in `backend/models/`:

```
backend/models/
├── final_q_learning.pkl            # Q-Learning navigator agents (TRAINED ✅)
├── final_policy_gradient_policy_agent_*.pth    # Task manager neural networks (TRAINED ✅)
├── final_policy_gradient_value_agent_*.pth     # Value networks (TRAINED ✅)
├── final_training_stats.json       # Complete training metrics
├── training_progress.png           # Training visualization charts
├── agent_comparison.png           # Q-Learning vs Policy Gradient comparison
├── normalized_performance.png     # Performance metrics over time
└── experiment_results_summary.json # Performance analysis
```

**Training Results Summary:**
- 🎯 **Collision Rate**: 0.010 (✅ Target <0.05 achieved)
- 📈 **Performance Improvement**: +58.4% over training
- 🧠 **Q-States Explored**: 609,461 unique states
- ⚡ **Training Duration**: 1.62 hours on Google Colab
- 🏆 **Statistical Significance**: p < 0.05 (highly significant learning)

## Current System State
✅ **Demo Working**: Complete system validation successful  
✅ **Training Complete**: Optimized models trained and deployed  
✅ **Production Ready**: Trained models loaded in backend/models/

## Project Structure
```
warehouse_rl_system/
├── backend/                 # FastAPI server
│   ├── environment/        # Grid world environment
│   ├── agents/            # RL agent implementations
│   ├── models/            # Model loading utilities
│   └── api/               # WebSocket & REST APIs
├── frontend/              # React dashboard
├── colab/                 # Training notebooks for Colab
├── models/                # Trained model storage
└── docs/                  # Documentation and results
```

## Requirements Met
- ✅ **Two RL Approaches**: Q-Learning + Policy Gradient
- ✅ **Multi-Agent System**: Coordinated warehouse robots
- ✅ **Agent Orchestration**: Dynamic task allocation
- ✅ **Real-time Learning**: Observable improvement over time
- ✅ **Performance Metrics**: Comprehensive evaluation
