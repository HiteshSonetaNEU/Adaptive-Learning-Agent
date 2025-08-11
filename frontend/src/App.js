import React, { useState, useEffect, useRef } from 'react';
import TutorialInterface from './components/TutorialInterface';
import PerformanceDashboard from './components/PerformanceDashboard';
import './App.css';

function App() {
  const [connected, setConnected] = useState(false);
  const [currentQuestion, setCurrentQuestion] = useState(null);
  const [sessionActive, setSessionActive] = useState(false);
  const [studentProgress, setStudentProgress] = useState(null);
  const [agentDecisions, setAgentDecisions] = useState(null);
  const [learningStats, setLearningStats] = useState(null);
  const [systemStatus, setSystemStatus] = useState(null);
  
  const ws = useRef(null);

  useEffect(() => {
    // Connect to WebSocket
    connectWebSocket();
    
    // Fetch initial system status
    fetchSystemStatus();
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);

  const connectWebSocket = () => {
    ws.current = new WebSocket('ws://localhost:8000/ws');
    
    ws.current.onopen = () => {
      console.log('Connected to tutorial system');
      setConnected(true);
    };
    
    ws.current.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    ws.current.onclose = () => {
      console.log('Disconnected from tutorial system');
      setConnected(false);
      
      // Attempt to reconnect after 3 seconds
      setTimeout(() => {
        if (!connected) {
          connectWebSocket();
        }
      }, 3000);
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const handleWebSocketMessage = (message) => {
    console.log('Received message:', message);
    
    switch (message.type) {
      case 'session_start':
        setCurrentQuestion(message.question);
        setSessionActive(true);
        setStudentProgress(null);
        setAgentDecisions(null);
        break;
        
      case 'answer_processed':
        setCurrentQuestion(message.next_question);
        setStudentProgress(message.student_progress);
        setAgentDecisions(message.agent_decisions);
        setLearningStats(message.learning_stats);
        break;
      case 'session_end':
        // Show end-of-session notice and allow restart
        setSessionActive(false);
        setCurrentQuestion(null);
        setLearningStats(prev => ({
          ...(prev || {}),
          recent_performance: message.recent_performance || (prev?.recent_performance || [])
        }));
        alert(`Session complete!\nQuestions: ${message.total_questions}\nCorrect: ${message.correct_answers}\nAccuracy: ${(message.accuracy*100).toFixed(1)}%`);
        break;
        
      case 'hint_provided':
        // Always show the latest hint on the current question
        setCurrentQuestion(prev => (prev ? { ...prev, hint: message.hint } : prev));
        break;
        
      default:
        console.log('Unknown message type:', message.type);
    }
  };

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/status');
      const status = await response.json();
      setSystemStatus(status);
    } catch (error) {
      console.error('Failed to fetch system status:', error);
    }
  };

  const startSession = () => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        type: 'start_session'
      }));
    }
  };

  const submitAnswer = (answer) => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN && currentQuestion) {
      ws.current.send(JSON.stringify({
        type: 'answer',
        answer: answer,
        question_id: currentQuestion.id
      }));
    }
  };

  // Hint feature removed
  const requestHint = () => {};

  return (
    <div className="app">
      <header className="app-header">
        <h1>🤖 Adaptive Tutorial Agent</h1>
        <div className="connection-status">
          <span className={`status-indicator ${connected ? 'connected' : 'disconnected'}`}>
            {connected ? '🟢 Connected' : '🔴 Disconnected'}
          </span>
          {systemStatus && (
            <span className="system-info">
              Q-Learning: {systemStatus.agents.q_learning ? '✅' : '❌'} | 
              Policy Gradient: {systemStatus.agents.policy_gradient ? '✅' : '❌'}
            </span>
          )}
        </div>
      </header>

      <main className="app-main">
        {!sessionActive ? (
          <div className="welcome-screen">
            <div className="welcome-content">
              <h2>Welcome to the Adaptive Tutorial System!</h2>
              <p>
                This system uses <strong>Reinforcement Learning</strong> to adapt to your learning style:
              </p>
              <ul className="feature-list">
                <li>🧠 <strong>Q-Learning Agent</strong> adapts question difficulty</li>
                <li>🎯 <strong>Policy Gradient Agent</strong> selects optimal content</li>
                <li>📊 Real-time learning analytics</li>
                <li>🎓 Personalized learning experience</li>
              </ul>
              
              {connected ? (
                <button 
                  className="start-button"
                  onClick={startSession}
                >
                  🚀 Start Learning Session
                </button>
              ) : (
                <div className="connection-warning">
                  ⚠️ Connecting to tutorial system...
                </div>
              )}
              
              {systemStatus && (
                <div className="system-status">
                  <h3>System Status</h3>
                  <div className="status-grid">
                    <div className="status-item">
                      <span>Topics Available:</span>
                      <span>{systemStatus.config.topics.join(', ')}</span>
                    </div>
                    <div className="status-item">
                      <span>Difficulty Levels:</span>
                      <span>{systemStatus.config.difficulty_levels.join(', ')}</span>
                    </div>
                    <div className="status-item">
                      <span>Question Types:</span>
                      <span>{systemStatus.config.question_types.join(', ')}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="tutorial-session">
            <div className="tutorial-main">
              <TutorialInterface
                question={currentQuestion}
                onSubmitAnswer={submitAnswer}
                onRequestHint={requestHint}
                connected={connected}
              />
            </div>
            
            <div className="tutorial-sidebar">
              <PerformanceDashboard
                studentProgress={studentProgress}
                agentDecisions={agentDecisions}
                learningStats={learningStats}
              />
            </div>
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>
          🎓 Reinforcement Learning for Adaptive Tutorial Systems | 
          Built with Q-Learning + Policy Gradient Methods
        </p>
      </footer>
    </div>
  );
}

export default App;
