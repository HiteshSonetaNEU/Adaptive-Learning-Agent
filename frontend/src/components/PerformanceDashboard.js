import React from 'react';
import './PerformanceDashboard.css';

const PerformanceDashboard = ({ studentProgress, agentDecisions, learningStats }) => {
  const formatPercentage = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatDecimal = (value) => {
    if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
    return Number(value).toFixed(3);
  };

  const qActionNames = [
    'make_easier',
    'keep_same',
    'make_harder',
    'change_topic',
    'provide_hint',
    'review_mode',
    'adaptive_pace'
  ];

  const pgActionNames = [
    'conceptual_question',
    'practice_problem',
    'application_example',
    'topic_transition',
    'prerequisite_check',
    'advanced_challenge',
    'interactive_exercise'
  ];

  const displayQAction = (action) => {
    if (action === null || action === undefined) return 'N/A';
    return qActionNames[action] || String(action);
  };

  const displayPGStrategy = (strategy) => {
    if (strategy === null || strategy === undefined) return 'N/A';
    return pgActionNames[strategy] || String(strategy);
  };

  const getDifficultyColor = (level) => {
    const colors = {
      'beginner': '#4CAF50',
      'intermediate': '#FF9800', 
      'advanced': '#F44336'
    };
    return colors[level] || '#666';
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return '#4CAF50'; // Green
    if (score >= 0.6) return '#FF9800'; // Orange
    return '#F44336'; // Red
  };

  return (
    <div className="performance-dashboard">
      <h3 className="dashboard-title">📊 Learning Analytics</h3>

      {/* Student Progress Section */}
      {studentProgress && (
        <div className="dashboard-section">
          <h4 className="section-title">👤 Student Performance</h4>
          <div className="metrics-grid">
            <div className="metric-card">
              <span className="metric-label">Current Score</span>
              <span 
                className="metric-value"
                style={{ color: getScoreColor(studentProgress.current_score) }}
              >
                {formatPercentage(studentProgress.current_score)}
              </span>
            </div>
            
            <div className="metric-card">
              <span className="metric-label">Questions Answered</span>
              <span className="metric-value">
                {studentProgress.questions_answered || 0}
              </span>
            </div>
            
            <div className="metric-card">
              <span className="metric-label">Correct Answers</span>
              <span className="metric-value">
                {studentProgress.correct_answers || 0}
              </span>
            </div>
            
            <div className="metric-card">
              <span className="metric-label">Current Difficulty</span>
              <span 
                className="metric-value difficulty-badge"
                style={{ backgroundColor: getDifficultyColor(studentProgress.current_difficulty) }}
              >
                {studentProgress.current_difficulty || 'N/A'}
              </span>
            </div>
            
            <div className="metric-card">
              <span className="metric-label">Learning Rate</span>
              <span className="metric-value">
                {formatDecimal(studentProgress.learning_rate)}
              </span>
            </div>
            
            <div className="metric-card">
              <span className="metric-label">Session Time</span>
              <span className="metric-value">
                {studentProgress.session_time ? `${Math.floor(studentProgress.session_time / 60)}m ${studentProgress.session_time % 60}s` : 'N/A'}
              </span>
            </div>
          </div>

          {studentProgress.topic_performance && (
            <div className="topic-performance">
              <h5>📚 Topic Performance</h5>
              <div className="topic-list">
                {Object.entries(studentProgress.topic_performance).map(([topic, score]) => (
                  <div key={topic} className="topic-item">
                    <span className="topic-name">{topic}</span>
                    <div className="topic-score-bar">
                      <div 
                        className="topic-score-fill"
                        style={{ 
                          width: `${score * 100}%`,
                          backgroundColor: getScoreColor(score)
                        }}
                      ></div>
                      <span className="topic-score-text">{formatPercentage(score)}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Agent Decisions Section */}
      {agentDecisions && (
        <div className="dashboard-section">
          <h4 className="section-title">🤖 AI Agent Decisions</h4>
          
          <div className="agent-section">
            <h5 className="agent-title">🧠 Q-Learning Agent</h5>
            <div className="agent-metrics">
              <div className="metric-row">
                <span className="metric-label">Action Taken:</span>
                <span className="metric-value action-badge">
                  {displayQAction(agentDecisions.q_learning?.action)}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Q-Value:</span>
                <span className="metric-value">
                  {formatDecimal(agentDecisions.q_learning?.q_value)}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Exploration Rate:</span>
                <span className="metric-value">
                  {formatPercentage(agentDecisions.q_learning?.epsilon)}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Reward:</span>
                <span className="metric-value">
                  {formatDecimal(agentDecisions.q_learning?.reward)}
                </span>
              </div>
            </div>
          </div>

          <div className="agent-section">
            <h5 className="agent-title">🎯 Policy Gradient Agent</h5>
            <div className="agent-metrics">
              <div className="metric-row">
                <span className="metric-label">Content Strategy:</span>
                <span className="metric-value strategy-badge">
                  {displayPGStrategy(agentDecisions.policy_gradient?.strategy)}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Policy Probability:</span>
                <span className="metric-value">
                  {formatPercentage(agentDecisions.policy_gradient?.probability)}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Baseline Value:</span>
                <span className="metric-value">
                  {formatDecimal(agentDecisions.policy_gradient?.baseline)}
                </span>
              </div>
              <div className="metric-row">
                <span className="metric-label">Advantage:</span>
                <span className="metric-value">
                  {formatDecimal(agentDecisions.policy_gradient?.advantage)}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Learning Statistics */}
      {learningStats && (
        <div className="dashboard-section">
          <h4 className="section-title">📈 Learning Statistics</h4>
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-label">Avg Response Time</span>
              <span className="stat-value">
                {learningStats.avg_response_time ? `${learningStats.avg_response_time.toFixed(1)}s` : 'N/A'}
              </span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Learning Efficiency</span>
              <span className="stat-value">
                {formatPercentage(learningStats.learning_efficiency)}
              </span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Improvement Rate</span>
              <span className="stat-value">
                {formatPercentage(learningStats.improvement_rate)}
              </span>
            </div>
            
            <div className="stat-card">
              <span className="stat-label">Engagement Level</span>
              <span className="stat-value">
                {formatPercentage(learningStats.engagement_level)}
              </span>
            </div>
          </div>

          {learningStats.recent_performance && (
            <div className="recent-performance">
              <h5>📊 Recent Performance Trend</h5>
              <div className="performance-trend">
                {learningStats.recent_performance.map((score, index) => (
                  <div 
                    key={index} 
                    className="performance-bar"
                    style={{ 
                      height: `${score * 100}%`,
                      backgroundColor: getScoreColor(score)
                    }}
                    title={`Question ${index + 1}: ${formatPercentage(score)}`}
                  ></div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Empty State */}
      {!studentProgress && !agentDecisions && !learningStats && (
        <div className="empty-state">
          <div className="empty-content">
            <span className="empty-icon">📊</span>
            <p>Start answering questions to see your learning analytics!</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default PerformanceDashboard;
