import React, { useState, useEffect } from 'react';
import './TutorialInterface.css';

const TutorialInterface = ({ question, onSubmitAnswer, onRequestHint, connected }) => {
  const [selectedAnswer, setSelectedAnswer] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  useEffect(() => {
    // Reset state when new question arrives
    if (question) {
      setSelectedAnswer('');
      setIsSubmitting(false);
      setShowFeedback(false);
    }
  }, [question]);

  const handleSubmit = async () => {
    if (!selectedAnswer.trim() || isSubmitting || !connected) return;
    
    setIsSubmitting(true);
    setShowFeedback(true);
    
    try {
      await onSubmitAnswer(selectedAnswer);
    } catch (error) {
      console.error('Error submitting answer:', error);
    }
    
    setIsSubmitting(false);
  };

  const handleHintRequest = () => {
    if (!connected || isSubmitting) return;
    onRequestHint();
  };

  const handleAnswerChange = (value) => {
    setSelectedAnswer(value);
    setShowFeedback(false);
  };

  if (!question) {
    return (
      <div className="tutorial-interface loading">
        <div className="loading-content">
          <div className="spinner"></div>
          <p>Generating your next question...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="tutorial-interface">
      <div className="question-header">
        <div className="question-meta">
          <span className="topic-badge">{question.topic}</span>
          <span className="difficulty-badge difficulty-{question.difficulty}">
            {question.difficulty}
          </span>
          <span className="type-badge">{question.type}</span>
        </div>
        
        {question.adaptive_reasoning && (
          <div className="adaptive-info">
            <span className="adaptive-badge">🤖 AI Adapted</span>
            <span className="reasoning">{question.adaptive_reasoning}</span>
          </div>
        )}
      </div>

      <div className="question-content">
        <h2 className="question-text">{question.text}</h2>
        
        {question.context && (
          <div className="question-context">
            <h4>Context:</h4>
            <p>{question.context}</p>
          </div>
        )}
      </div>

      <div className="answer-section">
         {question.type === 'multiple_choice' ? (
          <div className="multiple-choice">
            {(question.options || []).map((option, index) => (
              <label key={index} className="option-label">
                <input
                  type="radio"
                  name="answer"
                  value={option}
                  checked={selectedAnswer === option}
                  onChange={(e) => handleAnswerChange(e.target.value)}
                  disabled={isSubmitting}
                />
                <span className="option-text">{option}</span>
              </label>
            ))}
          </div>
         ) : (
          <div className="text-answer">
            <textarea
              value={selectedAnswer}
              onChange={(e) => handleAnswerChange(e.target.value)}
              placeholder={question.placeholder || "Type your answer here... (e.g., define, explain, compute)"}
              disabled={isSubmitting}
              rows={4}
              className="answer-input"
            />
            {question.example && (
              <div className="question-context">
                <h4>Example:</h4>
                <p>{question.example}</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Hints removed from UI */}

      <div className="action-buttons">
        {/* Hint button removed */}
        
        <button
          className="submit-button"
          onClick={handleSubmit}
          disabled={!selectedAnswer.trim() || isSubmitting || !connected}
        >
          {isSubmitting ? (
            <>
              <span className="spinner-small"></span>
              Processing...
            </>
          ) : (
            '✓ Submit Answer'
          )}
        </button>
      </div>

      {showFeedback && (
        <div className="feedback-section">
          <div className="processing-indicator">
            <span className="spinner-small"></span>
            <p>AI agents are analyzing your response...</p>
          </div>
        </div>
      )}

      {!connected && (
        <div className="connection-warning">
          ⚠️ Connection lost. Attempting to reconnect...
        </div>
      )}
    </div>
  );
};

export default TutorialInterface;
