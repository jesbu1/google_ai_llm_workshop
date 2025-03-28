import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

// API base URL
const API_URL = 'http://localhost:8000';

const App = () => {
  // User state
  const [userId, setUserId] = useState(localStorage.getItem('userId') || '');
  const [userExists, setUserExists] = useState(false);
  
  // Text generation state
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [maxLength, setMaxLength] = useState(100);
  const [temperature, setTemperature] = useState(0.8);
  
  // Training state
  const [trainingText, setTrainingText] = useState('');
  const [epochs, setEpochs] = useState(5);
  const [batchSize, setBatchSize] = useState(32);
  const [isTraining, setIsTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState([]);
  
  // Loading states
  const [isGenerating, setIsGenerating] = useState(false);
  const [message, setMessage] = useState('');
  
  // Check if a user model exists
  useEffect(() => {
    const checkUserExists = async () => {
      if (userId) {
        try {
          const response = await axios.get(`${API_URL}/users/${userId}/exists`);
          setUserExists(response.data.exists);
        } catch (error) {
          console.error('Error checking user:', error);
        }
      }
    };
    
    checkUserExists();
  }, [userId]);
  
  // Save user ID to localStorage when it changes
  useEffect(() => {
    if (userId) {
      localStorage.setItem('userId', userId);
    }
  }, [userId]);
  
  // Create a new user
  const createNewUser = async () => {
    try {
      setMessage('Creating new user...');
      
      // Send a training request with "new" as the user ID to create one
      const response = await axios.post(`${API_URL}/train`, {
        user_id: 'new',
        training_text: 'Hello world',  // Just a placeholder
        epochs: 1,
        batch_size: 8,
        create_new: true
      });
      
      // Set the new user ID
      setUserId(response.data.user_id);
      setUserExists(true);
      setMessage(`New user created with ID: ${response.data.user_id}`);
      
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('Error creating user:', error);
      setMessage('Error creating user');
    }
  };
  
  // Generate text
  const handleGenerate = async (e) => {
    e.preventDefault();
    
    if (!userId) {
      setMessage('Please create a user or enter a user ID first');
      return;
    }
    
    try {
      setIsGenerating(true);
      
      const response = await axios.post(`${API_URL}/generate`, {
        user_id: userId,
        prompt,
        max_length: maxLength,
        temperature
      });
      
      setGeneratedText(response.data.generated_text);
      setIsGenerating(false);
    } catch (error) {
      console.error('Error generating text:', error);
      setMessage('Error generating text');
      setIsGenerating(false);
    }
  };
  
  // Train model
  const handleTraining = async (e) => {
    e.preventDefault();
    
    if (!userId) {
      setMessage('Please create a user or enter a user ID first');
      return;
    }
    
    if (!trainingText.trim()) {
      setMessage('Please enter some training text');
      return;
    }
    
    try {
      setIsTraining(true);
      setMessage('Training model...');
      
      const response = await axios.post(`${API_URL}/train`, {
        user_id: userId,
        training_text: trainingText,
        epochs,
        batch_size: batchSize,
        create_new: false
      });
      
      setLossHistory(response.data.loss_history);
      setIsTraining(false);
      setMessage('Model trained successfully!');
      
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('Error training model:', error);
      setMessage('Error training model');
      setIsTraining(false);
    }
  };
  
  // Chart data and options
  const chartData = {
    labels: lossHistory.map((_, index) => `Epoch ${index + 1}`),
    datasets: [
      {
        label: 'Training Loss',
        data: lossHistory,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.1
      }
    ]
  };
  
  const chartOptions = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Training Loss'
      }
    }
  };
  
  return (
    <div className="container mt-5">
      <div className="header">
        <h1>Personal Text Generation Model</h1>
        <p className="text-muted">Train your own language model and generate text</p>
      </div>
      
      {/* User Section */}
      <div className="card mb-4">
        <div className="card-body">
          <h3 className="card-title">User Information</h3>
          
          {!userId ? (
            <div>
              <p>No user selected. Create a new user or enter an existing user ID.</p>
              <button 
                className="btn btn-primary me-2" 
                onClick={createNewUser}
              >
                Create New User
              </button>
              <div className="mt-3">
                <input 
                  type="text" 
                  className="form-control" 
                  placeholder="Or enter existing user ID"
                  value={userId}
                  onChange={(e) => setUserId(e.target.value)}
                />
              </div>
            </div>
          ) : (
            <div>
              <p>Current User ID: <strong>{userId}</strong></p>
              <p>
                User Model Status: 
                <span className={`badge bg-${userExists ? 'success' : 'warning'} ms-2`}>
                  {userExists ? 'Exists' : 'Not Found'}
                </span>
              </p>
              <button 
                className="btn btn-sm btn-outline-danger"
                onClick={() => {
                  setUserId('');
                  localStorage.removeItem('userId');
                }}
              >
                Change User
              </button>
            </div>
          )}
        </div>
      </div>
      
      {/* Message Alert */}
      {message && (
        <div className="alert alert-info">{message}</div>
      )}
      
      <div className="row">
        {/* Text Generation Section */}
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-body">
              <h3 className="card-title">Text Generation</h3>
              
              <form onSubmit={handleGenerate}>
                <div className="mb-3">
                  <label htmlFor="prompt" className="form-label">Prompt</label>
                  <textarea 
                    id="prompt"
                    className="form-control text-area"
                    placeholder="Enter text to start generation..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    required
                  ></textarea>
                </div>
                
                <div className="row mb-3">
                  <div className="col-md-6">
                    <label htmlFor="maxLength" className="form-label">Max Length</label>
                    <input 
                      type="number" 
                      id="maxLength"
                      className="form-control" 
                      min="10" 
                      max="500"
                      value={maxLength}
                      onChange={(e) => setMaxLength(parseInt(e.target.value))}
                    />
                  </div>
                  <div className="col-md-6">
                    <label htmlFor="temperature" className="form-label">Temperature</label>
                    <input 
                      type="range" 
                      id="temperature"
                      className="form-range" 
                      min="0.1" 
                      max="2" 
                      step="0.1"
                      value={temperature}
                      onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    />
                    <div className="text-center">{temperature}</div>
                  </div>
                </div>
                
                <button 
                  type="submit" 
                  className="btn btn-primary w-100"
                  disabled={isGenerating || !userId}
                >
                  {isGenerating ? 'Generating...' : 'Generate Text'}
                </button>
              </form>
              
              {generatedText && (
                <div className="mt-4">
                  <h5>Generated Text:</h5>
                  <div className="p-3 bg-light border rounded">
                    <p style={{ whiteSpace: 'pre-wrap' }}>{generatedText}</p>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* Training Section */}
        <div className="col-md-6">
          <div className="card h-100">
            <div className="card-body">
              <h3 className="card-title">Model Training</h3>
              
              <form onSubmit={handleTraining}>
                <div className="mb-3">
                  <label htmlFor="trainingText" className="form-label">Training Text</label>
                  <textarea 
                    id="trainingText"
                    className="form-control text-area"
                    placeholder="Enter text to train your model..."
                    value={trainingText}
                    onChange={(e) => setTrainingText(e.target.value)}
                    required
                  ></textarea>
                </div>
                
                <div className="row mb-3">
                  <div className="col-md-6">
                    <label htmlFor="epochs" className="form-label">Epochs</label>
                    <input 
                      type="number" 
                      id="epochs"
                      className="form-control" 
                      min="1" 
                      max="20"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
                    />
                  </div>
                  <div className="col-md-6">
                    <label htmlFor="batchSize" className="form-label">Batch Size</label>
                    <input 
                      type="number" 
                      id="batchSize"
                      className="form-control" 
                      min="1" 
                      max="64"
                      value={batchSize}
                      onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    />
                  </div>
                </div>
                
                <button 
                  type="submit" 
                  className="btn btn-primary w-100"
                  disabled={isTraining || !userId}
                >
                  {isTraining ? 'Training...' : 'Train Model'}
                </button>
              </form>
              
              {lossHistory.length > 0 && (
                <div className="mt-4">
                  <h5>Training Results:</h5>
                  <div style={{ height: '200px' }}>
                    <Line data={chartData} options={chartOptions} />
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      <footer className="mt-5 text-center text-muted">
        <p>Train a simple language model and generate text based on your input.</p>
      </footer>
    </div>
  );
};

export default App; 