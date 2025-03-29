import React, { useState, useEffect, useRef } from 'react';
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
const API_URL = "http://localhost:8000";

const App = () => {
  // User state
  const [userId, setUserId] = useState(localStorage.getItem('userId') || '');
  const [userExists, setUserExists] = useState(false);
  
  // Text generation state
  const [prompt, setPrompt] = useState('');
  const [generatedText, setGeneratedText] = useState('');
  const [maxLength, setMaxLength] = useState(40);
  const [temperature, setTemperature] = useState(0);
  
  // Training state
  const [trainingText, setTrainingText] = useState('');
  const [epochs, setEpochs] = useState(50);
  const [isTraining, setIsTraining] = useState(false);
  const [lossHistory, setLossHistory] = useState([]);
  const [accuracyHistory, setAccuracyHistory] = useState([]);
  
  // Fixed batch size
  const BATCH_SIZE = 32;
  
  // Loading states
  const [isGenerating, setIsGenerating] = useState(false);
  const [message, setMessage] = useState('');
  
  // Corpus selection state
  const [corpusType, setCorpusType] = useState('wikipedia');
  const [corpusCategory, setCorpusCategory] = useState('nature');
  const [corpusCount, setCorpusCount] = useState(5);
  const [isFetchingCorpus, setIsFetchingCorpus] = useState(false);
  const [fetchedTexts, setFetchedTexts] = useState([]);
  
  // Corpus options
  const corpusOptions = {
    wikipedia: [
      'nature',
      'technology',
      'history',
      'science',
      'art',
      'philosophy',
      'mathematics',
      'literature',
      'geography',
      'sports'
    ],
    books: [
      'fiction',
      'non-fiction',
      'classic',
      'poetry',
      'drama',
      'mystery',
      'romance',
      'adventure',
      'biography',
      'science-fiction'
    ]
  };
  
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
      
      // Send a training request with a temporary UUID
      const tempUserId = 'temp-' + Date.now();
      const sampleText = `This is a sample text to initialize the model. 
      The quick brown fox jumps over the lazy dog. 
      She sells seashells by the seashore. 
      To be or not to be, that is the question.
      All that glitters is not gold.
      A journey of a thousand miles begins with a single step.`;
      
      const response = await axios.post(`${API_URL}/train`, {
        user_id: tempUserId,
        training_text: sampleText,
        epochs: 1,
        batch_size: BATCH_SIZE,
        create_new: true
      });
      
      // Set the new user ID
      setUserId(response.data.user_id);
      setUserExists(true);
      setMessage(`New user created with ID: ${response.data.user_id}`);
      
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('Error creating user:', error);
      setMessage('Error creating user: ' + (error.response?.data?.detail || error.message));
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
  
  // Train model with SSE
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
    
    // Validate minimum text length
    if (trainingText.length < 50) {
      setMessage('Please enter more text for training (at least 50 characters)');
      return;
    }
    
    try {
      setIsTraining(true);
      setMessage('Training model...');
      setAccuracyHistory([]);
      
      // Prepare data for the POST request
      const trainingData = {
        user_id: userId,
        training_text: trainingText,
        epochs: epochs,
        batch_size: BATCH_SIZE,
        create_new: false
      };
      
      // First, send a POST request to initiate training
      await fetch(`${API_URL}/train_sse`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingData)
      }).then(response => {
        // Create an event reader from the response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        // Process the stream
        function processStream() {
          reader.read().then(({ done, value }) => {
            if (done) {
              console.log('Stream complete');
              setIsTraining(false);
              return;
            }
            
            // Decode the chunk
            const chunk = decoder.decode(value);
            
            // Process each SSE event in the chunk
            const events = chunk.split('\n\n');
            for (const event of events) {
              if (event.startsWith('data: ')) {
                try {
                  const data = JSON.parse(event.substring(6));
                  
                  switch (data.status) {
                    case 'starting':
                      setMessage(`Starting training with ${data.total_epochs} epochs...`);
                      break;
                    
                    case 'progress':
                      setMessage(`Training: ${data.epoch}/${data.total_epochs} epochs, Accuracy: ${data.accuracy.toFixed(4)}`);
                      setAccuracyHistory(data.accuracy_history);
                      break;
                    
                    case 'complete':
                      setMessage('Model trained successfully!');
                      setAccuracyHistory(data.accuracy_history || []);
                      setIsTraining(false);
                      setTimeout(() => setMessage(''), 3000);
                      break;
                    
                    case 'error':
                      setMessage(`Error: ${data.message}`);
                      setIsTraining(false);
                      break;
                    
                    default:
                      console.log('Unknown status:', data);
                  }
                } catch (e) {
                  console.error('Error parsing event data:', e, event);
                }
              }
            }
            
            // Continue reading the stream
            processStream();
          }).catch(err => {
            console.error('Error reading stream:', err);
            setMessage('Error in training connection');
            setIsTraining(false);
          });
        }
        
        processStream();
      });
    } catch (error) {
      console.error('Error training model:', error);
      setMessage('Error training model: ' + (error.response?.data?.detail || error.message));
      setIsTraining(false);
    }
  };
  
  // Fetch corpus
  const handleFetchCorpus = async () => {
    if (!userId) {
      setMessage('Please create a user or enter a user ID first');
      return;
    }

    try {
      setIsFetchingCorpus(true);
      setMessage('Fetching corpus...');

      const response = await axios.post(`${API_URL}/fetch_corpus`, {
        corpus_type: corpusType,
        category: corpusCategory,
        count: corpusCount
      });

      setFetchedTexts(response.data.texts);
      setMessage(`Successfully fetched ${response.data.texts.length} texts (${response.data.total_length} characters)`);
      setTimeout(() => setMessage(''), 3000);
    } catch (error) {
      console.error('Error fetching corpus:', error);
      setMessage('Error fetching corpus: ' + (error.response?.data?.detail || error.message));
    } finally {
      setIsFetchingCorpus(false);
    }
  };

  // Train with fetched corpus with SSE
  const handleTrainWithCorpus = async () => {
    if (!userId) {
      setMessage('Please create a user or enter a user ID first');
      return;
    }

    if (fetchedTexts.length === 0) {
      setMessage('Please fetch a corpus first');
      return;
    }

    try {
      setIsTraining(true);
      setMessage('Training model with corpus...');
      setAccuracyHistory([]);
      
      // Prepare data for the POST request
      const trainingData = {
        user_id: userId,
        training_text: fetchedTexts.join('\n\n'),
        epochs: epochs,
        batch_size: BATCH_SIZE,
        create_new: false
      };
      
      // First, send a POST request to initiate training
      await fetch(`${API_URL}/train_with_corpus_sse`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(trainingData)
      }).then(response => {
        // Create an event reader from the response
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        // Process the stream
        function processStream() {
          reader.read().then(({ done, value }) => {
            if (done) {
              console.log('Stream complete');
              setIsTraining(false);
              return;
            }
            
            // Decode the chunk
            const chunk = decoder.decode(value);
            
            // Process each SSE event in the chunk
            const events = chunk.split('\n\n');
            for (const event of events) {
              if (event.startsWith('data: ')) {
                try {
                  const data = JSON.parse(event.substring(6));
                  
                  switch (data.status) {
                    case 'starting':
                      setMessage(`Starting training with ${data.total_epochs} epochs...`);
                      break;
                    
                    case 'progress':
                      setMessage(`Training: ${data.epoch}/${data.total_epochs} epochs, Accuracy: ${data.accuracy.toFixed(4)}`);
                      setAccuracyHistory(data.accuracy_history);
                      break;
                    
                    case 'complete':
                      setMessage('Model trained successfully!');
                      setAccuracyHistory(data.accuracy_history || []);
                      setIsTraining(false);
                      setTimeout(() => setMessage(''), 3000);
                      break;
                    
                    case 'error':
                      setMessage(`Error: ${data.message}`);
                      setIsTraining(false);
                      break;
                    
                    default:
                      console.log('Unknown status:', data);
                  }
                } catch (e) {
                  console.error('Error parsing event data:', e, event);
                }
              }
            }
            
            // Continue reading the stream
            processStream();
          }).catch(err => {
            console.error('Error reading stream:', err);
            setMessage('Error in training connection');
            setIsTraining(false);
          });
        }
        
        processStream();
      });
    } catch (error) {
      console.error('Error training model:', error);
      setMessage('Error training model: ' + (error.response?.data?.detail || error.message));
      setIsTraining(false);
    }
  };
  
  // Chart data and options
  const chartData = {
    labels: accuracyHistory.map((_, index) => `Epoch ${index + 1}`),
    datasets: [
      {
        label: 'Token Prediction Accuracy',
        data: accuracyHistory,
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
        text: 'Token Prediction Accuracy'
      }
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 1,
        title: {
          display: true,
          text: 'Accuracy (0-1)'
        }
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
        {/* Corpus Selection Section */}
        <div className="col-md-12 mb-4">
          <div className="card">
            <div className="card-body">
              <h3 className="card-title">Corpus Selection</h3>
              
              <div className="row mb-3">
                <div className="col-md-4">
                  <label htmlFor="corpusType" className="form-label">Corpus Type</label>
                  <select 
                    id="corpusType"
                    className="form-select"
                    value={corpusType}
                    onChange={(e) => {
                      setCorpusType(e.target.value);
                      setCorpusCategory(corpusOptions[e.target.value][0]);
                    }}
                  >
                    <option value="wikipedia">Wikipedia</option>
                    <option value="books">Books</option>
                  </select>
                </div>
                
                <div className="col-md-4">
                  <label htmlFor="corpusCategory" className="form-label">Category</label>
                  <select 
                    id="corpusCategory"
                    className="form-select"
                    value={corpusCategory}
                    onChange={(e) => setCorpusCategory(e.target.value)}
                  >
                    {corpusOptions[corpusType].map(category => (
                      <option key={category} value={category}>
                        {category.charAt(0).toUpperCase() + category.slice(1)}
                      </option>
                    ))}
                  </select>
                </div>
                
                <div className="col-md-4">
                  <label htmlFor="corpusCount" className="form-label">Number of Texts</label>
                  <input 
                    type="number"
                    id="corpusCount"
                    className="form-control"
                    min="1"
                    max="20"
                    value={corpusCount}
                    onChange={(e) => setCorpusCount(parseInt(e.target.value))}
                  />
                </div>
              </div>
              
              <button 
                className="btn btn-primary me-2"
                onClick={handleFetchCorpus}
                disabled={isFetchingCorpus || !userId}
              >
                {isFetchingCorpus ? 'Fetching...' : 'Fetch Corpus'}
              </button>
              
              {fetchedTexts.length > 0 && (
                <button 
                  className="btn btn-success"
                  onClick={handleTrainWithCorpus}
                  disabled={isTraining}
                >
                  {isTraining ? 'Training...' : 'Train with entire Corpus'}
                </button>
              )}
              
              {fetchedTexts.length > 0 && (
                <div className="mt-3">
                  <h5>Fetched Texts:</h5>
                  <div className="list-group">
                    {fetchedTexts.map((text, index) => (
                      <div key={index} className="list-group-item">
                        <div className="d-flex justify-content-between align-items-center mb-2">
                          <small className="text-muted">Text {index + 1} ({text.length} characters)</small>
                          <button 
                            className="btn btn-sm btn-outline-primary"
                            onClick={() => {
                              setTrainingText(text);
                              setMessage("Text copied to training area");
                              setTimeout(() => setMessage(""), 2000);
                            }}
                          >
                            Copy only this text to training area to train on.
                          </button>
                        </div>
                        <div 
                          style={{ 
                            maxHeight: '250px', 
                            overflowY: 'auto', 
                            padding: '10px',
                            backgroundColor: '#f8f9fa',
                            borderRadius: '4px',
                            fontSize: '0.9rem'
                          }}
                        >
                          <p style={{ whiteSpace: 'pre-wrap' }}>
                            {text.substring(0, 2000)}
                            {text.length > 2000 && '...'}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        
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
                      max="200"
                      value={maxLength}
                      onChange={(e) => setMaxLength(Math.min(parseInt(e.target.value), 200))}
                    />
                  </div>
                  <div className="col-md-6">
                    <label htmlFor="temperature" className="form-label">Temperature</label>
                    <input 
                      type="range" 
                      id="temperature"
                      className="form-range" 
                      min="0" 
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
                  <div className="col-md-12">
                    <label htmlFor="epochs" className="form-label">Epochs</label>
                    <input 
                      type="number" 
                      id="epochs"
                      className="form-control" 
                      min="1" 
                      max="50"
                      value={epochs}
                      onChange={(e) => setEpochs(parseInt(e.target.value))}
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
              
              {accuracyHistory.length > 0 && (
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