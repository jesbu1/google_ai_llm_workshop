# Text Generation Model Application

This application allows users to train their own small language model for generating text based on their input. The model is pre-trained on a simple text corpus and can be fine-tuned iteratively with user-provided text.

## Features

- Pre-trained language model on a simple corpus
- User-specific model instances
- Text generation with customizable parameters
- Fine-tuning with your own text
- Training progress visualization
- Simple and intuitive UI

## Project Structure

```
text_model_app/
├── backend/              # FastAPI backend
│   ├── main.py           # API endpoints
│   ├── model.py          # Language model implementation
│   └── requirements.txt  # Python dependencies
│
└── frontend/             # React frontend
    ├── src/
    │   ├── App.jsx       # Main React component
    │   └── index.js      # React entry point
    ├── index.html        # HTML template
    └── package.json      # JavaScript dependencies
```

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd text_model_app/backend
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```
   python main.py
   ```
   The backend will run on http://localhost:8000.

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd text_model_app/frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the frontend development server:
   ```
   npm start
   ```
   The frontend will run on http://localhost:3000.

## Usage

1. **Create a User**: When you first visit the application, create a new user or enter an existing user ID.

2. **Generate Text**: Enter a prompt in the "Text Generation" section and click "Generate Text". Adjust the maximum length and temperature parameters as needed.

3. **Train Your Model**: Enter your own text in the "Model Training" section and click "Train Model". You can customize the number of epochs and batch size.

4. **Fine-tune Iteratively**: Continue adding more text samples to refine your model's output.

## Technical Details

- The language model is a character-based LSTM neural network.
- The model tokenizes text at the character level for simplicity.
- Models are stored on the server, with each user having their own model instance.
- The pre-trained model is trained on a small corpus of common English phrases.

## Notes

- This is a lightweight implementation designed for learning and experimentation.
- The models are relatively small and trained on limited data, so expect accordingly simple generations.
- Training may take some time depending on the amount of text and the number of epochs. 