# Text Generation Model Application

This application allows users to train their own small language model for generating text based on their input. The model is pre-trained on a diverse corpus including Wikipedia articles and can be fine-tuned iteratively with user-provided text.

## Features

- Pre-trained transformer-based language model (GPT-2 style architecture)
- Multi-head attention mechanism with causal masking
- Position embeddings and layer normalization
- Advanced text generation with temperature, top-k, and top-p sampling
- User-specific model instances with fine-tuning capabilities
- Training progress visualization
- Simple and intuitive UI
- Automatic Wikipedia article fetching for diverse training data

## Project Structure

```
text_model_app/
├── backend/              # FastAPI backend
│   ├── main.py           # API endpoints
│   ├── model.py          # Training and tokenization logic
│   ├── transformer_model.py  # Transformer model architecture
│   └── requirements.txt  # Python dependencies
│
└── frontend/             # React frontend
    ├── src/
    │   ├── App.jsx       # Main React component
    │   └── index.js      # React entry point
    ├── index.html        # HTML template
    └── package.json      # JavaScript dependencies
```

## Model Architecture

The model uses a lightweight transformer architecture similar to GPT-2, optimized for CPU training with the following specifications:

- 3 transformer layers (reduced from 6)
- 4 attention heads (reduced from 8)
- 64 hidden size (reduced from 256)
- 256 intermediate size (reduced from 1024)
- Character-level tokenization
- Position embeddings (128 max positions)
- Layer normalization
- GELU activation functions
- Dropout for regularization
- Causal attention masking
- 64-token sequence length
- Smaller batch size (8) for CPU training

This compact architecture allows for reasonable training times on CPU while maintaining the benefits of transformer-based language modeling.

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

2. **Generate Text**: Enter a prompt in the "Text Generation" section and click "Generate Text". You can adjust:
   - Maximum length of generated text
   - Temperature (higher values = more random, lower values = more focused)
   - Top-k sampling (limit vocabulary to k most likely tokens)
   - Top-p (nucleus) sampling (limit vocabulary to tokens whose cumulative probability exceeds p)

3. **Train Your Model**: Enter your own text in the "Model Training" section and click "Train Model". You can customize:
   - Number of epochs
   - Batch size
   - Create a new model or fine-tune existing one

4. **Fine-tune Iteratively**: Continue adding more text samples to refine your model's output.

## Technical Details

- The model uses a character-level tokenizer with special tokens (BOS, EOS, PAD, UNK)
- Pre-training corpus includes:
  - Random Wikipedia articles for diverse content
  - Curated text samples for quality control
- Training features:
  - AdamW optimizer with weight decay
  - Cosine annealing learning rate schedule
  - Gradient clipping
  - Dropout for regularization
- Text generation features:
  - Causal self-attention
  - Temperature-controlled sampling
  - Top-k and top-p filtering
  - Position-aware generation

## Notes

- The model is designed for learning and experimentation purposes
- Training time depends on:
  - Amount of training text
  - Number of epochs (default: 10)
  - Batch size (default: 8)
  - Available computational resources (CPU or GPU)
- The pre-trained model is automatically created on first run
- To rebuild the pre-trained model, delete the files in the models/pretrained directory and restart the server
- The model uses a smaller architecture optimized for CPU training 