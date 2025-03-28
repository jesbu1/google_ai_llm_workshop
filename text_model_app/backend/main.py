import os
import torch
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict

from model import (
    CharacterTokenizer,
    SimpleLanguageModel,
    LanguageModelTrainer,
    get_sample_corpus,
)

# Create model directories
os.makedirs("models", exist_ok=True)
os.makedirs("models/pretrained", exist_ok=True)
os.makedirs("models/user", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Text Generation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# Pydantic models for requests and responses
class TextGenerationRequest(BaseModel):
    user_id: str
    prompt: str
    max_length: int = 100
    temperature: float = 1.0


class TextGenerationResponse(BaseModel):
    generated_text: str


class TrainingRequest(BaseModel):
    user_id: str
    training_text: str
    epochs: int = 5
    batch_size: int = 32
    create_new: bool = False


class TrainingResponse(BaseModel):
    user_id: str
    message: str
    loss_history: List[float]


# Store user models in memory for quick access
user_models = {}

# Pre-trained model paths
PRETRAINED_MODEL_PATH = "models/pretrained/model.pt"
PRETRAINED_TOKENIZER_PATH = "models/pretrained/tokenizer.json"

# Create and train the pre-trained model if it doesn't exist
if not os.path.exists(PRETRAINED_MODEL_PATH) or not os.path.exists(
    PRETRAINED_TOKENIZER_PATH
):
    print("Creating pre-trained model...")

    # Get the sample corpus
    corpus = get_sample_corpus()

    # Create tokenizer and fit on corpus
    tokenizer = CharacterTokenizer()
    tokenizer.fit([corpus])

    # Create model
    model = SimpleLanguageModel(tokenizer.vocab_size)

    # Create trainer
    trainer = LanguageModelTrainer(tokenizer, model, device=device)

    # Train on corpus
    trainer.train_on_text(corpus, epochs=10, batch_size=8)

    # Save pre-trained model
    trainer.save_model(PRETRAINED_MODEL_PATH, PRETRAINED_TOKENIZER_PATH)
    print("Pre-trained model created and saved!")


def get_user_model_path(user_id: str) -> tuple:
    return f"models/user/{user_id}_model.pt", f"models/user/{user_id}_tokenizer.json"


def load_user_model(user_id: str) -> LanguageModelTrainer:
    model_path, tokenizer_path = get_user_model_path(user_id)

    # Check if model already loaded in memory
    if user_id in user_models:
        return user_models[user_id]

    # Check if user model exists
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        # Load existing user model
        tokenizer = CharacterTokenizer()
        tokenizer.load(tokenizer_path)

        model = SimpleLanguageModel(tokenizer.vocab_size)
        trainer = LanguageModelTrainer(tokenizer, model, device=device)
        trainer.load_model(model_path, tokenizer_path)
    else:
        # Create new model from pre-trained
        tokenizer = CharacterTokenizer()
        tokenizer.load(PRETRAINED_TOKENIZER_PATH)

        model = SimpleLanguageModel(tokenizer.vocab_size)
        trainer = LanguageModelTrainer(tokenizer, model, device=device)
        trainer.load_model(PRETRAINED_MODEL_PATH, PRETRAINED_TOKENIZER_PATH)
        trainer.save_model(model_path, tokenizer_path)

    # Cache model
    user_models[user_id] = trainer
    return trainer


@app.get("/")
async def root():
    return {"message": "Text Generation API is running!"}


@app.post("/generate", response_model=TextGenerationResponse)
async def generate_text(request: TextGenerationRequest):
    try:
        # Load user model
        trainer = load_user_model(request.user_id)

        # Generate text
        generated_text = trainer.model.generate_text(
            trainer.tokenizer,
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
        )

        return TextGenerationResponse(generated_text=generated_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    try:
        # Generate user ID if not provided
        if request.user_id == "new":
            user_id = str(uuid.uuid4())
        else:
            user_id = request.user_id

        model_path, tokenizer_path = get_user_model_path(user_id)

        # Create a new model or load existing
        if request.create_new or not os.path.exists(model_path):
            # Create new model from pre-trained
            tokenizer = CharacterTokenizer()
            tokenizer.load(PRETRAINED_TOKENIZER_PATH)

            model = SimpleLanguageModel(tokenizer.vocab_size)
            trainer = LanguageModelTrainer(tokenizer, model, device=device)
            trainer.load_model(PRETRAINED_MODEL_PATH, PRETRAINED_TOKENIZER_PATH)
        else:
            # Load existing model
            trainer = load_user_model(user_id)

        # Fine-tune the model
        loss_history = trainer.train_on_text(
            request.training_text, epochs=request.epochs, batch_size=request.batch_size
        )

        # Save the fine-tuned model
        trainer.save_model(model_path, tokenizer_path)

        # Update cache
        user_models[user_id] = trainer

        return TrainingResponse(
            user_id=user_id,
            message="Model trained successfully!",
            loss_history=loss_history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}/exists")
async def check_user_exists(user_id: str):
    model_path, _ = get_user_model_path(user_id)
    exists = os.path.exists(model_path)
    return {"exists": exists}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
