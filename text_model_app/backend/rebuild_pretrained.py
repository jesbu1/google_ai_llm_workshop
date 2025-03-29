#!/usr/bin/env python
"""
Script to rebuild the pre-trained model with enhanced parameters.
This script will delete the existing pre-trained model and create a new one.
"""

import os
import shutil
from model import (
    CharacterTokenizer,
    SimpleLanguageModel,
    LanguageModelTrainer,
    get_sample_corpus,
)
import torch

print("Starting to rebuild the pre-trained model...")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
# also mps
if torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

# Define model paths
PRETRAINED_DIR = "models/pretrained"
PRETRAINED_MODEL_PATH = f"{PRETRAINED_DIR}/model.pt"
PRETRAINED_TOKENIZER_PATH = f"{PRETRAINED_DIR}/tokenizer.json"

# Create directories if they don't exist
os.makedirs("models", exist_ok=True)
os.makedirs(PRETRAINED_DIR, exist_ok=True)

# Remove existing pre-trained model if it exists
if os.path.exists(PRETRAINED_MODEL_PATH):
    print(f"Removing existing model file: {PRETRAINED_MODEL_PATH}")
    os.remove(PRETRAINED_MODEL_PATH)

if os.path.exists(PRETRAINED_TOKENIZER_PATH):
    print(f"Removing existing tokenizer file: {PRETRAINED_TOKENIZER_PATH}")
    os.remove(PRETRAINED_TOKENIZER_PATH)

# Get sample corpus
corpus = get_sample_corpus()
print(f"Sample corpus length: {len(corpus)} characters")

# Create tokenizer and fit on corpus
tokenizer = CharacterTokenizer()
tokenizer.fit([corpus])
print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

# Create model with improved parameters
model = SimpleLanguageModel(
    vocab_size=tokenizer.vocab_size,
    embedding_dim=256,  # Increased from 128
    hidden_size=512,  # Increased from 256
    num_layers=3,  # Increased from 2
    dropout=0.3,  # Increased from 0.2
)
print("Enhanced model created with larger parameters")

# Create trainer with improved parameters
trainer = LanguageModelTrainer(
    tokenizer,
    model,
    sequence_length=75,  # Longer sequence length
    learning_rate=0.0005,  # Lower learning rate for better convergence
    device=device,
)

# Train on corpus with more epochs
print("Training enhanced pre-trained model...")
loss_history = trainer.train_on_text(
    corpus, epochs=20, batch_size=16
)  # Increased epochs, larger batch size

# Print final loss
final_loss = loss_history[-1] if loss_history else "N/A"
print(f"Final training loss: {final_loss}")

# Save pre-trained model
print(f"Saving enhanced pre-trained model to {PRETRAINED_MODEL_PATH}")
trainer.save_model(PRETRAINED_MODEL_PATH, PRETRAINED_TOKENIZER_PATH)

print("Enhanced pre-trained model created and saved!")
print("Restart the main server to use the new model.")
