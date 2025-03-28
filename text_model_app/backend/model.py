import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter


class CharacterTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit(self, texts: List[str]):
        # Create vocabulary from unique characters in the texts
        chars = set("".join(texts))
        # Add special tokens
        chars.update(["<PAD>", "<UNK>"])

        # Create mappings
        self.char_to_idx = {c: i for i, c in enumerate(sorted(chars))}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> List[int]:
        return [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in text]

    def decode(self, indices: List[int]) -> str:
        return "".join([self.idx_to_char.get(i, "<UNK>") for i in indices])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(
                {
                    "char_to_idx": self.char_to_idx,
                    "idx_to_char": {int(k): v for k, v in self.idx_to_char.items()},
                    "vocab_size": self.vocab_size,
                },
                f,
            )

    def load(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
            self.char_to_idx = data["char_to_idx"]
            self.idx_to_char = {int(k): v for k, v in data["idx_to_char"].items()}
            self.vocab_size = data["vocab_size"]


class SimpleLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super(SimpleLanguageModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hidden

    def generate_text(
        self,
        tokenizer: CharacterTokenizer,
        seed_text: str,
        max_length: int = 100,
        temperature: float = 1.0,
    ) -> str:
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            chars = seed_text
            hidden = None

            # Process the seed text character by character
            input_seq = (
                torch.tensor(tokenizer.encode(seed_text), dtype=torch.long)
                .unsqueeze(0)
                .to(device)
            )
            output, hidden = self(input_seq, hidden)

            # Generate new characters one by one
            for _ in range(max_length):
                # Get the predicted probabilities for the next character
                output_logits = output[:, -1, :] / temperature
                output_probs = F.softmax(output_logits, dim=-1)

                # Sample from the distribution
                predicted_idx = torch.multinomial(output_probs, 1).item()

                # Add the new character to the sequence
                chars += tokenizer.idx_to_char[predicted_idx]

                # Prepare the input for the next step
                input_seq = torch.tensor([[predicted_idx]], dtype=torch.long).to(device)
                output, hidden = self(input_seq, hidden)

            return chars


class LanguageModelTrainer:
    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        model: SimpleLanguageModel,
        sequence_length: int = 100,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.sequence_length = sequence_length
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def prepare_sequences(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode the text
        encoded = self.tokenizer.encode(text)

        # Create sequences
        sequences = []
        targets = []

        for i in range(0, len(encoded) - self.sequence_length):
            seq = encoded[i : i + self.sequence_length]
            target = encoded[i + 1 : i + self.sequence_length + 1]
            sequences.append(seq)
            targets.append(target)

        return (
            torch.tensor(sequences, dtype=torch.long).to(self.device),
            torch.tensor(targets, dtype=torch.long).to(self.device),
        )

    def train_on_text(
        self,
        text: str,
        batch_size: int = 32,
        epochs: int = 5,
        save_path: Optional[str] = None,
    ) -> List[float]:
        self.model.train()
        sequences, targets = self.prepare_sequences(text)

        # Create data batches
        num_batches = len(sequences) // batch_size
        loss_history = []

        for epoch in range(epochs):
            epoch_loss = 0

            # Shuffle the data
            indices = torch.randperm(len(sequences))
            sequences = sequences[indices]
            targets = targets[indices]

            for i in range(num_batches):
                # Get batch
                batch_sequences = sequences[i * batch_size : (i + 1) * batch_size]
                batch_targets = targets[i * batch_size : (i + 1) * batch_size]

                # Zero the gradients
                self.optimizer.zero_grad()

                # Forward pass
                outputs, _ = self.model(batch_sequences)

                # Reshape for loss calculation
                outputs = outputs.view(-1, self.model.fc.out_features)
                batch_targets = batch_targets.view(-1)

                # Calculate loss
                loss = self.criterion(outputs, batch_targets)

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / num_batches
            loss_history.append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        if save_path:
            torch.save(self.model.state_dict(), save_path)

        return loss_history

    def save_model(self, model_path: str, tokenizer_path: str):
        # Save model
        torch.save(self.model.state_dict(), model_path)

        # Save tokenizer
        self.tokenizer.save(tokenizer_path)

    def load_model(self, model_path: str, tokenizer_path: str):
        # Load model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Load tokenizer
        self.tokenizer.load(tokenizer_path)


def get_sample_corpus():
    """Returns a simple text corpus for pre-training."""
    return """
    The quick brown fox jumps over the lazy dog. She sells seashells by the seashore.
    To be or not to be, that is the question. All that glitters is not gold.
    A journey of a thousand miles begins with a single step. The early bird catches the worm.
    Actions speak louder than words. Don't judge a book by its cover.
    Where there's a will, there's a way. Time flies like an arrow, fruit flies like a banana.
    """
