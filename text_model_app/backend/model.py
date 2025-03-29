import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from typing import List, Dict, Optional, Tuple
from collections import Counter
import requests
from bs4 import BeautifulSoup
import random

from transformer_model import TransformerModel, TransformerConfig


class CharacterTokenizer:
    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

    def fit(self, texts: List[str]):
        # Create vocabulary from unique characters in the texts
        chars = set("".join(texts))
        # Add special tokens
        chars.update(["<PAD>", "<UNK>", "<BOS>", "<EOS>"])

        # Create mappings
        self.char_to_idx = {c: i for i, c in enumerate(sorted(chars))}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, text: str) -> List[int]:
        # Add BOS and EOS tokens
        return [self.char_to_idx["<BOS>"]] + [self.char_to_idx.get(c, self.char_to_idx["<UNK>"]) for c in text] + [self.char_to_idx["<EOS>"]]

    def decode(self, indices: List[int]) -> str:
        # Remove BOS and EOS tokens
        return "".join([self.idx_to_char.get(i, "<UNK>") for i in indices if i not in [self.char_to_idx["<BOS>"], self.char_to_idx["<EOS>"]]])

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


def fetch_wikipedia_articles(num_articles: int = 10) -> str:
    """Fetch random Wikipedia articles for training data."""
    base_url = "https://en.wikipedia.org/wiki/Special:Random"
    articles = []
    
    for _ in range(num_articles):
        try:
            response = requests.get(base_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Get the main content
            content = soup.find('div', {'class': 'mw-parser-output'})
            if content:
                # Extract paragraphs
                paragraphs = content.find_all('p')
                text = ' '.join(p.get_text() for p in paragraphs)
                articles.append(text)
        except Exception as e:
            print(f"Error fetching article: {e}")
            continue
    
    return '\n'.join(articles)

def get_sample_corpus() -> str:
    """Returns a diverse text corpus for pre-training."""
    # Fetch Wikipedia articles
    wiki_text = fetch_wikipedia_articles(10)
    
    # Add some curated text
    curated_text = """
    The quick brown fox jumps over the lazy dog. She sells seashells by the seashore.
    To be or not to be, that is the question. All that glitters is not gold.
    A journey of a thousand miles begins with a single step. The early bird catches the worm.
    Actions speak louder than words. Don't judge a book by its cover.
    Where there's a will, there's a way. Time flies like an arrow, fruit flies like a banana.
    
    Once upon a time, in a land far away, there lived a wise old king. The king had three daughters, each more beautiful than the last. The youngest princess was known throughout the kingdom for her kindness and gentle spirit.
    
    The village was nestled between rolling hills and a sparkling river. Farmers tended their fields during the day, while artisans crafted their wares in small workshops. Children played in the village square, their laughter echoing through the narrow streets.
    
    Science has transformed our understanding of the universe. From the smallest particles to the vast expanses of space, we continue to uncover the mysteries of existence. Technology advances at an unprecedented rate, changing how we live, work, and communicate.
    
    The ancient forest stood silent, its towering trees guardians of countless secrets. Sunlight filtered through the canopy, creating dappled patterns on the forest floor. A gentle stream wound its way between moss-covered rocks, the water clear and cool.
    
    The chef worked meticulously, combining ingredients with precision and creativity. Each dish told a story, a blend of tradition and innovation. The aroma filled the kitchen, promising delights to come.
    
    Music has the power to move us, to express what words cannot. It transcends language barriers and speaks directly to the heart. From classical symphonies to modern compositions, it reflects the full spectrum of human emotion.
    
    Democracy requires engaged citizens who participate in the process of governance. Debate and discourse shape policy, while compromise enables progress. The ideals of liberty and equality remain central to the democratic tradition.
    
    The detective examined the scene carefully, noting every detail that might provide a clue. Years of experience had taught him to observe what others missed. He would unravel this mystery, as he had many before.
    
    The mathematician found beauty in equations, elegance in proofs. Numbers told stories of their own, revealing patterns that explained the natural world. There was truth in mathematics, absolute and unwavering.
    
    Literature offers insights into the human condition, exploring themes of love, loss, courage, and redemption. Through stories, we experience lives different from our own, expanding our understanding and empathy.
    """
    
    # Combine and shuffle the text
    combined_text = wiki_text + "\n" + curated_text
    paragraphs = combined_text.split('\n')
    random.shuffle(paragraphs)
    return '\n'.join(paragraphs)


class LanguageModelTrainer:
    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        model: TransformerModel,
        sequence_length: int = 128,
        learning_rate: float = 0.0001,
        device: str = "cpu",
    ):
        self.tokenizer = tokenizer
        self.model = model.to(device)
        self.sequence_length = sequence_length
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def prepare_sequences(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare sequence pairs for training."""
        try:
            # Encode the text
            encoded = self.tokenizer.encode(text)

            # Check if encoded text is long enough
            if len(encoded) <= self.sequence_length:
                print(
                    f"Warning: Encoded text ({len(encoded)} tokens) is shorter than sequence_length ({self.sequence_length})"
                )
                return (
                    torch.tensor([], dtype=torch.long)
                    .reshape(0, self.sequence_length)
                    .to(self.device),
                    torch.tensor([], dtype=torch.long)
                    .reshape(0, self.sequence_length)
                    .to(self.device),
                )

            # Create sequences
            sequences = []
            targets = []

            for i in range(0, len(encoded) - self.sequence_length):
                seq = encoded[i : i + self.sequence_length]
                target = encoded[i + 1 : i + self.sequence_length + 1]
                sequences.append(seq)
                targets.append(target)

            print(f"Created {len(sequences)} training sequences")

            if not sequences:
                print("Warning: No sequences could be created. Returning empty tensors.")
                return (
                    torch.tensor([], dtype=torch.long)
                    .reshape(0, self.sequence_length)
                    .to(self.device),
                    torch.tensor([], dtype=torch.long)
                    .reshape(0, self.sequence_length)
                    .to(self.device),
                )

            return (
                torch.tensor(sequences, dtype=torch.long).to(self.device),
                torch.tensor(targets, dtype=torch.long).to(self.device),
            )
        except Exception as e:
            print(f"Error in prepare_sequences: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return (
                torch.tensor([], dtype=torch.long)
                .reshape(0, self.sequence_length)
                .to(self.device),
                torch.tensor([], dtype=torch.long)
                .reshape(0, self.sequence_length)
                .to(self.device),
            )

    def train_on_text(
        self,
        text: str,
        batch_size: int = 32,
        epochs: int = 5,
        save_path: Optional[str] = None,
    ) -> List[float]:
        """Train the model on the provided text."""
        self.model.train()

        # Ensure the text is long enough
        if len(text) < self.sequence_length + 1:
            print(
                f"Warning: Input text is too short ({len(text)} chars). Repeating to reach minimum length."
            )
            repetitions = (self.sequence_length + 1) // len(text) + 1
            text = text * repetitions

        try:
            print(f"Preparing sequences from text of length {len(text)}")
            sequences, targets = self.prepare_sequences(text)

            # Check if we have enough data for the batch size
            actual_batch_size = min(batch_size, len(sequences))
            if actual_batch_size < batch_size:
                print(
                    f"Warning: Reduced batch size from {batch_size} to {actual_batch_size} due to limited data"
                )
                batch_size = actual_batch_size

            if batch_size == 0:
                print("Error: No sequences could be generated from the text. Text may be too short.")
                return [0.0]

            # Create data batches
            num_batches = max(1, len(sequences) // batch_size)
            loss_history = []

            for epoch in range(epochs):
                epoch_loss = 0

                # Shuffle the data
                indices = torch.randperm(len(sequences))
                sequences = sequences[indices]
                targets = targets[indices]

                for i in range(num_batches):
                    # Get batch
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(sequences))
                    batch_sequences = sequences[start_idx:end_idx]
                    batch_targets = targets[start_idx:end_idx]

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(batch_sequences)

                    # Reshape for loss calculation
                    outputs = outputs.view(-1, self.model.config.vocab_size)
                    batch_targets = batch_targets.view(-1)

                    # Calculate loss
                    loss = self.criterion(outputs, batch_targets)

                    # Backward pass and optimization
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / num_batches
                loss_history.append(avg_loss)
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            if save_path:
                torch.save(self.model.state_dict(), save_path)

            return loss_history
        except Exception as e:
            print(f"Error in train_on_text: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return [0.0] * epochs

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
