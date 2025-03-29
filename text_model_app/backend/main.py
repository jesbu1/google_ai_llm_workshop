import os
import torch
import uuid
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

from model import (
    CharacterTokenizer,
    LanguageModelTrainer,
    get_sample_corpus,
)
from transformer_model import TransformerModel, TransformerConfig
from model_manager import ModelManager

# Create model directories
os.makedirs("models", exist_ok=True)
os.makedirs("models/pretrained", exist_ok=True)
os.makedirs("models/user", exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Text Generation API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add security headers middleware
@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Pydantic models for requests and responses
class GenerateRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    max_length: int = 128
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.95

class TrainRequest(BaseModel):
    user_id: str
    training_text: str
    epochs: int = 50
    batch_size: int = 32
    create_new: bool = False

class TextGenerationResponse(BaseModel):
    generated_text: str

class TrainingResponse(BaseModel):
    user_id: str
    message: str
    loss_history: List[float]

class FineTuneRequest(BaseModel):
    user_id: str
    text: str

class CorpusRequest(BaseModel):
    corpus_type: str  # "wikipedia" or "books"
    category: str     # Category for Wikipedia or book genre
    count: int = 5    # Number of articles/books to fetch

class CorpusResponse(BaseModel):
    texts: List[str]
    total_length: int
    message: str

model_manager = ModelManager()

# Pre-trained model paths
PRETRAINED_MODEL_PATH = "models/pretrained/model.pt"
PRETRAINED_TOKENIZER_PATH = "models/pretrained/tokenizer.json"

# Create and train the pre-trained model if it doesn't exist
@app.get("/")
async def root():
    return {"message": "Text Generation API is running!"}

@app.post("/generate")
async def generate_text(request: GenerateRequest):
    try:
        # Use user's LoRA model if available, otherwise use base model
        model = (model_manager.lora_models.get(request.user_id, model_manager.base_model) 
                if request.user_id else model_manager.base_model)
        
        inputs = model_manager.tokenizer(
            request.prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        outputs = model.generate(
            inputs.input_ids,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            do_sample=True
        )
        
        return {
            "generated_text": model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune")
async def fine_tune(request: FineTuneRequest):
    try:
        # Create LoRA model if it doesn't exist
        if request.user_id not in model_manager.lora_models:
            model = model_manager.create_lora_model_for_user(request.user_id)
        else:
            model = model_manager.lora_models[request.user_id]

        # Encode the text
        inputs = model_manager.tokenizer(
            request.text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        )

        # Single forward and backward pass for fine-tuning
        outputs = model(
            input_ids=inputs.input_ids,
            labels=inputs.input_ids
        )
        
        outputs.loss.backward()
        
        return {"message": "Fine-tuning step completed"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/exists")
async def check_user_exists(user_id: str):
    return {"exists": user_id in model_manager.lora_models}

@app.get("/model_info")
async def get_model_info():
    return {
        "base_model": model_manager.model_name,
        "parameters": "13M",
        "description": "Tiny-LLM with LoRA fine-tuning capability"
    }

@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # Create LoRA model if it doesn't exist
        if request.user_id not in model_manager.lora_models:
            model = model_manager.create_lora_model_for_user(request.user_id)
        else:
            model = model_manager.lora_models[request.user_id]

        # Encode the text
        inputs = model_manager.tokenizer(
            request.training_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Training loop with improved settings
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request.epochs)
        loss_history = []

        for epoch in range(request.epochs):
            # Forward pass
            outputs = model(
                input_ids=inputs.input_ids,
                labels=inputs.input_ids
            )
            
            # Backward pass
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            loss_history.append(loss.item())
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{request.epochs}, Loss: {loss.item():.4f}")

        # Generate a new user ID if this was a new user creation
        if request.create_new:
            new_user_id = str(uuid.uuid4())
            model_manager.lora_models[new_user_id] = model_manager.lora_models.pop(request.user_id)
            return {
                "user_id": new_user_id,
                "message": "Model trained successfully",
                "loss_history": loss_history
            }
        
        return {
            "user_id": request.user_id,
            "message": "Model trained successfully",
            "loss_history": loss_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_wikipedia_articles(category: str, count: int = 5) -> List[str]:
    """Fetch Wikipedia articles based on category."""
    # Category mapping to Wikipedia category names
    category_mapping = {
        'nature': 'Category:Nature',
        'technology': 'Category:Technology',
        'history': 'Category:History',
        'science': 'Category:Science',
        'art': 'Category:Art',
        'philosophy': 'Category:Philosophy',
        'mathematics': 'Category:Mathematics',
        'literature': 'Category:Literature',
        'geography': 'Category:Geography',
        'sports': 'Category:Sports'
    }
    
    # Get the proper Wikipedia category name
    wiki_category = category_mapping.get(category.lower(), category)
    
    # First, get a list of articles in the category
    category_url = f"https://en.wikipedia.org/wiki/{wiki_category}"
    articles = []
    
    try:
        print(f"Fetching category page: {category_url}")
        # Get the category page
        response = requests.get(category_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the main content
        content = soup.find('div', {'class': 'mw-category'})
        if not content:
            print("Category div not found, trying mw-parser-output")
            # Try the main content area if category page doesn't have the expected structure
            content = soup.find('div', {'class': 'mw-parser-output'})
        
        if not content:
            print("No content found in the page")
            return []
            
        # Find all article links
        article_links = content.find_all('a', href=True)
        article_urls = []
        
        # Filter and collect article URLs
        for link in article_links:
            href = link['href']
            if href.startswith('/wiki/') and not any(x in href for x in ['Category:', 'Special:', 'Help:', 'Template:']):
                article_urls.append(f"https://en.wikipedia.org{href}")
        
        print(f"Found {len(article_urls)} articles in category")
        
        if not article_urls:
            print("No article URLs found")
            return []
            
        # Randomly select articles up to the requested count
        import random
        selected_urls = random.sample(article_urls, min(count, len(article_urls)))
        
        # Fetch each selected article
        for url in selected_urls:
            try:
                print(f"Fetching article: {url}")
                article_response = requests.get(url)
                article_response.raise_for_status()
                article_soup = BeautifulSoup(article_response.text, 'html.parser')
                
                # Get the main content
                article_content = article_soup.find('div', {'class': 'mw-parser-output'})
                if article_content:
                    # Extract paragraphs
                    paragraphs = article_content.find_all('p')
                    text = ' '.join(p.get_text() for p in paragraphs)
                    if text.strip():  # Only add non-empty articles
                        articles.append(text)
                        print(f"Successfully added article from {url}")
                else:
                    print(f"No content found in article: {url}")
            except Exception as e:
                print(f"Error fetching article from {url}: {str(e)}")
                continue
                
    except Exception as e:
        print(f"Error fetching category {wiki_category}: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print(f"Successfully fetched {len(articles)} articles")
    return articles

def fetch_book_texts(genre: str, count: int = 5) -> List[str]:
    """Fetch book texts from Project Gutenberg based on genre."""
    # Project Gutenberg catalog URL
    base_url = "https://www.gutenberg.org/ebooks/search/?query="
    
    try:
        print(f"Searching for books with genre: {genre}")
        # Search for books in the genre
        search_url = f"{base_url}{genre}"
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find book links
        book_links = []
        for link in soup.find_all('a', href=True):
            if '/ebooks/' in link['href'] and link['href'].startswith('/'):
                book_links.append(f"https://www.gutenberg.org{link['href']}")
        
        print(f"Found {len(book_links)} book links")
        
        if not book_links:
            print("No book links found")
            return []
            
        texts = []
        # Fetch up to count books
        for book_url in book_links[:count]:
            try:
                print(f"Fetching book: {book_url}")
                # Get the book page
                book_response = requests.get(book_url)
                book_response.raise_for_status()
                book_soup = BeautifulSoup(book_response.text, 'html.parser')
                
                # Find the download link for plain text
                download_link = None
                for link in book_soup.find_all('a', href=True):
                    if 'txt' in link['href'] and 'utf-8' in link['href']:
                        download_link = link['href']
                        break
                
                if download_link:
                    print(f"Found download link: {download_link}")
                    # Download the text content
                    text_response = requests.get(download_link)
                    text_response.raise_for_status()
                    if text_response.status_code == 200:
                        # Clean up the text
                        text = text_response.text
                        # Remove headers and footers
                        text = text.split('*** START OF THIS PROJECT GUTENBERG EBOOK')[0]
                        text = text.split('*** END OF THIS PROJECT GUTENBERG EBOOK')[1]
                        # Remove multiple newlines
                        text = '\n'.join(line for line in text.split('\n') if line.strip())
                        if text.strip():  # Only add non-empty texts
                            texts.append(text)
                            print(f"Successfully added book from {book_url}")
                else:
                    print(f"No download link found for book: {book_url}")
            except Exception as e:
                print(f"Error fetching book: {str(e)}")
                continue
                
        print(f"Successfully fetched {len(texts)} books")
        return texts
    except Exception as e:
        print(f"Error fetching books: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return []

@app.post("/fetch_corpus")
async def fetch_corpus(request: CorpusRequest):
    try:
        print(f"Fetching corpus: type={request.corpus_type}, category={request.category}, count={request.count}")
        texts = []
        if request.corpus_type == "wikipedia":
            texts = fetch_wikipedia_articles(request.category, request.count)
        elif request.corpus_type == "books":
            texts = fetch_book_texts(request.category, request.count)
        else:
            raise HTTPException(status_code=400, detail="Invalid corpus type")
        
        if not texts:
            raise HTTPException(
                status_code=404, 
                detail=f"No texts found for {request.corpus_type} category: {request.category}. Please try a different category."
            )
        
        total_length = sum(len(text) for text in texts)
        
        return {
            "texts": texts,
            "total_length": total_length,
            "message": f"Successfully fetched {len(texts)} {request.corpus_type} texts"
        }
    except Exception as e:
        print(f"Error in fetch_corpus: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_with_corpus")
async def train_with_corpus(request: TrainRequest):
    try:
        # Create LoRA model if it doesn't exist
        if request.user_id not in model_manager.lora_models:
            model = model_manager.create_lora_model_for_user(request.user_id)
        else:
            model = model_manager.lora_models[request.user_id]

        # Combine all texts into one
        combined_text = "\n\n".join(request.training_text.split("\n"))

        # Encode the text
        inputs = model_manager.tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Training loop with improved settings
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request.epochs)
        loss_history = []

        for epoch in range(request.epochs):
            # Forward pass
            outputs = model(
                input_ids=inputs.input_ids,
                labels=inputs.input_ids
            )
            
            # Backward pass
            loss = outputs.loss
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            loss_history.append(loss.item())
            
            # Print progress every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{request.epochs}, Loss: {loss.item():.4f}")

        # Generate a new user ID if this was a new user creation
        if request.create_new:
            new_user_id = str(uuid.uuid4())
            model_manager.lora_models[new_user_id] = model_manager.lora_models.pop(request.user_id)
            return {
                "user_id": new_user_id,
                "message": "Model trained successfully",
                "loss_history": loss_history
            }
        
        return {
            "user_id": request.user_id,
            "message": "Model trained successfully",
            "loss_history": loss_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable or use default
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=False,  # Disable reload in production
        workers=1,  # Single worker for CPU training
        log_level="info"
    )
