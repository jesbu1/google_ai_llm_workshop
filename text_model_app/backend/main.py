import os
import torch
import uuid
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import requests
from bs4 import BeautifulSoup
from model_manager import ModelManager
import asyncio
import time
import urllib.parse
import json

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
device = "mps" if torch.backends.mps.is_available() else device
print(f"Using device: {device}")

# Pydantic models for requests and responses
class GenerateRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    max_length: int = 128
    temperature: float = 0.0
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
    accuracy_history: List[float] = []

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

# Add a semaphore to limit concurrent model usage
model_semaphore = asyncio.Semaphore(10)  # Only 2 concurrent models in memory

# Create and train the pre-trained model if it doesn't exist
@app.get("/")
async def root():
    return {"message": "Text Generation API is running!"}

@app.post("/generate")
async def generate_text(request: GenerateRequest, background_tasks: BackgroundTasks):
    async with model_semaphore:
        try:
            print(f"Starting text generation with prompt: {request.prompt[:50]}...")
            
            # Use user's LoRA model if available, otherwise use base model
            model = (model_manager.lora_models.get(request.user_id, model_manager.base_model) 
                    if request.user_id else model_manager.base_model)
            print(f"Using {'LoRA' if request.user_id in model_manager.lora_models else 'base'} model")
            
            # Move model to CPU/GPU as needed
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "mps" if torch.backends.mps.is_available() else device
            model = model.to(device)
            
            
            import gc
            gc.collect()
            # More aggressive garbage collection
            if device == "cuda":
                torch.cuda.empty_cache()
            
            # Tokenize with padding and attention mask
            print("Tokenizing input...")
            inputs = model_manager.tokenizer(
                request.prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_attention_mask=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            print(f"Input shape: {inputs['input_ids'].shape}")
            
            print("Starting generation...")
            # Generate with attention mask
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=request.max_length,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                do_sample=True if request.temperature > 0 else False,
                pad_token_id=model_manager.tokenizer.pad_token_id,
                eos_token_id=model_manager.tokenizer.eos_token_id,
                bos_token_id=model_manager.tokenizer.bos_token_id,
                use_cache=False,  # Disable KV cache completely
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                repetition_penalty=1.0
            )
            
            print("Decoding output...")
            generated_text = model_manager.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Successfully generated text: {generated_text[:50]}...")
            
            # Then schedule cleanup
            background_tasks.add_task(torch.cuda.empty_cache)
            
            return {
                "generated_text": generated_text
            }

        except Exception as e:
            print(f"Error in generate_text: {str(e)}")
            import traceback
            print(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/fine-tune")
async def fine_tune(request: FineTuneRequest):
    try:
        # Create LoRA model if it doesn't exist
        if request.user_id not in model_manager.lora_models:
            model = model_manager.initialize_lora_model(request.user_id)
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
        "parameters": "125M",
        "description": "Mobile-LLM"
    }

@app.post("/train")
async def train_model(request: TrainRequest):
    try:
        # For new user creation, generate the UUID first
        if request.create_new:
            new_user_id = str(uuid.uuid4())
            # Create the model with the new ID immediately
            model = model_manager.initialize_lora_model(new_user_id)
        else:
            # For existing users, get or create their model
            if request.user_id not in model_manager.lora_models:
                model = model_manager.initialize_lora_model(request.user_id)
            else:
                model = model_manager.lora_models[request.user_id]

        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps" if torch.backends.mps.is_available() else device
        model = model.to(device)
        print(f"Training on device: {device}")
        
        # More aggressive garbage collection
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Encode the text
        inputs = model_manager.tokenizer(
            request.training_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Training loop with improved settings
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request.epochs)
        loss_history = []
        accuracy_history = []

        for epoch in range(request.epochs):
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                labels=inputs['input_ids']
            )
            
            # Calculate token accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = inputs['input_ids']
            
            # Shift labels and predictions for autoregressive comparison
            shift_labels = labels[:, 1:].contiguous()
            shift_predictions = predictions[:, :-1].contiguous()
            
            # Calculate token-level accuracy
            correct_predictions = (shift_predictions == shift_labels).float().sum().item()
            total_tokens = shift_labels.numel()
            accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
            accuracy_history.append(accuracy)
            
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
                print(f"Epoch {epoch + 1}/{request.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        # Return the appropriate user ID
        return {
            "user_id": new_user_id if request.create_new else request.user_id,
            "message": "Model trained successfully",
            "loss_history": loss_history,
            "accuracy_history": accuracy_history
        }

    except Exception as e:
        print(f"Error in train_model: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_sse")
async def train_model_sse(request: TrainRequest):
    async def generate_updates():
        try:
            # For new user creation, generate the UUID first
            if request.create_new:
                new_user_id = str(uuid.uuid4())
                # Create the model with the new ID immediately
                model = model_manager.initialize_lora_model(new_user_id)
            else:
                # For existing users, get or create their model
                if request.user_id not in model_manager.lora_models:
                    model = model_manager.initialize_lora_model(request.user_id)
                else:
                    model = model_manager.lora_models[request.user_id]

            # Move model to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "mps" if torch.backends.mps.is_available() else device
            model = model.to(device)
            print(f"Training on device: {device}")
            
            # More aggressive garbage collection
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Encode the text
            inputs = model_manager.tokenizer(
                request.training_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Training loop with improved settings
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request.epochs)
            loss_history = []
            accuracy_history = []
            
            # Send a starting message
            yield f"data: {json.dumps({'status': 'starting', 'epoch': 0, 'total_epochs': request.epochs})}\n\n"
            await asyncio.sleep(0.1)  # Small delay for client to process

            for epoch in range(request.epochs):
                # Forward pass
                outputs = model(
                    input_ids=inputs['input_ids'],
                    labels=inputs['input_ids']
                )
                
                # Calculate token accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = inputs['input_ids']
                
                # Shift labels and predictions for autoregressive comparison
                shift_labels = labels[:, 1:].contiguous()
                shift_predictions = predictions[:, :-1].contiguous()
                
                # Calculate token-level accuracy
                correct_predictions = (shift_predictions == shift_labels).float().sum().item()
                total_tokens = shift_labels.numel()
                accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
                accuracy_history.append(accuracy)
                
                # Backward pass
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                loss_history.append(loss.item())
                
                # Print progress and send update every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch == request.epochs - 1:
                    print(f"Epoch {epoch + 1}/{request.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
                    
                    # Send progress update
                    update_data = {
                        'status': 'progress',
                        'epoch': epoch + 1,
                        'total_epochs': request.epochs,
                        'accuracy': accuracy,
                        'loss': loss.item(),
                        'accuracy_history': accuracy_history,
                        'loss_history': loss_history
                    }
                    yield f"data: {json.dumps(update_data)}\n\n"
                    await asyncio.sleep(0.1)  # Small delay
            
            # Final update with complete data
            final_data = {
                'status': 'complete',
                'user_id': new_user_id if request.create_new else request.user_id,
                'message': 'Model trained successfully',
                'loss_history': loss_history,
                'accuracy_history': accuracy_history
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            print(f"Error in train_model_sse: {str(e)}")
            import traceback
            print(traceback.format_exc())
            error_data = {
                'status': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_updates(), 
        media_type="text/event-stream"
    )

@app.post("/train_with_corpus_sse")
async def train_with_corpus_sse(request: TrainRequest):
    async def generate_updates():
        try:
            # Create LoRA model if it doesn't exist
            if request.user_id not in model_manager.lora_models:
                model = model_manager.initialize_lora_model(request.user_id)
            else:
                model = model_manager.lora_models[request.user_id]

            # Move model to appropriate device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            device = "mps" if torch.backends.mps.is_available() else device
            model = model.to(device)
            print(f"Training on device: {device}")
            
            # More aggressive garbage collection
            import gc
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # Combine all texts into one
            combined_text = "\n\n".join(request.training_text.split("\n"))

            # Encode the text
            inputs = model_manager.tokenizer(
                combined_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Training loop with improved settings
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request.epochs)
            loss_history = []
            accuracy_history = []
            
            # Send a starting message
            yield f"data: {json.dumps({'status': 'starting', 'epoch': 0, 'total_epochs': request.epochs})}\n\n"
            await asyncio.sleep(0.1)  # Small delay for client to process

            for epoch in range(request.epochs):
                # Forward pass
                outputs = model(
                    input_ids=inputs['input_ids'],
                    labels=inputs['input_ids']
                )
                
                # Calculate token accuracy
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                labels = inputs['input_ids']
                
                # Shift labels and predictions for autoregressive comparison
                shift_labels = labels[:, 1:].contiguous()
                shift_predictions = predictions[:, :-1].contiguous()
                
                # Calculate token-level accuracy
                correct_predictions = (shift_predictions == shift_labels).float().sum().item()
                total_tokens = shift_labels.numel()
                accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
                accuracy_history.append(accuracy)
                
                # Backward pass
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                loss_history.append(loss.item())
                
                # Print progress and send update every 5 epochs
                if (epoch + 1) % 5 == 0 or epoch == request.epochs - 1:
                    print(f"Epoch {epoch + 1}/{request.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
                    
                    # Send progress update
                    update_data = {
                        'status': 'progress',
                        'epoch': epoch + 1,
                        'total_epochs': request.epochs,
                        'accuracy': accuracy,
                        'loss': loss.item(),
                        'accuracy_history': accuracy_history,
                        'loss_history': loss_history
                    }
                    yield f"data: {json.dumps(update_data)}\n\n"
                    await asyncio.sleep(0.1)  # Small delay
            
            # Final update with complete data
            user_id = request.user_id
            # Generate a new user ID if this was a new user creation
            if request.create_new:
                new_user_id = str(uuid.uuid4())
                model_manager.lora_models[new_user_id] = model_manager.lora_models.pop(request.user_id)
                user_id = new_user_id
                
            final_data = {
                'status': 'complete',
                'user_id': user_id,
                'message': 'Model trained successfully',
                'loss_history': loss_history,
                'accuracy_history': accuracy_history
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        except Exception as e:
            print(f"Error in train_with_corpus_sse: {str(e)}")
            import traceback
            print(traceback.format_exc())
            error_data = {
                'status': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_updates(), 
        media_type="text/event-stream"
    )

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
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find all article links in the category
        article_urls = []
        
        # First try the category group divs
        category_groups = soup.find_all('div', {'class': 'mw-category-group'})
        if category_groups:
            for group in category_groups:
                links = group.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if href.startswith('/wiki/') and not any(x in href for x in ['Category:', 'Special:', 'Help:', 'Template:']):
                        article_urls.append(f"https://en.wikipedia.org{href}")
        
        # If no articles found in category groups, try the main content
        if not article_urls:
            main_content = soup.find('div', {'class': 'mw-parser-output'})
            if main_content:
                links = main_content.find_all('a', href=True)
                for link in links:
                    href = link['href']
                    if href.startswith('/wiki/') and not any(x in href for x in ['Category:', 'Special:', 'Help:', 'Template:']):
                        article_urls.append(f"https://en.wikipedia.org{href}")
        
        print(f"Found {len(article_urls)} articles in category")
        
        if not article_urls:
            print("No article URLs found")
            return []
            
        # Randomly select articles up to the requested count
        import random
        selected_count = min(count, len(article_urls))
        print(f"Selecting {selected_count} articles from {len(article_urls)} available")
        selected_urls = random.sample(article_urls, selected_count)
        print(f"Selected URLs: {selected_urls}")
        
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
                        print(f"Article content was empty: {url}")
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
    base_url = "https://www.gutenberg.org"
    
    # Define markers for the gutenberg text processing
    TEXT_START_MARKERS = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF PROJECT GUTENBERG EBOOK",
        "***START OF PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
        "STARTOF THE PROJECT GUTENBERG EBOOK",
        "START OF THE PROJECT GUTENBERG EBOOK",
        "START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    
    TEXT_END_MARKERS = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF PROJECT GUTENBERG EBOOK",
        "***END OF PROJECT GUTENBERG EBOOK",
        "End of Project Gutenberg",
        "End of the Project Gutenberg",
        "END OF THE PROJECT GUTENBERG EBOOK",
        "END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    
    LEGALESE_START_MARKERS = [
        "<<THIS ELECTRONIC VERSION OF",
    ]
    
    LEGALESE_END_MARKERS = [
        "SERVICE THAT CHARGES FOR DOWNLOAD",
    ]
    
    try:
        # Handle special cases for genre search
        print(f"Searching for books with genre: {genre}")
        
        # URL encode the genre for proper searching
        encoded_genre = urllib.parse.quote_plus(genre)
        
        # Map some common categories to better search terms
        genre_mapping = {
            "non-fiction": "nonfiction",
            "science-fiction": "science+fiction",
        }
        
        search_term = genre_mapping.get(genre, encoded_genre)
        search_url = f"{base_url}/ebooks/search/?query={search_term}"
        
        print(f"Using search URL: {search_url}")
        response = requests.get(search_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find book links - look for links that contain book numbers
        book_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            # Look for links that contain book numbers (e.g., /ebooks/1234)
            if '/ebooks/' in href and href.split('/ebooks/')[1].split('/')[0].isdigit():
                full_link = f"{base_url}{href}"
                if full_link not in book_links:  # Avoid duplicates
                    book_links.append(full_link)
        
        print(f"Found {len(book_links)} book links")
        
        if not book_links:
            # Try alternate search method using subject browsing
            subject_mapping = {
                "non-fiction": "Nonfiction",
                "science-fiction": "Science+fiction",
                "science": "Science",
                "biography": "Biography",
                "history": "History"
            }
            
            if genre in subject_mapping:
                subject_url = f"{base_url}/browse/subjects/{subject_mapping[genre]}"
                print(f"Trying subject browse URL: {subject_url}")
                subject_response = requests.get(subject_url)
                if subject_response.status_code == 200:
                    subject_soup = BeautifulSoup(subject_response.text, 'html.parser')
                    for link in subject_soup.find_all('a', href=True):
                        href = link['href']
                        if '/ebooks/' in href and href.split('/ebooks/')[1].split('/')[0].isdigit():
                            full_link = f"{base_url}{href}"
                            if full_link not in book_links:
                                book_links.append(full_link)
                    print(f"Found {len(book_links)} book links using subject browsing")
        
        if not book_links:
            print("No book links found after multiple search attempts")
            return []
            
        texts = []
        # Fetch up to count books
        selected_count = min(count, len(book_links))
        print(f"Attempting to fetch {selected_count} books from {len(book_links)} available")
        for book_url in book_links[:selected_count]:
            try:
                print(f"Fetching book: {book_url}")
                # Get the book page
                book_response = requests.get(book_url)
                book_response.raise_for_status()
                book_soup = BeautifulSoup(book_response.text, 'html.parser')
                
                # Find the download link for plain text
                download_link = None
                for link in book_soup.find_all('a', href=True):
                    href = link['href']
                    # Look for UTF-8 text download link
                    if 'txt' in href and 'utf-8' in href:
                        download_link = f"{base_url}{href}"
                        break
                
                if download_link:
                    print(f"Found download link: {download_link}")
                    # Download the text content
                    text_response = requests.get(download_link)
                    text_response.raise_for_status()
                    if text_response.status_code == 200:
                        # Get the raw text
                        raw_text = text_response.text
                        
                        # Strip headers and footers using the method from the Python gutenberg library
                        # This is a port of the algorithm from the gutenberg Python library
                        
                        lines = raw_text.splitlines()
                        cleaned_lines = []
                        i = 0
                        footer_found = False
                        ignore_section = False
                        
                        for line in lines:
                            reset = False
                            
                            # Check header (first 600 lines)
                            if i <= 600:
                                # Check if the header ends here
                                if any(marker in line for marker in TEXT_START_MARKERS):
                                    reset = True
                                
                                # If we found a header marker, reset any previously collected content
                                if reset:
                                    cleaned_lines = []
                                    i += 1
                                    continue
                            
                            # Check footer (after first 100 lines)
                            if i >= 100:
                                # Check if the footer begins
                                if any(marker in line for marker in TEXT_END_MARKERS):
                                    footer_found = True
                                
                                # If we found a footer marker, stop processing
                                if footer_found:
                                    break
                            
                            # Check for legalese sections
                            if any(marker in line for marker in LEGALESE_START_MARKERS):
                                ignore_section = True
                                i += 1
                                continue
                            elif any(marker in line for marker in LEGALESE_END_MARKERS):
                                ignore_section = False
                                i += 1
                                continue
                            
                            # Additional check for modern Project Gutenberg header
                            if i < 100 and line.startswith("The Project Gutenberg eBook of"):
                                # Skip this line and find a significant break after it
                                # Often there are several empty lines after the header
                                reset = True
                                cleaned_lines = []
                                i += 1
                                continue
                            
                            # If we're not in a section to ignore, add the line
                            if not ignore_section:
                                cleaned_lines.append(line.rstrip())
                            
                            i += 1
                        
                        # Join the cleaned lines
                        clean_text = "\n".join(cleaned_lines)
                        
                        # Additional cleanup: remove multiple consecutive blank lines
                        import re
                        clean_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', clean_text)
                        
                        if clean_text.strip():  # Only add non-empty texts
                            texts.append(clean_text)
                            print(f"Successfully added book from {book_url}")
                        else:
                            print(f"Book content was empty: {book_url}")
                else:
                    print(f"No download link found for book: {book_url}")
            except Exception as e:
                print(f"Error fetching book: {str(e)}")
                import traceback
                print(traceback.format_exc())
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
            model = model_manager.initialize_lora_model(request.user_id)
        else:
            model = model_manager.lora_models[request.user_id]

        # Move model to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "mps" if torch.backends.mps.is_available() else device
        model = model.to(device)
        print(f"Training on device: {device}")
        
        
        # More aggressive garbage collection
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Combine all texts into one
        combined_text = "\n\n".join(request.training_text.split("\n"))

        # Encode the text
        inputs = model_manager.tokenizer(
            combined_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Training loop with improved settings
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request.epochs)
        loss_history = []
        accuracy_history = []

        for epoch in range(request.epochs):
            # Forward pass
            outputs = model(
                input_ids=inputs['input_ids'],
                labels=inputs['input_ids']
            )
            
            # Calculate token accuracy
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            labels = inputs['input_ids']
            
            # Shift labels and predictions for autoregressive comparison
            shift_labels = labels[:, 1:].contiguous()
            shift_predictions = predictions[:, :-1].contiguous()
            
            # Calculate token-level accuracy
            correct_predictions = (shift_predictions == shift_labels).float().sum().item()
            total_tokens = shift_labels.numel()
            accuracy = correct_predictions / total_tokens if total_tokens > 0 else 0
            accuracy_history.append(accuracy)
            
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
                print(f"Epoch {epoch + 1}/{request.epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        # Generate a new user ID if this was a new user creation
        if request.create_new:
            new_user_id = str(uuid.uuid4())
            model_manager.lora_models[new_user_id] = model_manager.lora_models.pop(request.user_id)
            return {
                "user_id": new_user_id,
                "message": "Model trained successfully",
                "loss_history": loss_history,
                "accuracy_history": accuracy_history
            }
        
        return {
            "user_id": request.user_id,
            "message": "Model trained successfully",
            "loss_history": loss_history,
            "accuracy_history": accuracy_history
        }

    except Exception as e:
        print(f"Error in train_with_corpus: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Add a last_used timestamp to models
# Periodically unload models that haven't been used recently
def unload_inactive_models():
    current_time = time.time()
    for user_id in list(model_manager.lora_models.keys()):
        if current_time - model_manager.last_used.get(user_id, 0) > 3600:  # 1 hour
            print(f"Unloading inactive model for user {user_id}")
            del model_manager.lora_models[user_id]
            if device == "cuda":
                torch.cuda.empty_cache()

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
