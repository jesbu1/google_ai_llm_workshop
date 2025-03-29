from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from unittest.mock import patch
from pathlib import Path
from transformers.dynamic_module_utils import get_imports
import torch
import os


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """Work around for https://huggingface.co/microsoft/phi-1_5/discussions/72."""
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


class ModelManager:
    def __init__(self):
        # Just load the pre-trained Tiny-LLM model and tokenizer
        #self.model_name = "arnir0/Tiny-LLM"
        self.model_name = "facebook/MobileLLM-125M"
        self._tokenizer = None  # Protected attribute
        self.base_model = None
        self.lora_models = {}
        self.models_dir = Path("lora_models")
        self.models_dir.mkdir(exist_ok=True)
        self._initialize_model_and_tokenizer()

    def _initialize_model_and_tokenizer(self):
        """Initialize the tokenizer and model safely"""
        with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
            print(f"Initializing tokenizer for {self.model_name}")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, use_fast=False
            )

            # Add special tokens if they don't exist
            special_tokens = {
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>",
                "pad_token": "</s>",  # Use EOS as PAD by default
            }

            # Add any missing special tokens
            special_tokens_to_add = {}
            for token_name, token_value in special_tokens.items():
                if getattr(self._tokenizer, token_name) is None:
                    special_tokens_to_add[token_name] = token_value

            if special_tokens_to_add:
                print(f"Adding special tokens: {special_tokens_to_add}")
                self._tokenizer.add_special_tokens(special_tokens_to_add)

            print(f"Initializing base model for {self.model_name}")
            self.base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                pad_token_id=self._tokenizer.pad_token_id,
                bos_token_id=self._tokenizer.bos_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

            # Ensure the model's config has the correct token IDs
            self.base_model.config.pad_token_id = self._tokenizer.pad_token_id
            self.base_model.config.bos_token_id = self._tokenizer.bos_token_id
            self.base_model.config.eos_token_id = self._tokenizer.eos_token_id

            # Resize token embeddings if needed
            print("Resizing token embeddings if needed")
            self.base_model.resize_token_embeddings(len(self._tokenizer))

            print("Model and tokenizer initialization complete")

    @property
    def tokenizer(self):
        """Safe access to tokenizer"""
        if self._tokenizer is None:
            self._initialize_model_and_tokenizer()
        return self._tokenizer

    def initialize_lora_model(self, user_id: str):
        """Create a LoRA adapter for the base model"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.lora_models[user_id] = get_peft_model(self.base_model, lora_config)
        return self.lora_models[user_id] 