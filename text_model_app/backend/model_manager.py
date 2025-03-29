from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from pathlib import Path

class ModelManager:
    def __init__(self):
        # Just load the pre-trained Tiny-LLM model and tokenizer
        self.model_name = "arnir0/Tiny-LLM"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.lora_models = {}
        self.models_dir = Path("lora_models")
        self.models_dir.mkdir(exist_ok=True)

    def create_lora_model_for_user(self, user_id: str):
        """Create a LoRA adapter for the base model"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        self.lora_models[user_id] = get_peft_model(self.base_model, lora_config)
        return self.lora_models[user_id] 