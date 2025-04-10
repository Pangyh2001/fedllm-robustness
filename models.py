import torch
import torch.nn as nn
from transformers import GemmaForSequenceClassification, GemmaTokenizer
from peft import LoraConfig, TaskType, get_peft_model

def load_model(model_name, num_labels):
    """Load the model and tokenizer with PEFT/LoRA setup for multiple LLM architectures"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Determine model type and load appropriate model and tokenizer
    if "gemma" in model_name.lower():
        from transformers import GemmaForSequenceClassification, GemmaTokenizer
        
        # Load tokenizer first
        tokenizer = GemmaTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load the model with sequence classification head
        model = GemmaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            pad_token_id=tokenizer.pad_token_id,  # Pass the pad token ID to the model
        ).to(device)
        
        # Target modules for Gemma
        target_modules = ["q_proj", "v_proj"]
        
    elif "mistral" in model_name.lower():
        from transformers import MistralForSequenceClassification, AutoTokenizer
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Make sure pad_token is properly set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # This is crucial - make sure the model knows about the padding token
        
        # Load the model with sequence classification head
        model = MistralForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            pad_token_id=tokenizer.pad_token_id,  # Set the pad token ID explicitly\
            
            
        ).to(device)
        
        # Target modules for Mistral
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
    elif "zephyr" in model_name.lower():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load the model with sequence classification head
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            pad_token_id=tokenizer.pad_token_id,  # Set pad token ID explicitly
        ).to(device)
            
        # Target modules for Zephyr (which is based on Mistral)
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
    elif "llama" in model_name.lower():
        from transformers import LlamaForSequenceClassification, LlamaTokenizer
        
        # Load tokenizer first
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load the model with sequence classification head
        model = LlamaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            pad_token_id=tokenizer.pad_token_id,  # Set pad token ID explicitly
        ).to(device)
            
        # Target modules for Llama-2
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    # Add special tokens to the model's config to ensure it knows about the padding token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Configure PEFT-LoRA for parameter-efficient fine-tuning
    from peft import LoraConfig, TaskType, get_peft_model
    
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=32,
        target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Double-check padding settings after PEFT wrapping
    model.config.pad_token_id = tokenizer.pad_token_id
    
    return model, tokenizer, device