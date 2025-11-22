"""
Model loading utilities for white-box access to Mistral-7B
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple
import warnings

warnings.filterwarnings("ignore")


def load_model_and_tokenizer(
    model_name: str,
    device: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Mistral model and tokenizer for white-box access.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ("cuda" or "cpu")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory-efficient settings
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        # For CPU: use float16 to reduce memory from ~14GB to ~7GB
        # This is essential for systems with 16GB RAM where only ~10-12GB is available
        print("Using float16 precision on CPU (reduces memory usage by ~50%)")
        print("This may take a few minutes to load...")
        
        try:
            # Use float16 on CPU - uses ~7GB instead of ~14GB
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Half precision - much less memory
                device_map=None,
                low_cpu_mem_usage=True
            )
            model.to(device)
        except (RuntimeError, OSError) as e:
            # If float16 fails, try float32 as last resort
            print(f"Warning: float16 loading failed ({e})")
            print("Trying float32 - this requires ~14GB free RAM")
            print("If this fails, consider using a smaller model or closing other applications")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=None,
                low_cpu_mem_usage=True
            )
            model.to(device)
    
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer

