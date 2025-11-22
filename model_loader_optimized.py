"""
Memory-optimized model loading for CPU (alternative to model_loader.py)
Use this if you're running out of RAM with the standard loader
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import Tuple
import warnings
import gc

warnings.filterwarnings("ignore")


def load_model_and_tokenizer(
    model_name: str,
    device: str
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Mistral model with aggressive memory optimization for CPU.
    
    Args:
        model_name: HuggingFace model identifier
        device: Device to load model on ("cuda" or "cpu")
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print(f"Device: {device}")
    print("Using memory-optimized loading...")
    
    # Clear cache before loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with maximum memory efficiency
    if device == "cuda" and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        # For CPU: Use the most memory-efficient settings
        print("Loading model with CPU memory optimization...")
        print("This may take several minutes. Please be patient...")
        
        # Use float16 on CPU (half precision) - requires ~7GB instead of ~14GB
        # Note: Some operations may be slower, but it uses half the memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 even on CPU
            device_map=None,
            low_cpu_mem_usage=True,
            max_memory={"cpu": "20GiB"}  # Set memory limit
        )
        
        print("Moving model to CPU...")
        model.to(device)
        
        # Clear cache after loading
        gc.collect()
    
    model.eval()
    
    print("Model loaded successfully!")
    return model, tokenizer

