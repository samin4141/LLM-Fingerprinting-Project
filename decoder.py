"""
Decoder utilities for white-box logits and black-box sampling
"""

import torch
from typing import Dict, Optional, List
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_next_token_logits(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    device: str
) -> torch.Tensor:
    """
    Get logits for the next token position (white-box access to model A).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        device: Device to run on
    
    Returns:
        Tensor of shape [vocab_size] with logits for next token
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass without sampling
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Extract logits for last position: [batch_size, seq_len, vocab_size] -> [batch_size, vocab_size]
    next_token_logits = logits[:, -1, :]  # [1, vocab_size]
    
    # Return as 1D tensor
    return next_token_logits.squeeze(0)  # [vocab_size]


def get_logits_for_tokens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    token_ids: List[int],
    device: str
) -> Dict[int, float]:
    """
    Get logit values for specific tokens (white-box access to model A).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        token_ids: List of token IDs to extract logits for
        device: Device to run on
    
    Returns:
        Dictionary mapping token_id -> logit value
    """
    logits = get_next_token_logits(model, tokenizer, prompt, device)  # [vocab_size]
    
    return {tid: logits[tid].item() for tid in token_ids}


def sample_next_token_id_black_box(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int = 1,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None
) -> int:
    """
    Sample next token using black-box style sampling (model B behavior).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum number of new tokens to generate
        device: Device to run on
        generator: Optional random generator for reproducibility
    
    Returns:
        Sampled token ID as integer
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with specified decoding parameters
    # Note: top_k and top_p are both applied; top_k filters first, then top_p samples from remaining
    # Ensure chosen tokens (a, b, c) have rank < top_k in the distribution to avoid being filtered out
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            generator=generator
        )
    
    # Extract the last token (the newly generated one)
    generated_token_id = outputs[0, -1].item()
    
    return generated_token_id

