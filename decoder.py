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


def sample_next_tokens_batched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    num_samples: int,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int = 1,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
    batch_size: int = 32,
    show_progress: bool = True
) -> List[int]:
    """
    Sample multiple next tokens using batched generation (much faster than individual calls).
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt string
        num_samples: Total number of samples to generate
        temperature: Sampling temperature
        top_k: Top-k sampling parameter
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum number of new tokens to generate
        device: Device to run on
        generator: Optional random generator for reproducibility
        batch_size: Number of samples to generate per batch
        show_progress: Whether to print progress updates
    
    Returns:
        List of sampled token IDs (length = num_samples)
    """
    import time
    
    # Tokenize prompt once
    prompt_inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
    
    # Replicate prompt for batch
    batch_prompt_ids = prompt_inputs['input_ids'].repeat(batch_size, 1)
    batch_attention_mask = prompt_inputs['attention_mask'].repeat(batch_size, 1)
    
    all_token_ids = []
    num_batches = (num_samples + batch_size - 1) // batch_size
    batch_start_time = time.time()
    total_start_time = time.time()
    
    # Generate in batches
    with torch.no_grad():
        for batch_idx, batch_start in enumerate(range(0, num_samples, batch_size), 1):
            # Determine actual batch size for last batch
            current_batch_size = min(batch_size, num_samples - batch_start)
            
            if current_batch_size < batch_size:
                # Last batch might be smaller
                batch_prompt_ids = prompt_inputs['input_ids'].repeat(current_batch_size, 1)
                batch_attention_mask = prompt_inputs['attention_mask'].repeat(current_batch_size, 1)
            
            batch_inputs = {
                'input_ids': batch_prompt_ids[:current_batch_size],
                'attention_mask': batch_attention_mask[:current_batch_size]
            }
            
            # Generate batch
            batch_gen_start = time.time()
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                generator=generator
            )
            batch_gen_time = time.time() - batch_gen_start
            
            # Extract the last token from each sequence in the batch
            batch_token_ids = outputs[:, -1].cpu().tolist()
            all_token_ids.extend(batch_token_ids)
            
            # Progress reporting
            if show_progress:
                samples_done = len(all_token_ids)
                elapsed_total = time.time() - total_start_time
                avg_time_per_sample = elapsed_total / samples_done
                remaining_samples = num_samples - samples_done
                eta_seconds = avg_time_per_sample * remaining_samples
                
                print(f"  Batch {batch_idx}/{num_batches}: {samples_done}/{num_samples} samples | "
                      f"Batch time: {batch_gen_time:.2f}s | "
                      f"Total: {elapsed_total:.1f}s | "
                      f"Avg: {avg_time_per_sample*1000:.1f}ms/sample | "
                      f"ETA: {eta_seconds/60:.1f}min", end='\r')
    
    if show_progress:
        total_time = time.time() - total_start_time
        print()  # New line after progress
        print(f"  Completed {num_samples} samples in {total_time:.1f} seconds "
              f"({total_time/num_samples*1000:.2f} ms per sample)")
    
    return all_token_ids

