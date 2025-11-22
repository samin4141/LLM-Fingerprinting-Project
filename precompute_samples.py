"""
Precompute and store samples for Phase 1 experiment.

This script samples N_MAX tokens once for each context and stores them to disk.
The stored samples can then be reused for all smaller N values, making the
experiment much faster (especially on CPU).

Usage:
    python precompute_samples.py
"""

import os
import json
import numpy as np
import torch
import time
from pathlib import Path

from config import config
from model_loader import load_model_and_tokenizer
from decoder import sample_next_tokens_batched


def precompute_samples(output_dir: str = None, batch_size: int = 32):
    """
    Precompute samples for both contexts and save to disk.
    
    Args:
        output_dir: Directory to save samples (default: config.OUTPUT_DIR)
        batch_size: Batch size for batched generation
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("PRECOMPUTING SAMPLES")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {config.MODEL_NAME}")
    print(f"  Device: {config.DEVICE}")
    print(f"  N_MAX: {config.N_MAX}")
    print(f"  Batch size: {batch_size}")
    print(f"  Temperature: {config.B_TEMPERATURE}")
    print(f"  Top-k: {config.B_TOP_K}")
    print(f"  Top-p: {config.B_TOP_P}")
    print(f"  Random seed: {config.GLOBAL_SEED}")
    print()
    
    # Load model and tokenizer
    print("Loading model...")
    start_time = time.time()
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.1f} seconds\n")
    
    # Initialize random generator for reproducibility
    generator = torch.Generator(device=config.DEVICE)
    generator.manual_seed(config.GLOBAL_SEED)
    
    # Sample for context C1
    print("=" * 70)
    print(f"Sampling {config.N_MAX} tokens for context C1...")
    print(f"Prompt: {config.PROMPT_C1}")
    print("-" * 70)
    
    t1 = time.time()
    samples_C1 = sample_next_tokens_batched(
        model=model,
        tokenizer=tokenizer,
        prompt=config.PROMPT_C1,
        num_samples=config.N_MAX,
        temperature=config.B_TEMPERATURE,
        top_k=config.B_TOP_K,
        top_p=config.B_TOP_P,
        max_new_tokens=config.B_MAX_NEW_TOKENS,
        device=config.DEVICE,
        generator=generator,
        batch_size=batch_size
    )
    c1_time = time.time() - t1
    print(f"✓ C1 sampling completed in {c1_time:.1f} seconds ({c1_time/config.N_MAX*1000:.2f} ms per sample)")
    
    # Sample for context C2
    print("\n" + "=" * 70)
    print(f"Sampling {config.N_MAX} tokens for context C2...")
    print(f"Prompt: {config.PROMPT_C2}")
    print("-" * 70)
    
    t2 = time.time()
    samples_C2 = sample_next_tokens_batched(
        model=model,
        tokenizer=tokenizer,
        prompt=config.PROMPT_C2,
        num_samples=config.N_MAX,
        temperature=config.B_TEMPERATURE,
        top_k=config.B_TOP_K,
        top_p=config.B_TOP_P,
        max_new_tokens=config.B_MAX_NEW_TOKENS,
        device=config.DEVICE,
        generator=generator,
        batch_size=batch_size
    )
    c2_time = time.time() - t2
    print(f"✓ C2 sampling completed in {c2_time:.1f} seconds ({c2_time/config.N_MAX*1000:.2f} ms per sample)")
    
    # Save samples
    print("\n" + "=" * 70)
    print("Saving samples to disk...")
    print("-" * 70)
    
    # Save as NumPy arrays (efficient binary format)
    np_path_C1 = os.path.join(output_dir, "samples_C1.npy")
    np_path_C2 = os.path.join(output_dir, "samples_C2.npy")
    np.save(np_path_C1, np.array(samples_C1, dtype=np.int32))
    np.save(np_path_C2, np.array(samples_C2, dtype=np.int32))
    print(f"✓ Saved NumPy arrays:")
    print(f"  {np_path_C1}")
    print(f"  {np_path_C2}")
    
    # Save as JSON (human-readable, for inspection)
    json_path = os.path.join(output_dir, "samples_metadata.json")
    metadata = {
        "N_MAX": config.N_MAX,
        "model_name": config.MODEL_NAME,
        "device": config.DEVICE,
        "temperature": config.B_TEMPERATURE,
        "top_k": config.B_TOP_K,
        "top_p": config.B_TOP_P,
        "random_seed": config.GLOBAL_SEED,
        "prompt_C1": config.PROMPT_C1,
        "prompt_C2": config.PROMPT_C2,
        "num_samples_C1": len(samples_C1),
        "num_samples_C2": len(samples_C2),
        "sampling_time_C1_seconds": c1_time,
        "sampling_time_C2_seconds": c2_time,
        "total_time_seconds": time.time() - start_time
    }
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata: {json_path}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SAMPLE STATISTICS")
    print("=" * 70)
    
    unique_C1, counts_C1 = np.unique(samples_C1, return_counts=True)
    unique_C2, counts_C2 = np.unique(samples_C2, return_counts=True)
    
    print(f"\nContext C1:")
    print(f"  Total samples: {len(samples_C1)}")
    print(f"  Unique tokens: {len(unique_C1)}")
    print(f"  Top 10 tokens:")
    top_indices_C1 = np.argsort(counts_C1)[-10:][::-1]
    for idx in top_indices_C1:
        token_id = unique_C1[idx]
        count = counts_C1[idx]
        token_str = tokenizer.decode([token_id])
        print(f"    Token {token_id:5d} ({token_str:>10s}): {count:5d} ({count/len(samples_C1)*100:5.2f}%)")
    
    print(f"\nContext C2:")
    print(f"  Total samples: {len(samples_C2)}")
    print(f"  Unique tokens: {len(unique_C2)}")
    print(f"  Top 10 tokens:")
    top_indices_C2 = np.argsort(counts_C2)[-10:][::-1]
    for idx in top_indices_C2:
        token_id = unique_C2[idx]
        count = counts_C2[idx]
        token_str = tokenizer.decode([token_id])
        print(f"    Token {token_id:5d} ({token_str:>10s}): {count:5d} ({count/len(samples_C2)*100:5.2f}%)")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PRECOMPUTATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"\nSamples saved to: {output_dir}")
    print("\nYou can now run experiment_phase1.py - it will use these precomputed samples.")
    print()


def load_precomputed_samples(output_dir: str = None):
    """
    Load precomputed samples from disk.
    
    Args:
        output_dir: Directory containing samples (default: config.OUTPUT_DIR)
    
    Returns:
        Tuple of (samples_C1, samples_C2) as numpy arrays
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIR
    
    np_path_C1 = os.path.join(output_dir, "samples_C1.npy")
    np_path_C2 = os.path.join(output_dir, "samples_C2.npy")
    
    if not os.path.exists(np_path_C1) or not os.path.exists(np_path_C2):
        raise FileNotFoundError(
            f"Precomputed samples not found in {output_dir}.\n"
            f"Please run precompute_samples.py first."
        )
    
    samples_C1 = np.load(np_path_C1)
    samples_C2 = np.load(np_path_C2)
    
    return samples_C1, samples_C2


if __name__ == "__main__":
    precompute_samples()

