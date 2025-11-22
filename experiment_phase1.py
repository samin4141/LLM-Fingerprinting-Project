"""
Phase 1 Experiment: LLM Fingerprinting
Main orchestration script for running the convergence experiment
"""

import os
import json
import csv
import torch
from pathlib import Path
from typing import Dict, List
import time

from config import config
from model_loader import load_model_and_tokenizer
from decoder import get_logits_for_tokens
from estimators import estimate_T_and_delta_b, estimate_T_and_delta_b_ac
from precompute_samples import load_precomputed_samples
import numpy as np


def run_experiment_for_N(
    N: int,
    samples_C1: np.ndarray,
    samples_C2: np.ndarray,
    logits_C1: Dict[int, float],
    logits_C2: Dict[int, float]
) -> Dict:
    """
    Run the experiment for a given sample size N using precomputed samples.
    
    Args:
        N: Number of samples to use (takes first N from precomputed samples)
        samples_C1: Precomputed samples for context C1
        samples_C2: Precomputed samples for context C2
        logits_C1: Dictionary of token_id -> logit for context C1
        logits_C2: Dictionary of token_id -> logit for context C2
    
    Returns:
        Dictionary with estimation results
    """
    if N > len(samples_C1) or N > len(samples_C2):
        raise ValueError(
            f"Requested N={N} exceeds precomputed samples "
            f"(C1: {len(samples_C1)}, C2: {len(samples_C2)}). "
            f"Please run precompute_samples.py with N_MAX >= {N}."
        )
    
    token_a = config.TOKEN_A_ID
    token_b = config.TOKEN_B_ID
    token_c = config.TOKEN_C_ID
    
    print(f"  Using first {N} samples from precomputed data...")
    start_time = time.time()
    
    # Take first N samples
    sample_subset_C1 = samples_C1[:N]
    sample_subset_C2 = samples_C2[:N]
    
    # Count occurrences using numpy (much faster than loops)
    n_a_C1 = np.sum(sample_subset_C1 == token_a)
    n_b_C1 = np.sum(sample_subset_C1 == token_b)
    n_c_C1 = np.sum(sample_subset_C1 == token_c)
    
    n_a_C2 = np.sum(sample_subset_C2 == token_a)
    n_b_C2 = np.sum(sample_subset_C2 == token_b)
    n_c_C2 = np.sum(sample_subset_C2 == token_c)
    
    elapsed = time.time() - start_time
    print(f"  Count computation completed in {elapsed:.4f} seconds")
    
    # Compute estimates using (a, b) pair
    T_hat_ab, delta_b_hat_ab = estimate_T_and_delta_b(
        z_a_C1=logits_C1[token_a],
        z_b_C1=logits_C1[token_b],
        z_a_C2=logits_C2[token_a],
        z_b_C2=logits_C2[token_b],
        n_a_C1=n_a_C1,
        n_b_C1=n_b_C1,
        N1=N,
        n_a_C2=n_a_C2,
        n_b_C2=n_b_C2,
        N2=N
    )
    
    # Compute estimates using (a, c) pair
    T_hat_ac, delta_b_hat_ac = estimate_T_and_delta_b_ac(
        z_a_C1=logits_C1[token_a],
        z_c_C1=logits_C1[token_c],
        z_a_C2=logits_C2[token_a],
        z_c_C2=logits_C2[token_c],
        n_a_C1=n_a_C1,
        n_c_C1=n_c_C1,
        N1=N,
        n_a_C2=n_a_C2,
        n_c_C2=n_c_C2,
        N2=N
    )
    
    # Compute difference between estimates
    diff_T_hat = abs(T_hat_ab - T_hat_ac) if (T_hat_ab is not None and T_hat_ac is not None) else None
    
    # Build result dictionary
    result = {
        "N": N,
        "T_hat_ab": T_hat_ab,
        "T_hat_ac": T_hat_ac,
        "delta_b_hat_ab": delta_b_hat_ab,
        "delta_b_hat_ac": delta_b_hat_ac,
        "diff_T_hat": diff_T_hat,
        "true_temperature": config.B_TEMPERATURE,
        "n_a_C1": n_a_C1,
        "n_b_C1": n_b_C1,
        "n_c_C1": n_c_C1,
        "n_a_C2": n_a_C2,
        "n_b_C2": n_b_C2,
        "n_c_C2": n_c_C2,
        "elapsed_seconds": elapsed
    }
    
    return result


def save_results(results: List[Dict], output_dir: str):
    """
    Save results to CSV and JSON files.
    
    Args:
        results: List of result dictionaries
        output_dir: Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "convergence_results.csv")
    if results:
        fieldnames = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nResults saved to {csv_path}")
    
    # Save as JSON
    json_path = os.path.join(output_dir, "convergence_results.json")
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_path}")


def main():
    """Main experiment execution"""
    print("=" * 70)
    print("Phase 1: LLM Fingerprinting Experiment")
    print("=" * 70)
    print()
    
    # Validate configuration
    if not config.validate():
        print("ERROR: Token IDs not set in config!")
        print("Please set TOKEN_A_ID, TOKEN_B_ID, and TOKEN_C_ID in config.py")
        print("You can inspect the tokenizer to find appropriate token IDs.")
        return
    
    # Load precomputed samples
    print("Loading precomputed samples...")
    try:
        samples_C1, samples_C2 = load_precomputed_samples(config.OUTPUT_DIR)
        print(f"✓ Loaded {len(samples_C1)} samples for C1")
        print(f"✓ Loaded {len(samples_C2)} samples for C2")
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("\nPlease run precompute_samples.py first to generate samples.")
        return
    print()
    
    # Check that N_MAX is sufficient
    max_N = max(config.SAMPLE_SIZES)
    if max_N > len(samples_C1) or max_N > len(samples_C2):
        print(f"WARNING: Maximum requested N ({max_N}) exceeds precomputed samples.")
        print(f"Precomputed: {len(samples_C1)} samples")
        print(f"Please run precompute_samples.py with N_MAX >= {max_N}")
        return
    
    # Load model and tokenizer (needed for logits and tokenizer.decode)
    print("Loading model for logit computation...")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    print()
    
    # Pre-compute logits from model A (white-box access)
    print("Pre-computing logits from model A (white-box)...")
    token_ids = config.get_token_ids()
    
    logits_C1 = get_logits_for_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=config.PROMPT_C1,
        token_ids=token_ids,
        device=config.DEVICE
    )
    
    logits_C2 = get_logits_for_tokens(
        model=model,
        tokenizer=tokenizer,
        prompt=config.PROMPT_C2,
        token_ids=token_ids,
        device=config.DEVICE
    )
    
    print("Logits for context C1:")
    for tid, logit in logits_C1.items():
        token_str = tokenizer.decode([tid])
        print(f"  Token {tid} ({token_str}): {logit:.4f}")
    
    print("Logits for context C2:")
    for tid, logit in logits_C2.items():
        token_str = tokenizer.decode([tid])
        print(f"  Token {tid} ({token_str}): {logit:.4f}")
    print()
    
    # Run experiment for each sample size
    results = []
    
    print("=" * 70)
    print("Running experiments for different sample sizes...")
    print("=" * 70)
    print()
    
    for N in config.SAMPLE_SIZES:
        print(f"Sample size N = {N}")
        print("-" * 70)
        
        for repeat in range(config.REPEATS_PER_N):
            if config.REPEATS_PER_N > 1:
                print(f"  Repeat {repeat + 1}/{config.REPEATS_PER_N}")
            
            result = run_experiment_for_N(
                N=N,
                samples_C1=samples_C1,
                samples_C2=samples_C2,
                logits_C1=logits_C1,
                logits_C2=logits_C2
            )
            
            results.append(result)
            
            # Print summary
            print(f"  Results:")
            print(f"    T_hat (a,b): {result['T_hat_ab']:.4f}" if result['T_hat_ab'] is not None else "    T_hat (a,b): None")
            print(f"    T_hat (a,c): {result['T_hat_ac']:.4f}" if result['T_hat_ac'] is not None else "    T_hat (a,c): None")
            print(f"    True T: {result['true_temperature']:.4f}")
            print(f"    |T_hat_ab - T_hat_ac|: {result['diff_T_hat']:.4f}" if result['diff_T_hat'] is not None else "    |T_hat_ab - T_hat_ac|: None")
            print()
        
        # Save intermediate results
        save_results(results, config.OUTPUT_DIR)
        print()
    
    print("=" * 70)
    print("Experiment completed!")
    print("=" * 70)
    
    # Final save
    save_results(results, config.OUTPUT_DIR)


if __name__ == "__main__":
    main()

