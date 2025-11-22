"""
Quick script to check if digit tokens '1', '2', '3' appear in both contexts
"""

from model_loader import load_model_and_tokenizer
from decoder import get_next_token_logits
from config import config
import torch

def check_digit_tokens():
    """Check digit tokens in both contexts"""
    import time
    start_time = time.time()
    
    print("=" * 70)
    print("DIGIT TOKEN CHECKER")
    print("=" * 70)
    print(f"\nDevice: {config.DEVICE}")
    if config.DEVICE == "cpu":
        print("⚠ WARNING: Running on CPU. This will be slow (5-15 minutes).")
        print("   Consider using GPU (CUDA) if available for much faster execution.")
    print("\nLoading model (this may take 2-5 minutes if not cached)...")
    print("   Model: Mistral-7B-Instruct-v0.2 (~7GB)")
    
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    load_time = time.time() - t0
    print(f"✓ Model loaded in {load_time:.1f} seconds\n")
    
    # Get logits for both contexts
    print("Getting logits for context C1...")
    print(f"   Prompt: {config.PROMPT_C1[:60]}...")
    t1 = time.time()
    logits_C1 = get_next_token_logits(model, tokenizer, config.PROMPT_C1, config.DEVICE)
    c1_time = time.time() - t1
    print(f"✓ C1 logits computed in {c1_time:.2f} seconds")
    
    print("\nGetting logits for context C2...")
    print(f"   Prompt: {config.PROMPT_C2[:60]}...")
    t2 = time.time()
    logits_C2 = get_next_token_logits(model, tokenizer, config.PROMPT_C2, config.DEVICE)
    c2_time = time.time() - t2
    print(f"✓ C2 logits computed in {c2_time:.2f} seconds")
    
    # Find token IDs for '1', '2', '3'
    print("\n" + "=" * 70)
    print("Checking digit tokens '1', '2', '3'")
    print("=" * 70)
    
    t3 = time.time()
    digit_tokens = ['1', '2', '3']
    token_ids = {}
    for digit in digit_tokens:
        # Try to find the token ID
        test_input = tokenizer(digit, return_tensors="pt")
        if len(test_input['input_ids'][0]) > 0:
            tid = test_input['input_ids'][0][0].item()
            token_ids[digit] = tid
        else:
            print(f"Warning: Could not find token for '{digit}'")
    token_id_time = time.time() - t3
    print(f"Token ID lookup: {token_id_time:.4f} seconds")
    
    # Check each digit token
    print(f"\n{'Token':<10} {'Token ID':<12} {'Rank C1':<12} {'Rank C2':<12} {'Prob C1':<12} {'Prob C2':<12}")
    print("-" * 70)
    
    # Get top-k for ranking (more efficient than full softmax)
    k = 100  # Check top 100
    print(f"\nComputing top-{k} tokens for ranking...")
    t4 = time.time()
    top_k_C1 = torch.topk(logits_C1, k)
    top_k_C2 = torch.topk(logits_C2, k)
    topk_time = time.time() - t4
    print(f"Top-k computation: {topk_time:.4f} seconds")
    
    # Only compute softmax for the tokens we care about (much faster)
    print("Computing probabilities for digit tokens...")
    t5 = time.time()
    digit_token_ids = [tid for tid in token_ids.values() if tid < len(logits_C1)]
    
    # Compute log-sum-exp trick for numerical stability when getting specific token probs
    # For specific tokens: prob = exp(logit - log_sum_exp(all_logits))
    t5a = time.time()
    log_sum_exp_C1 = torch.logsumexp(logits_C1, dim=0)
    log_sum_exp_C2 = torch.logsumexp(logits_C2, dim=0)
    logsumexp_time = time.time() - t5a
    print(f"  Log-sum-exp computation: {logsumexp_time:.4f} seconds")
    
    # Get probabilities for our specific tokens only
    t5b = time.time()
    probs_C1_dict = {tid: torch.exp(logits_C1[tid] - log_sum_exp_C1).item() 
                     for tid in digit_token_ids}
    probs_C2_dict = {tid: torch.exp(logits_C2[tid] - log_sum_exp_C2).item() 
                     for tid in digit_token_ids}
    prob_compute_time = time.time() - t5b
    prob_total_time = time.time() - t5
    print(f"  Probability extraction: {prob_compute_time:.4f} seconds")
    print(f"Total probability computation: {prob_total_time:.4f} seconds")
    
    t6 = time.time()
    for digit, tid in token_ids.items():
        if tid < len(logits_C1):
            prob_C1 = probs_C1_dict.get(tid, 0.0)
            prob_C2 = probs_C2_dict.get(tid, 0.0)
            
            # Find rank
            rank_C1 = (top_k_C1.indices == tid).nonzero(as_tuple=True)
            rank_C2 = (top_k_C2.indices == tid).nonzero(as_tuple=True)
            
            rank_C1_val = rank_C1[0].item() + 1 if len(rank_C1[0]) > 0 else ">100"
            rank_C2_val = rank_C2[0].item() + 1 if len(rank_C2[0]) > 0 else ">100"
            
            print(f"{digit:<10} {tid:<12} {str(rank_C1_val):<12} {str(rank_C2_val):<12} {prob_C1:<12.6f} {prob_C2:<12.6f}")
    rank_time = time.time() - t6
    print(f"Rank computation and printing: {rank_time:.4f} seconds")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)
    
    # Check if all three digits are in top-k of both contexts
    all_in_top_k = all(
        (token_ids[d] in top_k_C1.indices) and (token_ids[d] in top_k_C2.indices)
        for d in digit_tokens if d in token_ids
    )
    
    if all_in_top_k:
        print("✓ All digit tokens '1', '2', '3' appear in top-100 of both contexts!")
        print("\nGood choice for TOKEN_A_ID, TOKEN_B_ID, TOKEN_C_ID:")
        for i, digit in enumerate(digit_tokens, 1):
            if digit in token_ids:
                print(f"  TOKEN_{['A', 'B', 'C'][i-1]}_ID = {token_ids[digit]}  # '{digit}'")
    else:
        print("⚠ Some digit tokens may not be in top-100 of both contexts.")
        print("Consider using other tokens from the common tokens list.")
    
    total_time = time.time() - start_time
    
    # Print timing breakdown
    print("\n" + "=" * 70)
    print("TIMING BREAKDOWN")
    print("=" * 70)
    print(f"{'Operation':<35} {'Time (s)':<12} {'% of Total':<12}")
    print("-" * 70)
    print(f"{'1. Model loading':<35} {load_time:<12.2f} {load_time/total_time*100:<12.1f}%")
    print(f"{'2. Getting logits for C1':<35} {c1_time:<12.2f} {c1_time/total_time*100:<12.1f}%")
    print(f"{'3. Getting logits for C2':<35} {c2_time:<12.2f} {c2_time/total_time*100:<12.1f}%")
    print(f"{'4. Token ID lookup':<35} {token_id_time:<12.4f} {token_id_time/total_time*100:<12.1f}%")
    print(f"{'5. Top-k computation':<35} {topk_time:<12.4f} {topk_time/total_time*100:<12.1f}%")
    print(f"{'6. Log-sum-exp':<35} {logsumexp_time:<12.4f} {logsumexp_time/total_time*100:<12.1f}%")
    print(f"{'7. Probability extraction':<35} {prob_compute_time:<12.4f} {prob_compute_time/total_time*100:<12.1f}%")
    print(f"{'8. Rank computation':<35} {rank_time:<12.4f} {rank_time/total_time*100:<12.1f}%")
    print("-" * 70)
    print(f"{'TOTAL':<35} {total_time:<12.2f} {'100.0%':<12}")
    print(f"\nTotal execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print()

if __name__ == "__main__":
    check_digit_tokens()


