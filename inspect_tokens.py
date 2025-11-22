"""
Helper script to inspect tokenizer and find appropriate token IDs
for tokens a, b, c that appear in next-token distributions
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLMting
from decoder import get_next_token_logits
from config import config


def inspect_tokenizer(tokenizer, model, device):
    """
    Inspect tokenizer and find tokens that appear in next-token distributions
    for both prompts C1 and C2.
    """
    print("=" * 70)
    print("Token Inspector")
    print("=" * 70)
    print()
    
    # Get logits for both contexts
    print("Getting logits for context C1...")
    logits_C1 = get_next_token_logits(model, tokenizer, config.PROMPT_C1, device)
    
    print("Getting logits for context C2...")
    logits_C2 = get_next_token_logits(model, tokenizer, config.PROMPT_C2, device)
    
    # Get top-k tokens for each context
    k = 20
    top_k_C1 = torch.topk(logits_C1, k)
    top_k_C2 = torch.topk(logits_C2, k)
    
    print(f"\nTop {k} tokens for context C1:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Token ID':<10} {'Token':<20} {'Logit':<12} {'Probability':<12}")
    print("-" * 70)
    
    probs_C1 = torch.softmax(logits_C1, dim=0)
    for i, (logit_val, token_id) in enumerate(zip(top_k_C1.values, top_k_C1.indices)):
        token_str = tokenizer.decode([token_id.item()])
        prob = probs_C1[token_id].item()
        print(f"{i+1:<6} {token_id.item():<10} {repr(token_str):<20} {logit_val.item():<12.4f} {prob:<12.6f}")
    
    print(f"\nTop {k} tokens for context C2:")
    print("-" * 70)
    print(f"{'Rank':<6} {'Token ID':<10} {'Token':<20} {'Logit':<12} {'Probability':<12}")
    print("-" * 70)
    
    probs_C2 = torch.softmax(logits_C2, dim=0)
    for i, (logit_val, token_id) in enumerate(zip(top_k_C2.values, top_k_C2.indices)):
        token_str = tokenizer.decode([token_id.item()])
        prob = probs_C2[token_id].item()
        print(f"{i+1:<6} {token_id.item():<10} {repr(token_str):<20} {logit_val.item():<12.4f} {prob:<12.6f}")
    
    # Find tokens that appear in top-k of both contexts
    print(f"\nTokens appearing in top {k} of BOTH contexts:")
    print("-" * 70)
    
    top_tokens_C1 = set(top_k_C1.indices.tolist())
    top_tokens_C2 = set(top_k_C2.indices.tolist())
    common_tokens = top_tokens_C1.intersection(top_tokens_C2)
    
    if common_tokens:
        print(f"{'Token ID':<10} {'Token':<20} {'Rank C1':<10} {'Rank C2':<10} {'Logit C1':<12} {'Logit C2':<12}")
        print("-" * 70)
        
        for token_id in sorted(common_tokens, key=lambda tid: probs_C1[tid] + probs_C2[tid], reverse=True):
            token_str = tokenizer.decode([token_id])
            rank_C1 = (top_k_C1.indices == token_id).nonzero(as_tuple=True)[0].item() + 1
            rank_C2 = (top_k_C2.indices == token_id).nonzero(as_tuple=True)[0].item() + 1
            logit_C1 = logits_C1[token_id].item()
            logit_C2 = logits_C2[token_id].item()
            print(f"{token_id:<10} {repr(token_str):<20} {rank_C1:<10} {rank_C2:<10} {logit_C1:<12.4f} {logit_C2:<12.4f}")
        
        print("\n" + "=" * 70)
        print("RECOMMENDATION:")
        print("=" * 70)
        print("Choose three tokens from the common tokens above.")
        print("Good candidates are tokens that:")
        print("  1. Have reasonable probabilities in both contexts (>0.01, ideally >0.05)")
        print("  2. Don't appear literally in the prompt text")
        print("  3. Are distinct from each other")
        print("  4. Have rank < B_TOP_K (currently {}) in both contexts".format(config.B_TOP_K))
        print("     (to ensure they're not filtered out by top-k sampling)")
        print()
        print("Example: If you see tokens like '1', '2', '3' or 'one', 'two', 'three',")
        print("these would be good candidates for TOKEN_A_ID, TOKEN_B_ID, TOKEN_C_ID.")
        print()
        print("Note: The prompts use words ('one', 'two', 'three'), so digit tokens")
        print("('1', '2', '3') are good choices as they don't appear literally in prompts.")
        print()
        print("Update config.py with your chosen token IDs:")
        print("  config.TOKEN_A_ID = <your_token_id>")
        print("  config.TOKEN_B_ID = <your_token_id>")
        print("  config.TOKEN_C_ID = <your_token_id>")
    else:
        print("No common tokens found in top-k of both contexts.")
        print("You may need to increase k or choose tokens manually.")
    
    print()


def main():
    """Main execution"""
    from model_loader import load_model_and_tokenizer
    
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.MODEL_NAME, config.DEVICE)
    print()
    
    inspect_tokenizer(tokenizer, model, config.DEVICE)


if __name__ == "__main__":
    main()

