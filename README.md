# Phase 1: LLM Fingerprinting Experiment

This project implements Phase 1 of an LLM fingerprinting experiment to empirically test whether we can estimate hidden decoding parameters (temperature, logit bias) using white-box access to model A and black-box access to model B.

## Project Structure

```
.
├── config.py                 # Configuration parameters
├── model_loader.py           # Model and tokenizer loading utilities
├── decoder.py                # White-box logits and black-box sampling
├── estimators.py             # Temperature and logit-bias estimation functions
├── precompute_samples.py     # Precompute and store samples (run once)
├── experiment_phase1.py      # Main experiment orchestration script
├── analysis_phase1.py        # Results analysis and plotting
├── inspect_tokens.py         # Helper script to find appropriate token IDs
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Setup

1. **Install dependencies:**
   
   **For CPU-only:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Inspect tokenizer to find token IDs:**
   ```bash
   python inspect_tokens.py
   ```
   This will show you the top tokens in the next-token distribution for both prompts. Choose three tokens (a, b, c) that:
   - Appear in top-k of both contexts
   - Have reasonable probabilities (>0.01)
   - Don't appear literally in the prompt text
   - Are distinct from each other

3. **Update config.py with your chosen token IDs:**
   ```python
   config.TOKEN_A_ID = <your_token_id>
   config.TOKEN_B_ID = <your_token_id>
   config.TOKEN_C_ID = <your_token_id>
   ```

## Running the Experiment

The experiment uses a two-step process to optimize performance:

1. **Precompute samples (run once):**
   ```bash
   python precompute_samples.py
   ```
   
   This will:
   - Load the Mistral-7B model
   - Sample N_MAX tokens (default: 20,000) for each context using batched generation
   - Store samples to disk as `.npy` files
   - This is the slow step (especially on CPU), but only needs to be done once
   
   **Note:** The samples are reused for all smaller N values, making subsequent runs very fast.

2. **Run the main experiment:**
   ```bash
   python experiment_phase1.py
   ```
   
   This will:
   - Load the precomputed samples from disk
   - Load the model to compute logits from model A (white-box)
   - For each sample size N, use the first N samples to compute counts
   - Estimate temperature T and logit-bias difference Δb
   - Save results to `results_phase1/convergence_results.csv` and `.json`
   
   **Note:** This step is now very fast since it just counts tokens from precomputed data.

3. **Analyze results:**
   ```bash
   python analysis_phase1.py
   ```
   
   This will:
   - Load the results
   - Compute summary statistics
   - Generate convergence plots showing:
     - Temperature estimates vs sample size
     - Convergence of estimator difference
     - Estimation error vs sample size
     - Sample counts visualization

## Configuration

Key parameters in `config.py`:

- **MODEL_NAME**: HuggingFace model identifier (default: "mistralai/Mistral-7B-Instruct-v0.2")
- **DEVICE**: "cuda" or "cpu" (auto-detected)
- **PROMPT_C1, PROMPT_C2**: The two context prompts
- **B_TEMPERATURE**: Temperature for black-box model B (default: 0.8)
- **B_TOP_K, B_TOP_P**: Decoding parameters for model B
- **SAMPLE_SIZES**: List of sample sizes N to test (default: [100, 500, 1000, 2000, 5000])
- **N_MAX**: Maximum number of samples to precompute per context (default: 20000)
- **GLOBAL_SEED**: Random seed for reproducibility (default: 42)

## Expected Results

As the sample size N increases, you should observe:

1. **Convergence of temperature estimates:**
   - `T_hat_ab` and `T_hat_ac` should both approach the true temperature
   - The difference `|T_hat_ab - T_hat_ac|` should tend to 0

2. **Consistency across token pairs:**
   - Both token pairs (a,b) and (a,c) should yield similar temperature estimates
   - This demonstrates the consistency of the estimator

3. **Reduced variance:**
   - Standard deviation of estimates should decrease as N increases

## Mathematical Background

The estimator solves the system:

```
r1 = (Δz1 + Δb) / T
r2 = (Δz2 + Δb) / T
```

Where:
- `r_i = log(p_a_Ci / p_b_Ci)` is the empirical log-ratio from model B samples
- `Δz_i = z_a_Ci - z_b_Ci` is the logit difference from model A (white-box)
- `T` is the unknown temperature
- `Δb` is the unknown logit-bias difference

Solution:
- `T = (Δz1 - Δz2) / (r1 - r2)`
- `Δb = r1 * T - Δz1`

## Notes

- The experiment assumes you have Mistral-7B available locally (cached by HuggingFace)
- **Sampling is done once**: `precompute_samples.py` generates all samples upfront, then `experiment_phase1.py` reuses them for all N values
- This makes the experiment much faster, especially on CPU where sampling is slow
- Results are saved incrementally, so you can stop and resume if needed
- **Device detection is automatic**: The code will use CUDA if available, otherwise CPU
- Make sure you have sufficient GPU memory if using CUDA (Mistral-7B needs ~14GB VRAM)
- All paths are relative, so the code will work when cloned to any directory
- **Reusing samples is valid** as long as you don't change the model, decoding parameters, or prompts

## Troubleshooting

1. **"Token IDs not set" error:**
   - Run `inspect_tokens.py` first to find appropriate token IDs
   - Update `config.py` with your chosen token IDs

2. **Out of memory errors:**
   - Reduce batch size or use CPU instead of CUDA
   - Use a smaller model variant if available

3. **Slow execution:**
   - Use CUDA if available (much faster)
   - The sampling step (`precompute_samples.py`) is the slow part - run it once and reuse
   - The experiment step (`experiment_phase1.py`) is now very fast since it just counts tokens
   - Reduce `N_MAX` in config.py for initial testing if needed

4. **"Precomputed samples not found" error:**
   - Run `precompute_samples.py` first before running `experiment_phase1.py`
   - Make sure `N_MAX` in config.py is >= max(SAMPLE_SIZES)

