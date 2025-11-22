# Quick Start Guide: Running the Phase 1 Experiment

## Step-by-Step Instructions

### Step 1: Install Dependencies ✅
```bash
pip install -r requirements.txt
```
*(Already done!)*

### Step 2: Inspect Tokens
Run the token inspector to find appropriate token IDs:

```bash
python inspect_tokens.py
```

This will:
- Load the Mistral-7B model
- Show top tokens for both prompts (C1 and C2)
- Display common tokens that appear in both contexts
- Help you choose three tokens (a, b, c)

**What to look for:**
- Tokens with probabilities > 0.01 (ideally > 0.05) in both contexts
- Tokens that don't appear literally in the prompt text
- Tokens with rank < 50 (since B_TOP_K = 50)
- Good candidates: digit tokens like '1', '2', '3' (since prompts use words "one", "two", "three")

### Step 3: Update Config with Token IDs
Open `config.py` and set your chosen token IDs:

```python
config.TOKEN_A_ID = <token_id_from_inspector>
config.TOKEN_B_ID = <token_id_from_inspector>
config.TOKEN_C_ID = <token_id_from_inspector>
```

### Step 4: Run the Experiment
```bash
python experiment_phase1.py
```

This will:
- Load the model and pre-compute logits
- Run sampling experiments for different sample sizes N
- Estimate temperature T and logit-bias Δb
- Save results to `results_phase1/` directory

**Note:** The experiment may take a while, especially for larger N values. Results are saved incrementally.

### Step 5: Analyze Results
```bash
python analysis_phase1.py
```

This will:
- Load the results from `results_phase1/convergence_results.csv`
- Print summary statistics
- Generate convergence plots showing:
  - Temperature estimates vs sample size
  - Convergence of estimator difference
  - Estimation error vs sample size

## Expected Output

After running the analysis, you should see:
- Summary statistics table showing convergence as N increases
- Plots saved to `results_phase1/convergence_plots.png`
- Both `T_hat_ab` and `T_hat_ac` should converge to the true temperature
- The difference `|T_hat_ab - T_hat_ac|` should approach 0

## Troubleshooting

**"Token IDs not set" error:**
- Make sure you've run `inspect_tokens.py` and updated `config.py`

**Out of memory:**
- The model will use CPU by default if CUDA is not available
- For large models, you may need to reduce batch size or use a smaller model

**Slow execution:**
- Start with smaller sample sizes in `config.py`: `[100, 500, 1000]`
- Increase to full sizes `[100, 500, 1000, 2000, 5000]` once you verify it works

## Quick Test Run

To quickly test the pipeline:
1. Run `inspect_tokens.py` → pick 3 tokens
2. Update `config.py` with token IDs
3. In `config.py`, set: `SAMPLE_SIZES = [100, 500]` (small test)
4. Run `experiment_phase1.py`
5. Run `analysis_phase1.py`

Then scale up to full experiment sizes once verified!

