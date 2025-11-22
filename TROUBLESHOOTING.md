# Troubleshooting: Model Loading Issues

## Problem: Script Stops at "Loading checkpoint shards"

This usually means the model is too large for your available RAM.

### Solutions:

### Solution 1: Use a Smaller Model for Testing

For initial testing, you can use a smaller model. Update `config.py`:

```python
# For testing - smaller model (1.3B parameters, ~2.6GB)
MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.1"  # Change this
# Or use an even smaller model:
# MODEL_NAME: str = "gpt2"  # Very small, for quick testing
```

**Note:** The experiment should work with any model, but results will differ. For the actual experiment, you'll want Mistral-7B.

### Solution 2: Check Available RAM

1. Open Task Manager (Ctrl+Shift+Esc)
2. Check "Memory" usage
3. Mistral-7B needs ~14-16GB RAM on CPU
4. If you have less, you'll need to:
   - Close other applications
   - Use a smaller model
   - Or use GPU if available

### Solution 3: Use Memory-Optimized Loader

Replace the import in `inspect_tokens.py`:

```python
# Change this line:
from model_loader import load_model_and_tokenizer

# To this:
from model_loader_optimized import load_model_and_tokenizer
```

The optimized loader uses float16 instead of float32, reducing memory by ~50%.

### Solution 4: Wait Longer

CPU loading can take 10-20 minutes for large models. Check Task Manager - if Python is using CPU and memory, it's still working.

### Solution 5: Use GPU (If Available)

If you have a CUDA-capable GPU:
1. Install CUDA-enabled PyTorch: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
2. The model will automatically use GPU if available

### Quick Test: Use GPT-2

For a quick test of the pipeline:

1. In `config.py`, change:
   ```python
   MODEL_NAME: str = "gpt2"
   ```

2. Run `inspect_tokens.py` - this will load in seconds

3. Once you verify the pipeline works, switch back to Mistral-7B

**Note:** GPT-2 is much smaller and will give different results, but it's good for testing the code works.

