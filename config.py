"""
Configuration file for Phase 1 LLM Fingerprinting Experiment
"""

from dataclasses import dataclass
from typing import List
import torch

@dataclass
class Config:
    """Configuration class for the experiment"""
    
    # Model configuration
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Prompts for contexts C1 and C2
    PROMPT_C1: str = "Pretend to be an RNG. Output an integer from one to three (format your output in the numerical form). Do nothing else."
    PROMPT_C2: str = "Sally bought some apples. The number of apples she bought is between one and three (inclusive). Output a plausible number of apples that Sally bought (in numerical form). Do nothing else."
    
    # Token IDs (to be filled manually after inspecting tokenizer)
    # These should be tokens that appear in next-token distribution for both C1 and C2
    # and don't appear literally in the prompt text
    TOKEN_A_ID: int = None  # Will be set after tokenizer inspection
    TOKEN_B_ID: int = None  # Will be set after tokenizer inspection
    TOKEN_C_ID: int = None  # Will be set after tokenizer inspection
    
    # Decoder parameters for black-box model B
    B_TEMPERATURE: float = 0.8
    B_TOP_K: int = 50
    B_TOP_P: float = 0.95
    B_MAX_NEW_TOKENS: int = 1  # Only care about first next token
    
    # Sample sizes for convergence analysis
    # Start with smaller sizes for testing, then increase for full experiment
    # Full experiment: [100, 1000, 5000, 10000, 20000]
    # Quick test: [100, 500, 1000, 2000, 5000]
    SAMPLE_SIZES: List[int] = None  # Will be set to [100, 500, 1000, 2000, 5000] for initial testing
    REPEATS_PER_N: int = 1  # Number of independent runs per N
    
    # Random seed for reproducibility
    GLOBAL_SEED: int = 42
    
    # Output directory
    OUTPUT_DIR: str = "results_phase1"
    
    def __post_init__(self):
        """Set default values for mutable fields"""
        if self.SAMPLE_SIZES is None:
            # Start with smaller sizes for initial testing
            # Increase to [100, 1000, 5000, 10000, 20000] for full experiment
            self.SAMPLE_SIZES = [100, 500, 1000, 2000, 5000]
    
    def get_token_ids(self) -> List[int]:
        """Get list of token IDs to track"""
        return [self.TOKEN_A_ID, self.TOKEN_B_ID, self.TOKEN_C_ID]
    
    def validate(self) -> bool:
        """Validate that required token IDs are set"""
        if any(tid is None for tid in [self.TOKEN_A_ID, self.TOKEN_B_ID, self.TOKEN_C_ID]):
            return False
        return True


# Global config instance
config = Config()

