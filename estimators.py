"""
Estimator functions for temperature T and logit-bias difference Δb
"""

import math
from typing import Tuple, Optional


def estimate_T_and_delta_b(
    z_a_C1: float,
    z_b_C1: float,
    z_a_C2: float,
    z_b_C2: float,
    n_a_C1: int,
    n_b_C1: int,
    N1: int,
    n_a_C2: int,
    n_b_C2: int,
    N2: int,
    eps: float = 1e-12
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate temperature T and logit-bias difference Δb_{a,b} using tokens a and b
    across contexts C1 and C2.
    
    The estimator solves the system:
        r1 = (Δz1 + Δb) / T
        r2 = (Δz2 + Δb) / T
    
    Where:
        r_i = log(p_a_Ci / p_b_Ci)  (empirical log-ratio)
        Δz_i = z_a_Ci - z_b_Ci      (logit difference from model A)
    
    Solution:
        T = (Δz1 - Δz2) / (r1 - r2)
        Δb = r1 * T - Δz1
    
    Args:
        z_a_C1: Logit of token a in context C1 (from model A)
        z_b_C1: Logit of token b in context C1 (from model A)
        z_a_C2: Logit of token a in context C2 (from model A)
        z_b_C2: Logit of token b in context C2 (from model A)
        n_a_C1: Count of token a in context C1 (from model B samples)
        n_b_C1: Count of token b in context C1 (from model B samples)
        N1: Total samples for context C1
        n_a_C2: Count of token a in context C2 (from model B samples)
        n_b_C2: Count of token b in context C2 (from model B samples)
        N2: Total samples for context C2
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Tuple of (T_hat, delta_b_hat) or (None, None) if estimation fails
    """
    # Compute empirical probabilities (with smoothing)
    p_a1 = (n_a_C1 + eps) / (N1 + 2 * eps)
    p_b1 = (n_b_C1 + eps) / (N1 + 2 * eps)
    p_a2 = (n_a_C2 + eps) / (N2 + 2 * eps)
    p_b2 = (n_b_C2 + eps) / (N2 + 2 * eps)
    
    # Compute empirical log-ratios
    r1 = math.log(p_a1 / p_b1)
    r2 = math.log(p_a2 / p_b2)
    
    # Compute logit differences
    delta_z1 = z_a_C1 - z_b_C1
    delta_z2 = z_a_C2 - z_b_C2
    
    # Avoid division by zero
    denom = r1 - r2
    if abs(denom) < 1e-8:
        return None, None
    
    # Estimate temperature
    T_hat = (delta_z1 - delta_z2) / denom
    
    # Estimate logit-bias difference
    delta_b_hat = r1 * T_hat - delta_z1
    
    return T_hat, delta_b_hat


def estimate_T_and_delta_b_ac(
    z_a_C1: float,
    z_c_C1: float,
    z_a_C2: float,
    z_c_C2: float,
    n_a_C1: int,
    n_c_C1: int,
    N1: int,
    n_a_C2: int,
    n_c_C2: int,
    N2: int,
    eps: float = 1e-12
) -> Tuple[Optional[float], Optional[float]]:
    """
    Estimate temperature T and logit-bias difference Δb_{a,c} using tokens a and c
    across contexts C1 and C2.
    
    Same as estimate_T_and_delta_b but for token pair (a, c).
    
    Args:
        z_a_C1: Logit of token a in context C1 (from model A)
        z_c_C1: Logit of token c in context C1 (from model A)
        z_a_C2: Logit of token a in context C2 (from model A)
        z_c_C2: Logit of token c in context C2 (from model A)
        n_a_C1: Count of token a in context C1 (from model B samples)
        n_c_C1: Count of token c in context C1 (from model B samples)
        N1: Total samples for context C1
        n_a_C2: Count of token a in context C2 (from model B samples)
        n_c_C2: Count of token c in context C2 (from model B samples)
        N2: Total samples for context C2
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Tuple of (T_hat, delta_b_hat) or (None, None) if estimation fails
    """
    return estimate_T_and_delta_b(
        z_a_C1, z_c_C1, z_a_C2, z_c_C2,
        n_a_C1, n_c_C1, N1,
        n_a_C2, n_c_C2, N2,
        eps
    )

