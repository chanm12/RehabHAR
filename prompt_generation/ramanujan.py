"""
RehabHAR Ramanujan Periodicity Transform (RPT) Utilities
Provides Ramanujan sums and RPT features for robust periodicity detection in IMU signals.
"""

import numpy as np
import math

def ramanujan_sum(n: int, q: int) -> int:
    """
    Calculate Ramanujan sum c_q(n).
    c_q(n) = sum_{k=1}^q, gcd(k,q)=1 of exp(2*pi*i*k*n/q)
    It is always an integer.
    """
    if q == 0: return 0
    val = 0
    for k in range(1, q + 1):
        if math.gcd(k, q) == 1:
            val += math.cos(2 * math.pi * k * n / q)
    return int(round(val))

def rpt_transform(signal: np.ndarray, max_period: int = 0) -> dict:
    """
    Project signal onto Ramanujan subspaces to find dominant integer periodicities.
    Returns a dictionary of {period_q: energy_in_subspace_Sq}.
    
    Args:
        signal: 1D array-like signal (e.g., vertical acceleration)
        max_period: Max period to check. If 0, defaults to len(signal)//2.
    """
    x = np.asarray(signal, dtype=float)
    x = x - np.mean(x) # Remove DC
    N = len(x)
    if max_period <= 0:
        max_period = N // 2
        
    energies = {}
    
    # Simple projection approach:
    # For each candidate period q, project x onto the subspace S_q spanned by c_q(n).
    # Since c_q(n) are orthogonal for different q (under specific N), we can estimate
    # the "Ramanujan energy" E_q. 
    # A simplified metric is the correlation with the Ramanujan basis c_q.
    
    for q in range(2, max_period + 1):
        # Basis vector for period q (Cosine-like)
        cq_0 = np.array([ramanujan_sum(n, q) for n in range(N)])
        norm_0 = np.linalg.norm(cq_0)
        
        # Basis vector shifted (Sine-like approximation)
        # For q=2, shift is irrelevant as dim=1. For q>2, this helps capture phase.
        shift = max(1, q // 4)
        cq_s = np.roll(cq_0, shift)
        norm_s = np.linalg.norm(cq_s)

        energy = 0.0
        if norm_0 > 1e-9:
            coeff_0 = np.dot(x, cq_0) / (norm_0**2)
            energy += (coeff_0 * norm_0)**2
            
        if q > 2 and norm_s > 1e-9:
             # Orthogonalize cq_s relative to cq_0 for proper energy sum
             # proj_s_on_0 = (dot(cq_s, cq_0)/norm_0^2) * cq_0
             # cq_s_orth = cq_s - proj_s_on_0
             proj_s0 = np.dot(cq_s, cq_0) / (norm_0**2)
             cq_s_orth = cq_s - proj_s0 * cq_0
             norm_orth = np.linalg.norm(cq_s_orth)
             
             if norm_orth > 1e-9:
                coeff_orth = np.dot(x, cq_s_orth) / (norm_orth**2)
                energy += (coeff_orth * norm_orth)**2

        energies[q] = energy
        
    return energies

def periodicity_strength(signal: np.ndarray, fs: float = 50.0) -> dict:
    """
    Analyze signal for Ramanujan periodicity.
    Returns:
        dominant_period_q: The integer period (in samples) with max energy
        periodicity_score: 0-1 score indicating how strong this period is relative to total energy
        estimated_hz: Converted frequency in Hz
        harmonic_ratio: Ratio of energy in q vs non-harmonic periods (measure of 'clean' rhythm)
    """
    x = np.asarray(signal)
    if len(x) < 10:
        return {'period_q': 0, 'score': 0.0, 'hz': 0.0, 'harmonic_ratio': 0.0}
        
    rpt = rpt_transform(x, max_period=int(fs * 2.0)) # Check up to 2s periods (0.5Hz)
    
    if not rpt:
        return {'period_q': 0, 'score': 0.0, 'hz': 0.0, 'harmonic_ratio': 0.0}

    # Find dominant q
    sorted_q = sorted(rpt.items(), key=lambda item: item[1], reverse=True)
    best_q, best_energy = sorted_q[0]
    
    if best_energy <= 1e-9:
         return {'period_q': 0, 'score': 0.0, 'hz': 0.0, 'harmonic_ratio': 0.0}

    # Normalize by total signal energy (variance since DC removed)
    signal_energy = np.sum(x**2)
    score = best_energy / signal_energy if signal_energy > 1e-9 else 0.0
    
    # Cap score at 1.0 (approximations might slightly exceed 1)
    score = min(1.0, score)
    
    hz = fs / best_q if best_q > 0 else 0.0
    
    # Harmonic ratio: Energy in q, 2q, 3q... vs total
    harmonic_energy = 0.0
    for k in range(1, 4): # q, 2q, 3q
        harmonic_energy += rpt.get(best_q * k, 0.0)
        
    # Use sum of RPT energies for harmonic ratio denominator to keep it relative to "periodicity energy"
    total_rpt_energy = sum(rpt.values())
    harmonic_ratio = harmonic_energy / total_rpt_energy if total_rpt_energy > 1e-9 else 0.0

    return {
        'period_q': best_q,
        'score': float(score),
        'hz': float(hz),
        'harmonic_ratio': float(harmonic_ratio)
    }
