
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from prompt_generation.ramanujan import periodicity_strength, rpt_transform

def test_on_sine_wave():
    print("Testing Ramanujan features on synthetic sine wave...")
    fs = 50.0
    T = 2.0
    t = np.arange(int(T * fs)) / fs
    
    # 2 Hz cosine wave (better phase alignment with Ramanujan sums)
    freq = 2.0
    signal = np.cos(2 * np.pi * freq * t)
    
    # Add some noise
    signal += 0.1 * np.random.randn(len(t))
    
    feats = periodicity_strength(signal, fs=fs)
    
    print(f"True Freq: {freq} Hz")
    print(f"Estimated Freq: {feats['hz']:.2f} Hz")
    print(f"Periodicity Score: {feats['score']:.2f}")
    
    # Expected period is fs / freq = 50 / 2 = 25 samples
    expected_q = int(fs / freq)
    print(f"Expected Period (q): {expected_q}")
    print(f"Estimated Period (q): {feats['period_q']}")
    
    if abs(feats['hz'] - freq) < 0.5:
        print("PASS: Frequency estimation is accurate.")
    else:
        print("FAIL: Frequency estimation is inaccurate.")

    if feats['score'] > 0.1:
         print("PASS: Significant periodicity score for signal.")
    else:
         print(f"FAIL: Low periodicity score ({feats['score']:.2f}) for signal.")

if __name__ == "__main__":
    test_on_sine_wave()
