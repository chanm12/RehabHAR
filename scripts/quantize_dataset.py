"""
quantize_dataset.py – Encode each trial as a sequence of discrete VQ token IDs.

Usage:
    PYTHONPATH=. python3 scripts/quantize_dataset.py \
        --npy data/uci_data.npy \
        --codebook checkpoints/vq_codebook_uci_K64.pkl \
        --dataset uci

Outputs:
    data/{dataset}_quantized.npy      – object array of (N_windows,) int arrays
    data/{dataset}_quantized_sample.txt – human-readable preview
"""

import argparse
import os
import sys
import numpy as np
from joblib import load
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.train_vq_codebook import extract_window_features


def quantize_trial(trial: np.ndarray, kmeans, window: int, stride: int, fs: float):
    """
    Given a (T, 6) trial array, return a (N_windows,) int array of token IDs.
    """
    trial = np.asarray(trial, dtype=float)
    T = trial.shape[0]
    tokens = []
    for start in range(0, T - window + 1, stride):
        win = trial[start:start + window]
        feat = extract_window_features(win, fs=fs)
        if feat is not None:
            token = int(kmeans.predict(feat.reshape(1, -1))[0])
            tokens.append(token)
    return np.array(tokens, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser(description="Quantize dataset using a trained VQ codebook")
    parser.add_argument("--npy", required=True, help="Path to .npy data file")
    parser.add_argument("--codebook", required=True, help="Path to trained codebook .pkl file")
    parser.add_argument("--dataset", required=True, help="Dataset name (used in output filename)")
    parser.add_argument("--output_dir", default="data", help="Output directory for quantized data")
    args = parser.parse_args()

    print(f"Loading codebook from {args.codebook}...")
    cb = load(args.codebook)
    kmeans = cb["kmeans"]
    window = cb["window"]
    stride = cb["stride"]
    fs = cb["fs"]
    K = cb["K"]
    print(f"  Codebook: K={K}, window={window}, stride={stride}, fs={fs}")

    print(f"Loading data from {args.npy}...")
    data = np.load(args.npy, allow_pickle=True)
    n = len(data)
    print(f"  Trials: {n}")

    all_tokens = []
    for i in tqdm(range(n), desc="Quantizing trials"):
        trial = data[i] if (hasattr(data, 'dtype') and data.dtype == object) else data[i]
        tokens = quantize_trial(trial, kmeans, window, stride, fs)
        all_tokens.append(tokens)

    # Save as object array
    os.makedirs(args.output_dir, exist_ok=True)
    out_npy = os.path.join(args.output_dir, f"{args.dataset}_quantized.npy")
    quantized_arr = np.empty(len(all_tokens), dtype=object)
    for i, t in enumerate(all_tokens):
        quantized_arr[i] = t
    np.save(out_npy, quantized_arr)
    print(f"Quantized data saved to: {out_npy}")

    # Save sample preview
    out_txt = os.path.join(args.output_dir, f"{args.dataset}_quantized_sample.txt")
    with open(out_txt, "w") as f:
        for i in range(min(5, n)):
            tokens_str = " ".join(map(str, all_tokens[i]))
            f.write(f"Trial {i} ({len(all_tokens[i])} tokens): {tokens_str}\n")
    print(f"Sample preview saved to: {out_txt}")


if __name__ == "__main__":
    main()
