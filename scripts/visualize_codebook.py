"""
visualize_codebook.py – Inspect codebook quality and token distribution.

Usage:
    PYTHONPATH=. python3 scripts/visualize_codebook.py \
        --codebook checkpoints/vq_codebook_uci_K64.pkl \
        --quantized data/uci_quantized.npy \
        --output_dir output/codebook_vis
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from joblib import load
from collections import Counter


def main():
    parser = argparse.ArgumentParser(description="Visualize VQ codebook stats")
    parser.add_argument("--codebook", required=True)
    parser.add_argument("--quantized", required=True, help="Quantized .npy file")
    parser.add_argument("--output_dir", default="output/codebook_vis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load codebook
    cb = load(args.codebook)
    kmeans = cb["kmeans"]
    K = cb["K"]
    centers = kmeans.cluster_centers_   # (K, D)

    # Load quantized data
    quantized = np.load(args.quantized, allow_pickle=True)
    all_tokens = np.concatenate([t for t in quantized if len(t) > 0])
    counts = Counter(all_tokens.tolist())

    # ---- Plot 1: Token usage (distribution) ----
    fig, ax = plt.subplots(figsize=(12, 4))
    ids = list(range(K))
    freqs = [counts.get(i, 0) for i in ids]
    ax.bar(ids, freqs, color="steelblue", edgecolor="none")
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Token Usage Distribution (K={K})")
    total = sum(freqs)
    used = sum(1 for f in freqs if f > 0)
    ax.set_title(f"Token Usage Distribution (K={K})  |  Used: {used}/{K}  |  Total windows: {total:,}")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "token_distribution.png"), dpi=120)
    plt.close(fig)
    print(f"Saved token_distribution.png  (used {used}/{K} tokens)")

    # ---- Plot 2: Codebook centroids heatmap ----
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(centers.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Token ID")
    ax.set_ylabel("Feature Dimension")
    ax.set_title("Codebook Centroids Heatmap")
    plt.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, "codebook_heatmap.png"), dpi=120)
    plt.close(fig)
    print("Saved codebook_heatmap.png")

    # ---- Plot 3: Perplexity (entropy-based) ----
    probs = np.array(freqs, dtype=float)
    probs = probs / (probs.sum() + 1e-8)
    entropy = -np.sum(probs * np.log(probs + 1e-8))
    perplexity = float(np.exp(entropy))
    print(f"Codebook Perplexity: {perplexity:.2f} / {K}  (higher = more uniform usage)")

    # Save summary text
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write(f"K (codebook size): {K}\n")
        f.write(f"Active tokens: {used} / {K}\n")
        f.write(f"Total windows: {total:,}\n")
        f.write(f"Perplexity: {perplexity:.2f}\n\n")
        f.write("Top 10 tokens:\n")
        for tok, cnt in sorted(counts.items(), key=lambda x: -x[1])[:10]:
            f.write(f"  Token {tok:3d}: {cnt:6d} ({100*cnt/total:.1f}%)\n")
    print(f"Summary saved to {args.output_dir}/summary.txt")


if __name__ == "__main__":
    main()
