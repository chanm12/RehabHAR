"""
train_vq_codebook.py – Fit a K-Means codebook from IMU windows.

Usage:
    PYTHONPATH=. python3 scripts/train_vq_codebook.py \
        --npy data/uci_data.npy --dataset uci --K 64 --window 64 --stride 32

Outputs:
    checkpoints/vq_codebook_{dataset}_K{K}.pkl  – sklearn KMeans model

Feature vector (25-d):
    Amplitude group   [0:7]   acc_rms (XYZ), svm_rms, gyro_rms (XYZ)
    Frequency group   [7:9]   dominant_freq_fft, zero_crossing_rate
    Variability group [9:12]  acc_std (XYZ)
    Ramanujan group   [12:16] rpt_score_acc, rpt_hz_acc, rpt_harmonic_acc, rpt_score_gyro
    Rehab group       [16:25] jerk_rms, angular_rom, svm_p2p, inter_peak_cv,
                               sample_entropy, acc_asymmetry, gyro_steadiness,
                               sustained_effort, harmonic_quality
"""

import argparse
import os
import sys
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from joblib import dump
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from prompt_generation.ramanujan import periodicity_strength
from prompt_generation.data_analysis import extract_rehab_quality_features


def extract_window_features(window: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """
    Extract a 25-d rehab-aware feature vector from a (T, 6) IMU window.
    The rehab-specific features (indices 16-24) are delegated to
    `prompt_generation.data_analysis.extract_rehab_quality_features` so that
    codebook training uses exactly the same computation as the prompt pipeline.
    Returns None for windows that are too short.
    """
    if window.shape[0] < 8:
        return None

    acc = window[:, :3].astype(float)
    gyro = window[:, 3:6].astype(float)
    svm = np.linalg.norm(acc, axis=1)

    # ── 0:3  Acc RMS per axis ────────────────────────────────────────────────
    acc_rms = np.sqrt(np.mean(acc ** 2, axis=0))

    # ── 3    SVM RMS ─────────────────────────────────────────────────────────
    svm_rms = float(np.sqrt(np.mean(svm ** 2)))

    # ── 4:7  Gyro RMS per axis ───────────────────────────────────────────────
    gyro_rms = np.sqrt(np.mean(gyro ** 2, axis=0))

    # ── 7    Dominant frequency (FFT) ────────────────────────────────────────
    fft = np.abs(np.fft.rfft(svm))
    freqs = np.fft.rfftfreq(len(svm), d=1.0 / fs)
    dom_freq = float(freqs[np.argmax(fft[1:]) + 1]) if len(fft) > 1 else 0.0

    # ── 8    Zero-crossing rate (vertical / Z axis) ──────────────────────────
    zcr = float(np.mean(np.diff(np.sign(acc[:, 2])) != 0))

    # ── 9:12 Acc std per axis ────────────────────────────────────────────────
    acc_std = np.std(acc, axis=0)

    # ── 12-14 Ramanujan periodicity (acc SVM) ────────────────────────────────
    rpt_acc = periodicity_strength(svm, fs=fs)
    rpt_score_acc    = rpt_acc['score']
    rpt_hz_acc       = rpt_acc['hz']
    rpt_harmonic_acc = rpt_acc['harmonic_ratio']

    # ── 15-24 Rehab-specific features (shared with prompt pipeline) ──────────
    rehab = extract_rehab_quality_features(acc, gyro, fs=fs)
    rpt_score_gyro  = rehab['rpt_score_gyro']

    # Jerk RMS (not in extract_rehab_quality_features, compute locally)
    jerk = np.diff(acc, axis=0) * fs
    jerk_rms = float(np.sqrt(np.mean(jerk ** 2)))

    feat = np.array([
        # Amplitude group
        acc_rms[0], acc_rms[1], acc_rms[2],    # 0-2
        svm_rms,                                 # 3
        gyro_rms[0], gyro_rms[1], gyro_rms[2],  # 4-6
        # Frequency group
        dom_freq,                                # 7
        zcr,                                     # 8
        # Variability group
        acc_std[0], acc_std[1], acc_std[2],     # 9-11
        # Ramanujan / periodicity group
        rpt_score_acc,                           # 12
        rpt_hz_acc,                              # 13
        rpt_harmonic_acc,                        # 14
        rpt_score_gyro,                          # 15
        # Rehab-specific group (from shared data_analysis module)
        jerk_rms,                                # 16  smoothness
        rehab['angular_rom_rad'],                # 17  ROM from gyro
        float(np.max(svm) - np.min(svm)),        # 18  ROM from acc (svm p2p)
        rehab['inter_peak_cv'],                  # 19  cadence regularity
        rehab['sample_entropy'],                 # 20  movement complexity
        rehab['acc_lateral_asymmetry'],          # 21  lateral compensation
        rehab['gyro_steadiness'],                # 22  rotation steadiness
        rehab['sustained_effort_frac'],          # 23  sustained effort
        rehab['harmonic_quality'],               # 24  harmonic quality
    ], dtype=np.float32)

    feat = np.where(np.isfinite(feat), feat, 0.0)
    return feat


def slide_and_extract(data: np.ndarray, window: int, stride: int, fs: float) -> np.ndarray:
    """
    Slide over all trials and extract a feature vector per window.
    data: object array of (T, 6) arrays  OR  fixed array of (N, T, 6).
    Returns array of shape (total_windows, D).
    """
    all_feats = []
    n = len(data)
    for i in tqdm(range(n), desc="Extracting features"):
        trial = data[i] if (hasattr(data, 'dtype') and data.dtype == object) else data[i]
        trial = np.asarray(trial, dtype=float)
        if trial.ndim == 1:
            continue
        T = trial.shape[0]
        for start in range(0, T - window + 1, stride):
            win = trial[start:start + window]
            feat = extract_window_features(win, fs=fs)
            if feat is not None:
                all_feats.append(feat)
    return np.array(all_feats, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train VQ codebook (K-Means) on IMU windows")
    parser.add_argument("--npy", required=True, help="Path to .npy data file")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. uci, keen_pad)")
    parser.add_argument("--K", type=int, default=64, help="Codebook size (number of clusters)")
    parser.add_argument("--window", type=int, default=64, help="Window length in samples (should be <= shortest trial length)")
    parser.add_argument("--stride", type=int, default=32, help="Stride in samples (50% overlap)")
    parser.add_argument("--fs", type=float, default=50.0, help="Sampling frequency in Hz")
    parser.add_argument("--output_dir", default="checkpoints", help="Output dir for codebook")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading data from {args.npy}...")
    data = np.load(args.npy, allow_pickle=True)
    print(f"  Data loaded: {data.shape}")

    print(f"Sliding window (size={args.window}, stride={args.stride}) and extracting features...")
    feats = slide_and_extract(data, window=args.window, stride=args.stride, fs=args.fs)
    if feats.ndim < 2 or feats.shape[0] == 0:
        print("ERROR: No features extracted! The window size may be larger than the trial length.")
        print(f"Hint: Try reducing --window (current: {args.window}).")
        print(f"Data shape: {data[0].shape if hasattr(data[0], 'shape') else 'unknown'}")
        return
    print(f"  Total windows extracted: {feats.shape[0]}, Feature dim: {feats.shape[1]}")

    print(f"Fitting MiniBatchKMeans with K={args.K}...")
    kmeans = MiniBatchKMeans(
        n_clusters=args.K,
        random_state=args.seed,
        batch_size=min(1024, feats.shape[0]),
        n_init=3,
        max_iter=300,
        verbose=0,
    )
    kmeans.fit(feats)
    inertia = kmeans.inertia_
    print(f"  K-Means converged. Inertia: {inertia:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"vq_codebook_{args.dataset}_K{args.K}.pkl")
    dump({"kmeans": kmeans, "window": args.window, "stride": args.stride, "fs": args.fs, "K": args.K}, out_path)
    print(f"Codebook saved to: {out_path}")


if __name__ == "__main__":
    main()
