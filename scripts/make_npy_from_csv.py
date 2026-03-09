"""
RehabHAR – Convert CSV sensor data to NumPy arrays
Based on: LanHAR prompt_generation/make_npy_from_csv.py

CLI tool that reads a CSV with stringified acc/gyro columns and
produces a (N, T, 6) .npy file for use by the prompt generation
and training pipelines.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def parse_acc_gyro_string(s: str) -> np.ndarray:
    """Parse a stringified list/array of numbers into (T,3) float ndarray.
    Accepts formats like "[1, 2, 3, 4, 5, 6, ...]" possibly with newlines.
    """
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.empty((0, 3), dtype=float)
    s_clean = str(s).replace("\n", " ").replace("[", " ").replace("]", " ")
    nums = np.fromstring(s_clean, sep=" ", dtype=float)
    if nums.size % 3 != 0:
        # try comma separation as fallback
        nums = np.fromstring(s_clean, sep=",", dtype=float)
    if nums.size % 3 != 0:
        raise ValueError("Parsed sensor list length is not a multiple of 3; cannot reshape to (T,3)")
    T = nums.size // 3
    return nums.reshape(T, 3)


def build_npy_from_csv(csv_path: Path, out_path: Path, acc_col: str = "acc", gyro_col: str = "gyro",
                        trim_to_min_len: bool = True, limit: int | None = None) -> None:
    df = pd.read_csv(csv_path)
    if acc_col not in df.columns or gyro_col not in df.columns:
        raise RuntimeError(f"CSV must contain '{acc_col}' and '{gyro_col}' columns. Found: {list(df.columns)}")

    acc_list, gyro_list = [], []
    rows = df.itertuples(index=False)
    count = 0
    for row in rows:
        acc_str = getattr(row, acc_col)
        gyro_str = getattr(row, gyro_col)
        acc = parse_acc_gyro_string(acc_str)
        gyro = parse_acc_gyro_string(gyro_str)
        if acc.shape[0] == 0 or gyro.shape[0] == 0:
            continue
        if acc.shape[0] != gyro.shape[0]:
            if not trim_to_min_len:
                raise RuntimeError(f"Mismatched lengths: acc={acc.shape[0]}, gyro={gyro.shape[0]}")
            T = min(acc.shape[0], gyro.shape[0])
            acc = acc[:T]
            gyro = gyro[:T]
        acc_list.append(acc.astype(float))
        gyro_list.append(gyro.astype(float))
        count += 1
        if limit is not None and count >= limit:
            break

    if len(acc_list) == 0:
        raise RuntimeError("No valid rows parsed from CSV.")

    # Stack to (N,T,6); allow variable T across samples by padding to max length
    T_max = max(a.shape[0] for a in acc_list)
    N = len(acc_list)
    data = np.zeros((N, T_max, 6), dtype=float)
    for i, (acc, gyro) in enumerate(zip(acc_list, gyro_list)):
        T = acc.shape[0]
        data[i, :T, :3] = acc
        data[i, :T, 3:6] = gyro
        # remaining rows stay zeros if sequences are shorter than T_max

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, data)
    print(f"Saved NumPy array: {out_path} with shape {data.shape}")


def main():
    parser = argparse.ArgumentParser(description="Convert df_<name>.csv to (N,T,6) .npy for prompt generation.")
    parser.add_argument("--csv", required=True, help="Path to CSV (e.g., models/data/df_uci.csv)")
    parser.add_argument("--out", required=True, help="Output .npy path (e.g., prompt_generation/data/uci_data.npy)")
    parser.add_argument("--acc_col", default="acc", help="Accelerometer column name")
    parser.add_argument("--gyro_col", default="gyro", help="Gyroscope column name")
    parser.add_argument("--no-trim", dest="trim", action="store_false", help="Do not trim to min length when acc/gyro differ")
    parser.set_defaults(trim=True)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit number of rows")
    args = parser.parse_args()

    build_npy_from_csv(Path(args.csv), Path(args.out), args.acc_col, args.gyro_col, args.trim, args.limit)


if __name__ == "__main__":
    main()
