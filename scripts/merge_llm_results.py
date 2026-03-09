import os
import glob
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Merge LLM batch output texts back into a CSV dataset.")
    parser.add_argument("--csv_in", required=True, help="Path to original CSV (e.g., data/df_uci.csv)")
    parser.add_argument("--csv_out", required=True, help="Path to save merged CSV (e.g., data/df_uci_A1.csv)")
    parser.add_argument("--res_dir", required=True, help="Path to the directory containing 00000_result.txt files")
    args = parser.parse_args()

    print(f"Loading {args.csv_in}...")
    try:
        df = pd.read_csv(args.csv_in)
    except Exception as e:
        print(f"Failed to load {args.csv_in}: {e}")
        return

    patterns = []
    missing = 0
    for i in range(len(df)):
        file_path = os.path.join(args.res_dir, f"{i:05d}_result.txt")
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                patterns.append(f.read().strip())
        else:
            patterns.append("")
            missing += 1
            
    df["pattern"] = patterns
    
    # Filter out missing records mapped to "" to allow subset training
    df = df[df["pattern"] != ""]
    
    # Save the new CSV 
    df.to_csv(args.csv_out, index=False)
    print(f"Successfully merged {len(df)} records into {args.csv_out} ({missing} missing filtered out)")

if __name__ == "__main__":
    main()
