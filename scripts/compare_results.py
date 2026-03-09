
import os
import argparse
import pandas as pd
import glob
import re
import sys

# Add project root to path
sys.path.append(os.getcwd())

from prompt_generation.eval import evaluate_answer

# Keyword mapping for rough accuracy check
LABEL_KEYWORDS = {
    "sit": ["sit", "sedentary", "sitting", "rest", "static"],
    "stand": ["stand", "standing", "still", "posture", "static"],
    "walk": ["walk", "gait", "locomotion", "step", "walking"],
    "stairsup": ["stair", "ascent", "climb", "up", "stairs"],
    "stairsdown": ["stair", "descent", "down", "stairs"],
}

def check_keyword_match(text, label):
    """
    Returns True if any keyword associated with the label is found in the text.
    """
    if not isinstance(label, str):
        return False
    
    keywords = LABEL_KEYWORDS.get(label.lower(), [])
    text_lower = text.lower()
    
    for kw in keywords:
        # Use regex to find whole words or distinct patterns if needed, 
        # but simple substring search is often enough for a first pass
        if kw in text_lower:
            return True
    return False

def parse_filename_index(filename):
    """
    Extracts sample index from filename like 'uci_sample_0_result.txt'
    """
    try:
        # Assumes format ..._sample_X_result.txt
        base = os.path.basename(filename)
        # Regex to find integer after "sample_"
        match = re.search(r"sample_(\d+)_", base)
        if match:
            return int(match.group(1))
    except:
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Compare LLM (Analysis) vs VLM (Visual) results.")
    parser.add_argument("--dir_a", required=True, help="Directory containing Analysis (Sensor Stats) results.")
    parser.add_argument("--dir_v", required=True, help="Directory containing Visual (Chart) results.")
    parser.add_argument("--ground_truth", default="data/df_uci.csv", help="Path to ground truth CSV.")
    parser.add_argument("--output", default="output/comparison_report.csv", help="Output comparison CSV.")
    args = parser.parse_args()

    # Load Ground Truth
    try:
        df_gt = pd.read_csv(args.ground_truth)
        print(f"Loaded ground truth with {len(df_gt)} samples.")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return

    # Find result files
    files_a = glob.glob(os.path.join(args.dir_a, "*_result.txt"))
    files_v = glob.glob(os.path.join(args.dir_v, "*_result.txt"))

    print(f"Found {len(files_a)} analysis files and {len(files_v)} visual files.")

    # Map index to files
    results = {}
    
    for f in files_a:
        idx = parse_filename_index(f)
        if idx is not None:
            if idx not in results: results[idx] = {}
            results[idx]['file_a'] = f

    for f in files_v:
        idx = parse_filename_index(f)
        if idx is not None:
            if idx not in results: results[idx] = {}
            results[idx]['file_v'] = f

    # Process Comparison
    comparison_data = []

    for idx, paths in results.items():
        if 'file_a' not in paths or 'file_v' not in paths:
            continue # verify overlap

        # Get Label
        try:
            true_label = df_gt.iloc[idx]['label']
        except:
            true_label = "unknown"

        row = {
            "sample_id": idx,
            "true_label": true_label
        }

        # --- Process Analysis (LLM) ---
        with open(paths['file_a'], 'r') as f:
            text_a = f.read()
        
        eval_a = evaluate_answer(text_a, mode="analysis")
        row['score_a_overall'] = eval_a.get('overall', 0)
        row['score_a_struct'] = eval_a['scores'].get('structure', 0)
        row['match_a'] = check_keyword_match(text_a, true_label)
        
        # --- Process Visual (VLM) ---
        with open(paths['file_v'], 'r') as f:
            text_v = f.read()
            
        # Note: Visual prompts essentially produce the same analysis structure
        eval_v = evaluate_answer(text_v, mode="analysis") 
        row['score_v_overall'] = eval_v.get('overall', 0)
        row['score_v_struct'] = eval_v['scores'].get('structure', 0)
        row['match_v'] = check_keyword_match(text_v, true_label)

        comparison_data.append(row)

    # Save Report
    if not comparison_data:
        print("No overlapping samples found to compare.")
        return

    df_res = pd.DataFrame(comparison_data)
    
    # Calculate Summary Stats
    print("\nXXX Comparison Summary XXX")
    print(f"Total Samples Compared: {len(df_res)}")
    print("-" * 30)
    print(f"Analysis (LLM) | Avg Score: {df_res['score_a_overall'].mean():.2f} | Label Match: {df_res['match_a'].sum()}/{len(df_res)}")
    print(f"Visual (VLM)   | Avg Score: {df_res['score_v_overall'].mean():.2f} | Label Match: {df_res['match_v'].sum()}/{len(df_res)}")
    print("-" * 30)
    
    df_res.to_csv(args.output, index=False)
    print(f"Detailed report saved to {args.output}")

if __name__ == "__main__":
    main()
