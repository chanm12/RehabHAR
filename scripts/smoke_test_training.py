"""
RehabHAR - Low-Resource Smoke Test for Training Pipeline
This script isolates a tiny 50-sample mini-dataset from UCI and HHAR,
injects synthetic NLP text to bypass OpenAI API calls, and triggers 
the Stage 1 and Stage 2 PyTorch trainers using a tiny BERT and tiny batch size.
"""

import os
import sys
import pandas as pd
import subprocess

def prepare_dummy_datasets(sample_size=50):
    print("Preparing synthetic datasets for the smoke test...")
    os.makedirs("data", exist_ok=True)
    
    # Isolate a tiny subset of UCI (source)
    if os.path.exists("data/df_uci.csv"):
        df_uci = pd.read_csv("data/df_uci.csv")
        df_uci_smoke = df_uci.head(sample_size).copy()
        
        # Inject dummy NLP patterns to save LLM tokens
        df_uci_smoke["pattern"] = df_uci_smoke["label"].apply(
            lambda lbl: f"This IMU reading represents a subject performing {lbl} with low-amplitude horizontal oscillation."
        )
        # Ensure our target labels exist in the cut (Stage 1 expects 'walk', 'still', 'stairsup', 'stairsdown')
        if 'walk' not in df_uci_smoke['label'].values:
            df_uci_smoke.loc[0, 'label'] = 'walk'
        if 'stairsup' not in df_uci_smoke['label'].values:
            df_uci_smoke.loc[1, 'label'] = 'stairsup'
        if 'stairsdown' not in df_uci_smoke['label'].values:
            df_uci_smoke.loc[2, 'label'] = 'stairsdown'
            
        df_uci_smoke.to_csv("data/df_smoke_uci.csv", index=False)
        print(f"Created df_smoke_uci.csv with {len(df_uci_smoke)} samples.")
    else:
        print("[Error] data/df_uci.csv not found. Are you in the project root?")
        sys.exit(1)

    # Isolate a tiny subset of HHAR (target)
    # Note: If df_hhar.csv doesn't exist, we'll just duplicate UCI and disguise it as HHAR
    # to guarantee the cross-dataset Stage 2 logic compiles.
    if os.path.exists("data/df_hhar.csv"):
        df_hhar = pd.read_csv("data/df_hhar.csv").head(sample_size).copy()
    else:
        print("df_hhar.csv not found, substituting df_uci slice as dummy target dataset...")
        df_hhar = df_uci_smoke.copy()
        
    df_hhar["pattern"] = df_hhar["label"].apply(
        lambda lbl: f"This IMU reading represents a subject performing {lbl} with low-amplitude horizontal oscillation."
    )
    df_hhar.to_csv("data/df_smoke_hhar.csv", index=False)
    print(f"Created df_smoke_hhar.csv with {len(df_hhar)} samples.")

def run_smoke_test():
    os.makedirs("output/models", exist_ok=True)
    
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    # 1. Stage 1 (Contrastive Pre-Training)
    print("\n" + "="*50)
    print("▶ STARTING STAGE 1 (Contrastive Pre-Training)")
    print("="*50)
    
    cmd_stage1 = [
        sys.executable, "training/train_stage1.py",
        "--source", "smoke_uci",
        "--model_name", "distilbert-base-uncased",
        "--batch_size", "4",
        "--max_len", "64",
        "--num_epochs", "1",
        "--model_save_path", "output/models"
    ]
    
    subprocess.run(cmd_stage1, env=env, check=True)
    
    # 2. Stage 2 (Cross-Dataset Alignment)
    print("\n" + "="*50)
    print("▶ STARTING STAGE 2 (Cross-Dataset Alignment)")
    print("="*50)
    
    cmd_stage2 = [
        sys.executable, "training/train_stage2.py",
        "--source", "smoke_uci",
        "--target", "smoke_hhar",
        "--model_name", "distilbert-base-uncased",
        "--batch_size", "4",
        "--max_len", "64",
        "--num_epochs", "1",
        "--model_save_path", "output/models"
    ]
    
    subprocess.run(cmd_stage2, env=env, check=True)
    
    print("\n✅ Smoke Test Complete! Both PyTorch stages compiled, passed forward/backward passes, and saved model checkpoints locally.")

if __name__ == "__main__":
    prepare_dummy_datasets(sample_size=30)
    run_smoke_test()
