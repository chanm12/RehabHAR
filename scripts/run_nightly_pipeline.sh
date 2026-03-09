#!/bin/bash
set -e

echo "============================================================"
echo " RehabHAR Nightly Pipeline Routine"
echo " Experiment: A1 (Text Modality Baseline on UCI)"
echo " LLM: claude-3-haiku-20240307"
echo "============================================================"

mkdir -p output/logs
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="output/logs/nightly_A1_${timestamp}.log"
exec > >(tee -i "$log_file")
exec 2>&1

export MODEL="claude-3-haiku-20240307"
export PYTHONPATH=.

echo " "
echo "▶ [1/4] Starting Batch LLM Generation (Stage 0)"
python3 scripts/batch_generate.py --dataset uci --exp_id A1 --mode analysis --call --resume

# Find the newly created result directory safely
LATEST_DIR=$(find output/experiments/A1/uci_analysis_* -maxdepth 1 -type d | sort -r | head -n 1)
echo "  ↳ Generated results found in: $LATEST_DIR"

echo " "
echo "▶ [2/4] Merging Batch Results into CSV"
python3 scripts/merge_llm_results.py --csv_in data/df_uci.csv --csv_out data/df_uci_A1.csv --res_dir "$LATEST_DIR"

echo " "
echo "▶ [3/4] Running Stage 1 (Text Encoder Contrastive Training)"
python3 training/train_stage1.py --source uci_A1 --num_epochs 10 --batch_size 16 --model_save_path output/models/A1_uci_stage1_best.pth

echo " "
echo "▶ [4/4] Running Stage 2 (Sensor Encoder Alignment)"
# Using uci_A1 as both source and target to verify local embedding convergence
python3 training/train_stage2.py --source uci_A1 --target uci_A1 --num_epochs 10 --batch_size 64 --model_save_path output/models/A1_uci_stage2_best.pth

echo " "
echo "✅ Nightly Pipeline Complete! Log saved to $log_file"
