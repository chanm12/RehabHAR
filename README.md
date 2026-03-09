# RehabHAR — Sensor-Based Human Activity Recognition with LLM Semantic Alignment

A rehabilitation-focused HAR system that leverages LLM-generated semantic descriptions of sensor signals
for cross-dataset activity recognition. Inspired by the [LanHAR](https://github.com/DASHLab/LanHAR) framework.

## Architecture

```
┌─────────────────────┐      ┌────────────┐      ┌──────────────────┐
│  IMU Sensor Signal   │─────▶│  Prompt     │─────▶│  LLM Semantic    │
│  (Acc + Gyro)        │      │  Generation │      │  Descriptions    │
└─────────────────────┘      └────────────┘      └────────┬─────────┘
                                                          │
         ┌────────────────────────────────────────────────┘
         ▼
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Stage 1: Train   │───▶│  Stage 2: Train   │───▶│  Inference:      │
│  Text Encoder     │    │  Sensor Encoder   │    │  Sensor → Label  │
│  (Semantic Align) │    │  (Signal → Lang)  │    │  via Similarity  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
```

**Pipeline:**
1. **Prompt Generation** — Preprocess IMU signals, extract features, build structured prompts for LLM
2. **Stage 1 (Text Encoder)** — Align sensor description embeddings with activity label embeddings
3. **Stage 2 (Sensor Encoder)** — Map raw sensor signals into the same semantic space
4. **Inference** — Compare sensor embeddings to label embeddings via cosine similarity

## Project Structure

```
RehabHAR/
├── configs/              # Hyperparams & experiment settings (YAML)
├── data/                 # Raw & processed data (gitignored)
├── prompt_generation/    # Phase 1: Feature extraction & LLM prompt building
├── models/               # Encoder architectures & loss functions
├── data_pipeline/        # PyTorch Datasets, DataLoaders, data reading
├── training/             # Stage 1 & Stage 2 training entry points
├── evaluation/           # Cross-dataset evaluation
├── scripts/              # CLI utilities
├── notebooks/            # Exploration & EDA
├── checkpoints/          # Saved model weights (gitignored)
└── logs/                 # Training logs (gitignored)
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Generate prompts from sensor data
python scripts/generate_prompts.py --data data/raw/your_data.npy

# Stage 1: Train text encoder
python -m training.train_stage1 --config configs/default.yaml

# Stage 2: Train sensor encoder
python -m training.train_stage2 --config configs/default.yaml

# Evaluate
python -m evaluation.testing --config configs/default.yaml
```

## References

- Yan et al., "Large Language Model-guided Semantic Alignment for Human Activity Recognition," IMWUT 2025.
