# RehabHAR Methodology & Experimental Plan

This document outlines the experimental methodology for the RehabHAR pipeline, specifically focusing on how different Large Language Model (LLM) prompting modalities (Text, Visual, and VQ Codebook) are integrated into the zero-shot Human Activity Recognition (HAR) framework.

## 1. Objective
The goal is to determine the most effective way to extract rich semantic descriptions of raw IMU sensor data (Accelerometer and Gyroscope) using Foundation Models. These descriptions serve as "teachers" to align a smaller sensor encoder (Stage 2) into a shared semantic embedding space with human-readable activity labels (Stage 1).

## 2. Experimental Modalities (Stage 0)
We have developed three independent methods for translating raw IMU sequences into prompts for an LLM/VLM:

### A. Text / Statistical Mode
* **Mechanism:** Computes a comprehensive suite of statistical, frequency, and clinical features (e.g., Jerk RMS, Sample Entropy, Ramanujan Periodic Transform) from the raw IMU window.
* **Prompting:** Feeds these raw numbers and qualitative summaries to a standard text-to-text LLM (e.g., GPT-4o, Claude 3.5 Sonnet).
* **Hypothesis:** LLMs can reason over biomechanical statistics and frequency distributions to accurately describe the underlying motion.

### B. Visual Mode
* **Mechanism:** Plots the IMU window into a 4-panel visual chart (Accel and Gyro amplitudes over time) and pairs it with a structured "reading guide."
* **Prompting:** Feeds the image and the reading guide to a Vision-Language Model (VLM).
* **Hypothesis:** VLMs excel at spatial reasoning and pattern recognition. "Seeing" the actual waveform shapes (e.g., impact spikes vs. smooth sinusoidal waves) may lead to more accurate descriptions than raw statistical text.

### C. VQ Codebook Mode
* **Mechanism:** Uses K-Means Vector Quantization to chunk the dataset into a universal dictionary of discrete "motion primitives." Each IMU window is represented as a sequence of integer tokens.
* **Prompting:** Feeds the discrete token sequence to an LLM.
* **Hypothesis:** LLMs are fundamentally designed to process discrete token sequences (words). By discretizing continuous motion into an alphabet of movement primitives, the LLM may better recognize rhythmic repetitions and phase transitions.

---

## 3. Execution Plan

### Phase 1: Independent Ablation Study
To rigorously evaluate which modality yields the highest zero-shot classification accuracy, we will run the pipeline end-to-end three separate times:

1. **Generate Descriptions:** Run `scripts/batch_generate.py` using `--mode analysis` (Text), `--mode visual`, and `--mode vq` on a small benchmark dataset (e.g., 500 samples of UCI or HHAR).
2. **Train Stage 1 (Text Encoder):** For each mode, fine-tune BERT (`train_stage1.py`) to map the generated long-form descriptions to their short ground-truth labels.
3. **Train Stage 2 (Sensor Encoder):** For each mode, train a 1D-CNN or Transformer (`train_stage2.py`) to map raw IMU signals into the specific semantic space learned in Stage 1.
4. **Evaluate:** Use `evaluation/testing.py` to test zero-shot accuracy.
* **Outcome:** Identify the single most performant prompting modality.

### Phase 2: The Multi-View / Ensembled Approach
If computation allows, we will explore combining the modalities to create an ultimate "super-teacher" label.

* **Concept:** Instead of picking just one modality, we concatenate the outputs of all three modes for a given IMU window into one massive, multi-dimensional description.
  * *Example:* `[Text Stats Reasoning] + [Visual Waveform Reasoning] + [VQ Structural Reasoning]`
* **Execution:** Train Stage 1 (BERT) against this combined representation.
* **Hypothesis:** An ensemble description provides the most comprehensive biomechanical grounding possible, forcing the sensor encoder (Stage 2) to learn a richer representation of the physical world.

### Phase 3: Data Augmentation for Senior Rehab Transfer
A critical challenge in HAR for rehabilitation is that most open-source datasets (like UCI and HHAR) consist of healthy, young adults. To improve zero-shot transfer to senior populations or individuals with mobility impairments, we will introduce a Data Augmentation strategy in Stage 0.

* **Mechanism:** Before feeding the IMU data to the prompt generator, we mathematically degrade the healthy signals to simulate senior/impaired motion.
  * *Techniques:* Time-warping (slowing down the execution speed), Amplitude-scaling (reducing the force/impact of the movement), and injecting Perlin noise (simulating tremor or reduced steadiness).
* **Execution:** The LLM receives the degraded signal properties and generates descriptions reflecting lower intensity, poorer rhythm, and reduced steadiness.
* **Hypothesis:** By artificially generating "impaired" semantic descriptions during Stage 1 pre-training, the sensor encoder in Stage 2 will be better equipped to project actual senior patient data into the correct semantic neighborhood during inference.

### Phase 4: Label Semantics Augmentation
In addition to augmenting the *sensor signal descriptions* (Phase 3), we can also augment the *ground truth label definitions* generated in Stage 0.

* **Mechanism:** When asking the LLM to define a label like "Squats", we generate two distinct semantic profiles: one describing the biomechanics of a healthy subject, and one describing the expected biomechanics of a senior rehab subject (e.g., reduced ROM, slower cadence, potential compensations).
* **Execution:** We combine these two profiles into a single, comprehensive "Super-Label" definition representing the full spectrum of the activity. Stage 1 (BERT) is then trained to align both healthy and impaired sensor signals against this unified, broad semantic target.
* **Hypothesis:** A broader, more inclusive ground-truth semantic target will create a larger "attractor basin" in the embedding space, further improving generalization and zero-shot transfer to unseen senior data.

## 4. Experimental Matrix

The following table summarizes the planned experimental runs to validate the pipeline. Because the ultimate goal is zero-shot Human Activity Recognition for a senior rehabilitation population, **Knee-Pad (a senior rehab dataset)** will serve as the universal hold-out evaluation metric. **UCI** (healthy) and **StrengthSense** (healthy multi-sensor) will serve as the diverse training corpus.

| Exp ID | Prompt Modality | Foundation Model | Train Dataset | Eval Dataset | Feature Augmentation (Phase 3) | Label Augmentation (Phase 4) | Primary Evaluation Goal |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | Text/Stats | `gpt-4o-mini` | UCI / StrengthSense | Knee-Pad | None | None | Baseline text alignment accuracy on senior data. |
| **A2** | Visual Charts | `gpt-4o` (VLM) | UCI / StrengthSense | Knee-Pad | None | None | Does visual reasoning of charts transfer better than statistics? |
| **A3** | VQ Codebook | `gpt-4o-mini` | UCI / StrengthSense | Knee-Pad | None | None | Does structural tokenization improve senior rhythmic activity detection? |
| **E1** | Ensemble (Text+Vis+VQ) | `claude-3-5-sonnet` | UCI / StrengthSense | Knee-Pad | None | None | Does multi-view prompting create a richer, more transferable semantic space? |
| **S1** | Winning Modality | `gpt-4o` | UCI / StrengthSense | Knee-Pad | **Yes** (Simulated Impairment) | None | Does artificially degrading healthy continuous data improve zero-shot accuracy? |
| **L1** | Winning Modality | `gpt-4o` | UCI / StrengthSense | Knee-Pad | None | **Yes** (Dual Healthy+Senior Text) | Does combining healthy & impaired semantic definitions improve the target space? |
| **U1** | **Ultimate (S1+L1)** | `claude-3-5-sonnet` | All Valid Training | Knee-Pad | **Yes** | **Yes** | Maximum possible zero-shot generalization to the senior domain. |

## 5. Next Steps
1. Execute **Phase 1 (Ablation)** on a small, fast-training dataset slice to establish baselines.
2. Review the resulting Stage 2 zero-shot accuracies.
3. Proceed to full-scale training on the chosen winning modality (or the Phase 2 ensembled approach).

## 6. Findings & Observations

### Experiment A2 (Visual Modality) Trade-offs
When developing the Visual inference pipeline (using GPT-4o Vision), we discovered a fundamental computer vision trade-off regarding how the IMU signal charts are presented to the model:

1. **Auto-Scaled Subplots (Axis Confusion):**
   * *Method:* Matplotlib automatically scales the Y-axis of every subplot perfectly to its data range.
   * *Pro:* The VLM is highly accurate at reading rhythms (~1 Hz movements), shape symmetries, and macro-synchronization over time because the movement envelopes are extremely visible.
   * *Con:* The VLM suffers from "Axis Confusion". If the Z-axis oscillates by only $\pm0.1$ and the X-axis oscillates by $\pm2.0$, autoscale makes both look visually identical in height. The VLM incorrectly assumes the Z-axis is just as dominant, whereas the Text Modality (A1) catches this immediately using mathematically extracted `rms` and `p2p` limits.

2. **Global Y-Axis Normalization (Amplitude Blindness):**
   * *Method:* The Accelerometer subplots are globally locked to `±20 m/s²` and Gyroscope subplots are globally locked to `±5 rad/s`.
   * *Pro:* It mathematically solves the Axis Confusion problem. The VLM immediately realizes which axes are *not* having wild oscillations because they appear relatively flat.
   * *Con:* It introduces "Amplitude Blindness". By forcing global limits, subtle, low-intensity rhythmic movements (like sitting down or standing up) are visually squashed into near-flat lines. Consequently, the VLM downgrades its rhythm assessment from a "rhythmic movement" to "likely still / no movement."

**Conclusion for Visual Prompting:**
Pure visual prompting struggles to balance macro-level rhythm detection with micro-level axis comparison. To achieve optimal results, a **Hybrid Approach** is likely required: overlaying the mathematical text summary (A1) *onto* the prompt alongside the image, essentially letting the VLM read the text statistics to confirm the dominant axis while looking at the auto-scaled chart for rhythm and shape.
