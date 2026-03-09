from prompt_generation.data_processing import preprocess_acc_segment, _rotate_gyro_to_aligned
from prompt_generation.data_analysis import *
from prompt_generation.eval import evaluate_answer
from prompt_generation.chart import generate_imu_chart
import numpy as np
import pandas as pd
import os
import argparse
import json
import base64
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None


_DATASET_DEVICE_LOCATION = {
    "uci": "waist",
    "motion": "front pockets",
    "hhar": "waists",
    "keen_pad": "Right Thigh (Rectus Femoris)",
    "shoaib": ["left_pocket", "right_pocket", "wrist", "upper_arm", "belt"],
    "strength_sense": ["Chest", "Left Upper Arm", "Right Upper Arm", "Left Forearm", "Right Forearm", "Waist", "Left Thigh", "Right Thigh", "Left Calf", "Right Calf"]
}

_DATASET_ACTIVITIES = {
    "uci": "[walking, biking, jogging, going upstairs, walking downstairs, sitting still]",
    "keen_pad": "[Squat (Correct), Squat (Weight on Healthy), Squat (Injured Forward), "
                "Seated Extension (Correct), Seated Extension (Lift Limb), Seated Extension (No ROM), "
                "Walking (Correct), Walking (Hip Abduction), Walking (Not Extended)]",
    "motion": "[walking, jogging, sitting, cycling, standing, going upstairs, going downstairs]",
    "hhar": "[biking, sitting, standing, walking, going upstairs, going downstairs]",
    "shoaib": "[walking, jogging, stairs, sitting, standing, biking]",
    "strength_sense": "[Walking (Flat), Walking (Incline/Decline), Rise from seat, Walk with shopping cart, Vacuuming, Squat to lying, Stand to sit, Stand to sit to lying, Greet camera, Drink water, Stairs, Push-ups, Sit-ups, Squats]"
}

def _get_device_location(dataset_name, sensor_idx=None, label=None):
    device_loc = _DATASET_DEVICE_LOCATION.get(dataset_name, "unknown location")
    if isinstance(device_loc, list):
        if sensor_idx is not None and sensor_idx < len(device_loc):
            return device_loc[sensor_idx]
        elif dataset_name == "shoaib" and label is not None and label < len(device_loc):
            return device_loc[label]
        return "multiple locations"
    return device_loc


def generate_vq_prompt(token_sequence, dataset_name, K=64, sensor_idx=None, label=None):
    """
    Generate an LLM prompt from a discrete VQ token sequence.

    Args:
        token_sequence: list or array of int token IDs, e.g. [42, 17, 17, 89, ...]
        dataset_name:   str, e.g. 'uci', 'keen_pad'
        K:              int, codebook size (vocabulary)

    Returns:
        str: prompt text
    """
    device_location = _get_device_location(dataset_name, sensor_idx, label)
    activities = _DATASET_ACTIVITIES.get(dataset_name, "[unknown]")
    tokens_str = " ".join(map(str, token_sequence))
    n_tokens = len(token_sequence)
    unique = sorted(set(token_sequence))
    n_unique = len(unique)

    prompt = f"""\
You are a motion analysis expert analyzing discretized human movement data.

────────────────────────────────
【Background】
The sensor signal has been segmented into fixed-length windows and each window has been
mapped to a discrete "motion primitive" token using K-Means Vector Quantization (VQ).

- Codebook size (vocabulary): K = {K} tokens
- Device location: {device_location}
- Possible activities: {activities}

Each token ID represents a learned short-pattern of accelerometer + gyroscope behavior.
Tokens are drawn from a fixed vocabulary; repeated tokens indicate periodic/rhythmic motion.

────────────────────────────────
【Quantized Token Sequence】
Length: {n_tokens} tokens | Unique tokens used: {n_unique} / {K}

{tokens_str}

────────────────────────────────
【Analysis Task】
Carefully study the token sequence above and reason about the following:

1. **Repetition & Rhythm**: Are there repeating patterns or runs of the same token?
   (Repetition → periodic/rhythmic motion like walking or cycling)
2. **Transitions**: Are there abrupt token changes? (Sudden change → direction reversal, start/stop)
3. **Token Diversity**: Is the sequence dominated by a few tokens or spread widely?
   (Low diversity → stationary or very repetitive; high diversity → complex or multi-phase motion)
4. **Sequence Structure**: Does the sequence have clear phases (warm-up, peak, cooldown)?

────────────────────────────────
【Output Format】
Respond with EXACTLY this structure:

**Activity**: <your predicted activity>
**Confidence**: <High / Medium / Low>
**Key Observations**:
- <Observation 1>
- <Observation 2>
- <Observation 3>
**Reasoning**: <1–2 sentences justifying your classification>
"""
    return prompt


def generate_promt(acc_raw, gyro_raw, dataset_name, label, fs=50.0, mode="auto", sensor_idx=None):

    pre = preprocess_acc_segment(acc_raw, fs=fs, mode=mode)

    acc_raw = pre["aligned"]["acc_xyz"]
    gyro_raw = _rotate_gyro_to_aligned(gyro_raw, pre["gravity"]["rot_R"])

    ACC_DATA_HERE = run_eda_slim(acc_raw, fs=fs)
    GYRO_FEATURES_JSON_HERE = extract_gyro_features(gyro_raw, fs=fs)
    GYRO_ACC_SUMMARY_JSON_HERE = gait_sync_and_impact(acc_raw, gyro_raw, fs=fs)
    REHAB_QUALITY_HERE = extract_rehab_quality_features(acc_raw, gyro_raw, fs=fs)

    device_location = _get_device_location(dataset_name, sensor_idx, label)
    activities = _DATASET_ACTIVITIES.get(dataset_name, "[unknown]")

    prompt = f"""
You are a professional wearable-device motion analysis expert, specializing in identifying motion patterns and gait characteristics from **short-window** accelerometer and gyroscope data. 
────────────────────────────────
【Data Introduction】
The data you need to anlyzez is a segment of IMU sensor reading. 
- It include (i) accelerometer data: coordinates are **gravity-aligned**, with +Z pointing **upward (vertical)**; Vert (vertical) and Horiz (horizontal) components can be used to distinguish **up–down oscillation vs. horizontal swinging**.
(ii) gyroscope data: raw angular velocity (rad/s), **no gravity alignment needed**.
- Sampling rate: 50 Hz
- Device location: {device_location}
- The context is that the IMU sensor reading may be in one of the following states: {activities}


────────────────────────────────
【Data Analysis】
All summary values for the current window; directly reference numbers with units
1) Accelerometer concise analysis (gravity-aligned):
{ACC_DATA_HERE}

Field description (Accelerometer):
- meta: N, fs
- stats: statistics of each axis and SVM (mean, std, min, max, p2p)
- body_summary: dynamic statistics after gravity separation (vert_rms, horiz_rms, vert_p2p, horiz_p2p)
- freq_summary: dominant frequency / step frequency and spectrum structure (dom_freq_hz, dom_freq_spm, low_high_energy_ratio, harmonic_2x_ratio)
- ramanujan_summary: Ramanujan Periodic Transform (RPT) features. 'periodicity_score' (0-1) indicates rhythm stability/periodicity strength. 'estimated_hz' is the RPT-derived frequency.
- peaks_summary: peaks / rhythm (peak_count, ibi_mean_s, ibi_std_s, ibi_cv)
- jerk_summary: jerk intensity (rms / p95 / vert_rms / vert_p95, in m/s³)

2) Gyroscope features (angular velocity):
{GYRO_FEATURES_JSON_HERE}
Field description (Gyroscope):
- time_stats: mean_xyz, rms_xyz, p2p_xyz, wmag_mean/rms/p2p, zcr_xyz
- energy: energy_frac_xyz (distribution of ∑ω²), net_angle_xyz_rad, abs_angle_xyz_rad
- spectral: welch_wmag / welch_wx (dom_freq_hz, top_freqs), peak_rate_hz, step_freq_est_hz

3) Gyro–Acc synchronization & vertical impact (cross-sensor alignment):
{GYRO_ACC_SUMMARY_JSON_HERE}

Field description (Sync / Impact):
Coordinates: gravity aligned with +Z vertical upward; Vert vs Horiz can be used for up–down vs sideways motion
- step_metrics: step_rate_acc_hz, step_rate_gyro_hz, step_rate_diff_hz, n_steps_acc/gyro
- sync_metrics: n_matched_pairs, mean_lag_s, median_abs_lag_s, phase_consistency_0to1
- vertical_impact: per_step_peak_to_valley_mps2 list, impact_median_mps2, impact_p95_mps2

4) Rehab Movement Quality:
{REHAB_QUALITY_HERE}

Field description (Rehab Quality):
- angular_rom_rad: joint angular displacement estimate from gyro integration (rad). Higher → greater ROM.
- acc_lateral_asymmetry: 0–1 imbalance between X and Y axis energy. Non-zero → lateral compensation / load-shifting.
- gyro_steadiness: smoothness of rotational arc (higher = steadier). Low → tremor or uneven joint motion.
- sustained_effort_frac: fraction of window where movement intensity ≥50% of peak. Low → ballistic/explosive; High → sustained/isometric.
- sample_entropy: movement unpredictability. Low = stereotyped/regular (correct form); High = complex/uncontrolled.
- inter_peak_cv: coefficient of variation of repetition intervals. Low = consistent pacing; High = fatigued/irregular.
- harmonic_quality: fundamental / (fundamental + harmonics) ratio. High → clean single-frequency motion (controlled form).
- rpt_score_gyro / rpt_hz_gyro: Ramanujan periodicity of rotation signal. High score = strong rhythmic joint movement.

────────────────────────────────
【Knowledge】

1. Vertical impact–related features are indicators of high-impact tasks. 
In IMU, stair descent usually produces the largest vertical impact (and correspondingly higher jerk) compared with ascent and level walking. 
Ascent/fast walking generally show moderate–strong vertical oscillation with a continuous rhythm; 
level walking shows medium amplitude with symmetric periodicity; still presents low amplitude and low jerk. 
Gyroscope 'rotational intensity' can help identify arm swing/trunk rotation typical of walking, but this depends on sensor placement and behavior

2. When the sensor is positioned near the body's center of mass (e.g., waist or front pocket) and the axes are gravity-aligned, the relative contributions of vertical and horizontal acceleration, together with the energy distribution across gyroscope axes, can help differentiate dominant movement mechanisms:
Vertical-dominant patterns often appear in movements with clear vertical displacement or impact.
Horizontal-dominant patterns are more typical of level walking or low-intensity motion.
Note that gait speed, handrail use, restricted arm swing, and sensor placement can all influence these tendencies.

3. During coordinated gait such as level walking, accelerometer and gyroscope signals often show similar step-related periodicity in rate and timing. 
When arm motion is limited or movement becomes asymmetric (e.g., using a handrail on stairs or carrying objects), accelerometer rhythms may persist while gyroscope activity weakens or becomes less synchronized, leading to partial decoupling.

────────────────────────────────
【Task Introduction】
You task is analyzing the pattern of the above IMU data segment.
We summarize the analysis into the following categories. 
Please respond strictly following the 7-point output format (numbers → direct verbal explanation; full units; mark data origin as [ACC] / [GYRO] / [SYNC]).
Do not directly label the activity as a specific class (e.g., "walking", "jogging"). 
If you think there is a pattern that particularly fits, you are also welcome to add.

• Category 1 **Strength (overall magnitude and whether clearly non-still)**
- [ACC] stats.SVM mean, std / p2p (m/s²)
- [ACC] jerk_summary rms / p95 (m/s³)
- [GYRO] wmag_rms / wmag_p2p (rad/s)
→ Conclusion: intensity is low / medium / high / others? Clearly non-still?

• Category 2 **Directional characteristics (contrast sustained oscillation vs. impact dominance; interpret movement mechanism)**
- [ACC] body_summary: vert_rms / horiz_rms, vert_p2p / horiz_p2p (m/s²):
  - First report vert_rms and horiz_rms (m/s²), and explicitly compare them using relational wording. Clearly state which direction has stronger sustained oscillation (RMS).
  - Then report vert_p2p and horiz_p2p (m/s²), and again compare them using ratio-style phrasing. If RMS and P2P suggest different dominance, describe it as a mixed profile rather than forcing a single label.
- [GYRO] energy_frac_xyz (distribution of ∑ω²; X=roll/arm-swing, Y=pitch, Z=yaw/torso-twist):
  -Report energy_frac_xyz (X=roll/arm-swing, Y=pitch, Z=yaw/torso-twist), and briefly state whether rotation is 
  roll-led, pitch-dominant, yaw-suppressed, or balanced.
→ Conclusion: Use a concise contrast statement in the form "Sustained sway leans direction, and peak forces land in direction (≈ratio×) → the motion pattern is vertical-leaning / horizontal-leaning / mixed？"

• Category 3 **Rhythm (final step frequency & stability)**
- [ACC] peaks_summary: ibi_mean_s / ibi_std_s / ibi_cv; peaks_freq_spm = 60 / ibi_mean_s
- [ACC] freq_summary: dom_freq_hz / dom_freq_spm, harmonic_2x_ratio, low_high_energy_ratio
- [GYRO] spectral: step_freq_est_hz (≈ peak_rate/2), welch_wmag.top_freqs (check for integer harmonic structure)
- (Conflict resolution: if [ACC] dom_freq_spm and peaks_freq_spm disagree, follow established rules – prefer peaks or 2×dom_freq_spm)
→ Conclusion: final step frequency (Hz / spm) and stability (IBI CV)? Is spectrum harmonically regular?

• Category 4 **Waveform shape (impact-like vs smooth; rising/falling symmetry)**
- [ACC] jerk_summary (rms / p95 / vert_rms / vert_p95) + qualitative interpretation of peak sharpness
- [ACC] or [GYRO] (after bandpass) describe symmetry of rise vs fall (e.g., half-height width ratio)
→ Conclusion: more impact-type or smooth transition / others ? Symmetric or asymmetric?

• Category 5 **Postural drift / slow orientation bias (only if relevant)**
- [GYRO] time_stats.mean_xyz (rad/s) and energy.net_angle_xyz_rad / abs_angle_xyz_rad (rad)
→ Conclusion: is there continuous forward-tilt / inward-rotation (e.g., mean_y<0, Δθ_y<0) or gradual drift instead of isolated wrist turns?

• Category 6 **Gyro–Acc sync**
- [SYNC] step_rate_acc_hz vs step_rate_gyro_hz (Hz) and difference
- [SYNC] median_abs_lag_s, mean_lag_s (s)
- [SYNC] phase_consistency_0to1 (0–1)
→ Conclusion: are step rates consistent but phase misaligned (decoupling, common in stair ascent using handrail)? Or well synchronized (typical in flat walking) / others?

• Category 7 **Vertical impact (cross-sensor: accelerometer vertical component)**
- [SYNC]/[ACC] vertical_impact: impact_median_mps2, impact_p95_mps2 (m/s²), and per-step dispersion
→ Conclusion: impact strength is "high and scattered" or "moderate and stable" or others

• Category 8 **Rehab Movement Quality** *(report for all datasets; especially critical for rehabilitation exercises)*
- [REHAB] angular_rom_rad: total joint angular displacement (rad) → range of motion estimate
- [REHAB] acc_lateral_asymmetry (0–1): lateral load-shifting; Non-zero → compensated gait or injury-avoidance
- [REHAB] gyro_steadiness: smooth arc vs. tremor/uneven rotation
- [REHAB] sustained_effort_frac: proportion at high intensity → ballistic vs. sustained effort profile
- [REHAB] sample_entropy: low = stereotyped/repetitive (correct form); high = variable/uncontrolled
- [REHAB] inter_peak_cv: rep-to-rep cadence consistency; low = correct pacing
- [REHAB] harmonic_quality: clean single-frequency form vs. multi-harmonic (jerky/compensated)
- [REHAB] rpt_score_gyro / rpt_hz_gyro: rhythmicity of joint rotation
→ Conclusion: movement quality is high / moderate / low? Are there signs of compensation (asymmetry), irregular pacing, or poor ROM?

────────────────────────────────
【Output template】
We provide a template of the final output (STRICTLY reuse this structure):
*Pattern Summary
- Strength:
- Axis dominance:
- Rhythm:
- Shape:
- Posture / drift:
- Gyro–Acc sync:
- Vertical impact:
- Rehab quality:


In addition, We provide an example of the final output (STRICTLY reuse this structure):
- Strength:
  - [ACC] SVM mean 10.27 m/s², std 2.29 m/s², p2p 10.00 m/s². Jerk rms 49.22 m/s³, p95 75.35 m/s³.
  - [GYRO] wmag_rms 0.97 rad/s, wmag_p2p 2.20 rad/s.
  - → Conclusion: intensity is medium–high; clearly non-still.

- Axis dominance:
  - [ACC] Sustained oscillation: vert_rms 2.31 m/s² vs horiz_rms 2.09 m/s² → vertical slightly stronger. Peak-to-peak: vert_p2p 10.17 m/s² vs horiz_p2p 4.95 m/s² → vertical ≈2.1× larger. Mixed but overall vertical-leaning (small vertical lead in RMS, clear vertical lead in peaks).
  - [GYRO] energy_frac_xyz = (X 0.106, Y 0.496, Z 0.398) → pitch-dominant rotation with modest yaw, low roll.
  - → Conclusion: Sustained sway leans vertical, and peak forces land in vertical (≈2.1×) → motion pattern is vertical-leaning.

"""

    return prompt


def generate_label_semantics(activity_label: str, is_impaired: bool = False) -> str:
    
    impaired_context = ""
    if is_impaired:
        impaired_context = """
CRITICAL IMPAIRED CONTEXT:
This definition is specifically for a senior rehabilitation patient or individual with mobility impairment. 
Describe the expected biomechanics considering age-related or pathological degradation. 
You MUST include variations such as:
- Reduced Range of Motion (ROM) or stiffness
- Slower execution speed (cadence/rhythm)
- Asymmetrical force application or lateral compensations
- Increased postural sway or reduced steadiness
"""

    prompt = f"""
You are a senior kinesiology and biomechanics expert. Write a concise, factual, and structured semantic interpretation for the activity label: "{activity_label}".
{impaired_context}

CRITICAL FORMATTING REQUIREMENTS:
- Use EXACTLY these three section headers (verbatim):
  1. "General Description:"
  2. "Sensor Patterns (Accelerometer & Gyroscope):"
  3. "Body Parts Involved:"
- Each section must have at least the minimum number of bullets:
  • General Description: ≥2 bullets
  • Sensor Patterns: ≥4 bullets (including one dedicated to placement variations)
  • Body Parts Involved: ≥2 bullets
- Start each bullet with "- " (dash-space).
- Do NOT fabricate dataset-specific numeric values. Generic ranges are acceptable (e.g., "step frequency typically 1–3 Hz").
- Keep tone neutral, factual, and grounded in biomechanics/IMU domain knowledge.

CONTENT REQUIREMENTS:

General Description:
- Define the activity: purpose, goal, typical contexts/environments.
- Note common variants if relevant (e.g., pace, incline/decline, indoor/outdoor, with/without load).

Sensor Patterns (Accelerometer & Gyroscope):
- Accelerometer: typical axis dominance (vertical vs horizontal), periodicity, impact characteristics, expected oscillation/step frequency ranges.
- Gyroscope: dominant rotational axes (roll/pitch/yaw), coordination with acceleration, torso/arm swing contributions, spectral structure.
- **Placement variations (REQUIRED):** Describe how sensor placement (waist, front/side pocket, wrist, upper arm) modulates these signatures.
- Additional pattern details: rhythm regularity, harmonic structure, jerk characteristics if applicable.

Body Parts Involved:
- Primary joints and muscle groups (e.g., ankle-knee-hip chain, core/trunk, shoulder-elbow for arm swing).
- Upper vs lower body roles, symmetry/asymmetry, balance/stability demands.

SELF-CHECKLIST (verify before finalizing):
☑ All 3 headers present exactly as specified
☑ Bullet counts: General ≥2, Sensor Patterns ≥4, Body Parts ≥2
☑ Placement-variation bullet included in Sensor Patterns section
☑ No fabricated dataset-specific numbers (generic ranges only)
☑ All bullets start with "- "
"""

    return prompt


def generate_visual_prompt(dataset_name: str, device_location: str = None, sensor_idx=None, label=None) -> str:
    if device_location is None:
        device_location = _get_device_location(dataset_name, sensor_idx, label)

    activities = _DATASET_ACTIVITIES.get(dataset_name, "[unknown]")

    prompt = f"""You are a professional wearable-device motion analysis expert. You are given a 4-panel chart of IMU sensor data from a device worn at the **{device_location}**.

The chart contains:
- **Panel 1 (top):** Accelerometer X, Y, Z axes (gravity-aligned; Z = vertical upward) in m/s²
- **Panel 2:** Accelerometer Signal Vector Magnitude (SVM) in m/s²
- **Panel 3:** Gyroscope X (roll), Y (pitch), Z (yaw) in rad/s
- **Panel 4 (bottom):** Gyroscope angular velocity magnitude in rad/s

Sampling rate: 50 Hz. The x-axis is time in seconds.

The context is that the IMU sensor reading may be in one of the following states: {activities}

────────────────────────────────
【Chart Reading Guidance】
To accurately analyze human biomechanics from this specific 4-panel layout, use the following intuition:
1. **Panel 1 (Acc XYZ)**: Reveals postural alignment and impact direction. The vertical Z-axis (gravity) will show near 9.8 m/s² when upright. Shifts in baseline indicate posture changes (e.g. leaning). Sharp spikes indicate translation/impact forces.
2. **Panel 2 (Acc SVM)**: Reveals overall movement intensity and rhythm. This absolute magnitude strips away direction, leaving a clean wave of purely dynamic human effort. It is ideal for counting repetitions, identifying stepping cadence, and assessing total exertion.
3. **Panel 3 (Gyro XYZ)**: Reveals joint mechanics and movement form. Because human joints act as hinges, the gyroscope directly measures the rotation around the sensor. It helps differentiate tasks that look similar in translation (like crouching vs sitting) by showing the accompanying trunk/limb rotation.
4. **Panel 4 (Gyro Mag |w|)**: Reveals rotational energy and smoothness. A healthy, controlled movement produces a smooth, bell-shaped magnitude curve. Jagged or stuttering lines indicate poor form, tremor, or instability. Comparing peaks here to SVM peaks (Panel 2) shows if rotational movement synchronizes with physical impacts.


────────────────────────────────
【Task】
Analyze the visual patterns in the chart and respond strictly following the 7-point output format below.
For each category, describe what you observe in the chart (amplitudes, oscillation patterns, periodicity, peak shapes, symmetry, etc.).
Mark data origin as [ACC] / [GYRO] / [SYNC]. Include approximate values read from the chart where visible.
Do not directly label the activity as a specific class.

• Category 1 **Strength (overall magnitude and whether clearly non-still)**
- [ACC] Approximate SVM mean and peak-to-peak range visible in Panel 2
- [GYRO] Approximate angular velocity magnitude range visible in Panel 4
→ Conclusion: intensity is low / medium / high? Clearly non-still?

• Category 2 **Directional characteristics (axis dominance from Panel 1 & 3)**
- [ACC] Which axis shows the strongest sustained oscillation? Which has the largest peaks?
- [GYRO] Which rotational axis dominates (roll/pitch/yaw from Panel 3)?
→ Conclusion: vertical-leaning / horizontal-leaning / mixed?

• Category 3 **Rhythm (periodicity and step frequency)**
- [ACC] Are there regular periodic peaks in Panel 1 or 2? Estimate frequency if visible.
- [GYRO] Does Panel 3/4 show matching periodicity?
→ Conclusion: estimated step frequency and rhythmic stability?

• Category 4 **Waveform shape (impact-like vs smooth)**
- [ACC] Are peaks in Panel 1/2 sharp (impact-like) or smooth (sinusoidal)? Symmetric rise/fall?
- [GYRO] Are gyroscope waveforms smooth or jerky?
→ Conclusion: impact-type or smooth transition? Symmetric or asymmetric?

• Category 5 **Postural drift / slow orientation bias**
- [GYRO] Any visible DC offset or slow drift in Panel 3 axes?
→ Conclusion: gradual tilt/rotation present?

• Category 6 **Gyro–Acc sync**
- [SYNC] Do acc peaks (Panel 1/2) and gyro peaks (Panel 3/4) appear time-aligned?
→ Conclusion: well synchronized or decoupled?

• Category 7 **Vertical impact**
- [ACC] Estimate peak-to-valley amplitudes of sharp vertical (Z-axis) features in Panel 1
→ Conclusion: impact strength is high/moderate/low? Consistent or scattered?

────────────────────────────────
【Output template】
*Pattern Summary
- Strength:
- Axis dominance:
- Rhythm:
- Shape:
- Posture / drift:
- Gyro–Acc sync:
- Vertical impact:
"""

    return prompt


if __name__ == "__main__":
    pass
    parser = argparse.ArgumentParser(description="Generate IMU analysis prompt and optionally call an LLM.")
    parser.add_argument("--npy", default="data/uci_data.npy", help="Path to a (N,T,6) NumPy array file.")
    parser.add_argument("--sample_index", type=int, default=0, help="Sample index in the N dimension to use.")
    parser.add_argument("--dataset", default="uci", help="Dataset name used in the prompt (e.g., uci, hhar, motion, shoaib).")
    parser.add_argument("--model", default=None, help="LLM model name (OpenAI SDK v1). If omitted, read from .env (MODEL/model) or fallback.")
    parser.add_argument("--api_key", default=None, help="Override API key. If omitted, load from OPENAI_API_KEY in .env or environment.")
    parser.add_argument("--call", dest="call", action="store_true", help="Call the LLM after building the prompt.")
    parser.add_argument("--no-call", dest="call", action="store_false", help="Do not call the LLM (default).")
    parser.set_defaults(call=False)
    parser.add_argument("--save_to", default=None, help="Optional path to save the LLM response.")
    parser.add_argument("--save_prompt", default=None, help="Optional path to save the generated prompt text.")
    parser.add_argument("--no-echo", dest="echo", action="store_false", help="Do not print the prompt to stdout.")
    parser.set_defaults(echo=True)
    parser.add_argument("--mode", choices=["analysis", "semantic", "visual"], default="analysis", help="analysis: sensor-window analysis prompt; semantic: label semantics prompt; visual: chart-based VLM analysis")
    parser.add_argument("--save_chart", default=None, help="Path to save the generated IMU chart PNG (visual mode).")
    parser.add_argument("--chart_mode", choices=["full", "compact"], default="full", help="Chart layout: full (4-panel) or compact (2-panel magnitudes only).")
    parser.add_argument("--rep_boundaries", default=None, help="Comma-separated rep boundary times in seconds (e.g., '0.5,1.0,1.5').")
    parser.add_argument("--vlm_model", default=None, help="VLM model for visual mode (default: gpt-4o).")
    parser.add_argument("--semantic_label", default=None, help="Activity label for --mode semantic.")
    parser.add_argument("--save_eval", default=None, help="Optional path to save evaluation JSON of the LLM output.")
    parser.add_argument("--max_attempts", type=int, default=1, help="Max regeneration attempts if eval fails (default 1 = no retry).")
    parser.add_argument("--accept_threshold", type=float, default=4.0, help="Minimum overall eval score to accept (default 4.0).")
    args = parser.parse_args()

    # Load key from .env if not provided
    load_dotenv()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    # Resolve model: CLI > env MODEL/model > default
    env_model = os.getenv("MODEL") or os.getenv("model")
    model_name = args.model or env_model or "gpt-4o-mini"

    # Resolve VLM model for visual mode
    vlm_model = args.vlm_model or os.getenv("VLM_MODEL") or "gpt-4o"

    # Build prompt (conditional on mode)
    if args.mode == "analysis":
        try:
            data = np.load(args.npy, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load '{args.npy}'. Ensure it's a valid NumPy array of shape (N,T,6).") from e

        if isinstance(data, np.ndarray) and data.dtype == object:
            try:
                data = np.array(data.tolist())
            except Exception:
                raise RuntimeError(
                    f"Loaded '{args.npy}' is object-dtype and could not be coerced into a numeric ndarray. "
                    "Please regenerate it as a numeric array of shape (N, T, 6)."
                )

        if data.ndim != 3 or data.shape[-1] != 6:
            raise RuntimeError(f"Expected data of shape (N,T,6), got {data.shape} from {args.npy}")
        if not (0 <= args.sample_index < data.shape[0]):
            raise RuntimeError(f"sample_index {args.sample_index} out of range for N={data.shape[0]}")

        acc = np.asarray(data[args.sample_index, :, :3], dtype=float)
        gyro = np.asarray(data[args.sample_index, :, 3:6], dtype=float)
        prompt = generate_promt(acc, gyro, args.dataset, '')
    elif args.mode == "semantic":
        if not args.semantic_label:
            raise RuntimeError("--semantic_label is required when --mode semantic")
        prompt = generate_label_semantics(args.semantic_label)
    else:  # visual
        try:
            data = np.load(args.npy, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load '{args.npy}'. Ensure it's a valid NumPy array of shape (N,T,6).") from e
        if isinstance(data, np.ndarray) and data.dtype == object:
            try:
                data = np.array(data.tolist())
            except Exception:
                raise RuntimeError("Could not coerce object-dtype array to numeric ndarray.")
        if data.ndim != 3 or data.shape[-1] != 6:
            raise RuntimeError(f"Expected data of shape (N,T,6), got {data.shape}")
        if not (0 <= args.sample_index < data.shape[0]):
            raise RuntimeError(f"sample_index {args.sample_index} out of range for N={data.shape[0]}")
        acc = np.asarray(data[args.sample_index, :, :3], dtype=float)
        gyro = np.asarray(data[args.sample_index, :, 3:6], dtype=float)

        # Determine device location
        ds = args.dataset
        if ds == "uci": device_location = "waist"
        elif ds == "motion": device_location = "front pockets"
        elif ds == "hhar": device_location = "waists"
        else: device_location = "unknown"

        # Parse rep boundaries if provided
        rep_bounds = None
        if args.rep_boundaries:
            rep_bounds = [float(x.strip()) for x in args.rep_boundaries.split(",")]

        # Generate chart
        chart_b64 = generate_imu_chart(
            acc, gyro, fs=50.0,
            title=f"IMU Window — {ds} sample {args.sample_index}",
            save_path=args.save_chart,
            mode=args.chart_mode,
            rep_boundaries=rep_bounds,
        )
        if args.save_chart:
            print(f"Chart saved to: {args.save_chart}")

        prompt = generate_visual_prompt(ds, device_location)

    # Optionally save the prompt and/or print it
    if args.save_prompt:
        os.makedirs(os.path.dirname(args.save_prompt) or ".", exist_ok=True)
        with open(args.save_prompt, "w", encoding="utf-8") as f:
            f.write(prompt)
    if getattr(args, "echo", True):
        print(prompt)

    # Optionally call LLM with iterative regeneration
    if args.call:
        # Determine initial models so we can check if we need both clients
        if args.mode == "visual":
            active_model = vlm_model
            system_role = "You are a senior motion analysis expert specializing in interpreting IMU sensor charts."
        elif args.mode == "semantic":
            active_model = model_name
            system_role = "You are a senior kinesiology and biomechanics expert. Produce precise, structured semantic descriptions."
        else:
            active_model = model_name
            system_role = "You are a senior motion analysis expert."
            
        mode_for_eval = "semantic" if args.mode == "semantic" else "analysis"

        # Initialize appropriate client
        is_anthropic = active_model.startswith("claude-")
        
        if is_anthropic:
            if Anthropic is None:
                raise RuntimeError("Anthropic SDK not installed. Add 'anthropic' to requirements and pip install.")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is not set. Provide it in .env or environment.")
            client = Anthropic(api_key=api_key)
        else:
            if OpenAI is None:
                raise RuntimeError("OpenAI SDK not installed. Add 'openai' to requirements and pip install.")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is not set. Provide --api_key or set it in .env/env.")
            client = OpenAI(api_key=api_key)

        best_content = None
        best_eval = None
        best_score = -1.0

        for attempt in range(1, args.max_attempts + 1):
            print(f"\n===== Attempt {attempt}/{args.max_attempts} =====")

            # Build messages with feedback from previous attempt if applicable
            messages = [{"role": "system", "content": system_role}]
            if args.mode == "visual":
                # VLM: send image + text prompt
                text_content = prompt
                if attempt > 1 and best_eval:
                    feedback = "\n".join([f"- {v}" for v in best_eval.get("violations", [])])
                    text_content = f"{prompt}\n\n**FEEDBACK FROM PREVIOUS ATTEMPT:**\nYour previous response had the following issues. Please fix them:\n{feedback}\n\nGenerate a corrected response now."
                messages.append({"role": "user", "content": [
                    {"type": "text", "text": text_content},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{chart_b64}",
                        "detail": "high"
                    }},
                ]})
                active_model = vlm_model
            else:
                if attempt == 1:
                    messages.append({"role": "user", "content": prompt})
                else:
                    feedback = "\n".join([f"- {v}" for v in best_eval.get("violations", [])])
                    retry_prompt = f"{prompt}\n\n**FEEDBACK FROM PREVIOUS ATTEMPT:**\nYour previous response had the following issues. Please fix them:\n{feedback}\n\nGenerate a corrected response now."
                    messages.append({"role": "user", "content": retry_prompt})
                active_model = model_name

            # Reasoning models (o1, o3) typically require temperature=1
            if active_model.startswith("o1") or active_model.startswith("o3"):
                temperature = 1.0
            else:
                temperature = 0.3 if attempt > 1 else 0.5

            try:
                if is_anthropic:
                    # Translate to Anthropic format
                    system_msg = ""
                    anthropic_messages = []
                    for msg in messages:
                        if msg["role"] == "system":
                            system_msg = msg["content"]
                        else:
                            if isinstance(msg["content"], list):
                                anth_content = []
                                for item in msg["content"]:
                                    if item["type"] == "text":
                                        anth_content.append({"type": "text", "text": item["text"]})
                                    elif item["type"] == "image_url":
                                        b64_data = item["image_url"]["url"].split("base64,")[1]
                                        anth_content.append({
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": "image/png",
                                                "data": b64_data
                                            }
                                        })
                                anthropic_messages.append({"role": msg["role"], "content": anth_content})
                            else:
                                anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
                    
                    resp = client.messages.create(
                        model=active_model,
                        system=system_msg,
                        messages=anthropic_messages,
                        max_tokens=2048,
                        temperature=temperature
                    )
                    content = resp.content[0].text
                else:
                    # OpenAI Format
                    resp = client.chat.completions.create(
                        model=active_model,
                        messages=messages,
                        temperature=temperature,
                    )
                    content = resp.choices[0].message.content
            except Exception as e:
                # If temperature is invalid (e.g. reasoning models), retry with temperature=1.0 for OpenAI
                if not is_anthropic and "temperature" in str(e) and ("1" in str(e) or "default" in str(e)):
                    print(f"  [info] Retrying with temperature=1.0 due to model constraint...")
                    resp = client.chat.completions.create(
                        model=active_model,
                        messages=messages,
                        temperature=1.0,
                    )
                    content = resp.choices[0].message.content
                else:
                    raise e

            # Evaluate
            try:
                eval_result = evaluate_answer(content, mode=mode_for_eval)
                overall = eval_result.get("overall", 0.0)

                print(f"Eval: overall={overall:.2f}, threshold={args.accept_threshold}")
                if eval_result.get("violations"):
                    print(f"Violations: {eval_result['violations']}")

                # Track best
                if overall > best_score:
                    best_score = overall
                    best_content = content
                    best_eval = eval_result

                # Accept if threshold met
                if overall >= args.accept_threshold:
                    print(f"✓ Accepted (score {overall:.2f} ≥ {args.accept_threshold})")
                    break
                else:
                    print(f"✗ Rejected (score {overall:.2f} < {args.accept_threshold})")
                    if attempt < args.max_attempts:
                        print("Regenerating with targeted feedback...")
            except Exception as e:
                print(f"[warn] evaluation failed on attempt {attempt}: {e}")
                if best_content is None:
                    best_content = content
                    best_eval = {"overall": 0.0, "scores": {}, "violations": ["eval error"]}

        # Use best result
        content = best_content
        eval_result = best_eval

        print("\n===== Final LLM Response =====\n")
        print(content)
        if args.save_to:
            os.makedirs(os.path.dirname(args.save_to) or ".", exist_ok=True)
            with open(args.save_to, "w", encoding="utf-8") as f:
                f.write(content)

        if eval_result:
            if args.save_eval:
                os.makedirs(os.path.dirname(args.save_eval) or ".", exist_ok=True)
                with open(args.save_eval, "w", encoding="utf-8") as f:
                    json.dump(eval_result, f, indent=2)
            print("\n===== Final Eval Summary =====")
            print(f"overall: {eval_result.get('overall', 0.0)}  |  scores: {eval_result.get('scores', {})}")
            if eval_result.get('violations'):
                print(f"violations: {eval_result['violations']}")
