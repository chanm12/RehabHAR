"""
RehabHAR – Rule-based LLM Output Evaluator
Based on: LanHAR prompt_generation/eval.py

Lightweight evaluator that checks LLM-generated motion analysis
and semantic descriptions for structural completeness, unit usage,
bullet counts, and anti-fabrication heuristics.
"""

import re
from typing import Dict, List, Literal

Mode = Literal["analysis", "semantic"]

REQUIRED_SECTIONS_ANALYSIS = [
    "Strength:",
    "Axis dominance:",
    "Rhythm:",
    "Shape:",
    "Posture / drift:",
    "Gyro–Acc sync:",
    "Vertical impact:",
]

REQUIRED_SECTIONS_SEMANTIC = [
    "General Description:",
    "Sensor Patterns (Accelerometer & Gyroscope):",
    "Body Parts Involved:",
]

UNITS_ANALYSIS = [
    r"m/s²",
    r"m/s³",
    r"rad/s",
    r"Hz",
    r"spm",
]


def _has_all_sections(text: str, sections: List[str]) -> List[str]:
    missing = []
    for s in sections:
        if s not in text:
            missing.append(f"Missing section header: {s}")
    return missing


def _has_units(text: str, units: List[str]) -> List[str]:
    findings = []
    for u in units:
        if re.search(u, text) is None:
            findings.append(f"Expected unit not found: {u}")
    return findings


def _score_from_violations(violations: List[str]) -> float:
    # Simple mapping: 0 violations => 5.0; each violation -0.6 down to min 0.0
    score = max(0.0, 5.0 - 0.6 * len(violations))
    return round(score, 2)


def evaluate_answer(answer_text: str, mode: Mode = "analysis") -> Dict:
    """
    Lightweight rule-based evaluator.
    Returns dict with per-dimension scores, violations list, and overall.
    """
    violations: List[str] = []

    if mode == "analysis":
        violations += _has_all_sections(answer_text, REQUIRED_SECTIONS_ANALYSIS)
        # Units expected somewhere in the text
        violations += _has_units(answer_text, UNITS_ANALYSIS)
        # Basic structure check: bullet lines begin with ' - ' under each section
        for sec in REQUIRED_SECTIONS_ANALYSIS:
            if sec in answer_text:
                sec_idx = answer_text.index(sec)
                snippet = answer_text[sec_idx: sec_idx + 400]
                if "- " not in snippet:
                    violations.append(f"Section lacks bullet details near: {sec}")
        # Placeholder logic checks can be expanded later
        coverage = 5.0 - 0.5 * len(_has_all_sections(answer_text, REQUIRED_SECTIONS_ANALYSIS))
        structure = 5.0 if not violations else max(0.0, 5.0 - 0.5 * len(violations))
        specificity = 4.0  # placeholder baseline
        consistency = 4.0  # placeholder baseline
        plausibility = 4.0  # placeholder baseline
        fidelity = 3.5  # placeholder baseline

    else:  # semantic
        violations += _has_all_sections(answer_text, REQUIRED_SECTIONS_SEMANTIC)

        # Bullet-count validation per section
        section_bullet_requirements = {
            "General Description:": 2,
            "Sensor Patterns (Accelerometer & Gyroscope):": 4,
            "Body Parts Involved:": 2,
        }
        for sec, min_bullets in section_bullet_requirements.items():
            if sec in answer_text:
                sec_idx = answer_text.index(sec)
                next_sec_idx = len(answer_text)
                for other_sec in REQUIRED_SECTIONS_SEMANTIC:
                    if other_sec != sec and other_sec in answer_text:
                        other_idx = answer_text.index(other_sec)
                        if other_idx > sec_idx:
                            next_sec_idx = min(next_sec_idx, other_idx)
                snippet = answer_text[sec_idx:next_sec_idx]
                bullet_count = snippet.count("\n- ") + snippet.count("\n -")
                if bullet_count < min_bullets:
                    violations.append(f"{sec} has {bullet_count} bullets, need ≥{min_bullets}")

        # Placement keyword check in Sensor Patterns section
        if "Sensor Patterns (Accelerometer & Gyroscope):" in answer_text:
            sec_idx = answer_text.index("Sensor Patterns (Accelerometer & Gyroscope):")
            next_sec_idx = len(answer_text)
            if "Body Parts Involved:" in answer_text:
                next_sec_idx = answer_text.index("Body Parts Involved:")
            snippet = answer_text[sec_idx:next_sec_idx].lower()
            placement_keywords = ["waist", "pocket", "wrist", "arm", "placement"]
            if not any(kw in snippet for kw in placement_keywords):
                violations.append("Sensor Patterns section missing placement-variation discussion")

        # Anti-fabrication check: flag suspicious dataset-specific numeric patterns
        fabrication_pattern = r'\b\d+\.\d{2,}\s*(m/s²|m/s³|rad/s|Hz|spm)\b'
        if re.search(fabrication_pattern, answer_text):
            violations.append("Possible fabricated dataset-specific numeric value detected (overly precise)")

        coverage = 5.0 - 0.5 * len(_has_all_sections(answer_text, REQUIRED_SECTIONS_SEMANTIC))
        structure = 5.0 if not violations else max(0.0, 5.0 - 0.5 * len(violations))
        specificity = 4.5 if not any("brief" in v for v in violations) else 3.5
        consistency = 4.5
        plausibility = 4.5 if "placement" not in " ".join(violations).lower() else 3.5
        fidelity = 4.5 if "fabricated" not in " ".join(violations).lower() else 2.0

    overall = min(
        round(max(0.0, min(5.0, coverage)), 2),
        _score_from_violations(violations),
        round(max(0.0, min(5.0, structure)), 2),
    )

    return {
        "scores": {
            "coverage": round(coverage, 2),
            "specificity": round(specificity, 2),
            "consistency": round(consistency, 2),
            "plausibility": round(plausibility, 2),
            "structure": round(structure, 2),
            "fidelity": round(fidelity, 2),
        },
        "violations": violations,
        "overall": overall,
    }
