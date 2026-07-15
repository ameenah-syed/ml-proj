"""Synthetic patient generator for the multi-disorder eating disorder
classifier.

Real eating-disorder patient records are protected health information and
none are used here. Patients are simulated from distributions grounded in
the clinical literature, principally Jennifer Gaudiani's "Sick Enough: A
Guide to the Medical Complications of Eating Disorders" (Routledge, 2019).
See README.md for the full source list.

This module is a thin orchestrator: it samples shared demographics once,
assigns each patient a label, and dispatches each label's subgroup to its
disorder module under src/disorders/ (see those modules' docstrings for the
clinical grounding behind each disorder's generative logic).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.disorders import DISORDER_MODULES, LABEL_PROBS, PREVALENCE_LABEL_PROBS

_ROUND1 = [
    "age", "height_cm", "current_weight_kg", "highest_past_weight_kg", "weight_loss_duration_months",
    "fear_of_weight_gain_score", "body_image_disturbance_score", "restrictive_intake_score",
    "exercise_compulsion_score", "binge_frequency_per_week", "sensory_sensitivity_score", "temperature_f",
]
_ROUND0 = [
    "resting_hr", "standing_hr", "ambulatory_hr", "systolic_bp_lying", "diastolic_bp_lying",
    "systolic_bp_standing", "diastolic_bp_standing", "qtc_msec", "ast_u_l", "alt_u_l",
    "glucose_mg_dl", "food_variety_count",
]
_ROUND2 = ["potassium_meq_l", "phosphorus_mg_dl", "magnesium_mg_dl"]


def generate_patients(n: int = 4000, seed: int = 42, label_probs: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label_probs = label_probs or LABEL_PROBS
    labels = list(label_probs.keys())
    probs = np.array([label_probs[label] for label in labels])
    probs = probs / probs.sum()

    sex = rng.choice(["F", "M"], size=n, p=[0.85, 0.15])
    age = np.clip(rng.normal(19, 6, size=n), 12, 45)
    height_cm = np.where(sex == "F", rng.normal(163, 7, size=n), rng.normal(176, 7, size=n))
    assigned_label = rng.choice(labels, size=n, p=probs)

    groups = []
    for target_label, module in DISORDER_MODULES.items():
        mask = assigned_label == target_label
        if not mask.any():
            continue
        raw = module.generate_group(rng, age[mask], sex[mask], height_cm[mask], target_label)
        group_df = pd.DataFrame({k: v for k, v in raw.items() if not k.startswith("_")})
        group_df["age"] = age[mask]
        group_df["sex"] = sex[mask]
        group_df["height_cm"] = height_cm[mask]
        group_df["label"] = target_label
        groups.append(group_df)

    df = pd.concat(groups, ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "patient_id", [f"P{i:05d}" for i in range(len(df))])

    for col in _ROUND1:
        df[col] = df[col].round(1)
    for col in _ROUND0:
        df[col] = df[col].round(0)
    for col in _ROUND2:
        df[col] = df[col].round(2)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic eating-disorder patient data.")
    parser.add_argument("--n", type=int, default=4000, help="number of synthetic patients")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="data/synthetic_patients.csv")
    parser.add_argument(
        "--prevalence",
        action="store_true",
        help="sample labels at approximate true population prevalence instead of the balanced training distribution",
    )
    args = parser.parse_args()

    label_probs = PREVALENCE_LABEL_PROBS if args.prevalence else None
    df = generate_patients(n=args.n, seed=args.seed, label_probs=label_probs)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} synthetic patients to {out_path}")
    print(df["label"].value_counts(normalize=True).rename("proportion"))


if __name__ == "__main__":
    main()
