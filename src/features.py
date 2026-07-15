"""Clinical feature engineering for the multi-disorder eating disorder
classifier.

Turns the raw simulated chart values from src/disorders/*.py into the
derived clinical metrics and risk flags described in Sick Enough (Gaudiani,
2019), then wires up an sklearn preprocessing pipeline. Chapter/page
citations match the ones used in the disorder modules; see README.md for
the full source list.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.clinical_reference import reference_median_bmi

LABEL_COL = "label"

# Single source of truth for the target classes -- imported by
# src/disorders/__init__.py (generation side) and src/evaluate.py
# (reporting side) so both stay in sync with this list.
LABELS = (
    "none",
    "anorexia_nervosa",
    "atypical_anorexia",
    "bulimia_nervosa",
    "atypical_bulimia_nervosa",
    "binge_eating_disorder",
    "arfid",
    "osfed_other",
)

RAW_NUMERIC = [
    "age",
    "height_cm",
    "current_weight_kg",
    "highest_past_weight_kg",
    "weight_loss_duration_months",
    "fear_of_weight_gain_score",
    "body_image_disturbance_score",
    "restrictive_intake_score",
    "exercise_compulsion_score",
    "binge_frequency_per_week",
    "food_variety_count",
    "sensory_sensitivity_score",
    "resting_hr",
    "standing_hr",
    "ambulatory_hr",
    "systolic_bp_lying",
    "diastolic_bp_lying",
    "systolic_bp_standing",
    "diastolic_bp_standing",
    "temperature_f",
    "potassium_meq_l",
    "phosphorus_mg_dl",
    "magnesium_mg_dl",
    "qtc_msec",
    "ast_u_l",
    "alt_u_l",
    "glucose_mg_dl",
]
RAW_BOOLEAN = [
    "purging_behaviors",
    "binge_behaviors",
    "loss_of_control_eating",
    "fear_of_aversive_consequences",
    "lanugo",
    "acrocyanosis",
    "cold_intolerance",
    "constipation",
    "amenorrhea",
    "russells_sign",
    "parotid_swelling",
    "dental_erosion",
]
CATEGORICAL = ["sex", "compensatory_behavior_type"]


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append derived clinical metrics and risk flags used as model features."""
    out = df.copy()

    out["current_bmi"] = out["current_weight_kg"] / (out["height_cm"] / 100) ** 2
    out["pct_median_bmi"] = out["current_bmi"] / reference_median_bmi(out["age"], out["sex"]) * 100

    # Weight suppression (Sick Enough Ch. 1, p.852-856): percentage of body
    # weight lost from the highest recent weight. >=5% is clinically
    # significant in the presence of ED symptoms.
    out["weight_suppression_pct"] = (
        (out["highest_past_weight_kg"] - out["current_weight_kg"]) / out["highest_past_weight_kg"] * 100
    ).clip(lower=0)
    out["weight_suppression_significant"] = out["weight_suppression_pct"] >= 5

    out["weight_loss_rate_kg_per_month"] = (
        out["highest_past_weight_kg"] - out["current_weight_kg"]
    ) / out["weight_loss_duration_months"].clip(lower=0.5)

    # Textbook underweight status, included as a feature (not a label
    # proxy): adolescents use %mBMI < 85, adults use BMI < 18.5.
    out["underweight"] = np.where(out["age"] < 18, out["pct_median_bmi"] < 85, out["current_bmi"] < 18.5)

    # "Walk across the room" test (Ch. 2, p.1392-1401): rise from resting to
    # ambulatory pulse of ~75%+ marks the "starving heart" pattern.
    out["hr_rise_pct"] = (out["ambulatory_hr"] - out["resting_hr"]) / out["resting_hr"] * 100
    out["bradycardic"] = out["resting_hr"] < 60
    out["significant_ambulatory_rise"] = out["hr_rise_pct"] >= 75

    # True orthostatic vitals (Ch. 2, p.1354-1358): SBP drop >=20 OR DBP
    # drop >=10, together with HR rise >=20 upon standing 3 minutes.
    sbp_drop = out["systolic_bp_lying"] - out["systolic_bp_standing"]
    dbp_drop = out["diastolic_bp_lying"] - out["diastolic_bp_standing"]
    standing_hr_rise = out["standing_hr"] - out["resting_hr"]
    out["true_orthostatic_vitals"] = ((sbp_drop >= 20) | (dbp_drop >= 10)) & (standing_hr_rise >= 20)
    out["starving_heart_pattern"] = out["significant_ambulatory_rise"] & ~out["true_orthostatic_vitals"]

    # "2-3-4 rule" (Ch. 10, p.5451-5459) and QTc/hepatitis thresholds
    # (Ch. 10, p.5471-5476; Ch. 3, p.2997-3013).
    out["hypokalemia"] = out["potassium_meq_l"] < 3.0
    out["hypophosphatemia"] = out["phosphorus_mg_dl"] < 3.0
    out["hypomagnesemia"] = out["magnesium_mg_dl"] < 1.5
    out["qtc_prolonged"] = out["qtc_msec"] >= 500
    out["transaminitis"] = (out["ast_u_l"] >= 150) | (out["alt_u_l"] >= 150)
    out["hypoglycemia"] = out["glucose_mg_dl"] < 70

    # Purging without being underweight -- a BN/atypical-BN signal,
    # distinguishing it from AN purge subtype which requires underweight
    # (Ch. 7, p.4558-4563).
    out["purging_without_underweight"] = out["purging_behaviors"] & ~out["underweight"]

    # DSM-5 BN frequency threshold: >=1x/week (Ch. 7, p.4558-4559).
    out["meets_bn_frequency_threshold"] = out["binge_frequency_per_week"] >= 1

    # ARFID's defining exclusion criterion (Ch. 14): high restriction
    # without weight/shape concern -- the inverse of the AN/AAN pattern.
    out["restriction_without_weight_concern"] = (
        (out["restrictive_intake_score"] >= 5) & (out["fear_of_weight_gain_score"] < 3) & (out["body_image_disturbance_score"] < 3)
    )

    return out


DERIVED_NUMERIC = [
    "current_bmi",
    "pct_median_bmi",
    "weight_suppression_pct",
    "weight_loss_rate_kg_per_month",
    "hr_rise_pct",
]
DERIVED_BOOLEAN = [
    "weight_suppression_significant",
    "underweight",
    "bradycardic",
    "significant_ambulatory_rise",
    "true_orthostatic_vitals",
    "starving_heart_pattern",
    "hypokalemia",
    "hypophosphatemia",
    "hypomagnesemia",
    "qtc_prolonged",
    "transaminitis",
    "hypoglycemia",
    "purging_without_underweight",
    "meets_bn_frequency_threshold",
    "restriction_without_weight_concern",
]

NUMERIC_FEATURES = RAW_NUMERIC + DERIVED_NUMERIC
BOOLEAN_FEATURES = RAW_BOOLEAN + DERIVED_BOOLEAN
FEATURE_COLUMNS = NUMERIC_FEATURES + BOOLEAN_FEATURES + CATEGORICAL


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("boolean", "passthrough", BOOLEAN_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Apply clinical feature engineering to a raw patient dataframe and
    return (X, y) ready for a pipeline."""
    df = add_clinical_features(df)
    X = df[FEATURE_COLUMNS].copy()
    for col in BOOLEAN_FEATURES:
        X[col] = X[col].astype(int)
    y = df[LABEL_COL]
    return X, y


def load_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load a raw synthetic patients CSV and return (X, y) ready for a pipeline."""
    df = pd.read_csv(csv_path)
    return prepare_dataframe(df)
