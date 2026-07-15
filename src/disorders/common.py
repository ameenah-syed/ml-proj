"""Shared generation helpers reused across the disorder modules in
src/disorders/. Centralizing these keeps each disorder module focused on
what makes it clinically distinct (see each module's docstring for chapter
citations) rather than re-deriving shared physiology.
"""

import numpy as np

from src.clinical_reference import reference_median_bmi


def baseline_row_defaults(rng, n: int) -> dict:
    """Population-normal/inapplicable defaults for columns not relevant to
    a given disorder; each module overrides only what it actually uses, so
    every module still emits the full raw column schema."""
    return {
        "purging_behaviors": np.zeros(n, dtype=bool),
        "compensatory_behavior_type": np.full(n, "none"),
        "binge_frequency_per_week": np.clip(rng.normal(0, 0.05, size=n), 0, None),
        "binge_behaviors": np.zeros(n, dtype=bool),
        "loss_of_control_eating": np.zeros(n, dtype=bool),
        "russells_sign": np.zeros(n, dtype=bool),
        "parotid_swelling": np.zeros(n, dtype=bool),
        "dental_erosion": np.zeros(n, dtype=bool),
        "food_variety_count": np.clip(rng.normal(25, 5, size=n), 10, 40),
        "fear_of_aversive_consequences": np.zeros(n, dtype=bool),
        "sensory_sensitivity_score": np.clip(rng.normal(2, 1.5, size=n), 0, 10),
    }


def behavioral_score(rng, latent_severity, n: int, noise_sd: float = 1.2):
    return np.clip(latent_severity + rng.normal(0, noise_sd, size=n), 0, 10)


def weight_trajectory(
    rng, age, sex, height_cm, n: int,
    pct_median_bmi_mean, pct_median_bmi_sd, pct_median_bmi_clip,
    suppression_mean, suppression_sd, suppression_clip,
    duration_shape, duration_scale, duration_clip,
) -> dict:
    """Returns current_weight_kg, highest_past_weight_kg,
    weight_loss_duration_months (raw columns) plus underscore-prefixed
    pct_median_bmi/weight_suppression_pct, used internally to drive
    physiology below -- features.py recomputes both from the raw columns,
    so these auxiliary values are never written to the output CSV."""
    ref_bmi = reference_median_bmi(age, sex)
    pct_median_bmi = np.clip(rng.normal(pct_median_bmi_mean, pct_median_bmi_sd, size=n), *pct_median_bmi_clip)
    current_weight_kg = pct_median_bmi / 100.0 * ref_bmi * (height_cm / 100.0) ** 2

    weight_suppression_pct = np.clip(rng.normal(suppression_mean, suppression_sd, size=n), *suppression_clip)
    highest_past_weight_kg = current_weight_kg / (1 - weight_suppression_pct / 100.0)

    weight_loss_duration_months = np.clip(rng.gamma(duration_shape, duration_scale, size=n), *duration_clip)

    return {
        "current_weight_kg": current_weight_kg,
        "highest_past_weight_kg": highest_past_weight_kg,
        "weight_loss_duration_months": weight_loss_duration_months,
        "_pct_median_bmi": pct_median_bmi,
        "_weight_suppression_pct": weight_suppression_pct,
    }


def physiologic_stress(latent_severity, weight_suppression_pct, resilience):
    """Ch. 2, p.1401-1404: "genetic variability" -- resilience dampens, but
    never fully masks, the physiologic effect of illness severity."""
    stress = np.clip(0.5 * (latent_severity / 10) + 0.5 * np.minimum(weight_suppression_pct / 30, 1), 0, 1)
    return stress * (1 - 0.6 * resilience)


def malnutrition_factor(pct_median_bmi, resilience):
    """Hepatitis of starvation (Ch. 3, p.2997-3013) tracks degree of
    underweight specifically, not general ED severity."""
    return np.clip((100 - pct_median_bmi) / 40, 0, 1.5) * (1 - 0.5 * resilience)


_COMPENSATORY_TYPES = ("vomiting", "laxative_abuse", "diuretic_abuse", "excessive_exercise")
_COMPENSATORY_PROBS = (0.55, 0.2, 0.1, 0.15)
# Ch. 7, p.4570-4571: exercise-purging "does not result in the type of
# physiology described in this chapter" -- much lower electrolyte impact.
_INTENSITY_BY_TYPE = {"vomiting": 1.0, "laxative_abuse": 0.9, "diuretic_abuse": 1.0, "excessive_exercise": 0.15, "none": 0.0}


def purging_profile(rng, n: int, purge_prob) -> dict:
    """Ch. 2 (p.1354-1358) true-orthostasis criteria and Ch. 10 (p.5455-5459)
    potassium-loss discussion both stem from purging mechanism, not just
    restriction -- this is shared by anorexia.py (purge subtype) and
    bulimia.py."""
    purging = rng.random(n) < purge_prob
    chosen = rng.choice(_COMPENSATORY_TYPES, size=n, p=_COMPENSATORY_PROBS)
    compensatory_behavior_type = np.where(purging, chosen, "none")
    purging_intensity = np.array([_INTENSITY_BY_TYPE[t] for t in compensatory_behavior_type])
    # Volume depletion / true orthostasis (Ch. 2, p.1354-1358) tracks
    # purging mechanism -- rare for exercise-type purging.
    volume_depleted = purging & (rng.random(n) < np.where(compensatory_behavior_type == "excessive_exercise", 0.05, 0.7))
    return {
        "purging_behaviors": purging,
        "compensatory_behavior_type": compensatory_behavior_type,
        "_purging_intensity": purging_intensity,
        "_volume_depleted": volume_depleted,
    }


def binge_profile(rng, n: int, freq_shape, freq_scale, freq_clip) -> dict:
    binge_frequency_per_week = np.clip(rng.gamma(freq_shape, freq_scale, size=n), *freq_clip)
    binge_behaviors = binge_frequency_per_week >= 0.1
    return {"binge_frequency_per_week": binge_frequency_per_week, "binge_behaviors": binge_behaviors}


def vitals_and_labs(rng, n: int, effective_stress, mal_factor, purging_intensity, volume_depleted) -> dict:
    resting_hr = np.clip(72 - 35 * effective_stress + rng.normal(0, 4, size=n), 32, 100)
    # "Walk across the room" test (Ch. 2, p.1392-1401): a rise of ~75%+ with
    # stable BP marks the "starving heart," distinct from true orthostasis.
    rise_fraction = np.clip(0.15 + 0.9 * effective_stress + rng.normal(0, 0.15, size=n), 0.05, 1.6)
    ambulatory_hr = np.clip(resting_hr * (1 + rise_fraction), 40, 170)

    systolic_bp_lying = np.clip(112 - 10 * effective_stress + rng.normal(0, 8, size=n), 80, 140)
    diastolic_bp_lying = np.clip(72 - 6 * effective_stress + rng.normal(0, 6, size=n), 45, 95)

    # True orthostatic vitals (Ch. 2, p.1354-1358): lying vs. standing 3 min,
    # SBP drop >=20 or DBP drop >=10, with HR rise >=20 -- reflects volume
    # depletion (e.g. purging), not pure restriction.
    standing_hr = np.where(volume_depleted, resting_hr + rng.uniform(20, 40, size=n), resting_hr + rng.normal(3, 2, size=n))
    systolic_bp_standing = np.where(
        volume_depleted, systolic_bp_lying - rng.uniform(20, 35, size=n), systolic_bp_lying + rng.normal(0, 5, size=n)
    )
    diastolic_bp_standing = np.where(
        volume_depleted, diastolic_bp_lying - rng.uniform(10, 20, size=n), diastolic_bp_lying + rng.normal(0, 4, size=n)
    )
    standing_hr = np.clip(standing_hr, 40, 160)
    systolic_bp_standing = np.clip(systolic_bp_standing, 70, 140)
    diastolic_bp_standing = np.clip(diastolic_bp_standing, 40, 95)

    temperature_f = np.clip(98.6 - 4 * effective_stress + rng.normal(0, 0.4, size=n), 94, 99.5)

    # "2-3-4 rule" (Ch. 10, p.5451-5453): Mg ~2 mg/dL, Phos ~3 mg/dL, K ~4
    # mEq/L. Purging drives potassium loss more than restriction alone.
    k_stress = effective_stress * 0.5 + purging_intensity * 1.1
    potassium_meq_l = np.clip(4.0 - k_stress * 1.3 + rng.normal(0, 0.3, size=n), 2.0, 4.8)
    phosphorus_mg_dl = np.clip(3.0 - effective_stress * 1.1 + rng.normal(0, 0.3, size=n), 1.0, 4.5)
    magnesium_mg_dl = np.clip(2.0 - effective_stress * 0.5 + rng.normal(0, 0.15, size=n), 1.0, 2.6)
    # QTc prolongation (Ch. 10, p.5471-5476): worsened by low potassium and illness severity.
    qtc_msec = np.clip(400 + (4.0 - potassium_meq_l) * 40 + 60 * effective_stress + rng.normal(0, 15, size=n), 350, 560)

    # Hepatitis of starvation (Ch. 3, p.2997-3013): AST/ALT elevation >=150
    # U/L independently predicts hypoglycemia.
    ast_u_l = np.clip(20 + 140 * mal_factor + rng.lognormal(0, 0.4, size=n) * 5, 8, 400)
    alt_u_l = np.clip(ast_u_l * rng.uniform(0.7, 1.3, size=n) + rng.normal(0, 5, size=n), 8, 420)
    hepatitis_flag = (ast_u_l >= 150) | (alt_u_l >= 150)
    glucose_mg_dl = np.clip(88 - 25 * mal_factor - np.where(hepatitis_flag, 10, 0) + rng.normal(0, 6, size=n), 40, 110)

    return {
        "resting_hr": resting_hr,
        "standing_hr": standing_hr,
        "ambulatory_hr": ambulatory_hr,
        "systolic_bp_lying": systolic_bp_lying,
        "diastolic_bp_lying": diastolic_bp_lying,
        "systolic_bp_standing": systolic_bp_standing,
        "diastolic_bp_standing": diastolic_bp_standing,
        "temperature_f": temperature_f,
        "potassium_meq_l": potassium_meq_l,
        "phosphorus_mg_dl": phosphorus_mg_dl,
        "magnesium_mg_dl": magnesium_mg_dl,
        "qtc_msec": qtc_msec,
        "ast_u_l": ast_u_l,
        "alt_u_l": alt_u_l,
        "glucose_mg_dl": glucose_mg_dl,
    }


def exam_findings(rng, n: int, effective_stress) -> dict:
    lanugo = rng.random(n) < np.clip(0.5 * effective_stress, 0, 0.6)
    acrocyanosis = rng.random(n) < np.clip(0.55 * effective_stress, 0, 0.65)
    cold_intolerance = rng.random(n) < np.clip(0.7 * effective_stress, 0, 0.8)
    constipation = rng.random(n) < np.clip(0.6 * effective_stress + 0.05, 0, 0.75)
    return {"lanugo": lanugo, "acrocyanosis": acrocyanosis, "cold_intolerance": cold_intolerance, "constipation": constipation}


def purging_exam_findings(rng, n: int, purging_intensity) -> dict:
    """Physical signs of purging (Ch. 8): Russell's sign (knuckle calluses
    from vomiting), parotid gland swelling, dental erosion. Scale with
    cumulative purging intensity, so exercise-type purging stays near zero
    per the same Ch.7 p.4570-4571 distinction used in vitals_and_labs."""
    p_base = np.clip(0.5 * purging_intensity, 0, 0.6)
    russells_sign = rng.random(n) < np.clip(p_base * 0.8, 0, 0.5)
    parotid_swelling = rng.random(n) < p_base
    dental_erosion = rng.random(n) < np.clip(p_base * 1.1, 0, 0.65)
    return {"russells_sign": russells_sign, "parotid_swelling": parotid_swelling, "dental_erosion": dental_erosion}


def amenorrhea(rng, n: int, sex, effective_stress, weight_suppression_pct):
    p = np.clip(0.65 * effective_stress + 0.5 * np.minimum(weight_suppression_pct / 30, 1), 0, 0.9) * 0.8
    return (sex == "F") & (rng.random(n) < p)
