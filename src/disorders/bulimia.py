"""Bulimia nervosa (BN) and atypical bulimia nervosa.

Sick Enough Ch. 7 (p.4548-4566): binge-and-purge behavior at least once a
week for 3+ months, patients "not typically underweight, although they may
be medically malnourished" (p.4558-4559). Unlike AN/AAN, BN has NO
restricting subtype -- binge+purge is definitional. Atypical BN's
differentiator from full BN is frequency, not weight status (p.4563-4566:
"binge/purge behaviors less often than for a diagnosis of bulimia
nervosa"), in contrast to atypical AN's weight-based split from AN.

Compensatory mechanism (vomiting/laxative/diuretic/exercise, p.4550-4551)
varies, and exercise-purging does NOT cause the same electrolyte physiology
as the others (p.4570-4571) -- handled via purging_intensity in
common.purging_profile / common.vitals_and_labs.
"""

import numpy as np

from src.disorders import common


def generate_group(rng, age, sex, height_cm, label: str) -> dict:
    n = len(age)
    row = common.baseline_row_defaults(rng, n)

    latent_severity = rng.beta(6, 1.8, size=n) * 10
    row["fear_of_weight_gain_score"] = common.behavioral_score(rng, latent_severity, n)
    row["body_image_disturbance_score"] = common.behavioral_score(rng, latent_severity, n)
    # Dietary restraint between binges is common but less purely restrictive
    # than AN, and exercise compulsion is present but not the primary driver.
    row["restrictive_intake_score"] = common.behavioral_score(rng, latent_severity * 0.6, n)
    row["exercise_compulsion_score"] = common.behavioral_score(rng, latent_severity * 0.8, n)

    # BN patients are "not typically underweight" (p.4558-4559): weight
    # status centers near population-normal, unlike AN/AAN.
    wt = common.weight_trajectory(
        rng, age, sex, height_cm, n,
        pct_median_bmi_mean=103, pct_median_bmi_sd=15, pct_median_bmi_clip=(75, 165),
        suppression_mean=6, suppression_sd=5, suppression_clip=(0, 25),
        duration_shape=3, duration_scale=2.5, duration_clip=(1, 36),
    )
    row["current_weight_kg"] = wt["current_weight_kg"]
    row["highest_past_weight_kg"] = wt["highest_past_weight_kg"]
    row["weight_loss_duration_months"] = wt["weight_loss_duration_months"]

    purge = common.purging_profile(rng, n, purge_prob=1.0)  # binge+purge is definitional
    row["purging_behaviors"] = purge["purging_behaviors"]
    row["compensatory_behavior_type"] = purge["compensatory_behavior_type"]

    # DSM-5 threshold: >=1x/week for 3mo = full BN; atypical BN sits below
    # that frequency (p.4563-4566), not split on weight like atypical AN.
    if label == "bulimia_nervosa":
        binge = common.binge_profile(rng, n, freq_shape=4, freq_scale=0.8, freq_clip=(1, 14))
    else:  # atypical_bulimia_nervosa
        binge = common.binge_profile(rng, n, freq_shape=2, freq_scale=0.25, freq_clip=(0.1, 1))
    row["binge_frequency_per_week"] = binge["binge_frequency_per_week"]
    row["binge_behaviors"] = np.ones(n, dtype=bool)

    resilience = rng.uniform(0, 1, size=n)
    # BN physiology is purging-driven, not starvation-driven -- patients
    # aren't underweight, so malnutrition_factor stays at zero.
    effective_stress = purge["_purging_intensity"] * 0.5 * (1 - 0.5 * resilience)
    mal_factor = np.zeros(n)

    row.update(common.vitals_and_labs(rng, n, effective_stress, mal_factor, purge["_purging_intensity"], purge["_volume_depleted"]))
    row.update(common.exam_findings(rng, n, effective_stress * 0.3))  # cold intolerance/lanugo are starvation signs, muted here
    row.update(common.purging_exam_findings(rng, n, purge["_purging_intensity"]))
    row["amenorrhea"] = common.amenorrhea(rng, n, sex, effective_stress * 0.5, wt["_weight_suppression_pct"])

    return row
